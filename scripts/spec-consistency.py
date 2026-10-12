#!/usr/bin/env python3
"""Teacher-forced consistency of a server's greedy decode (speculation gate).

T3.9h-b. Greedy continuations from two kernel shapes (a 1-row decode step vs a
K-row speculative verify step, or bs=1 vs bs=4) legitimately differ in bf16
reduction order, so "tokens identical to the spec-off serve" cannot tell a
numerics flip from a KV-cache bug. This measures what a KV bug would change:
whether the tokens the server emitted are the ones the model actually predicts
when the same sequence is scored through the prefill path (teacher forcing,
which speculation and batch shape never touch).

Per prompt:
  1. /generate the prompt greedily (temperature 0, N new tokens, per-token
     logprobs and the top-1 alternative);
  2. /generate again with input_ids = prompt ids + generated ids, one new
     token, logprobs from the first generated position: the teacher-forced
     logprob of every emitted token and the model's argmax at that position;
  3. report the argmax agreement rate, the positions where the emitted token
     loses to the teacher-forced argmax by more than --margin nats (a near-tie
     flipped by numerics is small; a token accepted against corrupted KV is
     not), and the largest |decode logprob - teacher-forced logprob|.

Run it on the speculation-on serve and on the spec-off serve; compare with
--compare. Exit 0 always; this is a measurement, not a gate.

  python3 scripts/spec-consistency.py --output evidence/t39hb/b-ngram/consistency.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers  # noqa: E402

import importlib.util  # noqa: E402

_bd = importlib.util.spec_from_file_location("bench_decode", pathlib.Path(__file__).resolve().parent / "bench-decode.py")
bench_decode = importlib.util.module_from_spec(_bd)
_bd.loader.exec_module(bench_decode)
CORPUS = pathlib.Path(__file__).resolve().parent / "fixtures" / "nll-corpus"


def generate(url: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{url.rstrip('/')}/generate", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **auth_headers()})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def chat_prompt(user: str) -> str:
    """DeepSeek-V4.1 chat-mode (thinking off) prompt, as encoding/test_encoding.py in the checkpoint
    renders it; /generate takes raw text, and an untemplated prompt stops after one or two tokens."""
    return "<｜begin▁of▁sentence｜><｜User｜>" + user + "<｜Assistant｜></think>"


def greedy(url: str, prompt: str, n: int, timeout: float) -> dict:
    r = generate(url, {"text": chat_prompt(prompt),
                       "sampling_params": {"max_new_tokens": n, "temperature": 0},
                       "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 1}, timeout)
    meta = r["meta_info"]
    prompt_ids = [e[1] for e in meta["input_token_logprobs"]]
    out = meta["output_token_logprobs"]
    return {"prompt_ids": prompt_ids, "output_ids": [e[1] for e in out], "output_logprobs": [float(e[0]) for e in out],
            "text": r.get("text", ""), "finish_reason": meta.get("finish_reason"),
            "spec_accept_length": meta.get("spec_accept_length")}


def teacher_force(url: str, prompt_ids: list[int], output_ids: list[int], timeout: float) -> dict:
    start = len(prompt_ids)
    r = generate(url, {"input_ids": prompt_ids + output_ids,
                       "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                       "return_logprob": True, "logprob_start_len": start - 1, "top_logprobs_num": 1}, timeout)
    meta = r["meta_info"]
    lps = meta["input_token_logprobs"]
    tops = meta["input_top_logprobs"]
    # Entries cover positions start-1 .. end; the prediction for the token at
    # position p is stored at index p - (start - 1). Position start-1 itself has
    # no prediction in this window (None); the first generated token is index 1.
    forced, argmax = [], []
    for k in range(len(output_ids)):
        e = lps[k + 1]
        t = tops[k + 1]
        assert e[1] == output_ids[k], (k, e[1], output_ids[k])
        forced.append(float(e[0]))
        argmax.append((int(t[0][1]), float(t[0][0])))
    return {"forced_logprobs": forced, "argmax": argmax}


def analyse(g: dict, tf: dict, margin: float) -> dict:
    out = g["output_ids"]
    agree = 0
    flips = []
    max_dlp = 0.0
    for k, tok in enumerate(out):
        top_id, top_lp = tf["argmax"][k]
        forced = tf["forced_logprobs"][k]
        max_dlp = max(max_dlp, abs(g["output_logprobs"][k] - forced))
        if top_id == tok:
            agree += 1
        else:
            gap = top_lp - forced
            flips.append({"pos": k, "emitted": tok, "argmax": top_id, "gap_nats": round(gap, 4)})
    big = [f for f in flips if f["gap_nats"] > margin]
    return {"tokens": len(out), "argmax_agree": agree, "argmax_agree_rate": agree / max(1, len(out)),
            "flips": len(flips), "flips_over_margin": len(big), "max_flip_gap_nats": max((f["gap_nats"] for f in flips), default=0.0),
            "max_abs_dlogprob_decode_vs_forced": max_dlp, "flip_detail": flips[:20]}


def long_prompt(target_tokens: int) -> str:
    text = ""
    for f in sorted(CORPUS.glob("*.txt")):
        text += f.read_text(errors="replace") + "\n\n"
        if len(text) >= target_tokens * 4:
            break
    return "Here is a document:\n\n" + text[: target_tokens * 4] + "\n\nSummary of the document above, section by section:\n\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--long-tokens", type=int, default=12000, help="0 = skip the long corpus prompt")
    ap.add_argument("--margin", type=float, default=0.5, help="nats; flips with a larger teacher-forced gap are counted separately")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--output", required=True)
    ap.add_argument("--compare", help="consistency.json from another server configuration")
    ap.add_argument("--label", default="")
    ap.add_argument("--flush-cache", action="store_true",
                    help="GET /flush_cache before each prompt: cold prefix cache and, for NGRAM, an empty draft corpus "
                         "(the server resets it on flush), so every verify step carries rejected rows")
    args = ap.parse_args()

    prompts = {"copy-code": bench_decode.COPY_FIXTURES[0], "copy-table": bench_decode.COPY_FIXTURES[1],
               "review": bench_decode.DISTINCT_FIXTURES[0]}
    if args.long_tokens:
        prompts[f"long{args.long_tokens}"] = long_prompt(args.long_tokens)

    results = {}
    for name, prompt in prompts.items():
        if args.flush_cache:
            urllib.request.urlopen(f"{args.url.rstrip('/')}/flush_cache", timeout=args.timeout).read()
            time.sleep(1)
        g = greedy(args.url, prompt, args.max_tokens, args.timeout)
        tf = teacher_force(args.url, g["prompt_ids"], g["output_ids"], args.timeout)
        a = analyse(g, tf, args.margin)
        a["prompt_tokens"] = len(g["prompt_ids"])
        a["finish_reason"] = g["finish_reason"]
        a["spec_accept_length"] = g["spec_accept_length"]
        a["text_head"] = g["text"][:200]
        results[name] = a
        print(f"{name:12} prompt={a['prompt_tokens']:6} out={a['tokens']:4} argmax-agree={a['argmax_agree']}/{a['tokens']} "
              f"({a['argmax_agree_rate']:.3f}) flips>{args.margin}nats={a['flips_over_margin']} max-gap={a['max_flip_gap_nats']:.3f} "
              f"max|dlogp| decode-vs-forced={a['max_abs_dlogprob_decode_vs_forced']:.3g} accept={a['spec_accept_length']}")

    report = {"label": args.label, "url": args.url, "max_tokens": args.max_tokens, "margin_nats": args.margin,
              "flush_cache": args.flush_cache, "prompts": results}
    if args.compare:
        base = json.load(open(args.compare))["prompts"]
        report["compare"] = {}
        for name, a in results.items():
            b = base.get(name)
            if b:
                report["compare"][name] = {"agree_rate_delta": a["argmax_agree_rate"] - b["argmax_agree_rate"],
                                          "flips_over_margin_delta": a["flips_over_margin"] - b["flips_over_margin"]}
                print(f"compare {name:12} agree-rate {b['argmax_agree_rate']:.3f} -> {a['argmax_agree_rate']:.3f}  "
                      f"flips>margin {b['flips_over_margin']} -> {a['flips_over_margin']}")
    stamp_or_exit(report, pathlib.Path(args.output), "spec-consistency.py")
    pathlib.Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

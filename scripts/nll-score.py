#!/usr/bin/env python3
"""Teacher-forced mean NLL of a fixed text: the quality metric for approximations.

Greedy-token comparison (greedy-logprobs.py) turned out to be noise on this
hybrid: at D=0, two bs=1 runs of the same prompt already pick different tokens
at near-ties, so "diverges at token N" says nothing about deferral or kernel
quality (E21 result 7). Teacher-forced NLL does not have that problem: the
tokens are fixed, and averaging −log p over a few thousand of them turns tie
flips into a few 1e-4 nats of noise.

Uses SGLang's native /generate with return_logprob + logprob_start_len=0; the
response carries meta_info.input_token_logprobs = [[logprob, token_id, None],
...] (first entry has logprob None). Scores the frozen corpus
scripts/fixtures/nll-corpus.txt (≈3.5K tokens) unless --text-file is given.

  # baseline on config A (run twice to see the noise floor)
  python3 scripts/nll-score.py --output evidence/x/nll-d0.json
  # on config B, compare
  python3 scripts/nll-score.py --output evidence/x/nll-d2.json --compare evidence/x/nll-d0.json

Reports mean NLL (nats/token), per-1K-token segments, and vs the baseline:
ΔNLL, max per-token |Δlogp|, and how many tokens changed by >0.5 nats.
"""
import argparse
import json
import math
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TEXT = HERE / "fixtures" / "nll-corpus.txt"


def score(url: str, text: str, timeout: float) -> dict:
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    req = urllib.request.Request(f"{url.rstrip('/')}/generate", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.load(resp)
    meta = r["meta_info"]
    entries = meta["input_token_logprobs"]
    lps = [e[0] for e in entries[1:]]  # position 0 has no prediction
    ids = [e[1] for e in entries]
    if any(lp is None for lp in lps):
        raise SystemExit("server returned None logprobs after position 0; is return_logprob supported?")
    return {"token_ids": ids, "logprobs": lps, "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"), "e2e_latency_s": meta.get("e2e_latency")}


def summarize(lps: list[float], seg: int) -> dict:
    n = len(lps)
    mean = -sum(lps) / n
    segments = []
    for i in range(0, n, seg):
        chunk = lps[i:i + seg]
        segments.append({"start": i, "n": len(chunk), "nll": -sum(chunk) / len(chunk)})
    return {"tokens_scored": n, "mean_nll": mean, "ppl": math.exp(mean), "segments": segments}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--text-file", type=pathlib.Path, default=DEFAULT_TEXT)
    ap.add_argument("--max-chars", type=int, default=0, help="truncate the text (0 = whole file)")
    ap.add_argument("--segment", type=int, default=1024, help="tokens per reported segment")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--compare", type=pathlib.Path, help="baseline JSON from an earlier run")
    ap.add_argument("--label", default="", help="free-text tag saved in the JSON (e.g. 'D=2')")
    args = ap.parse_args()

    text = args.text_file.read_text()
    if args.max_chars:
        text = text[:args.max_chars]
    res = score(args.url, text, args.timeout)
    summ = summarize(res["logprobs"], args.segment)
    out = {"label": args.label, "text_file": str(args.text_file.name), "chars": len(text), **res, **summ}

    print(f"tokens scored {summ['tokens_scored']}  mean NLL {summ['mean_nll']:.5f} nats/tok  "
          f"ppl {summ['ppl']:.3f}  cached_tokens {res['cached_tokens']}  latency {res['e2e_latency_s']:.1f} s")
    for s in summ["segments"]:
        print(f"  seg @{s['start']:>5} n={s['n']:>4}  nll {s['nll']:.5f}")

    if args.compare:
        base = json.loads(args.compare.read_text())
        if base["token_ids"] != res["token_ids"]:
            print("WARNING: tokenisation differs from baseline; comparing only the common prefix")
        n = min(len(base["logprobs"]), len(res["logprobs"]))
        d = [res["logprobs"][i] - base["logprobs"][i] for i in range(n)]
        delta = summ["mean_nll"] - base["mean_nll"]
        maxabs = max(abs(x) for x in d)
        mean_d = sum(d) / n
        sd = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
        se = sd / math.sqrt(n)
        z = mean_d / se if se else 0.0  # paired test: is the mean logp change distinguishable from 0?
        worse = sum(1 for x in d if x < -0.5)
        better = sum(1 for x in d if x > 0.5)
        out["compare"] = {"baseline": str(args.compare), "delta_mean_nll": delta, "max_abs_delta_logp": maxabs,
                          "mean_abs_delta_logp": sum(abs(x) for x in d) / n, "sd_delta_logp": sd,
                          "z_paired": z, "tokens_worse_gt_0p5": worse, "tokens_better_gt_0p5": better}
        print(f"vs {args.compare.name}: ΔNLL {delta:+.5f} nats/tok (z {z:+.2f})  max|Δlogp| {maxabs:.3f}  "
              f"mean|Δlogp| {out['compare']['mean_abs_delta_logp']:.4f} sd {sd:.3f}  "
              f"tokens >0.5 nats worse/better: {worse}/{better} of {n}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

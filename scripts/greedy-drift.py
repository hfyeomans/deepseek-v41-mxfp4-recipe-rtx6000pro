#!/usr/bin/env python3
"""Run-to-run greedy drift of one server: the same prompt, N times, from a flushed cache.

T3.15b. Every quality gate here is teacher-forced (prefill path) and bit-stable
across serves; the decode path is not (E21 result 7, T3.15: bs=1 greedy
continuations of one prompt diverge after 320-390 characters at D=0). This
script measures that drift directly so a candidate cause can be toggled and
compared:

  * each prompt is sent --repeats times at temperature 0 with the radix cache
    flushed before every request (so every run prefills from scratch and the
    only variable is the server's own numerics);
  * per prompt it reports whether all repeats are byte-identical, the earliest
    token index at which any repeat departs from the first, and the largest
    |delta logprob| on the shared prefix (drift that has not yet flipped a
    token);
  * --long-tokens adds prompts built from scripts/fixtures/nll-corpus/ of about
    that many tokens, so contexts past the indexer top-k (512 for V4.1-Flash)
    are covered separately from short ones. The server's prompt_tokens is
    recorded for each.

  python3 scripts/greedy-drift.py --output evidence/t315b/a-baseline/drift.json
  python3 scripts/greedy-drift.py --output evidence/t315b/b-toggle/drift.json \
      --compare evidence/t315b/a-baseline/drift.json

Exit 0 always; this is a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers  # noqa: E402

SHORT_PROMPTS = [
    "Why is the sky blue during the day and red at sunset? Explain the physics in detail.",
    "Explain how a CPU cache hierarchy (L1, L2, L3) works and why it matters for performance.",
    "What were the main causes of the fall of the Western Roman Empire? Answer in three short paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then explain its complexity.",
]
CORPUS = pathlib.Path(__file__).resolve().parent / "fixtures" / "nll-corpus"


def flush_cache(url: str) -> None:
    req = urllib.request.Request(f"{url.rstrip('/')}/flush_cache", data=b"", headers=auth_headers(), method="POST")
    urllib.request.urlopen(req, timeout=60).read()


def ask(url: str, model: str, prompt: str, n: int, timeout: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **auth_headers()})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = r["choices"][0]
    content = (ch.get("logprobs") or {}).get("content") or []
    toks = [(c["token"], float(c["logprob"])) for c in content]
    return toks, r.get("usage", {}).get("prompt_tokens")


def long_prompt(target_tokens: int) -> str:
    """About target_tokens of real prose from the NLL corpus (~4 chars per token), then a question."""
    text = ""
    for f in sorted(CORPUS.glob("*.txt")):
        text += f.read_text(errors="replace") + "\n\n"
        if len(text) >= target_tokens * 4:
            break
    text = text[: target_tokens * 4]
    return "Here is a document:\n\n" + text + "\n\nSummarize the document above in detail, section by section."


def analyse(runs: list[list[tuple[str, float]]]) -> dict:
    first = runs[0]
    identical = all(len(r) == len(first) and all(a[0] == b[0] for a, b in zip(r, first)) for r in runs[1:])
    first_div = None
    max_dlp = 0.0
    for r in runs[1:]:
        n = min(len(r), len(first))
        for i in range(n):
            if r[i][0] != first[i][0]:
                first_div = i if first_div is None else min(first_div, i)
                break
            max_dlp = max(max_dlp, abs(r[i][1] - first[i][1]))
        else:
            if len(r) != len(first):
                first_div = n if first_div is None else min(first_div, n)
    # Drift before the first token flip: logprob deltas on the shared prefix
    # tell "bit-identical" (0.0) apart from "numerically different, same argmax".
    return {
        "repeats": len(runs),
        "tokens": len(first),
        "all_identical": identical,
        "first_divergence_token": first_div,
        "max_abs_dlogprob_shared_prefix": max_dlp,
        "bitwise_logprobs_identical": identical and max_dlp == 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41-flash-hybrid")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--long-tokens", type=int, nargs="*", default=[3000],
                    help="also run corpus prompts of about these many tokens (0 or none to skip)")
    ap.add_argument("--long-max-tokens", type=int, default=128)
    ap.add_argument("--no-flush", action="store_true", help="keep the radix cache between repeats")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--output", required=True)
    ap.add_argument("--compare", help="drift.json from another server configuration")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    prompts = [("short", p, args.max_tokens) for p in SHORT_PROMPTS]
    for t in args.long_tokens:
        if t > 0:
            prompts.append((f"long{t}", long_prompt(t), args.long_max_tokens))

    results = []
    for kind, prompt, n in prompts:
        runs, ptoks = [], None
        for _ in range(args.repeats):
            if not args.no_flush:
                flush_cache(args.url)
            toks, ptoks = ask(args.url, args.model, prompt, n, args.timeout)
            runs.append(toks)
        row = {"kind": kind, "prompt_tokens": ptoks, "prompt_head": prompt[:60], **analyse(runs),
               "runs": [[t for t, _ in r] for r in runs],
               "logprobs": [[lp for _, lp in r] for r in runs]}
        results.append(row)
        print(f"{kind:9s} prompt_tokens={ptoks}  identical={row['all_identical']}  first_div={row['first_divergence_token']}  "
              f"max|dlogp| prefix={row['max_abs_dlogprob_shared_prefix']:.3g}  bitwise={row['bitwise_logprobs_identical']}")

    short = [r for r in results if r["kind"] == "short"]
    summary = {
        "short_prompts_all_identical": all(r["all_identical"] for r in short),
        "short_first_divergence_min": min((r["first_divergence_token"] for r in short if r["first_divergence_token"] is not None), default=None),
        "short_max_abs_dlogprob": max(r["max_abs_dlogprob_shared_prefix"] for r in short),
        "long": {r["kind"]: {k: r[k] for k in ("prompt_tokens", "all_identical", "first_divergence_token", "max_abs_dlogprob_shared_prefix")}
                 for r in results if r["kind"] != "short"},
    }
    report = {"label": args.label, "url": args.url, "model": args.model, "repeats": args.repeats,
              "flush_cache": not args.no_flush, "summary": summary, "prompts": results}
    if args.compare:
        base = json.loads(pathlib.Path(args.compare).read_text())
        report["compare"] = {"path": args.compare, "label": base.get("label"), "summary": base["summary"]}
        print("vs", args.compare, json.dumps(base["summary"]))
    print("summary", json.dumps(summary))
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    stamp_or_exit(report, out, "greedy-drift.py")
    out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

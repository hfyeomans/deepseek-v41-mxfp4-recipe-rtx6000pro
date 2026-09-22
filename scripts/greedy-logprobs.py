#!/usr/bin/env python3
"""Greedy-decode a fixed prompt set with logprobs and save/compare the result.

Detects approximation drift (expert deferral, quantised kernels, batch-size
dependent CPU paths) that a coherence check would miss: same prompts, same
greedy decoding, compare token-by-token and logprob-by-logprob against a saved
baseline from another server configuration.

  # baseline on config A
  python3 scripts/greedy-logprobs.py --output evidence/x/greedy-d0.json
  # on config B, compare
  python3 scripts/greedy-logprobs.py --output evidence/x/greedy-d2.json --compare evidence/x/greedy-d0.json
  # the same prompt inside a batch of K (K-1 filler streams), to test batch-size dependence
  python3 scripts/greedy-logprobs.py --batch 4 --output evidence/x/greedy-bs4.json --compare evidence/x/greedy-d0.json

CAVEAT (E21 result 7): on this hybrid the decode path is not bit-reproducible,
and greedy picks flip at near-tie tokens even between two runs of the same
server (D=0 vs D=0: token changes on 2/4 prompts, max |Δlogp| ≤ 0.32). Token
divergence therefore says nothing about approximation quality. Use this only
for gross sanity (garbage, NaN, wrong language); for a real quality number use
nll-score.py (teacher-forced NLL, deterministic on the prefill path).
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import urllib.request

PROMPTS = [
    "Explain how a CPU cache hierarchy (L1, L2, L3) works and why it matters for performance.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then explain its complexity.",
    "What were the main causes of the fall of the Western Roman Empire? Answer in three short paragraphs.",
    "A train leaves at 09:40 and arrives at 13:05. How long is the journey? Show the arithmetic.",
]
FILLERS = [
    "Describe the water cycle in detail.",
    "Write a short story about a lighthouse keeper.",
    "List 20 prime numbers and explain the sieve of Eratosthenes.",
    "Summarize the plot of Hamlet.",
    "Explain photosynthesis to a ten-year-old.",
    "What is the difference between TCP and UDP?",
    "Give a recipe for sourdough bread.",
]


def ask(url: str, model: str, prompt: str, n: int, logprobs: bool, timeout: int):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0,
        "logprobs": logprobs,
        "top_logprobs": 1 if logprobs else None,
        "chat_template_kwargs": {"thinking": False},
    }
    body = {k: v for k, v in body.items() if v is not None}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    ch = r["choices"][0]
    if not logprobs or not ch.get("logprobs"):
        return []
    return [(c["token"], round(c["logprob"], 4)) for c in ch["logprobs"]["content"]]


def compare(a, b):
    """a, b: list of (token, logprob). Returns (first divergent index or None, max|Δ| on common prefix, mean|Δ|)."""
    n = min(len(a), len(b))
    first = None
    diffs = []
    for i in range(n):
        if a[i][0] != b[i][0]:
            first = i
            break
        diffs.append(abs(a[i][1] - b[i][1]))
    return first, (max(diffs) if diffs else 0.0), (sum(diffs) / len(diffs) if diffs else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41-flash-hybrid")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--batch", type=int, default=1, help="run each prompt inside a batch of this size (fillers make up the rest)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--output", required=True)
    ap.add_argument("--compare", help="baseline JSON from a previous run")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    results = {}
    for p in PROMPTS:
        if a.batch <= 1:
            results[p] = ask(a.url, a.model, p, a.max_tokens, True, a.timeout)
        else:
            with cf.ThreadPoolExecutor(a.batch) as ex:
                main_f = ex.submit(ask, a.url, a.model, p, a.max_tokens, True, a.timeout)
                fill = [ex.submit(ask, a.url, a.model, f, a.max_tokens * 3, False, a.timeout)
                        for f in FILLERS[: a.batch - 1]]
                results[p] = main_f.result()
                for f in fill:
                    f.result()
    out = {"label": a.label, "model": a.model, "max_tokens": a.max_tokens, "batch": a.batch, "results": results}
    pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(out, indent=1))

    for p, toks in results.items():
        print(f"[{p[:50]}...] {''.join(t for t, _ in toks)!r}")
    if a.compare:
        base = json.loads(pathlib.Path(a.compare).read_text())["results"]
        print(f"\ncompare vs {a.compare} ({a.batch=} vs baseline batch {json.loads(pathlib.Path(a.compare).read_text()).get('batch')}):")
        worst = 0.0
        for p, toks in results.items():
            if p not in base:
                continue
            first, mx, mean = compare(base[p], toks)
            worst = max(worst, mx)
            status = "same tokens" if first is None else f"DIVERGES at token {first} ('{base[p][first][0]}' -> '{toks[first][0]}')"
            print(f"  {p[:50]:50s} {status}; max|Δlogp| {mx:.3f} mean {mean:.3f} over common prefix")
        print(f"  worst max|Δlogp| {worst:.3f}  (bs=1 run-to-run noise ≈ 0.06)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

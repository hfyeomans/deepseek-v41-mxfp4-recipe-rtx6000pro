#!/usr/bin/env python3
"""Decode throughput at fixed concurrency levels.

Method (matches the GLM and V4-Flash sibling recipes so numbers are comparable):
  - fixed code-agent style fixture, temperature 0.2, fixed seed, thinking off
  - --max-tokens forced output per sequence (default 256); min_tokens where supported
  - --warmup runs discarded, then --runs measured runs per concurrency level
  - each sequence is timed from its own first to last streamed token; aggregate
    decode = sum of per-sequence rates; TTFT excluded and reported separately
  - median / min / max across runs

If the server exposes Prometheus /metrics with spec-decode counters, the
acceptance rate delta is recorded per run (SGLang and vLLM both do).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import chat_stream, get_json  # noqa: E402

FIXTURE = (
    "You are a senior engineer. Review the following Python module for correctness and performance, then rewrite it "
    "with type hints, docstrings, and a small pytest suite. Explain each change briefly.\n\n"
    "```python\nimport os, json\n\ndef load(path):\n    d = {}\n    for f in os.listdir(path):\n        if f.endswith('.json'):\n"
    "            d[f] = json.load(open(os.path.join(path, f)))\n    return d\n\ndef merge(a, b):\n    for k in b:\n"
    "        if k in a and isinstance(a[k], dict):\n            merge(a[k], b[k])\n        else:\n            a[k] = b[k]\n"
    "    return a\n\ndef main():\n    cfg = merge(load('base'), load('override'))\n    print(json.dumps(cfg))\n```\n"
)

SPEC_PATTERNS = {
    "accepted": re.compile(r"^(?:sglang:|vllm:)?spec_decode_num_accepted_tokens_total(?:\{[^}]*\})?\s+([0-9.e+]+)", re.M),
    "draft": re.compile(r"^(?:sglang:|vllm:)?spec_decode_num_draft_tokens_total(?:\{[^}]*\})?\s+([0-9.e+]+)", re.M),
}


def spec_counters(url: str) -> dict[str, float] | None:
    try:
        text = urllib.request.urlopen(f"{url}/metrics", timeout=5).read().decode()
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for k, rx in SPEC_PATTERNS.items():
        vals = [float(m.group(1)) for m in rx.finditer(text)]
        if vals:
            out[k] = sum(vals)
    return out or None


def one_run(url: str, model: str, concurrency: int, max_tokens: int, seed: int) -> dict:
    before = spec_counters(url)
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(chat_stream, url, model, [{"role": "user", "content": FIXTURE}], max_tokens=max_tokens,
                          temperature=0.2, seed=seed + i, thinking=False,
                          extra={"min_tokens": max_tokens, "ignore_eos": True}) for i in range(concurrency)]
        results = [f.result() for f in futs]
    wall_s = time.perf_counter() - t0
    after = spec_counters(url)
    rates = [r.decode_tps() or 0.0 for r in results]
    # Wall-clock aggregate: honest when the server queues requests (e.g. max-num-seqs
    # below the concurrency level); the summed per-sequence rate is not, because
    # queued sequences run one after another yet each still reports its own full rate.
    total_tokens = sum(r.completion_tokens or 0 for r in results)
    wall_tps = total_tokens / wall_s if wall_s > 0 else 0.0
    acc = None
    if before and after and "draft" in before and "draft" in after:
        d = after["draft"] - before["draft"]
        a = after.get("accepted", 0) - before.get("accepted", 0)
        acc = (a / d) if d > 0 else None
    return {
        "aggregate_decode_tps": sum(rates),
        "wall_tps": wall_tps,
        "wall_s": wall_s,
        "per_seq_tps": rates,
        "ttft_s": [r.ttft_s for r in results],
        "completion_tokens": [r.completion_tokens for r in results],
        "finish_reasons": [r.finish_reason for r in results],
        "spec_acceptance": acc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    levels = [int(x) for x in args.concurrency.split(",")]
    report = {"url": args.url, "model": model, "max_tokens": args.max_tokens, "runs": args.runs, "warmup": args.warmup,
              "method": "per-sequence first-to-last-token rate summed (median_tps); total tokens / batch wall time incl. TTFT (median_wall_tps)",
              "levels": []}
    print(f"model={model}  runs={args.runs} warmup={args.warmup} max_tokens={args.max_tokens}")
    print(f"{'C':>3} {'median tok/s':>13} {'min':>9} {'max':>9} {'wall tok/s':>11} {'med TTFT s':>11} {'accept':>7}")
    for c in levels:
        for _ in range(args.warmup):
            one_run(args.url, model, c, args.max_tokens, args.seed)
        runs = [one_run(args.url, model, c, args.max_tokens, args.seed + 100 * i) for i in range(args.runs)]
        aggs = [r["aggregate_decode_tps"] for r in runs]
        walls = [r["wall_tps"] for r in runs]
        ttfts = [t for r in runs for t in r["ttft_s"]]
        accs = [r["spec_acceptance"] for r in runs if r["spec_acceptance"] is not None]
        level = {"concurrency": c, "median_tps": statistics.median(aggs), "min_tps": min(aggs), "max_tps": max(aggs),
                 "median_wall_tps": statistics.median(walls),
                 "median_ttft_s": statistics.median(ttfts), "median_spec_acceptance": statistics.median(accs) if accs else None,
                 "runs": runs}
        report["levels"].append(level)
        acc_s = f"{level['median_spec_acceptance'] * 100:6.2f}%" if accs else "    n/a"
        print(f"{c:>3} {level['median_tps']:13.2f} {level['min_tps']:9.2f} {level['max_tps']:9.2f} {level['median_wall_tps']:11.2f} "
              f"{level['median_ttft_s']:11.3f} {acc_s}")
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

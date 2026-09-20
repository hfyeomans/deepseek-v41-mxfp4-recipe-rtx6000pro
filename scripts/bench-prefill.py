#!/usr/bin/env python3
"""Prefill throughput versus prompt length.

For each size, builds a unique prompt of roughly that many tokens (seeded, so
prefix caching cannot help across runs), requests a single output token, and
records TTFT. Effective prefill = server-reported prompt_tokens / TTFT. TTFT
includes server tokenization and the one-token handoff; that is stated in
docs/BENCHMARKS.md rather than subtracted.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import chat_stream, get_json  # noqa: E402

WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit", "timber", "quartz",
         "velvet", "canyon", "pixel", "saffron", "anchor", "glacier", "marble", "cobalt", "ember", "willow",
         "compile", "buffer", "kernel", "socket", "thread", "cache", "vector", "schema", "token", "stride")


def make_prompt(tokens: int, rng: random.Random) -> str:
    body = " ".join(rng.choice(WORDS) + (str(rng.randrange(1000)) if rng.random() < 0.2 else "") for _ in range(int(tokens / 1.3)))
    return body + "\n\nReply with the single word DONE."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", default="8192,32768,131072,262144")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    sizes = [int(x) for x in args.tokens.split(",")]
    report = {"url": args.url, "model": model, "runs": args.runs, "method": "unique prompts; prompt_tokens/TTFT; TTFT includes tokenization", "sizes": []}
    print(f"{'target tok':>11} {'actual tok':>11} {'median prefill tok/s':>21} {'median TTFT s':>14}")
    for n in sizes:
        runs = []
        for i in range(args.runs):
            rng = random.Random(args.seed + 1000 * n + i)
            r = chat_stream(args.url, model, [{"role": "user", "content": make_prompt(n, rng)}], max_tokens=1, thinking=False, timeout=7200)
            pt = r.prompt_tokens or 0
            runs.append({"prompt_tokens": pt, "ttft_s": r.ttft_s, "prefill_tps": (pt / r.ttft_s) if pt else None})
        tps = [x["prefill_tps"] for x in runs if x["prefill_tps"]]
        size = {"target_tokens": n, "median_prompt_tokens": statistics.median(x["prompt_tokens"] for x in runs),
                "median_prefill_tps": statistics.median(tps) if tps else None,
                "median_ttft_s": statistics.median(x["ttft_s"] for x in runs), "runs": runs}
        report["sizes"].append(size)
        print(f"{n:>11,} {int(size['median_prompt_tokens']):>11,} {size['median_prefill_tps'] or 0:21.2f} {size['median_ttft_s']:14.3f}")
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

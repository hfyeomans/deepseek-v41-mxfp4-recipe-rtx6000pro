#!/usr/bin/env python3
"""Prefill throughput versus prompt length.

For each size, builds a unique prompt of roughly that many tokens (seeded, so
prefix caching cannot help across runs), requests a single output token, and
records TTFT. Effective prefill = server-reported prompt_tokens / TTFT. TTFT
includes server tokenization and the one-token handoff; that is stated in
docs/BENCHMARKS.md rather than subtracted.

Cold / warm (T3.19f): the first pass after a serve runs against whatever page-cache
state the box is in (the Engram tables are file-backed on NVMe, so a cold cache
costs prefill; T3.19d saw 1,471-1,504 vs 1,820 tok/s at 16K on unrecorded cache
state). `--warm-pass` repeats every size with fresh seeds (a repeat of the same
prompts would hit SGLang's radix cache, not the page cache) and reports both
passes with the Engram shards' resident fraction sampled before each. The
top-level `sizes` stay the cold pass so older readers keep working.
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
from evidence_manifest import engram_residency, locate, load, stamp_or_exit  # noqa: E402
from openai_client import chat_stream, get_json  # noqa: E402

WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit", "timber", "quartz",
         "velvet", "canyon", "pixel", "saffron", "anchor", "glacier", "marble", "cobalt", "ember", "willow",
         "compile", "buffer", "kernel", "socket", "thread", "cache", "vector", "schema", "token", "stride")


def make_prompt(tokens: int, rng: random.Random) -> str:
    body = " ".join(rng.choice(WORDS) + (str(rng.randrange(1000)) if rng.random() < 0.2 else "") for _ in range(int(tokens / 1.3)))
    return body + "\n\nReply with the single word DONE."


WARM_SEED_OFFSET = 500  # fresh prompts for the warm pass; same seeds would hit the radix cache


def engram_state() -> list[dict] | None:
    """Resident fraction of the Engram shards, from the manifest's model location (None without one)."""
    try:
        path = locate()
        if path is None or not path.exists():
            return None
        cfg = load(path)["config"]
        return engram_residency(cfg["HF_MODEL_DIR"], cfg["SNAPSHOT"])
    except Exception:  # noqa: BLE001 - residency is context, never a reason to skip the benchmark
        return None


def run_pass(url: str, model: str, sizes: list[int], runs: int, seed: int) -> list[dict]:
    out = []
    for n in sizes:
        rows = []
        for i in range(runs):
            rng = random.Random(seed + 1000 * n + i)
            r = chat_stream(url, model, [{"role": "user", "content": make_prompt(n, rng)}], max_tokens=1, thinking=False, timeout=7200)
            pt = r.prompt_tokens or 0
            rows.append({"prompt_tokens": pt, "ttft_s": r.ttft_s, "prefill_tps": (pt / r.ttft_s) if pt else None})
        tps = [x["prefill_tps"] for x in rows if x["prefill_tps"]]
        size = {"target_tokens": n, "median_prompt_tokens": statistics.median(x["prompt_tokens"] for x in rows),
                "median_prefill_tps": statistics.median(tps) if tps else None,
                "median_ttft_s": statistics.median(x["ttft_s"] for x in rows), "runs": rows}
        out.append(size)
        tps_str = f"{size['median_prefill_tps']:.2f}" if size['median_prefill_tps'] is not None else "n/a"
        print(f"{n:>11,} {int(size['median_prompt_tokens']):>11,} {tps_str:>21} {size['median_ttft_s']:14.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", default="8192,32768,131072,262144")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--warm-pass", action="store_true",
                    help="after the cold pass, repeat every size with fresh seeds (seed + %d) and report both" % WARM_SEED_OFFSET)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    sizes = [int(x) for x in args.tokens.split(",")]
    report = {"url": args.url, "model": model, "runs": args.runs, "seed": args.seed,
              "method": "unique prompts; prompt_tokens/TTFT; TTFT includes tokenization; sizes = cold pass", "sizes": [],
              "passes": []}
    print(f"{'target tok':>11} {'actual tok':>11} {'median prefill tok/s':>21} {'median TTFT s':>14}")
    for name, seed in (("cold", args.seed), ("warm", args.seed + WARM_SEED_OFFSET))[: 2 if args.warm_pass else 1]:
        if args.warm_pass:
            print(f"-- {name} pass (seed {seed})")
        before = engram_state()
        rows = run_pass(args.url, model, sizes, args.runs, seed)
        report["passes"].append({"pass": name, "seed": seed, "engram_residency_before": before, "sizes": rows})
    report["sizes"] = report["passes"][0]["sizes"]
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.output:
        stamp_or_exit(report, args.output, "bench-prefill.py")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

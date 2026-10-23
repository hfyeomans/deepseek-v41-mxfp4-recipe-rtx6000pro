#!/usr/bin/env python3
"""Decode benchmark harness template (C1/C4) — the contract every new recipe must honor so numbers
compare across recipes. Full reference implementation: the parent repo's scripts/bench-decode.py.

Contract (from METHODOLOGY.md section 8):
- fixed fixtures; temperature 0.2; fixed seed; thinking off; forced output length
  (min_tokens = max_tokens, ignore_eos)
- headline: 512 tokens, 1 warmup, 6 runs at C1, 3 at C4 (runs/warmups/tokens read from the stamped
  JSON, never typed into captions)
- C1 reported as median per-sequence AND wall tok/s (total completion tokens / batch wall incl. TTFT)
- C2+ measured with --distinct-prompts (same-prompt concurrency overstates MoE expert batching 2-3x);
  --same-prompt exists only for the batch-consistency gate and is labelled wherever its tok/s appears
- speculative acceptance = counter delta per run, reported beside the number
- power sampled alongside and summarized as busy-median
- every output JSON stamped with evidence.manifest_hash and copied beside the serve manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time


def openai_client():  # pragma: no cover - wired to your endpoint helper
    """Return a streaming client for the serve. The reference recipe reads URL/MODEL from
    `serve_config.py endpoint` so the probe can never address a different port than the serve."""
    raise NotImplementedError("wire to your OpenAI-compatible client; read URL/MODEL from serve_config.py endpoint")


def decode_one(client, prompt: str, seed: int, out_tokens: int) -> tuple[float, float, int]:
    """One streamed decode. Returns (per-seq tok/s, wall tok/s, completion tokens).

    Per-seq rate = (completion tokens - 1) / (last-token time - first-token time): TTFT excluded.
    Wall rate   = completion tokens / batch wall time: TTFT included (the honest number).
    """
    raise NotImplementedError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--concurrency", type=int, default=1, help="C1..Cn")
    ap.add_argument("--output-tokens", type=int, default=512)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=6 if 1 else 3)
    ap.add_argument("--distinct-prompts", action="store_true",
                    help="REQUIRED for any C2+ throughput claim (see contract)")
    ap.add_argument("--same-prompt", action="store_true",
                    help="consistency gate only; label the output accordingly")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.concurrency > 1 and not args.distinct_prompts and not args.same_prompt:
        print("refusing: C2+ throughput must be measured with --distinct-prompts "
              "(--same-prompt is the consistency gate only)")
        return 2

    client = openai_client()
    rng = random.Random(args.seed)
    prompts = [f"fixture-{i}-{rng.random()}" for i in range(args.concurrency)]  # real fixtures in production
    results = []
    for run in range(args.warmups + args.runs):
        t0 = time.time()
        rows = [decode_one(client, p, args.seed + run, args.output_tokens) for p in prompts]
        wall = time.time() - t0
        total_tokens = sum(r[2] for r in rows)
        results.append({
            "run": run - args.warmups if run >= args.warmups else f"warmup{run}",
            "per_seq_tok_s": [r[0] for r in rows],
            "wall_tok_s": total_tokens / wall if wall else 0.0,
            "aggregate_tok_s": sum(r[0] for r in rows),
            "concurrency": args.concurrency,
            "distinct_prompts": bool(args.distinct_prompts),
            "same_prompt": bool(args.same_prompt),
        })
    out = {
        "benchmark": "decode",
        "args": vars(args),
        "results": results[args.warmups:],
        # stamp with evidence.manifest_hash + copy manifest.json beside (evidence_manifest.py)
        "evidence": {"manifest_hash": None},
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

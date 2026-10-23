#!/usr/bin/env python3
"""Prefill / TTFT benchmark harness template — the cold/warm contract. Full reference
implementation: the parent repo's scripts/bench-prefill.py.

Contract (from METHODOLOGY.md section 8):
- unique seeded prompts per run (the prefix/radix cache must not hit), one output token, 3 runs
- effective prefill = server prompt_tokens / client TTFT; TTFT includes tokenization + one-token handoff
- cold / warm PAIR: the warm pass repeats every size with fresh seeds (seed+500) and the hot-table /
  page-cache residency is recorded before each pass — a comparator without recorded cache state is
  not a comparator (the reference project drifted 1,471-1,504 vs 1,820 tok/s at 16K on unrecorded state)
- report the cold-vs-warm gap itself: it is a finding (+1.7-3.1 s per cold request in the reference)
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def openai_client():  # pragma: no cover
    raise NotImplementedError("wire to your OpenAI-compatible client (same endpoint helper as bench-decode)")


def prefill_one(client, prompt_tokens: int, seed: int) -> dict:
    """One prefill. Returns {prompt_tokens, ttft_s, effective_tok_s}."""
    raise NotImplementedError


def page_cache_residency() -> dict:
    """Residency of the hot files (lookup tables / shards) before a pass. `fincore`-style: keys are
    file basenames, values are resident fractions. MUST be recorded beside every pass."""
    raise NotImplementedError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-tokens", type=int, nargs="+", default=[8192, 16384, 32768])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warm-pass", action="store_true",
                    help="run the warm pass (fresh seeds, seed+500) after the cold pass")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    client = openai_client()
    out = {"benchmark": "prefill", "args": vars(args), "passes": []}
    for pass_name, seed in (("cold", args.seed),) + ((("warm", args.seed + 500),) if args.warm_pass else ()):
        out["passes"].append({
            "pass": pass_name,
            "residency_before": page_cache_residency(),
            "sizes": [
                {"prompt_tokens": n,
                 "runs": [prefill_one(client, n, seed + run * 997) for run in range(args.runs)]}
                for n in args.prompt_tokens
            ],
        })
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

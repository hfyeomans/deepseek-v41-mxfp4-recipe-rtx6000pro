#!/usr/bin/env python3
"""Long-reasoning decode traffic: hard prompts with thinking on, so the model produces
thousands of reasoning tokens per request before its answer (T3.8b soak: the expert map
is fitted on prose / code counts and HumanEval reasoning is short, median ~300 tokens).

  python3 scripts/bench-reasoning.py --url http://127.0.0.1:8000 --model dsv41-flash-hybrid \
      --concurrency 4 --max-tokens 16384 --output evidence/<dir>/reasoning-c4.json

Reports per prompt: completion tokens (reasoning + answer, from usage), reasoning characters,
decode tok/s from the first streamed token, finish reason, and whether the final content
contains the known answer where one exists (a spot check on the map's quality under long
reasoning, not a benchmark score). Greedy, fixed seed, thinking on.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import chat_stream, get_json  # noqa: E402

# (prompt, answer substring or None). Answers verified with a script where numeric.
PROMPTS: list[tuple[str, str | None]] = [
    ("Find the remainder when 2^2024 is divided by 1000. Show the full derivation, then give the final "
     "answer as a single integer on the last line.", "216"),
    ("How many ordered pairs (a, b) of positive integers satisfy lcm(a, b) = 2^3 * 3^2 * 5? Derive it "
     "carefully and give the final count on the last line.", "105"),
    ("Count the lattice paths from (0,0) to (6,6) using only unit steps right (+1 in x) or up (+1 in y) "
     "that never visit any of the four points (3,3), (3,4), (4,3), (4,4). Reason it through and give the "
     "final number on the last line.", "224"),
    ("Does there exist a positive integer n such that n^2 + 1 is divisible by 2025? If yes give the "
     "smallest; if no, prove it. End with one line: either the integer or the words NO SUCH N.",
     "NO SUCH N"),
    ("How many five-digit positive integers have strictly increasing digits (left to right) and a digit "
     "sum divisible by 3? Work it out and give the final count on the last line.", "42"),
    ("Let f(n) be the number of trailing zeros of n!. Find the smallest positive integer n with "
     "f(n) = 100. Explain, then give n on the last line.", "405"),
    ("Design an O(n) algorithm that, given an integer array and an integer k, counts the subarrays whose "
     "sum is divisible by k. Prove its correctness, analyse the worst case including negative numbers, "
     "and give a complete Python implementation with tests.", None),
    ("The following Python snippet is meant to be a thread-safe LRU cache but has at least two "
     "concurrency bugs. Identify every bug, explain the interleavings that trigger each, and give a "
     "corrected implementation.\n\n```python\nimport threading\nfrom collections import OrderedDict\n\n"
     "class LRU:\n    def __init__(self, cap):\n        self.cap = cap\n        self.d = OrderedDict()\n"
     "        self.lock = threading.Lock()\n    def get(self, k):\n        if k in self.d:\n"
     "            with self.lock:\n                self.d.move_to_end(k)\n            return self.d[k]\n"
     "        return None\n    def put(self, k, v):\n        self.d[k] = v\n        with self.lock:\n"
     "            if len(self.d) > self.cap:\n                self.d.popitem(last=False)\n```", None),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    prompts = PROMPTS[: args.limit]

    def one(i: int) -> dict:
        prompt, answer = prompts[i]
        r = chat_stream(args.url, model, [{"role": "user", "content": prompt}], max_tokens=args.max_tokens,
                        temperature=0.0, seed=args.seed, thinking=not args.no_thinking)
        tail = "\n".join(r.text.strip().splitlines()[-3:])
        return {"i": i, "prompt": prompt[:60], "completion_tokens": r.completion_tokens,
                "reasoning_chars": len(r.reasoning), "answer_chars": len(r.text),
                "ttft_s": r.ttft_s, "total_s": r.total_s, "decode_tps": r.decode_tps(),
                "finish_reason": r.finish_reason, "expected": answer,
                "answer_found": (answer in r.text) if answer else None, "answer_tail": tail,
                "text": r.text, "reasoning": r.reasoning}

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        rows = list(ex.map(one, range(len(prompts))))
    wall = time.perf_counter() - t0
    toks = [x["completion_tokens"] or 0 for x in rows]
    tps = [x["decode_tps"] for x in rows if x["decode_tps"]]
    checked = [x for x in rows if x["answer_found"] is not None]
    summary = {"model": model, "concurrency": args.concurrency, "max_tokens": args.max_tokens, "seed": args.seed,
               "thinking": not args.no_thinking, "n": len(rows), "wall_s": wall,
               "median_completion_tokens": statistics.median(toks), "max_completion_tokens": max(toks),
               "total_completion_tokens": sum(toks),
               "median_reasoning_chars": statistics.median([x["reasoning_chars"] for x in rows]),
               "median_decode_tps": statistics.median(tps) if tps else None,
               "min_decode_tps": min(tps) if tps else None,
               "aggregate_tps": sum(toks) / wall,
               "answers_found": f"{sum(1 for x in checked if x['answer_found'])}/{len(checked)}",
               "finish_reasons": {k: sum(1 for x in rows if x["finish_reason"] == k)
                                  for k in sorted({str(x["finish_reason"]) for x in rows})}}
    print(f"{len(rows)} prompts at C{args.concurrency} in {wall:.0f} s: completion tokens median "
          f"{summary['median_completion_tokens']:.0f} (max {summary['max_completion_tokens']}, total {sum(toks)}), "
          f"decode median {summary['median_decode_tps']:.2f} tok/s per stream (min {summary['min_decode_tps']:.2f}), "
          f"aggregate {summary['aggregate_tps']:.1f}, answers {summary['answers_found']}, finish {summary['finish_reasons']}")
    for x in rows:
        print(f"  [{x['i']}] {x['completion_tokens']} tok, reasoning {x['reasoning_chars']} chars, "
              f"{(x['decode_tps'] or 0):.1f} tok/s, {x['finish_reason']}, "
              f"expected {x['expected']!r}: {x['answer_found']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = stamp_or_exit({"summary": summary, "rows": rows}, args.output, "bench-reasoning.py")
        args.output.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

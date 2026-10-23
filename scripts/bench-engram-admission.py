#!/usr/bin/env python3
"""T3.49 gate probe: does an admitted request's first-chunk Engram fetch overlap
the decode in flight?

Each round: a decode stream (fresh fixture) starts, and after `--delay` s a
prefill request of `--tokens` fresh tokens (make_prompt from bench-prefill, so
its Engram rows are page-cache cold for the round's seed) is admitted. Reports
the prefill's TTFT and the decode's tok/s alongside. With
ENGRAM_ADMISSION_PREFETCH=1 the prefill's chunk-0 rows ride the shared worker
during the decode, so the prefill TTFT should shrink toward the warm-page
value; the control serve pays the full cold fetch inside the prefill (E40:
+1.7-3.1 s per request, flat in prompt length).

The decode runs on a worker thread (chat_stream blocks until completion); the
prefill runs on the main thread so its timing is uncontended by the client.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import StreamResult, chat_stream  # noqa: E402

_spec = importlib.util.spec_from_file_location("bench_prefill", Path(__file__).resolve().parent / "bench-prefill.py")
_bench_prefill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench_prefill)
make_prompt = _bench_prefill.make_prompt

FIXTURE = (
    "You are a senior engineer. Review the following Python module for correctness and performance, then rewrite it "
    "with type hints, docstrings, and a small pytest suite. Explain each change briefly.\n\n"
    "```python\nimport os, json\n\ndef load(path):\n    d = {}\n    for f in os.listdir(path):\n"
    "        if f.endswith('.json'):\n            d[f] = json.load(open(os.path.join(path, f)))\n    return d\n```\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41-flash-hybrid")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--delay", type=float, default=6.0, help="seconds of decode before the prefill is admitted")
    ap.add_argument("--decode-max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    rounds = []
    for i in range(args.pairs):
        rng = random.Random(args.seed + 1000 * i)
        holder: list[StreamResult] = []
        dec = threading.Thread(
            target=lambda: holder.append(
                chat_stream(args.url, args.model,
                            [{"role": "user", "content": FIXTURE}],
                            max_tokens=args.decode_max_tokens, temperature=0.2,
                            seed=rng.randrange(10**6), thinking=False)
            ),
            daemon=True,
        )
        dec.start()
        time.sleep(args.delay)
        pre_prompt = make_prompt(args.tokens, rng)
        pre = chat_stream(args.url, args.model, [{"role": "user", "content": pre_prompt}],
                          max_tokens=32, temperature=0.0, seed=rng.randrange(10**6), thinking=False)
        pre_ttft = pre.ttft_s
        dec.join(timeout=600)
        d = holder[0] if holder else None
        rounds.append({
            "round": i,
            "prefill_ttft_s": round(pre_ttft, 3),
            "prefill_total_s": round(pre.total_s, 3),
            "prefill_prompt_tokens": pre.prompt_tokens,
            "decode_ttft_s": round(d.ttft_s, 3) if d else None,
            "decode_tps": round(d.decode_tps(), 2) if d and d.decode_tps() else None,
            "decode_completion_tokens": d.completion_tokens if d else None,
        })
        print(f"round {i}: prefill_ttft {round(pre_ttft, 3)} s (prompt {pre.prompt_tokens}), "
              f"decode {d.completion_tokens if d else 0} tok at {round(d.decode_tps(), 2) if d and d.decode_tps() else 0} tok/s")
    report = {"args": vars(args), "rounds": rounds}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stamp_or_exit(report, args.output, "bench-engram-admission.py")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

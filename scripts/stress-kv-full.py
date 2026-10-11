#!/usr/bin/env python3
"""Worst-case VRAM check: fill every running slot with a near-context prompt.

Sends N concurrent requests, each with a unique ~T-token prompt and M output
tokens, so all N KV slots are near their cap while chunked prefill and decode
run together. Samples `nvidia-smi` memory.used once a second during the run and
reports the peak per GPU. Use it after any change that spends VRAM (expert
placement, chunk size, capture list) - a single 32K prefill at C1 is not the
worst case a served config admits.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import chat_stream, get_json  # noqa: E402

PROMPT_MIN_FRACTION = 0.95  # of --prompt-tokens, by the server's prompt_tokens (R14); the recipe lands at ~97.3 %
WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit", "timber", "quartz",
         "velvet", "canyon", "pixel", "saffron", "anchor", "glacier", "marble", "cobalt", "ember", "willow",
         "compile", "buffer", "kernel", "socket", "thread", "cache", "vector", "schema", "token", "stride")


def make_prompt(tokens: int, rng: random.Random) -> str:
    # Same recipe as bench-prefill.py: unique per seed so prefix caching cannot help.
    body = " ".join(rng.choice(WORDS) + (str(rng.randrange(1000)) if rng.random() < 0.2 else "") for _ in range(int(tokens / 1.3)))
    return body + "\n\nWrite a long, detailed essay about the words above."


def sample_vram(stop: threading.Event, out: list[dict]) -> None:
    while not stop.is_set():
        try:
            txt = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            row = {"t": time.time()}
            for line in txt.strip().splitlines():
                idx, used, total = [x.strip() for x in line.split(",")]
                row[f"gpu{idx}_used_mib"] = int(used)
                row[f"gpu{idx}_total_mib"] = int(total)
            out.append(row)
        except Exception as exc:  # noqa: BLE001 - sampling must never abort the stress run
            out.append({"t": time.time(), "error": str(exc)})
        stop.wait(1.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prompt-tokens", type=int, default=30000)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    samples: list[dict] = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_vram, args=(stop, samples), daemon=True)
    sampler.start()

    def one(i: int) -> dict:
        rng = random.Random(args.seed + i)
        t0 = time.time()
        r = chat_stream(args.url, model, [{"role": "user", "content": make_prompt(args.prompt_tokens, rng)}],
                        max_tokens=args.max_tokens, thinking=False, timeout=7200)
        return {"stream": i, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                "ttft_s": r.ttft_s, "total_s": r.total_s, "finish_reason": r.finish_reason, "wall_s": time.time() - t0}

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(one, range(args.concurrency)))
    wall = time.time() - t_start
    stop.set()
    sampler.join(timeout=5)

    peak = {}
    for s in samples:
        for k, v in s.items():
            if k.endswith("_used_mib"):
                peak[k] = max(peak.get(k, 0), v)
    first_good = next((s for s in samples if "error" not in s), {})
    totals = {k: v for k, v in first_good.items() if k.endswith("_total_mib")}
    # Fail closed (R14): every slot must have finished, prefilled at least 95 % of the target
    # (a request the server truncated or that tokenised short did not exercise the KV it claims)
    # and produced at least one token (decode ran with every slot full, which is the point).
    min_prompt = int(args.prompt_tokens * PROMPT_MIN_FRACTION)
    for r in results:
        r["ok"] = (r["finish_reason"] is not None and (r["prompt_tokens"] or 0) >= min_prompt
                   and (r["completion_tokens"] or 0) >= 1)
    ok = all(r["ok"] for r in results)
    report = {"url": args.url, "model": model, "concurrency": args.concurrency, "prompt_tokens_target": args.prompt_tokens,
              "prompt_tokens_min": min_prompt,
              "max_tokens": args.max_tokens, "seed": args.seed, "wall_s": wall, "all_finished": ok,
              "peak_vram_used_mib": peak, "vram_total_mib": totals, "results": results, "vram_samples": samples,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(f"{'stream':>6} {'prompt tok':>10} {'out tok':>7} {'TTFT s':>8} {'total s':>8} finish  ok (prompt >= {min_prompt:,}, out >= 1)")
    for r in results:
        print(f"{r['stream']:>6} {r['prompt_tokens'] or 0:>10,} {r['completion_tokens'] or 0:>7} {r['ttft_s']:8.2f} {r['total_s']:8.2f} "
              f"{r['finish_reason']}  {'ok' if r['ok'] else 'FAIL'}")
    for k in sorted(peak):
        g = k.split("_")[0]
        tot = totals.get(f"{g}_total_mib")
        free = f", min free {tot - peak[k]:,} MiB" if tot else ""
        print(f"peak {g}: {peak[k]:,} MiB used{free}")
    print(f"wall {wall:.1f} s, all slots finished and filled: {ok}")
    if args.output:
        stamp_or_exit(report, args.output, "stress-kv-full.py")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

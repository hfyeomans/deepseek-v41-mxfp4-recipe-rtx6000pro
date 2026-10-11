#!/usr/bin/env python3
"""Where are the GPU-idle gaps? For each gap >= floor between GPU events in a
torch-profiler trace, report the kernel/memcpy before and after the gap and
the host-side CUDA runtime calls (memcpy, host-func launch, sync) that overlap
it. Companion to step-profile.py: that one says *how much* idle time there
is; this one says *what the GPU was waiting on*.

  python3 scripts/gap-context.py <trace.json.gz> [--floor-us 100] [--top 12]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def short(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--floor-us", type=float, default=100.0)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    ev = [e for e in load(args.trace) if e.get("ph") == "X"]
    gpu = sorted(
        (e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
        key=lambda e: e["ts"],
    )
    host = sorted(
        (e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver")),
        key=lambda e: e["ts"],
    )
    pyops = sorted((e for e in ev if e.get("cat") in ("cpu_op", "user_annotation")), key=lambda e: e["ts"])

    if not gpu:
        raise SystemExit(f"{args.trace}: no kernel / memcpy / memset events (is this a CPU-only trace?)")
    gaps = []
    cur_end = gpu[0]["ts"] + gpu[0]["dur"]
    prev = gpu[0]
    for e in gpu[1:]:
        s = e["ts"]
        if s > cur_end and (s - cur_end) >= args.floor_us:
            gaps.append((cur_end, s, prev, e))
        if s + e["dur"] > cur_end:
            cur_end = s + e["dur"]
            prev = e

    print(f"{len(gaps)} gaps >= {args.floor_us} us, total {sum(e - s for s, e, _, _ in gaps) / 1000:.2f} ms")

    pair = collections.Counter()
    pair_ms = collections.Counter()
    hostc = collections.Counter()
    hostc_ms = collections.Counter()
    opc = collections.Counter()
    for s, e, a, b in gaps:
        key = (short(a["name"], 45), short(b["name"], 45))
        pair[key] += 1
        pair_ms[key] += (e - s) / 1000
        # host runtime calls overlapping the gap
        seen = set()
        for h in host:
            hs, he = h["ts"], h["ts"] + h["dur"]
            if he < s:
                continue
            if hs > e:
                break
            seen.add(h["name"])
        for n in seen:
            hostc[n] += 1
            hostc_ms[n] += (e - s) / 1000
        # innermost-ish python/cpu op spanning the gap midpoint
        mid = (s + e) / 2
        best = None
        for o in pyops:
            if o["ts"] <= mid <= o["ts"] + o["dur"]:
                if best is None or o["dur"] < best["dur"]:
                    best = o
            if o["ts"] > mid:
                break
        if best is not None:
            opc[short(best["name"], 60)] += 1

    print("\nkernel before -> kernel after (count, total ms):")
    for k, n in pair.most_common(args.top):
        print(f"  {n:4d} {pair_ms[k]:8.2f} ms  {k[0]}  ->  {k[1]}")
    print("\nhost CUDA calls overlapping gaps (count, total gap ms):")
    for k, n in hostc.most_common(args.top):
        print(f"  {n:4d} {hostc_ms[k]:8.2f} ms  {k}")
    print("\nshortest cpu_op/annotation covering the gap midpoint:")
    for k, n in opc.most_common(args.top):
        print(f"  {n:4d}  {k}")

    print("\nlargest gaps:")
    for s, e, a, b in sorted(gaps, key=lambda g: -(g[1] - g[0]))[:8]:
        print(f"  {(e - s):8.0f} us  {short(a['name'], 40)} -> {short(b['name'], 40)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

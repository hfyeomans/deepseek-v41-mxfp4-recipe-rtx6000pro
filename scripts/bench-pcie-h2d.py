#!/usr/bin/env python3
"""Experiments E2 and E4: GPU memory bandwidth and PCIe host<->device bandwidth.

Requires torch with CUDA. The host Python has no torch; run inside a container:

  docker run --rm --gpus all -v "$PWD/scripts:/s:ro" -v "$PWD/results:/r" <torch-image> \
      python3 /s/bench-pcie-h2d.py --pinned --both --d2d --output /r/e4.json

Measures, per visible GPU and (with --both) all GPUs concurrently:
  H2D and D2H copy bandwidth for each --sizes buffer (pinned or pageable)
  D2D bandwidth (--d2d) as a GDDR7 proxy: bytes read + written per copy

Reports GiB/s medians and the Option B decode ceiling implied by the aggregate
H2D bandwidth: tokens/s <= aggregate_GiB_s / 1.64 GiB per token (39% of
routed experts streamed; see docs/FEASIBILITY.md section 4).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

try:
    import torch
except ImportError:
    print("torch is required; run inside a CUDA torch container (see docstring)", file=sys.stderr)
    sys.exit(2)

GIB = 2**30
OPTION_B_GIB_PER_TOKEN = 1.64


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 2**10, "M": 2**20, "G": 2**30}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def bw(fn, nbytes: int, iters: int, dev: int) -> float:
    torch.cuda.synchronize(dev)
    fn()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize(dev)
    return nbytes * iters / (time.perf_counter() - t0) / GIB


def per_device(dev: int, sizes: list[int], iters: int, pinned: bool, d2d: bool) -> dict:
    out = {"device": dev, "name": torch.cuda.get_device_name(dev), "h2d_gib_s": {}, "d2h_gib_s": {}, "d2d_gib_s": {}}
    with torch.cuda.device(dev):
        for n in sizes:
            host = torch.empty(n, dtype=torch.uint8, pin_memory=pinned)
            gpu = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dev}")
            out["h2d_gib_s"][n] = round(bw(lambda: gpu.copy_(host, non_blocking=True), n, iters, dev), 2)
            out["d2h_gib_s"][n] = round(bw(lambda: host.copy_(gpu, non_blocking=True), n, iters, dev), 2)
            if d2d:
                gpu2 = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dev}")
                out["d2d_gib_s"][n] = round(bw(lambda: gpu2.copy_(gpu), 2 * n, iters, dev), 2)
                del gpu2
            del host, gpu
            torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="256M,1G,4G")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--both", action="store_true", help="also run all GPUs concurrently")
    ap.add_argument("--d2d", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    sizes = [parse_size(s) for s in args.sizes.split(",")]
    n_dev = torch.cuda.device_count()
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "pinned": args.pinned, "iters": args.iters,
              "sizes": sizes, "devices": [], "concurrent": None}
    for d in range(n_dev):
        r = per_device(d, sizes, args.iters, args.pinned, args.d2d)
        report["devices"].append(r)
        print(f"GPU{d} {r['name']}: H2D {r['h2d_gib_s']}  D2H {r['d2h_gib_s']}  D2D {r['d2d_gib_s']}")

    if args.both and n_dev > 1:
        n = max(sizes)
        hosts = [torch.empty(n, dtype=torch.uint8, pin_memory=args.pinned) for _ in range(n_dev)]
        gpus = [torch.empty(n, dtype=torch.uint8, device=f"cuda:{d}") for d in range(n_dev)]
        results = [0.0] * n_dev

        def worker(d: int):
            with torch.cuda.device(d):
                results[d] = bw(lambda: gpus[d].copy_(hosts[d], non_blocking=True), n, args.iters, d)

        barrier_threads = [threading.Thread(target=worker, args=(d,)) for d in range(n_dev)]
        for t in barrier_threads:
            t.start()
        for t in barrier_threads:
            t.join()
        agg = sum(results)
        report["concurrent"] = {"size": n, "per_device_h2d_gib_s": [round(x, 2) for x in results], "aggregate_h2d_gib_s": round(agg, 2),
                                "option_b_decode_ceiling_tok_s": round(agg / OPTION_B_GIB_PER_TOKEN, 1)}
        print(f"concurrent H2D: per-device {report['concurrent']['per_device_h2d_gib_s']} aggregate {agg:.1f} GiB/s "
              f"-> Option B decode ceiling {agg / OPTION_B_GIB_PER_TOKEN:.0f} tok/s")
    else:
        single = statistics.median(v for d in report["devices"] for v in d["h2d_gib_s"].values())
        report["option_b_decode_ceiling_tok_s_single_link"] = round(single / OPTION_B_GIB_PER_TOKEN, 1)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

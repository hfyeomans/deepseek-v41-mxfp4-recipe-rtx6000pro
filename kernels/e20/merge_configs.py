#!/usr/bin/env python3
"""Merge two or more E20 tuning passes: per (shape, M) keep the config with the lower measured time.

    kernels/e20/merge_configs.py --out evidence/e20-dense-fp8-gemm/merged \
        evidence/e20-dense-fp8-gemm/pass1 evidence/e20-dense-fp8-gemm/pass2

Each input directory is a `tune_w8a8_block32.py --out` directory (results.json + configs/).  The
output directory gets the same layout plus results.json recording which pass won each M.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("passes", nargs="+", help="tuning output directories, in priority order for ties")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = []
    for d in args.passes:
        with open(os.path.join(d, "results.json")) as f:
            runs.append((d, json.load(f)))
    device_name = runs[0][1]["device_name"]
    bn, bk = runs[0][1]["block_shape"]

    merged = {"device_name": device_name, "block_shape": [bn, bk], "passes": args.passes, "shapes": {}}
    os.makedirs(os.path.join(args.out, "configs"), exist_ok=True)
    shapes = sorted({shape for _, r in runs for shape in r["shapes"]})
    for shape in shapes:
        per_m = {}
        for d, r in runs:
            for m, entry in r["shapes"].get(shape, {}).items():
                cur = per_m.get(m)
                if cur is None or entry["best_us"] < cur["best_us"]:
                    per_m[m] = {**entry, "from_pass": d}
        merged["shapes"][shape] = per_m
        N, K = (int(x.split("=")[1]) for x in shape.split(","))
        fname = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{bn}, {bk}].json"
        with open(os.path.join(args.out, "configs", fname), "w") as f:
            json.dump({m: per_m[m]["best_config"] for m in sorted(per_m, key=int)}, f, indent=4)
            f.write("\n")
        for m in sorted(per_m, key=int):
            e = per_m[m]
            print(f"{shape:>16} M={int(m):5d}: {e['best_us']:8.1f} us ({e['speedup']:5.2f}x)  <- {e['from_pass']}")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(merged, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Decide whether a DeepSeek-V4.1-Flash checkpoint fits a GPU/host memory layout.

Inputs are a budget JSON from scripts/checkpoint-budget.py plus a hardware
description (GPU count, GiB per GPU, host GiB). Output is a placement table
and a fits/does-not-fit verdict for each supported layout:

  gpu-only            everything resident in VRAM (what stock vLLM/SGLang do)
  engram-host         Engram tables in host DRAM (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1),
                      everything else in VRAM
  engram-nvme         Engram tables on NVMe with a host row cache, everything else in VRAM

The math is deliberately simple and conservative. Overheads are explicit
parameters so the numbers can be audited and re-run. Standard library only.

Exit status: 0 if the requested layout fits, 3 if it does not, 2 on input error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 2**30

# Default per-GPU reserve for CUDA context, NCCL buffers, cuBLAS/FlashInfer
# workspaces, CUDA-graph pools and activations. 0xSero's 4x RTX PRO 6000
# recipe and vLLM issue #56700 both land near 0.85-0.95 utilisation with 4-6
# GiB of non-weight, non-KV residency per GPU. 4 GiB is the optimistic end.
DEFAULT_GPU_RESERVE_GIB = 4.0
# KV cache: 890 bytes/token/sequence (vLLM recipe & LibertAI card; CSA2).
KV_BYTES_PER_TOKEN = 890
# Host reserve for OS, page cache, serving process, tokenizer, pinned staging.
DEFAULT_HOST_RESERVE_GIB = 24.0


def load_budget(path: Path) -> dict:
    data = json.loads(path.read_text())
    comps = data["components"]
    get = lambda k: comps.get(k, {"weight_bytes": 0, "scale_bytes": 0})
    tot = lambda k: get(k)["weight_bytes"] + get(k)["scale_bytes"]
    engram = tot("engram_table")
    dspark = tot("dspark")
    vision = tot("vision")
    total = data["header_total_bytes"]
    return {
        "source": f'{data["source"]}@{data["revision"]}',
        "total": total,
        "engram_tables": engram,
        "dspark": dspark,
        "vision": vision,
        "routed_experts": tot("routed_experts"),
        "resident_full": total - engram,
        "resident_text_no_dspark": total - engram - dspark - vision,
    }


def evaluate(budget: dict, *, gpus: int, gpu_gib: float, host_gib: float, util: float,
             gpu_reserve_gib: float, host_reserve_gib: float, context_tokens: int,
             sequences: int, include_dspark: bool, include_vision: bool, layout: str,
             engram_cache_gib: float) -> dict:
    resident = budget["resident_text_no_dspark"]
    if include_dspark:
        resident += budget["dspark"]
    if include_vision:
        resident += budget["vision"]
    if layout == "gpu-only":
        resident += budget["engram_tables"]

    kv = KV_BYTES_PER_TOKEN * context_tokens * sequences
    usable_vram = gpus * (gpu_gib * util - gpu_reserve_gib) * GIB
    vram_need = resident + kv
    vram_margin = usable_vram - vram_need

    if layout == "engram-host":
        host_need = budget["engram_tables"] + host_reserve_gib * GIB
    elif layout == "engram-nvme":
        host_need = (engram_cache_gib + host_reserve_gib) * GIB
    else:
        host_need = host_reserve_gib * GIB
    host_margin = host_gib * GIB - host_need

    fits = vram_margin >= 0 and host_margin >= 0
    return {
        "layout": layout,
        "gpus": gpus,
        "usable_vram_gib": round(usable_vram / GIB, 2),
        "resident_weights_gib": round(resident / GIB, 2),
        "kv_cache_gib": round(kv / GIB, 3),
        "vram_need_gib": round(vram_need / GIB, 2),
        "vram_margin_gib": round(vram_margin / GIB, 2),
        "per_gpu_need_gib": round(vram_need / gpus / GIB, 2),
        "host_need_gib": round(host_need / GIB, 2),
        "host_margin_gib": round(host_margin / GIB, 2),
        "fits": fits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("budget", type=Path, help="JSON from scripts/checkpoint-budget.py")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--gpu-gib", type=float, default=97887 / 1024, help="physical VRAM per GPU in GiB (default: RTX PRO 6000 97,887 MiB)")
    ap.add_argument("--host-gib", type=float, default=246.8)
    ap.add_argument("--util", type=float, default=0.95, help="fraction of VRAM the engine may claim")
    ap.add_argument("--gpu-reserve-gib", type=float, default=DEFAULT_GPU_RESERVE_GIB)
    ap.add_argument("--host-reserve-gib", type=float, default=DEFAULT_HOST_RESERVE_GIB)
    ap.add_argument("--context-tokens", type=int, default=262144)
    ap.add_argument("--sequences", type=int, default=1)
    ap.add_argument("--no-dspark", action="store_true", help="text-only, no DSpark speculative weights")
    ap.add_argument("--vision", action="store_true", help="include vision encoder + aligner")
    ap.add_argument("--layout", choices=["gpu-only", "engram-host", "engram-nvme", "all"], default="all")
    ap.add_argument("--engram-cache-gib", type=float, default=64.0, help="host row cache for engram-nvme layout")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        budget = load_budget(args.budget)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: cannot read budget: {exc}", file=sys.stderr)
        return 2

    layouts = ["gpu-only", "engram-host", "engram-nvme"] if args.layout == "all" else [args.layout]
    rows = [evaluate(budget, gpus=args.gpus, gpu_gib=args.gpu_gib, host_gib=args.host_gib, util=args.util,
                     gpu_reserve_gib=args.gpu_reserve_gib, host_reserve_gib=args.host_reserve_gib,
                     context_tokens=args.context_tokens, sequences=args.sequences,
                     include_dspark=not args.no_dspark, include_vision=args.vision, layout=l,
                     engram_cache_gib=args.engram_cache_gib) for l in layouts]
    out = {"source": budget["source"], "checkpoint_gib": {k: round(v / GIB, 2) for k, v in budget.items() if k != "source"}, "layouts": rows}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{budget['source']}")
        print(f"  checkpoint total {out['checkpoint_gib']['total']} GiB; engram tables {out['checkpoint_gib']['engram_tables']} GiB; "
              f"resident (no engram, +dspark) {out['checkpoint_gib']['resident_full']} GiB")
        print(f"  hardware: {args.gpus} x {args.gpu_gib:.1f} GiB GPU @ util {args.util} minus {args.gpu_reserve_gib} GiB reserve; host {args.host_gib} GiB")
        print(f"  KV: {args.sequences} x {args.context_tokens:,} tokens @ {KV_BYTES_PER_TOKEN} B/token")
        print(f"  {'layout':14s} {'usable VRAM':>12s} {'need':>9s} {'margin':>9s} {'per-GPU':>9s} {'host need':>10s} {'host margin':>12s}  verdict")
        for r in rows:
            print(f"  {r['layout']:14s} {r['usable_vram_gib']:12.1f} {r['vram_need_gib']:9.1f} {r['vram_margin_gib']:9.1f} "
                  f"{r['per_gpu_need_gib']:9.1f} {r['host_need_gib']:10.1f} {r['host_margin_gib']:12.1f}  {'FITS' if r['fits'] else 'DOES NOT FIT'}")
    return 0 if all(r["fits"] for r in rows) else 3


if __name__ == "__main__":
    sys.exit(main())

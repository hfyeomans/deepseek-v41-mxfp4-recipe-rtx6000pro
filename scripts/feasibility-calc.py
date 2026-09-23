#!/usr/bin/env python3
"""Decide whether a DeepSeek-V4.1-Flash checkpoint fits a GPU/host memory layout.

Inputs are a budget JSON from scripts/checkpoint-budget.py plus a hardware
description (GPU count, GiB per GPU, host GiB). Output is a placement table
and a fits/does-not-fit verdict for each supported layout:

  gpu-only            everything resident in VRAM (what stock vLLM/SGLang do)
  engram-host         Engram tables in host DRAM (SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1),
                      everything else in VRAM
  engram-nvme         Engram tables on NVMe with a host row cache, everything else in VRAM
  hybrid-cpu          attention/shared/DSpark + as many routed experts as fit in VRAM;
                      the remaining routed experts live in host DRAM and are *computed
                      on the CPU* (KTransformers-style); Engram per --engram. Also
                      reports the per-token DRAM traffic and the decode ceiling it
                      implies at --dram-gbps, plus the same ceiling if those experts
                      were instead streamed over PCIe at --pcie-gbps-per-gpu.

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
# Architecture constants from deepseek-ai/DeepSeek-V4.1-Flash config.json
# (text_config): 40 MoE layers x 384 routed experts, 6 routed per token.
MOE_LAYERS = 40
ROUTED_EXPERTS = 384
EXPERTS_PER_TOKEN = 6
ATTENTION_HEADS = 64  # TP degree must divide this for plain tensor parallelism
# Engram host footprint per mode, as a fraction of the checkpoint's FP8 tables.
# fp4: LibertAIDAI NVFP4 tables are 97.28 / 188.83 GiB of the FP8 ones.
# q3k: vcruz305's GGUF ladder quotes ~40,284 MiB per table at q3_K (2 tables).
ENGRAM_FRACTION = {"dram": 1.0, "dram-fp4": 97.28 / 188.83, "dram-q3k": 2 * 40284 / 1024 / 188.83}
# Measured on this host: E1b read-only DRAM bandwidth from the CPU cores is
# ~125 GB/s at 16 threads (evidence/e1b-dram-read-bandwidth.json); E1's 146 GB/s
# counted read+write of a copy. Theoretical 8ch DDR5-6400 is 409.6; the gap is a
# CPU-side (core-complex to IO-die) limit, not the DIMMs: E4b shows GPU DMA reads
# of the same DRAM add on top (evidence/e4b-dram-contention-cpu-vs-dma.json).
# E25 (C/OpenMP, pinned) confirmed it directly: 62-63 GB/s per CCD, 125 with both,
# 235 socket total with both GPUs' DMA (evidence/e25-dram-revalidation/).
# E27 (2026-09-23, BIOS slot 3: FCLK 2000 -> 2100 MHz with VSOC 1.05 V / VDDG
# 950 mV) lifted the per-CCD link to 66 GB/s and the socket read to 131-132
# (evidence/bios-trials/2026-09-23-boot{7,8,9,12}-*). Default moved 125 -> 131;
# pass --dram-gbps 125 to reproduce the FCLK-2000 figures quoted in older docs.
DEFAULT_DRAM_GBPS = 131.0
# PCIe 5.0 x16 pinned H2D per card, measured in E4/E4b: 53 GiB/s = 57 GB/s per
# card, both cards concurrently 114 GB/s aggregate (independent links).
DEFAULT_PCIE_GBPS_PER_GPU = 57.0
# GPU-side time per decoded token for the resident part of the model. The
# sibling V4-Flash NVFP4 TP2 deployment decodes at ~200 tok/s = 5 ms/token on
# these cards; V4.1 activates fewer parameters, so this is conservative.
DEFAULT_GPU_MS_PER_TOKEN = 5.0


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
        "tp_valid": ATTENTION_HEADS % gpus == 0,
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


def evaluate_hybrid(budget: dict, *, gpus: int, gpu_gib: float, host_gib: float, util: float,
                    gpu_reserve_gib: float, host_reserve_gib: float, context_tokens: int,
                    sequences: int, include_dspark: bool, include_vision: bool, engram: str,
                    engram_cache_gib: float, dram_gbps: float, pcie_gbps_per_gpu: float,
                    gpu_ms_per_token: float, activation_skew: float) -> dict:
    """VRAM holds everything but routed experts, then as many experts as fit.
    The rest live in DRAM and are computed on the CPU. `activation_skew` is the
    fraction of expert *activations* that land on GPU-resident experts beyond
    their share of the count (0 = uniform routing; 0.2 = a frequency-based
    placement catches 20 % more activations than uniform would)."""
    experts = budget["routed_experts"]
    non_expert = budget["resident_text_no_dspark"] - experts
    if include_dspark:
        non_expert += budget["dspark"]
    if include_vision:
        non_expert += budget["vision"]
    kv = KV_BYTES_PER_TOKEN * context_tokens * sequences
    usable_vram = gpus * (gpu_gib * util - gpu_reserve_gib) * GIB
    expert_capacity = usable_vram - non_expert - kv
    if expert_capacity < 0:
        gpu_fraction = 0.0
    else:
        gpu_fraction = min(1.0, expert_capacity / experts)
    per_expert = experts / (MOE_LAYERS * ROUTED_EXPERTS)
    gpu_experts = int(gpu_fraction * MOE_LAYERS * ROUTED_EXPERTS)
    cpu_bytes = experts * (1 - gpu_fraction)

    if engram == "nvme":
        engram_host = engram_cache_gib * GIB
    else:
        engram_host = budget["engram_tables"] * ENGRAM_FRACTION[engram]
    host_need = cpu_bytes + engram_host + host_reserve_gib * GIB
    host_margin = host_gib * GIB - host_need

    # Uniform routing: each of the 240 expert activations per token hits a
    # CPU expert with probability (1 - gpu_fraction), minus the placement skew.
    cpu_activation_share = max(0.0, (1 - gpu_fraction) * (1 - activation_skew))
    bytes_per_token = MOE_LAYERS * EXPERTS_PER_TOKEN * cpu_activation_share * per_expert
    dram_ms = bytes_per_token / (dram_gbps * 1e9) * 1e3
    pcie_ms = bytes_per_token / (pcie_gbps_per_gpu * gpus * 1e9) * 1e3
    decode_cpu = 1e3 / (gpu_ms_per_token + dram_ms) if bytes_per_token else 1e3 / gpu_ms_per_token
    decode_pcie = 1e3 / (gpu_ms_per_token + pcie_ms) if bytes_per_token else 1e3 / gpu_ms_per_token
    # Split: CPU cores compute some DRAM experts while the GPUs stream the rest
    # over PCIe. E4b measured the two DRAM read paths as additive (98 %), so the
    # DRAM side runs at the sum of both bandwidths. Upper bound: assumes CPU
    # compute keeps up with its share of the bytes (E6 stage 0 reached 94 of
    # 125 GB/s while computing) and perfect overlap with the GPU part.
    split_gbps = dram_gbps + pcie_gbps_per_gpu * gpus
    split_ms = bytes_per_token / (split_gbps * 1e9) * 1e3
    decode_split = 1e3 / (gpu_ms_per_token + split_ms) if bytes_per_token else 1e3 / gpu_ms_per_token

    fits = expert_capacity >= 0 and host_margin >= 0
    return {
        "layout": f"hybrid-cpu/{engram}",
        "gpus": gpus,
        "tp_valid": ATTENTION_HEADS % gpus == 0,
        "usable_vram_gib": round(usable_vram / GIB, 2),
        "non_expert_resident_gib": round(non_expert / GIB, 2),
        "kv_cache_gib": round(kv / GIB, 3),
        "gpu_expert_capacity_gib": round(max(expert_capacity, 0) / GIB, 2),
        "gpu_expert_fraction": round(gpu_fraction, 3),
        "gpu_experts": gpu_experts,
        "cpu_experts": MOE_LAYERS * ROUTED_EXPERTS - gpu_experts,
        "cpu_expert_gib": round(cpu_bytes / GIB, 2),
        "engram_host_gib": round(engram_host / GIB, 2),
        "host_need_gib": round(host_need / GIB, 2),
        "host_margin_gib": round(host_margin / GIB, 2),
        "cpu_activation_share": round(cpu_activation_share, 3),
        "dram_bytes_per_token_gib": round(bytes_per_token / GIB, 3),
        "dram_ms_per_token": round(dram_ms, 2),
        "decode_ceiling_cpu_tok_s": round(decode_cpu, 1),
        "pcie_ms_per_token": round(pcie_ms, 2),
        "decode_ceiling_pcie_stream_tok_s": round(decode_pcie, 1),
        "split_ms_per_token": round(split_ms, 2),
        "decode_ceiling_split_tok_s": round(decode_split, 1),
        "fits": fits,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("budget", type=Path, help="JSON from scripts/checkpoint-budget.py")
    ap.add_argument("--gpus", default="2", help="GPU count, or a comma list to compare (e.g. 2,3,4)")
    ap.add_argument("--gpu-gib", type=float, default=97887 / 1024, help="physical VRAM per GPU in GiB (default: RTX PRO 6000 97,887 MiB)")
    ap.add_argument("--host-gib", type=float, default=246.8)
    ap.add_argument("--util", type=float, default=0.95, help="fraction of VRAM the engine may claim")
    ap.add_argument("--gpu-reserve-gib", type=float, default=DEFAULT_GPU_RESERVE_GIB)
    ap.add_argument("--host-reserve-gib", type=float, default=DEFAULT_HOST_RESERVE_GIB)
    ap.add_argument("--context-tokens", type=int, default=262144)
    ap.add_argument("--sequences", type=int, default=1)
    ap.add_argument("--no-dspark", action="store_true", help="text-only, no DSpark speculative weights")
    ap.add_argument("--vision", action="store_true", help="include vision encoder + aligner")
    ap.add_argument("--layout", choices=["gpu-only", "engram-host", "engram-nvme", "hybrid-cpu", "all"], default="all")
    ap.add_argument("--engram-cache-gib", type=float, default=64.0, help="host row cache for engram-nvme layout and hybrid --engram nvme")
    ap.add_argument("--engram", choices=sorted(ENGRAM_FRACTION) + ["nvme"], default="nvme",
                    help="hybrid-cpu: where the Engram tables live (dram = FP8 as shipped, dram-fp4, dram-q3k, nvme + cache)")
    ap.add_argument("--dram-gbps", type=float, default=DEFAULT_DRAM_GBPS, help="hybrid-cpu: host DRAM read bandwidth for CPU experts")
    ap.add_argument("--pcie-gbps-per-gpu", type=float, default=DEFAULT_PCIE_GBPS_PER_GPU, help="hybrid-cpu: H2D bandwidth per card for the streaming comparison")
    ap.add_argument("--gpu-ms-per-token", type=float, default=DEFAULT_GPU_MS_PER_TOKEN, help="hybrid-cpu: GPU time per token for the resident part")
    ap.add_argument("--activation-skew", type=float, default=0.0, help="hybrid-cpu: extra share of activations caught by GPU experts via frequency placement (0..1)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        budget = load_budget(args.budget)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: cannot read budget: {exc}", file=sys.stderr)
        return 2
    try:
        gpu_counts = [int(g) for g in args.gpus.split(",")]
        if any(g < 1 for g in gpu_counts):
            raise ValueError
    except ValueError:
        print(f"error: --gpus must be a positive integer or comma list, got {args.gpus!r}", file=sys.stderr)
        return 2

    layouts = ["gpu-only", "engram-host", "engram-nvme", "hybrid-cpu"] if args.layout == "all" else [args.layout]
    common = dict(gpu_gib=args.gpu_gib, host_gib=args.host_gib, util=args.util,
                  gpu_reserve_gib=args.gpu_reserve_gib, host_reserve_gib=args.host_reserve_gib,
                  context_tokens=args.context_tokens, sequences=args.sequences,
                  include_dspark=not args.no_dspark, include_vision=args.vision)
    rows = []
    for gpus in gpu_counts:
        for layout in layouts:
            if layout == "hybrid-cpu":
                rows.append(evaluate_hybrid(budget, gpus=gpus, engram=args.engram, engram_cache_gib=args.engram_cache_gib,
                                            dram_gbps=args.dram_gbps, pcie_gbps_per_gpu=args.pcie_gbps_per_gpu,
                                            gpu_ms_per_token=args.gpu_ms_per_token, activation_skew=args.activation_skew, **common))
            else:
                rows.append(evaluate(budget, gpus=gpus, layout=layout, engram_cache_gib=args.engram_cache_gib, **common))
    out = {"source": budget["source"], "checkpoint_gib": {k: round(v / GIB, 2) for k, v in budget.items() if k != "source"}, "layouts": rows}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"{budget['source']}")
        print(f"  checkpoint total {out['checkpoint_gib']['total']} GiB; engram tables {out['checkpoint_gib']['engram_tables']} GiB; "
              f"routed experts {out['checkpoint_gib']['routed_experts']} GiB; resident (no engram, +dspark) {out['checkpoint_gib']['resident_full']} GiB")
        print(f"  hardware: {args.gpus} x {args.gpu_gib:.1f} GiB GPU @ util {args.util} minus {args.gpu_reserve_gib} GiB reserve; host {args.host_gib} GiB")
        print(f"  KV: {args.sequences} x {args.context_tokens:,} tokens @ {KV_BYTES_PER_TOKEN} B/token")
        resident_rows = [r for r in rows if not r["layout"].startswith("hybrid")]
        hybrid_rows = [r for r in rows if r["layout"].startswith("hybrid")]
        if resident_rows:
            print(f"  {'GPUs':>4s} {'layout':14s} {'usable VRAM':>12s} {'need':>9s} {'margin':>9s} {'per-GPU':>9s} {'host need':>10s} {'host margin':>12s}  verdict")
            for r in resident_rows:
                tp = "" if r["tp_valid"] else " (TP invalid: 64 heads; needs EP/PP)"
                print(f"  {r['gpus']:4d} {r['layout']:14s} {r['usable_vram_gib']:12.1f} {r['vram_need_gib']:9.1f} {r['vram_margin_gib']:9.1f} "
                      f"{r['per_gpu_need_gib']:9.1f} {r['host_need_gib']:10.1f} {r['host_margin_gib']:12.1f}  {'FITS' if r['fits'] else 'DOES NOT FIT'}{tp}")
        if hybrid_rows:
            print(f"  hybrid-cpu (experts that do not fit VRAM are computed on CPU from DRAM; Engram={args.engram}; "
                  f"DRAM {args.dram_gbps:.0f} GB/s; PCIe {args.pcie_gbps_per_gpu:.0f} GB/s/GPU; GPU {args.gpu_ms_per_token} ms/token; skew {args.activation_skew})")
            print(f"  {'GPUs':>4s} {'GPU experts':>11s} {'CPU experts':>11s} {'CPU GiB':>8s} {'host need':>10s} {'host margin':>12s} {'GiB/token':>9s} {'CPU ceiling':>11s} {'PCIe ceiling':>12s} {'split ceiling':>13s}  verdict")
            for r in hybrid_rows:
                tp = "" if r["tp_valid"] else " (TP invalid: 64 heads; needs EP/PP)"
                print(f"  {r['gpus']:4d} {r['gpu_experts']:5d} ({r['gpu_expert_fraction']*100:3.0f}%) {r['cpu_experts']:11d} {r['cpu_expert_gib']:8.1f} "
                      f"{r['host_need_gib']:10.1f} {r['host_margin_gib']:12.1f} {r['dram_bytes_per_token_gib']:9.2f} "
                      f"{r['decode_ceiling_cpu_tok_s']:8.0f} t/s {r['decode_ceiling_pcie_stream_tok_s']:9.0f} t/s {r['decode_ceiling_split_tok_s']:10.0f} t/s  {'FITS' if r['fits'] else 'DOES NOT FIT'}{tp}")
    return 0 if all(r["fits"] for r in rows) else 3


if __name__ == "__main__":
    sys.exit(main())

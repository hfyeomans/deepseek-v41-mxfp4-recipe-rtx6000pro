#!/usr/bin/env python3
"""Experiment E6 stage 0: can this CPU compute the DRAM-resident routed experts
of the 2-GPU three-tier layout fast enough? GPU-free; needs torch (run inside
any torch-bearing image on this host *without* --gpus).

Synthetic DeepSeek-V4.1-Flash expert: gate+up [2*2304, 5120] and down
[5120, 2304], 35.4 M parameters. A pool of --pool experts (default 256, well
over the 64 MiB L3) is allocated once; each "token" picks --experts-per-token
distinct experts at random (default 101 = 42 % of the 240 activations per
token, the CPU share on 2 GPUs) and runs gate/up GEMV, SiLU*up, down GEMV at
batch 1. Variants:

  bf16       weights stored bf16, torch.mv            -> memory-roofline of the
                                                        GEMV path (4x MXFP4 bytes)
  int8-deq   weights int8 + per-row scale, dequantised to bf16 on the fly,
             then torch.mv                            -> NOT a valid compute proxy:
                                                        torch materialises the
                                                        dequantised matrix (E6 s0
                                                        measured 584 ms/token).
                                                        Kept for the record; use a
                                                        real MXFP4 CPU kernel instead.
  read       torch.sum over an fp32 tensor             -> DRAM read bandwidth (E1b).
                                                        A bf16 sum is conversion-
                                                        bound (10 GB/s measured) and
                                                        was replaced by fp32.

Reports median ms/token, bytes touched per token and effective GB/s per
variant and thread count, plus the decode ceiling these imply if the MXFP4
experts moved 1.75 GiB/token at the measured GB/s (bytes-bound estimate) and
if the CPU took the measured ms/token (compute-bound estimate). Both are
estimates; the real kernel (KTransformers MXFP4) is measured in stage 1.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

try:
    import torch
except ImportError:  # pragma: no cover
    print("error: torch not importable; run inside a torch container, e.g.\n"
          "  docker run --rm -v \"$PWD:/w\" -w /w --entrypoint python3 dsv4-nvfp4:recipe scripts/bench-cpu-expert.py",
          file=sys.stderr)
    sys.exit(2)

HIDDEN = 5120
INTER = 2304
PARAMS_PER_EXPERT = 3 * HIDDEN * INTER
GIB = 2**30
# Calculator defaults (scripts/feasibility-calc.py): 1.75 GiB of MXFP4 expert
# bytes per token on 2 GPUs, 5 ms of GPU time per token.
MXFP4_GIB_PER_TOKEN = 1.75
GPU_MS_PER_TOKEN = 5.0


def make_pool(pool: int, variant: str, gen: torch.Generator) -> dict:
    if variant == "bf16":
        gu = torch.randn(pool, 2 * INTER, HIDDEN, generator=gen).to(torch.bfloat16)
        dn = torch.randn(pool, HIDDEN, INTER, generator=gen).to(torch.bfloat16)
        return {"gu": gu, "dn": dn, "bytes_per_expert": PARAMS_PER_EXPERT * 2}
    gu = torch.randint(-127, 128, (pool, 2 * INTER, HIDDEN), dtype=torch.int8, generator=gen)
    dn = torch.randint(-127, 128, (pool, HIDDEN, INTER), dtype=torch.int8, generator=gen)
    gu_s = torch.rand(pool, 2 * INTER, 1, generator=gen).to(torch.bfloat16) * 0.01
    dn_s = torch.rand(pool, HIDDEN, 1, generator=gen).to(torch.bfloat16) * 0.01
    return {"gu": gu, "dn": dn, "gu_s": gu_s, "dn_s": dn_s, "bytes_per_expert": PARAMS_PER_EXPERT * 1}


def expert_forward(p: dict, idx: int, h: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "bf16":
        gu = torch.mv(p["gu"][idx], h)
    else:
        gu = torch.mv(p["gu"][idx].to(torch.bfloat16) * p["gu_s"][idx], h)
    gate, up = gu[:INTER], gu[INTER:]
    act = torch.nn.functional.silu(gate) * up
    if variant == "bf16":
        return torch.mv(p["dn"][idx], act)
    return torch.mv(p["dn"][idx].to(torch.bfloat16) * p["dn_s"][idx], act)


def bench_variant(variant: str, pool: int, per_token: int, tokens: int, warmup: int, gen: torch.Generator) -> dict:
    p = make_pool(pool, variant, gen)
    h = torch.randn(HIDDEN, generator=gen).to(torch.bfloat16)
    times = []
    for t in range(warmup + tokens):
        idx = torch.randperm(pool, generator=gen)[:per_token].tolist()
        t0 = time.perf_counter()
        acc = torch.zeros(HIDDEN, dtype=torch.bfloat16)
        for i in idx:
            acc += expert_forward(p, i, h, variant)
        dt = time.perf_counter() - t0
        if t >= warmup:
            times.append(dt)
    ms = statistics.median(times) * 1e3
    bytes_per_token = per_token * p["bytes_per_expert"]
    del p
    return {
        "variant": variant,
        "ms_per_token_median": round(ms, 2),
        "ms_per_token_p90": round(sorted(times)[int(0.9 * (len(times) - 1))] * 1e3, 2),
        "bytes_per_token": bytes_per_token,
        "effective_gbps": round(bytes_per_token / (ms / 1e3) / 1e9, 1),
        "ms_per_expert": round(ms / per_token, 3),
    }


def bench_read(size_gib: float, reps: int, gen: torch.Generator) -> dict:
    n = int(size_gib * GIB / 4)
    x = torch.ones(n, dtype=torch.float32)  # fp32: the reduction is bandwidth-bound, unlike a bf16 sum
    x.sum()  # fault in
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        x.sum()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    del x
    return {"variant": "read", "size_gib": size_gib, "s_median": round(med, 4), "gbps": round(n * 4 / med / 1e9, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=256, help="experts in the synthetic pool (256 bf16 = 16.9 GiB)")
    ap.add_argument("--experts-per-token", type=int, default=101)
    ap.add_argument("--tokens", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", default="16,32", help="comma list of torch intra-op thread counts")
    ap.add_argument("--variants", default="read,bf16,int8-deq")
    ap.add_argument("--read-gib", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output")
    args = ap.parse_args()

    variants = args.variants.split(",")
    results = []
    for th in (int(t) for t in args.threads.split(",")):
        torch.set_num_threads(th)
        gen = torch.Generator().manual_seed(args.seed)
        for v in variants:
            t0 = time.perf_counter()
            if v == "read":
                r = bench_read(args.read_gib, 5, gen)
            else:
                r = bench_variant(v, args.pool, args.experts_per_token, args.tokens, args.warmup, gen)
            r["threads"] = th
            r["setup_plus_run_s"] = round(time.perf_counter() - t0, 1)
            results.append(r)
            print(json.dumps(r), flush=True)

    def ceiling(ms_cpu: float) -> float:
        return round(1e3 / (GPU_MS_PER_TOKEN + ms_cpu), 1)

    implied = []
    for r in results:
        if r["variant"] == "read":
            implied.append({"threads": r["threads"], "basis": "DRAM read GB/s (E1b), MXFP4 bytes-bound",
                            "cpu_ms_per_token": round(MXFP4_GIB_PER_TOKEN * GIB / (r["gbps"] * 1e9) * 1e3, 2)})
        else:
            implied.append({"threads": r["threads"], "basis": f"{r['variant']} GEMV GB/s, MXFP4 bytes-bound",
                            "cpu_ms_per_token": round(MXFP4_GIB_PER_TOKEN * GIB / (r["effective_gbps"] * 1e9) * 1e3, 2)})
            implied.append({"threads": r["threads"], "basis": f"{r['variant']} GEMV measured ms/token, compute-bound",
                            "cpu_ms_per_token": r["ms_per_token_median"]})
    for i in implied:
        i["decode_ceiling_tok_s"] = ceiling(i["cpu_ms_per_token"])

    out = {
        "experiment": "E6-stage0-cpu-expert",
        "torch": torch.__version__,
        "cpu_count": torch.get_num_interop_threads() and __import__("os").cpu_count(),
        "config": {"hidden": HIDDEN, "inter": INTER, "params_per_expert": PARAMS_PER_EXPERT, "pool": args.pool,
                   "experts_per_token": args.experts_per_token, "tokens": args.tokens, "warmup": args.warmup,
                   "mxfp4_gib_per_token": MXFP4_GIB_PER_TOKEN, "gpu_ms_per_token": GPU_MS_PER_TOKEN},
        "results": results,
        "implied_2gpu_decode_ceilings": implied,
    }
    text = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

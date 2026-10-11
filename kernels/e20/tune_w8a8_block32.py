#!/usr/bin/env python3
"""E20 — tune SGLang's Triton W8A8 block-FP8 GEMM for V4.1-Flash's [32, 32] dense weights on SM120.

SGLang ships tuned configs for this kernel only for block_shape [128, 128] (H100/H200/B200) and, for
DeepSeek-V4-family [32, 32] shapes, only for H200.  On the RTX PRO 6000 (SM120) it falls back to the
default config (BLOCK_M=64, BLOCK_N=32, BLOCK_K=32, 3 stages), which at decode M=1..4 launches a few
hundred tiny blocks each walking K serially in 32-wide slices: 2-3 % SM occupancy and ~20x off the
HBM roofline (E19/T3.6d3 profile: 189 calls, 40 ms of a 70 ms step).

SGLang's own tuner (`benchmark/kernels/quantization/tuning_block_wise_kernel.py`) cannot tune this
case: its search space has BLOCK_SIZE_K in {64, 128} and filters on `block_k % BLOCK_SIZE_K == 0`,
which is empty for block_k=32, and it never tries SPLIT_K / SWAP_AB.  This script reuses SGLang's
kernels (`_w8a8_block_fp8_matmul`, `_w8a8_block_fp8_matmul_hopper`, `_reduce_block_fp8_split_k`) and
searches a decode-shaped space (BLOCK_M=16, SWAP_AB, SPLIT_K up to 16) for small M and a conventional
space for large M, so prefill (M in the thousands) does not regress.  Every winning config is
checked against an FP32 dequantised reference before it is written.

Run inside the serving container (needs SGLang importable and a free-ish GPU):

    docker cp kernels/e20/tune_w8a8_block32.py dsv41-kt:/tmp/
    docker exec -e CUDA_VISIBLE_DEVICES=1 dsv41-kt python3 /tmp/tune_w8a8_block32.py --out /tmp/e20
    docker cp dsv41-kt:/tmp/e20 evidence/e20-dense-fp8-gemm/

Outputs (under --out):
    configs/N=..,K=..,device_name=..,dtype=fp8_w8a8,block_shape=[32, 32].json   SGLang config files
    results.json                                                                 baseline vs best per (N, K, M)

Copy the JSONs into the overlay's python/sglang/kernels/ops/quantization/configs/ (bind-mounted into
the container) and lift the `is_sm90` gate on the SPLIT_K/SWAP_AB kernel (overlay commit).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton

from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
    _reduce_block_fp8_split_k,
    _w8a8_block_fp8_matmul,
    _w8a8_block_fp8_matmul_hopper,
)
from sglang.srt.utils import get_device_name  # noqa: E402

# The seven dense (N, K) shapes the E19 server log reported as "Config file not found", all [32, 32].
DSV41_TP2_SHAPES: List[Tuple[int, int]] = [
    (16384, 1280),
    (1792, 5120),
    (2304, 5120),
    (25600, 6144),
    (4096, 1280),
    (5120, 1152),
    (5120, 4096),
]

BLOCK_N, BLOCK_K = 32, 32  # V4.1-Flash dense weight block-scale shape

DEFAULT_CONFIG = {  # what w8a8_block_fp8_matmul_triton uses when no JSON exists
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": BLOCK_N,
    "BLOCK_SIZE_K": BLOCK_K,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "num_stages": 3,
}

SMALL_M_MAX = 16  # overridable with --small-m-max; pass 1 shows a cliff at M=32 without split-K


def small_m_space() -> List[Dict[str, Any]]:
    """Decode-shaped configs: one M tile, spread the weight stream over many CTAs."""
    space = []
    for bn, split_k, swap_ab, warps, stages in itertools.product(
        [16, 32, 64, 128], [1, 2, 4, 8, 16], [False, True], [2, 4, 8], [2, 3, 4, 5]
    ):
        cfg = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": bn,
            "BLOCK_SIZE_K": BLOCK_K,
            "GROUP_SIZE_M": 32,
            "num_warps": warps,
            "num_stages": stages,
        }
        if swap_ab:
            cfg["SWAP_AB"] = True
        if split_k > 1:
            cfg["SPLIT_K"] = split_k
        space.append(cfg)
    return space


def large_m_space() -> List[Dict[str, Any]]:
    """Prefill-shaped configs: conventional tiles, no split-K."""
    space = []
    for bm, bn, warps, stages, group in itertools.product(
        [32, 64, 128], [32, 64, 128, 256], [4, 8], [2, 3, 4], [1, 32]
    ):
        space.append(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": BLOCK_K,
                "GROUP_SIZE_M": group,
                "num_warps": warps,
                "num_stages": stages,
            }
        )
    return space


def make_inputs(M: int, N: int, K: int, device: torch.device):
    fp8 = torch.float8_e4m3fn
    A = (torch.randn(M, K, device=device) * 0.5).to(fp8)
    B = (torch.randn(N, K, device=device) * 0.5).to(fp8)
    As = torch.rand(M, triton.cdiv(K, BLOCK_K), device=device) * 0.1 + 0.05
    Bs = torch.rand(triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K), device=device) * 0.1 + 0.05
    return A, B, As, Bs


def matches_reference(out: torch.Tensor, A, B, As, Bs, n_chunk: int = 2048) -> bool:
    """Compare `out` (M x N) against an FP32 dequantise-then-matmul reference, chunked over N.

    Chunking keeps the peak under ~0.5 GB even for 25600 x 6144 at M=4096, so the tuner can run
    beside a serving process that holds 91 of the 96 GB (the un-chunked version OOMed there).
    """
    M, K = A.shape
    N = B.shape[0]
    a = A.float() * As.repeat_interleave(BLOCK_K, dim=1)[:, :K]
    for n0 in range(0, N, n_chunk):
        n1 = min(n0 + n_chunk, N)
        bs_rows = Bs[n0 // BLOCK_N : triton.cdiv(n1, BLOCK_N)]
        b = B[n0:n1].float() * bs_rows.repeat_interleave(BLOCK_N, dim=0)[: n1 - n0].repeat_interleave(
            BLOCK_K, dim=1
        )[:, :K]
        ref = a @ b.t()
        # bf16 output carries ~3 significant digits (2^-8 relative); 1e-2 leaves room for fp32
        # accumulation-order differences between split-K and serial-K but rejects real tile bugs.
        # (The E20 configs shipped in the overlay were selected under the earlier 2e-2 tolerance;
        # they are not re-tuned here.)
        if not torch.allclose(out[:, n0:n1].float(), ref, rtol=1e-2, atol=1e-2):
            return False
    return True


def run_config(
    A, B, As, Bs, C, config: Dict[str, Any], partials: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mirror of sglang's w8a8_block_fp8_matmul_triton dispatch, minus the is_sm90 gate.

    `partials` is the (split_k, M, N) fp32 scratch for SPLIT_K > 1.  Callers that time this
    function should preallocate it (see `tune_shape`) so the measurement covers only the kernels;
    sglang's production wrapper allocates it per call through the caching allocator, which is
    host-side work the CUDA-event timing would not see anyway.
    """
    M, K = A.shape
    N = B.shape[0]
    split_k = config.get("SPLIT_K", 1)
    swap_ab = config.get("SWAP_AB", False)
    tuned_path = swap_ab or split_k > 1
    kernel = _w8a8_block_fp8_matmul_hopper if tuned_path else _w8a8_block_fp8_matmul
    if split_k > 1:
        if partials is None:
            partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
        assert partials.shape == (split_k, M, N) and partials.dtype == torch.float32
    else:
        partials = C
    needs_masking = bool(K % config["BLOCK_SIZE_K"] != 0)
    blocks = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"])
    grid = (blocks, split_k) if tuned_path else (blocks,)
    kernel[grid](
        A, B, partials, As, Bs, M, N, K, BLOCK_N, BLOCK_K,
        A.stride(-2), A.stride(-1), B.stride(1), B.stride(0),
        C.stride(-2), C.stride(-1), As.stride(-2), As.stride(-1),
        Bs.stride(1), Bs.stride(0),
        **config, needs_masking=needs_masking,
    )
    if split_k > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](partials, C, M * N, split_k, 256)
    return C


def bench(fn, warmup: int, rep: int) -> float:
    """Median kernel time in microseconds (CUDA events, L2 flushed per rep by triton.testing)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median") * 1000.0


def tune_shape(
    N: int, K: int, m_list: List[int], device, warmup: int, rep: int, quick: bool, small_m_max: int
):
    results = {}
    best_configs: Dict[int, Dict[str, Any]] = {}
    for M in m_list:
        A, B, As, Bs = make_inputs(M, N, K, device)
        C = torch.empty(M, N, device=device, dtype=torch.bfloat16)
        space = small_m_space() if M <= small_m_max else large_m_space()
        if quick:
            space = space[:: max(1, len(space) // 40)]
        timings: List[Tuple[float, Dict[str, Any]]] = []
        failed = 0
        scratch: Dict[int, torch.Tensor] = {}  # split_k -> preallocated fp32 partials
        for cfg in space:
            split_k = cfg.get("SPLIT_K", 1)
            partials = None
            if split_k > 1:
                partials = scratch.get(split_k)
                if partials is None:
                    partials = torch.empty((split_k, M, N), device=device, dtype=torch.float32)
                    scratch[split_k] = partials
            try:
                run_config(A, B, As, Bs, C, cfg, partials)
                torch.cuda.synchronize()
                us = bench(lambda: run_config(A, B, As, Bs, C, cfg, partials), warmup, rep)
            except Exception:  # out of shared memory / bad tile combo: skip
                failed += 1
                continue
            timings.append((us, cfg))
        timings.sort(key=lambda t: t[0])
        base_us = bench(lambda: run_config(A, B, As, Bs, C, DEFAULT_CONFIG), warmup, rep)

        # Pick the fastest config that also matches the FP32 reference.
        chosen: Optional[Tuple[float, Dict[str, Any]]] = None
        rejected = 0
        for us, cfg in timings:
            if matches_reference(run_config(A, B, As, Bs, C, cfg), A, B, As, Bs):
                chosen = (us, cfg)
                break
            rejected += 1
        if chosen is None:
            chosen = (base_us, dict(DEFAULT_CONFIG))
        best_us, best_cfg = chosen
        best_configs[M] = best_cfg
        weight_bytes = N * K + Bs.numel() * 4
        results[str(M)] = {
            "baseline_us": round(base_us, 2),
            "best_us": round(best_us, 2),
            "speedup": round(base_us / best_us, 2),
            "best_config": best_cfg,
            "weight_GBps_best": round(weight_bytes / best_us / 1e3, 1),
            "weight_GBps_baseline": round(weight_bytes / base_us / 1e3, 1),
            "configs_tried": len(timings),
            "configs_failed_to_launch": failed,
            "configs_rejected_numerically": rejected,
            "top5": [{"us": round(u, 2), **c} for u, c in timings[:5]],
        }
        print(
            f"N={N:6d} K={K:5d} M={M:5d}: default {base_us:8.1f} us -> best {best_us:8.1f} us "
            f"({base_us / best_us:5.2f}x, {results[str(M)]['weight_GBps_best']:7.1f} GB/s) {best_cfg}",
            flush=True,
        )
        del A, B, As, Bs, C, scratch
        torch.cuda.empty_cache()
    return best_configs, results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (configs/ and results.json)")
    ap.add_argument("--shapes", nargs="*", help="N,K pairs; default = the 7 V4.1-Flash TP=2 dense shapes")
    ap.add_argument(
        "--batch-sizes", nargs="*", type=int,
        default=[1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    ap.add_argument("--warmup", type=int, default=10, help="do_bench warmup ms")
    ap.add_argument("--rep", type=int, default=50, help="do_bench rep ms")
    ap.add_argument("--quick", action="store_true", help="subsample the search space (smoke test)")
    ap.add_argument(
        "--small-m-max", type=int, default=SMALL_M_MAX,
        help="use the decode-shaped (BLOCK_M=16, split-K, swap-AB) space for M up to this value",
    )
    args = ap.parse_args()

    shapes = (
        [tuple(int(x) for x in s.split(",")) for s in args.shapes] if args.shapes else DSV41_TP2_SHAPES
    )
    device = torch.device("cuda")
    device_name = get_device_name().replace(" ", "_")
    cfg_dir = os.path.join(args.out, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    print(f"device {device_name}, torch {torch.__version__}, triton {triton.__version__}", flush=True)
    print(
        f"search space: small-M {len(small_m_space())} configs, large-M {len(large_m_space())} configs",
        flush=True,
    )

    all_results: Dict[str, Any] = {
        "device_name": device_name,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "block_shape": [BLOCK_N, BLOCK_K],
        "default_config": DEFAULT_CONFIG,
        "small_m_max": args.small_m_max,
        "shapes": {},
    }
    t0 = time.perf_counter()
    for N, K in shapes:
        best, res = tune_shape(
            N, K, args.batch_sizes, device, args.warmup, args.rep, args.quick, args.small_m_max
        )
        all_results["shapes"][f"N={N},K={K}"] = res
        fname = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{BLOCK_N}, {BLOCK_K}].json"
        with open(os.path.join(cfg_dir, fname), "w") as f:
            json.dump({str(m): c for m, c in best.items()}, f, indent=4)
            f.write("\n")
        with open(os.path.join(args.out, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2)
    all_results["tuning_seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"done in {all_results['tuning_seconds']} s; configs in {cfg_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

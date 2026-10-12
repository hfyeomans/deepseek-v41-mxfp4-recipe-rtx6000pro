#!/usr/bin/env python3
"""T3.41 — decode-shape (M <= 4) block-[32, 32] FP8 GEMV for SM120, benched L2-cold against E20.

The seven dense W8A8 projections of the TP2 rank run at decode M = 1-4 through SGLang's Triton
`w8a8_block_fp8_matmul_triton` with the E20-tuned configs.  Those configs are tensor-core GEMM
tiles (BLOCK_M = 16 padded rows, BLOCK_K = 32, SPLIT_K up to 16 plus a reduce launch) and move the
weights at 428-748 GB/s against a 1,470 GB/s measured GDDR7 roof — the small shapes sit on a
~12 us floor regardless of bytes (`evidence/e20-dense-fp8-gemm/merged/results.json`).

This file holds a `tl.dot`-free GEMV: one CTA per BLOCK_N <= 32 output columns (so the CTA's
weight rows share one block-scale column), streaming K in BLOCK_K-wide contiguous slices.  Both
scales are folded into the activation slice once per k-slice
(`a_eff[m, k] = a[m, k] * As[m, k // 32] * Bs[n // 32, k // 32]`), so the inner loop is a plain
FP32 FMA row-dot per (m, n); M is a constexpr loop, the weight slice is loaded once per k-slice and
reused for all M rows.  Optional SPLIT_K writes FP32 partials and reuses SGLang's
`_reduce_block_fp8_split_k` (deterministic, no atomics).

Numerics: FP32 FMA accumulation in a fixed order.  It is NOT bit-identical to the tensor-core
kernel (an MMA accumulates in a different order with different intermediate rounding), and E20's
own SPLIT_K configs are already not bit-identical to the serial kernel; the end-to-end gate is the
serve-level NLL / tokens-moved check, exactly as E20 was gated.  Here every candidate is checked
against the FP32 dequantised reference (E20 tolerance, rtol = atol = 1e-2) on `--seeds` seeds, and
the max |diff| against the Triton winner is reported.

Run inside the recipe container with the overlay mounted and GPU 1 free (same pattern as T3.21):

  docker run --rm --gpus '"device=1"' --entrypoint python3 \\
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \\
    -v $PWD/kernels/t341:/tmp/t341:ro -v /tmp/t341-out:/out dsv41-kt:recipe \\
    /tmp/t341/fp8_gemv_block32.py --out /out/results.json --configs-out /out/gemv_configs.json

`--configs-out` writes the per-(N, K, M) winner table the overlay dispatch reads
(`SGLANG_DSV41_FP8_GEMV_CONFIGS`); a winner is written only if it beats the Triton winner by more
than 2 % and by more than the p10-p90 spread of either (T3.27 acceptance rule).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
BLOCK = 32
# The seven dense (N, K) shapes of the TP2 rank (E20 / T3.21).
SHAPES: List[Tuple[int, int]] = [
    (16384, 1280),
    (1792, 5120),
    (2304, 5120),
    (25600, 6144),
    (4096, 1280),
    (5120, 1152),
    (5120, 4096),
]


@triton.jit
def _fp8_gemv_block32(
    a_ptr, b_ptr, c_ptr, as_ptr, bs_ptr,
    N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_asm, stride_ask,
    stride_bsn, stride_bsk,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    OUT_FP32: tl.constexpr,
):
    """C[m, n] = sum_k (A[m, k] * As[m, k//32]) * (B[n, k] * Bs[n//32, k//32]), M <= 4 rows.

    grid = (cdiv(N, BLOCK_N), SPLIT_K).  BLOCK_N must divide 32 so every row of the CTA's weight
    tile shares the same block-scale row (n // 32).  With SPLIT_K > 1 the CTA writes FP32
    partials at c_ptr + pid_k * M * N (reduced by `_reduce_block_fp8_split_k`).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    n0 = pid_n * BLOCK_N
    offs_n = n0 + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    nb = n0 // 32  # one block-scale row for the whole tile (BLOCK_N | 32)

    k_per_split = tl.cdiv(tl.cdiv(K, BLOCK_K), SPLIT_K) * BLOCK_K
    k_begin = pid_k * k_per_split
    k_end = tl.minimum(k_begin + k_per_split, K)

    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(k_begin, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_end
        kb = offs_k // 32
        # weight tile [BLOCK_N, BLOCK_K], fp8 -> fp32, streamed once
        w = tl.load(
            b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk,
            mask=mask_n[:, None] & mask_k[None, :], other=0.0,
        ).to(tl.float32)
        # block scale of this tile's row-block along k: [BLOCK_K]
        bs = tl.load(bs_ptr + nb * stride_bsn + kb * stride_bsk, mask=mask_k, other=0.0)
        # row 0
        a = tl.load(a_ptr + 0 * stride_am + offs_k * stride_ak, mask=mask_k, other=0.0).to(tl.float32)
        s = tl.load(as_ptr + 0 * stride_asm + kb * stride_ask, mask=mask_k, other=0.0)
        acc0 += tl.sum(w * (a * s * bs)[None, :], axis=1)
        if M > 1:
            a = tl.load(a_ptr + 1 * stride_am + offs_k * stride_ak, mask=mask_k, other=0.0).to(tl.float32)
            s = tl.load(as_ptr + 1 * stride_asm + kb * stride_ask, mask=mask_k, other=0.0)
            acc1 += tl.sum(w * (a * s * bs)[None, :], axis=1)
        if M > 2:
            a = tl.load(a_ptr + 2 * stride_am + offs_k * stride_ak, mask=mask_k, other=0.0).to(tl.float32)
            s = tl.load(as_ptr + 2 * stride_asm + kb * stride_ask, mask=mask_k, other=0.0)
            acc2 += tl.sum(w * (a * s * bs)[None, :], axis=1)
        if M > 3:
            a = tl.load(a_ptr + 3 * stride_am + offs_k * stride_ak, mask=mask_k, other=0.0).to(tl.float32)
            s = tl.load(as_ptr + 3 * stride_asm + kb * stride_ask, mask=mask_k, other=0.0)
            acc3 += tl.sum(w * (a * s * bs)[None, :], axis=1)

    if SPLIT_K > 1:
        base = c_ptr + pid_k * M * N
        tl.store(base + 0 * N + offs_n, acc0, mask=mask_n)
        if M > 1:
            tl.store(base + 1 * N + offs_n, acc1, mask=mask_n)
        if M > 2:
            tl.store(base + 2 * N + offs_n, acc2, mask=mask_n)
        if M > 3:
            tl.store(base + 3 * N + offs_n, acc3, mask=mask_n)
    else:
        if OUT_FP32:
            tl.store(c_ptr + 0 * stride_cm + offs_n * stride_cn, acc0, mask=mask_n)
            if M > 1:
                tl.store(c_ptr + 1 * stride_cm + offs_n * stride_cn, acc1, mask=mask_n)
            if M > 2:
                tl.store(c_ptr + 2 * stride_cm + offs_n * stride_cn, acc2, mask=mask_n)
            if M > 3:
                tl.store(c_ptr + 3 * stride_cm + offs_n * stride_cn, acc3, mask=mask_n)
        else:
            tl.store(c_ptr + 0 * stride_cm + offs_n * stride_cn, acc0.to(c_ptr.dtype.element_ty), mask=mask_n)
            if M > 1:
                tl.store(c_ptr + 1 * stride_cm + offs_n * stride_cn, acc1.to(c_ptr.dtype.element_ty), mask=mask_n)
            if M > 2:
                tl.store(c_ptr + 2 * stride_cm + offs_n * stride_cn, acc2.to(c_ptr.dtype.element_ty), mask=mask_n)
            if M > 3:
                tl.store(c_ptr + 3 * stride_cm + offs_n * stride_cn, acc3.to(c_ptr.dtype.element_ty), mask=mask_n)


GEMV_MAX_M = 4


def fp8_gemv_block32(
    A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, C: torch.Tensor,
    config: Dict[str, Any], partials: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch the GEMV for A [M<=4, K] fp8, B [N, K] fp8, As [M, K/32], Bs [N/32, K/32] into C [M, N]."""
    M, K = A.shape
    N = B.shape[0]
    assert 1 <= M <= GEMV_MAX_M
    bn, bk, split_k = config["BLOCK_N"], config["BLOCK_K"], config.get("SPLIT_K", 1)
    assert 32 % bn == 0, "BLOCK_N must divide the 32-row block-scale granularity"
    if split_k > 1:
        if partials is None:
            partials = torch.empty((split_k, M, N), device=A.device, dtype=torch.float32)
        out = partials
    else:
        out = C
    grid = (triton.cdiv(N, bn), split_k)
    _fp8_gemv_block32[grid](
        A, B, out, As, Bs, N, K,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        M=M, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split_k, OUT_FP32=(C.dtype == torch.float32),
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )
    if split_k > 1:
        from sglang.kernels.ops.quantization.fp8_kernel import _reduce_block_fp8_split_k

        _reduce_block_fp8_split_k[(triton.cdiv(M * N, 256),)](partials, C, M * N, split_k, 256)
    return C


def gemv_space() -> List[Dict[str, Any]]:
    space = []
    for bn, bk, split_k, warps, stages in itertools.product(
        [8, 16, 32], [128, 256, 512, 1024], [1, 2, 4], [1, 2, 4, 8], [1, 2, 3]
    ):
        if bn * bk // 32 < warps * 32 // 4:  # fewer than 8 elements per thread: skip
            continue
        space.append({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": split_k, "num_warps": warps, "num_stages": stages})
    return space


# ---------------------------------------------------------------------------------------------
# bench harness: rotated L2-cold weights captured in one CUDA graph, CUDA events around the replay


def make(m: int, n: int, k: int, gen: torch.Generator):
    a = (torch.randn(m, k, device="cuda", generator=gen) * 2).to(FP8)
    b = (torch.randn(n, k, device="cuda", generator=gen) * 2).to(FP8)
    a_s = (torch.rand(m, k // BLOCK, device="cuda", generator=gen) + 0.5).to(torch.float32)
    b_s = (torch.rand(n // BLOCK, k // BLOCK, device="cuda", generator=gen) + 0.5).to(torch.float32)
    return a, b, a_s, b_s


def reference(a, b, a_s, b_s, chunk: int = 2048) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[0]
    a32 = a.float() * a_s.repeat_interleave(BLOCK, dim=1)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    for n0 in range(0, n, chunk):
        n1 = min(n, n0 + chunk)
        bs = b_s[n0 // BLOCK : n1 // BLOCK].repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
        out[:, n0:n1] = a32 @ (b[n0:n1].float() * bs).t()
    return out


def within_tolerance(out: torch.Tensor, ref: torch.Tensor) -> bool:
    return bool(torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2))


class Timer:
    """Per-launch microseconds for a kernel the way the decode step sees it.

    The T3.21 method (256 MiB flush, one event pair per launch) bottoms out at ~12.3 us for any
    kernel — the event pair itself — which hides everything a GEMV could win on the 5-20 MB
    shapes.  Here `fn(i)` is called over `rotations` distinct weight copies (total >= 2x the
    128 MB L2, so every launch streams from GDDR7, as the 40 per-layer weights do in a step),
    the whole rotation is captured in one CUDA graph (no Python / launch overhead, as in
    serving), and the graph replay is timed with one event pair; the reported figure is
    replay time / rotations.
    """

    def __init__(self, rotations: int):
        self.rotations = rotations

    def run(self, fn: Callable[[int], torch.Tensor], iters: int) -> List[float]:
        for i in range(self.rotations):
            fn(i)  # warm: JIT / config lookup outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(self.rotations):
                fn(i)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        times: List[float] = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            g.replay()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0 / self.rotations)
        del g
        return times


def rotations_for(weight_bytes: int, l2_bytes: int = 128 << 20) -> int:
    return max(2, -(-2 * l2_bytes // weight_bytes))


def summarize(times: List[float]) -> Tuple[float, float, float]:
    med = statistics.median(times)
    q = statistics.quantiles(times, n=10) if len(times) >= 2 else [med, med]
    return med, q[0], q[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=str, default="1,2,3,4")
    ap.add_argument("--iters", type=int, default=30, help="timed launches per candidate (final)")
    ap.add_argument("--screen-iters", type=int, default=5, help="launches per config in the screening pass")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--shapes", type=str, default=None, help="comma list of N:K, default all seven")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--configs-out", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="smaller config space")
    args = ap.parse_args()

    from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton

    ms = [int(x) for x in args.ms.split(",")]
    shapes = SHAPES if not args.shapes else [tuple(int(v) for v in s.split(":")) for s in args.shapes.split(",")]
    space = gemv_space()
    if args.quick:
        space = [c for c in space if c["num_stages"] == 2 and c["SPLIT_K"] in (1, 2)]
    props = torch.cuda.get_device_properties(0)
    results: Dict[str, Any] = {
        "device": props.name, "sm_count": props.multi_processor_count, "torch": torch.__version__,
        "triton": triton.__version__, "iters": args.iters, "seeds": args.seeds, "method": "rotated-weights-in-cuda-graph",
        "space_size": len(space), "rows": [],
    }
    winners: Dict[str, Dict[str, Any]] = {}
    print(f"{props.name}, {props.multi_processor_count} SMs; {len(space)} GEMV configs; median of {args.iters} graph replays, per launch")
    print(f"{'N':>6} {'K':>5} {'M':>2} {'candidate':<9} {'us':>8} {'p10-p90':>15} {'GB/s':>7} {'maxdiff':>9} {'ok':>3} config")
    for n, k in shapes:
        weight_bytes = n * k + (n // BLOCK) * (k // BLOCK) * 4
        rot = rotations_for(weight_bytes)
        timer = Timer(rot)
        for m in ms:
            gens = [torch.Generator(device="cuda").manual_seed(20261012 + s) for s in range(args.seeds)]
            inputs = [make(m, n, k, g) for g in gens]
            refs = [reference(*inp) for inp in inputs]
            a, b, a_s, b_s = inputs[0]
            # rotation set: distinct weight copies so every launch is L2-cold
            b_rot = [b] + [b.clone() for _ in range(rot - 1)]
            bs_rot = [b_s] + [b_s.clone() for _ in range(rot - 1)]
            c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            # baseline: the overlay's E20-tuned Triton path (allocates its output, as in serving)
            def base_fn(i):
                return w8a8_block_fp8_matmul_triton(a, b_rot[i], a_s, bs_rot[i], [BLOCK, BLOCK], output_dtype=torch.bfloat16)

            base_out = base_fn(0)
            torch.cuda.synchronize()
            base_ok = within_tolerance(base_out, refs[0])
            bt = timer.run(base_fn, args.iters)
            bmed, bp10, bp90 = summarize(bt)
            print(f"{n:>6} {k:>5} {m:>2} {'triton':<9} {bmed:>8.2f} {bp10:>6.2f}-{bp90:<8.2f} {weight_bytes/bmed/1e3:>7.0f} {'-':>9} {'y' if base_ok else 'N':>3} e20 rot={rot}")
            results["rows"].append({"N": n, "K": k, "M": m, "candidate": "triton_e20", "median_us": bmed, "p10_us": bp10, "p90_us": bp90, "within_tol": base_ok, "rotations": rot})

            # screening pass over the space
            screened: List[Tuple[float, Dict[str, Any]]] = []
            for cfg in space:
                split_k = cfg["SPLIT_K"]
                partials = torch.empty((split_k, m, n), device="cuda", dtype=torch.float32) if split_k > 1 else None
                fn = lambda i, cfg=cfg, partials=partials: fp8_gemv_block32(a, b_rot[i], a_s, bs_rot[i], c, cfg, partials)
                try:
                    fn(0)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001 — a config that cannot compile/launch is skipped
                    results["rows"].append({"N": n, "K": k, "M": m, "candidate": "gemv", "config": cfg, "error": str(exc)[:200]})
                    continue
                if not within_tolerance(c, refs[0]):
                    results["rows"].append({"N": n, "K": k, "M": m, "candidate": "gemv", "config": cfg, "rejected": "tolerance"})
                    continue
                t = statistics.median(timer.run(fn, args.screen_iters))
                screened.append((t, cfg))
            screened.sort(key=lambda x: x[0])
            # final pass: top 5, full iters, all seeds
            best: Optional[Dict[str, Any]] = None
            for _, cfg in screened[:5]:
                split_k = cfg["SPLIT_K"]
                partials = torch.empty((split_k, m, n), device="cuda", dtype=torch.float32) if split_k > 1 else None
                ok = True
                maxdiff = 0.0
                for (aa, bb, ass, bss), ref in zip(inputs, refs):
                    cc = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                    fp8_gemv_block32(aa, bb, ass, bss, cc, cfg, partials)
                    torch.cuda.synchronize()
                    ok &= within_tolerance(cc, ref)
                    tri = w8a8_block_fp8_matmul_triton(aa, bb, ass, bss, [BLOCK, BLOCK], output_dtype=torch.bfloat16)
                    maxdiff = max(maxdiff, (cc.float() - tri.float()).abs().max().item())
                fn = lambda i, cfg=cfg, partials=partials: fp8_gemv_block32(a, b_rot[i], a_s, bs_rot[i], c, cfg, partials)
                times = timer.run(fn, args.iters)
                med, p10, p90 = summarize(times)
                row = {"N": n, "K": k, "M": m, "candidate": "gemv", "config": cfg, "median_us": med, "p10_us": p10,
                       "p90_us": p90, "within_tol_all_seeds": ok, "max_abs_diff_vs_triton": maxdiff,
                       "bit_identical_to_triton": maxdiff == 0.0}
                results["rows"].append(row)
                print(f"{n:>6} {k:>5} {m:>2} {'gemv':<9} {med:>8.2f} {p10:>6.2f}-{p90:<8.2f} {weight_bytes/med/1e3:>7.0f} {maxdiff:>9.2e} {'y' if ok else 'N':>3} {cfg}")
                if ok and (best is None or med < best["median_us"]):
                    best = row
            if best is not None:
                spread = max(bp90 - bp10, best["p90_us"] - best["p10_us"])
                gain = bmed - best["median_us"]
                accepted = gain > 0.02 * bmed and gain > spread
                print(f"{'':>16} best gemv {best['median_us']:.2f} us vs triton {bmed:.2f} us: {gain/bmed*100:+.1f} % spread {spread:.2f} -> {'ACCEPT' if accepted else 'reject'}")
                if accepted:
                    winners.setdefault(f"N={n},K={k}", {})[str(m)] = {**best["config"], "us": best["median_us"], "triton_us": bmed}
            del inputs, refs, c, b_rot, bs_rot
            torch.cuda.empty_cache()
    results["winners"] = winners
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")
    if args.configs_out:
        with open(args.configs_out, "w") as f:
            json.dump({"device": props.name, "block_shape": [BLOCK, BLOCK], "max_m": GEMV_MAX_M, "shapes": winners}, f, indent=1)
        print(f"wrote {args.configs_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

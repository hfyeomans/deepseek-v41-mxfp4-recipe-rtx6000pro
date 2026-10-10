#!/usr/bin/env python3
"""T3.21 window 1 — large-M block-[32,32] FP8 GEMM candidates on SM120, timed L2-cold.

V4.1-Flash's dense projections are FP8 e4m3 weights with FP32 [32, 32] block scales and
per-token-group (1 x 32) FP32 activation scales.  Prefill runs them at M = chunk tokens
(8192 by default).  The serving path is SGLang's Triton `w8a8_block_fp8_matmul_triton`
with the E20-tuned configs (≈ 200–240 TFLOPS at M = 4096, well under the SM120 FP8 roof).

Candidates (same inputs, same math where marked exact):
  triton_tuned  SGLang Triton kernel + overlay configs                    exact (the baseline)
  cutile_32     FlashInfer cuTile `gemm_fp8_nt_groupwise(backend="cutile")`
                at scale_granularity_mnk=(1, 32, 32)                        exact-in-intent (checked)
  cutlass_128   FlashInfer CUTLASS SM120 groupwise at (1, 128, 128), scales
                subsampled to 128-blocks                                    NOT the same math; ceiling
                                                                            for a K-128 block kernel
  scaled_mm     torch._scaled_mm per-tensor FP8 -> bf16 (cuBLASLt)         NOT the same math; roofline

Every candidate is compared against an FP32 dequantised reference (max rel err, with the
caveat that cutlass_128 / scaled_mm use different scales and are expected to differ).

Timing: 256 MiB L2 flush before every launch, CUDA events, median of --iters.

Run inside the recipe container with the overlay mounted (GPU 1 free):

  docker run --rm --gpus '"device=1"' --entrypoint python3 \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v $PWD/kernels/t321:/tmp/t321:ro -v /tmp/t321-out:/out dsv41-kt:recipe \
    /tmp/t321/bench_fp8_prefill_gemm.py --out /out/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

FP8 = torch.float8_e4m3fn
BLOCK = 32
# The seven dense (N, K) shapes of the TP2 rank (E20).
SHAPES: List[Tuple[int, int]] = [
    (16384, 1280),
    (1792, 5120),
    (2304, 5120),
    (25600, 6144),
    (4096, 1280),
    (5120, 1152),
    (5120, 4096),
]


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


def rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    err = (out.float() - ref).abs()
    return (err / ref.abs().clamp_min(1.0)).max().item()


class Timer:
    def __init__(self, flush_bytes: int = 256 << 20):
        self.flush = torch.empty(flush_bytes, dtype=torch.uint8, device="cuda")

    def run(self, fn: Callable[[], torch.Tensor], iters: int) -> List[float]:
        fn()  # warm (JIT / autotune happens here, outside the timed region)
        torch.cuda.synchronize()
        times: List[float] = []
        for _ in range(iters):
            self.flush.zero_()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000.0)  # us
        return times


def candidates(a, b, a_s, b_s, out_bf16: torch.Tensor) -> Dict[str, Tuple[Callable[[], torch.Tensor], bool]]:
    """name -> (fn, exact_math)."""
    m, k = a.shape
    n = b.shape[0]
    cands: Dict[str, Tuple[Callable[[], torch.Tensor], bool]] = {}

    from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton

    cands["triton_tuned"] = (
        lambda: w8a8_block_fp8_matmul_triton(a, b, a_s, b_s, [BLOCK, BLOCK], output_dtype=torch.bfloat16),
        True,
    )

    import flashinfer.gemm as fg
    # The public dispatcher refuses anything but (1, 128, 128) for cuTile; the kernel module
    # itself takes any (block_n, block_k) (it asserts BLOCK_SIZE_N == block_n, BLOCK_SIZE_K == block_k).
    from flashinfer.gemm.kernels.cutile.gemm_fp8_nt_groupwise_cutile import gemm_fp8_nt_groupwise_cutile

    def cutile_32():
        return gemm_fp8_nt_groupwise_cutile(
            a, b, a_s, b_s, out_bf16, scale_granularity_mnk=(1, BLOCK, BLOCK), scale_major_mode="K",
        )

    cands["cutile_32"] = (cutile_32, True)

    if k % 128 == 0 and n % 128 == 0:
        a_s128 = a_s[:, ::4].contiguous()  # one of the four 32-scales per 128 block: wrong math, right cost
        b_s128 = b_s[::4, ::4].contiguous()

        def cutlass_128():
            return fg.gemm_fp8_nt_groupwise(
                a, b, a_s128, b_s128, scale_major_mode="K", scale_granularity_mnk=(1, 128, 128),
                out=out_bf16, backend="cutlass",
            )

        cands["cutlass_128"] = (cutlass_128, False)

        def cutile_128():
            return gemm_fp8_nt_groupwise_cutile(
                a, b, a_s128, b_s128, out_bf16, scale_granularity_mnk=(1, 128, 128), scale_major_mode="K",
            )

        cands["cutile_128"] = (cutile_128, False)

    one = torch.ones((), device="cuda", dtype=torch.float32)
    bt = b.t()  # (k, n) column-major view, what _scaled_mm wants for the second operand

    def scaled_mm():
        return torch._scaled_mm(a, bt, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

    cands["scaled_mm"] = (scaled_mm, False)
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", type=str, default="4096,8192,16384")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--shapes", type=str, default=None, help="comma list of N:K, default all seven")
    ap.add_argument("--skip", type=str, default="", help="comma list of candidate names to skip")
    args = ap.parse_args()

    ms = [int(x) for x in args.ms.split(",")]
    shapes = SHAPES if not args.shapes else [tuple(int(v) for v in s.split(":")) for s in args.shapes.split(",")]
    skip = set(filter(None, args.skip.split(",")))

    gen = torch.Generator(device="cuda").manual_seed(20261009)
    timer = Timer()
    props = torch.cuda.get_device_properties(0)
    results = {
        "device": props.name,
        "sm_count": props.multi_processor_count,
        "torch": torch.__version__,
        "iters": args.iters,
        "flush_mib": 256,
        "rows": [],
    }
    print(f"{props.name}, {props.multi_processor_count} SMs; median of {args.iters} L2-cold launches")
    print(f"{'N':>6} {'K':>5} {'M':>6} {'candidate':<13} {'us':>10} {'p10-p90':>16} {'TFLOPS':>8} {'max_rel':>9} exact")
    for n, k in shapes:
        for m in ms:
            a, b, a_s, b_s = make(m, n, k, gen)
            ref = reference(a, b, a_s, b_s)
            out_bf16 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
            flops = 2.0 * m * n * k
            for name, (fn, exact) in candidates(a, b, a_s, b_s, out_bf16).items():
                if name in skip:
                    continue
                try:
                    o = fn()
                    torch.cuda.synchronize()
                    err = rel_err(o, ref)
                    times = timer.run(fn, args.iters)
                except Exception as exc:  # noqa: BLE001 — report and keep going
                    print(f"{n:>6} {k:>5} {m:>6} {name:<13} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
                    results["rows"].append({"N": n, "K": k, "M": m, "candidate": name, "error": str(exc)[:400]})
                    continue
                med = statistics.median(times)
                q = statistics.quantiles(times, n=10)
                tflops = flops / med / 1e6
                print(f"{n:>6} {k:>5} {m:>6} {name:<13} {med:>10.1f} {q[0]:>7.1f}-{q[-1]:<8.1f} {tflops:>8.1f} {err:>9.2e} {'y' if exact else 'n'}")
                results["rows"].append({
                    "N": n, "K": k, "M": m, "candidate": name, "exact_math": exact,
                    "median_us": med, "p10_us": q[0], "p90_us": q[-1], "tflops": tflops, "max_rel_err": err,
                })
            del a, b, a_s, b_s, ref, out_bf16
            torch.cuda.empty_cache()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

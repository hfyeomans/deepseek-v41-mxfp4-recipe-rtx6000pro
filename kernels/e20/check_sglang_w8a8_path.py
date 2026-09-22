"""Run SGLang's real `w8a8_block_fp8_matmul` (overlay path, tuned configs) at the
M values a mixed decode/prefill batch produces and compare against an FP32 reference.

Purpose: rule the E20 tuned configs (SPLIT_K / SWAP_AB on SM120) in or out as a
cause of the `probability tensor contains inf/nan` crash. Runs inside the recipe
container with the overlay bind-mounted:

  docker run --rm --gpus '"device=1"' --entrypoint python3 \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v $PWD/kernels/e20:/tmp/e20:ro dsv41-kt:recipe /tmp/e20/check_sglang_w8a8_path.py

Exit code 1 if any shape/M fails (non-finite output or error above tolerance).
"""

import sys

import torch

from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton as w8a8_block_fp8_matmul

SHAPES = [(1792, 5120), (2304, 5120), (5120, 4096), (5120, 1152), (16384, 1280), (4096, 1280), (25600, 6144)]
MS = [1, 2, 3, 4, 5, 7, 768, 769, 1024, 1025]
BLOCK = [32, 32]
FP8 = torch.float8_e4m3fn


def make(m, n, k, gen):
    a = (torch.randn(m, k, device="cuda", generator=gen) * 2).to(FP8)
    b = (torch.randn(n, k, device="cuda", generator=gen) * 2).to(FP8)
    a_s = (torch.rand(m, k // BLOCK[1], device="cuda", generator=gen) + 0.5).to(torch.float32)
    b_s = (torch.rand(n // BLOCK[0], k // BLOCK[1], device="cuda", generator=gen) + 0.5).to(torch.float32)
    return a, b, a_s, b_s


def reference(a, b, a_s, b_s, m, n, k, chunk=4096):
    # chunked over N to bound memory (25600x6144 case)
    a32 = a.float()
    a32 = a32 * a_s.repeat_interleave(BLOCK[1], dim=1)
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    for n0 in range(0, n, chunk):
        n1 = min(n, n0 + chunk)
        b32 = b[n0:n1].float() * b_s[n0 // BLOCK[0]: n1 // BLOCK[0]].repeat_interleave(BLOCK[0], dim=0).repeat_interleave(BLOCK[1], dim=1)
        out[:, n0:n1] = a32 @ b32.t()
    return out


def main():
    gen = torch.Generator(device="cuda").manual_seed(20260922)
    bad = 0
    print(f"{'N':>6} {'K':>5} {'M':>5} {'max_abs_err':>12} {'max_rel_err':>12} finite")
    for n, k in SHAPES:
        for m in MS:
            a, b, a_s, b_s = make(m, n, k, gen)
            out = w8a8_block_fp8_matmul(a, b, a_s, b_s, BLOCK, output_dtype=torch.bfloat16)
            torch.cuda.synchronize()
            ref = reference(a, b, a_s, b_s, m, n, k)
            finite = bool(torch.isfinite(out).all())
            err = (out.float() - ref).abs()
            max_abs = err.max().item()
            rel = (err / ref.abs().clamp_min(1.0)).max().item()
            ok = finite and rel < 2e-2
            bad += not ok
            print(f"{n:>6} {k:>5} {m:>5} {max_abs:>12.4f} {rel:>12.5f} {finite} {'' if ok else '  <-- FAIL'}")
            del a, b, a_s, b_s, out, ref
    print("RESULT", "FAIL" if bad else "PASS", f"({bad} failing cases)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

import os, torch
os.environ["SGLANG_DSV41_FP8_GEMV"] = "1"
from sglang.kernels.ops.quantization.fp8_kernel import w8a8_block_fp8_matmul_triton
from sglang.kernels.ops.quantization import fp8_gemv_dsv41 as g
tbl = g.get_fp8_gemv_configs(32, 32)
print("table shapes:", len(tbl), sorted(tbl)[:3])
torch.manual_seed(0)
for (N, K) in [(4096, 1280), (16384, 1280), (1792, 5120), (25600, 6144)]:
    for M in (1, 2, 3, 4, 5):
        B = (torch.randn(N, K, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        Bs = torch.rand(N // 32, K // 32, device="cuda") * 0.01 + 0.005
        A = (torch.randn(1, M, K, device="cuda")).to(torch.float8_e4m3fn)  # 3-D batch dim
        As = torch.rand(1, M, K // 32, device="cuda") * 0.01 + 0.005
        ref = ((A.float() * As.repeat_interleave(32, dim=-1)) @ (B.float() * Bs.repeat_interleave(32, 0).repeat_interleave(32, 1)).T)
        C = w8a8_block_fp8_matmul_triton(A, B, As, Bs, [32, 32], torch.bfloat16)
        cfg = g.fp8_gemv_config(N, K, M, 32, 32)
        err = (C.float() - ref).abs().max().item(); rel = err / ref.abs().max().item()
        print(f"N={N} K={K} M={M} shape={tuple(C.shape)} gemv={'yes' if cfg else 'no '} maxabs={err:.4f} rel={rel:.2e}")
        assert C.shape == (1, M, N) and rel < 1e-2
print("DISPATCH OK")

"""Scratch: how long does the host block inside flashinfer cutlass_fused_moe (SM120 MXFP4 x MXFP8)?

Runs inside dsv41-kt:recipe on one GPU with the overlay's call shape (8192 rows, hidden 4096,
inter_tp 1440, E experts, top-k 6 with half the slots masked -1) and reports the host-side
duration of the call (return of the Python call) vs the GPU time (synchronize).
"""
import sys, time, statistics, torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from flashinfer import mxfp8_quantize, block_scale_interleave

E = int(sys.argv[1]) if len(sys.argv) > 1 else 165
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
hidden, inter = 5120, 1152
dev = torch.device("cuda:0")
torch.manual_seed(0)
w13 = torch.randint(0, 255, (E, 2 * inter, hidden // 2), dtype=torch.uint8, device=dev).view(torch.int8)
w2 = torch.randint(0, 255, (E, hidden, inter // 2), dtype=torch.uint8, device=dev).view(torch.int8)
s13 = torch.full((E, 2 * inter, hidden // 32), 127, dtype=torch.uint8, device=dev)
s2 = torch.full((E, hidden, inter // 32), 127, dtype=torch.uint8, device=dev)
gscale = torch.ones(E, dtype=torch.float32, device=dev)
x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev)
ids = torch.randint(0, E, (rows, 6), dtype=torch.int32, device=dev)
ids[:, 3:] = -1
w = torch.rand(rows, 6, dtype=torch.float32, device=dev)
out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
ws = torch.empty(2 * 1024**3, dtype=torch.int8, device=dev)

def call():
    xq, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    cutlass_fused_moe(
        input=xq, token_selected_experts=ids, token_final_scales=w,
        fc1_expert_weights=w13.view(torch.int64), fc2_expert_weights=w2.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[s13.view(torch.int32), gscale, s2.view(torch.int32), gscale],
        input_sf=sf, tp_size=2, tp_rank=0, ep_size=1, ep_rank=0,
        use_w4_group_scaling=False, use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu, tune_max_num_tokens=rows,
        output=out, workspace_buffer=ws,
    )

for _ in range(3):
    call(); torch.cuda.synchronize()
host, total = [], []
for _ in range(10):
    torch.cuda.synchronize(); t0 = time.perf_counter(); call(); t1 = time.perf_counter()
    torch.cuda.synchronize(); t2 = time.perf_counter()
    host.append((t1 - t0) * 1e3); total.append((t2 - t0) * 1e3)
print(f"E={E} rows={rows}: host call median {statistics.median(host):.1f} ms (min {min(host):.1f}), total median {statistics.median(total):.1f} ms")
if "--prof" in sys.argv:
    import cProfile, pstats
    torch.cuda.synchronize()
    pr = cProfile.Profile(); pr.enable(); call(); pr.disable(); torch.cuda.synchronize()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(18)

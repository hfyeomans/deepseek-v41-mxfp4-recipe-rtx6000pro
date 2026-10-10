"""Scratch: exercise the overlay's _pin_moe_runner_for_expert_count against real FlashInfer."""
import sys, time, statistics, torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe import core as fi_core
from flashinfer.fused_moe.core import ActivationType
from flashinfer import mxfp8_quantize
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as fc

rows = 8192
Ea, Eb = 165, 219
hidden, inter = 5120, 1152
dev = torch.device("cuda:0")
torch.manual_seed(0)


def make(E):
    w13 = torch.randint(0, 255, (E, 2 * inter, hidden // 2), dtype=torch.uint8, device=dev).view(torch.int8)
    w2 = torch.randint(0, 255, (E, hidden, inter // 2), dtype=torch.uint8, device=dev).view(torch.int8)
    s13 = torch.full((E, 2 * inter, hidden // 32), 127, dtype=torch.uint8, device=dev)
    s2 = torch.full((E, hidden, inter // 32), 127, dtype=torch.uint8, device=dev)
    g = torch.ones(E, dtype=torch.float32, device=dev)
    ids = torch.randint(0, E, (rows, 6), dtype=torch.int32, device=dev)
    ids[:, 3:] = -1
    return dict(w13=w13, w2=w2, s13=s13, s2=s2, g=g, ids=ids)


x = torch.randn(rows, hidden, dtype=torch.bfloat16, device=dev)
w = torch.rand(rows, 6, dtype=torch.float32, device=dev)
out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
ws = torch.empty(2 * 1024**3, dtype=torch.int8, device=dev)
sets = {Ea: make(Ea), Eb: make(Eb)}
ref = {}


def call(E, pin):
    s = sets[E]
    xq, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    if pin:
        fc._pin_moe_runner_for_expert_count(
            device=dev, num_experts_total=E, x_dtype=xq.dtype, weight_dtype=torch.int64,
            output_dtype=torch.bfloat16, use_w4_group_scaling=False, use_mxfp8_act_scaling=True,
            use_fused_finalize=True, use_wfp4afp8_humming=False,
        )
    cutlass_fused_moe(
        input=xq, token_selected_experts=s["ids"], token_final_scales=w,
        fc1_expert_weights=s["w13"].view(torch.int64), fc2_expert_weights=s["w2"].view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[s["s13"].view(torch.int32), s["g"], s["s2"].view(torch.int32), s["g"]],
        input_sf=sf, tp_size=2, tp_rank=0, ep_size=1, ep_rank=0,
        use_w4_group_scaling=False, use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu, tune_max_num_tokens=rows,
        output=out, workspace_buffer=ws, use_fused_finalize=True,
    )


# reference outputs on the shared runner (no pin)
for E in (Ea, Eb):
    call(E, False); torch.cuda.synchronize(); ref[E] = out.clone()

for pin in (False, True):
    host = {Ea: [], Eb: []}
    for E in [Ea, Eb] * 6:
        torch.cuda.synchronize(); t0 = time.perf_counter(); call(E, pin); t1 = time.perf_counter()
        torch.cuda.synchronize(); host[E].append((t1 - t0) * 1e3)
        assert torch.equal(out, ref[E]), f"output mismatch E={E} pin={pin}"
    for E, v in host.items():
        print(f"pin={pin} E={E}: host median {statistics.median(v):.1f} ms max {max(v):.1f}")
arch = "".join(str(v) for v in torch.cuda.get_device_capability(dev))
print("runners by count:", {k[1]: id(v) for k, v in fc._MOE_RUNNERS_BY_EXPERT_COUNT.items()})
print("runner_dict keys:", list(fi_core.get_cutlass_fused_moe_module(arch).MoERunner.runner_dict))
print("outputs bit-identical with and without pinning: OK")

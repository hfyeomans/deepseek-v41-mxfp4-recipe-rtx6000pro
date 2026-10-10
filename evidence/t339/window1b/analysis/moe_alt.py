"""Scratch: host stall when one flashinfer FusedMoeRunner alternates expert counts.

MoeGemmRunner::getMaxWorkspaceSize caches the workspace size for ONE expert count
(num_experts_); every call with a different count recomputes it by trying every
TMA-WS config and catching a TllmException (with a symbolized backtrace) per
unsupported one. The serve alternates 165 (streamed) / 219 (resident) experts on
the same runner, so every call pays that. Modes:
  same      : E_a only (cache hit after first call)
  alt       : E_a, E_b, E_a, E_b ... on the shared runner (serve pattern)
  alt-split : same alternation, but a separate C++ runner per expert count
"""
import sys, time, statistics, torch
from flashinfer.fused_moe import cutlass_fused_moe
from flashinfer.fused_moe import core as fi_core
from flashinfer.fused_moe.core import ActivationType
from flashinfer import mxfp8_quantize

mode = sys.argv[1] if len(sys.argv) > 1 else "alt"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
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

# --- alt-split: one C++ runner per expert count, swapped into runner_dict before each call
_split_runners = {}


def use_runner_for(E):
    arch = "".join(str(v) for v in torch.cuda.get_device_capability(dev))
    ns = fi_core.get_cutlass_fused_moe_module(arch)
    keys = list(ns.MoERunner.runner_dict)
    assert len(keys) == 1, keys  # populated by the warm-up call
    key = keys[0]
    if E not in _split_runners:
        raw = fi_core.gen_cutlass_fused_moe_sm120_module(False).build_and_load()
        _split_runners[E] = raw.init(*key)
    ns.MoERunner.runner_dict[key] = _split_runners[E]


def call(E):
    s = sets[E]
    xq, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    cutlass_fused_moe(
        input=xq, token_selected_experts=s["ids"], token_final_scales=w,
        fc1_expert_weights=s["w13"].view(torch.int64), fc2_expert_weights=s["w2"].view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[s["s13"].view(torch.int32), s["g"], s["s2"].view(torch.int32), s["g"]],
        input_sf=sf, tp_size=2, tp_rank=0, ep_size=1, ep_rank=0,
        use_w4_group_scaling=False, use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu, tune_max_num_tokens=rows,
        output=out, workspace_buffer=ws,
    )


seq = {"same": [Ea] * 12, "alt": [Ea, Eb] * 6, "alt-split": [Ea, Eb] * 6}[mode]
for E in seq[:2]:
    call(E); torch.cuda.synchronize()
host = {Ea: [], Eb: []}
for E in seq:
    if mode == "alt-split":
        use_runner_for(E)
    torch.cuda.synchronize(); t0 = time.perf_counter(); call(E); t1 = time.perf_counter()
    torch.cuda.synchronize()
    host[E].append((t1 - t0) * 1e3)
for E, v in host.items():
    if v:
        print(f"mode={mode} E={E}: host call median {statistics.median(v):.1f} ms (min {min(v):.1f}, max {max(v):.1f}) n={len(v)}")

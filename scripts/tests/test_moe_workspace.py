#!/usr/bin/env python3
"""GPU test for the persistent FlashInfer CUTLASS MXFP4 MoE workspace (recipe E34d-v (i)).

Runs INSIDE the recipe image with the overlay's python/sglang mounted over the installed
package (scripts/tests/run-moe-workspace-tests.sh); needs one GPU for a few seconds (the
SM120 `cutlass_fused_moe` MXFP4 x MXFP8 kernel). Safe while a server is not using GPU 0.

Background: without `workspace_buffer=` FlashInfer allocates and frees a ~1 GiB scratch
tensor on every `cutlass_fused_moe` call. Over a long prefill that churn fragments the
caching allocator until no contiguous block of that size is left (1M x 1 died at a ~306K
prefix, 262K x 4 at ~190K, both with >1.8 GiB reserved-but-free). The overlay's
`fused_experts_none_to_flashinfer_mxfp4` now hands FlashInfer one persistent per-device
buffer sized by `cutlass_fused_moe_workspace_size` for the prefill-chunk ceiling.

What a wrong implementation would get wrong, and which assertion catches it:
  - a buffer that changes the kernel's result (the kernel reading stale scratch it does
    not initialise, or a mis-sized buffer the binding does not reject) -> bit-equality of
    the with-buffer and per-call outputs at every row count, including the ceiling row
    count itself and a 1-row call.
  - a size estimate that does not match what the binding will demand (wrong hidden dim,
    top-k, expert count, dtype or flag) -> `size matches the binding`: a buffer of
    exactly `size_for(rows)` must be accepted for `rows` and a buffer 128 bytes short of
    it must be rejected by the binding; an undersized estimate would already fail the
    equality runs with the binding's "workspace_buffer too small". (The size is measured
    to be independent of the intermediate dim, so a wrong FP4 packing multiplier is
    harmless and not detectable here.)
  - a non-monotonic size (so a ceiling-sized buffer could be too small for fewer rows)
    -> `size is monotonic` over a ladder of row counts.
  - reallocating the buffer (a graph that captured its address would replay into freed
    memory) -> the data_ptr is the same after every call, including the one above the
    ceiling, which must fall back to per-call allocation and warn once.
  - allocating inside CUDA graph capture (the buffer would live in the graph pool) ->
    the capture guard raises when `is_current_stream_capturing` reports capture.
  - a kill switch that does not actually bypass the buffer -> with
    SGLANG_FLASHINFER_MOE_PERSISTENT_WORKSPACE=0 no buffer may exist after a call.

The model shape is DSv4.1-Flash's MoE slice per TP=2 rank (hidden 5120, intermediate
1152 per partition, top-k 6, SwigluStep with clamp 10) over 8 experts so the weights stay
small; the exact DSv4 workspace size (219 GPU experts, 8192-row chunk) is only reported.
"""
from __future__ import annotations

import contextlib
import logging
import sys

import torch

import sglang.srt.layers.moe.moe_runner.flashinfer_cutlass as fc
from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

DEV = torch.device("cuda", 0)  # the workspace store is keyed by the indexed device
HIDDEN = 5120
INTER = 1152  # moe_intermediate_size 2304 / TP 2
TOP_K = 6
EXPERTS = 8
CEILING = 2048  # stands in for the 8192-row prefill chunk


def make_quant_info():
    from flashinfer import block_scale_interleave

    g = torch.Generator(device=DEV).manual_seed(1234)
    w13 = torch.randint(0, 256, (EXPERTS, 2 * INTER, HIDDEN // 2), dtype=torch.uint8, device=DEV, generator=g)
    w2 = torch.randint(0, 256, (EXPERTS, HIDDEN, INTER // 2), dtype=torch.uint8, device=DEV, generator=g)
    # E8M0 scales near 2^-3 so random FP4 words do not overflow bf16 through the clamp.
    w13_s = torch.randint(122, 126, (EXPERTS, 2 * INTER, HIDDEN // 32), dtype=torch.uint8, device=DEV, generator=g)
    w2_s = torch.randint(122, 126, (EXPERTS, HIDDEN, INTER // 32), dtype=torch.uint8, device=DEV, generator=g)
    w13_s = block_scale_interleave(w13_s).reshape_as(w13_s).view(torch.float8_e8m0fnu)
    w2_s = block_scale_interleave(w2_s).reshape_as(w2_s).view(torch.float8_e8m0fnu)
    return fc.FlashInferCutlassMxfp4MoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_s,
        w2_weight_scale=w2_s,
        mxfp4_weight_global_scale=torch.ones(EXPERTS, dtype=torch.float32, device=DEV),
        w13_bias=None,
        w2_bias=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=torch.full((EXPERTS,), 10.0, dtype=torch.float32, device=DEV),
        use_swiglu_step=True,
        moe_tp_size=2,
        moe_tp_rank=0,
        moe_ep_size=1,
        moe_ep_rank=0,
        padded_hidden=None,
    )


def make_dispatch(rows: int, seed: int) -> StandardDispatchOutput:
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(rows, HIDDEN, dtype=torch.bfloat16, device=DEV, generator=g)
    logits = torch.randn(rows, EXPERTS, dtype=torch.float32, device=DEV, generator=g)
    w, ids = torch.topk(logits, TOP_K, dim=-1)
    w = torch.softmax(w, dim=-1)
    # some rows route only to a subset (masked experts, as the kt wrapper does)
    ids[::7, 0] = -1
    return StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(topk_weights=w, topk_ids=ids, router_logits=logits),
    )


def run(quant_info, dispatch) -> torch.Tensor:
    out = fc.fused_experts_none_to_flashinfer_mxfp4(dispatch, quant_info, MoeRunnerConfig())
    torch.cuda.synchronize()
    return out.hidden_states.clone()


def size_for(rows: int, *, experts=EXPERTS, inter=INTER, tp_size=2) -> int:
    from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
    from flashinfer.fused_moe.core import ActivationType

    return int(
        cutlass_fused_moe_workspace_size(
            rows,
            HIDDEN,
            inter,
            experts,
            TOP_K,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.int64,
            output_dtype=torch.bfloat16,
            activation_type=ActivationType.SwigluStep,
            tp_size=tp_size,
            tp_rank=0,
            use_mxfp8_act_scaling=True,
            use_fused_finalize=True,
            device=DEV,
        )
    )


def workspace():
    return fc._MOE_WORKSPACES.get(DEV)


def test_size_is_monotonic():
    ladder = [1, 2, 3, 4, 8, 16, 64, 100, 1000, 2047, 2048, 4096, 8192]
    sizes = [size_for(r) for r in ladder]
    for (r0, s0), (r1, s1) in zip(zip(ladder, sizes), zip(ladder[1:], sizes[1:])):
        assert s0 <= s1, f"size({r0})={s0} > size({r1})={s1}"
    dsv4 = size_for(8192, experts=219, inter=INTER)
    print(f"  size(8192 rows, 219 experts, DSv4 TP2 slice) = {dsv4 / 2**20:.1f} MiB")


def test_size_matches_the_binding(quant_info):
    rows = 1000
    exact = size_for(rows)
    dispatch = make_dispatch(rows, seed=7)
    with envs.SGLANG_FLASHINFER_MOE_PERSISTENT_WORKSPACE.override(0):
        ref = run(quant_info, dispatch)
    fc._MOE_WORKSPACES.clear()
    with envs.SGLANG_FLASHINFER_MOE_WORKSPACE_MAX_ROWS.override(rows):
        got = run(quant_info, dispatch)
    ws = workspace()
    assert ws is not None and ws.max_rows == rows and ws.buffer.numel() == exact, (ws and ws.buffer.numel(), exact)
    assert torch.equal(ref, got)
    # an undersized buffer must be refused by the binding, not silently used
    # (the config is pre-marked as fitting so the overlay's own check is bypassed)
    fc._MOE_WORKSPACES.clear()
    fc._MOE_WORKSPACES[DEV] = fc._MoeWorkspace(
        buffer=torch.empty(exact - 128, dtype=torch.uint8, device=DEV),
        max_rows=rows,
        fits=dict(ws.fits),
    )
    try:
        run(quant_info, dispatch)
    except Exception as e:  # noqa: BLE001
        assert "workspace_buffer too small" in str(e), str(e)[:300]
    else:
        raise AssertionError("binding accepted a buffer 128 bytes short of its own size")
    finally:
        fc._MOE_WORKSPACES.clear()


def test_outputs_match_per_call(quant_info):
    fc._MOE_WORKSPACES.clear()
    rows_list = [1, 5, 64, 777, CEILING]
    refs = {}
    with envs.SGLANG_FLASHINFER_MOE_PERSISTENT_WORKSPACE.override(0):
        for rows in rows_list:
            d = make_dispatch(rows, seed=rows)
            a, b = run(quant_info, d), run(quant_info, d)
            assert torch.equal(a, b), f"per-call path is not deterministic at {rows} rows"
            refs[rows] = a
        assert workspace() is None, "kill switch did not bypass the buffer"
    with envs.SGLANG_FLASHINFER_MOE_WORKSPACE_MAX_ROWS.override(CEILING):
        ptr = None
        for rows in rows_list + [3, CEILING]:
            got = run(quant_info, make_dispatch(rows, seed=rows))
            ws = workspace()
            assert ws is not None and ws.max_rows == CEILING, ws
            ptr = ptr or ws.buffer.data_ptr()
            assert ws.buffer.data_ptr() == ptr, "buffer was reallocated"
            if rows in refs:
                assert torch.equal(got, refs[rows]), f"output differs at {rows} rows"
            assert not got.isnan().any() and not got.isinf().any(), rows
        # above the ceiling: still correct, per-call allocation, one warning, no realloc
        rows = CEILING + 1
        d = make_dispatch(rows, seed=rows)
        with envs.SGLANG_FLASHINFER_MOE_PERSISTENT_WORKSPACE.override(0):
            ref = run(quant_info, d)
        logs = []
        handler = logging.Handler()
        handler.emit = lambda rec: logs.append(rec.getMessage())
        fc.logger.addHandler(handler)
        try:
            got = run(quant_info, d)
            got2 = run(quant_info, d)
        finally:
            fc.logger.removeHandler(handler)
        assert torch.equal(got, ref) and torch.equal(got2, ref)
        assert workspace().buffer.data_ptr() == ptr
        assert sum("exceeds the persistent workspace" in m for m in logs) == 1, logs


def test_capture_guard(quant_info):
    fc._MOE_WORKSPACES.clear()
    real = torch.cuda.is_current_stream_capturing
    torch.cuda.is_current_stream_capturing = lambda: True
    try:
        with envs.SGLANG_FLASHINFER_MOE_WORKSPACE_MAX_ROWS.override(CEILING):
            try:
                run(quant_info, make_dispatch(4, seed=4))
            except RuntimeError as e:
                assert "before CUDA graph capture" in str(e), str(e)
            else:
                raise AssertionError("allocation inside capture was not refused")
        assert workspace() is None
    finally:
        torch.cuda.is_current_stream_capturing = real
        fc._MOE_WORKSPACES.clear()


def main() -> int:
    if not torch.cuda.is_available():
        print("moe workspace tests: no GPU; skipped")
        return 3
    # the fused func reaches for the TP group only to pick an allocator for `out`
    fc.get_tp_group = lambda: None
    fc.use_symmetric_memory = lambda *a, **k: contextlib.nullcontext()
    fc.is_allocation_symmetric = lambda: False
    quant_info = make_quant_info()
    failures = 0
    for name, fn in (
        ("size is monotonic", test_size_is_monotonic),
        ("size matches the binding", lambda: test_size_matches_the_binding(quant_info)),
        ("outputs match per-call", lambda: test_outputs_match_per_call(quant_info)),
        ("capture guard", lambda: test_capture_guard(quant_info)),
    ):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{'OK' if failures == 0 else 'FAILED'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

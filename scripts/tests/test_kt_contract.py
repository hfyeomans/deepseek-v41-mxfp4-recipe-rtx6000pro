#!/usr/bin/env python3
"""Deterministic, GPU-free tests for the SGLang -> kt-kernel boundary contract (T3.16d).

Runs INSIDE the recipe image (needs torch); use scripts/tests/run-kt-contract-tests.sh,
which mounts the overlay's python/sglang/srt/layers/moe/kt_contract.py. CPU tensors only:
the checks are metadata-only (shape/dtype/stride/device), so what they decide on CPU is
what they decide under CUDA-graph capture.

What a wrong implementation would get wrong, and which test catches it:
  - accepting the qualified layout is not enough: every test below first proves the
    qualified TP2 / rank-0 / D=4 / 226-of-384 / top_k=6 configuration passes, then
    flips ONE value across the boundary and requires a KTContractError naming it.
  - off-by-one on the deferral bound: D == top_k protects zero experts and must be
    refused while D == top_k - 1 is legal -> test_deferral_bound.
  - checking rank 0 only: a wrapper on rank 1, or none on rank 0, both must fail
    -> test_wrapper_exactly_on_rank0.
  - dtype tolerance: kt's pinned input is bf16 and copy_ would cast fp16 silently
    -> test_hidden_must_be_bf16; float ids or integer weights -> test_topk_dtypes.
  - a transposed-then-viewed hidden tensor is the same shape but strided
    -> test_non_contiguous_refused.
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.moe.kt_contract import (
    KTContractError,
    check_moe_boundary,
    check_moe_topology,
)

HIDDEN = 64
TOP_K = 6
NUM_EXPERTS = 384
GPU_EXPERTS = 226

QUALIFIED_TOPOLOGY = dict(
    tp_rank=0,
    tp_size=2,
    moe_tp_rank=0,
    moe_tp_size=2,
    moe_ep_size=1,
    has_kt_wrapper=True,
    num_gpu_experts=GPU_EXPERTS,
    num_experts=NUM_EXPERTS,
    max_deferred=4,
    top_k=TOP_K,
    layer_idx=7,
)


def expect_error(fn, needle: str) -> None:
    try:
        fn()
    except KTContractError as e:
        assert needle in str(e), f"error did not mention {needle!r}: {e}"
        return
    raise AssertionError(f"expected KTContractError mentioning {needle!r}")


def topo(**overrides):
    kw = dict(QUALIFIED_TOPOLOGY)
    kw.update(overrides)
    return lambda: check_moe_topology(**kw)


def tensors(tokens: int = 4):
    x = torch.zeros((tokens, HIDDEN), dtype=torch.bfloat16)
    ids = torch.zeros((tokens, TOP_K), dtype=torch.int32)
    w = torch.zeros((tokens, TOP_K), dtype=torch.float32)
    return x, ids, w


def boundary(x, ids, w):
    return lambda: check_moe_boundary(x, ids, w, hidden_size=HIDDEN, top_k=TOP_K, layer_idx=7)


# ---- topology -------------------------------------------------------------


def test_qualified_topology_passes_on_both_ranks() -> None:
    topo()()
    topo(tp_rank=1, moe_tp_rank=1, has_kt_wrapper=False)()


def test_expert_parallel_refused() -> None:
    expect_error(topo(moe_ep_size=2), "moe_ep_size=2")


def test_moe_tp_group_must_equal_model_tp_group() -> None:
    expect_error(topo(moe_tp_size=1), "differs from the model")
    expect_error(topo(tp_rank=1, moe_tp_rank=0, has_kt_wrapper=False), "differs from the model")


def test_wrapper_exactly_on_rank0() -> None:
    expect_error(topo(has_kt_wrapper=False), "exactly rank 0")
    expect_error(topo(tp_rank=1, moe_tp_rank=1, has_kt_wrapper=True), "exactly rank 0")


def test_deferral_bound() -> None:
    topo(max_deferred=0)()
    topo(max_deferred=TOP_K - 1)()
    expect_error(topo(max_deferred=TOP_K), f"max_deferred_experts_per_token={TOP_K}")
    expect_error(topo(max_deferred=-1), "max_deferred_experts_per_token=-1")


def test_all_gpu_layer_must_not_build_hybrid() -> None:
    expect_error(topo(num_gpu_experts=NUM_EXPERTS), f"num_gpu_experts={NUM_EXPERTS}")


def test_b2_knobs_are_gone() -> None:
    # T3.16j: the B2 direct-UVA path was removed; the contract must not accept its knobs.
    import inspect

    from sglang.srt.layers.moe.kt_contract import check_moe_topology

    params = inspect.signature(check_moe_topology).parameters
    assert "b2_experts" not in params and "has_streamer" not in params, sorted(params)


# ---- boundary -------------------------------------------------------------


def test_qualified_tensors_pass_at_every_captured_width() -> None:
    for tokens in (1, 2, 3, 4, 13, 4096):
        boundary(*tensors(tokens))()


def test_hidden_size_and_rank_refused() -> None:
    x, ids, w = tensors()
    expect_error(boundary(torch.zeros((4, HIDDEN + 1), dtype=torch.bfloat16), ids, w), f"[tokens, {HIDDEN}]")
    expect_error(boundary(torch.zeros((1, 4, HIDDEN), dtype=torch.bfloat16), ids, w), f"[tokens, {HIDDEN}]")
    expect_error(boundary(torch.zeros((0, HIDDEN), dtype=torch.bfloat16), ids[:0], w[:0]), "0 tokens")


def test_topk_row_count_must_match_tokens() -> None:
    x, ids, w = tensors(4)
    expect_error(boundary(x, ids[:3], w), "topk_ids (3, 6) must be [4, 6]")
    expect_error(boundary(x, ids, w[:, :5]), "topk_weights (4, 5) must be [4, 6]")


def test_topk_dtypes() -> None:
    x, ids, w = tensors()
    boundary(x, ids.to(torch.int64), w)()
    boundary(x, ids, w.to(torch.bfloat16))()
    expect_error(boundary(x, ids.to(torch.float32), w), "topk_ids dtype torch.float32")
    expect_error(boundary(x, ids.to(torch.uint8), w), "topk_ids dtype torch.uint8")
    expect_error(boundary(x, ids, w.to(torch.int32)), "topk_weights dtype torch.int32")


def test_hidden_must_be_bf16() -> None:
    x, ids, w = tensors()
    expect_error(boundary(x.to(torch.float16), ids, w), "must be bfloat16")
    expect_error(boundary(x.to(torch.float32), ids, w), "must be bfloat16")


def test_non_contiguous_refused() -> None:
    x, ids, w = tensors(HIDDEN)  # square so the transpose keeps the shape
    strided = x.t()
    assert tuple(strided.shape) == (HIDDEN, HIDDEN) and not strided.is_contiguous()
    expect_error(boundary(strided, torch.zeros((HIDDEN, TOP_K), dtype=torch.int32), torch.zeros((HIDDEN, TOP_K))), "hidden_states is not contiguous")
    x, ids, w = tensors(4)
    wide = torch.zeros((4, 2 * TOP_K), dtype=torch.int32)[:, ::2]
    assert tuple(wide.shape) == (4, TOP_K) and not wide.is_contiguous()
    expect_error(boundary(x, wide, w), "topk_ids is not contiguous")


def test_device_mismatch_refused() -> None:
    x, ids, w = tensors()
    expect_error(boundary(x, ids.to("meta"), w), "topk_ids on meta")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

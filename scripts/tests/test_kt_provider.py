#!/usr/bin/env python3
"""GPU-free conformance tests for the CpuExpertProvider seam (recipe T3.19b).

Runs INSIDE the recipe image via scripts/tests/run-kt-provider-tests.sh: the overlay's
kt_contract.py / kt_provider.py / kt_runtime.py are imported through a stub `sglang` package, and
kt_kernel.experts_base (with the recipe patches applied) supplies the real Python
decision layer over the same C++ recorders test_kt_deferral_adjacency.py uses. No
CUDA is touched.

The suite is written once against the protocol and run against two providers:
  FakeProvider      a from-scratch implementation (no kt), the reference for what any
                    future CPU tier (another rank, a native coordinator) must satisfy.
  KtKernelProvider  the adapter the live server uses, over the recorder Wrapper.

What a wrong adapter would get wrong, and which test catches it:
  - accepting a shape kt keys its buffers on differently            -> test_validate_rejects_bad_shapes
  - reporting deferral the wrong way round (D=0 layer deferring,
    D>0 layer not)                                                   -> test_submission_reports_deferral
  - letting layer L+2 consume layer L's deferred slot, or a forward
    end with a slot outstanding                                      -> test_adjacency_is_enforced,
                                                                       test_forward_end_is_checked
  - waiting on another layer's handle                                 -> test_wait_rejects_foreign_submission
  - accepting a capture contract that shrinks or mismatches          -> test_reserve_contract
kt-only:
  - the adapter changing what kt sees (task order, slots, incremental,
    allow_pending) relative to the direct calls                      -> test_kt_calls_are_unchanged
  - the timed wait reordering kt's sync and H2D steps                 -> test_kt_timed_wait_order
  - a captured buffer set replaced underneath the graphs              -> test_kt_lease_detects_replaced_buffers
  - the two forward-end views (ledger vs kt flags) disagreeing        -> test_kt_forward_end_sees_kt_flags
"""
from __future__ import annotations

import sys
from typing import Callable, Dict, List, Optional

import torch

sys.path.insert(0, "/t")
import test_kt_deferral_adjacency as adj  # noqa: E402  (recorders + patched kt wrapper)

from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer  # noqa: E402
from sglang.srt.layers.moe import kt_provider as kp  # noqa: E402
from sglang.srt.layers.moe.kt_contract import KTContractError  # noqa: E402
from sglang.srt.layers.moe.kt_provider import (  # noqa: E402
    BufferLease,
    CaptureContract,
    ExpertBatchShape,
    ExpertSubmission,
    ForwardLedger,
    KtKernelProvider,
)
from sglang.srt.layers.moe.kt_runtime import KtRuntime  # noqa: E402

HIDDEN, TOP_K, D, STREAM = adj.HIDDEN, adj.TOP_K, adj.D, adj.STREAM
IDS, SCORES = adj.routing(
    [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]],
    [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]],
)


# --- reference provider (no kt) -----------------------------------------------------------

class FakeProvider:
    """Protocol reference: output = sum of weights per token broadcast over hidden (bf16)."""

    def __init__(self, layer_idx: int, hidden: int, top_k: int, max_deferred: int, ledger: ForwardLedger):
        self.layer_idx, self.hidden, self.top_k, self.max_deferred = layer_idx, hidden, top_k, max_deferred
        self.contract: Optional[CaptureContract] = None
        self._ledger = ledger
        self.pending_flags: Dict[int, bool] = {}
        self._out: Dict[int, torch.Tensor] = {}

    def reserve(self, contract: CaptureContract) -> None:
        if (contract.hidden, contract.top_k) != (self.hidden, self.top_k):
            raise KTContractError("contract/layer mismatch")
        if self.contract is not None and not set(self.contract.batch_sizes) <= set(contract.batch_sizes):
            raise KTContractError("contract shrank")
        self.contract = contract

    def validate(self, shape: ExpertBatchShape) -> None:
        if shape.tokens < 1 or shape.hidden != self.hidden or shape.top_k != self.top_k:
            raise KTContractError(f"bad shape {shape}")

    def submit(self, shape, hidden_states, compact_ids, weights, stream) -> ExpertSubmission:
        self.validate(shape)
        self._ledger.before_submit(self.layer_idx)
        deferred = self.max_deferred > 0 and bool((compact_ids >= 0).any())
        self.pending_flags[self.layer_idx] = deferred
        self._out[id(hidden_states)] = weights.sum(-1, keepdim=True).to(torch.bfloat16).expand(-1, self.hidden).contiguous()
        sub = ExpertSubmission(self.layer_idx, shape, self.layer_idx + 1 if deferred else None, int(stream), hidden_states)
        self._ledger.after_submit(sub)
        return sub

    def wait(self, submission, stream, mark=None) -> torch.Tensor:
        if submission.layer_idx != self.layer_idx:
            raise KTContractError("foreign submission")
        if mark is not None:
            mark("sync")
            mark("h2d")
        return self._out.pop(id(submission.hidden_states))

    def check_forward_end(self) -> None:
        self._ledger.check_forward_end(pending_flags=self.pending_flags, last_layer_idx=self.layer_idx)


# --- factories ----------------------------------------------------------------------------

def fake_factory():
    ledger = ForwardLedger()

    def make(layer_idx: int, d: int = D):
        return FakeProvider(layer_idx, HIDDEN, TOP_K, d, ledger)

    return make


def kt_factory():
    adj.reset()
    rt = KtRuntime(tp_rank=0)

    def make(layer_idx: int, d: int = D):
        return rt.new_provider(adj.make_layer(layer_idx, d=d, compact=True), layer_idx=layer_idx, hidden=HIDDEN, top_k=TOP_K, max_deferred=d)

    return make


def kt_provider(rt: KtRuntime, layer_idx: int, d: int = D) -> KtKernelProvider:
    return rt.new_provider(adj.make_layer(layer_idx, d=d, compact=True), layer_idx=layer_idx, hidden=HIDDEN, top_k=TOP_K, max_deferred=d)


FACTORIES: Dict[str, Callable] = {"fake": fake_factory, "kt": kt_factory}


def run(p, ids=IDS, scores=SCORES, mark=None):
    x = adj.hs(ids.shape[0])
    sub = p.submit(ExpertBatchShape.of(x, ids), x, ids, scores, STREAM)
    return sub, p.wait(sub, STREAM, mark=mark)


# --- conformance (both providers) --------------------------------------------------------

def conf_validate_rejects_bad_shapes(make) -> None:
    p = make(3)
    for bad in (ExpertBatchShape(0, HIDDEN, TOP_K), ExpertBatchShape(2, HIDDEN + 1, TOP_K), ExpertBatchShape(2, HIDDEN, TOP_K - 1)):
        try:
            p.validate(bad)
        except KTContractError:
            continue
        raise AssertionError(f"accepted {bad}")
    p.validate(ExpertBatchShape(2, HIDDEN, TOP_K))


def conf_submission_reports_deferral(make) -> None:
    a, b = make(6), make(7, d=0)
    sub, out = run(a)
    assert sub.layer_idx == 6 and sub.shape == ExpertBatchShape(2, HIDDEN, TOP_K) and sub.stream == STREAM
    assert sub.deferred_for_layer == 7, sub
    assert out.shape == (2, HIDDEN) and out.dtype == torch.bfloat16, (out.shape, out.dtype)
    sub2, _ = run(b)
    assert sub2.deferred_for_layer is None, "a D=0 layer must not report a deferred slot"


def conf_adjacency_is_enforced(make) -> None:
    a, b, c = make(6), make(7), make(8)
    run(a)
    try:
        run(c)
    except KTContractError:
        pass
    else:
        raise AssertionError("layer 8 consumed layer 6's deferred slot")
    run(b)  # the right consumer is still accepted after the refused submit
    run(c)


def conf_forward_end_is_checked(make) -> None:
    a, last = make(38), make(39, d=0)
    run(a)
    try:
        a.check_forward_end()
    except KTContractError:
        pass
    else:
        raise AssertionError("forward end accepted with layer 38's slot outstanding")
    run(last)
    last.check_forward_end()
    bad_last = make(39)  # D>0 on the rule layer
    run(bad_last)
    try:
        bad_last.check_forward_end()
    except KTContractError:
        pass
    else:
        raise AssertionError("forward end accepted with the last layer deferring")


def conf_wait_rejects_foreign_submission(make) -> None:
    a, b = make(10), make(11)
    sub, _ = run(a)
    run(b)
    try:
        b.wait(sub, STREAM)
    except KTContractError:
        return
    raise AssertionError("layer 11 waited on layer 10's handle")


def conf_reserve_contract(make) -> None:
    p = make(3)
    try:
        p.reserve(CaptureContract((1, 2, 4), HIDDEN + 1, TOP_K))
    except KTContractError:
        pass
    else:
        raise AssertionError("accepted a contract for another hidden size")
    p.reserve(CaptureContract((4, 1, 2), HIDDEN, TOP_K))
    assert p.contract.batch_sizes == (1, 2, 4)
    p.reserve(CaptureContract((1, 2, 4, 8), HIDDEN, TOP_K))  # accumulate: fine
    try:
        p.reserve(CaptureContract((1, 2), HIDDEN, TOP_K))
    except KTContractError:
        return
    raise AssertionError("accepted a shrinking contract")


# --- kt-specific --------------------------------------------------------------------------

def test_kt_calls_are_unchanged() -> None:
    """Direct kt calls and provider calls must leave identical recorder traces."""
    def trace(via_provider: bool):
        adj.reset()
        rt = KtRuntime(tp_rank=0)
        for L, d in ((6, D), (7, D), (8, 0)):
            w = adj.make_layer(L, d=d, compact=True)
            if via_provider:
                run(rt.new_provider(w, layer_idx=L, hidden=HIDDEN, top_k=TOP_K, max_deferred=d))
            else:
                adj.submit(w, IDS, SCORES)
        # buffer addresses differ between runs; compare which slot each task targets
        slots = {adj.slot_ptr(2, s): s for s in (0, 1)}
        return [(t.layer, t.kind, slots[t.out_ptr], t.incremental) for t in adj.INFER.queue], list(adj.INFER.syncs)

    direct, via = trace(False), trace(True)
    assert direct == via, f"the adapter changed what kt-kernel is asked to do:\n{direct}\n{via}"
    assert len(direct[0]) == 5 and len(direct[1]) == 3, direct


def test_kt_timed_wait_order() -> None:
    adj.reset()
    p = kt_provider(KtRuntime(tp_rank=0), 6)
    marks: List[str] = []
    _, out = run(p, mark=marks.append)
    assert marks == ["sync", "h2d"], marks
    assert adj.INFER.syncs == [("sync_stream", 1)], adj.INFER.syncs
    assert out.shape == (2, HIDDEN)


def test_kt_lease_detects_replaced_buffers() -> None:
    adj.reset()
    rt = KtRuntime(tp_rank=0)
    KExpertsCPUBuffer.capture_bs = [2]
    p = kt_provider(rt, 6)
    run(p)  # eager warmup promotes bs=2 into capture_buffers; no lease yet
    assert 2 not in rt.leases
    kp._capturing = lambda: True  # pretend a graph is capturing this submit
    try:
        run(kt_provider(rt, 7))
    finally:
        kp._capturing = lambda: False
    lease = rt.leases[2]
    assert isinstance(lease, BufferLease) and len(lease.ptrs) == 14
    run(kt_provider(rt, 8))  # same tuple: ok
    # Replace the captured tuple with a fresh one (what clear_buffer_cache + re-promotion does).
    BaseMoEWrapper.clear_buffer_cache()
    adj.INFER.queue.clear()
    try:
        run(kt_provider(rt, 9))
    except KTContractError as e:
        assert "not the one a CUDA graph captured" in str(e), e
    else:
        raise AssertionError("a replaced captured buffer set went unnoticed")
    KExpertsCPUBuffer.capture_bs = []


def test_kt_forward_end_sees_kt_flags() -> None:
    """The ledger can be clean while kt's own flag says the last layer deferred: still refused."""
    adj.reset()
    p = kt_provider(KtRuntime(tp_rank=0), 39, d=0)
    run(p)
    p.check_forward_end()
    BaseMoEWrapper._layer_has_pending_deferred[39] = True  # kt's view diverges
    try:
        p.check_forward_end()
    except KTContractError:
        return
    raise AssertionError("forward end trusted the ledger alone")


def test_kt_provider_validates_wrapper() -> None:
    adj.reset()
    rt = KtRuntime(tp_rank=0)
    w = adj.make_layer(5, compact=True)
    for kw in ({"layer_idx": 6}, {"hidden": HIDDEN + 1}, {"top_k": TOP_K + 1}, {"max_deferred": D - 1}):
        args = dict(layer_idx=5, hidden=HIDDEN, top_k=TOP_K, max_deferred=D)
        args.update(kw)
        try:
            rt.new_provider(w, **args)
        except KTContractError:
            continue
        raise AssertionError(f"provider accepted a wrapper mismatch: {kw}")


def main() -> int:
    failed = total = 0
    for name, factory in FACTORIES.items():
        for k, fn in sorted(globals().items()):
            if not k.startswith("conf_"):
                continue
            total += 1
            try:
                fn(factory())
                print(f"ok   {k}[{name}]")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {k}[{name}]: {type(e).__name__}: {e}")
    for k, fn in sorted(globals().items()):
        if not k.startswith("test_"):
            continue
        total += 1
        try:
            fn()
            print(f"ok   {k}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {k}: {type(e).__name__}: {e}")
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

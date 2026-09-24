#!/usr/bin/env python3
"""Deterministic, GPU-free tests for kt-kernel's expert-deferral adjacency rules (T3.16e).

Runs INSIDE the recipe image (it imports kt_kernel.experts_base with the recipe patches
applied) via scripts/tests/run-kt-deferral-tests.sh. No CUDA is touched: the C++ MoE and
CPUInfer are replaced by recorders, so what is under test is exactly the Python decision
layer -- which slot each task writes, with which `incremental` flag, in which order, and
what `sync(allow_pending)` is asked to wait for. The C++ contract those decisions rely on
was read from upstream kt-kernel (moe-tp.hpp merge_results, common.hpp should_skip_expert,
cpu_backend/task_queue.cpp) and is restated in each test's docstring so a change on
either side of the boundary fails a named test.

The deferral rule (max_deferred_experts_per_token = D of top_k = K):
  - per token, the K-D highest-scored routed slots are "protected"; an expert protected
    for ANY token in the batch is computed immediately for every token that routes to it;
    everything else is deferred (-1 in immediate_ids, the id in deferred_ids).
  - layer L's immediate task writes output slot L%2 with incremental=False (overwrite);
    its deferred task writes slot (L+1)%2 with incremental=False; layer L+1's immediate
    task then writes slot (L+1)%2 with incremental=True (accumulate), so L's deferred
    experts land in L+1's MoE output. The single FIFO TaskQueue makes that safe.
  - sync_forward(L) waits with allow_pending=1 iff L deferred something (the one task
    allowed to stay in flight is L's deferred task, which is behind L's immediate task).
  - upstream has no last-layer rule: layer 39's deferred task is computed and never
    consumed. The recipe overlay (kt_ep_wrapper.py) sets D=0 on layer num_layers-1;
    test_last_layer_deferral_has_no_consumer pins down why.

What a wrong implementation would get wrong, and which test catches it:
  - deferring exactly D per token (ignoring batch-wide protection)  -> test_protection_is_batch_wide
  - immediate/deferred not an exact partition of the routed ids     -> test_partition_is_exact
  - overlay compact selector diverging from upstream on plain ids   -> test_compact_selector_matches_upstream
  - deferred task aimed at the current slot, or next layer not
    accumulating (or accumulating when nothing was deferred)        -> test_deferred_lands_in_next_slot_and_next_layer_accumulates,
                                                                       test_no_deferral_means_next_layer_overwrites
  - sync waiting for zero pending (serialising the deferred task)
    or for more than one                                             -> test_sync_allow_pending_tracks_own_deferral
  - submission order that breaks the FIFO argument                   -> test_submission_order_is_immediate_then_deferred_per_layer
  - an all-protected batch treated as "no deferral" on one path but
    not the other (upstream inconsistency, documented)               -> test_all_protected_batch_still_submits_empty_deferred_task
"""
from __future__ import annotations

import ast
import os
import sys
from typing import Optional, Tuple

import torch

from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer

HIDDEN = 64
TOP_K = 6
NUM_EXPERTS = 384
D = 4  # the qualified deferral depth
STREAM = 0xC0FFEE  # opaque "cuda stream" handle; only recorded, never dereferenced

if not torch.cuda.is_available():
    def _unpinned(fn):
        def wrapped(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return fn(*args, **kwargs)

        return wrapped

    torch.zeros = _unpinned(torch.zeros)  # type: ignore[assignment]
    torch.full = _unpinned(torch.full)  # type: ignore[assignment]
    torch.empty = _unpinned(torch.empty)  # type: ignore[assignment]


# --- recorders standing in for the C++ side ---------------------------------------------

class Task:
    def __init__(self, layer: int, kind: str, ids_ptr: int, out_ptr: int, incremental: bool):
        self.layer, self.kind, self.ids_ptr, self.out_ptr, self.incremental = layer, kind, ids_ptr, out_ptr, incremental

    def __repr__(self) -> str:
        return f"Task(L{self.layer} {self.kind} out=0x{self.out_ptr:x} inc={self.incremental})"


class FakeMoE:
    def __init__(self, layer: int):
        self.layer = layer
        self.n = 0

    def forward_task(self, bsz_ptr, k, ids_ptr, w_ptr, in_ptr, out_ptr, incremental):
        # kt's Python submits immediate first, then (optionally) deferred; label by order.
        kind = "immediate" if self.n % 2 == 0 else "deferred"
        self.n += 1
        return Task(self.layer, kind, ids_ptr, out_ptr, bool(incremental))


class FakeCPUInfer:
    """One shared instance (kt shares one CPUInfer across layers); records the FIFO."""

    def __init__(self):
        self.queue: list[Task] = []
        self.syncs: list[tuple[str, int]] = []

    def submit(self, task):
        self.queue.append(task)

    def submit_with_cuda_stream(self, stream, task):
        assert stream == STREAM
        self.queue.append(task)

    def sync(self, allow_pending=0):
        self.syncs.append(("sync", int(allow_pending)))

    def sync_with_cuda_stream(self, stream, allow_pending=0):
        assert stream == STREAM
        self.syncs.append(("sync_stream", int(allow_pending)))


INFER = FakeCPUInfer()


class Wrapper(BaseMoEWrapper):
    def _get_cpu_infer(self, cpuinfer_threads, threadpool_count, numa_nodes=None):
        return INFER

    def load_weights(self, *a, **k):  # pragma: no cover - abstract stubs
        raise NotImplementedError

    def load_weights_from_tensors(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


def make_layer(layer_idx: int, d: int = D, compact: bool = False) -> Wrapper:
    w = Wrapper(
        layer_idx=layer_idx,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        hidden_size=HIDDEN,
        moe_intermediate_size=32,
        gpu_experts_mask=None,
        cpuinfer_threads=1,
        threadpool_count=1,
        weight_path="/nonexistent",
        chunked_prefill_size=64,
        max_deferred_experts_per_token=d,
    )
    w.moe = FakeMoE(layer_idx)
    if compact:
        import types

        w.select_deferred_experts = types.MethodType(compact_selector(), w)
    return w


_COMPACT = None


def compact_selector():
    """Load the overlay's _select_deferred_experts_compact without importing sglang."""
    global _COMPACT
    if _COMPACT is None:
        path = os.environ.get("KT_EP_WRAPPER_PY", "/overlay/kt_ep_wrapper.py")
        tree = ast.parse(open(path).read())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_select_deferred_experts_compact")
        ns = {"torch": torch, "Tuple": Tuple, "Optional": Optional}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
        _COMPACT = ns["_select_deferred_experts_compact"]
    return _COMPACT


def reset() -> None:
    BaseMoEWrapper.clear_buffer_cache()
    KExpertsCPUBuffer.capture_bs = []
    KExpertsCPUBuffer._retired = []
    BaseMoEWrapper._layer_has_pending_deferred.clear()
    INFER.queue.clear()
    INFER.syncs.clear()


def hs(bs: int) -> torch.Tensor:
    return torch.zeros((bs, HIDDEN), dtype=torch.bfloat16)


def routing(ids, scores):
    return torch.tensor(ids, dtype=torch.int64), torch.tensor(scores, dtype=torch.float32)


def slot_ptr(bs: int, slot: int) -> int:
    out = KExpertsCPUBuffer.get_buffer(hs(bs), TOP_K)[4]
    return out[slot].data_ptr()


def submit(w: Wrapper, ids, scores):
    x = hs(ids.shape[0])
    w.submit_forward(x, ids, scores, STREAM)
    w.sync_forward(x, STREAM)


def tasks(layer: int) -> list[Task]:
    return [t for t in INFER.queue if t.layer == layer]


# --- selection rule ----------------------------------------------------------------------

def test_partition_is_exact() -> None:
    """Every routed slot is in exactly one of immediate/deferred; the other side is -1."""
    reset()
    w = make_layer(3)
    ids, scores = routing(
        [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]],
        [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]],
    )
    imm, dfr = w.select_deferred_experts(ids, scores, TOP_K - D)
    assert dfr is not None
    assert torch.equal(torch.where(imm >= 0, imm, dfr), ids), (imm, dfr)
    assert not ((imm >= 0) & (dfr >= 0)).any(), "a slot is both immediate and deferred"
    assert not ((imm < 0) & (dfr < 0)).any(), "a routed slot was dropped from both"
    # disjoint experts across tokens -> exactly K-D immediate per token
    assert (imm >= 0).sum(dim=-1).tolist() == [TOP_K - D, TOP_K - D]
    assert set(imm[0].tolist()) - {-1} == {0, 1}
    assert set(imm[1].tolist()) - {-1} == {14, 15}


def test_protection_is_batch_wide() -> None:
    """Token B routes to expert 7 with its LOWEST score; token A protects 7 -> B's 7 is immediate too.

    A per-token-only rule would defer B's 7. The batch-wide rule exists because kt runs
    one CPU pass per expert: once 7 is loaded for A, B's row costs only the extra GEMV.
    """
    reset()
    w = make_layer(3)
    ids, scores = routing(
        [[7, 1, 2, 3, 4, 5], [7, 20, 21, 22, 23, 24]],
        [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4], [0.01, 0.9, 0.8, 0.7, 0.6, 0.5]],
    )
    imm, dfr = w.select_deferred_experts(ids, scores, TOP_K - D)
    assert imm[1, 0].item() == 7 and dfr[1, 0].item() == -1, (imm, dfr)
    assert (imm[1] >= 0).sum().item() == 3, "B keeps its own two protected experts plus the shared 7"


def test_compact_selector_matches_upstream() -> None:
    """The overlay's -1-tolerant selector equals upstream on plain ids and masks -1 through.

    Upstream scatter_'s a [num_experts] flag by expert id, so a -1 (GPU-resident slot in
    the compact CPU view) is out of bounds; the overlay routes -1 to a spare flag slot.
    """
    reset()
    up = make_layer(5)
    cp = make_layer(5, compact=True)
    torch.manual_seed(1)
    ids = torch.randint(0, NUM_EXPERTS, (4, TOP_K), dtype=torch.int64)
    scores = torch.rand(4, TOP_K)
    a = up.select_deferred_experts(ids, scores, TOP_K - D)
    b = cp.select_deferred_experts(ids, scores, TOP_K - D)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "compact selector diverged from upstream"
    ids_gpu = ids.clone()
    ids_gpu[0, 0] = -1  # this token's top expert lives on the GPU; the CPU view sees -1
    scores_gpu = scores.clone()
    scores_gpu[0, 0] = 1.0
    imm, dfr = cp.select_deferred_experts(ids_gpu, scores_gpu, TOP_K - D)
    assert imm[0, 0].item() == -1 and dfr[0, 0].item() == -1, "a GPU-resident slot must be -1 on both outputs"
    assert torch.equal(torch.where(imm >= 0, imm, dfr)[1:], ids_gpu[1:])
    assert (imm[0] >= 0).sum().item() >= TOP_K - D - 1, "-1 consumed a protected slot; the rest must still be protected"
    try:
        up.select_deferred_experts(ids_gpu, scores_gpu, TOP_K - D)
    except (RuntimeError, IndexError):
        pass  # upstream rejects -1: this is why the overlay binds the compact version
    else:
        raise AssertionError("upstream select_deferred_experts accepted -1; the compact override may be redundant now")


# --- adjacency across layers -------------------------------------------------------------

def test_deferred_lands_in_next_slot_and_next_layer_accumulates() -> None:
    """L: immediate -> slot L%2 overwrite, deferred -> slot (L+1)%2 overwrite; L+1: immediate -> same slot accumulate."""
    reset()
    bs = 2
    L = 6
    a, b = make_layer(L), make_layer(L + 1)
    ids, scores = routing(
        [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]],
        [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]],
    )
    submit(a, ids, scores)
    assert BaseMoEWrapper._layer_has_pending_deferred[L] is True
    ta = tasks(L)
    assert [t.kind for t in ta] == ["immediate", "deferred"], ta
    assert ta[0].out_ptr == slot_ptr(bs, L % 2) and ta[0].incremental is False
    assert ta[1].out_ptr == slot_ptr(bs, (L + 1) % 2) and ta[1].incremental is False
    assert ta[0].out_ptr != ta[1].out_ptr

    submit(b, ids, scores)
    tb = tasks(L + 1)
    assert tb[0].kind == "immediate" and tb[0].incremental is True, "L+1 must accumulate onto L's deferred output"
    assert tb[0].out_ptr == ta[1].out_ptr, "L+1's immediate output is not the slot L's deferred task wrote"
    assert tb[1].out_ptr == slot_ptr(bs, L % 2), "L+1's own deferral goes to L+2's slot (== L's, ring of 2)"


def test_no_deferral_means_next_layer_overwrites() -> None:
    """If L deferred nothing (D=0), L+1's immediate task must NOT accumulate stale slot contents."""
    reset()
    a, b = make_layer(6, d=0), make_layer(7)
    ids, scores = routing([[0, 1, 2, 3, 4, 5]], [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4]])
    submit(a, ids, scores)
    assert tasks(6) and all(t.kind == "immediate" for t in tasks(6)), "D=0 must submit no deferred task"
    assert BaseMoEWrapper._layer_has_pending_deferred[6] is False
    submit(b, ids, scores)
    assert tasks(7)[0].incremental is False


def test_sync_allow_pending_tracks_own_deferral() -> None:
    """sync_forward(L) may leave exactly one task (L's deferred) in flight, and only if L deferred."""
    reset()
    ids, scores = routing([[0, 1, 2, 3, 4, 5]], [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4]])
    submit(make_layer(6, d=0), ids, scores)
    submit(make_layer(7), ids, scores)
    submit(make_layer(8), ids, scores)
    assert INFER.syncs == [("sync_stream", 0), ("sync_stream", 1), ("sync_stream", 1)], INFER.syncs


def test_submission_order_is_immediate_then_deferred_per_layer() -> None:
    """The FIFO argument needs: L.imm, L.def, (L+1).imm, (L+1).def ... on ONE queue."""
    reset()
    ids, scores = routing([[0, 1, 2, 3, 4, 5]], [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4]])
    for L in (10, 11, 12):
        submit(make_layer(L), ids, scores)
    got = [(t.layer, t.kind) for t in INFER.queue]
    want = [(L, k) for L in (10, 11, 12) for k in ("immediate", "deferred")]
    assert got == want, got
    # and every layer's wrappers share the one CPUInfer (kt caches it class-wide)
    assert all(make_layer(L).cpu_infer is INFER for L in (10, 11))


def test_all_protected_batch_still_submits_empty_deferred_task() -> None:
    """Three tokens route to the same 6 experts with rotated scores -> every expert is protected.

    submit_forward() checks `deferred_ids is not None` (not `.any()`), so it still submits
    a deferred task whose ids are all -1. Per the C++ contract (should_skip_expert skips
    -1; merge_results with incremental=False stores the zero accumulator) that task
    ZEROES the next slot, and L+1 then accumulates onto zeros -- correct, one wasted
    enqueue. forward_on_pinned_buffers() (the NPU path) checks `.any()` and skips it.
    If upstream ever unifies the two, this test tells us which way they went.
    """
    reset()
    ids, scores = routing(
        [[0, 1, 2, 3, 4, 5]] * 3,
        [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.9, 0.8, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4, 0.9, 0.8]],
    )
    w = make_layer(20)
    imm, dfr = w.select_deferred_experts(ids, scores, TOP_K - D)
    assert (imm >= 0).all() and (dfr < 0).all(), "expected an all-protected batch"
    submit(w, ids, scores)
    t = tasks(20)
    assert [x.kind for x in t] == ["immediate", "deferred"], "submit_forward still enqueues the empty deferred task"
    assert BaseMoEWrapper._layer_has_pending_deferred[20] is True
    submit(make_layer(21), ids, scores)
    assert tasks(21)[0].incremental is True, "L+1 accumulates onto the zeroed slot (harmless)"


def test_last_layer_deferral_has_no_consumer() -> None:
    """Upstream: layer 39 (of 40) with D>0 submits a deferred task nobody reads; next step's
    layer 0 overwrites (incremental=False, it looks up layer -1). The overlay therefore
    forces D=0 on layer num_layers-1: no wasted task, and layer 39's 4 low-scored experts
    are applied instead of dropped. Both halves pinned here."""
    reset()
    ids, scores = routing([[0, 1, 2, 3, 4, 5]], [[0.9, 0.8, 0.1, 0.2, 0.3, 0.4]])
    # upstream default: D on every layer
    submit(make_layer(39), ids, scores)
    last = tasks(39)
    assert [t.kind for t in last] == ["immediate", "deferred"]
    submit(make_layer(0), ids, scores)  # next decode step
    first = tasks(0)
    assert first[0].incremental is False, "layer 0 must not accumulate the previous step's layer-39 leftovers"
    assert first[0].out_ptr == last[1].out_ptr, "...even though it writes the very slot layer 39's deferred task filled"
    # the consumer that would have read it does not exist:
    assert not any(t.incremental for t in INFER.queue if t.layer != 39 and t.out_ptr == last[1].out_ptr)
    # overlay policy: D=0 on the last layer
    reset()
    submit(make_layer(39, d=0), ids, scores)
    assert [t.kind for t in tasks(39)] == ["immediate"]
    assert INFER.syncs == [("sync_stream", 0)], "the last layer must wait for all of its work"


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

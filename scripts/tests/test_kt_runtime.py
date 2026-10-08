#!/usr/bin/env python3
"""GPU-free tests for the runner-owned CPU-tier state (recipe T3.19c).

Runs INSIDE the recipe image via scripts/tests/run-kt-runtime-tests.sh with the
overlay's python/sglang mounted over the installed package, so kt_runtime.py,
kt_loader.py and kt-kernel's own NativeMoEWrapper / MXFP4SafeTensorLoader are the
real ones. No CUDA is touched.

What a wrong implementation would get wrong, and which test catches it:
  - two ModelRunners in one process sharing providers, leases, the deferral
    ledger, the prefill coordinator or the timeline         -> test_two_runtimes_share_nothing
  - a capture contract reaching another runtime's providers  -> test_reserve_capture_sizes_is_per_runtime
  - the load_model handle leaking past load_model, or two
    runners loading at once                                  -> test_activate_scope
  - the compact loader loading the wrong experts, the wrong
    order, or all of them                                    -> test_compact_loader_loads_cpu_complement
  - kt loading through a plain loader (every expert under
    shifted ids) and nobody noticing                         -> test_load_cpu_experts_requires_kt_to_use_the_loader
"""
from __future__ import annotations

import os
import sys
import tempfile
import types

import torch
from safetensors.torch import save_file

from kt_kernel.utils.amx import NativeMoEWrapper  # noqa: E402
from kt_kernel.utils.loader import MXFP4SafeTensorLoader  # noqa: E402
from sglang.srt.layers.moe import kt_runtime  # noqa: E402
from sglang.srt.layers.moe.kt_contract import KTContractError  # noqa: E402
from sglang.srt.layers.moe.kt_loader import CompactMXFP4Loader  # noqa: E402
from sglang.srt.layers.moe.kt_provider import ExpertSubmission, ExpertBatchShape  # noqa: E402
from sglang.srt.layers.moe.kt_runtime import KtRuntime  # noqa: E402

HIDDEN, TOP_K, D = 4096, 6, 4
N_EXPERTS, N, K = 5, 64, 128  # 5 experts, [N, K/2] nibble-packed weights, [N, K/32] scales


class FakeKtWrapper:
    """Only what KtKernelProvider validates at construction."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.hidden_size = HIDDEN
        self.num_experts_per_tok = TOP_K
        self.max_deferred_experts_per_token = D


def write_fake_checkpoint(path: str) -> dict:
    """Five MXFP4 experts under the V4 key layout (no ``model.`` prefix)."""
    tensors = {}
    for e in range(N_EXPERTS):
        for proj, (rows, cols) in (("w1", (N, K // 2)), ("w3", (N, K // 2)), ("w2", (K, N // 2))):
            w = torch.full((rows, cols), 16 * e + 1, dtype=torch.uint8)  # expert id baked into the bytes
            s = torch.full((rows, cols * 2 // 32), 120 + e, dtype=torch.uint8)  # ue8m0 exponent 120+e
            tensors[f"layers.0.ffn.experts.{e}.{proj}.weight"] = w
            tensors[f"layers.0.ffn.experts.{e}.{proj}.scale"] = s
    save_file(tensors, os.path.join(path, "model-00001-of-00001.safetensors"))
    return tensors


def test_two_runtimes_share_nothing() -> None:
    a, b = KtRuntime(tp_rank=0), KtRuntime(tp_rank=0)
    pa = a.new_provider(FakeKtWrapper(3), layer_idx=3, hidden=HIDDEN, top_k=TOP_K, max_deferred=D)
    assert a.providers == [pa] and b.providers == []
    assert a.ledger is not b.ledger and a.leases is not b.leases
    # Deferral state recorded in one ledger is invisible to the other.
    a.ledger.after_submit(ExpertSubmission(3, ExpertBatchShape(1, HIDDEN, TOP_K), 4, 0, torch.zeros(1)))
    assert a.ledger.outstanding_for_layer == 4 and b.ledger.outstanding_for_layer is None
    a.leases[2] = object()
    assert 2 not in b.leases
    a.prefill_coordinator = object()
    assert b.prefill_coordinator is None
    assert a.timeline is None and b.timeline is None
    a._side_streams[0] = object()
    assert b._side_streams == {}


def test_reserve_capture_sizes_is_per_runtime() -> None:
    a, b = KtRuntime(tp_rank=0), KtRuntime(tp_rank=0)
    pa = a.new_provider(FakeKtWrapper(3), layer_idx=3, hidden=HIDDEN, top_k=TOP_K, max_deferred=D)
    pb = b.new_provider(FakeKtWrapper(3), layer_idx=3, hidden=HIDDEN, top_k=TOP_K, max_deferred=D)
    a.reserve_capture_sizes([2, 4])
    assert pa.contract is not None and set(pa.contract.batch_sizes) >= {2, 4}, pa.contract
    assert pb.contract is None, "a contract reached another runtime's provider"


def test_activate_scope() -> None:
    rt = KtRuntime(tp_rank=1)
    assert kt_runtime.current() is None
    with kt_runtime.activate(rt):
        assert kt_runtime.current() is rt
        try:
            with kt_runtime.activate(KtRuntime(tp_rank=0)):
                pass
        except KTContractError:
            pass
        else:
            raise AssertionError("nested activate() accepted")
        assert kt_runtime.current() is rt, "the failed nested activate clobbered the active runtime"
    assert kt_runtime.current() is None
    try:
        with kt_runtime.activate(rt):
            raise RuntimeError("load_model failed")
    except RuntimeError:
        pass
    assert kt_runtime.current() is None, "handle leaked past a failed load_model"


def test_compact_loader_loads_cpu_complement() -> None:
    with tempfile.TemporaryDirectory() as d:
        write_fake_checkpoint(d)
        mask = torch.tensor([True, True, False, True, False])
        loader = CompactMXFP4Loader(d, mask)
        out = loader.load_experts("model.layers.0")  # kt's first candidate; resolved through the stripped prefix
        assert sorted(out) == ["down", "down_scale", "gate", "gate_scale", "up", "up_scale"]
        assert loader.loaded_cpu_experts == 2 and [len(out[k]) for k in sorted(out)] == [2] * 6
        # compact id 0 is original expert 2, compact id 1 is original expert 4: the bytes say which
        assert [int(w[0, 0]) for w in out["gate"]] == [16 * 2 + 1, 16 * 4 + 1]
        assert [int(w[0, 0]) for w in out["down"]] == [16 * 2 + 1, 16 * 4 + 1]
        assert all(w.dtype == torch.uint8 and w.is_contiguous() for k in ("gate", "up", "down") for w in out[k])
        # ue8m0 exponent e -> bf16 2^(e-127): expert 2 -> 2^-5, expert 4 -> 2^-3
        assert [float(s[0, 0]) for s in out["gate_scale"]] == [2.0 ** -5, 2.0 ** -3]
        assert all(s.dtype == torch.bfloat16 for s in out["up_scale"])
        loader.close_all_handles()

        # mask length must match the checkpoint
        try:
            CompactMXFP4Loader(d, torch.tensor([True, False])).load_experts("model.layers.0")
        except ValueError as e:
            assert "gpu_experts_mask has 2 entries" in str(e), e
        else:
            raise AssertionError("a mask of the wrong length was accepted")

        # an all-CPU mask is kt's own full load, ids unchanged
        full = CompactMXFP4Loader(d, torch.zeros(N_EXPERTS, dtype=torch.bool))
        out = full.load_experts("model.layers.0")
        assert full.loaded_cpu_experts == N_EXPERTS and [int(w[0, 0]) for w in out["gate"]] == [16 * e + 1 for e in range(N_EXPERTS)]
        full.close_all_handles()


def _kt_like_wrapper(path: str, *, load_weights) -> NativeMoEWrapper:
    """A NativeMoEWrapper without its C++ side: method/weight_path/layer_idx plus a scripted load_weights."""
    w = NativeMoEWrapper.__new__(NativeMoEWrapper)
    w.method, w.weight_path, w.layer_idx = "MXFP4", path, 0
    w.loader = None
    w.load_weights = types.MethodType(load_weights, w)
    return w


def test_load_cpu_experts_requires_kt_to_use_the_loader() -> None:
    def kt_protocol(self, ids):
        # what kt's NativeMoEWrapper.load_weights does with the slot (amx.py)
        self.loader = NativeMoEWrapper._native_loader_instance
        self.got = self.loader.load_experts("model.layers.0")
        NativeMoEWrapper._release_loader(layer_idx=self.layer_idx)

    def own_loader(self, ids):
        # a kt that ignores the slot: every expert, shifted ids
        self.loader = MXFP4SafeTensorLoader(self.weight_path)
        self.got = self.loader.load_experts("model.layers.0")
        self.loader.close_all_handles()

    mask = torch.tensor([True, True, False, True, False])
    with tempfile.TemporaryDirectory() as d:
        write_fake_checkpoint(d)
        rt = KtRuntime(tp_rank=0)
        w = _kt_like_wrapper(d, load_weights=kt_protocol)
        rt.load_cpu_experts(w, torch.arange(2), mask)
        assert isinstance(w.loader, CompactMXFP4Loader) and len(w.got["gate"]) == 2
        assert NativeMoEWrapper._native_loader_instance is None, "slot not released after the layer"

        w = _kt_like_wrapper(d, load_weights=own_loader)
        try:
            rt.load_cpu_experts(w, torch.arange(2), mask)
        except KTContractError as e:
            assert "not the compact loader" in str(e), e
        else:
            raise AssertionError("kt loading through a plain loader went unnoticed")
        assert NativeMoEWrapper._native_loader_instance is None

        # kt's first wrapper __init__ leaves a plain loader in the slot: replaced (and closed), not used
        plain = MXFP4SafeTensorLoader(d)
        NativeMoEWrapper._native_loader_instance = plain
        w = _kt_like_wrapper(d, load_weights=kt_protocol)
        rt.load_cpu_experts(w, torch.arange(2), mask)
        assert isinstance(w.loader, CompactMXFP4Loader) and len(w.got["gate"]) == 2
        assert plain.file_handle_map == {} and NativeMoEWrapper._native_loader_instance is None

        # a compact loader left in the slot means kt stopped releasing per layer: refused up front
        for stale in (CompactMXFP4Loader(d, mask), object()):
            NativeMoEWrapper._native_loader_instance = stale
            try:
                rt.load_cpu_experts(_kt_like_wrapper(d, load_weights=kt_protocol), torch.arange(2), mask)
            except KTContractError as e:
                assert "still holds a" in str(e), e
            else:
                raise AssertionError("an occupied loader slot was overwritten")
            finally:
                NativeMoEWrapper._native_loader_instance = None

        # compact id count must match what the loader produced
        try:
            rt.load_cpu_experts(_kt_like_wrapper(d, load_weights=kt_protocol), torch.arange(3), mask)
        except KTContractError as e:
            assert "produced 2 experts for 3 compact ids" in str(e), e
        else:
            raise AssertionError("a compact id / expert count mismatch was accepted")

        # only kt's native MXFP4 wrapper is supported
        w = _kt_like_wrapper(d, load_weights=kt_protocol)
        w.method = "FP8"
        try:
            rt.load_cpu_experts(w, torch.arange(2), mask)
        except KTContractError as e:
            assert "method MXFP4" in str(e), e
        else:
            raise AssertionError("a non-MXFP4 wrapper was accepted")


def main() -> int:
    failed = total = 0
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

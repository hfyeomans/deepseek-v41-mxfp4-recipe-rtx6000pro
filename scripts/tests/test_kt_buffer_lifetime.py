#!/usr/bin/env python3
"""Deterministic, GPU-free tests for kt-kernel's KExpertsCPUBuffer lifetime rules.

Runs INSIDE the recipe image (it imports kt_kernel); use scripts/tests/run-kt-buffer-tests.sh,
which mounts the patched experts_base.py over the installed one. No CUDA is touched:
hidden_states live on the CPU and torch.cuda.is_current_stream_capturing is replaced by a
flag, so the test exercises exactly the Python decisions the patches make
(config/kt-kernel-patches/0001-*.patch fixes bug C; 0002-*.patch adds the assertions).

What a wrong implementation would get wrong, and which test catches it:
  - stock kt (no patches): a captured graph bakes the single temp tuple and the next
    different-size eager call frees it -> test_promoted_temp_is_never_retired.
  - assertions missing: capturing an unregistered size, or allocating during capture,
    silently succeeds -> test_capture_unregistered_size_fails, test_capture_unwarmed_size_fails.
  - set_capture_batch_sizes replacing instead of merging: the second graph runner
    deregisters the first one's sizes -> test_registration_merges.
"""
from __future__ import annotations

import gc
import sys
import weakref

import torch

from kt_kernel.experts_base import BaseMoEWrapper, KExpertsCPUBuffer

HIDDEN = 64
TOPK = 8
_capturing = {"on": False}
torch.cuda.is_current_stream_capturing = lambda: _capturing["on"]  # type: ignore[assignment]

if not torch.cuda.is_available():
    # get_buffer pins its host tensors, which needs the CUDA driver. The lifetime rules
    # under test do not depend on pinning, so without a GPU drop the flag and keep the
    # test runnable in a driverless container.
    def _unpinned(fn):
        def wrapped(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return fn(*args, **kwargs)

        return wrapped

    torch.zeros = _unpinned(torch.zeros)  # type: ignore[assignment]
    torch.full = _unpinned(torch.full)  # type: ignore[assignment]


def reset() -> None:
    BaseMoEWrapper.clear_buffer_cache()
    KExpertsCPUBuffer.capture_bs = []
    KExpertsCPUBuffer._retired = []
    _capturing["on"] = False


def hs(bs: int) -> torch.Tensor:
    return torch.zeros((bs, HIDDEN), dtype=torch.bfloat16)


def get(bs: int):
    return KExpertsCPUBuffer.get_buffer(hs(bs), TOPK)


def expect_assert(fn, needle: str) -> None:
    try:
        fn()
    except AssertionError as e:
        assert needle in str(e), f"assertion fired but with the wrong message: {e}"
        return
    raise AssertionError(f"expected an AssertionError containing {needle!r}")


def test_registration_merges() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1, 2, 3, 4])
    BaseMoEWrapper.set_capture_batch_sizes([6, 12, 18])  # a second runner (verify)
    assert BaseMoEWrapper.get_capture_batch_sizes() == [1, 2, 3, 4, 6, 12, 18]


def test_registered_sizes_are_permanent_and_identical() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4])
    a = get(2)  # eager warmup
    assert KExpertsCPUBuffer.capture_buffers[2] is a
    get(1)
    get(4)
    get(3)  # unregistered temp, replaces the temp slot
    get(1)
    assert get(2) is a, "a registered tuple must keep its identity across other sizes"
    _capturing["on"] = True
    assert get(2) is a, "capture must reuse the warmed tuple, not allocate"
    assert len(KExpertsCPUBuffer.capture_buffers) == 3


def test_capture_unregistered_size_fails() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4])
    get(3)  # temp
    _capturing["on"] = True
    expect_assert(lambda: get(3), "unregistered kt buffer size bs=3")
    expect_assert(lambda: get(5), "unregistered kt buffer size bs=5")


def test_capture_unwarmed_size_fails() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1, 2, 4])
    get(1)
    _capturing["on"] = True
    expect_assert(lambda: get(4), "temporary allocation during CUDA graph capture: bs=4")


def test_promoted_temp_is_never_retired() -> None:
    """Bug C shape: a size requested BEFORE registration (autotune dummy forward) lives in
    the temp slot; capture then hits it; a later different-size eager call must not free it."""
    reset()
    t = get(24)  # pre-registration request -> temp
    assert 24 not in KExpertsCPUBuffer.capture_buffers
    BaseMoEWrapper.set_capture_batch_sizes([6, 12, 18, 24])
    _capturing["on"] = True
    assert get(24) is t, "capture must get the exact temp tuple (promotion), not a copy"
    assert KExpertsCPUBuffer.capture_buffers[24] is t
    _capturing["on"] = False
    ref = weakref.ref(t[0][0])
    del t
    get(5)  # different-size eager call replaces the temp slot
    get(7)
    gc.collect()
    assert 24 in KExpertsCPUBuffer.capture_buffers
    assert ref() is not None, "the promoted tuple was freed by a later eager call"
    assert all(bs != 24 for _ev, bs, _buf in KExpertsCPUBuffer._retired)


def test_unregistered_temps_are_retired_not_freed_on_the_spot() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1])
    first = get(5)
    get(7)
    retired = [bs for _ev, bs, _buf in KExpertsCPUBuffer._retired]
    assert retired == [5], retired
    assert KExpertsCPUBuffer._retired[0][2] is first
    # CPU-only path records no event, so entries are kept for the process lifetime and
    # capped at _retired_max by waiting, never by dropping a pending one.
    for bs in range(8, 8 + KExpertsCPUBuffer._retired_max + 4):
        get(bs)
    assert len(KExpertsCPUBuffer._retired) <= KExpertsCPUBuffer._retired_max


def test_aliased_tuple_under_wrong_size_is_refused() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([1])
    t = get(3)  # temp
    KExpertsCPUBuffer.capture_buffers[9] = t  # corrupt: same tuple registered under another size
    expect_assert(lambda: get(4), "owned by the capture cache under sizes [9]")


def test_cached_shape_mismatch_is_refused() -> None:
    reset()
    BaseMoEWrapper.set_capture_batch_sizes([2])
    get(2)
    bad = torch.zeros((2, HIDDEN + 1), dtype=torch.bfloat16)
    expect_assert(lambda: KExpertsCPUBuffer.get_buffer(bad, TOPK), "cached tuple for bs=2 has shape")


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

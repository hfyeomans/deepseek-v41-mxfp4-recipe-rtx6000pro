#!/usr/bin/env python3
"""Deterministic, GPU-free tests for the SM120 page-split buffer seal (review C5).

Runs INSIDE the recipe image with the overlay's python/sglang mounted over the installed
package (scripts/tests/run-sm120-split-seal-tests.sh). CPU tensors only; the buffer
registry (`get_resources().buffers`) and `torch.cuda.is_current_stream_capturing` are
replaced so no CUDA context is needed. The decisions under test are pure dict/shape logic,
so what they decide here is what they decide on the GPU.

Background: E23 R1 bug B was a page-split buffer grown after CUDA-graph capture; the old
tensor was freed and every replay wrote a dangling pointer (Xid 31). `838490a6` made a
grow retire-not-free and pre-sized from the pool geometry; C5 asked that a grow after
reservation be *rejected*, because a grow at that point means the geometry was wrong.

What a wrong implementation would get wrong, and which test catches it:
  - sealing everything on first allocation (autotune's small buffer would then block the
    real reservation) -> test_grow_before_reservation_is_allowed_and_retires.
  - sealing only the mask, or only the split buffers -> test_reserved_mask_cannot_grow,
    test_reserved_split_cannot_grow.
  - sealing by device-agnostic key so an "extra"-tag buffer seals the ""-tag one
    -> test_seal_is_per_key.
  - a request within the reserved size must not raise (the whole point of reserving)
    -> test_request_within_reservation_is_a_no_op.
  - raising on the *first* allocation after seal for a key that was never reserved
    (a pool the caller did not list) would kill launches that worked yesterday; that key
    keeps the old grow-and-log behaviour -> test_unreserved_key_still_grows.
"""
from __future__ import annotations

import sys
import types

import torch

torch.cuda.is_current_stream_capturing = lambda: False  # no CUDA context in the test image

import sglang.srt.runtime_context as rc  # noqa: E402
from sglang.kernels.ops.attention import flash_mla_sm120 as m  # noqa: E402

DEV = torch.device("cpu")
SRC_PBS = 256  # a compressed-pool page size that needs re-paging to 64 (ratio 4)


def fresh_registry() -> dict:
    buffers: dict = {}
    rc.get_resources = lambda: types.SimpleNamespace(buffers=buffers)
    return buffers


def test_grow_before_reservation_is_allowed_and_retires():
    b = fresh_registry()
    small = m._ensure_mask(DEV, 8)
    big = m._ensure_mask(DEV, 16)
    assert big is not small and big.shape[0] == 16
    assert b[m._RETIRED_KEY] == [small], "old buffer must be retired, not dropped"


def test_reserved_mask_cannot_grow():
    fresh_registry()
    m.reserve_split_buffers([(100, SRC_PBS, ""), (300, SRC_PBS, "extra")], DEV)
    assert m._ensure_mask(DEV, 300).shape[0] == 300
    try:
        m._ensure_mask(DEV, 301)
    except m.SM120SplitBufferSealedError as e:
        assert "sealed" in str(e)
    else:
        raise AssertionError("mask grow after reservation must raise")


def test_reserved_split_cannot_grow():
    fresh_registry()
    m.reserve_split_buffers([(100, SRC_PBS, ""), (300, SRC_PBS, "extra")], DEV)
    # "" was reserved at 100 * 4 = 400 dst pages, "extra" at 1200.
    assert m._ensure_split_buffer(DEV, "", 400).shape[0] == 400
    assert m._ensure_split_buffer(DEV, "extra", 1200).shape[0] == 1200
    for tag, pages in (("", 401), ("extra", 1201)):
        try:
            m._ensure_split_buffer(DEV, tag, pages)
        except m.SM120SplitBufferSealedError:
            pass
        else:
            raise AssertionError(f"split buffer {tag!r} grow after reservation must raise")


def test_request_within_reservation_is_a_no_op():
    b = fresh_registry()
    m.reserve_split_buffers([(100, SRC_PBS, "")], DEV)
    before = m._ensure_split_buffer(DEV, "", 400)
    again = m._ensure_split_buffer(DEV, "", 37)
    assert again is before
    assert m._RETIRED_KEY not in b


def test_seal_is_per_key():
    fresh_registry()
    # Only the "extra" pool is listed; the "" split buffer is not reserved and not sealed.
    m.reserve_split_buffers([(300, SRC_PBS, "extra")], DEV)
    first = m._ensure_split_buffer(DEV, "", 10)
    grown = m._ensure_split_buffer(DEV, "", 20)  # unsealed key: grow-and-retire as before
    assert grown is not first and grown.shape[0] == 20


def test_unreserved_key_still_grows():
    fresh_registry()
    m.reserve_split_buffers([(64, m._PBS_DST, "")], DEV)  # page size already 64: nothing reserved
    a = m._ensure_mask(DEV, 5)
    c = m._ensure_mask(DEV, 6)
    assert c is not a and c.shape[0] == 6


def test_pools_already_at_64_seal_nothing():
    b = fresh_registry()
    m.reserve_split_buffers([(64, m._PBS_DST, ""), (10, 2, "extra")], DEV)
    assert not any(k.startswith(m._SEALED_KEY_PREFIX) for k in b), sorted(b)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

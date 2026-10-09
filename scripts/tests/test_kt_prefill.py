#!/usr/bin/env python3
"""Deterministic, GPU-free tests for the streamed-prefill protocol (recipe T3.19d).

Runs INSIDE the recipe image via scripts/tests/run-kt-prefill-tests.sh, which mounts
the overlay's kt_prefill.py (and kt_contract.py) through a stub package. The object
under test, PrefillPipeline, is the protocol LayerwisePrefillCoordinator used to keep
implicitly in `_buf` / `_pending` / `_expect_layer` / two CUDA events: pure Python,
no torch, so every legal transition can be walked and every illegal one provoked.

What a wrong implementation would get wrong, and which test catches it:
  - the double buffer not alternating, or layer L+1's export going into the buffer
    layer L is still DMAing from                              -> test_legal_cycle_two_forwards
  - overwriting a FILLED/EXPORTING buffer (unread export), or claiming a READING
    buffer without waiting for its DMA                         -> test_claim_refuses_unread_or_in_flight
  - accepting layers out of order / a skipped layer / a layer from
    outside the streaming set                                  -> test_layer_order
  - the token count changing mid-forward (the chunk-size invariant
    the old code only documented)                              -> test_token_count_fixed_per_forward
  - a new forward opened on top of an unfinished one           -> test_begin_forward_requires_closed
  - buffer transitions in the wrong phase (export_ready on a FREE
    buffer, dma_issued before export_ready, dma_complete twice)  -> test_buffer_phase_order
  - the forward not closing at the last layer / closing early -> test_forward_closes_at_last_layer
  - a single-layer streaming set, and an odd layer count (buffer 0 reused across
    the forward boundary with a DMA wait)                       -> test_single_layer_and_odd_count
  - malformed construction accepted                            -> test_construction
  - (E34d-v (ii)) the in-place bf16-bits -> E8M0 scale conversion `e8m0_from_bf16_bits_`
    not matching the old `(int16 -> int32 >> 7) -> uint8` expression on some bit
    pattern (a logical shift, a shift of the wrong width, or a narrowing cast that
    saturates instead of wrapping would all fail on exponents >= 128)
                                                                -> test_e8m0_from_bf16_bits
"""
from __future__ import annotations

import random
import sys

from sglang.srt.layers.moe.kt_contract import KTContractError
import torch

from sglang.srt.layers.moe.kt_prefill import (
    BufferState,
    PrefillPipeline,
    e8m0_from_bf16_bits_,
)

LAYERS = (1, 2, 3, 4)  # layer 0 dense, like DeepSeek-V4.1-Flash's first layer


def expect_raise(fn, *needle: str):
    try:
        fn()
    except KTContractError as e:
        for n in needle:
            assert n in str(e), (n, str(e))
        return e
    raise AssertionError(f"{fn} did not raise KTContractError")


def walk_layer(p: PrefillPipeline, layer: int, tokens: int) -> int:
    """The coordinator's per-layer IO order, as pipeline calls. Returns the next layer."""
    buf = p.layer_begin(layer, tokens)
    p.export_ready(buf)
    p.dma_issued(buf)
    nxt = p.layer_end(layer)
    if nxt is not None:
        nb = p.current_buf
        if p.needs_dma_wait(nb):
            p.dma_complete(nb)
        p.claim_for_export(nb, nxt)
    return nxt


def run_forward(p: PrefillPipeline, tokens: int) -> list:
    """Full forward; returns the buffer index used by each layer."""
    buf = p.begin_forward(tokens)
    if p.needs_dma_wait(buf):
        p.dma_complete(buf)
    p.claim_for_export(buf, p.first_layer)
    used = []
    layer = p.first_layer
    while layer is not None:
        used.append(p.current_buf)
        layer = walk_layer(p, layer, tokens)
    return used


def test_legal_cycle_two_forwards():
    p = PrefillPipeline(LAYERS)
    assert not p.open
    used = run_forward(p, 4096)
    assert used == [0, 1, 0, 1], used  # strict alternation within a forward
    assert not p.open and p.forwards_completed == 1
    # After the last layer its buffer (1) is still READING and buffer 0 is READING
    # too (layer 3's DMA event was never waited on: nothing claimed buffer 0 after it).
    assert [b.state for b in p.buffers] == [BufferState.READING, BufferState.READING]
    # Second forward: begin returns buffer 0, which needs its DMA wait first.
    buf = p.begin_forward(8192)
    assert buf == 0 and p.needs_dma_wait(0)
    expect_raise(lambda: p.claim_for_export(0, 1), "DMA has not been waited for")
    p.dma_complete(0)
    p.claim_for_export(0, 1)
    assert p.buffers[0].state is BufferState.EXPORTING and p.buffers[0].layer == 1
    layer = 1
    while layer is not None:
        layer = walk_layer(p, layer, 8192)
    assert p.forwards_completed == 2 and not p.open


def test_claim_refuses_unread_or_in_flight():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(2048)
    p.claim_for_export(0, 1)
    # EXPORTING buffer cannot be claimed again (would overwrite kt's in-flight write)
    expect_raise(lambda: p.claim_for_export(0, 1), "unread data would be overwritten")
    buf = p.layer_begin(1, 2048)
    p.export_ready(buf)
    # FILLED (synced, nobody has DMAed) cannot be overwritten either
    expect_raise(lambda: p.claim_for_export(0, 1), "filled", "unread data")
    p.dma_issued(buf)
    nxt = p.layer_end(1)
    assert nxt == 2 and p.current_buf == 1
    # the pipeline's next buffer is 1; claiming 0 (READING) is refused twice over
    e = expect_raise(lambda: p.claim_for_export(0, 2), "next buffer is 1")
    assert "buffer 0" in str(e)
    p.claim_for_export(1, 2)
    # and a wrong layer for the right buffer
    p.layer_begin(2, 2048)
    p.export_ready(1)
    p.dma_issued(1)
    assert p.layer_end(2) == 3
    expect_raise(lambda: p.claim_for_export(0, 4), "claimed while expecting 3")


def test_layer_order():
    p = PrefillPipeline(LAYERS)
    expect_raise(lambda: p.layer_begin(1, 100), "outside a forward")
    p.begin_forward(100)
    p.claim_for_export(0, 1)
    expect_raise(lambda: p.layer_begin(2, 100), "out of order (expected 1)")
    expect_raise(lambda: p.layer_begin(0, 100), "not a streaming layer")
    expect_raise(lambda: p.layer_begin(7, 100), "not a streaming layer")
    assert walk_layer(p, 1, 100) == 2
    # skipping layer 2
    expect_raise(lambda: p.layer_begin(3, 100), "out of order (expected 2)")
    # repeating layer 1
    expect_raise(lambda: p.layer_begin(1, 100), "out of order (expected 2)")
    # layer_end for a layer that is not the current one
    expect_raise(lambda: p.layer_end(3), "layer_end(3) while expecting 2")
    # layer_end before the layer's DMA was issued
    p.layer_begin(2, 100)
    expect_raise(lambda: p.layer_end(2), "DMA was not issued")


def test_token_count_fixed_per_forward():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(4096)
    p.claim_for_export(0, 1)
    walk_layer(p, 1, 4096)
    e = expect_raise(lambda: p.layer_begin(2, 4095), "has 4095 tokens", "began with 4096")
    assert "layer 2" in str(e)
    # the forward is still usable with the right count
    assert walk_layer(p, 2, 4096) == 3
    fresh = PrefillPipeline(LAYERS)
    expect_raise(lambda: fresh.begin_forward(0), "with 0 tokens")
    expect_raise(lambda: fresh.begin_forward(-5), "with -5 tokens")


def test_begin_forward_requires_closed():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(1024)
    p.claim_for_export(0, 1)
    walk_layer(p, 1, 1024)
    e = expect_raise(lambda: p.begin_forward(1024), "while a forward is open", "expected layer 2")
    assert "did not reach its last layer" in str(e)
    # finish it, then a new forward is fine
    layer = 2
    while layer is not None:
        layer = walk_layer(p, layer, 1024)
    assert not p.open
    p.begin_forward(1024)
    # an EXPORTING buffer left over (a forward that exported but never ran the
    # layer) blocks the next begin even though the forward counter says closed
    q = PrefillPipeline((5,))
    q.begin_forward(10)
    q.claim_for_export(0, 5)
    q.forward_tokens = None  # simulate an aborted forward with an unread export
    expect_raise(lambda: q.begin_forward(10), "buffer 0 exporting", "unread export")


def test_buffer_phase_order():
    p = PrefillPipeline(LAYERS)
    expect_raise(lambda: p.export_ready(0), "export_ready on host buffer 0 in state free")
    expect_raise(lambda: p.dma_issued(0), "dma_issued on host buffer 0 in state free")
    expect_raise(lambda: p.dma_complete(0), "dma_complete on host buffer 0 in state free")
    p.begin_forward(50)
    expect_raise(lambda: p.claim_for_export(1, 1), "next buffer is 0")
    p.claim_for_export(0, 1)
    # layer_begin requires the export to have been submitted into the current buffer
    q = PrefillPipeline(LAYERS)
    q.begin_forward(50)
    expect_raise(lambda: q.layer_begin(1, 50), "is free (layer None), not exporting layer 1")
    # DMA before the export is ready
    p.layer_begin(1, 50)
    expect_raise(lambda: p.dma_issued(0), "in state exporting")
    p.export_ready(0)
    expect_raise(lambda: p.export_ready(0), "in state filled")
    expect_raise(lambda: p.dma_complete(0), "in state filled")
    p.dma_issued(0)
    expect_raise(lambda: p.dma_issued(0), "in state reading")
    p.dma_complete(0)
    expect_raise(lambda: p.dma_complete(0), "in state free")
    # the buffer is now FREE while the layer is still in flight: layer_end must notice
    expect_raise(lambda: p.layer_end(1), "is free (layer None); the layer's DMA was not issued")


def test_forward_closes_at_last_layer():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(300)
    p.claim_for_export(0, 1)
    for layer, nxt in ((1, 2), (2, 3), (3, 4)):
        assert p.open
        assert walk_layer(p, layer, 300) == nxt
    assert p.open and p.expected_layer == 4
    assert walk_layer(p, 4, 300) is None
    assert not p.open and p.expected_layer is None and p.forwards_completed == 1
    expect_raise(lambda: p.layer_begin(4, 300), "outside a forward")
    expect_raise(lambda: p.layer_end(4), "while expecting no forward")
    snap = p.snapshot()
    assert snap["open"] is False and snap["forwards_completed"] == 1
    assert snap["buffers"] == [("reading", 3), ("reading", 4)]


def test_single_layer_and_odd_count():
    s = PrefillPipeline((9,))
    assert run_forward(s, 64) == [0]
    assert run_forward(s, 64) == [0]  # buffer 0 reused: needs_dma_wait path exercised
    assert s.forwards_completed == 2
    odd = PrefillPipeline((1, 2, 3))
    assert run_forward(odd, 7) == [0, 1, 0]
    # next forward starts at buffer 0, which layer 3 is still reading
    assert odd.needs_dma_wait(0) and odd.buffers[0].layer == 3
    assert run_forward(odd, 7) == [0, 1, 0]
    # random legal walks never raise and always alternate
    rng = random.Random(1234)
    for _ in range(200):
        n = rng.randint(1, 12)
        layers = sorted(rng.sample(range(0, 40), n))
        p = PrefillPipeline(layers)
        for _ in range(rng.randint(1, 4)):
            used = run_forward(p, rng.randint(1, 1 << 15))
            assert used == [i % 2 for i in range(n)], (layers, used)


def test_construction():
    expect_raise(lambda: PrefillPipeline(()), "no streaming layers")
    expect_raise(lambda: PrefillPipeline((3, 2)), "distinct and ascending")
    expect_raise(lambda: PrefillPipeline((2, 2)), "distinct and ascending")
    expect_raise(lambda: PrefillPipeline((1, 2), num_buffers=1), ">= 2 host buffers")
    expect_raise(lambda: PrefillPipeline([4, 1, 9]), "distinct and ascending")
    p = PrefillPipeline([1, 4, 9])  # list input is fine when ascending
    assert p.layers == (1, 4, 9) and p.first_layer == 1 and p.last_layer == 9
    assert len(p.buffers) == 2 and all(b.state is BufferState.FREE for b in p.buffers)


def test_e8m0_from_bf16_bits():
    # every 16-bit pattern once, in a [E, N, K/32]-shaped landing zone like the
    # streamed slots use (not only the e8m0 << 7 ones kt-kernel writes)
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    raw = bits.view(torch.bfloat16).reshape(4, 128, 128).clone()
    want = (raw.view(torch.int16).to(torch.int32) >> 7).to(torch.uint8)
    dst = torch.empty(raw.shape, dtype=torch.uint8)
    e8m0_from_bf16_bits_(raw, dst)
    assert dst.dtype == torch.uint8 and dst.shape == raw.shape
    assert torch.equal(dst, want), "e8m0 conversion differs from the int32 expression"
    # the e8m0 << 7 patterns in particular decode to the exponent itself
    exps = torch.arange(256, dtype=torch.int32)
    raw = (exps << 7).to(torch.int16).view(torch.bfloat16).reshape(1, 16, 16).clone()
    dst = torch.empty(raw.shape, dtype=torch.uint8)
    e8m0_from_bf16_bits_(raw, dst)
    assert torch.equal(dst.flatten().to(torch.int32), exps), "e8m0 << 7 does not round-trip"


def main() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok   {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

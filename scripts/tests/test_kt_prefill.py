#!/usr/bin/env python3
"""Deterministic, GPU-free tests for the streamed-prefill protocol (recipe T3.19d,
shared weight arenas T3.39 window 2).

Runs INSIDE the recipe image via scripts/tests/run-kt-prefill-tests.sh, which mounts
the overlay's kt_prefill.py (and kt_contract.py) through a stub package. Two objects
under test: PrefillPipeline, the protocol LayerwisePrefillCoordinator asks before every
IO step (pure Python, no torch), and ArenaLayout.from_kt, the check that kt's carve of
a layer arena is the constant-stride, E8M0 layout the staging copies assume.

With one GPU staging set per rank, the coordinator issues layer L+1's copies (on the
copy stream, after layer L's GEMM event) only after layer L's layer_end; layer_begin
of a layer must find exactly that layer staged. What a wrong implementation would get
wrong, and which test catches it:
  - layer L+1 staged before layer L's GEMM consumed the set, or a layer
    begun whose copy was never issued                       -> test_legal_cycle_two_forwards,
                                                               test_staging_order
  - accepting layers out of order / a skipped layer / a layer from
    outside the streaming set                                -> test_layer_order
  - the token count changing mid-forward (the chunk-size invariant)
                                                             -> test_token_count_fixed_per_forward
  - a new forward opened on top of an unfinished one, or on top
    of a stale staged copy                                   -> test_begin_forward_requires_closed
  - the forward not closing at the last layer / closing early -> test_forward_closes_at_last_layer
  - a single-layer streaming set and random legal walks      -> test_single_layer_and_random_walks
  - malformed construction accepted                          -> test_construction
  - kt's arena carve accepted with the wrong part count, expert
    count, non-E8M0 scales, or a non-constant expert stride  -> test_arena_layout
"""
from __future__ import annotations

import random

from sglang.srt.layers.moe.kt_contract import KTContractError
from sglang.srt.layers.moe.kt_prefill import ArenaLayout, PrefillPipeline

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
    """The coordinator's run_layer, as pipeline calls. Returns the next layer."""
    p.layer_begin(layer, tokens)
    nxt = p.layer_end(layer)
    if nxt is not None:
        p.staged(nxt)
    return nxt


def run_forward(p: PrefillPipeline, tokens: int) -> list:
    """Full forward; returns the layers in the order they ran."""
    first = p.begin_forward(tokens)
    p.staged(first)
    ran = []
    layer = first
    while layer is not None:
        ran.append(layer)
        layer = walk_layer(p, layer, tokens)
    return ran


def test_legal_cycle_two_forwards():
    p = PrefillPipeline(LAYERS)
    assert not p.open and p.staged_layer is None
    assert run_forward(p, 4096) == [1, 2, 3, 4]
    assert not p.open and p.forwards_completed == 1
    # nothing is staged across the forward boundary: the last layer's layer_end
    # returned None and no copy was issued
    assert p.staged_layer is None and not p.in_layer
    assert run_forward(p, 8192) == [1, 2, 3, 4]
    assert p.forwards_completed == 2 and not p.open


def test_staging_order():
    p = PrefillPipeline(LAYERS)
    # nothing may be staged before a forward is open
    expect_raise(lambda: p.staged(1), "staged(1) while expecting no forward")
    assert p.begin_forward(2048) == 1
    # layer_begin before its copy was issued
    expect_raise(lambda: p.layer_begin(1, 2048), "staging set holds nothing", "copy was not issued")
    p.staged(1)
    # the single staging set may not be refilled before its GEMM
    expect_raise(lambda: p.staged(1), "while layer 1 is staged", "overwritten before its GEMM")
    expect_raise(lambda: p.staged(2), "staged(2) while expecting 1")
    p.layer_begin(1, 2048)
    assert p.staged_layer is None and p.in_layer
    # no copy may be issued while the layer is running (the set is being read)
    expect_raise(lambda: p.staged(1), "inside layer 1", "before layer_begin")
    expect_raise(lambda: p.layer_begin(1, 2048), "layer_begin(1) twice without layer_end")
    assert p.layer_end(1) == 2
    # layer 2 begun before its copy
    e = expect_raise(lambda: p.layer_begin(2, 2048), "layer 2 begun but the staging set holds nothing")
    assert "copy was not issued" in str(e)
    p.staged(2)
    # a stale copy of the wrong layer: stage 2, then (illegally) ask to begin 3
    expect_raise(lambda: p.layer_begin(3, 2048), "out of order (expected 2)")
    p.layer_begin(2, 2048)
    assert p.layer_end(2) == 3


def test_layer_order():
    p = PrefillPipeline(LAYERS)
    expect_raise(lambda: p.layer_begin(1, 100), "outside a forward")
    expect_raise(lambda: p.layer_end(1), "while expecting no forward")
    p.begin_forward(100)
    p.staged(1)
    expect_raise(lambda: p.layer_begin(2, 100), "out of order (expected 1)")
    expect_raise(lambda: p.layer_begin(0, 100), "not a streaming layer")
    expect_raise(lambda: p.layer_begin(7, 100), "not a streaming layer")
    expect_raise(lambda: p.staged(0), "not a streaming layer")
    assert walk_layer(p, 1, 100) == 2
    # skipping layer 2
    expect_raise(lambda: p.layer_begin(3, 100), "out of order (expected 2)")
    # repeating layer 1
    expect_raise(lambda: p.layer_begin(1, 100), "out of order (expected 2)")
    # layer_end for a layer that is not the current one, and before layer_begin
    expect_raise(lambda: p.layer_end(3), "layer_end(3) while expecting 2")
    expect_raise(lambda: p.layer_end(2), "layer_end(2) without layer_begin")


def test_token_count_fixed_per_forward():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(4096)
    p.staged(1)
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
    p.staged(1)
    walk_layer(p, 1, 1024)
    e = expect_raise(lambda: p.begin_forward(1024), "while a forward is open", "expected layer 2")
    assert "did not reach its last layer" in str(e)
    # finish it, then a new forward is fine
    layer = 2
    while layer is not None:
        layer = walk_layer(p, layer, 1024)
    assert not p.open
    p.begin_forward(1024)
    # a staged copy left over (a forward that staged but never ran the layer)
    # blocks the next begin even though the forward counter says closed
    q = PrefillPipeline((5,))
    q.begin_forward(10)
    q.staged(5)
    q.forward_tokens = None  # simulate an aborted forward with an unconsumed copy
    expect_raise(lambda: q.begin_forward(10), "layer 5 still staged", "never consumed")


def test_forward_closes_at_last_layer():
    p = PrefillPipeline(LAYERS)
    p.begin_forward(300)
    p.staged(1)
    for layer, nxt in ((1, 2), (2, 3), (3, 4)):
        assert p.open
        assert walk_layer(p, layer, 300) == nxt
    assert p.open and p.expected_layer == 4 and p.staged_layer == 4
    assert walk_layer(p, 4, 300) is None
    assert not p.open and p.expected_layer is None and p.forwards_completed == 1
    assert p.staged_layer is None and not p.in_layer
    expect_raise(lambda: p.layer_begin(4, 300), "outside a forward")
    expect_raise(lambda: p.layer_end(4), "while expecting no forward")
    snap = p.snapshot()
    assert snap == {
        "open": False,
        "forward_tokens": None,
        "expected_layer": None,
        "staged_layer": None,
        "in_layer": False,
        "forwards_completed": 1,
    }, snap


def test_single_layer_and_random_walks():
    s = PrefillPipeline((9,))
    assert run_forward(s, 64) == [9]
    assert run_forward(s, 64) == [9]
    assert s.forwards_completed == 2
    # random legal walks never raise and visit every layer once in order
    rng = random.Random(1234)
    for _ in range(200):
        n = rng.randint(1, 12)
        layers = sorted(rng.sample(range(0, 40), n))
        p = PrefillPipeline(layers)
        for _ in range(rng.randint(1, 4)):
            assert run_forward(p, rng.randint(1, 1 << 15)) == layers
            assert p.staged_layer is None and not p.open


def test_construction():
    expect_raise(lambda: PrefillPipeline(()), "no streaming layers")
    expect_raise(lambda: PrefillPipeline((3, 2)), "distinct and ascending")
    expect_raise(lambda: PrefillPipeline((2, 2)), "distinct and ascending")
    expect_raise(lambda: PrefillPipeline([4, 1, 9]), "distinct and ascending")
    p = PrefillPipeline([1, 4, 9])  # list input is fine when ascending
    assert p.layers == (1, 4, 9) and p.first_layer == 1 and p.last_layer == 9
    assert p.next_layer(1) == 4 and p.next_layer(9) is None


def test_arena_layout():
    # kt's carve for E experts: gate, up, down blocks per expert at a constant stride
    gu, dn = 6_291_456, 6_291_456  # 64-aligned block sizes (values irrelevant to the check)
    stride = 2 * gu + dn
    n = 4

    def entry(e, e8=(True, True, True), shift=0):
        base = e * stride + shift
        return (base, base + gu, base + 2 * gu) + tuple(e8)

    good = [[entry(e) for e in range(n)]]
    lay = ArenaLayout.from_kt(good, n * stride, n)
    assert (lay.gate0, lay.up0, lay.down0, lay.stride, lay.nbytes) == (0, gu, 2 * gu, stride, n * stride)
    # a single expert has no stride to measure
    one = ArenaLayout.from_kt([[entry(0)]], stride, 1)
    assert one.stride == 0 and one.down0 == 2 * gu
    # two CPU TP parts: streaming reads one arena per layer
    expect_raise(lambda: ArenaLayout.from_kt([good[0], good[0]], n * stride, n), "2 CPU TP parts", "threadpool-count 1")
    # expert count mismatch
    expect_raise(lambda: ArenaLayout.from_kt(good, n * stride, n + 1), f"has {n} experts, expected {n + 1}")
    # scales left as bf16 (not compacted to E8M0) anywhere -> the copies would read the wrong bytes
    bad_first = [[entry(0, (True, False, True))] + [entry(e) for e in range(1, n)]]
    expect_raise(lambda: ArenaLayout.from_kt(bad_first, n * stride, n), "did not compact", "up=False")
    bad_later = [[entry(e) for e in range(n - 1)] + [entry(n - 1, (True, True, False))]]
    expect_raise(lambda: ArenaLayout.from_kt(bad_later, n * stride, n), f"expert {n - 1} is not at a constant stride")
    # a non-constant stride (expert 2 shifted by 64 bytes)
    shifted = [[entry(0), entry(1), entry(2, shift=64), entry(3)]]
    expect_raise(lambda: ArenaLayout.from_kt(shifted, n * stride, n), "expert 2 is not at a constant stride")
    # the expected tuple in the message names the stride-derived offsets
    e = expect_raise(lambda: ArenaLayout.from_kt(shifted, n * stride, n), f"vs expected {entry(2)[:3]}")
    assert "expert 2" in str(e)


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
    raise SystemExit(main())

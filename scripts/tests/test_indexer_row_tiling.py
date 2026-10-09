#!/usr/bin/env python3
"""GPU bit-equality test for the row-tiled dense fp4 indexer (recipe E34b, E34b-ii).

Runs INSIDE the recipe image with the overlay's python/sglang mounted over the installed
package (scripts/tests/run-indexer-tiling-tests.sh); needs one GPU (the deep_gemm
`fp8_fp4_mqa_logits` kernel, the fused ragged top-k and the block selection all run on
device). Takes the GPU for a few seconds only; safe while a server is not using GPU 0.

Background: `_low_ratio_index_topk_dense` scored every query row of a prefill chunk in one
`[rows, width]` fp32 logits tensor (4 B x prefix per row: 1 GiB at 64K, 1.75 GiB at 110K),
which is what OOMed the 128K ladder step (E34). E34b walks the rows in tiles sized by
`_TORCH_INDEXER_SCORE_BUDGET_BYTES`. Every kernel on the path is per row, so the tiling
must be bit-exact: the same `selected` indices and the same published candidate masks
whatever the tile size, including tiles that straddle a request boundary and 1-row tiles.

What a wrong implementation would get wrong, and which assertion catches it:
  - mapping a tile's rows to the wrong request (so the wrong `candidate_masks[b]` slice is
    applied, or a request's rows published under another request) -> consumer and source
    runs with 37-row tiles, whose tiles cross the 700-row request boundary.
  - slicing a request's mask by the tile-relative row instead of the request-relative row
    -> consumer run; masks are random per row, so a shifted slice changes `selected`.
  - passing whole-batch `ks` / `compress_lens` with a tile's logits (row i of the tile
    reads offset i of the batch) -> any run with more than one tile; request 1's rows
    would read request 0's offsets.
  - dropping or re-ordering published mask parts -> the concatenated mask equality.
  - forgetting `_mask_topk_scores` on a tile -> consumer run: underfilled rows keep -inf
    picks instead of -1.
  - a kernel that is not actually row-independent (accumulation that depends on T) ->
    the 1-row-tile run.

E34b-ii: the source publishes block-level masks `[rows, ceil(lc / block_size)]`
(`select_candidate_block_keep`) and each consumer tile expands its own rows
(`expand_candidate_blocks`). What a wrong implementation would get wrong:
  - an expansion that is not the old `repeat_interleave(block_size)[..., :width]` ->
    `test_block_expansion` compares it with an index-gather reference and checks the
    old per-position mask (`select_candidate_blocks`) equals the expanded keep.
  - a consumer that still indexes the block mask by position (or expands to the wrong
    width) -> consumer runs on block masks: a pick outside a kept block, or a shape error.
  - publishing per-position masks again -> the published shape check.

E34e e-i: the source publishes block-ID lists `[rows, min(topk_blocks, blocks)]` int32,
-1 padded (`select_candidate_block_ids`), prefix-independent in size, and the consumer
expands them (`expand_candidate_ids`). What a wrong implementation would get wrong:
  - an ID list that is not the keep mask (a dropped -inf check keeps unreachable
    blocks; a clamp of -1 to 0 inside the scatter can erase a real block-0 pick) ->
    `test_block_ids` compares expand(ids) with the keep path bit for bit at every lc,
    with rows that have fewer reachable blocks than topk (so -1 padding occurs while
    block 0 is kept), and counts the -1 entries against the keep mask.
  - publishing keep masks again -> the published shape / dtype check.
  - a consumer that reads the IDs as a mask -> consumer runs on ID lists.

Equality is on the picks sorted per row: the fused ragged top-k writes a row's k picks
in a launch-shape-dependent order (measured: identical sets, different order, at every
tile size), and `_low_ratio_index_topk_dense` sorts each row ascending before any
consumer sees it. The logits themselves are bit-identical at every tile size (checked
while writing this test); the sort is the only normalisation applied.
"""
from __future__ import annotations

import sys
import types

import torch

import sglang.srt.layers.attention.deepseek_v4_backend as backend
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.dsv4.indexer import (
    expand_candidate_blocks,
    expand_candidate_ids,
    select_candidate_block_ids,
    select_candidate_block_keep,
    select_candidate_blocks,
)

DEV = torch.device("cuda")
HEADS = 32  # index_n_heads
RATIO = 1  # the low-ratio dense path; compress_lens == pos + 1
# two requests: visible compressed positions and query rows (the newest rows)
LC = [6001, 1500]  # 6001: the last block is partial, so expansion must truncate
Q_LENS = [700, 300]
HUGE = 1 << 40
BLOCK = 8
TOPK_BLOCKS = 2048


def blocks_of(lc: int) -> int:
    return -(-lc // BLOCK)


def ids_width(lc: int) -> int:
    return min(TOPK_BLOCKS, blocks_of(lc))


def ids_of(keep: torch.Tensor) -> torch.Tensor:
    """A keep mask as the ID list a source publishes: kept block IDs, -1 padded."""
    top = keep.to(torch.int32).topk(ids_width(keep.shape[1] * BLOCK), dim=-1)
    return torch.where(top.values > 0, top.indices, -1).to(torch.int32)


def make_indexer(source: bool, consume: bool):
    return types.SimpleNamespace(
        is_candidate_source=source,
        uses_candidates=consume,
        index_topk=512,
        candidate_topk_blocks=TOPK_BLOCKS,
        candidate_block_size=BLOCK,
    )


def make_inputs(seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    num_tokens = sum(Q_LENS)
    total_kv = sum(LC)
    q = torch.randn((num_tokens, HEADS, 128), device=DEV, generator=g)
    k = torch.randn((total_kv, 128), device=DEV, generator=g)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(num_tokens, HEADS, 64)
    q_sf = q_sf.view(num_tokens, HEADS)
    k_fp4, k_sf = quantize_fp4_indexer_tensor(k, rne=True)
    weights = torch.rand((num_tokens, HEADS), device=DEV, generator=g) + 0.1
    pos_chunks, starts, start = [], [], 0
    for lc, t_len in zip(LC, Q_LENS):
        starts.append(start)
        pos_chunks.append(torch.arange(lc - t_len, lc, device=DEV))
        start += lc
    pos = torch.cat(pos_chunks)
    compress_lens = ((pos + 1) // RATIO).to(torch.int32)
    ks = torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.int32, device=DEV),
        torch.tensor(Q_LENS, dtype=torch.int64, device=DEV),
        output_size=num_tokens,
    )
    empty_mask = torch.zeros(0, 0, dtype=torch.int32, device=DEV)
    return (q_fp4, q_sf), (k_fp4, k_sf), weights, ks, compress_lens, empty_mask


def run(mode: str, budget: int, inputs, masks=None):
    """One `_dense_index_topk_rows` call with the budget monkeypatched; returns
    (selected, candidate_masks or None)."""
    q_fp4, kv_fp4, weights, ks, compress_lens, empty_mask = inputs
    source, consume = mode == "source", mode == "consume"
    stub = types.SimpleNamespace(candidate_masks=masks)
    saved = backend._TORCH_INDEXER_SCORE_BUDGET_BYTES
    backend._TORCH_INDEXER_SCORE_BUDGET_BYTES = budget
    try:
        selected = DeepseekV4AttnBackend._dense_index_topk_rows(
            stub,
            make_indexer(source, consume),
            q_fp4,
            kv_fp4,
            weights,
            ks,
            compress_lens,
            LC,
            Q_LENS,
            empty_mask,
        )
    finally:
        backend._TORCH_INDEXER_SCORE_BUDGET_BYTES = saved
    torch.cuda.synchronize()
    # the caller's normalisation: -1 (underfilled) last, positions ascending
    unselected = torch.iinfo(torch.int32).max
    ordered = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
    ordered = ordered.masked_fill(ordered == unselected, -1)
    return ordered, ([m.clone() for m in stub.candidate_masks] if source else None)


def tiles_for(rows_per_tile: int) -> int:
    width = backend.ceil_align(max(LC), 4)
    return rows_per_tile * width * 4


def random_masks(seed: int = 1):
    """Block-ID lists, the form a source publishes (E34e e-i), from random keep masks."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = []
    for lc, t_len in zip(LC, Q_LENS):
        m = torch.rand((t_len, blocks_of(lc)), device=DEV, generator=g) < 0.08
        m[:, -1] = True  # the block holding the newest position is always a candidate
        out.append(ids_of(m))
    return out


def check_shape_and_validity(selected, compress_lens, ks):
    assert selected.shape == (sum(Q_LENS), 512), selected.shape
    valid = selected >= 0
    assert valid.any(dim=1).all(), "every row must select something"
    col = selected.to(torch.int64) - ks[:, None].to(torch.int64)
    in_window = (col >= 0) & (col < compress_lens[:, None].to(torch.int64))
    assert bool((in_window | ~valid).all()), "a pick outside its row's window"


def test_plain_is_tile_invariant(inputs):
    one, _ = run("plain", HUGE, inputs)
    check_shape_and_validity(one, inputs[4], inputs[3])
    for rows in (37, 1):
        tiled, _ = run("plain", tiles_for(rows), inputs)
        assert torch.equal(one, tiled), f"plain: {rows}-row tiles differ"


def test_source_is_tile_invariant(inputs):
    one, masks_one = run("source", HUGE, inputs)
    check_shape_and_validity(one, inputs[4], inputs[3])
    assert [m.shape for m in masks_one] == [
        (t, ids_width(lc)) for lc, t in zip(LC, Q_LENS)
    ], [m.shape for m in masks_one]
    assert all(m.dtype == torch.int32 for m in masks_one), [m.dtype for m in masks_one]
    for rows in (37, 1):
        tiled, masks_tiled = run("source", tiles_for(rows), inputs)
        assert torch.equal(one, tiled), f"source: {rows}-row tiles differ"
        for b, (a, c) in enumerate(zip(masks_one, masks_tiled)):
            assert torch.equal(a, c), f"source: request {b} mask differs at {rows}-row tiles"
    return masks_one


def test_consume_is_tile_invariant(inputs, masks):
    one, _ = run("consume", HUGE, inputs, masks)
    check_shape_and_validity(one, inputs[4], inputs[3])
    # the consumer must never pick a non-candidate position
    col = one.to(torch.int64) - inputs[3][:, None].to(torch.int64)
    r0 = 0
    for b, (lc, t_len) in enumerate(zip(LC, Q_LENS)):
        rows = slice(r0, r0 + t_len)
        r0 += t_len
        picks = col[rows]
        ok = picks < 0
        valid = ~ok
        keep = expand_candidate_ids(masks[b], BLOCK, lc)
        gathered = keep.gather(1, picks.clamp(0, lc - 1))
        assert bool((gathered | ok).all()), f"consume: non-candidate picked in request {b}"
        assert bool((valid.sum(dim=1) <= keep.sum(dim=1)).all())
    for rows in (37, 1):
        tiled, _ = run("consume", tiles_for(rows), inputs, masks)
        assert torch.equal(one, tiled), f"consume: {rows}-row tiles differ"


def test_consume_sees_the_mask(inputs):
    """A sparse random mask must change the selection vs plain (otherwise the test
    above would pass with the consume branch deleted)."""
    plain, _ = run("plain", HUGE, inputs)
    masked, _ = run("consume", HUGE, inputs, random_masks())
    assert not torch.equal(plain, masked)


def test_block_expansion():
    """`expand_candidate_blocks` is the old per-position mask, bit for bit, including
    the truncated last block; and the old `select_candidate_blocks` is exactly
    keep -> expand on real block scores."""
    g = torch.Generator(device=DEV).manual_seed(2)
    for lc in (6001, 1500, 8, 9, 1):
        rows = 5
        keep = torch.rand((rows, blocks_of(lc)), device=DEV, generator=g) < 0.3
        got = expand_candidate_blocks(keep, BLOCK, lc)
        ref = keep[:, torch.arange(lc, device=DEV) // BLOCK]
        assert got.shape == (rows, lc), (got.shape, lc)
        assert torch.equal(got, ref), f"expansion differs at lc={lc}"
        logits = torch.randn((rows, lc), device=DEV, generator=g)
        lens = torch.randint(1, lc + 1, (rows, 1), device=DEV, generator=g)
        logits = logits.masked_fill(
            torch.arange(lc, device=DEV)[None, :] >= lens, -torch.inf
        )
        old = select_candidate_blocks(logits, lens, topk_blocks=64, block_size=BLOCK)
        new = select_candidate_block_keep(logits, lens, topk_blocks=64, block_size=BLOCK)
        assert new.shape == (rows, blocks_of(lc)), new.shape
        assert torch.equal(old, expand_candidate_blocks(new, BLOCK, lc)), lc


def test_block_ids():
    """`expand_candidate_ids(select_candidate_block_ids(...))` is the keep path bit for
    bit; the list is int32 `[rows, min(topk, blocks)]` and its -1 count per row is the
    number of unreachable blocks among the topk picks."""
    g = torch.Generator(device=DEV).manual_seed(3)
    for lc in (6001, 1500, 8, 9, 1):
        rows = 64
        logits = torch.randn((rows, lc), device=DEV, generator=g)
        # short lens: rows with fewer reachable blocks than topk, so -1 padding occurs
        # while block 0 is kept
        lens = torch.randint(1, lc + 1, (rows, 1), device=DEV, generator=g)
        lens[: rows // 2] = torch.randint(1, min(lc, 40) + 1, (rows // 2, 1), device=DEV, generator=g)
        logits = logits.masked_fill(
            torch.arange(lc, device=DEV)[None, :] >= lens, -torch.inf
        )
        for topk in (64, TOPK_BLOCKS):
            k = min(topk, blocks_of(lc))
            keep = select_candidate_block_keep(logits, lens, topk_blocks=topk, block_size=BLOCK)
            ids = select_candidate_block_ids(logits, lens, topk_blocks=topk, block_size=BLOCK)
            assert ids.dtype == torch.int32 and ids.shape == (rows, k), (ids.dtype, ids.shape)
            assert torch.equal((ids < 0).sum(1), k - keep.sum(1)), (lc, topk)
            assert torch.equal(
                expand_candidate_ids(ids, BLOCK, lc), expand_candidate_blocks(keep, BLOCK, lc)
            ), f"ids differ from keep at lc={lc} topk={topk}"
            # the collision case is exercised: block 0 kept in some row, -1 padding in some row
            assert bool(keep[:, 0].any()), (lc, topk)
            assert k == 1 or bool((ids < 0).any()), (lc, topk)


def main() -> int:
    if not torch.cuda.is_available():
        print("indexer row tiling tests: no GPU; skipped")
        return 3
    inputs = make_inputs()
    failures = 0
    for name, fn in (
        ("block expansion", test_block_expansion),
        ("block ids", test_block_ids),
        ("plain", lambda: test_plain_is_tile_invariant(inputs)),
        ("source", lambda: test_source_is_tile_invariant(inputs)),
        ("consume(random mask)", lambda: test_consume_is_tile_invariant(inputs, random_masks())),
        ("consume sees mask", lambda: test_consume_sees_the_mask(inputs)),
    ):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    # the consumer run on masks a source layer actually published
    try:
        masks = test_source_is_tile_invariant(inputs)
        test_consume_is_tile_invariant(inputs, masks)
        print("PASS consume(source-published mask)")
    except Exception as e:  # noqa: BLE001
        failures += 1
        print(f"FAIL consume(source-published mask): {type(e).__name__}: {e}")
    print(f"{'OK' if failures == 0 else 'FAILED'}: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

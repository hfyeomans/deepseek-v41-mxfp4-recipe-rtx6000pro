#!/usr/bin/env python3
"""GPU bit-equality test for the fp4 indexer key-row gather (recipe E34d-v (ii)).

Runs INSIDE the recipe image with the overlay's python/sglang mounted over the installed
package (scripts/tests/run-prefill-transient-tests.sh); needs one GPU for a few seconds.

Background: `DeepSeekV4IndexerPool.get_index_k_fp4(layer_id, slots)` returns the packed
fp4 key rows (payload int8 [n, 64], scales int32 [n]) at `slots` of a page buffer laid
out `[pages, page_size * 64 payload bytes | page_size * 4 scale bytes]`; the low-ratio
dense indexer passes the whole prefix of a request as `slots`. The previous
implementation gathered with per-byte column grids ([n, 64] + [n, 4] int64 = 544 B per
slot), a 332 MiB transient at the ~680K prefix where the 1M x 1 prefill OOMed (E34d-v (i)).
The fix indexes `[pages, page_size, 64]` / `[pages, page_size, 4]` views with the [n]
page and offset tensors (16 B per slot). It must return the same bytes.

The reference here is the old column-grid gather, written out inline (the production code
no longer has it); both read the same randomly filled buffer and are compared byte for
byte. What a wrong implementation would get wrong, and which case catches it:
  - splitting the page at the wrong byte (payload region `p * 63` or `p * 65` wide, or the
    scale region read from the payload bytes) -> every slot with off > 0 reads shifted
    bytes; the random fill makes any shift visible.
  - a page/offset mix-up (slot % p as the page) -> slots in page 1+ differ.
  - a view that drops the row stride (reshape of a non-contiguous slice copying the whole
    buffer, or viewing the full `[pages, p * 68]` row as `[pages, p, 68]`) -> the second
    returns 68-byte rows: shape or byte mismatch.
  - the last page: `size` is not a multiple of `page_size` here, so the gather at the
    highest slots reads the partial trailing page.
  - unsorted and repeated slots: an implementation that sorts or deduplicates (an
    `index_select` on unique pages, say) changes the row order.
  - `scales`: the 4 scale bytes must come back as one little-endian int32 per slot, so a
    non-contiguous gather that `.view(torch.int32)` rejects, or a byte-order change,
    fails the int32 equality.
The dequant readback `get_index_k_dequant` shares the gather and is compared against the
same reference's bf16 output. A size check asserts the gather allocates no more than
~96 B per slot on top of its output (the old path peaked at ~600 B per slot).
"""
from __future__ import annotations

import sys

import torch

from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool

DEV = torch.device("cuda")
PAGE = 256  # the recipe's served page size (launch log: c128_page_size 256)
HEAD_DIM = 128  # index_head_dim: 64 payload + 4 scale bytes per token
SIZE = 7 * PAGE + 133  # pool slots: the last page is partial


def make_pool() -> DeepSeekV4IndexerPool:
    pool = DeepSeekV4IndexerPool(
        SIZE,
        PAGE,
        torch.bfloat16,
        HEAD_DIM,
        layer_num=2,
        device="cuda",
        enable_memory_saver=False,
        use_fp4_indexer=True,
    )
    g = torch.Generator(device=DEV).manual_seed(20261009)
    for buf in pool.index_k_with_scale_buffer:
        buf.copy_(torch.randint(0, 256, buf.shape, generator=g, device=DEV, dtype=torch.int64).to(torch.uint8))
    return pool


def reference_gather(buf: torch.Tensor, p: int, slots: torch.Tensor):
    """The pre-E34d-v (ii) column-grid gather, verbatim."""
    slots = slots.to(torch.int64)
    page, off = (slots // p).unsqueeze(-1), slots % p
    payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
    scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(4, device=buf.device)
    payload = buf[page, payload_cols].view(torch.int8)
    scales = buf[page, scale_cols].contiguous().view(torch.int32).squeeze(-1)
    return payload, scales


def reference_dequant(buf: torch.Tensor, p: int, slots: torch.Tensor) -> torch.Tensor:
    slots = slots.to(torch.int64)
    page, off = (slots // p).unsqueeze(-1), slots % p
    payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
    scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(4, device=buf.device)
    u = buf[page, payload_cols].view(torch.uint8)
    codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)
    vals = DSV4_DEQUANT_FP4_TABLE.to(buf.device)[codes.long()].flatten(1)
    exps = buf[page, scale_cols].to(torch.int32) & 0xFF
    scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
    return (vals * scales).to(torch.bfloat16)


def slot_sets() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator(device="cpu").manual_seed(7)
    last = SIZE - 1
    sets = [
        ("single slot 0", torch.tensor([0])),
        ("single last slot", torch.tensor([last])),
        ("page-0 boundary", torch.tensor([PAGE - 1, PAGE, PAGE + 1])),
        ("whole prefix 255", torch.arange(255)),
        ("whole prefix 256", torch.arange(256)),
        ("whole prefix 257", torch.arange(257)),
        ("whole pool", torch.arange(SIZE)),
        ("unsorted repeated", torch.tensor([last, 3, 3, PAGE, 0, last, PAGE * 5 + 7, 3])),
        ("random 100000 with repeats", torch.randint(0, SIZE, (100000,), generator=g)),
        ("int32 slots", torch.randint(0, SIZE, (1000,), generator=g).to(torch.int32)),
    ]
    return [(name, s.to(DEV)) for name, s in sets]


def test_packed_gather(pool: DeepSeekV4IndexerPool) -> None:
    for layer in range(pool.layer_num):
        buf = pool.index_k_with_scale_buffer[layer]
        for name, slots in slot_sets():
            got_p, got_s = pool.get_index_k_fp4(layer, slots)
            ref_p, ref_s = reference_gather(buf, PAGE, slots)
            assert got_p.dtype == torch.int8 and got_s.dtype == torch.int32, (name, got_p.dtype, got_s.dtype)
            assert got_p.shape == ref_p.shape == (slots.numel(), 64), (name, got_p.shape)
            assert got_s.shape == ref_s.shape == (slots.numel(),), (name, got_s.shape)
            assert torch.equal(got_p, ref_p), f"payload differs: layer {layer} {name}"
            assert torch.equal(got_s, ref_s), f"scales differ: layer {layer} {name}"


def test_dequant(pool: DeepSeekV4IndexerPool) -> None:
    buf = pool.index_k_with_scale_buffer[1]
    for name, slots in slot_sets():
        got = pool.get_index_k_dequant(1, slots)
        ref = reference_dequant(buf, PAGE, slots)
        assert got.shape == ref.shape == (slots.numel(), HEAD_DIM), (name, got.shape)
        # bit patterns: random scale bytes make some rows 0 * inf = NaN, which torch.equal
        # would call unequal
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), f"dequant differs: {name}"
    got = pool.get_index_k_dequant(1)  # slots=None: every slot
    ref = reference_dequant(buf, PAGE, torch.arange(SIZE, device=DEV))
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), "dequant(None)"


def test_gather_is_small(pool: DeepSeekV4IndexerPool) -> None:
    """Peak allocation of the gather minus its output, per slot."""
    n = 100000
    slots = torch.randint(0, SIZE, (n,), device=DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    payload, scales = pool.get_index_k_fp4(0, slots)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    out = payload.numel() + scales.numel() * 4
    per_slot = (peak - out) / n
    assert per_slot <= 96, f"gather transient {per_slot:.0f} B per slot (expected <= 96: two int64 [n])"
    print(f"      gather transient {per_slot:.0f} B per slot above the {out / n:.0f} B output")
    # and the reference really was the fat path
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    ref_p, ref_s = reference_gather(pool.index_k_with_scale_buffer[0], PAGE, slots)
    torch.cuda.synchronize()
    ref_per_slot = (torch.cuda.max_memory_allocated() - base - out) / n
    assert ref_per_slot >= 500, f"reference transient {ref_per_slot:.0f} B per slot; expected >= 500"
    print(f"      reference transient {ref_per_slot:.0f} B per slot")


def main() -> int:
    if not torch.cuda.is_available():
        print("indexer k gather tests: no GPU; skipped")
        return 3
    pool = make_pool()
    failures = 0
    for name, fn in (
        ("packed gather == reference", lambda: test_packed_gather(pool)),
        ("dequant == reference", lambda: test_dequant(pool)),
        ("gather transient is small", lambda: test_gather_is_small(pool)),
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

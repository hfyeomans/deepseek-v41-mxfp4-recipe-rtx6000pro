# E34b — row-tiled dense fp4 indexer scoring (2026-10-08)

Protocol: [`plan/EXPERIMENTS.md` E34](../../../plan/EXPERIMENTS.md), part b. Task T3.26.
Overlay commit `c639cbee` on top of `1953a92e` / `9fc14817`: `_low_ratio_index_topk_dense` in
`deepseek_v4_backend.py` now calls `_dense_index_topk_rows`, which walks the chunk's query rows in
tiles of `_TORCH_INDEXER_SCORE_BUDGET_BYTES // (width x 4 B)` rows (width = the longest visible
prefix in the batch, 4-aligned) and, per tile, runs the fp4 MQA logits kernel, publishes or consumes
the candidate masks per request segment, and the fused ragged top-k. The transient is now
`[rows_per_tile, prefix]` (<= 256 MiB) instead of `[4096, prefix]` (1.75 GiB at 110K, 4 GiB at
262K). Same profile as E34a (226 / 158, D=4, spec off, chunk 4096, `MEM_FRACTION_STATIC=0.96`).

## Why rows, not key blocks

The E34 registration projected key-block tiling with a running top-k. Reading the path showed
every kernel on it is per query row: `fp8_fp4_mqa_logits` (row t -> `k[ks_t .. ke_t]`),
`select_candidate_blocks` (`topk` along dim -1), `topk_transform_ragged_v2` (row i selects over
`scores[i, :seq_lens[i]]`), `_mask_topk_scores` (gather per row). Tiling over rows therefore
changes nothing a row computes — it is bit-exact by construction, with no running-top-k merge and
no tie-break question. A key-block tiling would have needed a merge step with its own tie rules
and a second pass; rows needed neither.

## Bit-equality test (GPU)

[`scripts/tests/test_indexer_row_tiling.py`](../../../scripts/tests/test_indexer_row_tiling.py)
via `run-indexer-tiling-tests.sh` (inside the image, overlay mounted, GPU 0): two requests
(visible prefix 6000 / 1500, query rows 700 / 300, so tiles straddle the request boundary),
synthetic fp4 q / k from `quantize_fp4_indexer_tensor`, modes plain / source / consumer (random
sparse masks and masks a source run published), budget patched to one tile vs 37-row tiles vs
1-row tiles. Result: **5 / 5 pass** — `selected` identical after the caller's per-row sort, published
masks identical. Measured while writing the test: the logits are bit-identical at every tile size;
the fused top-k returns identical *sets* per row in a launch-shape-dependent *order* (every row's
order differs between tile sizes), which the caller's ascending sort normalises before any
consumer sees it. A mutant that slices the consumer mask tile-relative instead of request-relative
fails 3 / 5.

## Result (measured)

| Rung | Slots | Pool | Free after capture (GPU0 / GPU1) | (b) stress-kv | (c) needles | (d) NLL vs T3.16c | (e) health | (f) prefill 16K / 32K | C1 summed / wall / TTFT | Min free during stress | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 128K | 1 | 147,456 | 4.05 / 3.97 GB (= E34a) | **pass**: 125,773-token prompt + 256, TTFT 154.0 s, wall 160.2 s, peak 96,552 / 96,627 MiB | **6/6 at 110,666 tokens**, TTFT 109.8 s, `finish_reason=stop` | 0 of 4,795 moved (max abs logp 0.000) | 0 lines, EDAC 0/0, 0 Xid | **943.6 tok/s / 16.9 s** at 16K; **929.7 / 34.4 s** at 32K | 42.00 / 40.07 / 0.607 s | **1,335 / 1,260 MiB** | **pass** (E34a: OOM at 110K prefix, 196 MiB free) |
| 64K | 4 | 278,528 | 3.48 / 3.42 GB (= E34a) | **pass**: 4 x 62K all finished, TTFT 89.8 / 152.0 / 214.8 / 272.7 s, wall 290.1 s, peak 96,350 / 96,409 MiB | 6/6 at 55,206 tokens, TTFT 53.3 s | 0 of 4,795 moved | 0 lines, EDAC 0/0, 0 Xid | **1,048.3 tok/s / 15.2 s** at 16K; **1,041.7 / 30.7 s** at 32K | 42.14 / 40.20 / 0.609 s | **1,537 / 1,478 MiB** (E34a 737 / 650; projected +0.9 GB) | **pass** |

| 262K | 1 | 278,528 | 3.76 / 3.69 GB | **FAIL: CUDA OOM at 184K prefix** (chunk 46 of 64): `torch.cat(parts)` of the source layer's candidate mask asked for 720 MiB with 643 / 248 MiB free (GPU0 / GPU1) | not reached | — | 0 NaN / Xid (Python-level OOM, clean SIGQUIT) | per-chunk 929–963 tok/s, flat from 40K to 184K prefix | — | — | **fail, gate (b) — new binder** |

## 262K: the score matrix is gone; the candidate mask is next

The 46th chunk died allocating the **published candidate mask**, not scores: the candidate-source
layer (layer 20) publishes a `[chunk_rows, prefix]` bool per request that layers 21+ consume
(`self.candidate_masks[b]`), 1 B × prefix per row = 720 MiB at 184K, 1 GiB at 262K, 4 GiB at 1M.
The row tiling builds it in per-tile parts and `torch.cat`s them, so at the moment of the cat it
exists twice — the allocation that failed is the second copy. Both ranks failed at the same
size; GPU1 had 248 MiB free.

What a fix looks like (not done; registered as E34b-ii in EXPERIMENTS): the mask is 8× redundant —
`select_candidate_blocks` decides per 8-position block and `repeat_interleave(8)`s the result.
Publishing the block-level `[rows, ceil(prefix / 8)]` bool and expanding it per tile at the three
prefill consumers (`_dense_index_topk_rows`, `_low_ratio_index_topk_torch`, the late-layer tail
slice is row-only and unaffected) is bit-exact by construction and makes the resident mask 90 MiB
at 184K, 128 MiB at 262K, 512 MiB at 1M, with no duplicate at publish time. Preallocating the full
mask and writing tiles in place would only remove the duplicate (saves 720 MiB here) and leaves a
1 GiB resident mask at 262K — too thin to qualify, so the block-level representation is the one to
build.

Side finding: the per-chunk prefill rate is **flat** with prefix on this path — chunk throughput
929 / 946 / 958 / 955 / 963 tok/s at 40K / 80K / 120K / 160K / 184K of prefix (log lines). The
earlier reading that prefill "degrades per token beyond 32K" came from end-to-end TTFT (126K in
154 s = 817 tok/s); the 14 % between the two is tokenization and inter-chunk scheduling, not the
indexer.

## Prefill got faster, not slower

The gate was "<= 10 % prefill cost at 32K". Measured on the identical 64K x 4 profile (same
image, same experts, same chunk, E34a soak vs this serve): 16K prefill **842 -> 1,048 tok/s
(+24 %)**, 32K **845 -> 1,042 (+23 %)**; TTFT at 32K 38.1 -> 30.7 s. Decode is unchanged
(C1 40.20 wall vs 40.26 / 39.94). The 128K x 1 serve shows the same direction (944 / 930) with a
slot-count confound, so the 64K x 4 pair is the number to quote.

Why (projected, not traced): the old path allocated a fresh 1 GiB logits + a 256 MiB bool mask +
a 256 MiB pad copy per chunk from a pool with ~3.4 GB free after capture. The caching allocator
retries a failed `cudaMalloc` by freeing cached blocks and re-allocating (a device sync each
time) and only logs it when the retry also fails — E34a's 128K traceback shows exactly those
retry sizes; the 64K x 4 logs have no line because the retries succeeded. 256 MiB tiles recycle one
cached block and never reach the retry. A per-layer prefill trace (`KT_TIMELINE` covers decode
only) would settle it; it is not needed for the decision.

Margin: the 64K x 4 worst case moved from 737 / 650 MiB free (E34a soak) to 1,537 / 1,478 MiB —
+0.8 GB, against a projected +0.9 GB (the `[4096, lc]` logits + mask at 62K prefix, less one
256 MiB tile). Projection held this time.

## Files

- `128k/`: `launch.txt`, `launch-container.txt`, `stress-kv.json/.txt`, `verify-128k.json/.txt`,
  `nll.json/.txt`, `prefill-16k.json/.txt`, `prefill-32k.json/.txt`, `c1-off6.json`, `c1-bench.txt`.
- `64k-4slot/`: same shape at the deployment profile (`verify-64k.*` instead of `verify-128k.*`).
- `262k/`: `launch.txt`, `launch-container.txt` (traceback), `stress-kv.txt` (HTTP 500 after the crash).

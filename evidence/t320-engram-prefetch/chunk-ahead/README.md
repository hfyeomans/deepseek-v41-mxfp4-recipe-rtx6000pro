# T3.20 (c) — the next prefill chunk's Engram rows fetched during the current chunk (2026-10-10)

Two serves at launcher defaults (219 / 165 by the el8 mix map, D=4, chunk 8192, 64K × 4,
`MEM_FRACTION_STATIC=0.96`, `ENGRAM_FILE_PREFETCH=ahead`, `KT_CPUINFER=16`), back to back
(09:46–10:11 local), overlay `9d481b0b` (= `c777da2d` + this change), recipe at the commit that adds this
README. Serve `off/` = `ENGRAM_CHUNK_AHEAD=0` (the T3.8b qualified behaviour: each chunk's rows fetched from
its own post-hasher break); serve `on/` = the new default `ENGRAM_CHUNK_AHEAD=1`
(`SGLANG_DSV41_ENGRAM_CHUNK_AHEAD`). Driver `ab.sh` (committed beside this README). Every serve starts with the
checkpoint pages dropped by the loader, so the first pass of each `bench-prefill.py --warm-pass` is the
cold-box reading (both shards 0.01 % resident before both serves' cold passes, 5.11 % before the warm ones —
equal box state).

## What changed (overlay `9d481b0b`)

The T3.39 re-profile put the Engram row fetch at 2.2 s of a 7.5 s cold-box 16K TTFT (29 %): at every
8192-token chunk the post-hasher break submits both tables' gathers to the one shared `engram-ahead` worker,
each ≈ 1.1–1.3 s on a cold box, and layer 1 / layer 14 of that chunk wait for them. The hash ids are a pure
function of the input ids, so chunk k+1's rows can be fetched while chunk k runs:

- `ngram_embedding_manager._prepare_engram_history` records, for an extend batch that carries the chunked
  request, the next `chunked_prefill_size` tokens of `full_untruncated_fill_ids` plus the n−1 tokens of
  history before them (`ScheduleBatch.ne_next_chunk` → `ForwardBatch.engram_next_chunk`, a frozen
  `EngramNextChunk(tokens, history, start)`).
- `EngramHasher.hash_chunk` hashes those tokens with the extend kernel (no history commit); the result equals
  the rows the whole-prompt hasher produces for the same offsets (in-image check at four offsets).
- `_engram_file_prefetch` hashes + D2Hs the next chunk after submitting the current chunk's gathers and
  submits each table's rows with `chunk_ahead=True` into a separate `_ahead_chunk` slot, **serially on the one
  shared worker** (one worker per table was measured at −4 to −8 % prefill in T3.20 round 2, so the T3.39
  note's "both tables in parallel" was not built). The next chunk's normal `submit` adopts the slot when its
  ids are a row-major prefix of the slot's (`chunk-ahead hit`); a slot the next forward does not cover is
  drained and dropped (`chunk-ahead dropped`); `gather` slices the rows when the adopted fetch is longer
  (the last chunk of a prompt is shorter than `chunked_prefill_size`).
- Decode is untouched: `engram_next_chunk` is `None` on decode batches, and prefill runs eager
  (`--cuda-graph-backend-decode breakable` only).

Accepted limitation: a request's last chunk batched with another request's first chunk does not match the
slot (the slot was hashed for the chunked request alone) and falls back to a synchronous fetch, counted as
`ahead miss`. Interleaved decode steps can queue behind a chunk-ahead fetch on the worker (bounded by one
cold gather, ≈ 0.7 s; ≈ 0 warm).

## Result

| | `off` (`ENGRAM_CHUNK_AHEAD=0`) | **`on` (default)** | on vs off |
|---|---|---|---|
| prefill 16K cold (seed 777) / warm (seed 1277), median tok/s | 2,040 / 2,180 | **2,195 / 2,345** | **+7.6 % / +7.6 %** |
| 16K TTFT cold / warm | 7.87 / 7.34 s | **7.32 / 6.82 s** | −0.55 / −0.52 s |
| prefill 32K cold (seed 1277) / warm (seed 1777) | 2,259 / 2,347 | **2,499 / 2,612** | **+10.6 % / +11.3 %** |
| 32K TTFT cold / warm | 14.16 / 13.66 s | **12.81 / 12.31 s** | −1.36 / −1.35 s |
| first-ever 16K prefill of the serve (seed 777 run 1, cold Engram pages) | 690 tok/s (23.2 s) | 639 (25.0 s) | same band (chunk 0 of the first prompt is cold either way) |
| C1 median / wall tok/s (6 runs, 512 tokens, distinct prompts) | 56.84 / 55.12 | 57.89 / 56.08 | +1.8 % (noise band) |
| C4 aggregate / wall (3 runs) | 123.04 / 118.04 | 122.46 / 117.57 | −0.5 % (noise) |
| corpus NLL vs `evidence/t328/default-219-8192/corpus/` | ΔNLL −0.00033 ± 0.00101, z +0.33 | **identical: ΔNLL −0.00033 ± 0.00101, z +0.33, worse/better 1,921 / 1,944** | = the T3.8b qualification to the last digit (rows are rows) |
| same-prompt C4 consistency, before / after stress | PASS / PASS | **PASS / PASS** | |
| 4 × 62K stress (seed 31) | 4/4, wall 117.6 s, min free 2,171 / 2,044 MiB | **4/4, wall 103.6 s, min free 2,171 / 2,042 MiB** | −12 % wall, same trough |
| needles (`verify.py --context 65536`, 55,206 tokens) | 6/6, TTFT 20.77 s | **6/6, TTFT 19.09 s** | −1.7 s |
| VRAM after capture (GPU0 / GPU1) | 92,778 / 92,909 MiB | 92,778 / 92,909 MiB | = |
| health (NaN / Xid / Traceback / OOM) | 0 | 0 | |
| per table per rank at 4,096 lookups: `ahead hit / miss / dropped`, `chunk-ahead hit / dropped` | 4096 / 0 / 0, 0 / 0 | **4096 / 0 / 0, 24 / 0** | 24 = 6 × 1 (16K, 2 chunks) + 6 × 3 (32K, 4 chunks): every non-first chunk was served from the slot |
| break join wait, ms per lookup (cumulative, rank 0 layer 1 + layer 14 / rank 1 layer 1 + layer 14) | 2.46 + 0.02 / 4.72 + 3.17 | 1.76 + 1.27 / 1.82 + 1.14 | rank 1's lag behind rank 0 (T3.39 re-profile's open question) is gone: 7.9 → 3.0 ms on rank 1, both ranks now equal |

Reading. The gain is the non-first chunks: at 16K one of two chunks can be prefetched (−0.55 s of the
≈ 1.1 s a cold chunk costs), at 32K three of four (−1.36 s). The remaining Engram cost is chunk 0 of every
prompt (unavoidable without the ids before the request arrives) and the worker's serial order: chunk 1's
layer-14 rows are the fourth item on the worker after chunk 0's two gathers and chunk 1's layer-1 rows, so
rank 0's layer-14 join now waits a little where it never did (0.02 → 1.27 ms per lookup cumulative) while
rank 1's layer-1 and layer-14 joins fell 4.72 → 1.82 and 3.17 → 1.14 — the slack from fetching early is what
absorbed the per-rank gather skew the re-profile saw as rank 0 spinning in NCCL. Warm gains equal cold
gains (+7.6 % both at 16K) because the page-cache `index_select` of 8192 × 2 rows (≈ 0.2 s per table) is hidden
too, not only the NVMe reads.

## Gates and decision

Go bar (T3.39 / T3.20 notes): cold-box 16K / 32K prefill wall down — **met** (−0.55 / −1.36 s); `ahead hit`
100 % — **met**; `chunk-ahead hit` > 0 on every rank — **met** (24 / 24 / 24 / 24, dropped 0); NLL unchanged
— **met** (identical to the qualification); stress 4/4, needles 6/6, same-prompt PASS before and after,
C1 / C4 within noise, health 0 — **all met**. **`ENGRAM_CHUNK_AHEAD=1` is the launcher default**; `off`
stays as the A/B value. Stress ran with the slot in use (4 concurrent 62K prompts) and passed faster; its
own hit / miss counts were not logged (the table logs at powers of two of lookups and the serve ended before
8,192), so the batched-last-chunk miss path is exercised but uncounted here.

## Files

Per serve: `launch.txt`, `ready.txt`, `pre-serve-state.txt` (`free -g`), `vram-after-capture.txt`,
`prefill-16k.*`, `prefill-32k.*` (`--warm-pass`, shard residency per pass in the JSON), `c1.*`, `c4.*`,
`corpus/` + `corpus.txt` (NLL), `consistency.*` / `consistency-post.*` (same-prompt C4 before / after stress),
`stress-kv.*`, `verify.*`, `health.txt`, `engram-stats-after-prefill.txt` (empty on both serves: captured
before the 1,024-lookup log line), `engram-stats.txt` (the per-rank `engram file table` lines at 1,024 /
2,048 / 4,096 lookups), `manifest.json`. `ab-log.txt` is the driver's console; `ab.sh` the driver.

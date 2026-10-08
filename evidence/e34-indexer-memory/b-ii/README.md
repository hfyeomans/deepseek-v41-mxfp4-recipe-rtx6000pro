# E34b-ii — block-granular candidate masks (2026-10-08)

Protocol: [`plan/EXPERIMENTS.md` E34](../../../plan/EXPERIMENTS.md), part b-ii. Task T3.26.
Overlay commit `d6811728` on top of `c639cbee` / `9fc14817`. Same profile as E34b (226 / 158,
D=4, spec off, chunk 4096, `MEM_FRACTION_STATIC=0.96`).

## The change

E34b's 262K rung died at 184K prefix allocating the candidate-source layer's published mask: a
`[chunk_rows, prefix]` bool per request (720 MiB at 184K, 1 GiB at 262K), built in per-tile parts
and `torch.cat`-ed at publish, so it existed twice. `select_candidate_blocks` already decided per
`candidate_block_size` = 8 positions and `repeat_interleave(8)`-ed the result, so the mask was 8x
redundant.

`dsv4/indexer.py` now has `select_candidate_block_keep` (the `[rows, ceil(width / 8)]` keep mask)
and `expand_candidate_blocks` (the old `repeat_interleave(8)[..., :width]`); `select_candidate_blocks`
is their composition and still serves decode, whose mask is `[bs, lmax]` and small. Both prefill
paths (`_dense_index_topk_rows`, `_low_ratio_index_topk_torch`) publish the block form and each
consumer tile expands only its own rows (a `[tile_rows, lc]` bool <= 64 MiB that dies with the
tile). The consumers assert the published block count equals `ceil(lc / 8)` for their request.
The late-layer-tail slice is by row and is unaffected.

Resident mask per 4096-row chunk: 1 GiB -> 128 MiB at 262K, 4 GiB -> 512 MiB at 1M; no
publish-time duplicate of consequence (the `cat` now copies the block form).

## Test (GPU, bit-equality)

[`scripts/tests/test_indexer_row_tiling.py`](../../../scripts/tests/test_indexer_row_tiling.py)
extended: `LC = [6001, 1500]` so the last block is partial; `test_block_expansion` checks
`expand_candidate_blocks` against an index-gather reference and that the old
`select_candidate_blocks` output equals keep -> expand on random block scores at lc 6001 / 1500 /
8 / 9 / 1; the random consumer masks are block-level; the published shapes are checked to be
`[rows, ceil(lc / 8)]`; the "non-candidate picked" check gathers by `pick // 8`. **6 / 6 pass.**
Mutant (expansion shifted by one block): 2 / 6 fail (expansion equality, and a non-candidate pick
in the random-mask consumer run). The source-published consumer run does *not* catch that mutant
because with `topk_blocks` = 2048 > 751 blocks every reachable block is kept, which is why the
random-mask run exists.

## Result (measured)

| Rung | Slots | Free after capture (GPU0 / GPU1) | (b) stress-kv | (c) needles | (d) NLL vs T3.16c | (e) health | (f) prefill 16K / 32K | C1 summed / wall / TTFT | Min free during stress | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| **262K** | 1 | 3.76 / 3.69 GB | **pass**: 253,490-token prompt + 256, TTFT 285.9 s, wall 292.0 s, peak 96,080 / 96,153 MiB; per-chunk prefill 953-962 tok/s flat from 30K to 253K prefix; decode at 253K context 42.4 tok/s | **6/6 at 221,756 tokens**, TTFT 218.0 s, `finish_reason=stop` | 0 of 4,795 moved (max abs logp 0.000) | 0 lines, EDAC 0/0, 0 Xid | **974.3 tok/s / 16.4 s** at 16K; **945.8 / 33.8 s** at 32K | 42.27 / 40.34 / 0.607 s | **1,807 / 1,734 MiB** | **pass** (E34b: OOM at 184K prefix, 643 / 248 MiB free) |
| 64K | 4 | 3.48 / 3.42 GB (= E34b) | **pass**: 4 x 62K all finished, TTFT 87.6 / 151.6 / 214.9 / 273.4 s, wall 290.6 s, peak 96,346 / 96,403 MiB | 6/6 at 55,206 tokens, TTFT 56.7 s | 0 of 4,795 moved | 0 lines, EDAC 0/0, 0 Xid | **1,040.1 / 15.5 s** at 16K; **1,040.2 / 30.8 s** at 32K (reruns; see below) | 42.17 / 40.23 / 0.610 s | **1,541 / 1,484 MiB** (E34b 1,537 / 1,478) | **pass**, unchanged |

64K x 4 prefill: the first bench pair, taken directly after the 4-stream stress, read 964.8 / 945.8
tok/s (`prefill-16k.txt`, `prefill-32k.txt`, -8 % vs E34b). Re-run two minutes later with the
server idle: 1,040.1 / 1,040.2 (`prefill-*-rerun.txt`), within 1 % of E34b's 1,048 / 1,042. The
first pair is kept as evidence of the confound (the scheduler was still freeing the four 62K
streams' KV and the probe state) and is not the number; bench prefill only on an idle server.

## Reading

- Hypothesis held: 262K / 1 slot passes gate (b) with 1.7 GB min free, the published mask is gone
  from the binder list, and the margin at 262K (1,807 / 1,734 MiB) is *larger* than at 128K on
  E34b (1,335 / 1,260) because the 128K serve ran with a smaller pool and the mask and the pad
  copy were still resident there.
- The deployment profile is untouched: min free within 4 MiB, decode within 0.1 %, prefill within
  1 %, NLL bit-identical. At 64K the mask was never the binder, so no gain was expected.
- Per-chunk prefill is flat with prefix at 1 slot (953-962 tok/s from 30K to 253K); 262K
  end-to-end TTFT is 253K / 286 s = 887 tok/s, the rest is tokenization and inter-chunk gaps.
- Decode at 253K resident context is 42.4 tok/s on the log line (the C1 bench at 4K is 42.27),
  i.e. the sparse attention's decode cost is flat with context too, as the DSA design intends.
- The next rung (512K / 1 slot) needs the KV pool and the attention working set to fit: the
  compressed KV is ~ 0.1-0.2 GB per 65K (E33), so 512K's pool is ~ 1-1.5 GB over 262K's with
  1.7 GB spare — plausible, not projected to pass. 1M is not reachable at 226 / 158 without a
  placement trade (E34c, which so far has not been needed).

## Health

Both serves: health count 0 (no NaN / probability-tensor / Xid / Traceback / KTContract / Sealed
lines), EDAC 0/0, 0 Xid in the kernel log; the NaN probe was on (`KT_NAN_PROBE_MIN_TOKENS=1`).
Containers removed after each serve; GPUs 2 / 80 MiB at the end.

Files: `262k/` and `64k-4slot/` — `launch*.txt` (host launcher and container log, redacted),
`stress-kv.{txt,json}`, `verify-*.{txt,json}`, `nll.{txt,json}`, `prefill-16k*.{txt,json}`,
`prefill-32k*.{txt,json}`, `c1-bench.txt` / `c1-off6.json`.

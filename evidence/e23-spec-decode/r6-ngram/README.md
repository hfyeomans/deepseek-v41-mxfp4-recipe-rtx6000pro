# E23 R6 — n-gram speculation (`NGRAM=1`): verify crashes on the first request; closed (2026-10-09)

Serve: `NGRAM=1 bash scripts/serve-hybrid.sh` on the 64K × 4 default (219 / 165, D=4, chunk 8192,
`NGRAM_DRAFT_TOKENS=5 NGRAM_BREADTH=1`; overlay `eeb5320c` = `1eeb48cd` + three probe commits that
only admit `NGRAM` through upstream's guards). `manifest.json` is the launcher's manifest for the
serve; no bench JSON was produced. ~4 min from launch to crash; three launches in all.

## What happened

| attempt | result | file |
|---|---|---|
| 1 | `apply_deepseek_v4_defaults` asserts `speculative_algorithm in (EAGLE, DSPARK)` | `launch-container-attempt1-guard.txt` |
| 2 | `validate_deepseek_v41_features` rejects NGRAM (V4.1 feature guard lists DSPARK only) | `launch-container-attempt2-guard.txt` |
| 3 | both guards admitted (overlay `84cd3ec4`, `eeb5320c`); multi-stream overlap forced off (`16260ad3`, Bug A gate now covers NGRAM); mixed chunk disabled; target-verify graphs captured for bs 1–4 at `num_tokens_per_req=5`, 4.75 GB free after capture; pool 278,528. **First request: both TP ranks raise `AttributeError: 'DeepSeekV4TokenToKVPool' object has no attribute 'move_kv_cache'`** from `ngram_worker.forward_batch_generation` → `spec_utils.move_accept_tokens_to_target_kvcache`; scheduler SIGQUIT | `launch.txt`, `launch-container.txt` |

EDAC 0 / 0 (`edac.txt`). No health anomaly before the crash.

## Why, and why it stays closed

Upstream's EAGLE/DSPARK-only guard on DSV4 hides a missing pool API, not a policy. The n-gram
worker (like the EAGLE tree path) compacts the accepted tokens' KV into the request's contiguous
slots through `KVCache.move_kv_cache(tgt_loc, src_loc)`. Every other SGLang pool implements it
(MHA, MLA, DSA, the hybrid GDN pools); `DeepSeekV4TokenToKVPool` does not, because its KV is not a
row-addressable table: compressed c4 / c128 rings plus a per-request SWA ring keyed by request slot,
plus the indexer cache. The EAGLE/DSpark chain path on DSV4 sidesteps the move (a chain's accepted
prefix is already contiguous) and instead calls `clear_unaccepted_c128_draft_states` (needed in our
offline-C128 configuration, `SGLANG_OPT_USE_ONLINE_COMPRESS=False`), which the n-gram worker never
calls either.

A chain-only port (skip the move when `NGRAM_BREADTH=1`, add the c128 clear) is ½–1 day with a
silent-KV-corruption risk if the cpp n-gram cache ever emits a branch — and it cannot meet the R6 go
bar, by arithmetic rather than by luck:

- the worker always forwards a full `bs × draft_tokens` verify batch, match or no match
  (`ngram_worker.py:309`, "here we always enforce it"), so every decode step becomes a 5-row step;
- on this hybrid a 5-row step costs ≈ 75–80 ms of CPU-expert bytes (E23: C1 → C4 measured 3.1× the
  bytes, step 25 → 71 ms; a verify's K+1 rows route K+1 rows' worth of distinct CPU experts);
- ceiling on a perfect copy span (5 of 5 accepted): 5 / 0.08 s ≈ 62 tok/s, about +50 % over the
  spec-off 41.4; floor on fresh text (no match): 1 / 0.08 s ≈ 12–13 tok/s, about −70 %;
- the go bar's second clause (≤ 5 % loss on the review fixture) needs the 5-row step to cost ≈ 25 ms,
  which no drafter can buy. The best possible outcome is a per-workload serve flag for copy-heavy
  traffic, not a default.

Spec-off baselines for the copy fixtures (taken on the 262K × 4 chunk-4096 serve, for a later
comparison if one is ever run): copy-code 42.38 / 39.27 wall, copy-table 42.16 / 41.70 wall,
review C1 41.44 / 39.55 (`../../e34-indexer-memory/d-v-a/262k-4slot-chunk4096/c1-copy-*.json`).

Follow-up registered as **T3.9h-b** (low): n-gram with skip-verify-on-no-match + chain-skip +
c128 clear — only if a copy-heavy product workload appears, or a CPU-tier change (T3.20 / T3.22,
third GPU) makes verify rows materially cheaper than 25 → 71 ms for 1 → 4 rows.

The three overlay probe commits stay (guards only admit NGRAM; the Bug A gate change is correct
regardless); no qualified path changes. The knobs `NGRAM` / `NGRAM_DRAFT_TOKENS` / `NGRAM_BREADTH`
stay in `serve_config.py` for T3.9h-b.

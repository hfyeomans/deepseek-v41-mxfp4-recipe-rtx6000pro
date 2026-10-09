# E34e e-i — prefill candidate publish as block-ID lists (2026-10-09)

Overlay `b6dc5173` (this change), image `dsv41-kt:recipe`, launcher default 219 / 165 + chunk 8192,
`CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962` (`launch.txt`,
`manifest.json`), BIOS slot 3, 300 W, D=4, spec off. One serve, ~75 min. Comparators: T3.29's
512K × 1 serve (`evidence/t329/512k-1slot/`, overlay `bc73ef8d`, same config and seeds) and the
e-ii trace (`../e-ii/`, same overlay as T3.29).

Question (EXPERIMENTS Part e-i): e-ii found one live tensor that scales with the prefix and
outlives the request — the candidate-source layer's published block keep mask,
`[last-chunk rows, ceil(prefix / 8)]` bool (274.9 MiB at 384K, worst 0.5 GiB at 512K and 1.0 GiB at
1M, 2× while `torch.cat` runs). Does a prefix-independent representation remove it, bit-exactly and
for free, and does the 512K peak move?

## Change (overlay `b6dc5173`)

`select_candidate_block_ids` / `expand_candidate_ids` (`dsv4/indexer.py`) replace
`select_candidate_block_keep` / `expand_candidate_blocks` on both prefill publish/consume paths
(`_dense_index_topk_rows` and the torch dequant path in `deepseek_v4_backend.py`). The published
tensor is now `[rows, min(candidate_topk_blocks, blocks)]` int32, -1 padded: 4 B × 2048 = 8 KiB per
row whatever the prefix (64 MiB at 8192 rows) instead of one byte per block of prefix per row. The
block-ID list was chosen over a packed bitset because it is smaller than a bitset above 512K and
never larger than the byte mask, and because it is the keep mask's own `topk` output (bit-exact by
construction; no density heuristic). The consumer still expands only its own tile to a per-position
mask, bounded by the score budget. The -1 entries scatter into a spare column — a clamp to 0 races
a real block-0 pick inside `scatter_` (the mutant below). Decode (`two_level_decode_logits`,
per-position `[bs, lmax]`) is untouched.

`scripts/tests/test_indexer_row_tiling.py` 7/7 in the image (`run-indexer-tiling-tests.sh`):
new `test_block_ids` compares `expand_candidate_ids(select_candidate_block_ids(...))` with the keep
path bit for bit at lc 6001 / 1500 / 8 / 9 / 1 and topk 64 / 2048, with rows that have fewer
reachable blocks than topk (so -1 padding and a kept block 0 coexist) and counts the -1 entries
against the keep mask; the source / consume tile-invariance tests now run on ID lists and check the
published shape and dtype. Mutant (clamp -1 to 0 in the scatter): 3/7 fail, `block ids` first.

## Results (rank 0 / GPU 0; rank 1 within 3 MiB)

| measure | T3.29 / e-ii (`bc73ef8d`) | e-i (`b6dc5173`) | gate |
|---|---|---|---|
| published mask live bytes at 384K / 508K (`_dense_index_topk_rows`) | 274.9 / 200.0 MiB (5,959 / 3,390 rows × prefix / 8) | **51.6 / 24.1 MiB** (6,606 / 3,090 rows × 8 KiB) | prefix-independent: yes |
| torch allocated at 384K / 508K | 88.71 / 88.64 GiB | **88.51 / 88.48 GiB** (−205 / −166 MiB) | — |
| torch reserved at 384K / 508K | 93.28 / 93.28 GiB | 93.39 / 93.39 GiB (+115 MiB; the e-i idle snapshot was taken after the 508K stress prompt, e-ii's after a 4K smoke, so the cached-transient set differs) | — |
| inactive at 508K (blocks, largest) | 4.64 GiB (220, 980 MiB) | 4.91 GiB (223, 980 MiB) | — |
| stress: 508,445-token prompt, min free VRAM | 1,091 / 1,016 MiB, TTFT 369.5 s | **1,091 / 1,018 MiB**, TTFT **332.5 s** | **≥ +0.5 GiB: not met (+0 / +2 MiB)** |
| NLL, 65,653-token corpus vs `evidence/t328/default-219-8192/corpus/` | 0 moved | **0 of 65,653 moved** (bit-identical) | 0 moved: met |
| prefill 16K / 32K (`bench-prefill.py --runs 3 --seed 777`) | 1,650 / 1,669 tok/s | **1,627 / 1,660** (−1.4 / −0.6 %) | within 5 %: met |
| needles (`verify.py --context 524288`) | 6/6 at 443,617, TTFT 267 s | **6/6 at 443,617, TTFT 238 s** | met |
| C1 (`bench-decode.py` off6, 6 runs) | 41.63 / 39.66 wall | **41.48 / 39.57 wall**, TTFT 0.618 s | within noise |
| health (`health-grep.txt`), EDAC | 0, 0/0 | **0, 0/0** | met |

## Verdict — **the tensor is gone; the 512K peak did not move; Go gate not met as written**

- The one prefix-scaling live tensor e-ii found no longer scales: 52 → 24 MiB tracks the last
  chunk's rows only (8 KiB per row), allocated is 166–205 MiB lower at the snapshots, quality and
  speed are bit-identical / within noise, and the long-prompt TTFTs are 10 % faster (369 → 333 s,
  267 → 238 s; the publish no longer concatenates prefix-wide masks per tile).
- **Min free VRAM at 508K is unchanged (1,091 / 1,018 MiB).** The e-ii reading "the mask is the 1M
  binder" was half right: at 512K the mask (≤ 275 MiB live, ≤ 0.5 GiB worst) always fit inside the
  4.7–4.9 GiB of inactive blocks the first large prefill's transients had already reserved, so it
  never set the peak. The peak is the chunk-proportional transient set — the 980 MiB
  `cutlass_fused_moe` workspace and its neighbours (E34d-v (i)). This is Part e-i's stop rule
  ("only one bounded copy exists; the growth is elsewhere") landing after the fix rather than
  before it, which is fine: the fix cost nothing and the tensor had to go before 1M anyway.
- **What e-i buys at 1M:** the mask would have been up to 1.0 GiB live plus 1.0 GiB of `torch.cat`
  transient at the first full-width chunk, on top of the 512K transient set — now ≤ 64 MiB. The 1M × 1
  prefill high-water should therefore equal 512K's (≈ 4.2 GB above the post-capture floor) instead
  of exceeding it by up to 2 GiB. Against the ≈ 4.1 GB a 1,064,960-token pool leaves after capture at
  0.962 (T3.29 262K × 4 measured 3.88 GB at 0.96), the rung is a coin flip (~0.1 GB), not the
  0.5–1.0 GB shortfall e-ii projected. It is one serve with a harmless OOM as the failure mode;
  the alternative is 212 / 172 with `ALLOW_UNTESTED_FIT=1` (~2.5 % C1 on that rung only).

Files: `launch.txt`, `manifest.json`, `stress-kv.{txt,json}`, `trace.json` + `run.out`
(`trace-prefix-memory.py --prefixes 393216,507904`, idle snapshot after the stress prompt),
`nll.txt` + `corpus/`, `prefill-{16k,32k}.{txt,json}`, `verify-512k.{txt,json}`, `c1-bench.txt` +
`c1-off6.json`, `health-grep.txt`. Raw pickles stay out of the repo (`.amp/in/e-i-snap/`).

# E34e e-ii — allocation trace of the above-262K VRAM staircase (2026-10-09)

Overlay `bc73ef8d` (unchanged), image `dsv41-kt:recipe`, launcher default 219 / 165 + chunk 8192,
`CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962` (`launch.txt`,
`manifest.json`), BIOS slot 3, 300 W, D=4, spec off. One serve, ~20 min of prompts.

Question (EXPERIMENTS Part e / e-ii): is the free-VRAM staircase above 262K allocator high-water
retention (reserved grows, allocated flat, superseded buffers held `inactive`) or real per-token
state? Go = growth identified and ≤ 256 MiB from 128K to 512K.

## Method

`scripts/trace-prefix-memory.py` (new, CI-tested in `scripts/tests/test_trace_prefix_memory.py`):
for each prefix it calls SGLang's own `POST /start_profile {"activities": ["MEM"]}`, sends one
`max_tokens=8` prompt of that length, `POST /stop_profile` (SGLang dumps
`torch.cuda.memory._dump_snapshot` per TP rank), `docker cp`s the pickles out and summarises each
device: reserved / allocated / inactive bytes, inactive block count and largest block, live bytes
per allocation site (innermost `sglang/` or `kt` frame), and a replay of the recorded window
(allocs, frees, peak and net delta). Snapshots at idle (after a 4K smoke), 128K, 256K, 384K and
508K target prefix (`--prefixes 131072,262144,393216,507904`). An `nvidia-smi` sampler at 1 Hz
wrote `gpu-during.csv`; the script also records `vram_samples` in `trace.json`. No overlay change.

Caveat: SGLang's recorder keeps 100K events (`SGLANG_MEM_PROFILE_MAX_ENTRIES`); every large prompt
overflows it, so the window `peak_delta` values are lower bounds over the *last* 100K events
(flagged `possibly_truncated`). Live-block state (reserved / allocated / inactive / per-site live
bytes) is a complete picture at snapshot time and is what the verdict rests on.

## Results (rank 0 / GPU 0; rank 1 identical within 3 MiB)

| snapshot | prompt tok | TTFT s | reserved MiB | Δ | allocated MiB | Δ | inactive MiB (blocks, largest) | nvidia-smi used MiB |
|---|---|---|---|---|---|---|---|---|
| idle (after 4K smoke) | — | — | 92,746 | — | 90,558 | — | 2,188 (194, 480) | 93,916 |
| 128K | 127,822 | 95.4 | 95,342 | **+2,596** | 90,575 | +17 | 4,767 (216, **980**) | 96,506 |
| 256K | 255,456 | 164.6 | 95,496 | +154 | 90,607 | +32 | 4,889 (216, 980) | 96,512 |
| 384K | 382,855 | 240.6 | 95,516 | +20 | 90,838 | +231 | 4,678 (223, 980) | 96,668 |
| 508K | 494,910 | 317.3 | 95,516 | **+0** | 90,764 | −74 | 4,752 (220, 980) | 96,684 |

- **128K → 508K: reserved +174 MiB, allocated +189 MiB.** Inside the ≤ 256 MiB go-gate. From
  384K to 508K reserved did not move at all.
- **The staircase is the first large prefill's transient high-water.** The +2.6 GiB step at 128K
  is 99 % `inactive` (cached, reusable) blocks: 2,188 → 4,767 MiB inactive, allocated +17 MiB.
  The largest cached block is the 980 MiB `cutlass_fused_moe` prefill workspace (E34d). Nothing is
  "superseded and still held": the inactive total is stable at 4.7–4.9 GiB from 128K on and is
  reused by every later prefill.
- **One live tensor scales with the prefix:** `deepseek_v4_backend.py:3392 _dense_index_topk_rows`,
  the `self.candidate_masks = [... torch.cat(parts)]` publish of the E34b-ii block-granular
  candidate mask, held on the backend object after the request ends. Live bytes 16.8 → 45.8 →
  274.9 → 200.0 MiB. It is `[last-chunk rows, ceil(prefix / 8)]` bool: 5,959 rows × 382,855 / 8 B
  = 272 MiB (measured 274.9); 3,390 rows × 494,910 / 8 B = 200.0 MiB (measured 200.0) — the
  508K figure is *smaller* than 384K's because its final chunk had fewer rows. Worst case
  8,192 rows: 0.5 GiB at 512K, **1.0 GiB at 1M** (Part e hypothesis (a), confirmed; owner = e-i).
- **Every other live site moves by ≤ 4 MiB** (radix-cache bookkeeping `cache_unfinished_req`,
  `redistribute_on_node_split`). The KV pool's own bookkeeping does not grow with the prefix
  (Part e stop rule not met).
- **nvidia-smi used − torch reserved = 1,170 / 1,164 / 1,168 MiB at idle / 128K / 508K** — the
  non-torch footprint (CUDA context, NCCL, kt-kernel) is constant; all growth is the torch
  caching allocator.
- Health: `health-grep.txt` 0 lines, EDAC ce/ue 0/0, 0 Xid.

## Verdict — **GO on the letter of Part e; the 1M × 1 rung is reopened but does not fit yet**

The growth above 262K is identified and bounded: one retained candidate-mask publish (≤ 1 KiB
per token of prefix at 8,192 rows) plus allocator caching of the first prefill's transients.
There is no grow-on-demand retention to fix with an arena (e-ii's second stage): reserved is flat
from 128K on, so the arena stage is struck.

What this changes for 1M: the binder was never the per-token growth. A 1M × 1 serve uses the
same 1,064,960-token pool as T3.29's 262K × 4, which left 3.88 GB after capture at 0.96 (≈ 4.1 GB
at 0.962) and OOMed on the 980 MiB workspace at ~190K prefix; the prefill high-water measured
here and in T3.29 is ≈ 4.2 GB at 508K, of which ≤ 0.5 GiB is the mask and the rest is constant
transients. At 1M the mask alone adds up to another 0.5 GiB. **Projected shortfall at
219 / 8192 / 0.962: ~0.5–1.0 GB**, i.e. the Part e expectation "needs no expert change" was wrong
by about one mask. Two fixes already in the sequence close it: e-i (packed mask, 8× smaller:
−0.9 GiB worst case at 1M) and E34d-v (i) (row-tiled fused-MoE workspace, −0.5–0.9 GiB). The
rung is scheduled behind e-i rather than served now; the alternative (212 / 172 for that rung
only, `ALLOW_UNTESTED_FIT=1`) is an owner decision and costs ~2.5 % C1 on that profile.

Files: `trace.json` (stamped; per-snapshot device state, top live sites, window replay,
`vram_samples`), `run.out` (per-snapshot site diffs), `gpu-during.csv`, `launch.txt`,
`health-grep.txt`, `manifest.json`. Raw pickles stay out of the repo (`.amp/in/e34e-ii/snapshots/`).

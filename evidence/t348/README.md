# T3.48 — user-space Engram row cache (first cut, 2026-10-23)

Overlay branch `t348-rowcache` (worktree `/tmp/sglang-t348`, commits `b3fbfe001`, `ea9ef5a5e`,
`d767598f9` on `deec7a6f`): a direct-mapped row cache per file table
(`_EngramRowCache`: 264 B rows = [dim] FP8 weights + [scale_cols] UE8M0 scales), served from RAM
on the single shared prefetch worker. Launcher knob `ENGRAM_ROW_CACHE_MB` (default 0; env
`SGLANG_DSV41_ENGRAM_ROW_CACHE_MB`, split evenly across the two tables). Hits: no NVMe read, no
WILLNEED; misses: the mmap path + insert. The launch-thread sync path bypasses the cache, so one
thread owns it (race-free by construction); a conflict overwrites the slot, and the tag always
matches the stored row, so a hit is byte-exact.

Two load-time bugs found and fixed on the way (both in this branch's history): the loader's
`with torch.device` contexts land bare `torch.empty` buffers on the GPU (fixed with explicit
`device="cpu"`; 8192 MiB then OOM'd the load at the kt_prefill staging peak), and fp8 tensors
have no `index_put` (fixed with uint8 stores + uint8-view writes; byte-identical for 1-byte
dtypes).

## Regression gates (8192 MiB cache, release-smoke.sh serve --full, 262K x 4)

**All gates passed** — same-prompt C4, NLL corpus identity (0/65,653 tokens moved vs the 1017
baseline), needles 6/6, stress 4x260K, C1 median/wall 60.14/58.23,
C4 121.56/116.72, prefill benches, power summary, health
(`evidence/t348/smoke/`). The cache adds a ~50-id lookup per step on the worker and is flat on
decode within run noise.

## The win: Engram rows survive a page-cache drop

`evidence/t348/warmth-warm.json` + `warmth-afterdrop.json` (stamped): the same 16K prompts run,
then `scripts/drop-engram-cache.py` evicts the Engram pages, and the prompts run again:

| prompt | TTFT before drop | TTFT after drop | pages loaded after drop |
|---|---|---|---|
| real-16384-0 (16,210 tok) | 3.503 s | 3.488 s | **0** |
| real-16384-1 (18,103 tok) | 4.828 s | 4.821 s | **0** |
| synthetic-16384 (16,033 tok) | 3.434 s | 3.414 s | **0** (first contact was 5.701 s, 2.47 GiB from NVMe) |

Control (no cache, `evidence/t349/control-cold/`): the same drop leaves the prompts cold at
5.56 / 6.74 / 5.57 s (+1.7-2.0 s). **The 8 GiB row cache therefore removes the ~2 s cold cost
for returning content - the T3.48 projection, confirmed.** The cache's 264 B rows are ~22x
denser than the page cache, so an 8 GiB budget holds the whole served working set while the
page cache may be evicted under host pressure.

## Verdict

**Works and is the real TTFT lever for repeated content** (coding-agent loops that re-send
context, re-reviews, corpus-heavy traffic): −2.1 s (−38 %) on a 16K synthetic after a page-cache
drop. Cost: RAM (the knob's budget, e.g. 8192 MiB) and none on decode/NLL. Stays **off by
default** (RAM budget is an owner call per deployment); a coding-agent profile could set it on.
First-contact rows still pay the 4 KiB page fault (row-granular pread would be the follow-up if
ever needed).

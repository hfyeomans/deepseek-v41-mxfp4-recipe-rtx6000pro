# T3.29 — context ladder re-qualified on the 219 / 165 + chunk 8192 default (2026-10-08)

Overlay `bc73ef8d` (T3.19d, unchanged), image `dsv41-kt:recipe`, BIOS slot 3 (FCLK 2100), 300 W,
D=4, spec off, `KT_NAN_PROBE_MIN_TOKENS=1`, `ALLOW_UNTESTED_FIT=1` (the launcher guard at the time
required 226 / 4096 above 64K). One serve per rung, E33 gates (a)–(f) plus the T3.28 pooled
corpus (`nll-score.py --corpus`, 65,653 tokens, same-shape vs `evidence/t328/default-219-8192/corpus/`
so 0 moved is the expected reading).

## Why

T3.28 moved the default to 219 / 165 + chunk 8192 but the 128K / 262K / 512K × 1 rungs had only been
qualified at 226 / 4096 (E34b, E34b-ii, 512K rung). Chunk 8192 adds ~2.2 GB of prefill transient
and seven fewer GPU experts return ~2.3 GB (0.34 GB per expert per rank, E28), so the projection was
"neutral at 128K / 262K, tight at 512K". Both the projection and its sign were wrong: the new default
has *more* headroom on every rung and faster long-prompt TTFT.

## Results

| rung | config | stress (context − 2K) | needles | NLL (corpus) | prefill 16K / 32K tok/s | C1 summed / wall | health | free after capture | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `262k-4slot/` | 262K × 4, fraction 0.96 | **CUDA OOM at ~190K prefix of the first request** (980 MiB `cutlass_fused_moe` workspace, other three requests still queued) | — | — | — | — | 0 NaN / Xid before the OOM | 3.88 GB, pool 1,064,960 tokens | **fail (b)** |
| `262k-1slot/` | 262K × 1, fraction 0.96 | 253,242-token prompt, TTFT **193 s** (226 / 4096: 286 s), min free 1,669 / 1,596 MiB | 6/6 at 221,756 tokens, TTFT **133 s** (was 218 s) | 0 of 65,653 moved | **1,400 / 1,493** (runs 12.28 / 11.47 / 11.24 s at 16K) | 41.48 / 39.53, TTFT 0.634 s | 0, EDAC 0/0 | 5.87 GB, pool 278,528 | **pass** |
| `1m-1slot/` | 1M × 1, fraction 0.962, overlay `b6dc5173` (after E34e e-i), run 2026-10-09 | **CUDA OOM at ~306K prefix (37th chunk)** on the 980 MiB `cutlass_fused_moe` workspace: 499 MiB free, 1.87 GiB reserved-but-unallocated (fragmented), 48 MiB graph pools; prefill 1,600–1,694 tok/s per chunk until then | — | — | — | — | 0 NaN / Xid before the OOM, EDAC 0/0 | 4.19 / 4.11 GB, pool 1,064,960 tokens | **fail (b)** |
| `../e34-indexer-memory/d-v-i/262k-4slot/` | 262K × 4, fraction 0.96, overlay `2323052b` (E34d-v (i)), run 2026-10-09 | **CUDA OOM on the first 8192-token chunk** (`kt_prefill.load` 234 MiB after 320 / 400 / 640 MiB retries; 818 MiB reserved-but-unallocated): the resident workspace leaves 2.92 / 2.86 GB after bs-4 capture, the remaining chunk-8192 transients need ≈ 3 | — | — | — | — | 0 NaN / Xid, EDAC 0/0 | 2.92 / 2.86 GB, pool 1,064,960 tokens | **fail (b)** |
| `../e34-indexer-memory/d-v-i/1m/` | 1M × 1, fraction 0.962, overlay `2323052b` (E34d-v (i)), run 2026-10-09 | **CUDA OOM at ≈ 680K prefix** (was ~306K) on a 332 MiB int64 index tensor in `get_index_k_fp4` (512 B per prefix token), 301 MiB free, 1.20 GiB reserved-but-unallocated; prefill 1,500 → 1,150 tok/s per chunk until then | — | — | — | — | 0 NaN / Xid, EDAC 0/0 | 3.25 / 3.17 GB, pool 1,064,960 tokens | **fail (b)**, next binder E34d-v (ii) |
| `../e34-indexer-memory/d-v-ii/1m/` | 1M × 1, fraction 0.962, overlay `1eeb48cd` (E34d-v (ii)), run 2026-10-09 | **1,019,251-token prompt completed**, TTFT **803 s** (≈ 1,269 tok/s averaged over the prompt), min free **547 / 574 MiB** (two earlier serves on the same day OOMed at ≈ 298K and ≈ 860K on the next two chunk-proportional transients, see that README) | 6/6 at 887,387 tokens, TTFT **550 s** | 0 of 65,653 moved | **1,659 / 1,657** | 41.31 / 39.41, TTFT 0.617 s | 2 benign sealed-buffer lines, 0 NaN / Xid, EDAC 0/0 | 3.25 / 3.18 GB, pool 1,064,960 | **pass — ladder top; per-rung override, not a default** |
| `../e34-indexer-memory/d-v-a/262k-4slot/` | 262K × 4, fraction 0.96, overlay `1eeb48cd`, chunk 8192, run 2026-10-09 | **CUDA OOM on the third 8192-token chunk of the first request** (320 MiB `mhc_post` output, 266 MiB free, 1.05 GiB reserved-but-unallocated — fragmentation) | — | — | — | — | 0 NaN / Xid, EDAC 0/0 | 2.92 / 2.86 GB, pool 1,064,960 | **fail (b)** |
| `../e34-indexer-memory/d-v-a/262k-4slot-chunk4096/` | 262K × 4, fraction 0.96, overlay `1eeb48cd`, **chunk 4096**, run 2026-10-09 | 4 × ~253K prompts all finished, wall **1,082 s**, TTFT 289 / 810 / 1,064 / 550 s, min free **877 / 738 MiB** | 6/6 at 221,756 tokens, TTFT 216 s | pooled ΔNLL −0.00114 ± 0.00097 (shape change) | **975 / 975** | 41.44 / 39.55, TTFT 0.618 s | 0, EDAC 0/0 | 3.65 / 3.59 GB, pool 1,064,960 | **pass** (first multi-slot rung above 64K) |
| `512k-1slot/` | 512K × 1, **fraction 0.962** | 508,445-token prompt, TTFT **369 s** (was 596 s), min free **1,091 / 1,016 MiB** (was 713 / 638) | 6/6 at 443,617 tokens, TTFT **267 s** | 0 of 65,653 moved | **1,650 / 1,669** (9.71 / 9.74 / 9.71 s; 19.17 / 19.28 / 19.11 s) | 41.63 / 39.66, TTFT 0.633 s | 0, EDAC 0/0 | 5.32 GB, pool 540,672 | **pass** |

128K × 1 was not run: its pool (147,456 tokens) is smaller than 262K × 1's and its stress prefix
shorter, and every binder found on this ladder grows with the pool or the prefix. Treated as implied
by the 262K × 1 pass; a 128K serve is one command if that inference is challenged.

Comparators: 262K × 1 at 226 / 4096 is `evidence/e34-indexer-memory/b-ii/` (TTFT 286 s, needle 218 s,
min free 1,807 / 1,734 MiB, prefill 974 / 946); 512K × 1 at 226 / 4096 + 0.962 is
`evidence/e34-indexer-memory/b-ii/512k/` (TTFT 596 s, min free 713 / 638, prefill 943 / 943); 64K × 4
on the same default leaves 5.50 GB after capture (E34d-ii).

## What the numbers say

- **TTFT at 253K / 508K drops 32 % / 38 %** on the new default. The per-chunk prefill rate is what
  moved (chunk 8192 halves the chunk count and the fixed per-chunk cost: router, barrier, Engram,
  indexer tiles); DSA decode is unchanged (C1 39.5–39.7 wall, same as the 64K × 4 reading).
- **Headroom went up, not down.** 262K × 1 leaves 5.87 GB after capture against 3.76 GB at 226 / 4096
  (`e34-indexer-memory/b-ii/262k/`, same 278,528-token pool; "Memory pool end" 6.39 vs 4.18 GB). Seven
  experts (2.3 GB) buy more than the chunk-8192 transient costs at rest; the transient only shows
  during prefill, where min free is 1.67 / 1.60 GB at 262K — 140 MiB less than 226 / 4096 at the same
  prefix — and 1.09 / 1.02 GB at 512K, **380 MiB more** than 226 / 4096 had.
- **E34e growth flattens.** On the 512K run free VRAM went flat from ~251K to ~440K of prefix and then
  fell again; it is not linear in prefix as the E34b-ii / 512K readings (1,807 → 783 → 713 MiB at
  253K / 503K / 522K) suggested. Still unidentified; still the 1M binder.
- **262K × 4 fails on pool size, not on concurrency.** The OOM is in the first request's prefill
  while the other three are queued, so a 4-slot pool at 262K (1.06M tokens, 3.88 GB left after
  capture) does not leave room for the chunk-proportional fused-MoE workspace + the E34e transient
  (~4.2 GB at 190K prefix). Any config whose pool leaves ≤ ~4 GB after capture is projected to fail
  the same way at a prefix of ~190K regardless of slot count.

## Launcher change

`scripts/serve-hybrid.sh` above 64K now (1) refuses `MAX_RUNNING_REQUESTS > 1` and (2) accepts only
219 / 8192 or 226 / 4096, both overridable with `ALLOW_UNTESTED_FIT=1`. The old guard required
226 / 4096 above 64K. `DRY_RUN=1` prints the resolved line after every guard and exits before any
docker call (added after a guard probe on this task launched a real server, which was removed
before the first rung). `scripts/tests/test-serve-config.sh` (11 DRY_RUN cases, in `ci.sh`) pins the
accepted and refused profiles.

## Default-context candidates (projections, not measured)

The deployment default stays 64K × 4. If the owner wants a larger default, the candidates that share
the 512K × 1 pool size (540,672 tokens, fraction 0.962, 5.32 GB after capture) are **128K × 4** (keeps
the C4 gate shape) and **262K × 2**; both are projected to pass with ~1 GB of stress margin and each
is one serve to confirm. 262K × 4 needs either ≥ 7–10 fewer GPU experts (−2 to −3 % more decode) or
the E34d-v / E34e memory fixes.

## Files

- `*/launch.txt`, `*/launch-container.txt` — launcher and container logs (`resolved:` line,
  `max_total_num_tokens`, `available_gpu_mem`; health grep 0 hits on the passing rungs).
- `*/stress-kv.{txt,json}` — `stress-kv-full.py --concurrency 1 --prompt-tokens <ctx − 2048>`; peak VRAM
  samples and min free per GPU. `262k-4slot/stress-kv.txt` is the HTTP 500 traceback.
- `*/verify-*.{txt,json}` — `verify.py --context <ctx>`: alias, ceiling, arithmetic, thinking toggle,
  tool call, 6 needles.
- `*/nll.txt`, `*/corpus/` — pooled corpus vs `evidence/t328/default-219-8192/corpus/`.
- `*/prefill-{16k,32k}.{txt,json}` — `bench-prefill.py --runs 3 --seed 777` on the idle server.
- `*/c1-bench.txt`, `*/c1-off6.json` — `bench-decode.py --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242`.
- `262k-4slot/pre-serve-state.txt` — page cache / swap state before the serve (T3.19d finding; the
  1-slot rungs were run back to back on the same warm box).

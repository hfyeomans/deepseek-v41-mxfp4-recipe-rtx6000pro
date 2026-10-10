# T3.39 window 1 — where the 37 ms per layer-chunk of GPU idle in a 16K prefill goes (2026-10-10)

Two serves on the qualified profile (overlay `ad973440`, launcher defaults 219 / 165, D=4, chunk 8192, 64K × 4,
`ENGRAM_FILE_PREFETCH=ahead`, expandable-after-capture on), one per `KT_CPUINFER` value — **16** (the default,
one worker per physical core of the 9955WX) and **14** (the H2 probe: two cores left for the launch threads and
the Engram worker). Each serve: one warm 16K pass (seed 777, the serve's first prefill) → two unprofiled 16K
passes (seed 1277) → `/start_profile` with **CPU + GPU** activities → one 16K pass (seed 1777) → `/stop_profile`.
Driver and analysis scripts are in `window1/analysis/`; traces, bench outputs, launch / ready / manifest in
`window1/cpuinfer16/` and `window1/cpuinfer14/`.

**Result.** Both hypotheses are real, but they explain different parts of the TTFT.

* **H1 confirmed — the per-layer idle is the host blocked in kt-kernel's export.** Inside the layer loop, every
  layer-chunk has one GPU gap of 25–35 ms (after `quantize_with_block_size`, before the resident `cutlass`
  grouped GEMM) during which **no `cpu_op` is running on the launch thread at all** — the host is inside a
  non-aten call, i.e. the kt pybind `sync_write_weight_scale_to_buffer` wait. 2,137 ms of such "no host op"
  gap time at 16 workers, 1,782 ms at 14; the layer period is **90.2 vs 90.3 ms (unchanged)**. Fewer workers do
  not shorten the export wait, so the in-loop idle is not scheduler starvation — it is the export itself.
* **H2 real, but it lives at the chunk start, not in the loop.** `KT_CPUINFER=14` is **+9.4 % on 16K prefill**
  (1,659 → 1,815 tok/s unprofiled, 1,637 → 1,866 profiled). The whole gain is the **Engram chunk-start break**:
  after the hasher's one D2H of the chunk's hash ids (`_engram_file_prefetch`, 1.5 MB `[187584]` int64 = 7,816
  tokens × 24 ids), the GPU idles until the first Engram layer's 48 MB pinned H2D arrives. That gap was
  **400 / 1,459 ms** at the two chunk starts with 16 workers and **342 / 319 ms** with 14 (rank 0; rank 1's
  chunk-2 gap was 793 ms at 16, then it waited 669 ms in the all-reduce for rank 0). Chunk-start idle 1,859 ms
  vs 661 ms = the 1.2 s TTFT difference between the two profiled runs. At 16 workers the page-cache gather /
  `MADV_WILLNEED` worker competes with the 16 export threads for the 16 cores (the launch thread's `aten::item`
  also stalls up to 134 ms at 16 vs 15 ms at 14).

So the **90 ms layer period** (81 % of TTFT) is export-bound as T3.21 said, and the fix for it is still window 2
(export-free streaming). The 14-worker knob is a separate, cheap **+9 %** that is not yet a default: E26 A1
measured 14 and 16 as a decode plateau, but that was C1; the C4 / stress / 64K × 4 soak for a default change has
not been run (owner decision).

## Numbers

16K prefill (`scripts/bench-prefill.py --tokens 16384`), one serve per column, same seeds:

| pass | `KT_CPUINFER=16` tok/s (TTFT s) | `KT_CPUINFER=14` tok/s (TTFT s) | delta |
|---|---|---|---|
| warm, seed 777 (first prefill of the serve, cold Engram pages) | 642 (24.89) | 600 (26.62) | n/a (cold box) |
| unprofiled, seed 1277, run 1 / run 2 | 1,577 / 1,741 (10.13 / 9.18) | 1,787 / 1,842 (8.93 / 8.68) | **+9.4 % on the median** (1,659 → 1,815) |
| profiled (CPU + GPU activities), seed 1777 | 1,637 (9.78) | 1,866 (8.58) | +14 % |

Trace accounting, rank 0 (`analysis/gaps.py`, `analysis/timeline.py`; GPU events = kernels + memcpy + memset,
all on one stream):

| | 16 workers | 14 workers |
|---|---|---|
| trace span (first GPU event → last) | 9,739 ms | 8,539 ms |
| GPU busy | 5,043 ms | 5,073 ms |
| GPU gaps ≥ 1 ms, total | 4,618 ms | 3,394 ms |
| layer period (w13 HtoD → next w13 HtoD), median / p10 / p90, n = 77 | **90.2** / 87.8 / 98.4 ms | **90.3** / 88.3 / 96.5 ms |
| in-loop gaps ≥ 1 ms (period minus chunk starts), ≈ per layer-chunk | ≈ 2,697 ms ≈ 35 ms | ≈ 2,667 ms ≈ 35 ms |
| of which gaps ≥ 15 ms with **no `cpu_op` at all** (host in the kt export wait) | 2,137 ms | 1,782 ms |
| chunk-start gap (Engram ids D2H → first Engram rows H2D), chunk 1 / chunk 2 | 400 / **1,459** ms | 342 / 319 ms |
| longest launch-thread host op | `aten::item` 134 ms (then 24, 20) | `aten::item` 15 ms (then 14, 11) |

What the chunk-start gap is (`analysis/boundary.py` on the 4,580–6,260 ms window at 16, 4,310–5,060 at 14):
the last w13 HtoD of chunk 1 → the layer's all-reduce → `Memcpy DtoH (Device → Pageable)` 1,500,672 B (the
`hash_ids.to("cpu")` in `_engram_file_prefetch`) → idle → `Memcpy HtoD (Pinned → Device)` 48,021,504 B +
1,500,672 B (the first Engram table's rows and scales, `_engram_file_rows`) → all-reduce → the first dense GEMM of
chunk 2 → w13 HtoD of layer 0. The only host ops in the window are a `gloo:barrier` on the TP thread and the
4.6 ms `aten::to`; the rest is the Engram worker's `MADV_WILLNEED` + `index_select` gather, which runs in a
Python thread that the 16 export workers out-schedule.

## What this buys

* Window 2 (export-free streaming, TASKS T3.39) targets the ≈ 28–35 ms per layer-chunk of export wait =
  2.1–2.7 s of a 16K TTFT; ceiling unchanged (≈ 52 ms period serialised, ≈ +70 %).
* `KT_CPUINFER=14` is a +9 % 16K prefill candidate with zero code; it needs the C1 / C4 / stress soak before a
  default flip (E26 A1 says decode is flat between 14 and 16). If window 2 lands, the export threads stop
  competing with the Engram worker during the gather only if the export is gone, so the two are complementary,
  not alternatives.
* The chunk-start Engram gather is itself a target: 0.3–0.4 s per 8192-chunk even at 14 workers (6–8 % of a
  16K TTFT; proportionally the same at 262K). It is one Python thread doing `index_select` out of a page-cached
  mmap; issuing the `WILLNEED` for chunk *n+1* during chunk *n*'s layers (the ids are known once the hasher has
  run on the next chunk's tokens — they are not, today, until the next forward starts) would hide it. Registered
  as a T3.20 follow-up, not scheduled.

## Files

| path | what |
|---|---|
| `window1/analysis/profile.sh` | the driver (serve → warm → unprofiled ×2 → profile → stop → copy, per `KT_CPUINFER`) |
| `window1/analysis/gaps.py` | span / busy / gap totals, layer period, "no cpu_op" gap time, longest host ops |
| `window1/analysis/timeline.py` | chunk boundaries and every ≥ 15 ms gap with its bracketing GPU events and overlapping host op |
| `window1/analysis/boundary.py` | host ops and GPU events inside one time window |
| `window1/cpuinfer{16,14}/*-TP-{0,1}.trace.json.gz` | torch-profiler traces (CPU + GPU, no stacks), one per rank |
| `window1/cpuinfer{16,14}/prefill-16k-{warm,unprofiled,profiled}.{json,txt}` | bench outputs |
| `window1/cpuinfer{16,14}/{launch,ready,start-profile,stop-profile,prefill-log}.txt`, `manifest.json` | serve provenance |

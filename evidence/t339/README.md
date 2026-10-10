# T3.39 window 1 + 1b — where the 37 ms per layer-chunk of GPU idle in a 16K prefill goes (2026-10-10)

Two serves on the qualified profile (overlay `ad973440`, launcher defaults 219 / 165, D=4, chunk 8192, 64K × 4,
`ENGRAM_FILE_PREFETCH=ahead`, expandable-after-capture on), one per `KT_CPUINFER` value — **16** (the default,
one worker per physical core of the 9955WX) and **14** (the H2 probe: two cores left for the launch threads and
the Engram worker). Each serve: one warm 16K pass (seed 777, the serve's first prefill) → two unprofiled 16K
passes (seed 1277) → `/start_profile` with **CPU + GPU** activities → one 16K pass (seed 1777) → `/stop_profile`.
Driver and analysis scripts are in `window1/analysis/`; traces, bench outputs, launch / ready / manifest in
`window1/cpuinfer16/` and `window1/cpuinfer14/`.

**Result.** Both hypotheses are real, but they explain different parts of the TTFT.

* ~~**H1 confirmed — the per-layer idle is the host blocked in kt-kernel's export.**~~ **Corrected by window 1b
  (below): the gap is the host inside FlashInfer's `getWorkspaceInfo`, recomputing the MoE workspace size on
  every call because the hybrid alternates two expert counts on one C++ runner; the kt export wait is only
  6–17 ms per layer and hides behind the GPU.** The trace reading that follows is still accurate as a trace
  reading — it just named the wrong non-aten callee. Inside the layer loop, every
  layer-chunk has one GPU gap of 25–35 ms (after `quantize_with_block_size`, before the resident `cutlass`
  grouped GEMM) during which **no `cpu_op` is running on the launch thread at all** — the host is inside a
  non-aten call, ~~i.e. the kt pybind `sync_write_weight_scale_to_buffer` wait~~ which py-spy identifies as
  the FlashInfer workspace-size recompute, not the kt export. 2,137 ms of such "no host op"
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

~~So the **90 ms layer period** (81 % of TTFT) is export-bound as T3.21 said, and the fix for it is still window 2
(export-free streaming).~~ Window 1b: the period was host-bound on a FlashInfer cache miss, not export-bound; with
one runner per expert count the 16K prefill is 1,659 → 2,059 tok/s on the same seeds, and window 2's ceiling
has to be re-derived from a new profile. The 14-worker knob is a separate, cheap **+9 %** that is not yet a default: E26 A1
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

## Window 1b — the 27–42 ms host stall per layer-chunk was FlashInfer, not the kt export (2026-10-10)

Window 1 read "no `cpu_op` on the launch thread" as "host blocked in the kt export wait" (H1) because the export is
the only thing the layer loop is known to wait on. That was an inference, not an observation, so before building
window 2 on it the host side was measured directly: first by timing the layer period's host anatomy from the
window-1 traces (`analysis/period*.py`, `analysis/gpugaps.py`: the export join is **6–17 ms** per layer, not
25–35), then with `py-spy record --native` on the TP0 scheduler during a steady 16K prefill on the qualified
profile (`diag/`; the sampled bench itself read 1,316 tok/s because 250 Hz native unwinding slows the process).

**Result: 3,083 of 3,521 main-thread samples (88 %) were inside
`FusedMoeRunner::runMoe → getWorkspaceInfo → MoeGemmRunner::getMaxWorkspaceSize → calcMaxWorkspaceSize →
dispatchMoeGemmSelectClusterShapeTmaWarpSpecialized → TllmException ctor / getTrace`** — FlashInfer 0.6.18's
CUTLASS MoE runner recomputing its workspace size by trying every TMA warp-specialized config and catching one
exception, with a symbolized backtrace, per config SM120 does not support. `getMaxWorkspaceSize` caches the
answer for *one* expert count (`num_experts_`), and FlashInfer keeps one C++ runner per dtype/quant-mode key
(`MoERunner.runner_dict`). The hybrid prefill calls the fused MoE twice per layer on that one runner — the
streamed CPU set (165 experts) and the resident set (219) — so every call missed. Microbench in the image
(`analysis/moe_alt.py`): same count every call 0.1 ms host; alternating 165 / 219 **14.1 / 14.2 ms**;
alternating with a separate C++ runner per count 0.1 ms. The serve's stalls were larger (27–42 ms) — the
backtrace symbolization is slower in the big process and competes with the 16 kt workers (not measured
separately).

**Fix (overlay `99128a7e`, `SGLANG_FLASHINFER_MOE_RUNNER_PER_EXPERT_COUNT=1`, default on):** one C++
`FusedMoeRunner` per (instance key, expert count), swapped into `runner_dict` before each call; the first count
keeps FlashInfer's own runner, the second is built from the same module (`analysis/pin_check.py` in the image:
alternating host 14.4 / 14.5 → 0.1 / 0.1 ms, outputs bit-identical with and without the pin). The serve log
shows the dedicated 165-expert runner built on both ranks at the first prefill.

One serve, launcher defaults otherwise (219 / 165, D=4, chunk 8192, 64K × 4, `KT_CPUINFER=16`, `ahead`):

| pass | window 1, `KT_CPUINFER=16`, overlay `ad973440` | **window 1b, overlay `99128a7e`** | delta |
|---|---|---|---|
| 16K warm, seed 777 (serve's first prefill, cold Engram pages) | 642 tok/s (24.89 s) | 675 (23.67 s) | n/a (cold box) |
| 16K unprofiled, seed 1277, run 1 / run 2 | 1,577 / 1,741 (10.13 / 9.18 s) | **2,024 / 2,093 (7.89 / 7.64 s)** | **median 1,659 → 2,059, +24 %** |
| 32K, seed 2277, 2 runs (vs the E35 control / T3.20 r2 on `ad973440`) | 1,620–1,666 / 1,910 | **2,192 (14.59 s)** | **+32 % / +15 %** |
| C1 / C4 decode (`bench-decode.py`, offset 6, 512 tokens) | 41.9 / 56.7 summed | 42.7 / 57.0 summed, 40.7 / 54.8 wall | unchanged (decode never alternates counts) |
| `verify.py --context 60000` | — | ALL PASS, needles 6/6 at 50,543 tokens | — |
| py-spy `--native`, TP0, 5 s during a 16K prefill (`pyspy-fixed.raw.gz`) | `getWorkspaceInfo` 80 % of samples (`diag/pyspy-stale.raw.gz`) | **0 samples**; residue = Engram `madvise` / `index_select` worker 17 %, kt export wait 6 %, `select` 7 % | — |

~~Not run on this serve: the 4 × 62K stress, NLL and same-prompt gates — **`99128a7e` is not yet the qualified
profile**.~~ **Qualification serve (2026-10-10, `qual-99128a7e/`, driver `qual-99128a7e/qual.sh`, launcher defaults,
`KT_CPUINFER=16`): every gate met.**

| gate | `ad973440` (T3.20 r2b / r2-stress) | **`99128a7e`** |
|---|---|---|
| 16K prefill cold / warm (seed 777 / 1277, `--warm-pass`) | 1,811 / 1,870 tok/s (TTFT 8.87 / 8.54 s) | **2,040 / 2,171 (7.87 / 7.37 s)** |
| 32K prefill cold / warm (seed 1277 / 1777) | 1,910 / 1,929 (16.75 / 16.57 s) | **2,272 / 2,353 (14.09 / 13.63 s)** |
| C1 summed / wall | 41.94 / 39.96 | 42.41 / 40.43 |
| C4 summed / wall | 56.66 / 54.46 | 55.75 / 53.63 |
| NLL vs the T3.28 corpus | 0 of 65,653 moved | **0 of 65,653 moved** |
| same-prompt C4 | PASS | PASS (before and after stress) |
| 4 × 62K stress | 4/4, wall 165.2 s, min free 2,161 / 2,122 MiB | **4/4, wall 124.0 s, min free 2,149 / 2,022 MiB** |
| VRAM after capture | 92,778 / 92,835 MiB | 92,778 / 92,837 |
| needles | 6/6 at 55,206 | 6/6 at 55,206 (TTFT 20.6 s) |
| health | 0 | 0 |

The stress wall drops 25 % because the four 62K prefills are the thing that got faster. GPU 1's stress trough is
100 MiB lower; GPU 0's is 12 MiB lower — within the day-to-day spread of this gate (T3.31 soak 2,179 / 2,122,
round 1 2,203 / 2,144) and still ≥ 2 GiB. ~~The `KT_CPUINFER=14` rider goes on the next serve.~~

### `KT_CPUINFER=14` soak on `99128a7e` — withdrawn (2026-10-10, `cpuinfer14-soak/`, driver `soak14.sh`)

Same gates, same seeds, both serves' benches at 0.01 % Engram residency:

| | `99128a7e`, 16 workers | `99128a7e`, **14 workers** |
|---|---|---|
| 16K prefill cold / warm | 2,040 / 2,171 | **1,523 / 1,934 (−25 / −11 %)** |
| 32K prefill cold / warm | 2,272 / 2,353 | **1,960 / 2,047 (−14 / −13 %)** |
| C1 summed / wall | 42.41 / 40.43 | 42.87 / 40.68 |
| C4 summed / wall | 55.75 / 53.63 | 56.14 / 53.78 |
| NLL | 0 moved | 0 moved |
| 4 × 62K stress | 4/4, 124.0 s, min free 2,149 / 2,022 | 4/4, **148.4 s**, 2,173 / 2,046 |
| needles / same-prompt / health | 6/6 / PASS / 0 | 6/6 / PASS / 0 |

Window 1's +9.4 % at 14 workers was measured with the FlashInfer stall holding the layer period at 90 ms, where
the export was hidden and the only exposed host work was the Engram chunk-start gather. With the stall gone the
period is shorter and kt's core-bound export (E22a: 3.3 GB per layer at ≈ 52 GB/s on 16 cores) is the term on
the critical path; 14 workers make it 16/14 longer, and the gather's gain cannot pay for that. Decode is flat
(E26 A1's plateau). `KT_CPUINFER` stays 16; the next step is a fresh CPU + GPU profile of one chunk on
`99128a7e` to size window 2 (export-free streaming), which is now the exposed term rather than a stale ceiling.

What this changes upstream of window 2: the layer period's host side is no longer 90 ms, so the pipeline is
now GPU-side (≈ 52 ms: 29 HtoD + 3 fix-up + ≈ 20 compute) plus whatever the 6–17 ms export join still exposes.
Window 2's ceiling (≈ +70 % serialised) was derived from the 90 ms period and is stale; re-profile (GPU + CPU,
one chunk) on `99128a7e` before scoping it. The two lessons: a "no host op" gap in a torch-profiler trace says
the host is in *some* non-aten native call, not which one — sample the native stack before attributing it; and
a C++ library that caches by one key silently thrashes when the call pattern alternates two keys.

## Re-profile on `99128a7e` — what a 16K TTFT is made of now (2026-10-10, `reprofile-99128a7e/`, driver `analysis/reprofile.sh`)

One serve on the qualified profile (overlay `99128a7e`, launcher defaults, `KT_CPUINFER=16`), window 1's
sequence: warm 16K seed 777 → two unprofiled seed 1277 → CPU + GPU profiler on one seed-1777 16K pass. The box
was cold for Engram (both shards **3 %** resident at launch, 2.9 GB swap in use from before the serve; the
qualification serve read 26 %), so the Engram terms below are a cold-box reading. Analyses: window 1's `gaps.py`,
window 1b's `period.py`, and two new scripts (`analysis/regaps.py` — big gaps with timestamps, per-period idle,
in-loop gap boundaries; `analysis/rebusy.py` — GPU time per period by class, copy / kernel overlap, stream ids,
whole-trace class totals per rank); their outputs are beside them as `*-tp0.txt` / `*-tp1.txt`.

| pass | tok/s (TTFT s) |
|---|---|
| warm, seed 777 (serve's first prefill, cold pages) | 649 (24.63) |
| unprofiled, seed 1277, run 1 / run 2 | 2,062 / 2,093 (7.74 / 7.64) |
| profiled (CPU + GPU activities), seed 1777 | 2,131 (7.51) |

**Layer period 90.3 → 65.0 ms, and it is now export-bound, not host-bound.** Both ranks: median **65.0 ms**
(p10 61.8, p90 66.4, n = 75). Inside a period the GPU is busy **56.0 ms** in one contiguous stretch from the w13
HtoD to the resident MoE's last kernel: **29.2 ms of HtoD** (w13 973 MB 17.1 ms + w2 487 MB 8.5 ms + scales 3.2 ms
+ 0.4 DtoD — 1.64 GB at 57 GB/s, the PCIe 5 ×16 roof) and **26.8 ms of kernels**, with **zero overlap** between
the two: `_StreamedExpertSlots.load` issues the four copies on the compute stream (all 80 × (w13 + s13 + w2 + 2 × s2)
streamed-weight memcpys are on stream 30 with the kernels; the side streams 74 / 343–348 carry only the small
Engram / kt staging copies). Then
one **≈ 9 ms gap** (median 8.9, p10 1.2, p90 10.9; 67 of 75 periods; 552 ms per rank) between the resident MoE's
`finalizeMoeRoutingKernel` and the next layer's w13 HtoD. The host is busy only 3.9 ms per period and idle in one
53 ms stretch from offset ≈ 8 ms — blocked in the next layer's export join. So period = max(GPU 56, kt export
≈ 65) = 65: the core-bound export (E22a: 3.3 GB per layer at ≈ 52 GB/s ≈ 68 ms) is on the critical path, which is
what the `KT_CPUINFER=14` soak showed from the other side (16/14 longer export, −11 to −25 % prefill).

Top GPU items per period (rank 0; rank 1 identical to ±0.1 ms): HtoD pinned 28.8 ms, NCCL all-reduce 5.3,
cutlass grouped GEMM (resident + streamed experts) 4.3, `_w8a8_block_fp8_matmul` 3.9, elementwise 3.4 (mostly
the w13 gate / up half swap after the copy), `sparse_mla_prefill` 3.0, `_hc_mix_stats_partial` 1.5, `mhc_post` 0.9,
`finalizeMoeRouting` 0.4, `block_scale_interleave` 0.3.

**Where the 7,482 ms trace span (TTFT 7.51 s) goes, rank 0 = the critical path:**

| term | ms | share | note |
|---|---|---|---|
| streamed-expert HtoD, serialised with compute | 2,309 | 31 % | w13 1,366 + w2 683 + scales 260; 80 layer-chunks × 29 ms |
| kernels (all classes, real all-reduce ≈ 418 as on rank 1) | 2,235 | 30 % | 80 × 27 ms + chunk-start / hasher work |
| **NCCL spin waiting for rank 1** | **1,549** | **21 %** | four all-reduces of 304 / 474 / 302 / 443 ms — each right after an Engram row fetch |
| GPU gaps ≥ 1 ms | 1,321 | 18 % | Engram chunk-start 347 + 307; export-bound in-loop 552; first layer of each chunk waiting for its export 52 + 54 |
| sub-ms gaps | 68 | 1 % | |

The Engram term is bigger than it looks from rank 0's own gaps. Per chunk the one post-hasher break submits
both tables' gathers to the one `engram-ahead` worker, serially: layer 1's rows, then layer 14's. On rank 0
layer 1's rows land 347 ms after the ids' D2H (chunk 2: 307) and layer 14's are ready when needed (its join is
2 ms); on rank 1 layer 1's take **648 / 609 ms** and layer 14's land **474 / 443 ms late** (ids D2H at 1,602 ms,
rows H2D at 2,076; rank 1's GPU idles, rank 0 spins in the all-reduce). Each table's gather is ≈ 1.1–1.3 s of worker
time on this cold box; rank 1 starts its second one 300 ms later and overruns layer 14. **Engram on the critical
path = 654 + 1,549 = 2.2 s = 29 % of the TTFT** at 3 % residency — the largest idle term now, ahead of the
export-bound gap (0.66 s, 9 %). Why rank 1's gathers are consistently ~300 ms slower is open (same shards, same
residency, same worker code; candidates: NVMe queue arbitration between the two processes' `WILLNEED` pools,
CPU scheduling against the 32 kt export threads of the two ranks); the per-table `ahead hit / ms per lookup`
log line is per rank and will answer it on the next serve that keeps the container log. T3.20 (c)'s chunk-ahead
`WILLNEED` was priced at 0.3–0.4 s per chunk from window 1; on this profile it is ≈ 1.1 s per chunk because the
loop got faster around a gather whose duration did not.

### Window 2 sizing (derived from this profile, both ranks; 80 layer-chunks per 16K prefill)

| variant | period model | loop | 16K TTFT | 16K prefill | VRAM | needs |
|---|---|---|---|---|---|---|
| today | max(GPU 29 + 27, export ≈ 65) = 65 ms | 5.2 s | 7.5 s | 2,130 | — | — |
| (i) export-light only: DMA gate / up from the `cudaHostRegister`ed arena, host does down + scales (≈ 20–25 ms) | GPU-bound, serialised: 56 ms | 4.5 s | ≈ 6.8 s | ≈ +10 % | 0 | registered arena, 2D copies; export ≤ 56 ms |
| (ii) (i) + the four copies on a side stream, **one** staging set: copy of L+1 starts when L's streamed MoE ends and overlaps L+1's attention / dense / resident MoE (≈ 20 ms); streamed MoE waits on an event | ≈ copy 29 + streamed MoE ≈ 7 → 36–40 ms | ≈ 3.0 s | ≈ 5.3 s | **≈ +40 %** | 0 | the host must not block in the export join before launching L+1's attention (join at the streamed-MoE call); export ≤ ≈ 36 ms (down + scales only) |
| (iii) (ii) + a second staging set | ≈ max(29, 27) + ε ≈ 30–33 ms | ≈ 2.5 s | ≈ 4.8 s | ≈ +55 % | +1.6 GiB / rank | not affordable: 64K × 4 trough 2.0 GiB, 262K × 4 none |

Variant (i) alone misses the +25 % go bar; (ii) clears it with no VRAM and is the one to build. Two caveats the
model does not hide: every variant shortens the loop around an Engram gather of fixed length, so layer 14's
lateness grows (at a 36 ms period layer 14 is needed ≈ 0.5 s after the chunk starts and the cold gather takes
≈ 1.2 s) — the **chunk-ahead Engram gather (T3.20 (c), both tables in parallel, submitted a chunk early) has to
land with or before window 2**, and on a cold box it is worth ≈ 2 s of today's 7.5 s by itself, more than (i);
and the two ranks' exports share the 16 cores, so (i)'s host saving counts twice.

### Window 2 design (2026-10-10, `window2/`) — (ii) cannot be built on the wheel as shipped; the fix is a kt-kernel allocator patch, not a kernel

The (ii) row above assumed the overlay could `cudaHostRegister` kt-kernel's resident expert arena and DMA from it.
Four findings from reading the overlay, the wheel and upstream kt-kernel source, plus one measurement, change the
shape of the work (not the ceiling):

1. **The overlay already joins the export at the streamed-MoE call.** `KTEPWrapperMethod.apply` runs the resident
   FlashInfer MoE first and only then `PrefillPipeline.run_layer()`: rank 0 `sync_write_weight_scale_to_buffer()`
   + gloo barrier → `_StreamedExpertSlots.load()` (4 H2D copies on the **compute** stream, gate / up half swap,
   `e8m0_from_bf16_bits_`, `block_scale_interleave`) → CUTLASS MoE → `_submit_export(next)` (waits
   `dma_done[buf]`, barrier, one `submit_write_weight_scale_to_buffer(tp_size, e, ptrs)` per expert). The host
   buffers are a POSIX shm of 2 × tp × slice (5.8 GB), `cudaHostRegister`ed by each rank. So (ii)'s "join late"
   requirement is already met; what is missing is the side-stream copy and an export that fits under ≈ 36 ms.
2. **kt-kernel 0.7.0.post4 is a binary wheel (PyPI, uploaded 2026-09-13) with no C++ source in the image, and its
   pybind surface has no accessor for the resident weights**: `kt_kernel_ext.moe.AMXFP4_KGroup_MOE` exposes only
   `forward[_task]`, `load_weights[_task]`, `warm_up[_task]`, `run_layerwise_fp8_batch`,
   `write_weight_scale_to_buffer_task`. There is no pointer to register. Upstream source at `3a19fd6` (2026-09-01,
   `kt-kernel/operators/amx/fp4-moe.hpp`, `moe_base.hpp`, `la/amx_buffers.hpp`): the resident storage is three
   separate `std::aligned_alloc(64, buffer_b_required_size(n, k))` per expert (gate, up `[I, H/2]` nibble-packed
   row-major; down `[H, I/2]`), each followed by its FP32 `[N, K/32]` scales, which `finalize_scale_e8()` compacts
   **in place to uint8 E8M0** after load (`scale_e8_valid` per tensor; a tensor whose scales are not all positive
   powers of two stays FP32). kt copies the loader's tensors into that storage and Python drops them. The export
   (`write_weights_to_buffer`) is one `do_work_stealing_job` per expert — 100 tasks here (cpu_tp 1 < gpu_tp 2):
   8 chunk memcpys each for gate / up per rank, 32 down tasks of per-row 720-byte `fast_memcpy` (AVX-512
   `loadu`/`storeu`, no non-temporal stores) and a scalar `copy_scale_to_bf16`. That is the 3.2 GB/s-per-thread,
   core-bound 65 ms (E22).
3. **Rank 1 cannot DMA from rank 0's arena as allocated.** The ranks are separate processes; a private
   `aligned_alloc` arena is not mappable by another process (`/proc/<pid>/mem` cannot be `mmap`ed, and
   `cudaHostRegister` needs a mapping in the registering process). Any arena-DMA design puts the arena in
   shareable memory mapped by both ranks. `/dev/shm` on this host is tmpfs at 50 % of RAM ≈ 123 GiB < the 132 GB
   arena, so the right primitive is `os.memfd_create` in rank 0 (not bounded by `/dev/shm`'s size, bounded by
   RAM + swap) opened by rank 1 through `/proc/<pid0>/fd/<n>` (same container, same user). Host THP is
   `enabled=madvise`, `shmem_enabled=never`, `AnonHugePages 0`: kt's arena already lives on 4 KiB pages, so a
   shmem arena is neutral for decode (and `shmem_enabled=advise` + `MADV_HUGEPAGE` is a later lever, not a cost).
4. **Measured on the idle box, in-image (`window2/dma2d.py` → `dma2d.txt`):** today's contiguous w13 slice,
   928.1 MiB, 17.09 ms = 57.0 GB/s; the down column slice as `cudaMemcpy2DAsync` straight from a pinned row-major
   `[H, I/2]` host layout (720-byte rows at pitch 1,440, 4,096 rows × 165 experts, 464.1 MiB) **9.16 ms one call /
   9.36 ms per expert = 52–53 GB/s, bit-identical for both ranks' halves**. Down needs **no host gather**: the
   2D engine runs at 91 % of the contiguous roof, so with a shared arena the host export disappears entirely
   (period = max(copy ≈ 29 ms overlapped, GPU work) rather than the export's 65).

**Design (ii′) — shared arena, export-free streaming.**

- *kt-kernel patch (allocator + pybind, ≈ 40–60 lines C++, no kernel code):* a `MOEConfig.weight_arena` pointer
  (+ byte count); when set, the `AMX_MOE_BASE` constructor carves gate_e / up_e / down_e BufferB storage
  sequentially (64-aligned) from the arena instead of `aligned_alloc`; a static
  `weight_arena_bytes(expert_num, hidden, inter, group)` so Python can size it before construction; and a
  post-load `weight_arena_layout()` returning per-expert `(gate_off, up_off, down_off)` and the three
  `scale_e8_valid` flags. Load, forward and the existing export are untouched (same bytes, same addresses modulo
  base). Built from source at the commit matching the wheel (pin by diffing the wheel's `experts.py` /
  `utils/amx.py` against upstream between 2026-09-01 and 09-13), `avx512_bf16` variant only, added to
  `config/Dockerfile.dsv41-kt` next to the existing `config/kt-kernel-patches/` (image has cmake, g++ 13.3,
  `numa.h`, `hwloc.h`).
- *Overlay:* rank 0 creates one memfd per MoE layer sized by `weight_arena_bytes`, maps it, passes
  `weight_arena` into the wrapper, and `cudaHostRegister`s it; rank 1 opens the same memfds, maps and registers
  them. `PrefillPipeline` loses the export, the 5.8 GB shm staging (back to the Engram page cache), the gloo
  barriers and `sync_write_weight_scale_to_buffer`; per layer, on one side stream after an event recorded at the
  end of layer L−1's streamed MoE: up rows → `w13[:, :I_tp]`, gate rows → `w13[:, I_tp:]` (the half swap folds
  into the copy destination), E8M0 uint8 scales straight (an FP32 → E8M0 GPU fallback when a flag is false),
  down via the 2D copy, down scales via a 2D copy (45-byte rows, ≈ 30 MB) or a full copy + GPU slice; the
  compute stream waits on the copy event before the streamed MoE. The `test_kt_prefill.py` pipeline tests are
  rewritten for the event-only protocol.
- *Expected:* period ≈ copy 29 ms overlapping ≈ 20 ms of attention / dense / resident MoE, plus ≈ 7 ms streamed
  MoE → **≈ 36 ms (65 today), i.e. the (ii) row's ≈ +40 % on 16K prefill**, and the 16 cores are idle during
  prefill (the Engram chunk-0 gather no longer competes with 16 export threads — a second, unsized gain).
  Decode unchanged by construction (same resident bytes, same kt forward). No VRAM change (one staging set).

**Alternatives.** (K1) a getter + rank-mask export patch instead of the shared arena: rank 0 DMAs from its own
registered arena, the host exports rank 1's slice only (≈ half the tasks) → period ≈ max(29 overlapped, ≈ 42
export) ≈ 42 ms, ≈ +25 % — at the go bar, still a kt rebuild, and leaves the barrier / shm / 8 cores busy.
(0) stop window 2 here at 2,195 / 2,499. Side-stream copies alone (no kt change) give nothing: the period is
export-bound, not copy-bound.

**Risks to carry into the decision.** The wheel ↔ source commit match (a mismatch shows up at load, not at
run; the pin is verifiable by diff); `cudaHostRegister` of 132 GB in each rank at startup (pin rate ≈ 1–3 GB/s →
≈ 1–2 min per rank, concurrent; `--ulimit memlock=-1` is already set); memfd-backed shmem counts as shared
memory in cgroup accounting (same total as today's anonymous arena + 5.8 GB less); and the policy point —
**this is a kt-kernel allocator / pybind change, not CUDA kernel work (T3.21 stays closed), but it adds a
from-source kt-kernel build to the image**, which the recipe has avoided so far. Go / no-go is the owner's call
before the build starts.

### Files (window 1b)

| path | what |
|---|---|
| `window1b/diag/` | the diagnostic serve on `ad973440`: launch / ready / manifest, three benches (seed 1277 cold 1,792; `bench2` is the prefix-cache hit at the same seed, 14,815 "tok/s"; `bench3` seed 4242 under py-spy, 1,316), `pyspy-stale.raw.gz` (py-spy raw, 15 s at 250 Hz, `--native`) |
| `window1b/analysis/period*.py`, `gpugaps.py` | host anatomy of one layer period and GPU gap census from the window-1 traces |
| `window1b/analysis/moe_host.py`, `moe_alt.py` | in-image microbench: host time of `cutlass_fused_moe` at one count vs alternating counts vs one runner per count |
| `window1b/analysis/pin_check.py` | in-image check of the overlay helper: host ms and bit-identical outputs |
| `window1b/{launch,serve-log}.txt`, `manifest.json` | the fixed serve (`99128a7e`), container log (dedicated-runner lines) |
| `window1b/prefill-16k-{warm,unprofiled,pyspy}.*`, `prefill-32k.*` | bench outputs; `-pyspy` ran under the 5 s sample |
| `window1b/pyspy-fixed.raw.gz`, `record-pyspy.txt` | py-spy raw on the fixed serve |
| `window1b/c1*`, `c4*`, `verify.*` | decode and correctness sanity |
| `cpuinfer14-soak/` | the `KT_CPUINFER=14` soak on `99128a7e`, same layout as `qual-99128a7e/` plus `soak14.sh` / `soak-log.txt` |
| `window2/dma2d.py`, `dma2d.txt` | in-image, idle-box microbench of the H2D copies window 2 needs: contiguous w13 slice vs `cudaMemcpy2DAsync` of the down column slice from a row-major host layout (per expert and one call), bit-identity of both ranks' halves |
| `qual-99128a7e/` | the qualification serve: `qual.sh` driver + `qual-log.txt`, launch / ready / manifest / `pre-serve-state`, `prefill-{16k,32k}.*`, `c1.*`, `c4.*`, `corpus/` + `corpus.txt` (NLL), `consistency*.{json,txt}`, `stress-kv.*`, `verify.*`, `vram-after-capture.txt`, `runner-log.txt` (dedicated-runner lines) |

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
| `reprofile-99128a7e/analysis/{reprofile.sh,regaps.py,rebusy.py}` + `*-tp{0,1}.txt` | re-profile driver, the two new analyses and the outputs of all four (`gaps`, `period`, `regaps`, `rebusy`) per rank |
| `reprofile-99128a7e/*-TP-{0,1}.trace.json.gz`, `prefill-16k-*.{json,txt}`, `{launch,ready,start-profile,stop-profile,prefill-log}.txt`, `manifest.json` | traces, bench outputs and provenance of the re-profile serve |

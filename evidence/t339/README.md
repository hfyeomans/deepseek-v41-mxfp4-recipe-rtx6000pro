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

Not run on this serve: the 4 × 62K stress, NLL and same-prompt gates — **`99128a7e` is not yet the qualified
profile**; that is the next serve (and the one the approved `KT_CPUINFER=14` rider can ride on, one change per
serve).

What this changes upstream of window 2: the layer period's host side is no longer 90 ms, so the pipeline is
now GPU-side (≈ 52 ms: 29 HtoD + 3 fix-up + ≈ 20 compute) plus whatever the 6–17 ms export join still exposes.
Window 2's ceiling (≈ +70 % serialised) was derived from the 90 ms period and is stale; re-profile (GPU + CPU,
one chunk) on `99128a7e` before scoping it. The two lessons: a "no host op" gap in a torch-profiler trace says
the host is in *some* non-aten native call, not which one — sample the native stack before attributing it; and
a C++ library that caches by one key silently thrashes when the call pattern alternates two keys.

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

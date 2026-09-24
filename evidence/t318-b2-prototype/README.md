# T3.18 / E30 — B2 decode prototype: DRAM-resident experts computed by the GPU straight out of pinned host memory (2026-09-24)

**Verdict: B2 is dead as a single-user lever on this box. Kept as an off-by-default experiment.**
C1 41.92 → **42.71** summed / 39.86 → **40.71** wall tok/s at |S| = 40 (+2 %), **40.65 / 38.73** at
|S| = 60 (−3 %); TTFT and C4 unchanged once B2 is gated to bs = 1. The FEASIBILITY "78 tok/s B2
ceiling" assumed that additive DRAM *bandwidth* (E4b) turns into additive *throughput*; a real
forward shows why it does not (§4).

Server: launcher defaults (226 / 158, D=4, spec off, chunk 4096, fraction 0.96, capture 1 2 3 4,
`KT_GPU_PREFILL_MIN_TOKENS=1024`, `KT_NAN_PROBE_MIN_TOKENS=1`) plus `KT_B2_UVA_EXPERTS=40|60`,
overlay `7e41317e` (`~/dev/src/sglang-dsv41-overlay`, branch `dsv41-kt-hybrid`), image
`dsv41-kt:recipe` unchanged. Baseline = `evidence/t316-review-intake/c1-off6-t316c.json`
(41.92 / 39.86 / TTFT 0.609, overlay `8ad514ed`), same prompts, seed and fixture offset.

## 1. What was built (deviations from the TASKS row, all recorded in `plan/TASKS.md` T3.18)

| TASKS row said | Built | Why |
|---|---|---|
| rank-0-only | **symmetric TP**: each rank computes its TP slice of the B2 experts; the layer's existing all-reduce sums them, exactly like the resident tier | no new collective, both PCIe links used (E4b: additive) |
| DMA into a fixed device buffer, then E13 kernel | **direct UVA**: the weights *stay* in pinned host memory (`cudaHostAlloc`, in SGLang's FlashInfer layout) and the existing FlashInfer SM120 CUTLASS MXFP4 grouped GEMM runs on their device pointers; the GEMM issues the PCIe reads itself | no custom kernel, no staging copy; the grouped GEMM reads only routed experts' rows so bytes are data-dependent while the captured work is not |
| S ≈ 40 by hotness | S = the **last N** CPU experts (ids 384−N..383); no hotness sorting | the qualified placement (E28) uses contiguous ids, no hotness data on file |
| kt-kernel keeps the rest | kt **still loads all 158** CPU experts and is handed S back for forwards above `KT_B2_MAX_TOKENS` | short prefill (< 1024 tokens, not streamed) hits most of S per layer — see §3 |
| side stream | first run on the **main stream before `sync()`**, then re-run **forked** on a per-device side stream after `submit()`, joined after `sync()` | §3 explains the difference |

Code: overlay `python/sglang/srt/layers/moe/kt_ep_wrapper.py` (`_B2HostPool`, `_B2UvaExperts.run/fork/join`,
`remap_topk_ids_for_b2`, `KTEPWrapperMethod.apply/submit`), launcher `KT_B2_UVA_EXPERTS` (default 0)
and `KT_B2_MAX_TOKENS` (default 1), microbench `scripts/bench-b2-uva-moe.py`.

Correctness argument for the fork (no `record_stream`): each layer's fork starts with
`side.wait_stream(main)` enqueued *after* the previous layer's join and add on main, so the side
stream's allocator pool cannot hand the previous B2 output's block to a kernel that could run before
main has read it; the same ordering holds inside a CUDA graph's private pool. FlashInfer's
`cutlass_fused_moe` allocates its workspace per call on the current stream (`getWorkspaceInfo`,
`alloc_tensor`) and sets tactics host-side under `mMutex`, so the B2 call and the resident tier's call
share no device state.

## 2. Microbench (idle GPU0, `scripts/bench-b2-uva-moe.py`, random FP4, TP-2 slice = 8.96 MiB/expert/rank)

| file | measurement | result |
|---|---|---|
| `uva-moe-gpu0.json`, `uva-moe-both-gpu{0,1}.json` | raw PCIe: `cudaMemcpyAsync` H2D and a copy kernel over UVA, 1 and 6 slices | **51.4 / 52.8 GiB/s** per link, both cards, memcpy = UVA copy kernel; `index_select` over UVA 6.7–13 GiB/s (do not gather through torch) |
| same, `moe` | FlashInfer grouped GEMM M=1, 6 distinct hits: device weights vs UVA weights | device 0.17 ms; **UVA 1.35 ms = 39 GiB/s effective, bit-equal**; graph replay 1.28 ms bit-equal; a gather-then-GEMM variant 3.9 ms |
| `uva-moe-sparse-idle.json` | M=1 with 0 / 1 / 2 / 6 valid B2 slots | 0.089 / 0.326 / 0.533 / 1.37 ms → **≈ 0.09 ms floor + 0.21 ms per expert hit** |
| `uva-moe-sparse-dramload.json` | same under a 130 GB/s host-DRAM read hog (16 threads, 8 GiB) | eager launches slow to ≈ 1 ms (host-side launch latency), **graph replay unchanged (1.30 vs 1.28 ms)** → DRAM contention is not what limits B2 |
| `uva-moe-overlap-gpu0.json` | B2 on a side stream beside a 226-expert device tier, graphed | M=1 h=1: tier 0.113, B2 0.239, sequential (`sequential` key) 0.349, **forked 0.241** = max(tier, B2); h=6: 1.386 → 1.279; M=4 h=1: 1.154 → 0.867 (eager "serial 26 ms" rows are an alternating-runner artefact, ignore) |

kt-kernel's CPU tier reads a **full** expert (17.9 MiB) in ≈ 0.146 ms at 125 GB/s (E21/E26). Per expert
hit B2 is therefore **1.4× slower than the CPU tier** even with both PCIe links busy (two 8.96 MiB
slices in 0.21 ms = 85 GB/s aggregate; at the 106 GiB/s raw ceiling it would still be 0.165 ms).

## 3. Served runs

| file | config | C1 summed / wall tok/s | TTFT | notes |
|---|---|---|---|---|
| `c1-off6-t316c.json` (baseline, other dir) | B2 off | 41.92 / 39.86 | 0.609 s | ≈ 23.9 ms/step |
| `c1-off6-s40.json` | S=40, **main stream**, no token gate | **33.12 / 28.54** | **2.525 s** | regression; `verify-c4-s40.json` PASS; `hsmp-c1-s40.txt` DRAM 0–125 GB/s bursts (not starved); `prof-s40-gpuonly/`: M=1 steps GPU busy 25–30 of 31–36 ms (baseline 15.4), 160 grouped GEMMs/step = 4/layer, B2 ≈ 10–13 ms/step on the critical path; short-prefill layers (M = 9–13) hit most of S |
| `c1-off6-s40-fork.json` | S=40, **forked**, `KT_B2_MAX_TOKENS=1` | **42.71 / 40.71** (min–max 41.87–42.94) | 0.617 s | `verify-c4-s40-fork.json` PASS (bs=1 with B2 agrees with bs=4 without it: 225-char prefix); `prof-s40-fork-gpuonly/`: union GPU busy 17.3 ms/step, B2-sized grouped GEMMs 37/step summing 4.9 ms → ≈ 70 % hidden, step ≈ 24.3 ms in-profile |
| `c4-off0-s40-fork.json` | same server, C4 | 57.22 / 55.02 | 1.50 s | B2 off at bs=4 by the gate; E27/E28 C4 reference 56.2–59.0 summed — unchanged |
| `c1-off6-s60-fork.json` | S=60, forked | **40.65 / 38.73** | 0.639 s | `verify-c4-s60-fork.json` PASS; `prof-s60-fork-gpuonly/`: union busy 19.8 ms/step, B2 GEMMs 51/step summing 7.3 ms → B2 is now the longest pole |

RAM: S=40 pinned 14.8 GiB/rank (`free -g`: used 184, available 62); S=60 21.0 GiB/rank (used 215,
available 31, Engram page cache squeezed) — S ≥ 80 would not fit beside kt's own copy of S.

Health across all three launches: 0 NaN / `probability tensor` / Xid / Traceback lines, EDAC 0/0,
no phantom-busy GPU after teardown. Launch logs: `launch-s40-mainstream.txt`, `launch-s40-fork.txt`,
`launch-s60-fork.txt` (Decode/Prefill batch lines stripped, bridge IP redacted).

The NLL gate was not run: `nll-score.py` prompts are long enough to take the streamed-prefill path,
where B2 is not used, so it would measure nothing about B2. The same-prompt bs=1-vs-bs=4 consistency
check *is* the B2 correctness gate here (bs=1 uses B2, bs=4 does not, and they agree), on top of the
microbench's bit-equality against device weights.

## 4. Why additive bandwidth did not become additive throughput

Per layer at bs=1 the CPU-resident work is ≈ 6 × 158/384 ≈ 2.5 expert hits. Split S of them off to B2:

| |S| | B2 hits/layer | B2 time (0.09 + 0.21 × hits) | kt hits/layer | kt time (0.146 × hits) | longest pole |
|---|---|---|---|---|---|
| 0 | 0 | — | 2.47 | 0.36 ms | kt |
| 40 | 0.63 | 0.22 ms | 1.84 | 0.27 ms | kt (≈ balanced) |
| 60 | 0.94 | 0.29 ms | 1.53 | 0.22 ms | **B2** |

The model puts the crossover near |S| ≈ 45–50 and the measured curve (0 → 41.9, 40 → 42.7, 60 → 40.7)
agrees. Two things cap the gain at the balance point:

1. **B2 is slower per expert than the path it replaces** (0.21 vs 0.146 ms), so the balanced split saves
   at most ≈ 0.11 ms/layer ≈ 4.4 ms/step of *CPU-path* time — and only if that time was exposed.
2. **Most of the CPU path is already hidden at bs=1**: D=4 deferral moves part of each layer's CPU work
   behind the next layer's attention, and E21 found a fixed ≈ 0.45 ms/layer component of the wait that
   does not scale with bytes. Removing 25 % of the CPU tier's bytes (S=40) therefore removed little
   exposed time, while B2's own GEMM added 4.9 ms/step of GPU work of which ≈ 1.9 ms was *not* hidden
   (union busy 15.4 → 17.3 ms/step).

Contrast with E28, where moving experts to VRAM cut C1 step time by the full byte reduction: that
lever removes CPU bytes at ≈ zero added cost (the device grouped GEMM is 15–24 µs per call), B2 removes
them at 1.4× their cost. B2 would pay only if (a) the exposed CPU wait were large (D=0 — but D=4 is worth
+21 % on its own, E26) or (b) the GPU could read host memory faster than the CCD fabric, which one x16
link per rank cannot. Neither is available on this hardware; a third GPU (T4.1b) changes (b) by holding
the experts in VRAM, not by streaming them.

What this leaves for the roadmap: the CPU tier's *fixed* per-layer cost (E21's 0.45 ms, the 40
submit/sync boundaries) is the un-hidden part, and only T3.19(a)'s per-layer timeline can separate it
from bytes. B2's two useful by-products stay: `_B2HostPool` / direct-UVA compute is a working,
graph-capturable way to run a grouped GEMM on pinned host weights on SM120 (no custom kernel), and the
`fork/join` pattern is the template for T3.20's Engram staging.

## 5. Method notes (also in TROUBLESHOOTING)

- `/start_profile … num_steps 8` counts scheduler steps, so a trace started before a request straddles
  its prefill: split the decode region at the last > 3 ms gap and divide by all-reduces/80, not by 8.
- Raw `/v1/completions` prompts of a few tokens make this model emit EOS at once (1 completion token);
  use `/v1/chat/completions` for profiling requests.
- `step-profile.py` sums kernel durations, which double-counts overlapped streams; the per-stream union
  in §3 was computed ad hoc from the trace (`args.stream`) — worth folding into the script (T3.19a).

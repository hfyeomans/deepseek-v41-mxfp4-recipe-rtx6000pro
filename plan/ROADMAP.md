# Roadmap and readout

The one page to update when an experiment lands. Protocols live in [EXPERIMENTS.md](EXPERIMENTS.md), task breakdown in [TASKS.md](TASKS.md), phases and tracks in [README.md](README.md), sources in [RESEARCH.md](RESEARCH.md). This page holds the *readout*: what each experiment has told us so far and which decision it fed.

Conventions: one row per experiment; `Result so far` is a measured number with its condition, or `—`; `Decision fed` names the doc/flag/choice that changed. Add a new experiment by giving it the next `E` number in EXPERIMENTS.md, a task in TASKS.md, and a row here. Add an option by giving it a letter in the options register below. Never delete a row; strike it through and say why.

Status vocabulary (same as TASKS.md): `done`, `ready`, `quiet-host`, `gpu-window`, `blocked`, `decision`, plus `partial` when a stage has run.

## Where we are (2026-09-21)

```diagram
P0 research ──▶ P1 host micro ──▶ P2 baseline ──▶ P3 2-GPU hybrid ──▶ P4 3/4-GPU
   DONE          DONE               DONE*          design + kernel    blocked
                 (E1, E1b, E2, E3,  (E8, E12 done; (E6 s0 done;       (cards, power;
                  E4, E4b, E6 s0)    *E8b needs     E13 s1 runs:       250 W is the
                                     sudo)          1.45 TB/s decode)  cap floor)
```

Blocking condition for everything below marked `quiet-host` or `gpu-window`: the GLM container `glm53-flash-2x-rtxpro6000` is serving and benchmarking on both GPUs. Owner announces windows.

## Experiment readout

| E | Name | Needs | Status | Result so far | Decision fed |
|---|---|---|---|---|---|
| E1 | Host DRAM bandwidth | quiet host | done | 146.2 GB/s copy (read+write) at 32 workers, Python; superseded by E1b for the read figure | Historical; the ceiling now rests on E1b |
| E1b | Host DRAM **read** bandwidth, torch, quiet host | quiet host | done | **124.7 GB/s** read at 16 threads (67 at 8, 118 at 32); 30 % of 409.6 theoretical | `--dram-gbps 125` default → CPU-compute ceiling 50 tok/s. Limit is the 2-CCD fabric, not DIMMs (E4b) → upgrade path is CPU CCD count, not faster RAM |
| E2 | GPU memory bandwidth | GPU | done | D2D ≈1,470 GB/s per card (82 % of 1,792 derived) at 300 W | Weight-streaming roofline; E13 decode reaches 93 % of it |
| E3 | Engram gather DRAM vs NVMe | quiet host | done | DRAM 4.15 M rows/s (86K tok-equiv/s); 9100 PRO 403 K rows/s (8.4K tok-equiv/s); T705 347 K rows/s | Engram default = NVMe + 64 GiB DRAM cache; NVMe gather is not the decode bottleneck at the hybrid's 56 tok/s |
| E4 | PCIe H2D/D2H per GPU | GPU | done | pinned 53.1 / 52.9 GiB/s per card; **106.1 GiB/s (114 GB/s) both concurrently**; D2H 53.3 / 52.4 | `--pcie-gbps-per-gpu 57`; PCIe-streaming fallback **46 tok/s**, within 10 % of CPU compute; layerwise prefill 1.06 s per 4,096-token chunk |
| E4b | CPU read vs GPU DMA contention on host DRAM | GPU | done | CPU 123.7 + H2D 114.0 solo → **233.9 GB/s together, additivity 0.98** | **Option B2 (split)**: DRAM experts served by CPU compute *and* PCIe streaming at once; calculator `split ceiling` 78 tok/s (71 at the 94 GB/s CPU-sustained figure) |
| E5 | V4.1 on 2 GPUs, stock generic offload | GPU + 475 GiB download | gpu-window | — | Evidence that a MoE-aware engine is required (expected OOM/unusable) |
| E6 s0 | CPU expert GEMV, 16 cores | quiet host | done | bf16 GEMV 76 ms/token at 16 thr = **94 GB/s effective** (75 % of the E1b read ceiling); 32 thr worse; `read` and `int8-deq` variants invalid | DRAM bandwidth, not compute, is the CPU-side bound; CPU-compute ceiling 40–50 tok/s (94–125 GB/s). A real MXFP4 CPU kernel is still needed to confirm compute keeps pace at 4-bit |
| E6 s1 | sglang-kt hybrid, V4-Flash, 2 GPUs | GPU | gpu-window | — | Toolchain proof on SM120; calibrates `--gpu-ms-per-token` and `--dram-gbps` against a real hybrid |
| E6 s2 | Port `deepseek_v41` into sglang-kt; first V4.1 serve on 2 GPUs | build, then GPU | design doc next (`plan/E6-DESIGN.md`) | — | The primary deliverable |
| E7 | Routing-skew histogram | GPU | gpu-window | — | `--activation-skew`; frequency placement gain; whether Option D (pruning) is worth anything |
| E8 | Baseline re-measure V4-Flash NVFP4 TP2 | GPU | **done** | C1 **171–188 tok/s** (256-tok fixture, acceptance 40–45 %), **207–239** (512 tok, acceptance 51–63 %); C2 354 summed / 300 wall; wall-clock ceiling ≈ 300 tok/s at `max_num_seqs=2`. Prefill **7.8K tok/s at 32K → 3.2K at 512K → 1.9K at 1M** (length curve). 6/6 needles at 222K. GPU draw 242–249 W decode (uncapped), 300 W prefill (throttled to 1.95 GHz) | The number every 2-GPU result is compared with; carry the 512-tok row and always report acceptance beside tok/s |
| E8b | tok/s per watt curve (power caps) | GPU | **blocked on sudo** (script ready) | driver floor is **250 W**, 225 W does not exist; prediction from E8 draw: 250 W costs decode ≈ 0, prefill ≈ 15 % | T4.0 power decision (2,000 W PSU vs 250 W cap) — run `sudo bash scripts/bench-power-sweep.sh` |
| E9 | 4-GPU V4.1 qualification | 4 cards | blocked | calculator: 100% experts resident, ~200 tok/s, Engram-in-DRAM fits | Option A profile in `serve.sh` (unqualified) |
| E10 | DSpark on/off under CPU-bound MoE | GPU, after E6 s1 | blocked on E6 s1 | — | Whether speculative decoding pays when the CPU is the bottleneck |
| E11 | Engram tiering / quantization A/B | quiet host, then GPU | quiet-host (sim) | calculator: FP8 tables in DRAM do not fit 2-GPU (−78 GiB); FP4 +13 GiB, q3_K +32 GiB headroom | Engram mode per GPU count; first quality number for quantized tables |
| E12 | Kernel census on SM120 | GPU (from sibling torch-profiler traces) | **done for V4-Flash**; V4.1 engines pending E6 | **Not Marlin**: experts run on a CUTLASS SM120 block-scaled PtrArray grouped GEMM (E13's family, NVFP4 types), 24.5 % of kernel time; dense FP8 27 %, bf16 GEMM 12 %, attention 11 %, DSpark draft 10 %. 15.3 ms union per 6-token verify step → 3.8–5.1 ms per accepted token | `--gpu-ms-per-token 5` confirmed as the conservative end; expert-GEMM speedups cap at ~25 % on this engine; the hybrid's GPU term is dense FP8 + attention + draft |
| E13 s1 | MXFP4 grouped GEMM on SM120 (CUTLASS 79d type swap; W4A4 and W4A8 variants) | build GPU-free, run GPU | **done** (W4A4 + W4A8 + census-shaped runs; Marlin ratio moot — see E12) | **Compiles, verifies and runs on SM120.** Decode (6 experts, M=1): gate/up 53.8 µs = **1,397 GB/s** of weights (93 % of E2), down 39.6 µs = 949 GB/s; ≈3.8 ms per token for all 40 layers on one GPU. Prefill 64 tok × 384 experts 335 TFLOPS; dense proxy **1,060 TFLOPS** at 300 W. 128×128×128 beats 128×64×{128,256} at every decode M | Kernel is lossless w.r.t. the checkpoint and already at the memory roofline for gate/up; the GPU term of the hybrid is not where the time goes. Census-shaped addendum: TP2 × 36 groups **1,314–1,435 GB/s**, TP1 × 36 groups (451 MB > L2) **1,421–1,451 GB/s** = 97–99 % of E2; at 15 groups the launch time equals the deployed vLLM CUTLASS NVFP4 kernel's (72.5 vs 73 µs). Open: decode-shaped M tile (K10), activation-error check |
| E13 s2 | Integrate winning kernel into the E6 engine | build, GPU | decision after s1 | — | Re-run E12 |

## Kernel experiment menu (E13 sub-items and successors; pick by measured need)

Ordered by how directly each moves a number in the readout above.

| ID | Experiment | Hypothesis | Cost | Status |
|---|---|---|---|---|
| K1 | mxf4×mxf4 grouped PtrArray GEMM on SM120 (E13 s1, W4A4) | CUTLASS's builder accepts it; runs at V4.1 expert shapes; ≥ 1.3× Marlin at decode M | source + compile now (GPU-free); run in window | **done** — builds, verifies, 1,397 GB/s at M=1 (`kernels/e13/`); Marlin ratio pending K5 |
| K2 | mxf8×mxf4 grouped GEMM (E13 s1, W4A8) | Lower activation-quant error than K1 at similar speed; `kind::mxf8f6f4` path proven by example 79c and the shipped `mxf8_mxf4` grouped test | `E13_A8=1 kernels/e13/build.sh` | **done** — builds, verifies, but **2× slower at decode** (104 µs vs 54; 724 GB/s): the 128-row M tile makes M=1 cost M=128 of MMA, and the FP8-rate MMA is compute-bound there. "Similar speed" rejected for decode; fine for prefill (untested) |
| K3 | Tile sweep under 99 KiB smem | 128×128×128 (79d) vs 128×64×256 / 128×32×256 (the mxf4 unit-test tiles) for decode M ≤ 64 | compile variants | **done for N=64** — both smaller tiles lose at every decode M (−30 to −35 %); hypothesis rejected; next idea is a small **M** tile, not a small N tile |
| K4 | Cooperative vs pingpong schedule at decode shapes | Pingpong wins at small M | free (79d has both) | **done** — pingpong +1–5 % at decode; cooperative +6–8 % at 384-group prefill |
| K5 | Marlin W4A16 reference on the same shapes (`VLLM_MOE_FORCE_MARLIN=1`, inside `dsv4-nvfp4:recipe`) | Gives the denominator for K1–K4 | GPU window | **dropped** — E12 shows the deployed NVFP4 path is already a roofline CUTLASS SM120 grouped GEMM (73 µs/launch, matched by E13 at 72.5 µs); Marlin is 0.24 ms/step. The denominator that matters is the deployed kernel, and E13 ties it |
| **K10** | **Decode-shaped MXFP4 grouped GEMM**: M tile 16–32 (mma.sync M=16), non-TMA mainloop (cp.async / LDG.128), `kind::mxf4nvf4` directly | Removes the 128-row padding MMA that makes W4A4 two-thirds compute at M=1 and W4A8 fully compute-bound; should hold ≈1,400 GB/s for both projections and make W4A8 free at decode | new kernel (CuTe, no builder); `TILE_M=64` proved the SM120 TMA WS mainloop cannot do it | next after K5 |
| K6 | Activation quantization kernel (bf16 → MXFP4/MXFP8 block-32, fused with gather) | Needed to feed K1/K2 in a real engine; must not eat the win | write after K1 | after K1 |
| K7 | Hybrid glue: pinned-host activation gather/scatter with stream-ordered H2D/D2H overlapped with CPU expert compute (TK.5) | Overlap hides the PCIe hop entirely at batch 1–8 | after E6 s1 | blocked |
| K8 | Engram row gather from pinned host (zero-copy) vs GPUDirect Storage from the 9100 PRO | Zero-copy beats staging for 264-byte rows | after E6 s1 | blocked |
| K9 | Sub-4-bit / pruned experts | No SM120 kernels; explicitly out of scope unless E7 shows strong skew | — | parked |

## Options register

| Option | What | Fits 2 GPUs? | Calculator ceiling | Status |
|---|---|---|---|---|
| A | 4-GPU GPU-resident (SGLang `dev-dsv41` TP4, Engram in DRAM) | no (needs 4) | ~200 tok/s | profile written, unqualified; blocked on cards + power |
| B | **2-GPU three-tier hybrid**: VRAM 58% experts + attention, CPU/DRAM 42% experts, Engram NVMe + cache | yes (200.1 GiB host need, +46.7 margin) | 50 tok/s @125 GB/s read (E1b); 40 @94 (E6 s0 sustained) | primary mechanism; superseded as a target by B2 |
| B' | B with PCIe expert streaming instead of CPU compute | yes | 46 tok/s @114 GB/s measured | fallback; now within 10 % of B and needs no CPU kernel |
| **B2** | **B with the DRAM-resident experts split between CPU compute and PCIe streaming** (E4b: the two DRAM read paths are additive, 234 GB/s) | yes | **78 tok/s** @125+114; 71 @94+114 | **new primary target**; needs E6 s1/s2 engine work plus a per-step streaming path (K7) |
| C | Smaller Engram in DRAM (FP4/q3_K) on 2 GPUs | yes | same as B, better TTFT | E11 |
| D | Expert pruning to fit resident on 2 GPUs | only with heavy pruning | — | research-grade; needs E7 skew |
| E | 3-GPU EP3/PP, 91% experts resident | needs 3 | ~126 tok/s @146; 156 @300 | profile to write (T4.1b) |
| Baseline | V4-Flash-0731 NVFP4 TP2 (qualified) | yes | 200.6 tok/s measured | re-measure with this harness (E8) |
| W4A4 vs W4A8 | Activation format for E13 kernels | n/a | — | decided by K1/K2 error and speed |

## Standing assumptions to re-check when data lands

| Assumption | Value | Replaced by |
|---|---|---|
| GPU ms/token for resident part of the hybrid | **5.0 confirmed as the conservative end by E12**: 15.3 ms union per 6-token verify step, 3.8–5.1 ms per accepted token at 40–63 % acceptance; experts are only ~25 % of it | E12 on the V4.1 engine once E6 builds |
| Prefill baseline | **length curve, not a scalar**: 7.8K tok/s at 32K, 6.2K at 128K, 5.2K at 222K, 3.2K at 512K, 1.9K at 1M (E8) | E8 on V4.1 |
| Power cap floor | **250 W** (`power.min_limit`); the HARDWARE.md 225 W option is gone | — |
| Host DRAM read bandwidth | **125 GB/s measured (E1b)**; 94 GB/s sustained while computing (E6 s0) | done; a C STREAM build would only confirm |
| PCIe per GPU | **57 GB/s measured (E4)**, additive with CPU reads (E4b) | done |
| Host reserve | 24 GiB | — (policy) |
| Engram DRAM cache | 64 GiB | E11 |
| CPU expert compute | bf16 GEMV measured at 76 ms/token (7.15 GiB read); MXFP4 kernel at the same GB/s → ~20 ms/token | E6 s0 done; MXFP4 CPU kernel (KT/llama.cpp) still to measure |
| CPU DRAM limit is CCD fabric, not DIMMs | inferred from 2 CCDs × ~64 B/clk and confirmed indirectly by E4b additivity | a C STREAM run or `pcm-memory` DIMM counters would make it direct |

## Change log

- 2026-09-21 (GPU window 2) — **E8 done** with this harness (171–239 tok/s C1 depending on output length/acceptance; prefill length curve 7.8K → 1.9K tok/s; 6/6 needles at 222K; 242–249 W decode draw). `bench-decode.py` gained `wall tok/s` after C4/C8 summed rates proved invalid under `max_num_seqs=2`. **E12 done for V4-Flash** from the sibling's torch traces via new `scripts/kernel-census.py`: experts already on a CUTLASS SM120 block-scaled grouped GEMM (not Marlin), 24.5 % of kernel time; dense FP8 is the largest category. **E13 census-shaped runs** (`run.py --tp/--decode-groups`): 1,421–1,451 GB/s at 451 MB working set — L2 was not inflating stage 1; E13 ties the deployed kernel's launch time. **K5 dropped** as moot. **E8b blocked**: needs sudo and the driver's cap floor is 250 W (225 W option removed from HARDWARE.md); `scripts/bench-power-sweep.sh` + `summarize-power.py` ready. GLM container relaunched via its `serve.sh` (its `stop.sh` removes the `--rm` container; `docker start` does not work).

- 2026-09-21 (GPU window, later) — K2 W4A8 built and verified: 2× slower at decode than W4A4 (MMA-bound through 128-row M padding). `TILE_M=64` rejected by the SM120 TMA mainloop → K10 (decode-shaped kernel) added to the menu.
- 2026-09-21 (GPU window) — P1 host micro-benchmarks complete: E2 (≈1,470 GB/s GDDR7), E4 (53 GiB/s per card, 106 both), E1b (125 GB/s DRAM read), E6 s0 (94 GB/s while computing), **E4b new: CPU and DMA DRAM reads additive → Option B2 split, 78 tok/s calculator ceiling**. E13 stage 1 (K1, K3, K4) built and run: `kernels/e13/` mxf4×mxf4 grouped GEMM verifies on SM120, 1,397 GB/s at decode M=1, 1,060 TFLOPS dense. Calculator defaults: `--dram-gbps 125`, `--pcie-gbps-per-gpu 57`, new `split ceiling` column. `scripts/bench-dram-read.py`, `scripts/bench-dram-contention.py` added.

- 2026-09-21 — `docs/LEARNING.md` added: reading map per puzzle piece and per experiment.
- 2026-09-21 — Page created. E13 format question resolved from PTX ISA + CUTLASS source (no re-quant needed). CUTLASS v4.7.1 cloned to `~/dev/src/cutlass` for the GPU-free E13 build. Kernel menu K1–K9 added.

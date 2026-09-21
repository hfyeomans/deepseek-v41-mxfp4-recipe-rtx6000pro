# Roadmap and readout

The one page to update when an experiment lands. Protocols live in [EXPERIMENTS.md](EXPERIMENTS.md), task breakdown in [TASKS.md](TASKS.md), phases and tracks in [README.md](README.md), sources in [RESEARCH.md](RESEARCH.md). This page holds the *readout*: what each experiment has told us so far and which decision it fed.

Conventions: one row per experiment; `Result so far` is a measured number with its condition, or `—`; `Decision fed` names the doc/flag/choice that changed. Add a new experiment by giving it the next `E` number in EXPERIMENTS.md, a task in TASKS.md, and a row here. Add an option by giving it a letter in the options register below. Never delete a row; strike it through and say why.

Status vocabulary (same as TASKS.md): `done`, `ready`, `quiet-host`, `gpu-window`, `blocked`, `decision`, plus `partial` when a stage has run.

## Where we are (2026-09-21)

```diagram
P0 research ──▶ P1 host micro ──▶ P2 baseline ──▶ P3 2-GPU hybrid ──▶ P4 3/4-GPU
   DONE          partial            waiting        design only        blocked
                 (E1, E3 done;      (GPU window)   (E6 s0 script      (cards, power)
                  E6 s0, E1b, E2,                   ready; E13 s1
                  E4 waiting)                       source next)
```

Blocking condition for everything below marked `quiet-host` or `gpu-window`: the GLM container `glm53-flash-2x-rtxpro6000` is serving and benchmarking on both GPUs. Owner announces windows.

## Experiment readout

| E | Name | Needs | Status | Result so far | Decision fed |
|---|---|---|---|---|---|
| E1 | Host DRAM bandwidth | quiet host | partial | 146.2 GB/s median at 32 workers, Python memcpy lower bound, **measured while GLM was serving** | `--dram-gbps 146` default → 56 tok/s hybrid ceiling. Re-run on a quiet host (E1b, torch read-only) before trusting the ceiling |
| E2 | GPU memory bandwidth | GPU | gpu-window | — | Sanity check for GDDR7 1.79 TB/s spec |
| E3 | Engram gather DRAM vs NVMe | quiet host | done | DRAM 4.15 M rows/s (86K tok-equiv/s); 9100 PRO 403 K rows/s (8.4K tok-equiv/s); T705 347 K rows/s | Engram default = NVMe + 64 GiB DRAM cache; NVMe gather is not the decode bottleneck at the hybrid's 56 tok/s |
| E4 | PCIe H2D/D2H per GPU | GPU | gpu-window | — | `--pcie-gbps-per-gpu 50` assumption; sets PCIe-streaming fallback (42 tok/s calc) and layerwise prefill cost |
| E5 | V4.1 on 2 GPUs, stock generic offload | GPU + 475 GiB download | gpu-window | — | Evidence that a MoE-aware engine is required (expected OOM/unusable) |
| E6 s0 | CPU expert GEMV, 16 cores | quiet host | quiet-host (script ready) | — | Whether CPU compute or DRAM bandwidth bounds the hybrid; 15 ms/token compute estimate vs 12.9 ms DRAM |
| E6 s1 | sglang-kt hybrid, V4-Flash, 2 GPUs | GPU | gpu-window | — | Toolchain proof on SM120; calibrates `--gpu-ms-per-token` and `--dram-gbps` against a real hybrid |
| E6 s2 | Port `deepseek_v41` into sglang-kt; first V4.1 serve on 2 GPUs | build, then GPU | design doc next (`plan/E6-DESIGN.md`) | — | The primary deliverable |
| E7 | Routing-skew histogram | GPU | gpu-window | — | `--activation-skew`; frequency placement gain; whether Option D (pruning) is worth anything |
| E8 | Baseline re-measure V4-Flash NVFP4 TP2 | GPU | gpu-window | prior repo: 200.6 tok/s single-stream, 226 at C2, 1M needle OK (not this harness) | The number every 2-GPU result is compared with |
| E8b | tok/s per watt curve (power caps) | GPU | gpu-window | — | T4.0 power decision (2,000 W PSU vs 225 W cap) |
| E9 | 4-GPU V4.1 qualification | 4 cards | blocked | calculator: 100% experts resident, ~200 tok/s, Engram-in-DRAM fits | Option A profile in `serve.sh` (unqualified) |
| E10 | DSpark on/off under CPU-bound MoE | GPU, after E6 s1 | blocked on E6 s1 | — | Whether speculative decoding pays when the CPU is the bottleneck |
| E11 | Engram tiering / quantization A/B | quiet host, then GPU | quiet-host (sim) | calculator: FP8 tables in DRAM do not fit 2-GPU (−78 GiB); FP4 +13 GiB, q3_K +32 GiB headroom | Engram mode per GPU count; first quality number for quantized tables |
| E12 | Kernel census on SM120 (nsys) | GPU | gpu-window | — | Is the 5 ms/token GPU figure a Marlin W4A16 number? DSpark acceptance under that path |
| E13 s1 | MXFP4 grouped GEMM on SM120 (CUTLASS 79d type swap; W4A4 and W4A8 variants) | build GPU-free, run GPU | ready to write source; format question **answered from source** | SM120 takes MXFP4/UE8M0 block-32 natively (`kind::mxf4nvf4 … 2X … ue8m0`, `mx_float4_t`, shipped dense `sm120_bs_gemm_mxf4_mxf4` test); no re-quant to NVFP4. Grouped (PtrArray) mxf4×mxf4 has no shipped test — that is what s1 compiles and runs | Kernel is lossless w.r.t. the checkpoint; build for `sm_120a`, CUDA ≥ 12.8 (image has 13.0), CUTLASS v4.7.1 cloned to `~/dev/src/cutlass` |
| E13 s2 | Integrate winning kernel into the E6 engine | build, GPU | decision after s1 | — | Re-run E12 |

## Kernel experiment menu (E13 sub-items and successors; pick by measured need)

Ordered by how directly each moves a number in the readout above.

| ID | Experiment | Hypothesis | Cost | Status |
|---|---|---|---|---|
| K1 | mxf4×mxf4 grouped PtrArray GEMM on SM120 (E13 s1, W4A4) | CUTLASS's builder accepts it; runs at V4.1 expert shapes; ≥ 1.3× Marlin at decode M | source + compile now (GPU-free); run in window | next |
| K2 | mxf8×mxf4 grouped GEMM (E13 s1, W4A8) | Lower activation-quant error than K1 at similar speed; `kind::mxf8f6f4` path proven by example 79c and the shipped `mxf8_mxf4` grouped test | same build | next |
| K3 | Tile sweep under 99 KiB smem | 128×128×128 (79d) vs 128×64×256 / 128×32×256 (the mxf4 unit-test tiles) for decode M ≤ 64 | compile variants | after K1 |
| K4 | Cooperative vs pingpong schedule at decode shapes | Pingpong wins at small M | free (79d has both) | with K1 |
| K5 | Marlin W4A16 reference on the same shapes (`VLLM_MOE_FORCE_MARLIN=1`, inside `dsv4-nvfp4:recipe`) | Gives the denominator for K1–K4 | GPU window | with K1 |
| K6 | Activation quantization kernel (bf16 → MXFP4/MXFP8 block-32, fused with gather) | Needed to feed K1/K2 in a real engine; must not eat the win | write after K1 | after K1 |
| K7 | Hybrid glue: pinned-host activation gather/scatter with stream-ordered H2D/D2H overlapped with CPU expert compute (TK.5) | Overlap hides the PCIe hop entirely at batch 1–8 | after E6 s1 | blocked |
| K8 | Engram row gather from pinned host (zero-copy) vs GPUDirect Storage from the 9100 PRO | Zero-copy beats staging for 264-byte rows | after E6 s1 | blocked |
| K9 | Sub-4-bit / pruned experts | No SM120 kernels; explicitly out of scope unless E7 shows strong skew | — | parked |

## Options register

| Option | What | Fits 2 GPUs? | Calculator ceiling | Status |
|---|---|---|---|---|
| A | 4-GPU GPU-resident (SGLang `dev-dsv41` TP4, Engram in DRAM) | no (needs 4) | ~200 tok/s | profile written, unqualified; blocked on cards + power |
| B | **2-GPU three-tier hybrid**: VRAM 58% experts + attention, CPU/DRAM 42% experts, Engram NVMe + cache | yes (200.1 GiB host need, +46.7 margin) | 56 tok/s @146 GB/s; 89 @300 | primary; engine port not started |
| B' | B with PCIe expert streaming instead of CPU compute | yes | 42 tok/s | fallback if E6 s0 shows CPU-bound |
| C | Smaller Engram in DRAM (FP4/q3_K) on 2 GPUs | yes | same as B, better TTFT | E11 |
| D | Expert pruning to fit resident on 2 GPUs | only with heavy pruning | — | research-grade; needs E7 skew |
| E | 3-GPU EP3/PP, 91% experts resident | needs 3 | ~126 tok/s @146; 156 @300 | profile to write (T4.1b) |
| Baseline | V4-Flash-0731 NVFP4 TP2 (qualified) | yes | 200.6 tok/s measured | re-measure with this harness (E8) |
| W4A4 vs W4A8 | Activation format for E13 kernels | n/a | — | decided by K1/K2 error and speed |

## Standing assumptions to re-check when data lands

| Assumption | Value | Replaced by |
|---|---|---|
| GPU ms/token for resident part of the hybrid | 5.0 (Marlin-path guess) | E12 |
| Host DRAM read bandwidth | 146 GB/s (contended, lower bound) | E1b |
| PCIe per GPU | 50 GB/s | E4 |
| Host reserve | 24 GiB | — (policy) |
| Engram DRAM cache | 64 GiB | E11 |
| CPU expert compute | ~15 ms/token estimate | E6 s0 |

## Change log

- 2026-09-21 — Page created. E13 format question resolved from PTX ISA + CUTLASS source (no re-quant needed). CUTLASS v4.7.1 cloned to `~/dev/src/cutlass` for the GPU-free E13 build. Kernel menu K1–K9 added.

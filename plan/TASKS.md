# Tasks

Status: `done`, `ready` (can run now), `quiet-host` (GPU-free but needs the host idle: no GLM benchmark running, since CPU/DRAM/NVMe numbers are contended otherwise), `gpu-window` (needs the GLM container stopped), `blocked` (external dependency), `decision` (owner call).

## P0 — Research and data

| ID | Task | Status | Output |
|---|---|---|---|
| T0.1 | Read shard headers of the three V4.1-Flash checkpoints; classify bytes per component | done | `evidence/budget-*.json`, `scripts/checkpoint-budget.py` |
| T0.2 | Hardware inventory without identifiers | done | `evidence/hardware-inventory.json`, `scripts/hardware-inventory.sh` |
| T0.3 | Feasibility calculator; validate against the 0xSero 4-GPU layout | done | `scripts/feasibility-calc.py`, FEASIBILITY.md §3 |
| T0.7 | Extend the calculator with the three-tier `hybrid-cpu` layout, Engram tiering modes, 2/3/4-GPU comparison and decode ceilings; tests | **done** | FEASIBILITY.md §3a, README verdict |
| T0.8 | Research CPU-expert engines: KTransformers sglang-kt (V4-Flash only, MXFP4 on AVX2), llama.cpp `deepseek41` branch, community 2-GPU attempts | **done** | RESEARCH.md |
| T0.4 | Audit the source document's claims against primary sources | done | FEASIBILITY.md §6, RESEARCH.md |
| T0.5 | DIMM part number, rated and configured speed, CAS latency | **done** (KF564R32-32 × 8, SPD 4800, configured 6400 = EXPO; CL not exposed by SMBIOS, vendor CL32) | HARDWARE.md memory table |
| T0.6 | Record PSU model and wall circuit capacity | **done** (1,500 W; circuit still unrecorded) — verdict: 4 × 300 W does not fit, see T4.0 | HARDWARE.md power table |

## P1 — Host microbenchmarks

| ID | Task | Status | Output |
|---|---|---|---|
| T1.1 | E3 Engram gather: DRAM, nvme0, nvme1 | **done** | `evidence/e3-engram-gather-*.json` |
| T1.2 | E1 DRAM bandwidth | **done** | `evidence/e1-dram-bandwidth.json` |
| T1.3 | E4 PCIe H2D/D2H per GPU and both concurrently | **done** — 53 GiB/s per card, 106 GiB/s both | `evidence/e2-e4-gpu-pcie-bandwidth.json` |
| T1.4 | E2 GDDR7 device bandwidth | **done** — ≈1,470 GB/s D2D | `evidence/e2-e4-gpu-pcie-bandwidth.json` |
| T1.5 | Recompute the hybrid ceilings from T1.3/T1.7/T1.8 (`--dram-gbps`, `--pcie-gbps-per-gpu`, CPU ms/token); update FEASIBILITY.md §3a | **done** — defaults 125 / 57; CPU 50, PCIe 46, split 78 tok/s | FEASIBILITY.md §3a, README |
| T1.7 | E6 stage 0: `bench-cpu-expert.py` CPU expert GEMV in the torch container, GPU-free | **done** — bf16 GEMV 76 ms/token = 94 GB/s at 16 thr; `read`/`int8-deq` variants invalid (see E6) | `evidence/e6-stage0-cpu-expert-gemv.json` |
| T1.8 | E1b: torch DRAM read bandwidth (`scripts/bench-dram-read.py`), replacing the Python lower bound | **done** — 125 GB/s read at 16 thr; CCD-fabric limit | `evidence/e1b-dram-read-bandwidth.json` |
| T1.9 | E4b: CPU read vs GPU DMA contention on host DRAM (`scripts/bench-dram-contention.py`) | **done** — additive (0.98), 234 GB/s combined → Option B2 split | `evidence/e4b-dram-contention-cpu-vs-dma.json` |
| T1.6 | Hugepage reservation test: `vm.nr_hugepages` for ~190 GiB of 2 MiB pages while GLM is running; confirm allocation succeeds and revert | ready (needs root) | TROUBLESHOOTING.md note |

## P2 — Baseline

| ID | Task | Status | Output |
|---|---|---|---|
| T2.1 | Port fixture/method from the GLM and V4-Flash recipes into `bench-decode.py` / `bench-prefill.py` so numbers are comparable | done (harness written) | `scripts/bench-*.py` |
| T2.2 | E8 run against the V4-Flash NVFP4 TP2 deployment | **done** (incl. DSpark-off 89 tok/s and 1M prefill 1,898 tok/s) — decode C1–C8, prefill 8K–512K, 222K needles, power samples; DSpark-off and 1M prefill rows not re-run | `evidence/baseline-v4-flash/`, BENCHMARKS.md |
| T2.2b | E8b power sweep (325/300/275/250 W; 225 W is below the driver floor) | **done** — decode 216–229 tok/s at every cap, prefill 8,770/8,428/7,991/7,482 tok/s; see BENCHMARKS.md §E8b | `evidence/e8b-power-sweep/summary.json` |
| T2.3 | Fill BENCHMARKS.md baseline tables from JSON | after T2.2 | `docs/BENCHMARKS.md` |
| T2.4 | Baseline improvement sweep on V4-Flash: DSpark block 3/5/7, FP8 KV, C8/C16 | gpu-window | BENCHMARKS.md |
| T2.5 | E14 model-to-model table: GLM-5.3-Flash (owner's reference recipe) vs V4-Flash vs V4.1 on this box, same harness/fixtures/power policy | scoped; GLM column needs a GPU window with the GLM container up; V4-Flash column needs C16 + footprint only | `evidence/e14-*/`, BENCHMARKS.md §E14 |

## P3 — 2-GPU hybrid build

| ID | Task | Status | Output |
|---|---|---|---|
| T3.1 | Download `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…` (475 GiB) to nvme0 | ready (disk: 1.7 TB free) | HF cache |
| T3.2 | Pull `lmsysorg/sglang:dev-dsv41` and `approachingai/ktransformers:DSV4-specific`; record digests in PROVENANCE.md | ready | PROVENANCE.md |
| T3.3 | E5 2-GPU attempt with stock SGLang and vLLM generic offload; scrub and record logs | gpu-window, after T3.1–T3.2 | TROUBLESHOOTING.md |
| T3.4 | E6 stage 1: sglang-kt hybrid with V4-Flash on 2 GPUs; sweep `--kt-num-gpu-experts`; calibrate the calculator | gpu-window, after T3.2 | `results/e6-stage1.json` |
| T3.5 | E6 stage 2 design doc: `deepseek_v41` in sglang-kt (CSA2, Engram host table, hyper-connections, DSpark, MXFP4 format mapping) | ready (GPU-free) | `plan/E6-DESIGN.md` |
| T3.6 | E6 stage 2 implementation and first V4.1-Flash serve on 2 GPUs | after T3.5, gpu-window | fork + BENCHMARKS.md |
| T3.7 | llama.cpp `runtime/deepseek41` build; watch ggml-org/llama.cpp#28696 and CSA2; run Q4_K_M with `-ot` offload when it lands | ready to build, blocked on CSA2 | BENCHMARKS.md |
| T3.8 | E7 routing-skew histogram on V4-Flash; feed `--activation-skew` | gpu-window | `evidence/e7-routing/` |
| T3.9 | E10 DSpark on/off under the hybrid | after T3.4 | BENCHMARKS.md |
| T3.10 | E11 Engram tiering: cache-hit simulation (GPU-free) then NVMe vs FP4/q3_K A/B | quiet-host / after T3.6 | BENCHMARKS.md |
| T3.11 | `serve.sh` profile `2gpu-hybrid` (KT flags, Engram NVMe, hugepages) — labelled unqualified until T3.6 | after T3.5 | `config/example.env`, `scripts/serve.sh` |

## Kernel track (cross-cutting)

| ID | Task | Status | Output |
|---|---|---|---|
| TK.1 | Research SM120 FP4 kernel status (CUTLASS SM120 block-scaled GEMM, FlashInfer/vLLM Marlin fallback, TRT-LLM Triton MoE, cuBLAS gap, smem/TMEM constraints) | **done** | RESEARCH.md |
| TK.2 | E12 kernel census on the V4-Flash baseline: nsys profile + backend log grep; measured GPU ms/token breakdown into the calculator | **done** (from the sibling's torch-profiler traces; nsys cannot attach) — experts on CUTLASS SM120 grouped GEMM, not Marlin; 15.3 ms/verify step; 5 ms/token confirmed conservative. Open: prefill census, V4.1 engines | `scripts/kernel-census.py`, `evidence/e12-kernel-census-v4-flash-tp2.json`, BENCHMARKS.md |
| TK.2b | Settle MXFP4-UE8M0 vs NVFP4-UE4M3 acceptance on SM120 from PTX ISA + CUTLASS source (open question 9) | **done** — native; type swap `nv_float4_t` → `mx_float4_t`, build `sm_120a` | RESEARCH.md "MXFP4 is native" |
| TK.3 | E13 stage 1: mxf4×mxf4 grouped GEMM from example 79d (W4A4) and mxf8×mxf4 (W4A8) microbench at V4.1 expert shapes vs Marlin, with dequant-reference correctness | **done** — W4A4 1,421–1,451 GB/s at a 451 MB working set (97–99 % of E2), W4A8 2× slower at decode, census-shaped TP2 runs tie the deployed vLLM kernel; Marlin denominator dropped (moot per E12). Still open, moved to TK.3b: activation-error check |
| TK.3b | E13 step 4: bf16-activation vs MXFP4/FP8-activation error on real V4.1 expert weights | open | `kernels/e13/`, EXPERIMENTS.md | `kernels/e13/`, `evidence/e13-stage1-*.json` |
| TK.4 | E13 stage 2: integrate the winning kernel into the E6 engine's MoE backend; re-run E12 | decision, after TK.3 | fork + BENCHMARKS.md |
| TK.5 | Hybrid glue kernels: activation gather/scatter to pinned host, stream-ordered H2D/D2H overlapped with CPU expert compute; Engram row gather from pinned host (zero-copy) or GPUDirect Storage from the 9100 PRO | after E6 stage 1 | fork |

## P4 — 3/4-GPU comparison

| ID | Task | Status | Output |
|---|---|---|---|
| T4.0 | Power decision for four cards: 2,000 W-class PSU (needs 240 V or 30 A circuit) **or** 250 W cap (driver floor) on the 1,500 W PSU | **decided by E8b: 250 W cap** for experiments — costs 0 % decode, 11 % prefill, best tok/s/W (15.0). 2,000 W PSU only if sustained 4-card serving is wanted (four cards at 250 W + 200 W CPU PPT ≈ 1,300 W, 87 % of the supply) | HARDWARE.md power table |
| T4.1 | Acquire and install one or two more RTX PRO 6000 (Max-Q preferred for power); verify P2P OK and x16 on all slots | blocked on T4.0 | HARDWARE.md |
| T4.1b | 3-GPU profile: EP3 or PP layout (TP3 invalid), 91% experts resident + hybrid remainder; llama.cpp layer-split as the simple variant | after T4.1 with 3 cards | `scripts/serve.sh` |
| T4.2 | `preflight.sh` passes the memory gate on 4 GPUs | after T4.1 | — |
| T4.3 | E9 serve + verify + benchmark, Engram-in-DRAM and NVMe modes | after T4.2 | BENCHMARKS.md, PROVENANCE.md pins |
| T4.4 | README verdict update; systemd unit qualified | after T4.3 | README.md |

## Standing rules

- Before any `gpu-window` task: `docker stop glm53-flash-2x-rtxpro6000`; after: `docker start` it and confirm `/health`.
- Before any `quiet-host` task: confirm no benchmark is running against the GLM deployment (`uptime` load < 2, `nvidia-smi` utilisation idle); the owner announces the window. Torch benches run in `dsv4-nvfp4:recipe` without `--gpus`.
- Every JSON promoted from `results/` to `evidence/` goes through `scripts/ci.sh`'s identifier scan.
- No pushes from automation. Commits are local until the owner says otherwise.

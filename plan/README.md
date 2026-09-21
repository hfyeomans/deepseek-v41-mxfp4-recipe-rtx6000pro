# Plan: DeepSeek-V4.1-Flash FP4 on RTX PRO 6000 Blackwell

## Goal

Produce a revision-pinned, benchmarked recipe for the best DeepSeek FP4 model this host can serve, with honest statistics, and push the 2-GPU boundary as far as it measurably goes. "Best" means: V4.1-Flash if it can be made to run; otherwise the qualified V4-Flash-0731 NVFP4 baseline, improved.

## Scope decision (Phase 0 result)

The memory math in [docs/FEASIBILITY.md](../docs/FEASIBILITY.md) rules out V4.1-Flash as a GPU-resident model on two RTX PRO 6000 (shortfall 112 GiB, 42% of the routed experts) and rules **in** a three-tier layout: VRAM for attention + 58% of the experts, host DRAM + CPU compute for the other 42% (112 GiB), NVMe + DRAM cache for the Engram tables. No engine runs that layout for `deepseek_v41` today, so the primary track is a build with a measured ceiling (~56 tok/s decode at the 146 GB/s DRAM bandwidth measured so far; ~89 at 300 GB/s). Three tracks run in parallel and share one benchmark harness:

| Track | Outcome | Status |
|---|---|---|
| **Baseline** | V4-Flash-0731 NVFP4 TP2 re-measured with this repo's harness — the number the hybrid competes with | scripts ready; needs a GPU window |
| **2-GPU hybrid (primary)** | An engine that runs the three-tier layout for V4.1-Flash on this host, with decode/prefill/needle numbers. Routes, in order: (a) port `deepseek_v41` into KTransformers' sglang-kt; (b) llama.cpp `deepseek41` branch once CSA2 lands; (c) own PCIe expert-streaming prototype | microbenchmarks runnable now; port not started |
| **Kernels (cross-cutting)** | Know which GPU kernels each engine actually dispatches on SM120 for this model family (native block-scaled FP4 vs Marlin W4A16 dequant), measure the GPU ms/token they cost, and build custom CUDA kernels where the stock path is a fallback: MXFP4 grouped GEMM sized for SM120's 99 KiB shared memory, hybrid glue (activation gather/scatter to pinned host, overlapped H2D/D2H), Engram row gather from pinned host or GPUDirect Storage | research done; E12 census needs a GPU window; E13 CUTLASS microbench buildable GPU-free, runs in a window |
| **3/4-GPU comparison** | EP3/PP profile for a third card (91% experts resident, ~126 tok/s) and the SGLang `dev-dsv41` TP4 + Engram-in-DRAM profile for four (~200 tok/s), both ready to qualify when cards and power exist | `serve.sh` 4-GPU profile written, unqualified; 3-GPU profile to write |

## Phases

```diagram
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ P0 Research & data │──▶│ P1 Host micro-     │──▶│ P2 Baseline        │──▶│ P3 2-GPU hybrid    │──▶│ P4 3/4-GPU         │
│ checkpoint budget, │   │ benchmarks (no     │   │ V4-Flash NVFP4 TP2 │   │ build: sglang-kt   │   │ comparison         │
│ hardware, sources, │   │ model): DRAM, NVMe,│   │ decode/prefill/    │   │ dsv41 port, llama  │   │ (needs cards and   │
│ feasibility math   │   │ CPU GEMV, PCIe H2D │   │ 1M needle          │   │ .cpp, streaming    │   │ a power decision)  │
│ DONE               │   │ E1/E3 done         │   │ needs GPU window   │   │ E5/E6/E7/E10/E11   │   │ blocked on hardware│
└────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘
```

### P0 — Research and data (done)

- Exact byte budget of the three V4.1-Flash checkpoints from shard headers ([`evidence/budget-*.json`](../evidence/)).
- Hardware inventory ([`evidence/hardware-inventory.json`](../evidence/hardware-inventory.json)); DIMMs verified (8 × Kingston FURY Renegade Pro KF564R32-32 at 6400 MT/s); PSU 1,500 W.
- Feasibility calculator with the 0xSero 4-GPU layout as the calibration point, extended with the `hybrid-cpu` layout (GPU/CPU expert split, Engram tiering, DRAM and PCIe decode ceilings) and 2/3/4-GPU comparison.
- Source audit of the Google Doc: two correct claims, four nonexistent flags, one wrong-model size. See FEASIBILITY.md §6 and [RESEARCH.md](RESEARCH.md).

Exit criterion met: a placement table that says, per layout and GPU count, fits or does not, with margins.

### P1 — Host microbenchmarks (no model weights, no GPU contention except E2/E4)

E1 DRAM bandwidth (done: 146 GB/s Python lower bound; C/STREAM follow-up pending), E3 Engram-style random gather from DRAM and both NVMe drives (done), **E6 stage 0 CPU expert GEMV throughput** (GPU-free, in a torch container without `--gpus`), E2 GPU memory bandwidth, E4 PCIe host-to-device bandwidth per GPU. These replace every *vendor spec* and *estimate* row in HARDWARE.md with a measurement and set the hybrid's decode ceiling.

Exit criterion: HARDWARE.md has no "estimate" rows left; FEASIBILITY.md §3a is recomputed with measured `--dram-gbps` and a measured CPU compute time per token.

### P2 — Baseline

Run this repo's `bench-decode.py`, `bench-prefill.py` and `verify.py` against the already-qualified `deepseek-v4-flash-nvfp4-tp2` deployment so that every later number is directly comparable (same fixture, same method, same power condition).

Exit criterion: `docs/BENCHMARKS.md` baseline tables filled from `results/` JSON produced by this harness, not copied from the other repo.

### P3 — 2-GPU hybrid build (the "push" track)

- E5: load V4.1-Flash on 2 GPUs with the generic CPU offload of stock SGLang/vLLM and document the exact failure mode or throughput. Expected: OOM or unusable. Purpose: evidence that the hybrid needs a MoE-aware engine.
- E6: the hybrid itself. Stage 0 (GPU-free): CPU expert GEMV throughput on 16 cores. Stage 1 (GPU): KTransformers sglang-kt with **V4-Flash** on 2 GPUs — proves the toolchain on SM120 and gives a hybrid-vs-resident delta on a model that also runs resident here. Stage 2: port `deepseek_v41` into sglang-kt (CSA2 attention, Engram host table, hyper-connections, DSpark) and serve V4.1-Flash with 58% of experts on GPU. Parallel: build llama.cpp `runtime/deepseek41` and run the Q4_K_M GGUF with `-ot` expert offload as soon as CSA2 lands.
- E7: routing-skew histogram (feeds `--activation-skew` and expert placement, and decides Option D).
- E10: DSpark on/off under the CPU-bound hybrid.
- E11: Engram tiering A/B: NVMe + cache vs FP4/q3_K tables in DRAM; quality check of the quantized tables.

Exit criterion: a V4.1-Flash decode/prefill/needle table on 2 GPUs in BENCHMARKS.md with a pinned engine build, or a documented reason none of the three routes could run — including negative results.

### P4 — 3/4-GPU comparison (blocked on hardware and power)

Decide the power path (T4.0: 2,000 W PSU or 225 W cap). With a third card: EP3 or PP profile, 91% experts resident, hybrid for the rest. With a fourth: `preflight.sh` passes, `serve.sh` 4-GPU SGLang `dev-dsv41` profile with Engram in DRAM, full verify + benchmark suite, compared with 0xSero's numbers, the P2 baseline and the P3 hybrid.

Exit criterion: one table with 2-, 3- and 4-GPU rows for the same fixture and power condition.

## Innovation candidates (ranked by measurability)

1. **Three-tier V4.1-Flash on two GPUs** (Option B): first published 2-GPU number for this model. Measurable as decode tok/s vs the calculator's ceiling, prefill tok/s with layerwise GPU prefill, and needle accuracy at 262K. The engine port is the risk; the memory and bandwidth arithmetic is done.
2. **Frequency-based expert placement** on top of (1): keep the hottest 58% of experts on the GPU instead of an arbitrary 58%. Gain = measured skew from E7; zero if routing is uniform.
3. **Engram fully in DRAM on a 256 GB host** (4-GPU variant): nobody has published a consumer-GPU V4.1 result with the whole FP8 Engram table DRAM-resident. Measurable as prefill/TTFT delta vs NVMe mode.
4. **Engram table tiering and quantization** (E11): NVMe + DRAM row cache vs FP4/q3_K tables in DRAM; the first quality measurement of quantized Engram tables.
5. **Expert pruning for 2-GPU fit** (Option D): research-grade; only if E7 shows skew.
6. **Custom SM120 kernels** (E12/E13): native MXFP4 grouped GEMM for the GPU-resident experts where every engine today falls back to Marlin W4A16 (40–50% slower, and it shifts DSpark acceptance because draft heads were trained on native FP4 activations); hybrid dispatch kernels that overlap CPU expert compute with GPU work; Engram gather straight from pinned host memory or NVMe via GPUDirect Storage. Measurable as GPU ms/token before/after under the same fixture, and as DSpark acceptance rate.
7. **Baseline improvements** that need no new hardware: DSpark block size sweep, FP8 KV, concurrency scaling to C8/C16 on the existing V4-Flash recipe.

## Non-goals

- Shipping a V4.1-Flash 2-GPU "recipe" that has not run. `preflight.sh` fails closed until a hybrid engine is verified here.
- Quantizing weights ourselves. Experts are already 4-bit; sub-4-bit has no SM120 kernels.
- Publishing any host identifiers. `scripts/ci.sh` scans for UUIDs, PCI IDs, home paths and LAN addresses.

**Running readout (update this first when a result lands): [ROADMAP.md](ROADMAP.md).** Detailed experiments: [EXPERIMENTS.md](EXPERIMENTS.md). Task list with owners and status: [TASKS.md](TASKS.md). Sources and open questions: [RESEARCH.md](RESEARCH.md).

# Learning guide

A reading map for every piece of this project, written for the owner. Each section says why the piece matters *for this build*, what to read (primary sources first), and where it shows up in this repo so you can connect the reading to a file you can run. Read in order the first time; afterwards jump to whichever experiment you are about to run and read its row in the last table.

Difficulty marks: ★ background, ★★ working knowledge needed to run and interpret experiments, ★★★ needed to write or modify kernels.

## 0. The one-paragraph map

We want a 475 GiB model on 192 GiB of VRAM. The model is a Mixture-of-Experts (MoE): most of its weights are "experts" of which only 6 of 384 fire per token, so weights that are not in VRAM only cost bandwidth when they are used. That turns the problem into memory-hierarchy engineering — VRAM (1.79 TB/s) → host DRAM (~150–300 GB/s) → NVMe (~14 GB/s, but 400K random rows/s) — and into kernel engineering, because the tensor cores on our RTX PRO 6000 (SM120) run 4-bit math natively only through a specific instruction that no serving engine uses yet on this card. Everything in `plan/` is one of those two threads.

## 1. The model: DeepSeek-V4.1-Flash

Why here: the byte budget in `evidence/budget-upstream-bf16fp8.json` and every number in `docs/FEASIBILITY.md` come from the model's structure. You cannot reason about offload without knowing which tensors are hot.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| Mixture-of-Experts routing (top-k gating, load balance, why only k experts cost compute) | ★ | Switch Transformers (Fedus et al., arXiv 2101.03961) §2; DeepSeekMoE (arXiv 2401.06066) for fine-grained + shared experts, which is the V4 lineage | `scripts/checkpoint-budget.py` classifier: how experts are counted (15,360 = 40 layers × 384) |
| Multi-head Latent Attention (MLA) and why the KV cache is tiny | ★★ | DeepSeek-V2 report (arXiv 2405.04434) §2.1 | `feasibility-calc.py` `kv_bytes` (890 B/token) |
| Sparse attention (DSA → CSA2 in V4.1): lightning indexer, top-k token selection | ★★ | DeepSeek-V3.2 technical report (DeepSeek Sparse Attention); V4.1 model card on Hugging Face `deepseek-ai/DeepSeek-V4.1-Flash` | RESEARCH.md "checkpoint facts"; the indexer weights are part of the non-expert resident set |
| Engram conditional memory (n-gram hash → table lookup, why it is 183 GiB and read-only) | ★★ | DeepSeek "Engram" paper (search arXiv, Jan 2026, "conditional memory via scalable lookup"); V4.1 model card `engram_*` config keys | E3 gather benchmark (`scripts/bench-engram-gather.py`): 264-byte rows, 48 rows/token |
| Multi-token prediction / DSpark speculative heads | ★★ | DeepSeek-V3 report (arXiv 2412.19437) §2.2 MTP; Leviathan et al. speculative decoding (arXiv 2211.17192) for acceptance-rate maths | E10; the 7.39 GiB DSpark tensors in the budget |
| Hyper-connections (replacing residual stream) | ★ | Zhu et al. "Hyper-Connections" (arXiv 2409.19606) | Relevant only to the E6 stage-2 port |
| Reading a safetensors header without loading weights | ★ | huggingface.co/docs/safetensors (format section: 8-byte length + JSON header) | `scripts/checkpoint-budget.py` reads only headers over HTTPS |

Exercise: run `python3 scripts/checkpoint-budget.py --help`, then reproduce the 268.9 GiB expert figure by hand from 15,360 experts × (4608×5120×2 + 5120×2304) × 0.5 byte + scales.

## 2. Number formats: FP4, block scaling, MXFP4 vs NVFP4

Why here: the experts are stored as E2M1 with one UE8M0 scale per 32 values. Whether a tensor core can consume that *exact* layout decided open question 9 and shapes E13.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| Floating-point mini-formats: E2M1 (FP4), E4M3/E5M2 (FP8), what mantissa/exponent bits buy | ★ | OCP "Microscaling Formats (MX) Specification v1.0" (opencompute.org), §5 element types — short and definitive | RESEARCH.md "Quantization as shipped" |
| Block scaling: shared per-block scale, UE8M0 (power-of-two, 8 bits) vs UE4M3 | ★★ | Same OCP spec §4–6 for MX (block 32, UE8M0); NVIDIA blog "Introducing NVFP4" for NVFP4 (block 16, UE4M3 + a per-tensor FP32 scale) | E13 procedure step 1 (`mx_float4_t` vs `nv_float4_t`) |
| Why MXFP4 and NVFP4 are *not* interchangeable numerically (re-quantising changes what the model was trained on) | ★★ | Compare the two block sizes and scale types above; then RESEARCH.md "MXFP4 is native" | Open question 9 (resolved) |
| Weight-only (W4A16) vs weight-and-activation (W4A4/W4A8) quantization; why Marlin dequantises | ★★ | MARLIN paper (arXiv 2408.11743) §1–3 | RESEARCH.md "Engines" (Marlin fallback); E13 W4A4/W4A8 variants |
| INT4-pair packing (two E2M1 nibbles per byte) | ★ | OCP spec packing section; `cutlass/include/cutlass/float_subbyte.h` | Budget classifier: "INT8-packed FP4 pairs" |

Exercise: write a 20-line Python function that quantises a 32-float block to MXFP4 (find max, pick power-of-two scale, round each value to the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}) and dequantises it back; plot the error. This is exactly what the E13 reference does.

## 3. The GPU: RTX PRO 6000 Blackwell, SM120

Why here: SM120 is *consumer* Blackwell. It shares the FP4 tensor-core instruction family with B200 (SM100) but not the memory system (no TMEM, no `tcgen05`, 99 KiB shared memory), so datacenter kernels do not run and engines fall back. Knowing the difference explains every "SM120 unsupported" issue in RESEARCH.md.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| GPU execution model: SMs, warps, blocks, shared memory, registers, occupancy | ★ | CUDA C++ Programming Guide ch. "Programming Model" and "Hardware Implementation" (docs.nvidia.com/cuda/cuda-c-programming-guide) | `docs/HARDWARE.md` GPU section |
| Tensor cores and `mma.sync`: what a 16×8×64 block-scaled MMA does per warp | ★★★ | PTX ISA (docs.nvidia.com/cuda/parallel-thread-execution) → "Warp Level Matrix Multiply-Accumulate Instructions" → block scaling section; find `kind::mxf4nvf4` | RESEARCH.md "MXFP4 is native" item 1 |
| SM100 vs SM120: TMEM, `tcgen05`, cluster launch, smem size; why "Blackwell" is two architectures | ★★ | NVIDIA Blackwell architecture whitepaper (datacenter) and the RTX Blackwell whitepaper (GeForce/PRO); Colfax "NVFP4 blockscaled GEMM on SM12x" (research.colfax-intl.com) | RESEARCH.md "Hardware/CUTLASS" |
| Compute capability targets: `sm_120`, `sm_120a` (arch-specific), `sm_120f` (family); what `-gencode` does | ★★ | CUDA Compiler Driver NVCC guide, "GPU Compilation" → "Feature-specific / family-specific" targets | E13 build flag decision (`compute_120a,code=sm_120a`) |
| Memory bandwidth and the roofline model (are we bound by FLOPs or bytes?) | ★★ | Williams, Waterman, Patterson "Roofline" (CACM 2009); NVIDIA "GPU Performance Background User's Guide" | E2; every "ceiling" number in `feasibility-calc.py` is a roofline |
| PCIe Gen5 x16 realities (~50–60 GB/s per direction, pinned vs pageable, P2P) | ★★ | CUDA Best Practices Guide "Data Transfer Between Host and Device" | E4 (`scripts/bench-pcie-h2d.py`) |

Exercise: after E2 and E4 run, put GDDR7, PCIe and DRAM bandwidth on one log-scale bar chart. That picture *is* the three-tier design.

## 4. CUTLASS and CuTe

Why here: the only production-quality open FP4 GEMM code for SM120 is CUTLASS. E13 is a small edit of a CUTLASS example, but reading it cold is hard: it is template metaprogramming over a layout algebra (CuTe). Budget real time here if you want to modify kernels rather than only run them.

Local checkout: `~/dev/src/cutlass` (v4.7.1). Docs are in `media/docs/` in that tree and at docs.nvidia.com/cutlass.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| What a GEMM kernel does: tiling, shared-memory staging, register blocking, software pipelining | ★★ | Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance" (siboehm.com) — the best 2-hour intro; then NVIDIA "CUTLASS: Fast Linear Algebra in CUDA C++" blog | — |
| CuTe layouts: shapes, strides, `Layout`, `Tensor`, composition, `local_tile`, `partition`, MMA atoms | ★★★ | `media/docs/cpp/cute/00_quickstart.md` → `01_layout.md` → `02_layout_algebra.md` → `03_tensor.md` → `0t_mma_atom.md` → `0x_gemm_tutorial.md` in the CUTLASS tree | `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA` in 79d: how the scale-factor tensor is laid out |
| CUTLASS 3.x structure: `CollectiveBuilder`, mainloop / epilogue collectives, `GemmUniversal`, dispatch policies and kernel schedules (TMA warp-specialized, cooperative vs pingpong) | ★★★ | `media/docs/cpp/cutlass_3x_design.md`, `media/docs/cpp/gemm_api_3x.md`, `media/docs/cpp/pipeline.md`; Colfax "CUTLASS Tutorial: Mastering the NVIDIA Tensor Memory Accelerator (TMA)" | `examples/79_blackwell_geforce_gemm/79d_*.cu` lines defining `CollectiveMainloop`, `CollectiveEpilogue`, `GemmKernel` |
| Block-scaled GEMM in CUTLASS: `OpClassBlockScaledTensorOp`, `nv_float4_t` / `mx_float4_t` pairs, SF vector size, `blockscaled_type`, `select_instr` | ★★★ | `include/cutlass/gemm/collective/builders/sm1xx_common.inl` (namespace `detail::blockscaled`), `sm120_blockscaled_mma_builder.inl`; `test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32.cu` as the smallest working MXFP4 instance | RESEARCH.md items 3–4 of "MXFP4 is native" |
| Grouped GEMM (one launch, many problems — the MoE primitive): `GroupProblemShape`, pointer arrays, device-side TMA descriptor modification | ★★★ | 79d header comment and `run()`; `media/docs/cpp/grouped_scheduler.md`; `media/docs/cpp/blackwell_functionality.md` for the SM120 column of the support matrix; CUTLASS 4.3 release notes (MoE grouped GEMM API) | E13 step 3 (decode vs prefill problem lists) |
| Epilogue fusions (`LinearCombination`, `LinCombBlockScaleFactor`) | ★★ | `include/cutlass/epilogue/fusion/operations.hpp` | E13 step 1: replacing FP4-output fusion with bf16 |
| Building CUTLASS examples: CMake `-DCUTLASS_NVCC_ARCHS=120a`, or standalone `nvcc -I include -I tools/util/include` | ★★ | `README.md` "Building" in the CUTLASS tree; `examples/79_blackwell_geforce_gemm/CMakeLists.txt` | `kernels/e13/build.sh` (to be written) |

Exercise: build unmodified `79d` for `sm_120a` inside `dsv4-sm120:cuda-devel-https` (no GPU needed to compile), then read the `--help` it prints. Then change one type and see what the compiler says — the `static_assert` messages in `sm1xx_common.inl` are the fastest way to learn what the builder allows.

## 5. Writing and profiling CUDA kernels

Why here: the kernel track (E12/E13, K1–K8) needs you to read a profile and tell native FP4 from a Marlin dequant path, and to know when a kernel is bandwidth-bound.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| CUDA C++ basics: kernels, streams, events, pinned memory, `cudaMemcpyAsync` overlap | ★ | CUDA C++ Programming Guide ch. 1–3; "An Even Easier Introduction to CUDA" (NVIDIA blog) | K7 hybrid glue design |
| Nsight Systems: timeline, kernel names, CPU-GPU overlap, `nsys profile --trace=cuda,nvtx` | ★★ | docs.nvidia.com/nsight-systems user guide, "CLI profiling" | E12 procedure |
| Nsight Compute: per-kernel roofline, memory throughput, achieved occupancy | ★★★ | docs.nvidia.com/nsight-compute, "Kernel Profiling Guide" | E13 speed analysis |
| Reading SASS/PTX to confirm which MMA instruction ran (`cuobjdump -sass`, grep `MMA` variants) | ★★★ | CUDA Binary Utilities guide | Confirming `kind::mxf4nvf4` in the compiled E13 binary |
| Inline PTX in CUDA (`asm volatile`) | ★★★ | "Inline PTX Assembly in CUDA" guide | `cute/arch/mma_sm120.hpp` |
| Community course | ★★ | GPU MODE lecture series (YouTube; "Lecture 1: Profiling", "CUTLASS" lectures) | — |

## 6. Serving engines and how they schedule MoE

Why here: three engines appear in the plan (SGLang `dev-dsv41`, vLLM, KTransformers' sglang-kt, plus llama.cpp). The E6 port is against sglang-kt; the baseline is SGLang; E5 uses stock offload in SGLang/vLLM.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| Continuous batching, prefill vs decode, KV cache paging | ★ | vLLM paper (Kwon et al., arXiv 2309.06180) §3–4; SGLang paper (arXiv 2312.07104) for RadixAttention | `scripts/bench-decode.py` / `bench-prefill.py` measure exactly these two phases |
| Tensor parallel vs expert parallel vs pipeline parallel; why TP3 is invalid for 64 heads | ★★ | Megatron-LM (arXiv 1909.08053) §3 for TP; GShard (arXiv 2006.16668) for EP | `feasibility-calc.py` `tp_valid`; open question 8 |
| KTransformers: CPU experts + GPU attention, `--kt-num-gpu-experts`, frequency placement | ★★ | github.com/kvcache-ai/ktransformers → `doc/en/DeepSeek-V4-Flash.md` and the sglang-kt docs | E6 stage 1 flags |
| llama.cpp expert offload (`-ot` tensor override regex) and GGUF quant types (Q4_K_M, q3_K) | ★★ | github.com/ggml-org/llama.cpp `docs/` and the `-ot` help text; GGUF spec in `ggml/docs/gguf.md` | T3.7; E11 q3_K Engram option |
| **CUDA graphs**: why a decode step of a few hundred tiny kernels is launch-bound, what `--max-cudagraph-capture-size` / `cuda_graph_sizes` capture (one graph per batch size; speculative steps need size = seqs × (K+1)), why capture needs a fixed memory pool, and why a CPU call inside the step (KTransformers experts) breaks or must be re-expressed inside the graph | ★★ | NVIDIA blog "Getting Started with CUDA Graphs" (2019); CUDA Programming Guide §"CUDA Graphs"; vLLM docs `design/cuda_graphs` (piecewise capture); KTransformers `ktransformers/util/cuda_graph_runner.py` for the CPU-expert pattern | GLM `serve.sh` line 55 (`96` = 16 seqs × 6); sibling `.env` `MAX_CUDAGRAPH_CAPTURE_SIZE=16`; E15 measures the eager cost; E12 census gives the launch count |
| vLLM MoE kernel selection and the Marlin fallback (`VLLM_MOE_FORCE_MARLIN`) | ★★ | vLLM source `vllm/model_executor/layers/quantization/` (search `marlin`, `sm120`); vLLM issue #30135 | RESEARCH.md "Engines"; K5 |
| SGLang DeepSeek V4.1 branch: Engram host table env var, tokenizer/reasoning parsers | ★★ | SGLang repo branch `dev-dsv41`; RESEARCH.md vLLM recipe notes for flag names | `scripts/serve.sh` 4-GPU profile |

## 7. Memory hierarchy and offloading (the "innovation" half)

Why here: the 2-GPU verdict is a bandwidth budget. Every offload design is bounded by the slowest tier it touches per token.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| DRAM bandwidth: channels × width × MT/s; why 8 × DDR5-6400 is ~410 GB/s theoretical and ~250–300 achievable; STREAM | ★★ | John McCalpin's STREAM page (cs.virginia.edu/stream); Kingston KF564R32-32 datasheet for the DIMMs installed | E1/E1b; `docs/HARDWARE.md` memory table |
| NUMA, hugepages (2 MiB / 1 GiB), THP, `numactl`, why page faults kill gather workloads | ★★ | Linux kernel docs `Documentation/admin-guide/mm/hugetlbpage.rst`; `man numactl` | T1.6 hugepage test; K7 pinned host memory |
| CPU SIMD for dequant-GEMV: AVX-512, VNNI, BF16 extensions on Zen 5; what AMX is and why not having it matters less than it sounds | ★★ | AMD Zen 5 ISA notes; Intel Intrinsics Guide (filter AVX-512); KTransformers' `MXFP4` CPU kernel source | E6 stage 0 (`scripts/bench-cpu-expert.py`); HARDWARE.md CPU note |
| NVMe random-read throughput vs sequential, queue depth, `io_uring`, O_DIRECT | ★★ | Samsung 9100 PRO datasheet; `man io_uring`; "fio" docs for the `randread` model | E3 (`bench-engram-gather.py`), `evidence/e3-*.json` |
| GPUDirect Storage (GPU reads NVMe without host bounce) and zero-copy pinned host access from kernels | ★★★ | docs.nvidia.com/gpudirect-storage; CUDA Programming Guide "Mapped Memory" | K8 |
| Caching theory for Engram rows: hit rate vs cache size, Zipf-like access | ★★ | Any textbook LRU/LFU chapter; measure instead of assume | E11 cache-hit simulation |
| Prior art in CPU-GPU hybrid MoE inference | ★★ | KTransformers paper (search arXiv for "KTransformers: Unleashing the Full Potential of CPU/GPU Hybrid Inference"); "MoE-Lightning" (arXiv 2411.11217); "Fiddler" (arXiv 2402.07033) | Option B/B' design in FEASIBILITY.md |

Exercise: compute the per-token DRAM traffic by hand. 6 routed experts × 40 MoE layers = 240 expert activations per token; 42% of them are CPU-resident ≈ 101; each activation reads one expert's three matrices = 17.9 MiB; 101 × 17.9 MiB ≈ 1.77 GiB per token (the calculator prints 1.75 with exact counts). Divide by 146 GB/s and by 300 GB/s: those are the 56 and 89 tok/s ceilings in FEASIBILITY.md.

## 8. Host hardware in this box

Why here: the specs that appear in `docs/HARDWARE.md` and gate feasibility.

| Component | Know | Read |
|---|---|---|
| AMD Threadripper PRO 9955WX (16C Zen 5) | 8 DDR5 channels, 128 PCIe 5.0 lanes, AVX-512, no AMX, 64 MiB L3 | AMD product page; `lscpu`; `evidence/hardware-inventory.json` |
| Kingston FURY Renegade Pro DDR5-6400 RDIMM (KF564R32-32 × 8) | RDIMM vs UDIMM, ECC, EXPO profile (SPD 4800 → configured 6400), rank | Kingston datasheet; `sudo dmidecode -t memory` (you already ran it) |
| ASUS Pro WS WRX90E-SAGE SE | Slot lane map (which x16 slots share lanes), 8 DIMM slots one per channel | ASUS manual, block diagram page |
| RTX PRO 6000 Blackwell Max-Q ×2 | 96 GB GDDR7, ~1.79 TB/s, 300 W Max-Q, SM120, no NVLink (P2P over PCIe) | NVIDIA product brief; `nvidia-smi -q` |
| Samsung 9100 PRO, Crucial T705 | PCIe 5.0 x4 NVMe, ~14 GB/s seq, random 4K IOPS spec vs our measured 403K/347K rows/s | Vendor datasheets; `evidence/e3-*.json` |
| 1,500 W PSU | 2 × 300 W GPUs + 350 W CPU + platform ≈ 1.1 kW peak; 4 GPUs do not fit | HARDWARE.md power table; T4.0 |

## 9. Benchmarking method and statistics

Why here: "honest statistics" is in the goal. Every claim in `docs/BENCHMARKS.md` must be reproducible from a JSON in `results/` produced by this harness.

| Topic | ★ | Read | Look at here |
|---|---|---|---|
| Decode tok/s from `usage` counts, excluding TTFT; prefill tok/s; warm-up; repeated trials; medians not means | ★ | `scripts/bench-decode.py` docstring and `test-openai-client.py` | — |
| Fixed fixtures across recipes so numbers compare (same prompt lengths, same sampling, same concurrency) | ★ | The GLM recipe's BENCHMARKS.md method section (`~/dev/src/glm-5.3-flash-2x-rtx-pro-6000-blackwell/`) | T2.1 |
| Needle-in-a-haystack long-context checks | ★ | "Lost in the Middle" (arXiv 2307.03172) for why position matters | `scripts/verify.py` |
| Power/thermal conditioning: `nvidia-smi -pl`, clocks, sustained vs burst | ★★ | `nvidia-smi --help-query-gpu`; NVIDIA power-capping docs | E8b tok/s-per-watt; E8 1M prefill (88 °C, −6 % clocks after 8 min at the cap) |
| Speculative decoding as a measurement hazard: acceptance depends on content, so report it beside tok/s and always take a speculation-off number | ★ | DSpark/MTP section of the V4 tech report; `bench-decode.py` `accept` column | E8 DSpark-off 89 tok/s vs 171–239 on |
| Prefix caching as a measurement hazard: a repeated prompt "prefills" in 0.1 s | ★ | vLLM automatic prefix caching docs | E8b first run (223K tok/s bogus row) |
| Contended vs quiet measurements (why E1 must be re-run) | ★ | ROADMAP.md standing assumptions | E1b |

## 10. What each experiment teaches you

| E | Running it teaches | Read before |
|---|---|---|
| E1/E1b | What "memory bandwidth" means in practice and how far Python/torch get from STREAM | §7 DRAM row |
| | *Surprise:* the CPU reads only 125 GB/s of a 410 GB/s memory system. The limit is the Zen 5 core-complex-to-IO-die fabric (16 cores = 2 CCDs), so the fix is a CPU with more CCDs, not faster DIMMs. Read up: AMD "Infinity Fabric" / GMI bandwidth per CCD, and why STREAM on low-core-count Threadrippers disappoints. [`e1b`](../evidence/e1b-dram-read-bandwidth.json) | |
| E2 | GDDR7 bandwidth vs spec; how to measure with torch and CUDA events | §3 roofline row |
| | *Surprise:* none — 82 % of the derived 1,792 GB/s, as expected for a copy. The lesson is in how a D2D copy counts bytes twice (read+write) so "1,370 GiB/s" and "1,470 GB/s" describe the same run. [`e2-e4`](../evidence/e2-e4-gpu-pcie-bandwidth.json) | |
| E3 | Random-access throughput of DRAM vs NVMe; why rows/s, not GB/s, is the Engram metric | §7 NVMe + caching rows |
| E4 | PCIe H2D reality and pinned memory; why ~46 tok/s is the streaming ceiling | §3 PCIe row |
| | *Surprise:* both x16 links run at full rate together (106 GiB/s), and that aggregate is within 10 % of what the CPU itself can read from DRAM. [`e2-e4`](../evidence/e2-e4-gpu-pcie-bandwidth.json) | |
| E4b | Whether two consumers of one memory share a bottleneck: measure each alone, then together, and compute additivity | §7 DRAM row; NUMA/fabric topology |
| | *Surprise:* CPU reads and GPU DMA reads of the same DIMMs add up (234 GB/s, 0.98 additivity). That is what turned "the CPU is too slow" into "use both paths at once" (Option B2). Read up: how a PCIe root complex on the IO die reaches memory without crossing the CCD links. [`e4b`](../evidence/e4b-dram-contention-cpu-vs-dma.json) | |
| E5 | How stock engines fail on an oversize model (OOM traces, offload flags that exist vs don't) | §6 |
| E6 s0 | Whether a 16-core CPU can do 101 expert GEMVs per token; roofline on the CPU side | §7 SIMD row |
| | *Surprise:* two of the three script variants measured torch, not the machine — a bf16 `sum` runs at 10 GB/s because it converts every element, and an int8 dequant written with tensor temporaries is 8× slower than the bf16 GEMV it was meant to proxy. Lesson: a micro-benchmark must be checked against a known roofline before its number is believed. The valid variant (bf16 GEMV, 94 GB/s) says bandwidth, not compute, binds. [`e6-stage0`](../evidence/e6-stage0-cpu-expert-gemv.json) | |
| E6 s1 | KTransformers' hybrid in practice; expert placement flags; SM120 toolchain (flashinfer, tilelang) | §6 KTransformers row |
| E6 s2 | The whole model architecture, by porting it (§1 end-to-end) | §1, all rows |
| E7 | MoE routing statistics; Zipf-or-not; why placement can beat random | §1 MoE row |
| E8/E8b | Clean benchmarking method; power caps vs throughput | §9 |
| | *Surprise (E8):* three separate lessons. (1) The "single-stream tok/s" of a speculative-decoding engine is not a hardware number: the same server gave 171 and 239 tok/s depending only on how many tokens the fixture made it write, because DSpark acceptance went from 40 % to 63 % with the content. Always print acceptance beside tok/s and fix the output length. (2) A harness that sums per-sequence rates lies as soon as the server queues: C8 "1,239 tok/s" was really 277 — check TTFT jumps and compute total tokens ÷ wall time. (3) Prefill throughput is a curve over context length (7.8K tok/s at 32K → 1.9K at 1M), so a single "prefill tok/s" without its prompt length is meaningless. Bonus: decode draws 242–249 W with a 300 W cap, prefill sits on the cap with clocks throttled — memory-bound work does not need the power budget compute-bound work does. Read up: speculative decoding acceptance rate and its effect on step yield; Little's law for queued throughput; `nvidia-smi --query-gpu=power.draw,clocks.current.sm`. [`baseline-v4-flash/`](../evidence/baseline-v4-flash/) | |
| | *Surprise (E8b):* a power cap is a *ceiling*, and decode never touches it: 216–229 tok/s from 325 W down to 250 W because a memory-bound decode step draws only 228–251 W per card. Prefill is compute-bound and sits exactly on whatever cap you set, so throughput tracks SM clock (8,770 → 7,482 tok/s for 2.09 → 1.72 GHz) and tok/s per watt *rises* as the cap falls — the physics of \(P \propto f V^2\): lower clock lets the driver lower voltage too. Practical: on a fixed power budget, more cards at a lower cap beat fewer at 325 W. Two harness lessons cost more time than the physics: (a) an inference server's *prefix cache* will happily "prefill" 32K tokens in 0.14 s if you send the same prompt twice — change the seed per run and treat a TTFT that is too good as a bug, not a result; (b) a background sampler that outlives a failed script keeps writing at its own file offset; when the next run truncates the same file you get a NUL hole and stale rows. Always kill helpers from the EXIT trap. From the DSpark-off run: the one truly repeatable decode number is the no-speculation one (89 tok/s, ±0.2 %); measure it first and treat every speculative number as "89 × (1 + acceptance gain)". Also: the first cap you test runs on cold cards, which draw ~20 W less at the same clock (leakage rises with temperature), so either warm up first or randomise order before reading tok/s/W differences under 10 %. |
| E10 | Speculative decoding acceptance vs bottleneck location | §1 MTP row |
| E11 | Quantizing a lookup table; cache simulation | §2 formats, §7 caching |
| E12 | Reading a profiler trace; identifying kernels by name; Marlin vs native | §5 |
| | *Surprise:* the hypothesis was wrong in a useful way. The deployed vLLM NVFP4 path already runs a CUTLASS SM120 block-scaled **grouped** GEMM — the mangled name `GemmUniversal<GroupProblemShape…MainloopSm120ArrayTmaWarpSpecializedBlockScaled…float_e2m1_t, float_ue4m3_t>` is E13's own kernel family with NVFP4 types (demangle with `c++filt` or read the template arguments: `PtrArray` = grouped, `Sm120` = arch, `128,64` = tile, `e2m1/ue4m3` = formats). Experts were 25 % of GPU time, not the 50 %+ the whole plan assumed; the dense FP8 projections (`_w8a8_triton_block_scaled_mm`, 236 launches per step) were bigger. Lesson: profile before optimising — a 2× faster expert kernel would have bought 12 %. Second lesson: nsys cannot attach to a running container, but vLLM's `/start_profile` torch-profiler route can; and a categorised kernel census (`scripts/kernel-census.py`) is a 100-line script once someone has parsed the trace. Read up: Amdahl's law; `torch.profiler` Chrome-trace format; CUTLASS kernel naming (`KernelPtrArrayTmaWarpSpecializedCooperative`). [`e12`](../evidence/e12-kernel-census-v4-flash-tp2.json) | |
| E13 | CUTLASS block-scaled grouped GEMM end to end: types, builder, compile targets, correctness vs reference, TFLOP/s | §2, §3 PTX row, §4 all rows |
| | *Surprise:* it just worked — a 40-line type swap on example 79d compiled in 34 s and passed the host reference; the "SM120 needs NVFP4 re-quant" folklore was a CUTLASS 3.8 limitation. Second surprise: at decode M=1 the kernel is already at 93 % of the memory roofline, so a "faster kernel" cannot help there; only fewer bytes (fewer experts per token, or a smarter M tile for the short-K down projection) can. Third: smaller N tiles made decode slower, because a 128-row M tile at M=1 is 99 % padding and each extra CTA pays fixed cost. Fourth (from the W4A8 run): a kernel can look bandwidth-bound and still be two-thirds compute — at M=1 the 128-row M tile issues M=128 worth of MMA, so the FP8-rate `mxf8f6f4` variant doubled the time. Roofline lesson: count the *issued* FLOPs, not the useful ones. Fifth (census-shaped runs): always check the working set against L2 — the RTX PRO 6000's 128 MB L2 holds a 6-group or 15-group decode case entirely, so back-to-back iterations can read from L2 and post "bandwidth" above HBM (1,611 GB/s at 15 groups × M=4). The 36-group runs (226–451 MB) are the honest HBM numbers, and they came out *higher* than the small cases (1,421–1,451 GB/s), so the small cases were latency-bound, not L2-flattered. Sixth: E13 at the deployed kernel's shapes ties it to within 1 µs — two roofline kernels cannot differ, so kernel work should chase *bytes* (format, expert count, M tile for short K) rather than speed. Walk `kernels/e13/mxf4_grouped_gemm.cu` next to 79d with `diff` to see exactly what changed. [`e13`](../evidence/e13-stage1-mxf4-grouped-gemm.json), [`tp2-g36`](../evidence/e13-stage1-decode-tp2-g36.json) | |
| E15 | *Question it answers:* how much of a decode step is CPU overhead rather than GPU work, and therefore whether a V4.1 engine that cannot capture graphs is acceptable. *Answer:* 52 of 63 ms per step in eager mode — 4.4–5.6×; not acceptable. *Surprise:* I guessed 15–35 % from kernel-launch gaps alone; the real cost is Python dispatch over thousands of ops plus the fusion torch.compile provides — launch latency was the smallest part. Second surprise: 16 slots would not start at 1M because weights leave 5.7 GiB of KV per card — the scheduler setting in the sibling was a memory outcome. *Read first:* §6 CUDA graphs row; the vLLM `SpecDecoding metrics` log line is the cheapest per-position acceptance instrument you have. |
| E14 | How to compare two inference stacks honestly: same fixture, same output length, speculation on *and* off, wall-clock not summed rates, same power cap, footprint next to speed; why a 16-slot scheduler and a 2-slot one cannot be compared at C8 without saying so | §9 |
| | *Surprise (E14):* the power cap lesson from E8b was model-specific. GLM's EXL3 trellis weights have to be *decoded* on the SMs before every GEMM, so its decode runs ~10 % hotter at C1 (256–259 W) and pins both cards at 300 W as soon as there is any concurrency; V4-Flash's NVFP4 is consumed directly by the tensor cores and never reaches the cap. *And a lesson about measuring:* my first write-up said "297 W at C1" because the sampler median covered the whole C1→C16 run — always check what window a median spans before attributing it to one condition. The owner's 250 W sweep, which isolates C1, corrected it. Same GPUs, opposite bottleneck — read §2 (formats) and §3 (SM120 FP4 instructions) with that pair in mind: "4-bit" says how many bytes you move, not how much work the SMs do to use them. Second: the two engines' prefill curves cross — dense attention wins short (7.8K vs 4.2K tok/s at 32K), sparse MLA wins long (3.8K vs 1.9K at 1M) — which is exactly the trade CSA2 in V4.1 is supposed to remove. Third: different tokenizers turn the same text into 16 % different token counts; compare *time per fixture*, or actual tokens, never target tokens. [`e14-glm53`](../evidence/e14-glm53/) |

## 11. What the box has taught us so far (read this before any of the sections above)

The measured facts every later decision rests on. Each links to the experiment that produced it; if a number here disagrees with a section above, the section is stale.

| Fact | Number | Why it matters | From |
|---|---|---|---|
| The CPU cannot read its own DRAM at spec | 125 GB/s of 410 (30 %); 94 GB/s while also computing | Two CCDs on the 9955WX cap the fabric; the DRAM-resident experts cost 15–20 ms/token on the CPU path alone. Faster DIMMs would not help; more CCDs would | E1b, E6 s0 |
| GPU DMA reads of DRAM add on top of CPU reads | 120 + 114 = 234 GB/s, 98 % additive | The hybrid should feed DRAM-resident experts through *both* paths (Option B2, 78 tok/s ceiling) | E4b |
| PCIe 5.0 x16 per card, both together | 53 GiB/s, 106 GiB/s | Ceiling for weight streaming (~46 tok/s) and layerwise prefill (~1 s per 4K chunk) | E4 |
| GDDR7 on these cards | ~1,470 GB/s copy; E13 kernel reaches 1,400–1,450 | Decode on GPU-resident experts is at the roofline already | E2, E13 |
| Native MXFP4 works on SM120 | CUTLASS 79d type swap compiles and verifies | The "SM120 needs re-quant to NVFP4" folklore is dead; format choice is about bytes and quality, not kernel availability | E13 |
| vLLM already uses a roofline SM120 grouped GEMM for NVFP4 experts | 24.5 % of the decode step; dense FP8 27 %, attention 11 %, draft 10 % | A faster expert kernel cannot move the 5 ms GPU step; fewer bytes per token and the dense/attention side can | E12 |
| V4-Flash NVFP4 TP2 decode, no speculation | **89 tok/s** (±0.2 %), 11.2 ms/step | The repeatable number; everything above it is speculation acceptance | E8 |
| Same, with DSpark K5 | 171–239 tok/s at 40–63 % acceptance | Acceptance follows the generated content, so report it or the number is meaningless | E8 |
| Power cap vs throughput | Decode never reaches 250 W; prefill pins to any cap; 250 W = −11 % prefill, 0 % decode | Four cards at 250 W fit the 1,500 W PSU for experiments; efficiency rises as the cap falls | E8b |
| Sustained load | 8 min at 300 W → 88 °C, clocks −6 % vs bursts | Short benches flatter long-context throughput; thermal headroom is the constraint for a 3rd/4th card | E8 1M prefill |
| Decode power depends on the weight format, not just the GPU | GLM (EXL3) C1: 256–259 W, pinned at 300 W for C2+; 250 W cap → 0 % C1 decode, −11.6 % prefill. V4-Flash (NVFP4): 233–250 W, never capped | Formats the tensor cores eat directly (MXFP4/NVFP4) leave power headroom under a 250 W cap; trellis/K-quants spend it on dequant. T4.0 holds for both | E14 + sweep |
| Eager mode is not a fallback, it is a 4–6× loss | V4-Flash eager: 15.9 tok/s no-DSpark (63 ms/step vs 11.2), 49.5 with DSpark (vs 217); GPUs at ~150 W, starved | A V4.1 engine must capture/compile its GPU step; the E6 CPU-expert call must live inside the graph. `--enforce-eager` advice from the source doc is wrong for this box | E15 |
| Speculation amortises a CPU-bound step | Eager+DSpark 49.5 vs eager 15.9 tok/s (3.1×); acceptance unchanged (51–60 %) | The hybrid's 15–20 ms DRAM-bound step gets ~4 tokens per verify — DSpark is load-bearing for E6, not optional (E10) | E15 |
| Slot count is a KV-budget outcome | V4-Flash: 83.2 GiB weights/card → 5.7 GiB KV → one 1M request; at 262K, 16 slots → C16 1,258 tok/s (2.2× GLM) | Context ceiling and concurrency trade against each other inside ~10 GiB/card; V4.1 on 2 GPUs will have even less KV room — pick 262K/multi-slot or 1M/single | T2.4, E15 |
| K=5 draft is not too long | Per-position acceptance 0.85/0.71/0.58/0.48/0.40, mean 4.0 tokens/step | The 5th token still pays; verification is free on a bandwidth-bound step. Longer would likely still help | E15 |
| The 2-GPU incumbents to beat | GLM: 132 tok/s C1, 569 C16, 3.8K tok/s at 1M; V4-Flash: 207–239 C1, 1.9K at 1M | V4.1 on 2 GPUs will not win single-stream (hybrid ceiling 50–78); its case is quality/context, and speed needs the 3-/4-GPU layouts | E14 |
| Prefill is a length curve | 7.8K (32K) → 1.9K tok/s (1M) | Any V4.1 CSA2 engine is judged against this curve, not a scalar | E8 |
| Engram lookups from NVMe keep up with decode | both drives sustain the random-row rate | Engram on NVMe + 64 GiB DRAM cache is the default tiering; DRAM stays for experts | E3 |

## Keeping this current

When an experiment resolves a question, add one line under its row in §10 saying what surprised you, and link the evidence JSON. The point of the guide is the delta between what the sources said and what the box did.

# Experiments

Each experiment states the question, the hypothesis with a number attached, the exact procedure, what counts as pass, and where the result lands. Negative results are results. All output JSON goes to `results/` (git-ignored) and the summarised numbers move into `docs/BENCHMARKS.md` by hand with the JSON committed under `evidence/` once reviewed for identifiers.

Legend: **GPU-free** experiments do not touch the cards but still need a quiet host (no GLM benchmark in flight) for CPU/DRAM/NVMe numbers to mean anything. **GPU** experiments need `docker stop glm53-flash-2x-rtxpro6000` first and `docker start` afterwards. Torch-based GPU-free benches run as `docker run --rm -v "$PWD:/w" -w /w --entrypoint python3 dsv4-nvfp4:recipe scripts/<bench>.py` (no `--gpus`).

## E1 — Host DRAM bandwidth (GPU-free)

- Question: how much of the 409.6 GB/s theoretical does the 16-core 9955WX reach?
- Hypothesis: 250–320 GB/s copy bandwidth with 16–32 workers; a single worker ≈ 20–30 GB/s.
- Procedure: `scripts/bench-dram.py --workers 1,2,4,8,16,32 --size-gib 8 --output results/e1.json` (stdlib `mmap` memcpy via `memoryview` slicing in N *processes*; threads hold the GIL and measure one core). STREAM inside a container is the optional follow-up.
- Pass: a measured number replaces the "estimate" row in HARDWARE.md. There is no fail.
- Feeds: the hybrid decode ceiling (`feasibility-calc.py --dram-gbps`); Engram-in-DRAM prefill estimate.
- **Result (run):** 60.0 GB/s single process, 139.9 GB/s at 16, 146.2 GB/s at 32 (read+write). Below hypothesis; a Python lower bound. **This is now the number the 2-GPU verdict's ceiling rests on** (56 tok/s at 146; 89 at 300). Follow-up E1b: a C/STREAM or `torch` CPU read benchmark in the `dsv4-nvfp4:recipe` container, GPU-free, to find the real read bandwidth. [`evidence/e1-dram-bandwidth.json`](../evidence/e1-dram-bandwidth.json).

## E2 — GPU memory bandwidth (GPU)

- Question: achieved GDDR7 bandwidth per RTX PRO 6000 Max-Q at 300 W.
- Hypothesis: 1,450–1,650 GB/s device-to-device copy; well below 1,792 GB/s theoretical.
- Procedure: `scripts/bench-pcie-h2d.py --d2d` inside any torch-bearing image on this host (e.g. the `dsv4-sm120` images), 1–8 GiB buffers, 20 iterations, median.
- Pass: number recorded. Used for the decode roofline on the 4-GPU profile (weights per GPU ÷ bandwidth ≈ per-token floor).

## E3 — Engram-style random row gather: DRAM vs NVMe (GPU-free)

- Question: can either drive or DRAM serve Engram lookups at decode and prefill rates?
- Hypothesis: DRAM ≥ 5 M rows/s single-process; Samsung 9100 PRO ≥ 600K random 4 KiB reads/s with O_DIRECT at QD 32 from Python threads (well below the 2.2 M IOPS spec because of Python overhead; document that as a lower bound); T705 lower.
- Procedure: `scripts/bench-engram-gather.py --backend dram --rows 20000000 --output results/e3-dram.json`; `--backend nvme --path results/engram-nvme0.bin` (nvme0, root fs) and `--path /mnt/data/dsv41-engram-bench.bin` (nvme1). Synthetic 20 M × 264 B table (4.9 GiB); no model download. Report rows/s, p50/p99 latency per 48-row token batch.
- Pass: decode batch (48 rows) p99 < 1 ms on both backends; prefill 3.1 M rows in < 10 s on DRAM. Failure on NVMe means Option A must use Engram-in-DRAM mode, which this host can.
- Feeds: HARDWARE.md storage section; choice of `OFFLOAD_MODE` in the 4-GPU profile.
- **Result (run):** DRAM 4.15 M rows/s, batch p99 0.015 ms, 64K-prefill equivalent 0.75 s — pass. Samsung 9100 PRO 403K rows/s, batch p50 1.90 ms; Crucial T705 347K rows/s, p50 2.20 ms — the `< 1 ms` criterion **fails as measured**, but the harness issues a token's 48 reads serially, so the batch latency is 48 × the drive's ≈ 40–46 µs unloaded read latency, not a queueing limit. Corrected reading: both drives sustain decode-rate lookups (≥ 7K token-equivalents/s); NVMe costs ≈ 7.7–8.9 s of lookups per 64K-token prefill versus 0.75 s in DRAM. Decision: the 4-GPU profile keeps `OFFLOAD_MODE=engram-host` (DRAM); NVMe is the fallback if DRAM is needed for something else. Evidence: [`e3-engram-gather-dram.json`](../evidence/e3-engram-gather-dram.json), [`e3-engram-gather-nvme0.json`](../evidence/e3-engram-gather-nvme0.json), [`e3-engram-gather-nvme1.json`](../evidence/e3-engram-gather-nvme1.json).

## E4 — PCIe host-to-device bandwidth per GPU (GPU)

- Question: sustained pinned H2D and D2H bandwidth per card, and both cards concurrently.
- Hypothesis: 48–55 GiB/s per card pinned; both concurrently ≥ 90 GiB/s aggregate (separate x16 links); pageable ≈ half.
- Procedure: `scripts/bench-pcie-h2d.py --sizes 256M,1G,4G --pinned --both` inside a torch image.
- Pass: numbers recorded. Sets the PCIe-streaming fallback ceiling (decode tok/s ≤ 1000 / (5 ms + 1.75 GiB ÷ aggregate GiB/s)) and the layerwise-prefill cost of the hybrid (112 GiB of CPU experts per 4,096-token chunk).

## E5 — V4.1-Flash on 2 GPUs with generic CPU offload (GPU, needs 475 GiB download)

- Question: what exactly happens when the two available engines are asked to do it?
- Hypothesis: SGLang `dev-dsv41` with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 --tp 2` OOMs during weight load at ~140 GiB/GPU demand. vLLM `--cpu-offload-gb 110` either fails to construct the deepseek_v41 loader on SM120 or produces < 5 tok/s because it streams all offloaded parameters every step.
- Procedure: `scripts/download-model.sh` (pinned), then `ALLOW_UNQUALIFIED=1 scripts/serve.sh --profile 2gpu-attempt` with each engine; capture the log tail (scrubbed) into `results/e5-*.log`.
- Pass criterion is documentation: the failure mode is recorded verbatim in TROUBLESHOOTING.md. If it unexpectedly serves, run `verify.py` and `bench-decode.py` and treat the number as the new 2-GPU result.

## E6 — Three-tier hybrid: CPU-computed experts (stage 0 GPU-free; stages 1–2 GPU)

- Question: can 42% of V4.1-Flash's routed experts run on this host's CPU from DRAM at a rate that makes the 2-GPU layout useful, and can an engine be made to do it for `deepseek_v41`?
- Hypothesis: decode 40–60 tok/s at batch 1 with 58% of experts on GPU (calculator ceiling 56 tok/s at 146 GB/s); CPU compute ≤ DRAM time when overlapped; prefill 2–3K tok/s with layerwise GPU prefill.
- **Stage 0 — CPU expert GEMV (GPU-free, runnable now).** `scripts/bench-cpu-expert.py` inside `dsv4-nvfp4:recipe` without `--gpus`: a synthetic pool of experts (3 × 5120×2304 matrices each) larger than L3, 101 random experts per token, gate/up/down GEMVs at batch 1, 16 and 32 threads, weights in bf16 (memory roofline) and int8-dequant (compute proxy for MXFP4). Report ms/token and effective GB/s. Pass: a measured CPU ms/token to put next to the 12.9 ms DRAM figure; if bf16 GEMV already exceeds 25 ms for 101 experts, CPU compute — not DRAM — is the bound and the plan changes to PCIe streaming or a 3rd GPU.
- **Stage 1 — sglang-kt with V4-Flash on 2 GPUs (GPU).** Docker `approachingai/ktransformers:DSV4-specific`, `--kt-method MXFP4 --kt-num-gpu-experts N --kt-expert-placement-strategy frequency --kt-enable-dynamic-expert-update --kt-gpu-prefill-token-threshold 4096`, model V4-Flash (validated by KT). Sweep N from all-on-GPU down to 58% and record decode vs the resident 200 tok/s baseline. Purpose: prove the toolchain on SM120 (flashinfer ≥ 0.6.9, tilelang) and calibrate the calculator's `--gpu-ms-per-token` and `--dram-gbps` against a real hybrid before touching V4.1.
- **Stage 2 — port `deepseek_v41` into sglang-kt (build).** Design doc first (`plan/E6-DESIGN.md`): CSA2 attention and indexer, Engram host table (reuse SGLang `dev-dsv41`'s `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE`), hyper-connections, DSpark, and the MXFP4 expert format mapping (upstream UE8M0 g32 → KT's MXFP4 CPU kernel). Then serve V4.1-Flash with `--kt-num-gpu-experts` ≈ 8,958 total, Engram on NVMe + cache, and run `verify.py` + E8's benchmark set.
- **Parallel route — llama.cpp `deepseek41`.** Build vcruz305/llama.cpp `runtime/deepseek41`, load `vcruz305/DeepSeek-V4.1-Flash-GGUF` Q4_K_M (or Q3_K_M, 323 GiB) with `-ot` regex offload of ~42% of `ffn_*_exps` to CPU and `--split-mode layer`, once CSA2 is merged. Lower GPU efficiency than SGLang but zero porting.
- Pass: any route producing a V4.1-Flash decode/prefill/needle table on 2 GPUs. Record the engine commit, expert split, Engram mode and power in BENCHMARKS.md. Negative results (route blocked, number, reason) are recorded too.

## E7 — Routing skew study for expert pruning (GPU, uses the V4-Flash baseline deployment)

- Question: is routing uniform enough that pruning 42% of experts would destroy quality, or skewed enough that frequency-based GPU placement (E6) gains and a pruned checkpoint is plausible?
- Hypothesis: `noaux_tc` load balancing makes per-expert activation frequency within ±30% of uniform on general text; pruning is not viable without retraining.
- Procedure: enable expert-load logging on the V4-Flash baseline (same MoE family, 256 experts) over a 5 M-token mixed corpus; compute per-layer activation histograms and the cumulative share of the top-61% experts.
- Pass: report the share of activations captured by the top 58% of experts per layer; that number is `--activation-skew` for the calculator and the expected gain of frequency placement. If the top 58% capture ≥ 90% of activations on ≥ 80% of layers, Option D graduates to a build task; otherwise it is closed with the histogram as evidence.

## E8 — Baseline re-measurement (GPU, existing V4-Flash deployment)

- Question: what does the qualified V4-Flash-0731 NVFP4 TP2 deployment do under *this* harness?
- Hypothesis: within ±5% of the numbers in the sibling repo (200 tok/s single-stream with DSpark K5, ~226 tok/s at C2, ~2K tok/s prefill).
- Procedure: `scripts/bench-decode.py --concurrency 1,2,4,8 --runs 5 --warmup 2`, `scripts/bench-prefill.py --tokens 8192,32768,131072,524288,1000000`, `scripts/verify.py --needles 6 --context 1000000`.
- Pass: tables in BENCHMARKS.md; JSON in `evidence/baseline-v4-flash/`.
- **E8b — power sweep (same window).** The host's PSU is 1,500 W; four Max-Q cards at 300 W do not fit it (HARDWARE.md power table). Repeat the C1 decode and 32K prefill rows at `nvidia-smi -pl 300`, `275`, `250`, `225` on both cards, logging `power.draw` and, if a wall meter is available, whole-system draw. Hypothesis: decode (bandwidth-bound) loses < 10 % from 300 → 225 W; prefill (compute-bound) loses 15–25 %. Output: the tok/s-per-watt curve that decides between "cap four cards at 225 W on the 1,500 W PSU" and "buy a 2,000 W PSU" for Option A. Restore `-pl 300` afterwards.

## E9 — 4-GPU V4.1-Flash qualification (blocked on hardware)

- Question: does the SGLang `dev-dsv41` + Engram-in-DRAM profile serve V4.1-Flash on 4 × RTX PRO 6000 Max-Q, and how does it compare with 0xSero's NVMe-Engram numbers and with E8?
- Hypothesis: 180–230 tok/s at C1 with DSpark 5, 600–750 tok/s at C8, 5–7K tok/s prefill, DSpark acceptance 75–88%; Engram-in-DRAM improves prefill TTFT on long prompts versus NVMe mode by a measurable margin.
- Procedure: `preflight.sh` (must pass on 4 GPUs), `serve.sh --profile 4gpu-sglang`, `wait-ready.py`, `verify.py`, E8's benchmark set, then the same with `OFFLOAD_MODE=nvme` for the A/B.
- Pass: verify 6/6 needles at 262,144 and at the configured maximum; `17*19=323` sanity; tables recorded with power condition (300 W Max-Q).

## E10 — Speculative decoding under a CPU-bound MoE (GPU, after E6 stage 1)

- Question: does DSpark (block 5) help or hurt when 42% of expert activations are DRAM-bound?
- Hypothesis: a 5-token block activates ~3–4× the distinct CPU experts of a single token, so DRAM traffic per step rises faster than accepted tokens per step; net decode falls unless acceptance ≥ 80%. Community reports on CPU-bandwidth-bound hybrids say spec decode hurts.
- Procedure: on the E6 stage 1 (V4-Flash) and stage 2 (V4.1-Flash) deployments, C1 decode with DSpark off, block 3, block 5; log accepted tokens/step and host DRAM bandwidth (`perf stat` uncore counters or `pcm-memory`).
- Pass: tok/s and GiB/step per setting; the recipe's default follows the measurement.

## E11 — Engram tiering and quantization (GPU-free data prep; GPU for quality)

- Question: which Engram placement is best for the 2-GPU hybrid, where DRAM is the scarce tier?
- Hypothesis: NVMe + 64 GiB DRAM row cache costs < 5% decode versus DRAM-resident tables (48 reads/token at ~40 µs each, parallel) and adds ~8 s per 64K-token prefill; FP4 or q3_K tables in DRAM remove that cost at an unmeasured quality price.
- Procedure: (a) extend `bench-engram-gather.py` with an LRU/hot-set cache in front of NVMe and a Zipf request distribution to estimate hit rate vs cache size; (b) once E6 serves, A/B NVMe-cache vs LibertAI FP4 tables on `verify.py` needles and a small perplexity set.
- Pass: decode and prefill delta per mode; a quality verdict on quantized tables.

## E12 — Kernel census on SM120 (GPU, uses the V4-Flash baseline deployment first)

- Question: which kernels do SGLang `dev-dsv41`, vLLM and sglang-kt actually run on these cards for the MoE experts, attention and DSpark, and how many ms/token do they cost? The calculator's `--gpu-ms-per-token 5` is an assumption taken from the V4-Flash TP2 result; this replaces it with a breakdown.
- Hypothesis: the routed-expert GEMMs run on Marlin W4A16 (FP4 dequantised to FP16, no FP4 tensor-core use) rather than a native SM120 block-scaled path; experts are ≥ 50% of GPU time at batch 1; attention (CSA2/MLA via FlashInfer) ≤ 25%.
- Procedure: `nsys profile --trace cuda,nvtx` of 200 decode steps and one 32K prefill on the V4-Flash baseline, then on each V4.1 engine as it becomes runnable; server logs grepped for backend selection (`Marlin`, `cutlass`, `flashinfer`, `trtllm`, `triton`). Record top-10 kernels by time, ms/token per category, and whether any kernel is a documented SM120 fallback.
- Pass: a table in BENCHMARKS.md; the calculator's `--gpu-ms-per-token` becomes a measured value with a breakdown. Also records DSpark acceptance under the dispatched path (feeds E10: acceptance drops are reported on the Marlin dequant path because MTP heads were trained on native FP4 activations).

## E13 — Custom MXFP4 grouped GEMM for SM120 (build GPU-free; measure in a GPU window)

- Question: how much of the Marlin penalty can a native SM120 block-scaled kernel recover for V4.1-Flash's expert shapes (per expert: gate/up [4608 × 5120], down [5120 × 2304], MXFP4 E2M1 with UE8M0 block-32 scales; decode batch 1–8 tokens × 6 experts; prefill 4096-token chunks)?
- Hypothesis: CUTLASS's SM120 block-scaled grouped GEMM (SM120 mainloops since CUTLASS 3.9; mixed-input block-scaled grouped GEMM since 4.2; MoE grouped GEMM API in 4.3), tiled for 99 KiB shared memory (no TMEM/tcgen05 on SM120), reaches ≥ 2× Marlin W4A16 throughput at prefill shapes and ≥ 1.3× at decode shapes, where the GEMV is bandwidth-bound and dequant cost is what Marlin pays.
- Format question (answered from source, RESEARCH.md "MXFP4 is native"): SM120 accepts MXFP4 E2M1 + UE8M0 block-32 directly via `mma.sync … kind::mxf4nvf4 … scale_vec::2X … ue8m0` on `sm_120a`; CUTLASS's SM120 block-scaled builder takes `cutlass::mx_float4_t<float_e2m1_t>` (SF vector 32) and ships a dense `sm120_bs_gemm_mxf4_mxf4` unit test. No re-quantisation to NVFP4/UE4M3 is needed; the community claim came from CUTLASS 3.8. What is still untested anywhere is the mxf4 × mxf4 **grouped** (PtrArray) instantiation on SM120 — that is stage 1.
- Procedure, stage 1 (build GPU-free, run in a GPU window):
  1. Start from `examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu` (CUTLASS ≥ 4.2; `ArchTag Sm120`, `OpClassBlockScaledTensorOp`, TMA warp-specialized cooperative and pingpong, tile 128×128×128). Swap `ElementA/ElementB` from `cutlass::nv_float4_t<float_e2m1_t>` to `cutlass::mx_float4_t<float_e2m1_t>` (builder derives `float_ue8m0_t`, SF vector 32 → `BlockScaledInstr::MXF4_NVF4`); replace the FP4-output `LinCombBlockScaleFactor` epilogue with a plain bf16 `LinearCombination` so D is bf16 like the MoE down-projection expects. Build with `-gencode arch=compute_120a,code=sm_120a` (the `kind::mxf4*` PTX is arch-specific; `compute_120f` is not enough) in `dsv4-sm120:cuda-devel-https` (CUDA ≥ 12.8 required for the 2X-UE8M0 atom; check the image's `nvcc --version`).
  2. Second variant for W4A8: A as `cutlass::mx_float8_t<float_e4m3_t>`, B as `mx_float4_t` (the `kind::mxf8f6f4` mixed path example 79c uses). Same build.
  3. Feed real V4.1 expert shapes: per expert gate/up [4608 × 5120] and down [5120 × 2304] as K-major B; grouped problem lists for decode (6 experts × 1–8 tokens, M padded to the tile) and prefill (4096-token chunks spread over 384 experts ≈ 64 tokens/expert). Random E2M1 payload and UE8M0 scales in the checkpoint's block-32 layout.
  4. Correctness: compare against a host/torch reference that dequantises B (and A, in the same block-32 scheme) to fp32 and multiplies; W4A4 and W4A8 must match their own dequant reference bit-exactly in fp32 accumulate order tolerance (≤ 1 ulp bf16 after rounding). Also record W4A4 and W4A8 error against a **bf16-activation** reference so the activation-quantisation cost is visible, since Marlin W4A16 pays none.
  5. Speed: TFLOP/s and effective weight GB/s for each variant versus vLLM's Marlin MoE kernel on the same shapes (`VLLM_MOE_FORCE_MARLIN=1` path, run inside `dsv4-nvfp4:recipe`).
- Stage 2, if stage 1 wins: wire the kernel into the engine chosen by E6 (sglang-kt or SGLang `dev-dsv41`) behind the existing MoE backend selection, and re-run E12.
- Pass: microbench numbers recorded either way in `results/e13-stage1.json`. Stage 2 proceeds only if stage 1 shows ≥ 1.3× over Marlin at decode shapes with the variant's activation-quantisation error judged acceptable (W4A8 preferred if W4A4 error is visible in `verify.py`).
- Known prior art to read first: FlashInfer #2577 and PR #2725 (SM120 NVFP4 MoE patches), CUTLASS #3096 (TMA WS grouped GEMM failing on SM120; `compute_120f` fix → 39 tok/s native FP4), TRT-LLM #12706 (`TRTLLM_MOE_BACKEND=TRITON` 6.7× over CUTLASS on SM120/121), Colfax "NVFP4 blockscaled GEMM on SM12x", chsasank/blackwell custom CUTLASS NVFP4 kernels (GEMM at 38% of peak).

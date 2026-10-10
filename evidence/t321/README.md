# T3.21 window 1 — FP8 prefill GEMM candidates, and where a 16K prefill actually goes (2026-10-10) — closed, stop clause

Two parts, both on the qualified profile (overlay `ad973440`, launcher defaults 219 / 165, D=4, chunk 8192,
64K × 4, `ENGRAM_FILE_PREFETCH=ahead`, expandable-after-capture on):

| part | what | where |
|---|---|---|
| 1a | L2-cold microbench of five large-M FP8 GEMM candidates on the seven DSV4-Flash dense shapes at M = 4096 / 8192 / 16384 (GPU 1, container, no serve) | `gemm-bench/` (`results.json`, `bench-log.txt`), script `kernels/t321/bench_fp8_prefill_gemm.py` |
| 1b | GPU-only torch-profiler trace of one steady 16K prefill (two 8192 chunks) on a live serve | `prefill-profile/` (TP-0 / TP-1 traces, `step-profile-tp0.json` = `scripts/step-profile.py --steps 1`, the bench outputs, launch / ready / manifest) |

**Result.** The dense FP8 GEMM is **4.0 % of the 16K TTFT** (351 ms of 8,760 ms on rank 0; 360 ms, 4.1 % on
rank 1). The task's go bar (16K prefill ≥ +20 %) is unreachable by arithmetic even with an infinitely fast
kernel, so T3.21 closes on its stop clause with no kernel work. The only exact-math alternative found,
FlashInfer's cuTile `(1,32,32)` groupwise kernel, is **2.3–2.7× slower** than the E20-tuned Triton kernel
anyway. What the trace shows instead: a 16K prefill is the **E22 streamed-expert pipeline** — 90 ms per
layer per chunk (7.1 s of the 8.76 s), export-bound on the host — which is the next prefill lever (T3.39,
proposed).

## 1a — GEMM microbench

Shapes are the dense projections of one TP rank at the served TP=2 (N × K): 1792×5120, 2304×5120, 4096×1280,
5120×1152, 5120×4096, 16384×1280, 25600×6144. FP8 e4m3 weights with FP32 [32,32] block scales, per-token-group
(1×32) FP32 activation scales — the serving math. 256 MiB L2 flush before every launch, CUDA events, median of
20; `max_rel` is against an FP32 dequantised reference.

| candidate | math | sum of medians, M = 8192 (ms) | vs Triton |
|---|---|---|---|
| `triton_tuned` — SGLang `w8a8_block_fp8_matmul_triton` + the overlay's E20 configs (**the served kernel**) | exact | **14.32** (230–291 TFLOPS) | 1.00 |
| `cutile_32` — FlashInfer `gemm_fp8_nt_groupwise_cutile` at `(1,32,32)`, activation scales expanded `[M/32,K/32]`→`[M,K/32]` and timed in | exact (max_rel 0.0047, same as Triton) | 37.82 (97–111 TFLOPS) | **2.64× slower** |
| `cutlass_128` — FlashInfer CUTLASS SM120 groupwise `(1,128,128)`, scales subsampled | **not** the served math (K-128 blocks; max_rel ≈ 10² is the different scaling, not a bug) — a ceiling for a K-128 kernel | 7.75 (475–545) | 1.85× faster |
| `cutile_128` — cuTile at `(1,128,128)` | not the served math | 10.98 (334–403) | 1.30× faster |
| `scaled_mm` — `torch._scaled_mm` per-tensor FP8 (cuBLASLt) | not the served math; the per-tensor roof | **7.07** (505–591) | 2.03× faster |

M = 4096 and 16384 rank the same (cuTile 2.3–2.7× slower on every shape; Triton 1.9–2.0× the cuBLASLt roof).
`cutile_32` is called directly (`flashinfer.gemm.kernels.cutile.gemm_fp8_nt_groupwise_cutile`) because the public
dispatcher only admits `(1,128,128)` on SM120; the CUTLASS SM120 groupwise kernel rejects a K-granularity other than
128 (`gemm_groupwise_sm120.cu`), so there is no exact-math CUTLASS candidate without a new instantiation.

Toolchain note (the hour it cost): FlashInfer 0.6.18 / `cuda.tile` 1.6.0rc5 in the pinned image has no
`tileiras`; the 13.2.86 wheel fails (rc 5) because the JIT needs a matching `ptxas` / `libnvvm`. A throwaway
`pip install --no-deps --target` of `nvidia-cuda-tileiras nvidia-cuda-nvcc nvidia-nvvm nvidia-cuda-crt`
(all 13.4.92) mounted at `/tl` with `PATH` / `CUDA_HOME` / `LD_LIBRARY_PATH` pointed at `/tl/nvidia/cu13` made
it compile. Nothing of this is in the image; the bench is reproducible from the script's docstring.

## 1b — where a 16K prefill goes (rank 0; rank 1 agrees to within 3 %)

Serve → one cold 16K pass (seed 777, 651 tok/s, the serve's first prefill) → `/start_profile` (GPU only) →
16K prefill seed 1777: **1,822 tok/s, TTFT 8.785 s** (profiler on) → `/stop_profile`. `step-profile.py --steps 1`
over the 8,760 ms span, 11,837 GPU events, everything serialised on one stream:

| bucket | ms | share | what it is |
|---|---|---|---|
| HtoD memcpy | 2,341 | 26.7 % | 398 pinned→device copies, 131.6 GB, 57 GB/s: 80 × (973 + 487 + 122 + 61 MB) = **the 165 CPU experts of every layer streamed to this rank per chunk** (E22), not overlapped with compute |
| idle gaps ≥ 100 µs | 3,485 | 39.8 % | 2,159 ms in 78 gaps of 27–36 ms **inside the resident-expert fused-MoE call** (after its `quantize_with_block_size`, before its `blockExpertPrefixSum`) in 78 of the 80 layer-chunks; 577 ms in 63 gaps of ≈ 9 ms between the resident GEMM's `finalizeMoeRouting` and the layer's HtoD (where the host blocks in `sync_write_weight_scale_to_buffer`); 706 ms in 4 gaps at chunk starts (371 / 331 ms after a pageable DtoH) |
| NCCL all-reduce | 1,120 | 12.8 % | 166 calls, 6.7 ms each — mostly rank skew (rank 1 spends 394 ms) |
| GPU-resident + streamed expert MoE | 515 | 5.9 % | 320 grouped-GEMM launches (two per layer per chunk) |
| **dense FP8 GEMM** (`_w8a8_block_fp8_matmul`) | **351** | **4.0 %** | 462 calls, 756 µs avg — T3.21's target |
| elementwise / index | 339 | 3.9 % | 240 × 1.0 ms = the `_StreamedExpertSlots.load` w13 swap after each layer's HtoD (3 ms/layer-chunk), plus small copies |
| other (hc_mix, mhc_post, combine) | 271 | 3.1 % | |
| attention + DSA indexer | 262 | 3.0 % | `sparse_mla_prefill_mg_dual_kernel` 76 × 3.1 ms |
| fp8 quant, rope/norm/act, Engram, bf16, sampling | 46 | 0.5 % | |

Per layer per chunk the period between consecutive 973 MB copies is **90.3 ms median (p10 86.9, p90 97.0,
n = 77, sum 7,122 ms = 81 % of TTFT)** on both ranks, matching E22's "pipelined ≈ 87 ms/layer". Inside it:
≈ 29 ms HtoD (1.64 GB at 57 GB/s), ≈ 3 ms layout fix-up, ≈ 20 ms of the layer's compute + all-reduce, and
≈ 37 ms with the GPU idle. The trace is GPU-only, so the idle is attributed by elimination, two candidates:
(H1) kt-kernel's export of 3.3 GB (both ranks' slices) into the shared pinned buffer — 68 ms/layer at 52 GB/s
core-bound (E22a), longer than the GPU side of the layer, so the host blocks on `sync_write_weight_scale_to_buffer`
and the GPU drains; (H2) the launch thread starved by the export itself: kt's 16 export workers are the box's 16 physical cores
(CONFIG `KT_CPUINFER`: extra workers "float on SMT siblings and starve the two TP-rank launch threads"), so the
host enqueues the resident fused-MoE call's kernels slowly while the export runs — which would put the gap
inside a host call rather than at the explicit wait, as observed (the `cutlass_fused_moe` workspace is already
persistent since E34d-v (i), so allocation is not a candidate). H1 and H2 are the same cost seen from two sides
and have the same fix family; one serve with CPU + GPU profiler activities settles the split (T3.39's window 1).

## What this closes and what it opens

- **T3.21 closed (stop clause):** dense GEMM = 4 % of prefill; a 2× kernel is ≈ +2 %. The cuTile `(1,32,32)`
  path is slower than Triton; a CUTLASS `(1,32,32)` instantiation would be new kernel work for ≤ +4 %.
- **T3.27's prefill scope shrinks** to the same +4 % ceiling on the dense shapes; its decode scope (E20 decode
  GEMMs, resident-expert MXFP4) is unchanged.
- **Opened: T3.39 (proposed) — the streamed-expert pipeline period.** 90 ms/layer-chunk is 81 % of a 16K
  TTFT; the GPU side of it is ≈ 52 ms (29 HtoD + 3 fix-up + 20 compute). Window 1: a CPU + GPU profile of one
  chunk to split the 37 ms idle between H1 and H2 (H2 also has a one-knob probe: export with 14 workers).
  Window 2: export-free streaming
  — DMA kt-kernel's native expert bytes and do the layout transform on the GPU (1.6 GB is ≈ 1 ms there), which
  removes the 68 ms host export; the period floor is then ≈ 52 ms serialised (≈ +70 % prefill at 16K) or
  ≈ max(29, 23) ≈ 30 ms with the DMA on a side stream and a second staging set (+1.6 GiB VRAM per rank; up to
  ≈ 2.5×). Its first step is reading `write_weight_scale_to_buffer` to establish whether kt's layout →
  FlashInfer's is a GPU-expressible permutation, and measuring pageable vs `cudaHostRegister`ed HtoD out of the
  136 GiB expert arena (a second host copy does not fit: 246 GiB total).

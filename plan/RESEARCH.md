# Research log

What was read, what it established, and what is still open. Claims marked *primary* were verified against the artifact itself (checkpoint headers, config.json, this host). Claims marked *secondary* come from a vendor or community source and are cited so they can be re-checked; they were read, not reproduced.

## Model: DeepSeek-V4.1-Flash

Primary (`deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277`, `config.json` and shard headers):

- `DeepseekV41ForCausalLM`, `model_type deepseek_v41`; text config: 40 layers, hidden 5120, 384 routed experts + 1 shared, top-6 (`noaux_tc`, `sqrtsoftplus`), MoE intermediate 2304, 64 heads / 1 KV head, head_dim 512, q_lora 1280, o_lora 1024 in 8 groups.
- Attention is CSA2: `compress_ratios` per layer (0 for layers 0–1 and 40–42, 2 for 2–19, 1 for 20–39), `kv_source_layer_ids [2,8,14,20]`, 8 index-source layers, `index_topk 512`, sliding window 128, candidate top-2048 blocks of 8 from layer 20. This is why KV is ~890 B/token: only four source layers project the global KV.
- Engram: layers 1 and 14; tables of 384,006,168 and 384,016,682 rows × 256 (FP8 e4m3) with `[rows, 8]` UE8M0 scales; 8 heads; `engram_max_ngram_size 4`; `engram_vocab_size 16,000,000`.
- Context: `max_position_embeddings 1,048,576` = YaRN factor 16 over 65,536 original.
- Quantization as shipped: `fp8` block [32,32] UE8M0 scales, `expert_dtype fp4`. Experts are stored as INT8-packed FP4 pairs: 253.1 GiB payload + 15.8 GiB scales = 268.9 GiB for 15,360 experts (17.9 MiB each).
- DSpark: `mtp.0..2` with their own attention, 128-expert MoE, Markov head and confidence head; 7.39 GiB.
- Vision: 32-block ViT (hidden 1024) + aligner; 0.90 GiB.
- License: MIT (retained in `THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt`, sha256 `f2c6c602…`).

Secondary:

- HF model card: 552B backbone + 196B Engram; 8B active at prefill / 16B at decode (paper: HF papers/2609.19969).
- vLLM recipe (recipes.vllm.ai, "DeepSeek-V4.1", updated 2026-09-20): memory table lists routed+DSpark experts MXFP4 259.5 GiB, Engram FP8 183.1 GiB, attention/dense 6.9, embedding/head 3.9, UE8M0 scales 21.9; `vram_minimum_gb 614` for the all-GPU layout. **Our header-derived totals agree** (Engram FP8 payload 183.1 GiB exactly; total 475.2 GiB). Requires `vllm/vllm-openai:nightly` ≥ 2026-09-10 (PR #56228) or tag `deepseekv41-flash-0909`; flags `--tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --language-model-only`, `--engram-config '{"embedding_across_dp":true}'`; DSpark `--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`; thinking on by default at effort 50 (pin `thinking:false` in requests); no torch.compile; sanity `17*19=323`.
- SGLang: cookbook `docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`; preview image `lmsysorg/sglang:dev-dsv41` (NVIDIA card also lists `lmsysorg/sglang:dev-cu13-dsv41`); not in a pip release. Backends auto-select. `--reasoning-parser deepseek-v41`, `--tool-call-parser deepseekv41`, `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`, opt-in `--enable-decoder-swa-bounded-replay`. Engram host table: env `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared|per_rank`; needs hugepages; bitwise-identical output; +36% KV capacity on 4×GB300. Blog: lmsys.org/blog/2026-09-10-deepseek-v41.
- `nvidia/DeepSeek-V4.1-Flash-NVFP4@3431dde3…` (MIT, 2026-09-16): experts NVFP4 W4A4 g16; tested GB300 TP4; accuracy ≈ source. Header total 491.06 GiB (the index metadata still carries the upstream 475.24 GiB `total_size`; header bytes are authoritative).
- `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4@dfce15b9…` (MIT): additionally quantizes Engram FP8→FP4 (97.28 GiB, cosine 0.9934, lossy, unevaluated). The card states the 302 GiB non-Engram set must be resident and that "two 120 GB boxes cannot hold it". Header total 399.90 GiB.

## Community SM120 deployments (secondary)

- vLLM issue #56700: 8 × RTX PRO 6000 works on tag `deepseekv41-flash-0909`, TP8, FP8 KV block 128, 0.85 util, ~91 GiB/GPU, 4,584 tok/s prefill eager, 950K needle OK. Pitfalls: DeepGEMM autotune hang, FlashInfer sparse-MLA SM120 dispatch gaps, FP8 KV scale layout bug visible only ≥ 100K, CUDA-graph capture crash, MXFP4 → Marlin fallback (#30135).
- `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000` (MIT; SGLang base `sha256:c4ca6511…`; model rev `fb2764a5…`): 4 × 96 GB + 128 GB DDR5 + NVMe. Engram via NVMe direct I/O + 64 GiB DDR5 row cache (`OFFLOAD_MODE=nvme|ram`; ram needs 189 + 24 GiB). TP4/EP4, DSpark 5, ctx 409,600–524,288, 0.85–0.95 mem. Results: C1 ~200–230 tok/s, C8 ~700–750 tok/s, prefill ~7,000 tok/s, DSpark acceptance 79–88%, 4.06 M populated KV, GPUs at 275 W. **This is the calibration point for `feasibility-calc.py`** and the template for our 4-GPU profile.
- vLLM issue #47266 (V4, not 4.1): SM120 needed nine workarounds incl. `--disable-custom-all-reduce`.
- Earlier V4 SM120 recipes: hermia-ai (4×, stock vLLM 0.25.1, Marlin), lastloop-ai vllm-blackwell-guide (2×, 150–170 tok/s), jacklarmer/deepseek-v4-flash-0731-sm120.

## CPU-expert hybrid engines (secondary, read 2026-09-21)

- **KTransformers + sglang-kt** (kvcache-ai/ktransformers, `doc/en/DeepSeek-V4-Flash.md`): routed experts computed on CPU with `--kt-method MXFP4` (AVX2 sufficient; AMX not required — this corrects HARDWARE.md's earlier note), hot experts on GPU via `--kt-num-gpu-experts N`, `--kt-enable-dynamic-expert-update`, `--kt-expert-placement-strategy frequency`, layerwise GPU prefill `--kt-gpu-prefill-token-threshold`. Docker `approachingai/ktransformers:DSV4-specific`; needs flashinfer ≥ 0.6.9, transformers 4.57.1, tilelang. Validated for **DeepSeek-V4-Flash only**: 20–28 tok/s with all experts on CPU, 1 × RTX 5090 + Xeon. **No `deepseek_v41` support**; porting CSA2, Engram host table, hyper-connections and DSpark is the build (E6 stage 2).
- **llama.cpp**: `vcruz305/DeepSeek-V4.1-Flash-GGUF` (arch `deepseek41`; Q2_K 246.3 GiB, Q3_K_M 323.4 GiB, Q4_K_M pending; Engram tables ≈ 40,284 MiB each at q3_K; Q1_0 withdrawn). Runtime branch vcruz305/llama.cpp `runtime/deepseek41`: loader, Engram and hyper-connections verified, **CSA2 sparse attention pending**; conversion PR ggml-org/llama.cpp#28696. Not runnable yet; when it is, `-ot` regex expert offload gives the three-tier layout without a port.
- Community: r/BlackwellPerformance asks whether V4.1-Flash can run on 2 × RTX PRO 6000 — unanswered. HF discussion #28 on the model: 0xSero's 4 × 6000 recipe; a vLLM patch reaching ~170 tok/s on 4 × 6000 without vision. Reddit reports that speculative decoding hurts CPU-bandwidth-bound hybrids (E10).
- Why CPU compute rather than PCIe weight streaming (own analysis, FEASIBILITY.md Option B): same 1.75 GiB/token of expert weights either way; DRAM read 146 GB/s measured / 409.6 theoretical versus ~100 GB/s over two PCIe 5.0 x16 links; CPU kernels exist, a `deepseek_v41` streaming engine does not.

## SM120 kernel status (secondary, read 2026-09-21)

- **Hardware/CUTLASS.** SM120 has hardware block-scaled `mma.sync` (Colfax, "NVFP4 blockscaled GEMM on SM12x"); no TMEM/tcgen05, 99 KiB shared memory per block versus 228 KiB on SM100, so SM100 kernels trap and SM100-sized tiles fail to launch. CUTLASS 3.9 added SM120 block-scaled dense and sparse mainloops; 4.2 added SM120 mixed-input block-scaled grouped GEMM and a blockwise example (87); 4.3 added a MoE grouped-GEMM API. cuBLAS exposes no FP4 GEMM on SM120; Triton `tl.dot_scaled` is a software fallback there.
- **Engines.** FlashInfer `mm_fp4` and its CUTLASS NVFP4 MoE fail on SM120 (issue #2577: cudnn unsupported, cutlass returns zeros, trtllm refuses CC 120; PR #2725 carries SM120 patches). vLLM detects the failure and falls back to **Marlin W4A16** (FP4 → FP16 dequant, no FP4 tensor cores; 40–50% throughput cost; `VLLM_MOE_FORCE_MARLIN=1`), which is why the earlier MXFP4 → Marlin note (#30135) applies to V4.1 experts. CUTLASS #3096 documents the TMA warp-specialized grouped-GEMM failure and a `compute_120f` fix reaching 39 tok/s native FP4 on a 397B MoE across 4 × RTX PRO 6000 (vs 50.5 tok/s on Marlin). TRT-LLM #12706: `TRTLLM_MOE_BACKEND=TRITON` is 6.7× faster than its CUTLASS backend on SM120/121. SGLang 0.5.8 produced NaN on SM120 for that model; `dev-dsv41` on SM120 is only proven by 0xSero's V4.1 run.
- **Speculative decoding interaction.** On the Marlin path MTP acceptance dropped (61–85% vs 89%) and net throughput fell 22%, attributed to draft heads trained on native FP4 activations seeing W4A16-dequantised values. DSpark on our stack will be measured under whichever path E12 finds (E10).
- **Custom kernels in the wild.** chsasank/blackwell: custom CUTLASS NVFP4 GEMM + attention for SM120 in development (GEMM at 38% of peak). VincentKaufmann/fp4-cuda-kernel: CUTLASS 3.8 FP4 GEMM for SM120/121, 143 TFLOP/s, reports that MXFP4 (E2M1 + E8M0) had to be re-quantised to NVFP4 (UE4M3) for the tensor cores — **to be verified against the PTX ISA's mxf4 kinds in E13**, because V4.1's experts are MXFP4 with UE8M0 scales and a forced re-quantisation would change the numerics DeepSeek trained for.
- Consequence for this plan: the GPU side of the hybrid (5 ms/token assumed) is a Marlin-path number until E12 measures it, and a native SM120 MXFP4 grouped GEMM (E13) is the one place a custom CUDA kernel could move both the 2-GPU hybrid and the 4-GPU profile.

## Baseline on this host (primary, sibling repo `deepseek-v4-flash-nvfp4-tp2`)

`nvidia/DeepSeek-V4-Flash-0731-NVFP4@f1caa71142bd0be02f728c79f75042ac1e461579`, TP2, 1,048,576 context at 96% memory, DSpark K5: 93.66 → 200.64 tok/s single-stream, 226 tok/s at C2, 998,869-token prompt in 513.8 s. Checkpoint is cached locally (164 GB).

## Source document audit

The Google Doc supplied with the request is an AI-generated plan. Its correct claims: CSA2 KV ≈ 890 B/token; Engram tables ≈ 189 GiB fit in 256 GB DDR5; SGLang has an Engram host path. Its incorrect claims: "~151 GB 4-bit backbone" (that is V4-Flash, not V4.1), SGLang flag `--offload-engram-to host-dram` and model id `sgl-project/DeepSeek-V4.1-Flash-FP4` (neither exists), vLLM `--offload-config` (does not exist). Its eight numbered citations resolve to homepages (mindstudio.ai, reddit.com, github.com, remio.ai, yottalabs.ai), not to the pages that would support the claims. Treated as a hypothesis list, fully re-derived here.

## Open questions

1. ~~**DIMM specification.**~~ Resolved 2026-09-20: 8 × Kingston KF564R32-32 (FURY Renegade Pro), SPD 4800 MT/s, configured 6400 MT/s (EXPO active), one per channel. Remaining unknown: running voltage (SMBIOS reports 1.1 V; the 6400 profile is specified at 1.4 V) and CAS latency, neither exposed by SMBIOS. Affects E1 and nothing else in the verdict.
2. **Measured DRAM read bandwidth and CPU expert GEMV time.** The two numbers that set the 2-GPU hybrid ceiling (E1b, E6 stage 0); PCIe H2D (E4) sets the streaming fallback and layerwise prefill cost.
3. **Which MoE kernel runs on SM120.** 0xSero proves SGLang `dev-dsv41` runs at TP4; whether the experts use a native block-scaled path or a Marlin W4A16 fallback decides GPU ms/token, prefill speed and DSpark acceptance. E12 answers it with nsys.
9. **Does SM120 accept MXFP4 (UE8M0 scales) directly in block-scaled MMA, or only NVFP4 (UE4M3)?** Decides whether E13's kernel is lossless for the upstream checkpoint.
4. **Engram-in-DRAM vs NVMe A/B** has no published data; E9 provides it.
5. **Does `--language-model-only` in vLLM / the SGLang equivalent skip loading the 0.90 GiB vision and 7.39 GiB DSpark tensors?** Changes the resident total by ≤ 8.3 GiB; does not change the 2-GPU verdict.
6. **Engram FP4 (LibertAI) / q3_K quality.** Unevaluated by their authors. Matters for the 2-GPU hybrid, where full FP8 tables in DRAM do not fit (E11).
7. **MXFP4 format mapping.** Upstream experts are INT4-pair packed with UE8M0 block-32 scales; KT's CPU `MXFP4` kernel and llama.cpp's Q4 formats each expect their own layout. Whether a lossless repack exists (it should: same 4-bit values, same block size) decides whether the hybrid is bit-identical to the resident model.
8. **3-GPU parallelism.** 64 heads / 384 experts do not divide by 3; EP3 with replicated attention or uneven PP are the candidates. Which one sglang-kt or llama.cpp supports decides the 3-card profile.

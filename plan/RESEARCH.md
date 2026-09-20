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

## Baseline on this host (primary, sibling repo `deepseek-v4-flash-nvfp4-tp2`)

`nvidia/DeepSeek-V4-Flash-0731-NVFP4@f1caa71142bd0be02f728c79f75042ac1e461579`, TP2, 1,048,576 context at 96% memory, DSpark K5: 93.66 → 200.64 tok/s single-stream, 226 tok/s at C2, 998,869-token prompt in 513.8 s. Checkpoint is cached locally (164 GB).

## Source document audit

The Google Doc supplied with the request is an AI-generated plan. Its correct claims: CSA2 KV ≈ 890 B/token; Engram tables ≈ 189 GiB fit in 256 GB DDR5; SGLang has an Engram host path. Its incorrect claims: "~151 GB 4-bit backbone" (that is V4-Flash, not V4.1), SGLang flag `--offload-engram-to host-dram` and model id `sgl-project/DeepSeek-V4.1-Flash-FP4` (neither exists), vLLM `--offload-config` (does not exist). Its eight numbered citations resolve to homepages (mindstudio.ai, reddit.com, github.com, remio.ai, yottalabs.ai), not to the pages that would support the claims. Treated as a hypothesis list, fully re-derived here.

## Open questions

1. **DIMM specification.** Owner reports Kingston FURY DDR5-6400; `dmidecode` needs root. Configured speed may be below 6400 with 8 RDIMMs populated on WRX90 (JEDEC 5600 vs XMP/EXPO 6400). Affects E1 and nothing else in the verdict.
2. **Measured PCIe H2D.** The single number that moves the Option B ceiling (E4).
3. **SGLang `dev-dsv41` on SM120 with `--tp 4`.** 0xSero proves it runs; whether the FlashInfer MXFP4 grouped-GEMM path is used or a Marlin fallback (as in vLLM) decides prefill speed. Check the server log's kernel selection in E9.
4. **Engram-in-DRAM vs NVMe A/B** has no published data; E9 provides it.
5. **Does `--language-model-only` in vLLM / the SGLang equivalent skip loading the 0.90 GiB vision and 7.39 GiB DSpark tensors?** Changes the resident total by ≤ 8.3 GiB; does not change the 2-GPU verdict.
6. **Engram FP4 (LibertAI) quality.** Unevaluated by its author. Only matters if host DRAM becomes the constraint (Option B), not for Option A on this host.

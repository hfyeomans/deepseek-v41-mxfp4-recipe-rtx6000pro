# Feasibility: DeepSeek-V4.1-Flash on 2× RTX PRO 6000 Blackwell 96 GB

Verdict, stated up front: **the 4-bit DeepSeek-V4.1-Flash checkpoint does not fit two RTX PRO 6000 Blackwell GPUs with any serving stack that exists today, even with the Engram tables moved off the GPUs.** The shortfall is about 105–112 GiB of VRAM, roughly 39% of the routed-expert weights. Closing it requires one of: two more GPUs (proven), an expert-offload engine that does not exist for this architecture on SM120, or a smaller checkpoint (sub-3-bit experts or expert pruning) for which no SM120 kernels exist yet.

Every number below is either measured on this host, read from the checkpoint headers by [`scripts/checkpoint-budget.py`](../scripts/checkpoint-budget.py), or derived by [`scripts/feasibility-calc.py`](../scripts/feasibility-calc.py). Re-run them; do not trust the prose.

## 1. What the checkpoint actually weighs

Byte totals read from the safetensors headers of every shard (no weights downloaded). Committed in [`evidence/`](../evidence/).

| Component | `deepseek-ai/DeepSeek-V4.1-Flash` (upstream MXFP4 experts) | `nvidia/…-NVFP4` | `LibertAIDAI/…-NVFP4` |
|---|---:|---:|---:|
| Routed experts (40 layers × 384) | **268.95 GiB** (253.1 INT4-packed + 15.8 UE8M0 g32 scales) | 284.77 GiB (253.1 + 31.6 FP8 g16 scales) | 284.77 GiB |
| Engram tables (layers 1, 14) | **188.83 GiB** (183.1 FP8 e4m3 + 5.7 scales) | 188.83 GiB | 97.28 GiB (FP4, lossy, unevaluated) |
| DSpark (3 MTP stages) | 7.39 GiB | 7.39 GiB | 7.78 GiB |
| Attention (CSA2, indexer, compressor) | 4.80 GiB | 4.80 GiB | 4.80 GiB |
| Embedding + LM head | 2.47 GiB | 2.47 GiB | 2.47 GiB |
| Shared expert | 1.32 GiB | 1.32 GiB | 1.32 GiB |
| Vision encoder + aligner | 0.90 GiB | 0.90 GiB | 0.90 GiB |
| Engram projections, router, hyper-connections, norms | 0.59 GiB | 0.59 GiB | 0.59 GiB |
| **Total** | **475.24 GiB** | 491.06 GiB | 399.90 GiB |
| **Resident excluding Engram tables** | **286.41 GiB** | 302.23 GiB | 302.62 GiB |
| Resident, text-only, no DSpark | **278.12 GiB** | 293.94 GiB | 293.94 GiB |

Three consequences:

1. The upstream checkpoint is *already* 4-bit where it matters. Experts are MXFP4 (`expert_dtype: fp4`, UE8M0 block-32 scales). There is no "FP4 quantization" step that makes it smaller than what DeepSeek ships. NVIDIA's NVFP4 re-quantization is 15.8 GiB **larger** because g16 FP8 scales double the scale overhead; its value is kernel compatibility (W4A4 cuBLAS/CUTLASS NVFP4 paths), not size.
2. The average routed expert is 17.9 MiB. Per layer, all 384 experts are 6.72 GiB. Decode activates 6 of 384 per layer (1.6%).
3. The Engram tables are two FP8 embedding tables stored as `[384,006,168, 256]` and `[384,016,682, 256]` e4m3 weights with `[rows, 8]` UE8M0 scales (one scale per 32-wide block, i.e. 8 heads × 32 dims). They are pure lookups: 48 random 264-byte row reads per decoded token (2 layers × 8 heads × 3 n-gram sizes, `engram_max_ngram_size: 4`). Getting them off the GPU is architecturally free; SGLang already does it.

## 2. What this host can hold

Measured by [`scripts/hardware-inventory.sh`](../scripts/hardware-inventory.sh); full sheet in [HARDWARE.md](HARDWARE.md).

| Resource | Value |
|---|---:|
| GPU VRAM | 2 × 97,887 MiB = 191.19 GiB physical |
| Usable for weights + KV at 0.95 utilisation, 4 GiB/GPU reserve | **173.6 GiB** |
| Host DRAM | 246.8 GiB (≈256 GB DDR5, 8 channels) |
| Host DRAM after 24 GiB OS/process reserve | 222.8 GiB |
| NVMe (Samsung 9100 PRO 4 TB, PCIe 5.0 x4) | 1.7 TB free |

The 4 GiB/GPU reserve covers CUDA context, NCCL, FlashInfer workspaces and activations. Community SM120 deployments land at 0.85–0.95 utilisation with 4–6 GiB of non-weight, non-KV residency per card, so 4 GiB is the *optimistic* end.

## 3. The placement table

`scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --context-tokens 262144` (one 262K sequence; KV at 890 B/token is 0.22 GiB and irrelevant):

| Layout | Usable VRAM | VRAM needed | Margin | Per GPU | Host needed | Host margin | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Everything on GPU (stock vLLM/SGLang) | 173.6 | 474.6 | **−300.9** | 237.3 | 24 | 222.8 | does not fit |
| Engram tables in host DRAM (SGLang host table) | 173.6 | 285.7 | **−112.1** | 142.9 | 212.8 | 34.0 | does not fit |
| Engram on NVMe + 64 GiB host cache (0xSero pattern) | 173.6 | 285.7 | **−112.1** | 142.9 | 88 | 158.8 | does not fit |
| Same, text-only, DSpark and vision removed | 173.6 | 278.3 | **−104.7** | 139.2 | — | — | does not fit |

Each GPU would need to hold ~140 GiB against 95.6 GiB physical. No memory-utilisation flag changes that. The Google Doc's "~151 GB 4-bit backbone" is the size of DeepSeek-**V4**-Flash (284B); V4.1-Flash's backbone is 552B and its non-Engram weights are 286 GiB.

The same calculator applied to the only proven consumer-GPU deployment (0xSero: 4 × RTX PRO 6000, 128 GB DDR5, Engram on NVMe) returns FITS with 60 GiB VRAM margin and 40 GiB host margin, and correctly returns DOES NOT FIT for Engram-in-host-DRAM on 128 GB. That is the calibration point for the model.

## 4. What would close the ~105 GiB gap

Non-expert resident weights (text-only) are 9.2 GiB; the remaining 164 GiB of the 173.6 GiB budget is the **expert ceiling**. Current experts are 268.9 GiB, so experts must shrink or move by a factor of 0.61.

### Option A — Add two GPUs (4× RTX PRO 6000). Proven.
286.4 GiB across four cards is 71.8 GiB each, leaving ~20 GiB/GPU for KV (≈22 M tokens of KV at 890 B/token, so context is compute-bound, not memory-bound). 0xSero reports ~200–230 tok/s at C1 with DSpark, ~700–750 tok/s at C8, ~7,000 tok/s prefill, on SGLang `dev-dsv41` with Engram served from NVMe + DDR5 cache. This host's 246.8 GiB DRAM is enough to keep the full FP8 Engram tables **in DRAM** (188.8 + 24 reserve = 212.8 GiB), which 0xSero could not do on 128 GB; that is the one variable where this host is better positioned. Cost: two more cards, a second PSU or power budget review (4 × 300 W Max-Q + 350 W CPU platform), and PCIe lane planning (WRX90E-SAGE SE has seven x16 slots off 128 lanes). This is the only path with a qualified result on similar hardware.

### Option B — MoE-aware expert offload to host DRAM. Not implemented for this architecture.
Keep 164 GiB of experts on GPU, stream the other 105 GiB (39%) from pinned host DRAM per step. Host budget: 105 GiB experts + 97 GiB FP4 Engram (LibertAI) + 24 GiB reserve = 226 GiB against 246.8 GiB — it *fits the DRAM*. The problem is bandwidth and software:

- Decode, batch 1: 240 expert activations per token; 39% offloaded and uniform routing (DeepSeek's `noaux_tc` load balancing makes routing close to uniform, so hot-expert caching gains little) → 93.6 × 17.9 MiB = **1.64 GiB per token** across the two GPUs' PCIe 5.0 x16 links. At a realistic ~50 GiB/s pinned H2D per link the ceiling is **~61 tok/s** before any compute, versus 200 tok/s for the existing V4-Flash-0731 baseline on the same two cards. DSpark does not help: a 5-token block activates more distinct experts per step.
- Prefill, 4096-token chunks: every offloaded expert is touched per chunk → 52 GiB per GPU per chunk → ~1.05 s at 50 GiB/s → **≤ 3,900 tok/s** ceiling, probably 2–3K with imperfect overlap. This part is tolerable.
- Software: vLLM `--cpu-offload-gb` and SGLang's generic CPU offload are parameter-level and stream *all* offloaded parameters every step regardless of routing; they do not know which experts a token selected. Neither has a `deepseek_v41`-aware expert offloader on SM120. KTransformers/ktransformers-style CPU-expert execution targets Intel AMX; this CPU is Zen 5 with AVX-512 but no AMX, and 16 cores cannot compute 8 B active MoE parameters per token at useful speed (DRAM roofline ≈ 409.6 GB/s theoretical / ~4.2 GiB per token if *all* experts were CPU-resident → <100 tok/s even with perfect kernels; the realistic number is far lower).

Verdict: feasible as a research prototype yielding ≤ ~60 tok/s decode; not a recipe. Measured PCIe H2D bandwidth ([`bench-pcie-h2d.py`](../scripts/bench-pcie-h2d.py)) is the first number that would move this estimate.

### Option C — Smaller expert weights. No kernels.
- 3.0 bpw effective experts → ~190 GiB. Still 26 GiB over.
- ≤ 2.6 bpw effective (e.g. 2-bit trellis/EXL3-style with group scales) → ~164 GiB. Fits with zero KV headroom. No SM120 kernels exist for sub-4-bit MoE grouped GEMM in vLLM or SGLang for `deepseek_v41`; the GLM-5.3 EXL3 K3 recipe on this host required a bespoke fork (tpurtell/B12x) that took months.
- Quality at 2.6 bpw on a model whose experts are *already* 4-bit is unknown and likely poor; DeepSeek trained the experts FP4-aware.

### Option D — Expert pruning. Research, no evaluated artifact.
Keep 4-bit experts but drop ~39% of them per layer (384 → ≈ 232) via calibration-based pruning (REAP / frequency-based). Fits at ~164 GiB. Requires a calibration pipeline, a re-indexed router, and a quality evaluation; nobody has published this for V4.1-Flash. Interesting as a boundary experiment; not shippable.

### Option E — Three GPUs. Marginal, no parallelism fit.
3 × 95.6 GiB at 0.97 utilisation, 3 GiB reserve → 269.2 GiB usable versus 278.3 GiB text-only need: **−9.2 GiB**. And TP3 is not valid for 64 attention heads / 384 experts in either engine. Not viable.

## 5. What *does* fit two RTX PRO 6000 today

`nvidia/DeepSeek-V4-Flash-0731-NVFP4@f1caa71142bd0be02f728c79f75042ac1e461579` (V4-Flash, 284B, NVFP4 experts) — already qualified on this exact host in the owner's `deepseek-v4-flash-nvfp4-tp2` recipe: TP2, 1,048,576 context, DSpark K5, 93.66 → 200.64 tok/s single-stream, 226 tok/s at C2, a 998,869-token prompt in 513.8 s. This is the **baseline to beat**, and the only DeepSeek FP4 model that runs here without adding hardware. The V4.1 plan's benchmark harness must run against it first so every V4.1 number (on 4 GPUs or in a prototype) is comparable.

## 6. Audit of the source document's claims

| Claim | Status | Evidence |
|---|---|---|
| CSA2 KV cache ≈ 890 B/token; 262K ≈ 233 MB | **Correct** | vLLM recipe; LibertAI card; `feasibility-calc.py` KV column |
| "~151 GB 4-bit backbone" fits in 192 GB | **Wrong model.** V4-Flash (284B) is ~151 GiB; V4.1-Flash non-Engram weights are 286.4 GiB | `evidence/budget-upstream-bf16fp8.json` |
| Engram tables ~189–203 GB in 256 GB DDR5 | **Correct size** (188.83 GiB); host DRAM can hold them with 34 GiB spare | budget JSON; calc `engram-host` row |
| `--offload-engram-to host-dram` flag; model `sgl-project/DeepSeek-V4.1-Flash-FP4` | **Do not exist.** Real mechanism: `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` (+ `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared\|per_rank`), image `lmsysorg/sglang:dev-dsv41`; model `deepseek-ai/DeepSeek-V4.1-Flash` | SGLang cookbook `DeepSeek-V4_1.mdx` |
| `--disable-flashinfer-autotune`, `--chunked-prefill-size 4096` | Plausible but unverified flags for `dev-dsv41`; cookbook says let backends auto-select | plan task T2.3 |
| vLLM `--offload-config '{"cpu_tier_size_gb":…}'` | **Does not exist** in vLLM CLI; the recipe uses `--engram-config` and Engram CPU offload in the B300 profile | recipes.vllm.ai DeepSeek-V4.1 page |
| `--enforce-eager` required on V4-family | Partially supported: SM120 reports show CUDA-graph capture crashes (vLLM #56700); eager costs decode speed | TROUBLESHOOTING.md |
| SGLang "recommended" for this build | Correct in spirit: SGLang has the Engram host-table path and the only SM120 4-GPU result; but no stack fits 2 GPUs | this document |

## 7. Bottom line for the plan

- **Ship nothing labelled "qualified" for V4.1-Flash on 2 GPUs.** `scripts/preflight.sh` fails closed on the memory gate above.
- Run the host microbenchmarks (DRAM/NVMe Engram gather, PCIe H2D) so Options A and B have measured inputs instead of estimates.
- Re-baseline V4-Flash-0731 NVFP4 with this repo's harness.
- Keep a 4-GPU SGLang profile in `serve.sh`, explicitly labelled unqualified, ready to run the day two more cards are installed.
- Treat Options B and D as the "pushing the boundaries" track: measurable prototypes with explicit ceilings, not promises.

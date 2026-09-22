# DeepSeek-V4.1-Flash FP4 on RTX PRO 6000 Blackwell: feasibility, plan, and recipe scaffold

A measured answer to "can two RTX PRO 6000 Blackwell 96 GB serve DeepSeek-V4.1-Flash at 262K context, and if not, what would?" — plus a phased plan to benchmark what does run, and the scripts to do it. Modeled on the [GLM-5.3 Flash dual-RTX-PRO-6000 recipe](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell): pinned revisions, fail-closed preflight, reproducible tables, no host identifiers in public evidence.

## Verdict

**Two RTX PRO 6000 cannot hold DeepSeek-V4.1-Flash in VRAM — the shortfall is 112 GiB, not a flag. They can serve it as a three-tier model: VRAM + host DRAM/CPU + NVMe. The cost is decode speed, and building the engine is the work.**

| | GiB |
|---|---:|
| V4.1-Flash weights that must be GPU-resident in a stock engine (everything except the Engram tables; experts already 4-bit) | **286.4** |
| Usable VRAM on 2 × 95.6 GiB at 0.95 utilisation minus 4 GiB/GPU reserve | **173.6** |
| Margin with Engram tables in host DRAM (the best stock 2-GPU layout) | **−112.1** |
| KV cache for one 262,144-token sequence (CSA2, 890 B/token) | 0.22 |

The bytes are read from the shard headers of `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…` ([`evidence/budget-upstream-bf16fp8.json`](evidence/budget-upstream-bf16fp8.json)); the placement math is [`scripts/feasibility-calc.py`](scripts/feasibility-calc.py). The routed experts are **already MXFP4** in the upstream checkpoint (268.9 GiB); NVIDIA's NVFP4 re-quantization is 15.8 GiB *larger*. The 262K context itself is trivial — the source document was right about that and wrong about the model size (its "~151 GB backbone" is DeepSeek-**V4**-Flash).

### The 2-GPU plan: three tiers

`scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2,3,4 --layout hybrid-cpu`:

| Tier | What lives there | 2 GPUs | 3 GPUs (future) | 4 GPUs (future) |
|---|---|---:|---:|---:|
| **VRAM** | attention, shared expert, DSpark, embeddings (16.6 GiB) + as many routed experts as fit | 58% of experts (8,958) | 91% (13,916) | 100% |
| **DRAM (246.8 GiB), CPU compute** | routed experts that do not fit VRAM, computed on the 16 Zen 5 cores (KTransformers `MXFP4` method, AVX-512, no AMX needed) | 42% = **112.1 GiB** | 25.3 GiB | 0 |
| **NVMe** (Samsung 9100 PRO) + DRAM row cache | Engram tables, 188.8 GiB FP8, 48 random 264 B reads per token | 64 GiB cache | 64 GiB cache | or all in DRAM |
| Host needed / margin | | 200 / **+47 GiB** | 113 / +134 | 88 / +159 |
| DRAM read per decoded token | | 1.75 GiB | 0.40 GiB | 0 |
| **Decode ceiling**, CPU compute at 125 GB/s measured DRAM read / **B2 split** with PCIe streaming (E4b) | | **~50 / ~78 tok/s** | ~119 / ~155 | ~200 (GPU-bound) |
| Parallelism | | TP2 | **TP3 invalid** (64 heads) → EP3/PP | TP4 |
| Power on the 1,500 W PSU | | fine | CPU PPT cap | **needs 2,000 W or 225 W/card** |

Engram tiering on 2 GPUs: full FP8 tables in DRAM do **not** fit (−78 GiB); FP4 (+13 GiB) or q3_K (+32 GiB) tables do; NVMe + 64 GiB cache (+47 GiB) is the default because E3 showed both drives sustain decode-rate lookups. Full derivation, options and the 3/4-GPU columns: [docs/FEASIBILITY.md](docs/FEASIBILITY.md).

Measured so far on this host ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)): CPU DRAM read 125 GB/s (E1b; a two-CCD fabric limit, 30 % of the 409.6 theoretical); GPU DMA reads of the same DRAM 114 GB/s over both x16 links and **additive** with the CPU's (E4/E4b, 234 GB/s together); GDDR7 ≈1,470 GB/s per card (E2); Engram-style random row gather 4.15 M rows/s from DRAM, 403K reads/s from the Samsung 9100 PRO, 347K from the Crucial T705 (E3); a native MXFP4×MXFP4 grouped GEMM for SM120 (`kernels/e13/`) that verifies and streams expert weights at 1,397 GB/s at decode shapes and reaches 1,060 TFLOPS dense at 300 W (E13).

### Paths, ranked

| Path | Status | Expected result |
|---|---|---|
| **B. Three-tier hybrid on 2 GPUs** — port `deepseek_v41` into KTransformers' sglang-kt (validated for V4-Flash only), or run llama.cpp's `deepseek41` branch when CSA2 lands, or our own PCIe expert-streaming prototype | no engine runs it today; build track P3 | 56–89 tok/s decode at batch 1 (DRAM-bound), 2–3K tok/s prefill via layerwise GPU prefill; the first published 2-GPU V4.1-Flash number |
| **E. Three GPUs**, 91% of experts resident, EP3/PP | future comparison; no card | ~119–155 tok/s |
| **A. Four RTX PRO 6000** + Engram tables in DRAM (SGLang `dev-dsv41` host-table path) | proven pattern on 4× SM120 by 0xSero; profile written, unqualified; **1,500 W PSU cannot carry four cards at 300 W** ([power](docs/HARDWARE.md#power-and-cooling-planning-inputs-for-option-a)) | ~200 tok/s C1, ~700 tok/s C8, ~7K tok/s prefill (0xSero, 275 W) |
| **C/D. Sub-3-bit experts or 42% expert pruning** | no kernels / no evaluated artifact | research only; D gated on E7 routing skew |
| **Baseline: DeepSeek-V4-Flash-0731 NVFP4, TP2** | qualified on this host;

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

Exercise: write a 20-line Python function that quantises a 32-float block to MXFP4 (find max, pick power-of-two scale, round each value to the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 

# Feasibility: DeepSeek-V4.1-Flash on 2× RTX PRO 6000 Blackwell 96 GB (with 3- and 4-GPU comparison)

Verdict, stated up front: **the 4-bit DeepSeek-V4.1-Flash checkpoint does not fit two RTX PRO 6000 Blackwell GPUs as a GPU-resident model, even with the Engram tables moved off the GPUs** — the shortfall is 112 GiB of VRAM, 42% of the routed-expert weights. **It does fit this host as a three-tier model**: attention, shared expert, DSpark and 58% of the routed experts in VRAM; the other 42% (112 GiB) in host DRAM and computed on the CPU, KTransformers-style; Engram tables on NVMe behind a DRAM row cache (or quantized into DRAM). The memory works with 13–47 GiB of host to spare depending on how Engram is tiered. The price is decode speed: the CPU cores can read host DRAM at only ~125 GB/s on this 16-core (two-CCD) part, so CPU-computed experts alone cap decode at ~50 tok/s at batch 1 (40 at the 94 GB/s sustained while computing), against 200 tok/s for the V4-Flash model already qualified on these two cards. Measured in this repo's GPU window: the GPUs' DMA reads of the same DRAM (114 GB/s over both x16 links) are additive with the CPU's reads, so splitting the DRAM-resident experts between CPU compute and per-step PCIe streaming (Option B2) lifts the ceiling to ~71–78 tok/s. No engine runs this layout for `deepseek_v41` yet; §4 Option B is the build. Two more GPUs (Option A) remain the proven all-VRAM path and a third GPU (Option E) is the middle case; both are kept as comparison columns.

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
|---|---:|---:|---:|---:|---:|---
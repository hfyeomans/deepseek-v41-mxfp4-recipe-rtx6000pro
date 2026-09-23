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
|---|---:|---:|---:|---:|---:|---:|---|
| Everything on GPU (stock vLLM/SGLang) | 173.6 | 474.6 | **−300.9** | 237.3 | 24 | 222.8 | does not fit |
| Engram tables in host DRAM (SGLang host table) | 173.6 | 285.7 | **−112.1** | 142.9 | 212.8 | 34.0 | does not fit |
| Engram on NVMe + 64 GiB host cache (0xSero pattern) | 173.6 | 285.7 | **−112.1** | 142.9 | 88 | 158.8 | does not fit |
| Same, text-only, DSpark and vision removed | 173.6 | 278.3 | **−104.7** | 139.2 | — | — | does not fit |

Each GPU would need to hold ~140 GiB against 95.6 GiB physical. No memory-utilisation flag changes that. The Google Doc's "~151 GB 4-bit backbone" is the size of DeepSeek-**V4**-Flash (284B); V4.1-Flash's backbone is 552B and its non-Engram weights are 286 GiB.

The same calculator applied to the only proven consumer-GPU deployment (0xSero: 4 × RTX PRO 6000, 128 GB DDR5, Engram on NVMe) returns FITS with 60 GiB VRAM margin and 40 GiB host margin, and correctly returns DOES NOT FIT for Engram-in-host-DRAM on 128 GB. That is the calibration point for the model.

### 3a. The three-tier layout, by GPU count

`scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2,3,4 --layout hybrid-cpu --engram nvme` (defaults: Engram on NVMe with a 64 GiB DRAM row cache, 24 GiB host reserve, DRAM 125 GB/s read as measured in E1b, PCIe 57 GB/s per GPU as measured in E4, GPU side 5 ms/token, uniform routing). VRAM holds everything that is not a routed expert (16.6 GiB with DSpark, text-only) plus as many routed experts as fit; the rest live in DRAM and are computed on the CPU.

| GPUs | Routed experts in VRAM | Experts in DRAM (CPU) | DRAM read per decoded token | Decode ceiling, CPU compute @125 GB/s | Same experts streamed over PCIe @114 GB/s | **Split** (both paths, E4b) | Host needed / margin (Engram NVMe) | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| **2** | 8,958 of 15,360 (**58%**), 156.9 GiB | 6,402 (42%), **112.1 GiB** | 1.75 GiB | **~50 tok/s** (40 @94) | ~46 tok/s | **~78 tok/s** (71 @94+114) | 200.1 / **+46.7** GiB | **fits** |
| 3 | 13,916 (91%) | 1,444 (9%), 25.3 GiB | 0.40 GiB | ~126 tok/s | ~128 tok/s | 113.3 / +133.5 GiB | fits; **TP3 invalid** (64 heads) → EP3 or PP |
| 4 | 15,360 (100%) | 0 | 0 | ~200 tok/s (GPU-bound) | — | 88.0 / +158.8 GiB | fits (= Option A) |

*2026-09-23 (E27, BIOS slot 3 adopted):* the CPU-reachable read ceiling is now **131–132 GB/s** (FCLK 2100 MHz; per-CCD 66 GB/s), so the calculator default moved to `--dram-gbps 131`. The table above keeps the FCLK-2000 figures as the historical baseline; at 131 the 2-GPU row reads **~52 tok/s CPU compute / ~79 split**, the 3-GPU row ~121 / ~156, the 4-GPU row is unchanged (GPU-bound). Pass `--dram-gbps 125` to reproduce the table. The served hybrid moved the same way (C1 36.08 → 37.19 wall, +3.1 %, BENCHMARKS).

Engram tiering on 2 GPUs (`--engram …`): FP8 tables fully in DRAM need 324.9 GiB host → **−78.1 GiB, does not fit**; LibertAI FP4 tables (97.3 GiB) fit with +13.4 GiB; q3_K tables (78.7 GiB, GGUF ladder) fit with +32.0 GiB; NVMe + 64 GiB cache fits with +46.7 GiB. Default is NVMe because E3 showed both drives sustain decode-rate lookups and DRAM is the scarce resource in this layout.

Where the decode number comes from: 240 expert activations per token (40 layers × 6); with uniform routing 42% of them hit a DRAM-resident expert, so 101 × 17.9 MiB = 1.75 GiB of weights must be read from DRAM per token. At the 125 GB/s the CPU cores can read (E1b) that is 15.0 ms, plus ~5 ms of GPU work (the V4-Flash NVFP4 TP2 deployment decodes at 200 tok/s on these cards), ≈ 20 ms → 50 tok/s; at the 94 GB/s a bf16 GEMV actually sustained (E6 s0) it is 40 tok/s. The 125 GB/s is not a Python artefact: it is 30 % of the 409.6 GB/s theoretical because the 16-core 9955WX has two CCDs and the CCD-to-IO-die links cap reads (HARDWARE.md). Faster DIMMs would not help; a CPU with more CCDs would (`--dram-gbps 250` → ~83 tok/s). What *does* help on this box is that the GPUs read host DRAM through a different path: E4b measured CPU reads (120 GB/s) and both GPUs' DMA (114 GB/s) running together at 234 GB/s, 98 % additive. **Option B2** therefore splits the DRAM-resident experts: about half computed on the CPU, half streamed each step over PCIe to the GPU and run there with the E13 kernel. The calculator's `split ceiling` is 1000 / (5 + 1.75 GiB ÷ 239 GB/s) ≈ **78 tok/s** (71 at 94 + 114). It is an upper bound: it assumes the CPU kernel keeps up with its share of the bytes and that the streamed experts' GEMMs (E13: ≈1,400 GB/s, so ~0.4 ms for 50 experts) hide under the transfer. Frequency-based placement that keeps hot experts on the GPU raises it further (`--activation-skew`), and E7 measures whether V4.1 routing has any skew to exploit. The 5 ms GPU figure is confirmed by the E12 census as the conservative end of 3.8–5.1 ms per accepted token — and it is **not** a Marlin number: vLLM already dispatches the NVFP4 experts to a CUTLASS SM120 block-scaled grouped GEMM at the memory roofline, and those experts are only ~25 % of the step (dense FP8 projections 27 %, bf16 GEMMs 12 %, attention 11 %, DSpark draft 10 %). A faster expert kernel therefore cannot move the 5 ms much; what shrinks it is fewer GPU-resident bytes per token and, with 3 or 4 GPUs, the TP split of the dense layers. Speculative decoding (DSpark) is *not* free here: a 5-token block touches more distinct CPU experts per step, so its acceptance gain competes with extra DRAM traffic (E10). The stakes are larger than they look: the 5 ms is a *per accepted token* figure with DSpark on; the DSpark-off measurement (E8) is 11.2 ms per step, so a hybrid that cannot speculate is at 1000 / (11.2 + 15) ≈ **38 tok/s** even before CPU-side overheads, while one that keeps DSpark's 2.5–3 accepted tokens per verify step amortises both the GPU step and the DRAM reads.

## 4. What would close the ~112 GiB gap

Non-expert resident weights are 16.6 GiB with DSpark (9.2 GiB without); the remaining 157 GiB of the 173.6 GiB budget is the **expert ceiling**. Current experts are 268.9 GiB, so 42% of them must shrink or move.

### Option A — Add two GPUs (4× RTX PRO 6000). Proven.
286.4 GiB across four cards is 71.8 GiB each, leaving ~20 GiB/GPU for KV (≈22 M tokens of KV at 890 B/token, so context is compute-bound, not memory-bound). 0xSero reports ~200–230 tok/s at C1 with DSpark, ~700–750 tok/s at C8, ~7,000 tok/s prefill, on SGLang `dev-dsv41` with Engram served from NVMe + DDR5 cache. This host's 246.8 GiB DRAM is enough to keep the full FP8 Engram tables **in DRAM** (188.8 + 24 reserve = 212.8 GiB), which 0xSero could not do on 128 GB; that is the one variable where this host is better positioned. Cost: two more cards, **a power upgrade** (the host's 1,500 W PSU cannot carry 4 × 300 W Max-Q + a 350 W CPU ≈ 1,650 W; either a 2,000 W-class PSU on a 240 V/30 A circuit or a 225 W per-card cap — HARDWARE.md power table), and PCIe lane planning (WRX90E-SAGE SE has seven x16 slots off 128 lanes). This is the only path with a qualified result on similar hardware.

### Option B — Three-tier hybrid: 42% of experts computed on the CPU from DRAM. The primary 2-GPU track; software does not exist for `deepseek_v41` yet.
Keep 157 GiB of routed experts on the GPUs (8,958 of 15,360), leave the other 112 GiB in host DRAM, and run those experts **on the CPU** with the activations shipped over PCIe instead of the weights. Engram tables on NVMe with a DRAM row cache. Host budget: 112 GiB experts + 64 GiB Engram cache + 24 GiB reserve = 200 GiB against 246.8 GiB (§3a). Why CPU compute rather than streaming the weights to the GPU: the DRAM traffic is identical (1.75 GiB/token either way), but DRAM read bandwidth on this board is 146 GB/s measured / 409.6 GB/s theoretical versus ~100 GB/s aggregate over two PCIe 5.0 x16 links, and the CPU kernels already exist. What has to be true:

- **DRAM bandwidth.** 1.75 GiB/token at 125 GB/s (E1b, CPU read ceiling) → 15.0 ms → **~50 tok/s** with 5 ms of GPU work; at the 94 GB/s sustained by a bf16 GEMV (E6 s0) → 20 ms → ~40 tok/s. The CPU-side number is now measured and is a fabric limit of this two-CCD part, not a tuning problem. **Split path (B2):** add the GPUs' 114 GB/s of DMA reads, measured additive in E4b, and the DRAM side drops to 7.9 ms → ~78 tok/s. Remaining measurement: an actual MXFP4 CPU kernel (KTransformers `MXFP4` or llama.cpp) at 4-bit byte counts, to confirm compute keeps pace at ~94–125 GB/s.
- **CPU compute.** 101 CPU-expert activations/token × 3 GEMVs of 5120×2304 ≈ 3.6 GMAC/token in MXFP4. 16 Zen 5 cores with AVX-512 (no AMX) peak near 2 TFLOP/s FP32; a dequant-and-FMA kernel at a quarter of that is ~15 ms/token — the same order as the DRAM time, so compute must overlap DRAM streaming and be measured, not assumed. KTransformers' `MXFP4` CPU method requires only AVX2 and is the reference kernel; it reports 20–28 tok/s for **all** experts of V4-Flash on one RTX 5090 plus a Xeon. Here only 42% of activations go to the CPU.
- **Prefill.** Every DRAM-resident expert is touched by a 4,096-token chunk; the CPU cannot do that work at prefill rates. KTransformers solves it with layerwise GPU prefill (`--kt-gpu-prefill-token-threshold`): for long prompts, stream each layer's CPU experts to the GPU once per chunk and compute there. 112 GiB per chunk over two x16 links = 1.06 s at the measured 106 GiB/s aggregate (E4) → ≤ 3,900 tok/s prefill ceiling from the transfer alone; the GEMMs themselves are cheap (E13: 64 tokens × 384 experts of gate/up in 3.5 ms at 335 TFLOPS). Realistically 2–3K tok/s; 262K of prompt in ~1.5–2 minutes.
- **Software.** vLLM `--cpu-offload-gb` and SGLang's generic CPU offload are parameter-level and stream *all* offloaded parameters every step; they do not know which experts a token selected. The only engine with routed-expert CPU compute plus GPU-resident hot experts (`--kt-num-gpu-experts N`, `--kt-expert-placement-strategy frequency`, `--kt-enable-dynamic-expert-update`) is **KTransformers' sglang-kt hybrid**, validated for DeepSeek-**V4**-Flash only. Porting `deepseek_v41` (CSA2 attention, Engram host table, hyper-connections, DSpark) into sglang-kt is the build. Second route: **llama.cpp's `deepseek41` branch** (vcruz305/llama.cpp `runtime/deepseek41`; conversion PR ggml-org/llama.cpp#28696) — loader, Engram and hyper-connections work, CSA2 sparse attention pending; `-ot` regex expert offload and CPU MXFP4/Q4 kernels come for free once it runs, but at lower GPU efficiency than SGLang. Third route: a PCIe weight-streaming prototype of our own (the ~42 tok/s column) as the fallback if neither port lands.
- **Speculative decoding** interacts badly with a CPU-bound MoE: a DSpark block of 5 activates up to 5× the distinct CPU experts per step. Whether acceptance outruns the extra traffic is E10.

Verdict: **fits the memory of this host; decode ceiling 40–50 tok/s with CPU compute alone (measured DRAM read limits), 71–78 tok/s with the B2 split that also streams experts over PCIe; 0.2–0.4× the V4-Flash baseline's 200 tok/s, for a model with 2× the parameters.** Feasible as a build with measurable milestones (E6), not as a download-and-run recipe.

### Option C — Smaller expert weights. No kernels.
- 3.0 bpw effective experts → ~190 GiB. Still 26 GiB over.
- ≤ 2.6 bpw effective (e.g. 2-bit trellis/EXL3-style with group scales) → ~164 GiB. Fits with zero KV headroom. No SM120 kernels exist for sub-4-bit MoE grouped GEMM in vLLM or SGLang for `deepseek_v41`; the GLM-5.3 EXL3 K3 recipe on this host required a bespoke fork (tpurtell/B12x) that took months.
- Quality at 2.6 bpw on a model whose experts are *already* 4-bit is unknown and likely poor; DeepSeek trained the experts FP4-aware.

### Option D — Expert pruning. Research, no evaluated artifact.
Keep 4-bit experts but drop ~39% of them per layer (384 → ≈ 232) via calibration-based pruning (REAP / frequency-based). Fits at ~164 GiB. Requires a calibration pipeline, a re-indexed router, and a quality evaluation; nobody has published this for V4.1-Flash. Interesting as a boundary experiment; not shippable.

### Option E — Three GPUs. Fits as a hybrid; needs expert or pipeline parallelism instead of TP.
3 × 95.6 GiB at 0.95 utilisation, 4 GiB reserve → 260.4 GiB usable versus 285.7 GiB resident need: **−25.3 GiB** GPU-resident, so 91% of the routed experts fit and only 1,444 experts (25.3 GiB, 0.40 GiB/token of DRAM reads) stay on the CPU. Decode ceiling ~119 tok/s at 125 GB/s CPU compute, ~134 tok/s streaming the same experts over three x16 links (171 GB/s), ~155 tok/s split — at that traffic PCIe streaming beats CPU compute outright, so the third card removes the need for a CPU MXFP4 kernel. The catch is parallelism: 64 attention heads and 384 experts do not divide by 3, so plain TP3 is invalid in both engines; the layout must be EP3 (128 experts per card, attention replicated) or 2-stage PP with uneven layer splits. llama.cpp's layer-split mode handles 3 cards without that constraint. Power: three Max-Q at 300 W + CPU ≈ 1,350 W, inside the 1,500 W PSU only with the CPU PPT capped (HARDWARE.md power table). Worth keeping as the intermediate column: one card buys ~2.2× the 2-GPU ceiling.

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
| `--enforce-eager` required on V4-family | Partially supported as a symptom (SM120 capture crashes, vLLM #56700) but **not usable as a remedy**: E15 measured eager at 4.4× slower with DSpark and 5.6× without (15.9 tok/s) on V4-Flash. If the V4.1 engine cannot capture on SM120, fixing capture is on the critical path | BENCHMARKS.md §E15 |
| SGLang "recommended" for this build | Correct in spirit: SGLang has the Engram host-table path, the only SM120 4-GPU result, and the KTransformers hybrid (sglang-kt) is an SGLang fork; but no stock build fits 2 GPUs | this document |

## 7. Bottom line for the plan

- **The 2-GPU target is the three-tier hybrid (Option B), not a GPU-resident model.** `scripts/preflight.sh` keeps failing closed on the resident memory gate until an engine that runs the hybrid layout exists and has been verified here.
- ~~Finish the host measurements that set the hybrid's ceiling~~ — done (E1b 125 GB/s read, E6 s0 94 GB/s sustained, E4 57 GB/s per GPU, E4b additive). Calculator defaults updated; the `split ceiling` column is the B2 target. Next measurement: a real MXFP4 CPU kernel's ms/token (E6 s1 with KTransformers calibrates it).
- Re-baseline V4-Flash-0731 NVFP4 with this repo's harness; it is the number the hybrid competes with.
- Build track, in order of expected payoff: port `deepseek_v41` into sglang-kt (KTransformers hybrid); track llama.cpp's `deepseek41` branch and run it the day CSA2 lands; own PCIe-streaming prototype as the fallback.
- Keep the 3-GPU (EP3/PP) and 4-GPU (TP4, Engram in DRAM) profiles as comparison columns; both are labelled unqualified until cards exist, and both need the power decision in HARDWARE.md.
- Option D (pruning) stays a study gated on E7's routing histogram; Option C (sub-4-bit experts) is closed unless SM120 kernels appear.

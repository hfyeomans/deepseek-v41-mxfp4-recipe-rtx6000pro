# Engine research — five candidates for the recipe generator's engine-selection layer

Date: 2026-10-23. Fresh fetches (repos, READMEs, PRs) for all five; every claim cited. Complements
`recipe-generator/research/landscape.md` (placement-planning landscape). Verdict vocabulary reuses the ledger's
ADOPT / REJ / PARKED / CLOSED.

## Summary table

| Engine | What it is (2026-10-23) | Placement model | Unique contribution | Verdict |
|---|---|---|---|---|
| NVIDIA FLARE | Federated-learning application runtime — training orchestration + privacy across sites, **not a serving engine** | None (no weights/KV/placement concepts) | Job/workflow governance for multi-site fleets (wrong shape: aggregation- and privacy-centric) | REJ as engine · CLOSED (see §FLARE) |
| DeepInfer | DeepSeek's production inference runtime, open-sourced **2026-10-16**; Apache-2.0; offline inference for DeepSeek-V4.1-Flash; 21k★, active | Attention DP / balanced CP, **MoE EP + EPLB**; SM100 + Ascend 950; no CPU/NVMe offload | First-party EP+EPLB patterns at production scale; `step_peak` EPLB policy; batch-invariant + deterministic-sampling gates | ADOPT as multi-GPU-EP tier (watch) · borrow EPLB policy math |
| SGLang | Active serving framework, Apache-2.0, 37.7k★, pushed same day | TP/PP/EP+EPLB; `--cpu-offload-gb`; **Paged Experts** (merged 2026-10-19, budget→K); Waterfill/LPLB dispatch; HiCache KV offload | GPU-computes/host-stores expert paging with auto-K from memory budget; dispatch-time balancing | ADOPT as primary CPU-hybrid tier |
| TensorFold | Zig single-binary engine (Metal + CUDA), Apache-2.0, active; no Python/MLX at serve time | Quant-shrunk single/paired-machine; pinned-revision pulls; memory-floor reporting; prompt-cache/learn disk tier; no CPU offload | EXL3/TR3/EXLR8/MLX recipe culture; portable benchmark receipts; revision-pinned pulls | ADOPT for quant-shrunk tier · borrow pin/floor conventions |
| vLLM | Largest serving engine, Apache-2.0, 95.2k★, active | TP/EP/PD; **weight offload** (`--cpu-offload-gb`, `--cpu-offload-params`, UVA backend); Engram host offload; **KV tiering** (CPU + filesystem); EPLB; MXFP4/NVFP4 | recipes schema + VRAM-fit formula + verified-per-hardware recipes; richest offload surface | ADOPT as schema source + GPU-resident + secondary CPU-hybrid tier |

## NVIDIA FLARE — the one that is not a serving engine

1. **What it is.** "NVIDIA Federated Learning Application Runtime Environment … a domain-agnostic, open-source,
   extensible Python SDK that allows researchers and data scientists to adapt existing ML/DL workflows to a
   federated paradigm" — built-in FedAvg/FedProx/FedOpt/Scaffold, scatter-and-gather workflows, differential
   privacy / homomorphic encryption / PSI, simulator + POC mode, dashboard, authorization. Apache-2.0, active
   (pushed 2026-10-23), 988★. ([repo](https://github.com/NVIDIA/NVFlare), [README](https://raw.githubusercontent.com/NVIDIA/NVFlare/main/README.md))
2. **Placement model.** None. It schedules *jobs across sites* for collaborative training; it has no notion of
   checkpoint components, VRAM/DRAM/NVMe tiers, or serving flags.
3. **Unique contribution.** Only by analogy: "server-controlled workflows + dashboard + authorization + fault
   tolerance" is a fleet *governance* shape. If our generator ever had to roll recipe revisions out to a fleet
   and collect gate results back, FLARE is the wrong tool — its machinery (model aggregation, privacy
   preservation, client-side training loops) would all be dead weight.
4. **Verdict: REJ as engine, CLOSED.** Reopen trigger: a real multi-host *governance* requirement appears — even
   then, evaluate plain orchestration (Ansible/systemd/our manifests) before FLARE.

## DeepSeek DeepInfer — brand-new, not deprecated

1. **What it is.** "The high-performance inference framework that powers DeepSeek's large-scale production
   services. This open-source release provides offline inference" for DeepSeek-V4.1-Flash (default checkpoint
   `deepseek-ai/DeepSeek-V4.1-Flash-1017`). Apache-2.0; repo **created 2026-10-16**, pushed 2026-10-21, ~21k★,
   2.3k forks, discussions active. Built primarily on vLLM, with SGLang and TensorRT-LLM inspiration. Ships no
   KV-cache system or VE/Prefill/Decode disaggregation yet (listed as "required for a complete inference
   service, not included"). Hardware: NVIDIA SM100, Huawei Ascend 950. ([repo](https://github.com/deepseek-ai/DeepInfer), [README](https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/README.md))
2. **Placement model.** Parallelism only: Attention DP / balanced CP, **MoE EP + EPLB**. No CPU/NVMe offload
   knobs — it is a datacenter-scale, GPU-resident runtime. Placement is hand-authored per deployment
   (EP size, EPLB policy), not budget-computed.
3. **Unique contribution.** First-party, production-grade EP+EPLB configuration for V4.x models; `step_peak`
   EPLB placement policy (being ported elsewhere: ROCm/ATOM PR #2673 "add step_peak placement policy (ported
   from DeepInfer)"); batch-invariant inference and deterministic sampling as explicit correctness gates;
   confidence-scheduled DSpark verification; SWA bounded replay. ([ATOM PR #2673](https://github.com/ROCm/ATOM/pull/2673), [release coverage](https://gigazine.net/gsc_news/en/20261021-deepseek-deepinfer-recipe))
4. **Verdict: ADOPT as the multi-GPU EP tier (watch).** One week old and offline-only, so it cannot be a
   primary serving target yet; but for V4.x-family recipes needing EP+EPLB layouts it is the authoritative flag
   source, and its batch-invariance/determinism gates are worth copying into our qualification suite. Reopen
   trigger: DeepInfer ships a server/KV/disaggregation layer → promote to primary tier.

## SGLang — primary CPU-hybrid tier

1. **What it is.** Apache-2.0 serving framework, 37.7k★, pushed 2026-10-23; cookbook migrated into
   docs.sglang.io with hardware/variant/strategy axes. ([repo](https://github.com/sgl-project/sglang))
2. **Placement model.** TP/PP/EP + EPLB (static EPLB placements feed Waterfill/LPLB dispatch balancing);
   `--cpu-offload-gb` weight streaming; **Paged Experts** (merged 2026-10-19): K-of-E experts resident on GPU,
   all E in pinned host RAM, GPU computes, LRU residency + wave planning, **auto-K sized from the memory budget**
   (exact safetensors-header bytes + mem-fraction + KV reserve) — the only engine-side budget→placement loop
   found; HiCache KV offload (GPU→CPU→disk) with write_through mode in cookbook use.
   ([Paged Experts PR #42557](https://github.com/sgl-project/sglang/pull/42557), [waterfill/LPLB blog](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb), [HiCache cookbook PR #46017](https://github.com/sgl-project/sglang/pull/46017))
3. **Unique contribution.** Budget-computed expert residency (sizing.py reads safetensors headers — the same
   pattern as our `checkpoint-budget.py`); bit-identical paging results; measured offload trade-off tables
   (vs llama.cpp `--n-cpu-moe`).
4. **Verdict: ADOPT as the primary CPU-hybrid MoE tier.** Our generator should emit
   `--enable-paged-experts` / `--paged-experts-num-resident` (with our own K math) plus HiCache KV tiers, and
   borrow its auto-K sizing for calibration.

## TensorFold — quant-shrunk tier + recipe culture

1. **What it is.** Single Zig binary serving on Apple Silicon (Metal) and NVIDIA (CUDA), no Python/MLX at serve
   time; Apache-2.0; created 2026-06, active (pushed 2026-10-23), 1.5k★. Per-model native recipes under
   `docs/recipes/`; `tensorfold info REPO` reports a checkpoint's memory floor. DeepSeek checkpoints are
   explicitly **not served yet** (whole-model gate #827 open). ([repo](https://github.com/ashhart/TensorFold), [README](https://raw.githubusercontent.com/ashhart/TensorFold/blob/main/README.md))
2. **Placement model.** Quant-shrunk, mostly single-machine or explicitly paired two-machine (two GB10s TP=2,
   two M5 Ultra Macs for GLM-5.3 EXLR8); pinned-revision pulls; `--prompt-cache-gib` prefix budget and
   `--learn`/`--learn-gib` disk tiers for prefixes; no CPU/NVMe weight offload. Formats: MLX 4/6/8-bit, NVFP4,
   EXL3/TR3 (GLM-5.3-Flash 4bpw), EXLR8.
3. **Unique contribution.** The EXL3/TR3/EXLR8 quant-recipe culture; revision-pinned model pulls; portable
   `tensorfold benchmark` receipts; agent-oriented contributor skill (docs/recipes/SKILL.md).
4. **Verdict: ADOPT for the quant-shrunk single/paired-machine shape; borrow its pin-and-floor conventions.**
   Not a fit for CPU-hybrid or NVMe tiering. Reopen trigger: DeepSeek support lands (#827) → evaluate as a
   third CPU-hybrid engine.

## vLLM — schema source + richest offload surface

1. **What it is.** Apache-2.0 serving engine, 95.2k★, active (pushed 2026-10-23); owns the recipes schema
   (`vllm-project/vllm → recipes/models/`, rendered by vllm-project/recipes to recipes.vllm.ai + JSON API).
   ([repo](https://github.com/vllm-project/vllm), [recipes schema](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md))
2. **Placement model.** TP/EP/TEP/DEP/PD; **weight offload** via `--cpu-offload-gb` + `--cpu-offload-params`
   (named experts) + `--offload-backend uva`; per-component Engram host offload
   (`--engram-config '{"cpu_offload":true}'`); **KV tiering** with CPU + persistent-filesystem tiers
   (`TieringOffloadingSpec`, `--kv-offloading-backend`); EPLB; MXFP4/NVFP4/FP8 quants. All evidenced in the
   DeepSeek-V4.1-Flash recipe (component budget table, DGX Station row offloading 60 GiB of named experts,
   per-variant offload budget). ([DeepSeek-V4.1-Flash YAML](https://raw.githubusercontent.com/vllm-project/recipes/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml), [EPLB docs](https://docs.vllm.ai/en/latest/api/vllm/distributed/eplb/policy/default/))
3. **Unique contribution.** The only machine-readable recipe ecosystem: YAML schema, `ceil(params×B×1.2)` VRAM
   formula + mixed-precision correction, per-hardware verified map, provenance (repo/commit/path), JSON API
   endpoints — the publication format our generator should emit into.
4. **Verdict: ADOPT as schema/provenance source and GPU-resident + secondary CPU-hybrid tier.** Placement is
   author-planned (per recipe YAML), not budget-computed — that stays ours.

## Confirm/deny (the two specifics)

- **(a) "DeepInfer deprecated in favor of SGLang" — DENIED.** The repo did not exist until 2026-10-16, is Apache-2.0,
  unarchived, pushed 2026-10-21, ~21k★, with active discussions; the README announces it as DeepSeek's production
  framework. Its EPLB story: "MoE EP + EPLB" is a shipped feature, and its `step_peak` EPLB policy is being ported
  into ROCm/ATOM ([PR #2673](https://github.com/ROCm/ATOM/pull/2673)). If "deprecated" refers to anything, it may
  be a confusion with DeepSeek's older per-model serving repos or with SGLang's KTransformers integration —
  no evidence supports it for DeepInfer.
- **(b) NVIDIA FLARE scope — CONFIRMED as the federated-learning runtime.** ([repo](https://github.com/NVIDIA/NVFlare),
  [README](https://raw.githubusercontent.com/NVIDIA/NVFlare/main/README.md)). "Using" it in a serving-recipe
  generator would only make sense as *fleet recipe governance* (roll out recipe R, collect per-host gate results,
  enforce authorization). Its core machinery (FedAvg-family aggregation, differential privacy, homomorphic
  encryption, cross-site training loops) is training-shaped and would be entirely unused — REJ/CLOSED.

## Fold-in map — engine per placement shape and generator phase

| Placement shape | Primary engine | Why | Generator phase |
|---|---|---|---|
| CPU-hybrid MoE (VRAM+DRAM+NVMe) | SGLang | Paged Experts auto-K, `--cpu-offload-gb`, HiCache KV tiers | P3 build (emit flags), P4 qualify |
| CPU-hybrid MoE (secondary) | vLLM | `--cpu-offload-gb`/`--cpu-offload-params`/UVA, Engram offload, KV tiering | P3 build (emit flags) |
| GPU-resident multi-GPU | vLLM | TP/EP/TEP/DEP/PD + recipes schema for publication | P0 fit math (VRAM formula), P3 build, publish |
| Multi-GPU EP fleet (V4.x scale) | DeepInfer | Production EP+EPLB, step_peak policy, batch-invariance gates | P4 reference + gate vocabulary (watch) |
| Quant-shrunk single/paired machine | TensorFold | EXL3/TR3/EXLR8/MLX/NVFP4, pinned pulls, memory floor | P0 fit (floor), P3 build (serve flags), receipts |
| Fleet governance | (none of these) | FLARE REJ/CLOSED; use plain orchestration if ever needed | out of scope for now |

Cross-cutting borrows already recorded in `recipe-generator/research/landscape.md`: vLLM recipes schema/provenance
+ VRAM formula (P0), dynamo find-serving-recipe gates (P0 scan), llama.cpp offload semantics + LRU-over-static
finding (hot-expert maps), FlexGen/EPLB/PrefixPlace math (tiering engine). This pass adds: DeepInfer's
batch-invariance/determinism gates (P4 qualification), SGLang's budget→K sizing as our calibration point, and
TensorFold's revision-pinned pulls + receipt culture (evidence discipline).

## Sources

[DeepInfer repo](https://github.com/deepseek-ai/DeepInfer) · [DeepInfer README](https://raw.githubusercontent.com/deepseek-ai/DeepInfer/main/README.md) ·
[ATOM PR #2673 step_peak](https://github.com/ROCm/ATOM/pull/2673) · [DeepInfer release coverage](https://gigazine.net/gsc_news/en/20261021-deepseek-deepinfer-recipe) ·
[NVFlare repo](https://github.com/NVIDIA/NVFlare) · [NVFlare README](https://raw.githubusercontent.com/NVIDIA/NVFlare/main/README.md) ·
[SGLang repo](https://github.com/sgl-project/sglang) · [Paged Experts PR #42557](https://github.com/sgl-project/sglang/pull/42557) ·
[waterfill/LPLB blog](https://www.lmsys.org/blog/2026-06-26-waterfill-lplb) · [HiCache cookbook PR #46017](https://github.com/sgl-project/sglang/pull/46017) ·
[TensorFold repo](https://github.com/ashhart/TensorFold) · [TensorFold README](https://raw.githubusercontent.com/ashhart/TensorFold/blob/main/README.md) ·
[TensorFold whole-model gate #827](https://github.com/ashhart/TensorFold/pull/827) ·
[vLLM repo](https://github.com/vllm-project/vllm) · [vLLM recipes SCHEMA.md](https://github.com/vllm-project/recipes/blob/main/SCHEMA.md) ·
[DeepSeek-V4.1-Flash YAML](https://raw.githubusercontent.com/vllm-project/recipes/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) ·
[vLLM EPLB docs](https://docs.vllm.ai/en/latest/api/vllm/distributed/eplb/policy/default/)

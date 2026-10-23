# ROADMAP.md — engine selection and the phased rollout (don't do everything at once)

The owner's rule for extending the generator: **each capability lands in its own phase, with an entry
criterion and an exit criterion; no phase starts before its entry criterion is met; no two engine adapters
are built in the same iteration.** This file is the phased plan for folding engine choice into the generator.
The engine research itself is [research/engines.md](research/engines.md); the machine-readable engine
profiles are [engines.json](engines.json); the placement-side rules live in
[PLACEMENT.md §8](PLACEMENT.md).

## 0. The principle (from this project's own history)

Every big lever in the reference recipe was landed one at a time, gated, revertible: D=4 deferral (+22 %),
BIOS FCLK (+3.1 %), 226-expert placement (+6.3 %), indexer tiling (~2× prefill), the hot-expert map
(+37 % C1), streamed prefill (+34–41 %), host pinning (kills a C4 bimodality). The generator inherits that
discipline: **engine support is a lever family, and levers land one per phase with a calibration anchor** —
never all five engines at once.

## 1. The phases

| Phase | What lands | Entry criterion | Exit criterion (verifiable) | Do not start before |
|---|---|---|---|---|
| **A — engine-fit classifier** (now, GPU-free) | `engines.json` (static profiles: placement model, quant formats, tiering knobs, when-it-wins, verdict) + the decision rules in PLACEMENT.md §8 + `placement_calc.py --engine` selecting engine defaults (chunk-workspace model, deferral availability, hot-map runtime support, KV offload model) | engine research committed | every worked example names exactly one engine with a "why"; the classifier never recommends an engine the ledger REJ-ed without its reopen trigger | — |
| **B — llama.cpp adapter** | full emission support for llama.cpp recipes (flags `-ngl/-ot/--n-cpu-moe/-fa/-c`, Q4/Q5/Q6/Q8 size factors, published-reference calibration) | first new recipe that is small/consumer-grade OR the CSA2 blocker lifts (ggml-org/llama.cpp#28696) | an emitted llama.cpp recipe passes its own smoke + the plan reproduces a published anchor (already true for Qwen3-30B-A3B) | A |
| **C — vLLM adapter** | full emission support for vLLM recipes: `--cpu-offload-gb/--cpu-offload-params` weight offload, KV offload tiers (`TieringOffloadingSpec`), NVFP4 path, recipes.vllm.ai schema import/export | a GPU-resident or weight-offload recipe demand arises, or vLLM #30135 (SM120 MXFP4 Marlin fallback) lands | emitted vLLM recipe reproduces a vLLM measured anchor; the VRAM-fit formula matches vLLM recipes' own (schema-borrowed, research/landscape.md) | A, B |
| **D — sglang-hybrid deep adapter** | the full-rigor adapter: typed registry surface, 61-patch overlay inventory, gate checklist, el8-style map emission (largely built — this phase finishes it: expert-map file emission + guard mirroring) | a CPU-hybrid MoE recipe demand (the reference recipe's own shape) | emitted recipe resolves through the parent registry without refusal; plan reproduces evidence/t352 | A |
| **E — TensorFold adapter** | EXL3 quant-recipe emission (per-layer bit allocation, measurement-driven; pin-and-floor + receipt conventions borrowed now) | TensorFold serves DeepSeek (gate #827) or a published SM120 EXL3 MoE kernel with a KL number appears (the ledger's reopen trigger for sub-4-bit) | emitted EXL3 recipe reproduces a TensorFold measured anchor; NLL gate passes on the quantized weights | A; gated on the kernel, not on us |
| **F — DeepInfer / EPLB tier** | multi-GPU EP placement plans (EPLB/step_peak vocabulary, replication + packing from activation stats; batch-invariance + deterministic-sampling gates borrowed into the qualification suite now) | the owner's 3rd/4th-GPU hardware decision (GitHub #5), or DeepInfer ships a server/KV/disaggregation layer (today it is offline-only, V4.1-Flash only, SM100/Ascend 950 - a reference tier, not a runnable target on this SM120 box) | plan reproduces a measured multi-GPU anchor; TP/EP divisibility checks pass (attention heads, experts) | A; hardware-gated |
| **FLARE — deferred by verdict** | anything FLARE-shaped: fleet-level recipe governance (aggregating gate results across hosts, federated ledger) | a fleet exists (≥2 recipe hosts) and the research verdict says FLARE fits that shape | — | everything above; likely never for single-box serving; even for a fleet, plain orchestration is evaluated first (research/engines.md: the aggregation/privacy machinery is dead weight) |

Order is demand-driven after A: the phases are ranked by *expected next real recipe*, not by ambition.
Phases B–F are adapters; **each is one iteration, one engine, one calibration anchor** — the anti-pattern
being avoided is a single "engine abstraction layer" written against five engines nobody has served yet
(poteto: trust before parallelism; and the ledger: build the lever, measure it, then generalize).

## 2. What each phase costs (so the owner can cut)

- **A** is GPU-free and nearly free: static profiles + a flag. It changes the plan output (engine line) but
  not the math.
- **B** is small: llama.cpp flags are known; published anchors exist; the CSA2 blocker is the only reason
  B is not the reference engine today.
- **C** is medium: vLLM's knobs exist and its recipes schema is the best flag reference; the work is
  mapping our lever vocabulary onto its `--cpu-offload-*`/KV-tiering flags and importing its VRAM formula.
- **D** is large but mostly done: the registry surface and patch inventory are already emitted; finishing
  means mirroring guards and emitting expert-map files.
- **E** and **F** are hardware-gated: no work starts until the kernel (E) or the cards (F) exist.

## 3. How engine choice folds into the existing pipeline

```
budget JSON ──► placement_calc.py --engine <name>   (A: engine defaults feed the plan)
hardware JSON ┘
        │  plan now carries: engine, why-this-engine, engine lever surface
        ▼
emit_recipe.py --engine <name>   (adapter phases B..F: engine-specific launcher/profiles/notes)
```

The placement math is engine-agnostic (bytes, bandwidths, rooflines); engines differ in *which levers
exist* (deferral: sglang-hybrid only; LRU expert cache: llama.cpp/vLLM; KV offload: vLLM; EP: DeepInfer/
SGLang) and *how a recipe expresses them*. The classifier maps plan-shape → engine; the adapters map
lever-settings → engine flags. One rule from the ledger stays supreme: **an engine the ledger REJ-ed
(TensorFold as engine, E30/T3.32) can only be recommended by the classifier when its reopen trigger is
cited in the same breath.**

## 4. Status

- **Phase A: landed this iteration.** `research/engines.md` (the five-engine research, cited), `engines.json`
  (six profiles with verdicts + reopen triggers + capability flags), PLACEMENT.md §8 (fold-in map + rules),
  and the `placement_calc.py --engine` flag (engine defaults feed the plan; REJ-ed engines warn with their
  reopen trigger; engines without deferral force D=0). Self-tested: the reference example still reproduces
  its anchor with `--engine sglang-hybrid`.
- Phases B–F: not started; entry criteria above.
- FLARE: deferred by verdict (see research/engines.md).

# Recipe Factory — a reusable system for building local-model serving recipes

**What it is:** the methodology, tooling templates, knob taxonomy, tried-options ledger and UI plan extracted from the
end-to-end journey of this repository — taking DeepSeek-V4.1-Flash (a 475 GiB MoE checkpoint) from a spreadsheet
verdict ("does not fit two RTX PRO 6000") to a qualified, revision-pinned, gate-passing three-tier serving recipe
(VRAM + host DRAM/CPU + NVMe, C1 58→60.8 tok/s, C4 118.9 wall, 262K×4 default, 1M×1 ladder, 6/6 needles).

**What it is for:** spinning up *"recipe X for a new model on hardware Y"* with the same rigor — any open-source
model, any hardware (single card, multi-GPU, CPU-hybrid, Apple Silicon), with per-use-case profiles, KV settings,
TTFT/tok-s trade-offs, and a record of what was tried and rejected so a new recipe never re-measures a dead end.

This folder is **documentation and templates only**. It touches nothing the serving recipe reads: no script under
`scripts/`, no config, no evidence manifest, no CI check. The recipe keeps serving and CI stays green.

---

## 1. Does something like this already exist? (researched 2026-10-23)

We looked for a system that packages the whole loop — feasibility math, host microbenchmarks, baseline, build,
qualification gates, profiles, ledger, UI. **No single system covers it.** The landscape, with what each piece
covers and lacks:

| System | What it is | Covers | Lacks (why it isn't enough) |
|---|---|---|---|
| [vllm-project/recipes](https://github.com/vllm-project/recipes) → [recipes.vllm.ai](https://recipes.vllm.ai) | Machine-readable YAML recipe schema (`models/<org>/<repo>.yaml`) rendered to a site + JSON API; per-hardware overrides, serving strategies (TP/EP/DEP/PD), a VRAM-fit formula, provenance to the owning repo+commit, auto-synced pinned sources | "How do I run model X on hardware Y for task Z?" for **vLLM only**; the DeepSeek-V4.1-Flash recipe is excellent flag reference (Engram host tables, DSpark per-hardware configs) | No measurement methodology: no evidence directories, no qualification gates, no NLL/needle/stress discipline, no tried-options ledger, no profiles-with-measured-trade-offs, no host-microbenchmark phase, no CPU-hybrid tiering, no UI beyond a command builder |
| [sgl-project/sgl-cookbook](https://github.com/sgl-project/sgl-cookbook) → [docs.sglang.io/cookbook](https://docs.sglang.io/cookbook/intro) | Community markdown guides per model (DeepSeek-V4_1, GLM-5.3, Qwen3.8-27B NVFP4 recipes…) | Model-specific flag guidance for SGLang | Prose recipes; no schema, no reproducibility pins, no gates, no evidence discipline, nothing reusable across models |
| [NVIDIA LLM Inference Quick Start Recipes](https://docs.nvidia.com/llm-inference-quick-start-recipes/overview) + [ai-dynamo/dynamo recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) | Deployment guides and disaggregated (Dynamo + SGLang/TRT-LLM/vLLM) recipes for NVIDIA hardware | Production multi-node deployments on NVIDIA data-center hardware | Enterprise infra oriented (PD split, routers, NIXL); no local-workstation methodology, no feasibility-to-gate pipeline, no evidence/ledger discipline |
| [0xSero Local AI Registry](https://local-ai-registry.vercel.app/hardware/rtx-pro-6000-blackwell-96gb) (docs: [0xsero.github.io/local-ai-registry](https://0xsero.github.io/local-ai-registry/)) | Normalized catalog: 158 RTX PRO 6000 recipes, hardware records (specs + measured), market prices, provenance, JSON API | **The best catalog model** — hardware × model × recipe links with provenance, an API the UI could consume | A catalog, not a method: recipes are one-command launches, no qualification gates, no evidence, no profiles, no methodology for *new* recipes |
| [samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell) | A single revision-pinned, safety-hardened recipe (pins, fail-closed preflight, verify, million-token needle tests, canary, systemd, `ci.sh`, provenance) | **The discipline model this repo was modeled on** — the gold standard for *one* recipe | One model, one hardware: no generator, no template, no ledger, no profile system, no UI |
| [ai-dynamo/dynamo PR #14211 — "find-serving-recipe" agent skill](https://github.com/ai-dynamo/dynamo/pull/14211) | An agent skill that hunts serving recipes with provenance gates | Closest existing "recipe generator" concept: agentic discovery + provenance | Finds recipes; does not build, measure, or qualify them. No methodology, no ledger |
| Community single-recipe repos ([0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero), [0xSero/glm-5.2-sm120](https://github.com/0xSero/glm-5.2-sm120), [1austinanderson/aa-glm53-flash-rtx-pro-6000](https://github.com/1austinanderson/aa-glm53-flash-rtx-pro-6000), [J-M-Recipes/recipes](https://github.com/J-M-Recipes/recipes), TensorFold recipe repos) | One-command launch recipes with receipts | Same as above: results without the method that produced them |
| HuggingFace blueprints | [huggingface/ai-blueprint](https://github.com/huggingface/ai-blueprint) is RAG/extraction/fine-tuning; no serving-recipe org exists | — | Not serving |
| SkyPilot / Modal / Replicate | Cloud deployment examples (SkyServe YAML, modal-examples `vllm_inference.py`) | Cloud orchestration | Nothing about local hardware feasibility, microbenchmarks, or qualification |

**Verdict: build ours, borrow theirs where they are strong.** No existing system packages the end-to-end loop this
repository learned. What exists and is worth adopting rather than reinventing:

1. **vLLM recipes' schema/provenance conventions** — a new recipe should also emit a machine-readable descriptor
   (model id, hardware rows, args, provenance to repo+commit). See [TEMPLATE/README.md](TEMPLATE/README.md).
2. **Local AI Registry's catalog model + JSON API** — the natural publication target for recipe metadata; the UI
   plan in [UI/macOS-UI-PLAN.md](UI/macOS-UI-PLAN.md) reads both registry-style metadata and our evidence JSONs.
3. **samuelcardillo's discipline** — pins, fail-closed preflight, verify, canary, `ci.sh`. This repository already
   generalized it; the generalization lives in [METHODOLOGY.md](METHODOLOGY.md).
4. **The dynamo find-serving-recipe skill** — later, as an agent front-end over the ledger + registry.

The differentiators of this system (what ours adds that none of the above has):

- **A numbered end-to-end methodology with exit criteria per phase** (P0 research → P1 host microbenchmarks →
  P2 baseline → P3 build → P4 scale), not a command dump.
- **Feasibility math before any download** — the checkpoint-budget → placement-calculator pattern
  (`checkpoint-budget.py` reads only safetensors headers; `feasibility-calc.py` compares layouts × GPU counts
  against measured bandwidths, calibrated on a known deployment, and exits nonzero on "does not fit").
- **Host microbenchmarks that decide specific engineering questions** (DRAM read → decode ceiling; NVMe random
  row gather → Engram-tiering decision; PCIe additivity → stream-vs-compute split; GPU GEMM → roofline).
- **A gate discipline** (NLL identity, verify/needles, stress, batch consistency, health, power, spec-consistency)
  that makes "qualified" a mechanical claim, plus a **tried-options ledger** so rejected paths are recorded with
  numbers, not re-measured.
- **A typed knob registry** (every knob declared once with type/default/sink/evidence; unknown, mistyped, removed
  and unsafe values refused before launch) and **profiles** layered under the environment.
- **Per-use-case profile presets with measured trade-offs** (coding-agent, multi-user, long-context ladder,
  power-capped, quality-first) — see [KNOBS-AND-PROFILES.md](KNOBS-AND-PROFILES.md).
- **An evidence discipline** (per-serve manifests, stamped JSON, generated headline tables) that a macOS UI can
  read directly — see [UI/](UI/).

---

## 2. The methodology distilled: five phases

Every phase has one exit criterion. A phase is not "done" until its evidence is committed and its exit criterion
is verifiable from the repo alone. Full playbook: [METHODOLOGY.md](METHODOLOGY.md).

### P0 — Research and data (no GPU time)

Read the checkpoint's *structure* without downloading weights: safetensors header budget per component (experts,
Engram/embedding tables, draft heads, attention, embeddings), the hardware inventory, and every existing recipe
for the model family (the landscape above, plus the source-document audit pattern of `docs/FEASIBILITY.md §6`:
every claim audited against evidence). Run the placement calculator across layouts × GPU counts with measured
bandwidths and a **calibration point** — a known deployment whose numbers the calculator must reproduce before
its new rows are trusted. **Exit:** a placement table that says, per layout and GPU count, *fits / does not fit,
with margins*, and a ranked list of paths.

### P1 — Host microbenchmarks (no model weights, minimal GPU contention)

Replace every vendor spec with a measurement and attach each number to the decision it feeds: DRAM read
bandwidth (thread sweep; CCD/fabric limits) → CPU-offload decode ceiling; GDDR7 copy → GPU roofline; NVMe random
row gather → whether lookup tables can live on disk; PCIe H2D + CPU/DMA additivity → stream-vs-compute split and
prefill ceiling; CPU GEMV throughput → compute-bound vs bandwidth-bound. **Exit:** zero "estimate" rows remain
in the hardware doc; the calculator is recomputed on measured numbers.

### P2 — Baseline

Re-measure the already-qualified incumbent deployment with *this* recipe's harness (same fixtures, same power
condition) so every later number is comparable. **Exit:** baseline tables produced by this harness, not copied
from another repo.

### P3 — Build (the push track)

Order the build routes by expected payoff (port into a hybrid engine → alternate engine → own prototype), and
run the full gate suite at every milestone: NLL identity against a same-model baseline, verify/needles, stress,
batch consistency, health, power. Every negative result lands in the ledger with numbers. **Exit:** a
decode/prefill/needle table with a pinned engine build — or a documented reason no route ran.

### P4 — Scale and compare

The 3/4-GPU or upgraded-CPU columns, re-qualified on the same harness. **Exit:** one table with all hardware
rows for the same fixture and power condition.

This project's own phase map is [plan/README.md](../plan/README.md); its P0–P4 artifacts are the reference
implementations (`scripts/checkpoint-budget.py`, `scripts/feasibility-calc.py`, the `scripts/bench-*` suite,
`docs/FEASIBILITY.md`, `docs/HARDWARE.md`, `docs/BENCHMARKS.md`).

---

## 3. What is in this folder

| File | Purpose |
|---|---|
| [README.md](README.md) | This file: vision, landscape, phases, repo-split plan |
| [METHODOLOGY.md](METHODOLOGY.md) | The reusable playbook: feasibility math, microbenchmark suite and what each number decides, engine/kernel research checklist per GPU arch, gate discipline, evidence discipline, benchmark-harness spec |
| [KNOBS-AND-PROFILES.md](KNOBS-AND-PROFILES.md) | Knob taxonomy generalized from `scripts/serve_config.py`, with the **measured** trade-off numbers from this project per knob, plus ready-made profile presets (coding-agent, multi-user, long-context ladder, power-capped, quality-first) |
| [TRIED-OPTIONS-LEDGER.md](TRIED-OPTIONS-LEDGER.md) | Every option tried in this project with outcome/verdict/why, formatted so a new recipe consults it instead of re-measuring |
| [TEMPLATE/](TEMPLATE/) | The new-recipe starter: skeleton dirs, typed knob registry template, serve launcher template, benchmark harness template, profiles template, CI template |
| [UI/](UI/) | macOS UI plan ([macOS-UI-PLAN.md](UI/macOS-UI-PLAN.md)), the JSON schema the UI consumes, and a minimal SwiftUI skeleton |

The measured numbers quoted throughout are this project's own evidence (see
[`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) and the `evidence/` tree). They are **host- and model-specific
calibration values**, never universal claims — the methodology's first rule is that every number is re-measured
per host.

---

## 4. Repo-split plan

Today everything lives in one repository (this one). The split below is deliberately late: the factory knowledge
is only trustworthy because it was extracted from a *working* recipe in the same tree, and the recipe keeps
serving while the factory evolves. Proposed split, when the owner is ready (see "decisions needed" in §5):

**Stays in this repo (the recipe, `deepseek-v41-mxfp4-recipe-rtx6000pro`):**
- Everything the serve reads: `scripts/`, `config/`, `kernels/`, `patches/`, `systemd/`, `examples/`
- Everything the recipe's evidence depends on: `evidence/`, `docs/`, `plan/`, provenance/license files
- `recipe-factory/` itself for now (co-located so the recipe's artifacts stay the worked examples the docs cite)

**Moves to a new repo `hfyeomans/recipe-factory` (or similar) when it earns it:**
- `recipe-factory/TEMPLATE/` → the new-recipe starter (own CI, own tests, no model weights)
- `recipe-factory/UI/` → the SwiftUI app (its own repo is *required* for notarization and releases anyway)
- `recipe-factory/METHODOLOGY.md`, `KNOBS-AND-PROFILES.md`, `TRIED-OPTIONS-LEDGER.md` → the factory docs
- Generalized, model-agnostic versions of the harness scripts (today they live in `scripts/` and carry
  V4.1-specific fixtures; the split extracts the generic core and keeps the fixtures here)

**Shared between the two (the contract):**
- The evidence JSON schema (manifest fields, stamped-JSON rules) — `UI/JSON-SCHEMA.md` is the draft
- CI conventions: identifier scan, executable-bit rules, generated-doc staleness checks
- The knob-registry conventions (typed knobs, sinks, evidence citations, profile layering)

**Upstreamed, not kept anywhere private:** the six candidate PRs in
[`docs/research/OPEN-SOURCE-PLAN.md`](../docs/research/OPEN-SOURCE-PLAN.md) (SM120 W8A8 configs, block-32 tuner,
FP8 scratch fix, page-split graph lifetime, kt-kernel buffer lifetime, indexer row-tiling) and the exact-top-k
fix (sgl-project/sglang #35257). The factory's long-term bet is that engine-specific patches shrink as upstream
lands; the methodology, ledger and UI do not.

---

## 5. Decisions needed from the owner

See the report to the Lead; the five are: (1) repo-split timing — now, at first new recipe, or at UI v1;
(2) UI scope — plan-only vs building the SwiftUI dashboard v1 (which needs the evidence-JSON puller + ssh
transport agreed first); (3) template depth — model-agnostic engine-agnostic core only, vs. including the
SGLang-hybrid specifics as a "first recipe" worked example; (4) whether the factory's canonical home is this
repo or a new public repo (the ledger and methodology are publishable; the 1017 checkpoint pins are not);
(5) how much of the 61-patch overlay gets upstreamed before new recipes are started, since that decides how
portable the launcher template really is.

## 6. Rules inherited from the parent repo (non-negotiable in any new recipe)

- Never ship a recipe that has not run: preflight fails closed until the engine is verified on this host.
- No host identifiers in committed evidence: no paths under user home dirs, no private IPs, no GPU UUIDs,
  no PCI IDs, no serials. `scripts/ci.sh` enforces the scan.
- Every knob in an owned name family is declared in the typed registry or the launch is refused.
- A removed knob is refused with the experiment that removed it. A mistyped knob cannot launch the default.
- Every performance claim in a generated table is read from manifest-stamped JSON; a stale table fails CI.

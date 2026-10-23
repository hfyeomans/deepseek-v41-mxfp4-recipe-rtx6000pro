# Recipe Generator — mise en place for models

**What this is:** the placement engine, tuning model, and **recipe emitter** that turn the
`recipe-factory/` knowledge base into a working product: given any open-source model (a budget JSON from
safetensors headers) and any target hardware (a JSON of capacities plus *measured* bandwidths), it decides
**where the parts go** — GPU VRAM / host DRAM+CPU / NVMe — picks the levers that realize it, projects the
serving numbers (C1/C4 decode, prefill, TTFT) with an enforced trust label per projection, and **emits a
recipe tree that can actually be run** (launcher, profiles, preflight, verify, provenance, deep-lever and
patch notes), optionally publishing it to a GitHub repo. The app **forms and witnesses; the host executes** —
the UI is discerning about that boundary and never pretends ([RECIPE-EMISSION.md §4](RECIPE-EMISSION.md)).

**What this is not:** a replacement for `recipe-factory/`. That folder holds the methodology, the ledger, the
knob taxonomy and the UI plan — the *theory*. This folder is the *engine* that executes the P0 feasibility
math, the tuning-mode front-end for P3, and the P3-to-recipe emission. It reads the same evidence tree, cites
the same experiment ids, and does not duplicate recipe-factory's documents. **It touches nothing the serving
recipe reads** — no `scripts/`, `config/`, `evidence/`, `docs/`, `plan/` files are modified.

The metaphor (borrowed from [poteto's Michelin kitchen](research/poteto.md)): the ingredients (checkpoint
components) are placed and prepped — quantized, tiered, cached — so service is fast. The chef builds the
kitchen once (budget JSON + hardware JSON + gate suite); every dish after that is cheap.

## The pipeline

```
checkpoint (headers only) ──► budget JSON ──┐
                                            ├──► placement_calc.py --engine <name> ──► tier plan + levers + projections
host measurements (P1) ─────► hardware JSON ┘          │        (engines.json feeds the plan)
                                                       ├─ trust: calibrated | calibrated-model | provisional
                                                       ├─ exits 0 (plan) / 3 (does not fit)  ← CI-pinnable
                                                       └──► emit_recipe.py --engine <name> ──► runnable recipe tree
                                                                    │      (launcher, profiles, preflight,
                                                                    │       verify, provenance, patch notes)
                                                                    └─ --repo-url → optional GitHub publication
```

## File map

| File | What it is |
|---|---|
| [README.md](README.md) | This file: vision, pipeline, decisions, status |
| [PLACEMENT.md](PLACEMENT.md) | The planner spec: input schemas, tiering rules, every formula, the constant register, **the calibration rule** |
| [placement_calc.py](placement_calc.py) | The working minimal calculator (stdlib only, GPU-free): budget JSON + hardware JSON → tier plan + projections |
| [emit_recipe.py](emit_recipe.py) | The emitter: plan → **runnable recipe tree** (README/resolved line/launcher/preflight/verify/provenance/patch notes), optional GitHub repo stamping |
| [RECIPE-EMISSION.md](RECIPE-EMISSION.md) | Emission spec: what a generated recipe contains, the deep-lever surface, the advanced patch inventory, GitHub publication, **the UI discernment boundary** |
| [ROADMAP.md](ROADMAP.md) | The phased rollout: engine support lands one adapter per phase, each with entry/exit criteria — no doing everything at once |
| [PROJECT-PLAN.md](PROJECT-PLAN.md) | The end-to-end plan: phases 0–4, milestones M1–M5, the new-model trial queue (T1–T4), owner decisions D1–D6 |
| [engines.json](engines.json) | Engine profiles (sglang-hybrid / llama-cpp / vLLM / TensorFold / DeepInfer / FLARE): placement model, lever availability, verdict + reopen trigger |
| [registry-surface.json](registry-surface.json) | The parent recipe's complete 70-knob typed registry, machine-readable for the UI's knob panel |
| [quantization.json](quantization.json) | Quantization options with size + quality trade-offs (measured + published reference factors, each with a declared baseline dtype) |
| [LEVERS.md](LEVERS.md) + [levers.json](levers.json) | The lever dictionary: what each lever moves, its cost, when it helps, measured numbers, generator mapping |
| [PROFILES.md](PROFILES.md) + [profiles/](profiles/) | Per-use-case presets (coding-agent / multi-user / long-context / power-capped / quality-first) as lever deltas |
| [EXAMPLES.md](EXAMPLES.md) + [examples/](examples/) | Three worked examples: real DSV4.1 (475 GiB), synthetic 211 GiB, Qwen3-30B-A3B small model |
| [emitted/](emitted/) | Two sample emitted recipes (Qwen3 on RTX 3090 via llama.cpp — runnable anywhere; synthetic 211 GiB via sglang-hybrid — runnable shape) |
| [UI-THEME.md](UI-THEME.md) | The cyberpunk × AI-tropes × Frankenstein UI theme (green reserved for "qualified") |
| [research/poteto.md](research/poteto.md) | The poteto method distilled (identity correction included) and mapped onto the generator |
| [research/landscape.md](research/landscape.md) | The landscape check: what exists, what to borrow vs build |
| [research/engines.md](research/engines.md) | The five-engine research (FLARE / DeepInfer / SGLang / TensorFold / vLLM): what each is, its placement model, verdict, fold-in map |
| [tests/test-placement-calc.py](tests/test-placement-calc.py) | Self-tests: the calculator must reproduce the repo's measured qualified deployment; the emitter must produce a clean runnable tree |

## Quickstart (GPU-free)

```bash
# The three worked examples + self-tests:
bash recipe-generator/examples/run-examples.sh

# One plan, interactively:
python3 recipe-generator/placement_calc.py \
  recipe-generator/examples/dsv41-flash/budget.json \
  --hardware recipe-generator/examples/dsv41-flash/hardware.json

# Tuning mode: turn the levers and watch the projections move:
python3 recipe-generator/placement_calc.py ... --chunk-size 4096 --deferral 0 --map checkpoint_order
python3 recipe-generator/placement_calc.py ... --json   # machine-readable plan

# Form a recipe that can be run (and stamp an optional GitHub repo):
python3 recipe-generator/emit_recipe.py \
  --budget recipe-generator/examples/qwen3-30b-a3b/budget.json \
  --hardware recipe-generator/examples/qwen3-30b-a3b/hardware-rtx3090.json \
  --quant gguf-q4_k_m --context-tokens 4096 --sequences 1 \
  --engine llama-cpp --out recipes/qwen3-rtx3090 \
  --model Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M \
  --repo-url https://github.com/<owner>/<repo> --init-git
cd recipes/qwen3-rtx3090 && bash preflight.sh && bash launch.sh && bash verify.sh
```

For a new model: produce the budget with the parent repo's `scripts/checkpoint-budget.py` (headers only),
add the `arch` section per [PLACEMENT.md §2.1](PLACEMENT.md), measure the bandwidths per
`recipe-factory/METHODOLOGY.md` §4, and add a `calibration` section once the first serve produces numbers —
that is the moment the plan's projections become `calibrated`.

## The calibration rule (the generator's prime directive)

**Every projection must be validated against at least one measured datapoint before it is trusted.** The
calculator enforces it mechanically: no `calibration` in the hardware JSON → every projection is labeled
`provisional` and the plan header says UNVALIDATED. The self-test pins the calculator to the repo's own
measured deployment (219/165 experts, C1 60.8, C4 124.0, prefill 4,344 — `evidence/t352`), so the tool itself
never ships unvalidated either. Full mechanics: [PLACEMENT.md §5](PLACEMENT.md).

## The five decisions the owner must make

1. **Repo home.** Keep the generator in this repo (co-located with the evidence it calibrates against — the
   recommendation until it has earned a split), move it to a new `recipe-factory` repo, or move only
   `research/` + docs. The recipe-factory/README repo-split plan already anticipates this; the generator's
   worked examples reference `evidence/` by relative path, which is the main coupling to preserve or replace.
2. **UI integration and the emit flow.** Whether the UI owns the full emit flow (plan → tune → **emit a
   runnable recipe** → ask for an optional GitHub repo → scaffold/push with consent), or emits through the
   CLI and only witnesses. Either way the contract is `MANIFEST.json` + `registry-surface.json` + the
   UI-THEME trust switch; the discernment boundary (RECIPE-EMISSION.md §4: the app never serves, never
   downloads, never builds images) is fixed.
3. **Real-engine invocation.** Whether the planner should later *invoke* engines (auto-run the calibration
   serve/microbench to close the loop like `gpu-container` does) or stay pure planning that humans/serves
   feed. GPU-free is the constraint for now; the calibration schema is already the interface for a future
   auto-calibrator.
4. **Runtime placement as a first-class tier.** The static hot-expert map has a measured 10–15 % CPU-share
   floor; runtime placement (SGLang Paged Experts PR #42557, llama.cpp #27861 LRU, vLLM RFC #38256) is the
   only path below it. Decide whether the generator models `runtime_oracle` placements as plan output now or
   stays advisory.
5. **CI wiring.** Whether to wire `recipe-generator/tests/` into the parent `scripts/ci.sh` (a one-line
   addition to a serving script — needs the owner's OK under the do-not-modify rule), or keep the generator's
   CI self-contained behind `run-examples.sh`.

## Status (2026-10-23)

Built and GPU-free-tested this session: the calculator reproduces the repo's measured qualified deployment
from JSON inputs alone (the calibration anchor), the synthetic 211 GiB case shows 100 % of experts in VRAM +
Engram in DRAM with C1 ≈ 73.8 (vs 60.8 today), the small-model case reproduces published reference numbers,
and the emitter produces complete runnable recipe trees (two samples under `emitted/`) including the full
70-knob registry surface and the 61-patch overlay inventory with portability verdicts. Not yet: a real
new-model run (needs a first serve for its anchor), the UI panel, engine invocation (decision 3). Research is
in `research/` — read [poteto.md](research/poteto.md) for what the prep-first method adds and
[landscape.md](research/landscape.md) for the borrow-vs-build verdict.

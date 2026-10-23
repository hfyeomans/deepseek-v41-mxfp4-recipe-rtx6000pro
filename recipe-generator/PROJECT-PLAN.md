# PROJECT-PLAN.md — the recipe generator, end to end

What has to happen from today to "the app forms a recipe that can be run, and we feed other models through it."
Phases have entry and exit criteria (the house rule: no phase starts before its entry criterion; no two big
things land at once). Milestones M1–M5 gate the phases. The engine-adapter sub-phases live in
[ROADMAP.md](ROADMAP.md) and are referenced, not duplicated.

**State today (M1 done):** `recipe-factory/` (methodology, ledger, knob taxonomy, UI plan) + `recipe-generator/`
(placement engine with the calibration rule, emitter of runnable recipes, lever dictionary, 70-knob registry
surface, 61-patch inventory, engine classifier, five profiles, UI theme, poteto + landscape + engine research)
+ MobileBuildMCP tooling. Self-tests reproduce the measured qualified deployment (219/165, C1 60.8, C4 124.0);
quorra CI green at every commit; serving files untouched.

---

## Phase 0 — Validate the generator against a second real checkpoint (GPU-free, can start now)

**Status: DONE (2026-10-23).** All three tasks complete and pinned as self-test #9: the 1017 budget
re-derived (byte-identical components — weights changed, shapes did not); the plan reproduces
`evidence/t1017` (219/165, D=4, chunk 8192, C1 61.9, C4 123.6, prefill 3,095 cold, TTFT 16K 5.4 s vs
measured 5.18 s); and the emitted recipe matches the serving recipe's `qualified.env` knob-for-knob with
zero conflicts (including `KT_EXPERT_MAP`, via the new `--expert-map` flag). The generator is now
validated against **two** real checkpoints.

The calibration rule says every projection needs a measured anchor. We have exactly one anchor today
(evidence/t352). Before feeding new models, prove the generator on a *different* checkpoint whose numbers the
repo already measured: **DeepSeek-V4.1-Flash-1017** (the currently-served weights).

Tasks:
1. `scripts/checkpoint-budget.py` on the 1017 snapshot (headers only) → `budget-1017.json`.
2. Run `placement_calc.py` on it + the reference hardware JSON with the `evidence/t1017` calibration
   (C1 61.9/55.9, C4 123.6/114.0, prefill 4,504/4,613, 14.4 % CPU share).
3. **Exit:** the plan reproduces the 1017 deployment (same 219/165 split, same levers, same trust label),
   and `emit_recipe.py` output for 1017 matches the serving recipe's own knobs. A fail here is a generator
   bug — fix before anything else.

## Phase 1 — First generated serve on a NEW model (small serve window)

Trial order (each is one iteration, one anchor):

| # | Model | Hardware shape | Why first/next | Adapter |
|---|---|---|---|---|
| T1 | Qwen3-30B-A3B (Q4_K_M) | 1× RTX PRO 6000 | tiny, fast serve, published anchors seed the plan; first *generator-made* recipe | llama.cpp (ROADMAP B) |
| T2 | GLM-5.3-Flash | 2× RTX PRO 6000 | the repo already measured it (e14-glm53, t29-glm-325w, e16-glm-32slots) — anchors exist | engine classifier picks (SGLang/vLLM/EXL3 path) |
| T3 | the real ~211 GiB model (when the owner names it) | 2× 96 GiB | validates the VRAM-rich placement + Engram-in-DRAM plan (the synthetic example is its prior) | sglang-hybrid |
| T4 | DeepSeek-V4.1-Flash-NVFP4 (nvidia quant variant) | 2× 96 GiB | tests the quantization layer against a real alternative format | vLLM (ROADMAP C) |

Per trial: budget JSON (GPU-free now) → plan → emit recipe tree → owner runs
`preflight.sh && launch.sh && verify.sh` → gate suite (NLL identity, needles, stress, batch consistency,
health, power) → the measured numbers become that model's `calibration` anchor → **trust label promoted to
`calibrated`**. Anything that fails lands in the tried-options ledger with numbers.

**Exit (milestone M2):** one emitted recipe served and qualified on a model that is not the reference one;
the repo gains its second calibration anchor. GPU time needed: ~1 serve window per trial — the owner
schedules them (the box is busy).

## Phase 2 — UI v1 (macOS app, MobileBuildMCP loop)

Entry: M2 + owner go on UI scope (decision D2 below). **Go given 2026-10-23; the screen-by-screen build
plan with acceptance criteria and the M3a/M3b/M3c milestones is [UI-V1-PLAN.md](UI-V1-PLAN.md). The app
builds in a git worktree (`ui-app` branch) so the session filesystem guard cannot roll it back.**

1. Scaffold the app target (`mobilebuildmcp project-scaffolding` or an Xcode project around
   `recipe-factory/UI/RecipeFactoryUI`).
2. Build the screens from the existing contracts: plan viewer (tier plan + trust banner), tuning panel
   (levers from `levers.json` + `registry-surface.json`), profile diff, emission flow (emit → optional
   GitHub repo ask → publish), evidence witness (stamped JSON renderer).
3. Theme per [UI-THEME.md](UI-THEME.md): cyberpunk × AI-tropes × Frankenstein, green reserved for
   "qualified"; the trust switch is "the creature opens its eyes".
4. Iterate with MobileBuildMCP: `simulator build-and-run` → `ui-automation screenshot` → `snapshot-ui`
   (elementRefs) → tap/type against the real app; screenshots shown to the owner each iteration.
5. Wire what the UI may NOT do as explicit hand-offs (RECIPE-EMISSION.md §4): serve, benchmark, gate,
   census, BIOS — the app forms and witnesses; the host executes.

**Exit (milestone M3):** the app builds, runs, and the full emit flow (plan → tune → emit → GitHub repo
ask → publish) works from the UI, demonstrated on T1/T2 recipes.

## Phase 3 — Three-model proof and hardening

Entry: M3.

1. Finish trials T2–T4 (each is a ROADMAP adapter phase with its own calibration anchor).
2. Generator hardening driven by the trials: expert-map file emission + guard mirroring (ROADMAP D),
   vLLM schema import/export (publish the recipes in the vLLM recipes shape), ledger entries for every
   failed experiment.
3. Wire `recipe-generator/tests/` into `scripts/ci.sh` if the owner approves (decision D5).

**Exit (milestone M4):** three models across at least two hardware shapes, each recipe emitted by the
generator, served, qualified, and anchored; the classifier never recommended a ledger-REJ-ed engine without
its reopen trigger.

## Phase 4 — Scale and publish

Entry: M4.

1. Publication: Local AI Registry records (catalog + `validated` flag) and/or vLLM-recipes-schema export.
2. Fleet/multi-GPU questions: 3rd/4th GPU (GitHub #5), DeepInfer watch (server/KV layer release →
   ROADMAP F), runtime placement as a first-class plan tier (T3.8d measured 1.8–2.2 % held-out CPU share —
   update the `runtime_oracle` calibration row when it ships).
3. Repo-home decision execution (D1): split `recipe-factory` + generator into their own repo when the owner
   calls it; upstream the six candidate PRs.

**Exit (milestone M5):** the generator is public-facing: documented, CI-wired, with published recipes and a
decision recorded for every open question.

---

## Milestones

| Milestone | What | Evidence |
|---|---|---|
| **M1 — the kitchen** (done) | factory knowledge + placement engine + emitter + engine classifier + Phase 0 | 9/9 self-tests; 4 worked examples (2 real checkpoints); emitted 1017 recipe matches the serving recipe; CI green |
| **M2 — first generated serve** | Phase 0 + T1 | a non-reference model served from an emitted recipe; second calibration anchor committed |
| **M3 — UI v1** | Phase 2 | app builds/runs; full emit+publish flow from the UI; screenshots reviewed |
| **M4 — three-model proof** | Phase 3 | 3 models × 2 hardware shapes, all qualified + anchored |
| **M5 — published** | Phase 4 | registry/schema exports; repo-home decision executed |

## Owner decisions (numbered for reuse)

- **D1. Repo home** — keep co-located vs split to a new `recipe-factory` repo, and when.
- **D2. UI go** — approve building UI v1 now (Phase 2) and who owns the JSON contract; MobileBuildMCP is
  installed and the theme is specced.
- **D3. Real-engine invocation** — later auto-calibration (the app invokes a serve/microbench) or keep the
  planner human-fed. GPU-free stays the default until then.
- **D4. Serve windows** — the GPU box is busy; trials T1–T4 each need a scheduled window. The owner picks
  the order and timing; everything before the serve is GPU-free and can be done now.
- **D5. CI wiring** — add `recipe-generator/tests/` to `scripts/ci.sh` (one line in a serving script, needs
  the owner's OK) or keep the generator's CI self-contained.
- **D6. Which 211 GiB model** — T3's real model, when it exists (the synthetic example is the placeholder
  prior).

## What can be done right now (GPU-free)

1. ~~Phase 0 in full~~ **done** (see above).
2. Budget JSONs for T1/T2/T4 via `scripts/checkpoint-budget.py` (headers only) — so every trial starts
   from an exact component budget, not an estimate.
3. ROADMAP phase B (llama.cpp adapter) — the small model's adapter, needed by T1.
4. UI scaffolding (Phase 2 tasks 1–2) and the theme tokens — everything short of running serves.

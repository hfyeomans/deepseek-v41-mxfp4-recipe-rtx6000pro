# RECIPE-EMISSION.md — from plan to a recipe that can be run

The product expectation (owner, 2026-10-23): **the app forms a recipe that can be run** — it may ask for an
optional GitHub repo to put it in — and it must **dig deep**: KV cache and every other lever, including the
most advanced and innovative patches this project built, wherever they apply. Some things are not possible
for the UI; the app must be **discerning** about that boundary instead of pretending.

This file is the spec for [`emit_recipe.py`](emit_recipe.py) and the contract between the generator, the
emitted recipe, and the UI. Sample emitted recipes live in [`emitted/`](emitted/).

## 1. What emission produces (the runnable recipe tree)

```
<recipe>/
  README.md            plan-derived: model/hardware/quant, tier plan table, resolved line,
                       projections + trust banner, run commands, gate checklist, publish section
  MANIFEST.json        plan + pins + registry surface (what the UI renders; no parallel formats)
  PROVENANCE.md        immutable pins: model snapshot, engine base/overlay/image, k-factors, anchor
  config/profiles/*.env  profile files from the plan's lever settings + the preset deltas
  launch.sh            engine-specific launcher (sglang-hybrid docker shape / llama-server flags)
  preflight.sh         refuse to serve below the plan's margins
  verify.sh            smoke: endpoint + one completion (full gates stay the host's job - checklist in README)
  engine-notes.md      deep-lever surface + the advanced patch inventory with portability classification
  registry-surface.json  the parent recipe's 70-knob typed registry, machine-readable for the UI's knob panel
  .gitignore
```

Emission is **GPU-free by design**: it only writes files. The owner executes the recipe on the host
(`bash preflight.sh && bash launch.sh && bash verify.sh`), which is exactly the poteto split — *"the app
forms and witnesses; the host executes"*.

## 2. Digging deep: the lever surface a generated recipe inherits

### 2.1 The complete knob surface (70 knobs)

`registry-surface.json` carries the parent recipe's typed registry verbatim in shape
(name/type/default; extracted 2026-10-23 from `scripts/serve_config.py`). The generated sglang-hybrid
profile files populate every family the plan touches, and the UI's knob panel renders this surface — so the
UI can never invent a knob the registry would refuse. Families and their measured anchors are in
[LEVERS.md](LEVERS.md) §8; the deep summary is in the emitted `engine-notes.md`:

- **KV cache:** context × slots (262K×4 default; ladder 512K/1M needs fraction 0.962 — the pool silently
  caps at 505,856 tokens at 0.96), `MAX_TOTAL_TOKENS` (engine default sized the pool 15× too large, E26),
  logprob chunk 512 (a logprob request otherwise materializes 1.06 GB of fp32 logits and OOMs the server).
- **Placement/CPU tier:** expert count + hot map (42→15 % CPU share, +37 % C1, +120 % C4), `KT_CPUINFER=16`,
  D=4 (D=5 fails NLL — guarded), GPU prefill min tokens.
- **Prefill:** chunk 8192 (16384: +22 % prefill for −4.4 % C1 — guarded beside >219 experts).
- **Engram/NVMe:** ahead-prefetch + chunk-ahead + admission prefetch + page-cache hygiene.
- **Graphs/allocator:** capture "1 2 3 4", expandable segments after capture (stress 678 vs 1,082 s).
- **Determinism:** fused-finalize off + sorted top-k + exact top-k (bit-reproducible greedy at zero cost).
- **Robustness:** per-request NaN rejection, max_new_tokens clamp, API-key file.
- **Host/power:** `HOST_THREAD_CPUS=auto` (kills the C4 bimodality), BIOS FCLK 2100 (+3.1 % C1), the power
  sweep protocol (numbers are per-model — the protocol transfers, the numbers don't).
- **Kernel/quality:** persistent kernel cache, FP8 GEMV opt-in, reasoning effort.
- **Speculation:** off, with reopen triggers; guards/escape hatches exist so experiments are recorded.

### 2.2 The advanced patch inventory (the project's most innovative work)

`engine-notes.md` carries the full 61-patch SGLang overlay inventory grouped by function with a
portability verdict per group: **ports** (W8A8 FP8 configs → PR 1; page-split seals → PR 3/4; indexer
tiling + score budget → PR 6; the DSpark overlap gate; robustness clamps; deterministic top-k — the
upstream post is ready), **stays recipe-local** (Engram file layout + prefetch — model-specific tables;
the el8 hot-expert map — fitted to this model's gates; B2 UVA — retired, do not port; FP8 GEMV — opt-in
until a pinned A/B clears +2 %), and **diagnostics** (BCG debug, timelines, step fingerprints — opt-in,
inert by default). Plus the kt-kernel side (3 patches, `config/kt-kernel-patches/`) and the kernel artifacts
(`e13` MXFP4 grouped GEMM, `e20` W8A8 configs, `t321`, `t341`). A new recipe inherits the classification,
not a blind copy of 61 patches.

### 2.3 Honest gaps (discernment in the *content*, too)

- llama.cpp recipes get the llama.cpp lever set (`-ot` regex offload, `--n-cpu-moe`, draft
  `--moe-expert-cache` LRU, `-fa`, `-c`) — there is **no** deferral, Engram, DSpark or CSA2 there; the
  deepseek41 branch is blocked on the upstream CSA2 merge. The emitted notes say so instead of pretending.
- The prefill √chunk law carries a documented k_prefill error (0.79 SGLang-hybrid, 0.35/0.21 llama.cpp);
  the trust banner makes it visible.

## 3. Optional GitHub publication

`--repo-url https://github.com/<owner>/<repo>` stamps the repo into README/MANIFEST/PROVENANCE and emits a
publish section (`git init -b main && git add -A && git commit && gh repo create <owner>/<repo> --source=.
--private --push`). `--init-git` performs the local part immediately. Rules:

- The app **asks** for the repo; it never pushes without an explicit answer (repo URL + `gh` authed).
- What is publishable differs from what is not: methodology, ledger, plan JSONs, profile files → public-safe;
  model weights, evidence with host identifiers, the 1017 checkpoint pins → never (the identifier scanner in
  the parent CI already refuses host paths/IDs; the emitter inherits that discipline by construction).

## 4. UI discernment: what the app can and cannot do

The UI is a **former and a witness, not an operator**. The boundary is mechanical, and the UI states it:

| The app CAN | The app CANNOT (hands the owner exact commands instead) |
|---|---|
| read budget/hardware JSONs and run the placement calc | download checkpoints or read them off-host (GPU box busy; weights are large) |
| turn every lever and see projections re-light (tuning mode) | run serves, microbenchmarks, or the gate suite — those stay on the host, executed by the owner/agent |
| label trust per projection (the creature opens its eyes) | do the kernel census (needs one GPU window) |
| emit the runnable recipe tree + provenance + manifest | build the docker image or apply the 61-patch overlay |
| scaffold + commit + push a recipe to GitHub (with consent) | change BIOS/power caps or measure bandwidths (the host does; the app records the results) |
| render stamped evidence JSON and gate results | invent evidence — numbers without a manifest render as glass (UI-THEME.md) |
| diff profiles and explain each delta from a measured row | promise numbers a plan has not calibrated |

Every boundary is a **hand-off artifact**: the emitted recipe's README contains the exact commands the owner
runs on the host; the app's job is to make those commands correct, complete, and honest about their trust
level. That is the discerning version of "form a recipe that can be run": the app forms it; the kitchen
runs it.

## 5. Regeneration and staleness

- `emit_recipe.py` is idempotent per `--out` (refuses to overwrite a non-empty tree).
- Re-emit after any plan change: the recipe tree is derived, never hand-edited (same discipline as the
  parent repo's generated tables). The MANIFEST stamps the emitter version and the exact calc command.
- Test coverage: [`tests/test-placement-calc.py`](tests/test-placement-calc.py) also emits a tree and checks
  its key files exist and are identifier-clean.

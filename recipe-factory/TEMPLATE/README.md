# Recipe Factory — New-Recipe Template

Copy this folder to start *"recipe X for model Y on hardware Z"*. The template is **model-agnostic and
engine-agnostic**: it carries the *shape* of the discipline (typed knobs, profiles, gates, evidence, CI),
not the DeepSeek-V4.1 specifics. The parent repository is the worked example for every file here —
read the corresponding file there (pointed at in each header) before customizing.

```
recipe-<model>-<hardware>/            # your new repo (or a folder in an existing one)
├── README.md                          # verdict-first: fits/does-not-fit + ranked paths (see parent README)
├── PROVENANCE.md                      # every pin: model snapshot, engine, image, overlay
├── config/
│   ├── profiles/
│   │   ├── qualified.env              # the serve shape every headline number is measured on
│   │   └── developer.env              # same shape + the only profile allowed to carry guard bypasses
│   └── example.env                    # legacy-style env for any download/serve.sh chain
├── scripts/
│   ├── checkpoint-budget.py           # P0: safetensors-header byte budget (copy from the parent repo)
│   ├── feasibility-calc.py            # P0: placement calculator (copy from the parent repo)
│   ├── hardware-inventory.sh          # P1: identifier-free hardware sheet
│   ├── bench-dram.py                  # P1: DRAM read bandwidth
│   ├── bench-pcie-h2d.py              # P1: PCIe H2D + additivity
│   ├── bench-engram-gather.py         # P1: random row gather, DRAM vs NVMe (rename per model's hot tables)
│   ├── serve_config.py                # ← THIS template: the typed knob registry
│   ├── serve.sh                       # ← THIS template: the launcher (evals the registry)
│   ├── wait-ready.py                  # readiness probe (health ≠ ready; watch the startup warmup)
│   ├── verify.py                      # P3 gate: alias/context/arithmetic/thinking/tools/needles
│   ├── nll-score.py                   # P3 gate: teacher-forced NLL identity (copy from the parent repo)
│   ├── stress-kv-full.py              # P3 gate: fill every slot, report peak VRAM
│   ├── verify-batch-consistency.py    # P3 gate: same-prompt batch agreement
│   ├── spec-consistency.py            # P3 gate: only if speculation is ever tried
│   ├── bench-decode.py                # ← THIS template: C1/C4 harness contract
│   ├── bench-prefill.py               # ← THIS template: cold/warm prefill + TTFT contract
│   ├── evidence_manifest.py           # evidence discipline (copy from the parent repo)
│   ├── benchmarks_headline.py         # generated headline table (copy from the parent repo)
│   └── ci.sh                          # ← THIS template: the CI gate set
├── evidence/                          # one dir per experiment: README.md + stamped JSON + manifest.json
├── docs/                              # FEASIBILITY, HARDWARE, BENCHMARKS, CONFIG, TROUBLESHOOTING, LEARNING
├── plan/                              # README (P0-P4), ROADMAP (running readout), EXPERIMENTS, TASKS, RESEARCH
├── systemd/                           # the service unit template
└── results/                           # gitignored scratch
```

**Use order** (the phases; full playbook in [../METHODOLOGY.md](../METHODOLOGY.md)):

1. `checkpoint-budget.py` + `hardware-inventory.sh` → `feasibility-calc.py` (P0; calibrate on a known
   deployment first).
2. The P1 microbenchmarks; every number lands in `docs/HARDWARE.md`, zero "estimate" rows left.
3. Re-measure the incumbent deployment with `bench-decode.py` / `bench-prefill.py` / `verify.py` (P2).
4. Customize `serve_config.py` (declare the engine's real knobs) and `serve.sh`; build; run every gate (P3).
5. The scale columns (P4), then the profile presets from [../KNOBS-AND-PROFILES.md](../KNOBS-AND-PROFILES.md) §3.

**Rules carried over from the parent repo (non-negotiable):** never ship a recipe that has not run; no host
identifiers in committed evidence; every knob in an owned name family is declared or the launch refuses; a
removed knob refuses with the experiment that removed it; every performance claim in a generated table is
read from manifest-stamped JSON.

# DeepSeek-V4.1-Flash FP4 on RTX PRO 6000 Blackwell: feasibility, plan, and recipe scaffold

A measured answer to "can two RTX PRO 6000 Blackwell 96 GB serve DeepSeek-V4.1-Flash at 262K context, and if not, what would?" — plus a phased plan to benchmark what does run, and the scripts to do it. Modeled on the [GLM-5.3 Flash dual-RTX-PRO-6000 recipe](https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell): pinned revisions, fail-closed preflight, reproducible tables, no host identifiers in public evidence.

## Verdict

**Two RTX PRO 6000 cannot serve DeepSeek-V4.1-Flash today. The shortfall is ~112 GiB of VRAM, not a flag.**

| | GiB |
|---|---:|
| V4.1-Flash weights that must be GPU-resident (everything except the Engram tables, 4-bit experts as shipped) | **286.4** |
| Usable VRAM on 2 × 95.6 GiB at 0.95 utilisation minus 4 GiB/GPU reserve | **173.6** |
| Margin with Engram tables in host DRAM (the best 2-GPU layout) | **−112.1** |
| KV cache for one 262,144-token sequence (CSA2, 890 B/token) | 0.22 |

The bytes are read from the shard headers of `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…` ([`evidence/budget-upstream-bf16fp8.json`](evidence/budget-upstream-bf16fp8.json)); the placement math is [`scripts/feasibility-calc.py`](scripts/feasibility-calc.py). The routed experts are **already MXFP4** in the upstream checkpoint (268.9 GiB); NVIDIA's NVFP4 re-quantization is 15.8 GiB *larger*. The 262K context itself is trivial — the source document was right about that and wrong about the model size (its "~151 GB backbone" is DeepSeek-**V4**-Flash).

Full derivation, five options and their ceilings: [docs/FEASIBILITY.md](docs/FEASIBILITY.md).

Measured so far on this host (GPU-free, [docs/BENCHMARKS.md](docs/BENCHMARKS.md)): host DRAM copy 146 GB/s aggregate (E1); Engram-style random row gather 4.15 M rows/s from DRAM, 403K reads/s from the Samsung 9100 PRO, 347K from the Crucial T705 (E3). Engram lookups are not a bottleneck in DRAM and are decode-viable from either NVMe; the missing 112 GiB is the whole problem.

### What this host can do

| Path | Status | Expected result |
|---|---|---|
| **A. Four RTX PRO 6000** + Engram tables in this host's 246.8 GiB DRAM (SGLang `dev-dsv41` host-table path) | proven pattern on 4× SM120 by 0xSero; profile written, **unqualified** until cards exist; **the 1,500 W PSU cannot carry four cards at 300 W** — needs a 2,000 W-class PSU or a 225 W cap ([HARDWARE.md power](docs/HARDWARE.md#power-and-cooling-planning-inputs-for-option-a)) | ~200 tok/s C1, ~700 tok/s C8, ~7K tok/s prefill (0xSero, 275 W); this host could be the first with the whole Engram in DRAM instead of NVMe |
| **B. Stream 39% of experts from DRAM over PCIe** on 2 GPUs | no MoE-aware offloader exists for `deepseek_v41` on SM120; prototype planned | ≤ ~60 tok/s decode ceiling at batch 1 (PCIe-bound), 2–4K tok/s prefill |
| **C/D. Sub-3-bit experts or 39% expert pruning** | no kernels / no evaluated artifact | research only |
| **Baseline: DeepSeek-V4-Flash-0731 NVFP4, TP2** | already qualified on this host (sibling recipe) | 200.6 tok/s single-stream with DSpark K5, 1M context, 226 tok/s C2 |

## What is in this repository

- [docs/FEASIBILITY.md](docs/FEASIBILITY.md) — the memory math, placement table, five options with ceilings, and an audit of the source document's claims.
- [docs/HARDWARE.md](docs/HARDWARE.md) — ASUS WRX90E-SAGE SE, Threadripper PRO 9955WX, 246.8 GiB DDR5 (8 × Kingston FURY Renegade Pro KF564R32-32, configured 6400 MT/s), 2 × RTX PRO 6000 Max-Q, Samsung 9100 PRO / Crucial T705, with the derived bandwidths that feed the verdict and which ones are still vendor specs.
- [plan/README.md](plan/README.md) — five phases (research → host microbenchmarks → baseline → boundary experiments → 4-GPU qualification), innovation candidates ranked by measurability, non-goals.
- [plan/EXPERIMENTS.md](plan/EXPERIMENTS.md) — E1–E9 with numeric hypotheses and pass criteria. [plan/TASKS.md](plan/TASKS.md) — status per task. [plan/RESEARCH.md](plan/RESEARCH.md) — sources and open questions.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — method and tables (checkpoint budget filled; throughput tables await GPU windows). [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — SM120 pitfalls, hugepages, thinking mode.
- `scripts/` — see below. `evidence/` — committed JSON. `config/example.env`, `systemd/`, `examples/`.
- [PROVENANCE.md](PROVENANCE.md), [ATTRIBUTIONS.md](ATTRIBUTIONS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [CONTRIBUTING.md](CONTRIBUTING.md); Apache-2.0 for this repository, MIT text retained for the model.

## Scripts

Runnable now, no model download, no GPU contention:

| Script | Purpose |
|---|---|
| `checkpoint-budget.py <repo\|dir>` | exact per-component bytes from safetensors headers (HTTP range or local), stdlib only |
| `feasibility-calc.py <budget.json> --gpus N ...` | fits / does-not-fit per layout with margins; exit 3 when it does not fit |
| `hardware-inventory.sh [--json]` | identifier-free hardware sheet |
| `bench-engram-gather.py --backend dram\|nvme` | E3: 48-row random gathers per token, DRAM vs NVMe |
| `bench-dram.py` | E1: host copy bandwidth |

Need a GPU window (stop the GLM container first):

| Script | Purpose |
|---|---|
| `bench-pcie-h2d.py` (inside a torch container) | E2/E4: GDDR7 and PCIe bandwidth; prints the Option B decode ceiling |
| `download-model.sh` → `preflight.sh` → `serve.sh` → `wait-ready.py` → `verify.py` | the recipe chain; `preflight.sh` **fails closed** on 2 GPUs |
| `bench-decode.py`, `bench-prefill.py` | the shared benchmark harness (also for the V4-Flash baseline) |

`scripts/ci.sh` runs unit tests, checks that the README verdict still agrees with the calculator, and scans tracked files for identifiers.

## Quick start (what you can do today)

```bash
git clone <this repo> && cd deepseek-v41-fp4-recipe-rtx6000pro
scripts/ci.sh                                   # tests + consistency + identifier scan
scripts/hardware-inventory.sh
scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2
scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 4 --host-gib 246.8 --layout engram-host
scripts/bench-engram-gather.py --backend dram --rows 20000000
scripts/bench-engram-gather.py --backend nvme --path results/engram-nvme0.bin --rows 20000000   # results/ lives on nvme0
```

When two more cards are installed:

```bash
cp config/example.env .env && $EDITOR .env      # MODEL_DIR on nvme0, GPU_DEVICES=0,1,2,3
sudo sysctl -w vm.nr_hugepages=97280            # ~190 GiB for the Engram host table
docker pull lmsysorg/sglang:dev-dsv41           # then pin the digest in .env and PROVENANCE.md
scripts/download-model.sh                       # 475.24 GiB
scripts/serve.sh                                # runs preflight.sh first
scripts/wait-ready.py && scripts/verify.py --context 262144 --output results/verify.json
scripts/bench-decode.py --output results/decode.json
scripts/bench-prefill.py --output results/prefill.json
```

## Requirements (for the 4-GPU profile)

- Linux, Docker with NVIDIA Container Toolkit, `hf` CLI, Python ≥ 3.10 (stdlib only on the host)
- Four RTX PRO 6000 Blackwell 96 GB with P2P read OK between all pairs
- ≥ 213 GiB host RAM for Engram-in-DRAM (189 GiB tables + reserve), or ≥ 88 GiB with Engram on NVMe
- ~500 GiB free NVMe for the checkpoint

Evidence in this repository was gathered on Ubuntu 26.04.1, kernel 7.0.0-31, driver 615.71.09, Docker 29.8.1. Those are the tested conditions, not minimums.

## Status

Phase 0 (research and data) complete. Phase 1 microbenchmarks runnable. Phases 2–3 need a GPU window on the existing deployment. Phase 4 is blocked on hardware. See [plan/TASKS.md](plan/TASKS.md).

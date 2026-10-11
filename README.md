# DeepSeek-V4.1-Flash MXFP4 on RTX PRO 6000 Blackwell: feasibility, plan, and recipe

Repository: `github.com/hfyeomans/deepseek-v41-mxfp4-recipe-rtx6000pro` (private, created 2026-10-10). The checkpoint's routed experts are MXFP4 (UE8M0 block-32 scales, the upstream checkpoint as shipped; [PROVENANCE](PROVENANCE.md)), not NVIDIA's NVFP4 re-quantization, which is why the remote carries `mxfp4` while the original local checkout directory still says `fp4`.

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
| Same at the 131 GB/s the BIOS slot-3 profile reaches (E27, FCLK 2100; calculator default since 2026-09-23) | | ~52 / ~79 tok/s | ~121 / ~156 | ~200 |
| Parallelism | | TP2 | **TP3 invalid** (64 heads) → EP3/PP | TP4 |
| Power on the 1,500 W PSU | | fine | CPU PPT cap | **needs 2,000 W or 225 W/card** |

Engram tiering on 2 GPUs: full FP8 tables in DRAM do **not** fit (−78 GiB); FP4 (+13 GiB) or q3_K (+32 GiB) tables do; NVMe + 64 GiB cache (+47 GiB) is the default because E3 showed both drives sustain decode-rate lookups. Full derivation, options and the 3/4-GPU columns: [docs/FEASIBILITY.md](docs/FEASIBILITY.md).

Measured so far on this host ([docs/BENCHMARKS.md](docs/BENCHMARKS.md)): CPU DRAM read 125 GB/s (E1b; a two-CCD fabric limit, 30 % of the 409.6 theoretical — E25 measured it directly: 62–63 GB/s per CCD, 235 GB/s with both GPUs' DMA on top); GPU DMA reads of the same DRAM 114 GB/s over both x16 links and **additive** with the CPU's (E4/E4b, 234 GB/s together); GDDR7 ≈1,470 GB/s per card (E2); Engram-style random row gather 3.15 M whole rows/s from DRAM (Python-bound lower bound; re-measured 2026-10-11), 403K reads/s from the Samsung 9100 PRO, 347K from the Crucial T705 (E3); a native MXFP4×MXFP4 grouped GEMM for SM120 (`kernels/e13/`) that verifies and streams expert weights at 1,397 GB/s at decode shapes and reaches 1,060 TFLOPS dense at 300 W (E13).

### Paths, ranked

| Path | Status | Expected result |
|---|---|---|
| **B. Three-tier hybrid on 2 GPUs** — port `deepseek_v41` into KTransformers' sglang-kt (validated for V4-Flash only), or run llama.cpp's `deepseek41` branch when CSA2 lands, or our own PCIe expert-streaming prototype | **serving** (E19–E34, `scripts/serve-hybrid.sh`: SGLang overlay + kt-kernel, ~~226 GPU / 158 CPU experts~~ 219 GPU / 165 CPU experts + prefill chunk 8192 since 2026-10-08 (T3.28), per-layer hot-expert map `config/expert-map-el8-mix-219.json` since 2026-10-10 (T3.8b), D=4 deferral, streamed prefill, spec off, **64K context × 4 slots** since 2026-10-07, FCLK 2100 BIOS profile) | measured: **C1 58.0 tok/s median / 56.2 wall, C4 123.9 aggregate / 118.8 wall (30 per stream) since 2026-10-10 (T3.8b: each layer's hot experts take the GPU slots via `KT_EXPERT_MAP`, the el8 mix map; CPU-tier route share 42 → 15 %)** — ~~C1 41.5 tok/s summed / 39.5 wall (226 / 158 read 42 / 40), C4 56 / 54 wall (13.5 per stream)~~ on checkpoint-order placement, prefill **2,195 / 2,499 tok/s at 16K / 32K cold since T3.20 (c), 2026-10-10** (16K TTFT 7.3 s; ~~1,887 / 1,901 since E34d-v (ii), 2026-10-09~~, 2,040 / 2,272 after T3.39 window 1b; 1,400–1,820 on the same default before the three transient removals; ≈ 1,040 at chunk 4096, ≈ 850 before the E34b indexer tiling), 6/6 needles at 222K, NLL bit-identical across configs ([BENCHMARKS](docs/BENCHMARKS.md) E33/E34); **512K × 1 slot passes every gate since the 2026-10-08 ladder rung and again on the 219 / 8192 default (T3.29)** (508K prompt, TTFT 369 s — 596 s at 226 / 4096; needles 6/6 at 444K, decode 41.6 tok/s; needs `MEM_FRACTION_STATIC=0.962`; above 64K the launcher allows one slot only without `ALLOW_UNTESTED_FIT=1` — ~~262K × 4 OOMs~~ **262K × 4 passes every gate at chunk 4096 since 2026-10-09 (E34d-v (b): 4 × 253K prompts, needles 6/6 at 222K, prefill 975 tok/s; OOMs at chunk 8192)**; ~~ladder closed at 512K~~ **1M × 1 slot passes every gate since 2026-10-09 (E34d-v (ii): 1,019K-token prompt TTFT 803 s, needles 6/6 at 887K, decode 39.4 wall; `MEM_FRACTION_STATIC=0.962`)**) — ~~262K × 1 slot passes every gate since E34b-ii (253K prompt, TTFT 286 s, decode 42 tok/s at 253K context)~~ — the question in the first line is answered yes, single-slot; ~~128K × 1 slot passes every gate since E34b, 262K OOMs on the indexer's candidate mask (E34b-ii next)~~. Every C3/C4 figure before 2026-09-23 is an artifact (BENCHMARKS "Erratum"). Pre-build projection was 56–89 tok/s at batch 1 |
| **E. Three GPUs**, 91% of experts resident, EP3/PP | future comparison; no card | ~119–155 tok/s |
| **A. Four RTX PRO 6000** + Engram tables in DRAM (SGLang `dev-dsv41` host-table path) | proven pattern on 4× SM120 by 0xSero; profile written, unqualified; **1,500 W PSU cannot carry four cards at 300 W** ([power](docs/HARDWARE.md#power-and-cooling-planning-inputs-for-option-a)) | ~200 tok/s C1, ~700 tok/s C8, ~7K tok/s prefill (0xSero, 275 W) |
| **C/D. Sub-3-bit experts or 42% expert pruning** | no kernels / no evaluated artifact | research only; D gated on E7 routing skew |
| **Baseline: DeepSeek-V4-Flash-0731 NVFP4, TP2** | qualified on this host; **re-measured with this harness (E8)** | 171–239 tok/s C1 with DSpark K5 (acceptance 40–63 % drives the spread), ~300 tok/s wall-clock aggregate at `max_num_seqs=2`, prefill 7.8K tok/s at 32K → 1.9K at 1M, 6/6 needles at 222K — the number to beat per parameter ([BENCHMARKS](docs/BENCHMARKS.md#e8-baseline-deepseek-v4-flash-0731-nvfp4-tp2-2--rtx-pro-6000-max-q)) |

## What is in this repository

- [docs/FEASIBILITY.md](docs/FEASIBILITY.md) — the memory math, placement table, five options with ceilings, and an audit of the source document's claims.
- [docs/HARDWARE.md](docs/HARDWARE.md) — ASUS WRX90E-SAGE SE, Threadripper PRO 9955WX, 246.8 GiB DDR5 (8 × Kingston FURY Renegade Pro KF564R32-32, configured 6400 MT/s), 2 × RTX PRO 6000 Max-Q, Samsung 9100 PRO / Crucial T705, with the derived bandwidths that feed the verdict and which ones are still vendor specs. [docs/BIOS-OPTIMIZATION-GUIDE.md](docs/BIOS-OPTIMIZATION-GUIDE.md) — firmware settings map for this exact board/CPU/8-DIMM build, inventory, staged trial protocol (nothing changed yet; every tier is owner-approved).
- [plan/README.md](plan/README.md) — five phases (research → host microbenchmarks → baseline → 2-GPU hybrid build → 3/4-GPU comparison), innovation candidates ranked by measurability, non-goals.
- [plan/ROADMAP.md](plan/ROADMAP.md) — running readout: every experiment's status, result so far and the decision it fed; kernel experiment menu; options register. [plan/EXPERIMENTS.md](plan/EXPERIMENTS.md) — E1–E13 with numeric hypotheses and pass criteria. [plan/TASKS.md](plan/TASKS.md) — status per task. [plan/RESEARCH.md](plan/RESEARCH.md) — sources and open questions.
- [docs/LEARNING.md](docs/LEARNING.md) — reading map per puzzle piece (model architecture, FP4 formats, SM120, PTX, CUTLASS/CuTe, profiling, engines, memory hierarchy, host hardware, benchmarking) and what each experiment teaches.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — method and tables (checkpoint budget filled; throughput tables await GPU windows). [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — SM120 pitfalls, hugepages, thinking mode.
- [docs/research/](docs/research/README.md) — durable design notes with measured/projected labels: Mac (exo/MLX) and third-GPU feasibility, the interactive-vs-local router, the E19→E23 benchmark evolution and E20 FP8 tuning in detail, the current hybrid architecture and Engram decision, the open-source/upstreaming plan for the SGLang overlay, and the [Python-to-native hybrid migration roadmap](docs/research/MIGRATION-ROADMAP-PYTHON-TO-NATIVE-HYBRID.md).
- `scripts/` — see below. `evidence/` — committed JSON. `config/example.env`, `systemd/`, `examples/`.
- [PROVENANCE.md](PROVENANCE.md), [ATTRIBUTIONS.md](ATTRIBUTIONS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [CONTRIBUTING.md](CONTRIBUTING.md); Apache-2.0 for this repository, MIT text retained for the model.

## Scripts

Runnable now, no model download, no GPU contention:

| Script | Purpose |
|---|---|
| `checkpoint-budget.py <repo\|dir>` | exact per-component bytes from safetensors headers (HTTP range or local), stdlib only |
| `feasibility-calc.py <budget.json> --gpus 2,3,4 [--layout hybrid-cpu --engram nvme\|dram-fp4\|dram-q3k\|dram --dram-gbps N]` | fits / does-not-fit per layout and GPU count with margins; hybrid rows report GPU/CPU expert split, DRAM GiB/token and decode ceilings; exit 3 when it does not fit |
| `hardware-inventory.sh [--json]` | identifier-free hardware sheet |
| `bench-engram-gather.py --backend dram\|nvme` | E3: 48-row random gathers per token, DRAM vs NVMe |
| `bench-dram.py` | E1: host copy bandwidth |
| `evidence_manifest.py write\|show\|verify\|check-evidence` | T3.19f: the per-serve provenance manifest (recipe/overlay SHA, image + container id, resolved knobs, BIOS slot + HSMP clocks, GPU caps, pre-serve memory and Engram page-cache state, content hash). The launcher writes one; the evidence scripts stamp their JSON with its hash and copy it beside the output as `manifest.json`; `ci.sh` checks `evidence/` |
| `hsmp-clocks.py [--watch 1 --n 30]` | live FCLK / MCLK / DDR-controller GB/s / socket W from the kernel `hsmp_acpi` driver, unprivileged — the BIOS Tier 3 clock check (E27) |
| `export-sglang-series.sh` | T3.24c: re-exports the SGLang overlay branch as `patches/sglang/*.patch` + `SERIES.env` (52 patches on upstream `da64c5cb`; the README there has the apply recipe and a patch → task index); `ci.sh` re-applies the series and checks the tree hash |

Need a GPU window (stop the GLM container first):

| Script | Purpose |
|---|---|
| `benchmarks_headline.py markdown` / `check` / `write` | T3.24e: the headline table at the top of [docs/BENCHMARKS.md](docs/BENCHMARKS.md) is generated from the manifest-stamped JSON of the rungs listed in `config/benchmarks-headline.json` (a file whose stamp does not match its sibling `manifest.json` is refused); `ci.sh` fails when the block is stale. |
| `release-smoke.sh units` / `release-smoke.sh serve --output-dir evidence/<dir> [--context N --slots N --full]` | T3.24d: the pre-release GPU smoke — the three GPU unit suites with the cards free, then the qualification gate set (same-prompt C4, NLL corpus vs the T3.28 baseline, needles, log health; `--full` adds stress + decode / prefill benches) against a running `serve-hybrid.sh` server; see CONTRIBUTING |
| `bench-pcie-h2d.py` (inside a torch container) | E2/E4: GDDR7 and PCIe bandwidth; prints the Option B decode ceiling |
| `bench-dram-read.py` (inside a torch container, GPU-free) | E1b: read-only host DRAM bandwidth (fp32/int64 sum) at 8/16/32 threads |
| `bench-dram-contention.py` (inside a torch container, `--gpus all`) | E4b: CPU DRAM reads and both-GPU H2D DMA alone and together; additivity |
| `kernels/e13/build.sh`, `kernels/e13/run.py` | E13: build the SM120 MXFP4×MXFP4 grouped GEMM (GPU-free, `dsv4-sm120:cuda-devel-https`) and run it at V4.1 expert shapes |
| `download-model.sh` → `preflight.sh` → `serve.sh` → `wait-ready.py` → `verify.py` | the legacy all-GPU chain; `preflight.sh` **fails closed** on 2 GPUs. `stop.sh` removes whichever serve is up (hybrid `dsv41-kt` or legacy); `systemd/dsv41-flash.service.in` runs the qualified hybrid |
| `bench-decode.py`, `bench-prefill.py` | the shared benchmark harness (also for the V4-Flash baseline) |
| `bench-humaneval.py`, `bench-reasoning.py` | E32 / T3.8b soak: HumanEval pass@1 greedy (`--thinking` optional; programs run in a no-network container) and long-reasoning traffic (hard prompts, thinking on, up to 16K tokens, known answers spot-checked) — real coding / reasoning decode for the recorder and the map |
| `stress-kv-full.py` | E28 fit gate for any VRAM spend: fills every running slot with a ≈ 30K prompt, decodes them together and reports the peak `nvidia-smi` memory per GPU — a single 32K prefill at C1 is not the worst case a config admits |
| `serve-hybrid.sh`, `serve_config.py` | E19–E21: the 2-GPU + CPU-expert (kt-kernel) hybrid; every knob (`KT_*`, `DSPARK*`, `MEM_FRACTION_STATIC`, `CUDA_GRAPH_BS`, `EXTRA_*`, …) is declared once in the typed registry `serve_config.py` (T3.19e) — table in [docs/CONFIG.md](docs/CONFIG.md); a removed knob, or an undeclared variable in any owned name family (`KT_*` … `CONTEXT_LENGTH*`, the `OWNED_PREFIXES` list), is refused before docker starts; the hybrid reads the environment and `config/profiles/*.env`, never the legacy `.env`; `DRY_RUN=1` / `--print` probe without launching; `PROFILE=<name>` loads `config/profiles/<name>.env` under the environment (`qualified` by default, `developer` for the guard bypasses; T3.24a) and the resolved line names the profile plus every env override |
| `step-profile.py`, `gap-context.py` | E20/E21: per-step kernel timeline from a torch-profiler trace (`/start_profile`), and what surrounds each GPU idle gap |
| `kt-cpu-per-step.sh`, `sched-wait-per-step.py` | E21: kt CPU-seconds per decode step, and per-thread run-queue wait / context switches (`/proc/<tid>/schedstat`) around one bench run |
| `nll-score.py` | quality metric for approximations (expert deferral, kernels): teacher-forced mean NLL of `scripts/fixtures/nll-corpus.txt`, `--compare` a baseline JSON |
| `verify-batch-consistency.py` | E23 R3b / E26 gate for any concurrency number: `--same-prompt` sends N identical greedy prompts in one batch and requires every row to agree (min pairwise prefix, per-slot prefix vs a bs=1 reference, gibberish ratio); the check that exposed the bug-C bs=4 graph behind the E19–E23 C4 figures |
| `verify-disconnect-abort.py` | T3.44 client contract: four 32K+ streaming requests, one client closes its socket mid-decode; passes when `#running-req` drops within 10 s (stock SGLang frees the slot on the next decode batch) |
| `greedy-logprobs.py` | gross sanity only — greedy picks flip at near-ties run-to-run on the hybrid (E21 r7); use `nll-score.py` for numbers |
| `kt-export-bench.py` | E22a: exports one layer's CPU experts from kt-kernel into pinned host buffers (`gpu_tp_count=2`), byte-compares the slices against the safetensors shards, times export / H2D / pipelined streaming, and checks the staged FlashInfer SM120 MoE output against the resident layout and an fp32 reference (`--check-tokens`, `--skip-timing`, `--json`); runs inside `dsv41-kt:recipe` without a server |

`scripts/ci.sh` runs unit tests, checks that the README verdict still agrees with the calculator, and scans tracked files for identifiers.

## Quick start (what you can do today)

```bash
git clone <this repo> && cd deepseek-v41-fp4-recipe-rtx6000pro
scripts/ci.sh                                   # tests + consistency + identifier scan
scripts/hardware-inventory.sh
scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2,3,4            # resident layouts: only 4 GPUs fit
scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2,3,4 --layout hybrid-cpu --engram nvme --dram-gbps 125
scripts/bench-engram-gather.py --backend dram --rows 20000000
scripts/bench-engram-gather.py --backend nvme --path results/engram-nvme0.bin --rows 20000000   # results/ lives on nvme0; the file is created and deleted by the run (--keep-file keeps it; an existing foreign file is refused unless --force)
```

When two more cards are installed:

```bash
cp config/example.env .env && $EDITOR .env      # MODEL_DIR on nvme0, GPU_DEVICES=0,1,2,3
sudo sysctl -w vm.nr_hugepages=97280            # ~190 GiB for the Engram host table
docker pull lmsysorg/sglang:dev-dsv41           # then pin the digest in .env and PROVENANCE.md
scripts/download-model.sh                       # 475.24 GiB
scripts/serve.sh                                # runs preflight.sh first
scripts/wait-ready.py --url http://127.0.0.1:8010 --container dsv41-flash-rtxpro6000   # legacy port / name; the defaults are the hybrid's
scripts/verify.py --url http://127.0.0.1:8010 --context 262144 --output results/verify.json
scripts/bench-decode.py --output results/decode.json
scripts/bench-prefill.py --output results/prefill.json
```

## Requirements

For the 2-GPU three-tier hybrid (the build target; no engine ships it yet):

- Two RTX PRO 6000 Blackwell 96 GB, PCIe 5.0 x16 each, P2P OK
- ≥ 200 GiB host RAM: 112 GiB CPU-resident experts + 64 GiB Engram row cache + 24 GiB reserve (this host: 246.8 GiB, 8-channel DDR5-6400)
- A CPU with AVX-512 or at least AVX2 and enough DRAM bandwidth to matter: decode tok/s ≈ 1000 / (5 ms + 1.75 GiB ÷ DRAM GB/s)
- NVMe with ≥ 500 GiB free for the checkpoint and ~190 GiB of Engram tables served by random 264 B reads (Samsung 9100 PRO measured 403K reads/s)

For the 4-GPU resident profile:

- Linux, Docker with NVIDIA Container Toolkit, `hf` CLI, Python ≥ 3.10 (stdlib only on the host)
- Four RTX PRO 6000 Blackwell 96 GB with P2P read OK between all pairs, and a PSU that carries them (2,000 W-class, or 225 W per card on 1,500 W)
- ≥ 213 GiB host RAM for Engram-in-DRAM (189 GiB tables + reserve), or ≥ 88 GiB with Engram on NVMe
- ~500 GiB free NVMe for the checkpoint

Evidence in this repository was gathered on Ubuntu 26.04.1, kernel 7.0.0-31, driver 615.71.09, Docker 29.8.1. Those are the tested conditions, not minimums.

## Status

**Where the model lives at the shipped default (T3.30 snapshot, 2026-10-10; full tables with evidence in [docs/BENCHMARKS.md § Deployment snapshot](docs/BENCHMARKS.md#deployment-snapshot-t330-2026-10-10)):**

| Tier | Contents | Size |
|---|---|---|
| 2 × RTX PRO 6000 (TP2) | FP8 dense stack + **219 of 384 routed experts per layer, MXFP4**, chosen per layer by the hot-expert map | 86.3 GB per rank at load (≈ 153 GiB of experts across both cards); KV pool 0.45 GB per rank at 64K × 4, 1.74 GB at 262K × 4 / 1M × 1 (FP8, 1,630 B per token per rank) |
| Host DRAM (246.8 GiB) | **165 of 384 routed experts per layer, MXFP4**, computed by kt-kernel (16 threads, D=4) | ≈ 115 GiB, one shared copy for both ranks |
| NVMe (page cache) | **Engram tables** (layers 1 and 14), FP8, read in place with ahead-prefetch | 188.83 GiB; the first 16K prefill after a serve runs at ~600 tok/s while rows are faulted in |

Progression: C1 **14 → 58 tok/s**, C4 **— → 118 wall**, 16K prefill **216 → 2,934–3,264 tok/s**, qualified context **32K × 4 → 64K × 4 default + 262K × 4 / 512K × 1 / 1M × 1** over 2026-09-22 → 2026-10-10 (same hardware; FCLK 2000 → 2100).

**2026-10-10: the recipe is out of experimental-overlay form** — the six-point completion definition of the migration roadmap's gate ledger (G6, T3.24) holds: typed profiles, no unsafe flag in a release profile, the overlay as a patch series on upstream `da64c5cb` (50 at the gate; 52 as of T3.43c) re-applied and tree-hashed by `ci.sh`, every CPU suite in `ci.sh` plus the GPU smoke in `release-smoke.sh`, provenance for every artefact, and a generated headline table whose every number is read from manifest-stamped evidence ([docs/BENCHMARKS.md](docs/BENCHMARKS.md#headline--qualified-rungs-of-the-v41-flash-hybrid-generated-2026-10-10)). Qualified rungs: 64K × 4 (launcher default), 262K × 4 and 512K × 1 / 1M × 1 as per-rung overrides. The overlay is tagged `recipe-2026-10` at `682fc6a0`. What remains is throughput work and strengthening (T3.30 onwards in [plan/TASKS.md](plan/TASKS.md)); the earlier phase history is in [plan/README.md](plan/README.md).

# Tasks

Status: `done`, `ready` (can run now), `gpu-window` (needs the GLM container stopped), `blocked` (external dependency), `decision` (owner call).

## P0 — Research and data

| ID | Task | Status | Output |
|---|---|---|---|
| T0.1 | Read shard headers of the three V4.1-Flash checkpoints; classify bytes per component | done | `evidence/budget-*.json`, `scripts/checkpoint-budget.py` |
| T0.2 | Hardware inventory without identifiers | done | `evidence/hardware-inventory.json`, `scripts/hardware-inventory.sh` |
| T0.3 | Feasibility calculator; validate against the 0xSero 4-GPU layout | done | `scripts/feasibility-calc.py`, FEASIBILITY.md §3 |
| T0.4 | Audit the source document's claims against primary sources | done | FEASIBILITY.md §6, RESEARCH.md |
| T0.5 | DIMM part number, rated and configured speed, CAS latency | **done** (KF564R32-32 × 8, SPD 4800, configured 6400 = EXPO; CL not exposed by SMBIOS, vendor CL32) | HARDWARE.md memory table |
| T0.6 | Record PSU model and wall circuit capacity | **done** (1,500 W; circuit still unrecorded) — verdict: 4 × 300 W does not fit, see T4.0 | HARDWARE.md power table |

## P1 — Host microbenchmarks

| ID | Task | Status | Output |
|---|---|---|---|
| T1.1 | E3 Engram gather: DRAM, nvme0, nvme1 | **done** | `evidence/e3-engram-gather-*.json` |
| T1.2 | E1 DRAM bandwidth | **done** | `evidence/e1-dram-bandwidth.json` |
| T1.3 | E4 PCIe H2D/D2H per GPU and both concurrently | gpu-window | `results/e4.json` |
| T1.4 | E2 GDDR7 device bandwidth | gpu-window | `results/e2.json` |
| T1.5 | Recompute Option B ceiling from T1.3; update FEASIBILITY.md §4 | after T1.3 | FEASIBILITY.md |
| T1.6 | Hugepage reservation test: `vm.nr_hugepages` for ~190 GiB of 2 MiB pages while GLM is running; confirm allocation succeeds and revert | ready (needs root) | TROUBLESHOOTING.md note |

## P2 — Baseline

| ID | Task | Status | Output |
|---|---|---|---|
| T2.1 | Port fixture/method from the GLM and V4-Flash recipes into `bench-decode.py` / `bench-prefill.py` so numbers are comparable | done (harness written) | `scripts/bench-*.py` |
| T2.2 | E8 run against the V4-Flash NVFP4 TP2 deployment | gpu-window | `evidence/baseline-v4-flash/` |
| T2.3 | Fill BENCHMARKS.md baseline tables from JSON | after T2.2 | `docs/BENCHMARKS.md` |
| T2.4 | Baseline improvement sweep on V4-Flash: DSpark block 3/5/7, FP8 KV, C8/C16 | gpu-window | BENCHMARKS.md |

## P3 — Boundary

| ID | Task | Status | Output |
|---|---|---|---|
| T3.1 | Download `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…` (475 GiB) to nvme0 | ready (disk: 1.7 TB free) | HF cache |
| T3.2 | Pull `lmsysorg/sglang:dev-dsv41`; record image digest in PROVENANCE.md | ready | PROVENANCE.md |
| T3.3 | E5 2-GPU attempt with SGLang and vLLM; scrub and record logs | gpu-window, after T3.1–T3.2 | TROUBLESHOOTING.md |
| T3.4 | E6 stage 1 synthetic expert-streaming loop | gpu-window, after T1.3 | `results/e6-stage1.json` |
| T3.5 | E6 stage 2 design doc (only if stage 1 ≥ 40 tok/s) | decision | `plan/E6-DESIGN.md` |
| T3.6 | E7 routing-skew histogram on V4-Flash | gpu-window | `evidence/e7-routing/` |

## P4 — 4-GPU qualification

| ID | Task | Status | Output |
|---|---|---|---|
| T4.0 | Power decision for four cards: 2,000 W-class PSU (needs 240 V or 30 A circuit) **or** 225 W cap on the 1,500 W PSU; decided by E8b's tok/s-per-watt curve | decision, after E8b | HARDWARE.md power table |
| T4.1 | Acquire and install two RTX PRO 6000 (Max-Q preferred for power); verify P2P 4×4 OK and x16 on all slots | blocked on T4.0 | HARDWARE.md |
| T4.2 | `preflight.sh` passes the memory gate on 4 GPUs | after T4.1 | — |
| T4.3 | E9 serve + verify + benchmark, Engram-in-DRAM and NVMe modes | after T4.2 | BENCHMARKS.md, PROVENANCE.md pins |
| T4.4 | README verdict update; systemd unit qualified | after T4.3 | README.md |

## Standing rules

- Before any `gpu-window` task: `docker stop glm53-flash-2x-rtxpro6000`; after: `docker start` it and confirm `/health`.
- Every JSON promoted from `results/` to `evidence/` goes through `scripts/ci.sh`'s identifier scan.
- No pushes from automation. Commits are local until the owner says otherwise.

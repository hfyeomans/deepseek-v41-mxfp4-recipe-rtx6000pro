# Plan: DeepSeek-V4.1-Flash FP4 on RTX PRO 6000 Blackwell

## Goal

Produce a revision-pinned, benchmarked recipe for the best DeepSeek FP4 model this host can serve, with honest statistics, and push the 2-GPU boundary as far as it measurably goes. "Best" means: V4.1-Flash if it can be made to run; otherwise the qualified V4-Flash-0731 NVFP4 baseline, improved.

## Scope decision (Phase 0 result)

The memory math in [docs/FEASIBILITY.md](../docs/FEASIBILITY.md) rules out V4.1-Flash on two RTX PRO 6000 with any existing engine (shortfall ≈ 105–112 GiB). The plan therefore has three tracks that run in parallel and share one benchmark harness:

| Track | Outcome | Status |
|---|---|---|
| **Baseline** | V4-Flash-0731 NVFP4 TP2 re-measured with this repo's harness | scripts ready; needs a GPU window |
| **4-GPU profile** | SGLang `dev-dsv41` + Engram-in-DRAM profile for V4.1-Flash, ready to qualify when two more cards are installed | `serve.sh` written, labelled unqualified |
| **Boundary** | Measured ceilings for expert streaming (Option B) and a feasibility study of expert pruning (Option D) on 2 GPUs | microbenchmarks runnable now |

## Phases

```diagram
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ P0 Research & data │──▶│ P1 Host micro-     │──▶│ P2 Baseline        │──▶│ P3 Boundary        │──▶│ P4 4-GPU           │
│ checkpoint budget, │   │ benchmarks (no     │   │ V4-Flash NVFP4 TP2 │   │ experiments        │   │ qualification      │
│ hardware, sources, │   │ model): DRAM, NVMe,│   │ decode/prefill/    │   │ expert streaming   │   │ (needs 2 more      │
│ feasibility math   │   │ PCIe H2D, hugepages│   │ 1M needle          │   │ prototype, pruning │   │ cards)             │
│ DONE               │   │ runnable today     │   │ needs GPU window   │   │ study              │   │ blocked on hardware│
└────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘   └────────────────────┘
```

### P0 — Research and data (done)

- Exact byte budget of the three V4.1-Flash checkpoints from shard headers ([`evidence/budget-*.json`](../evidence/)).
- Hardware inventory ([`evidence/hardware-inventory.json`](../evidence/hardware-inventory.json)); DIMM part numbers still pending root.
- Feasibility calculator with the 0xSero 4-GPU layout as the calibration point.
- Source audit of the Google Doc: two correct claims, four nonexistent flags, one wrong-model size. See FEASIBILITY.md §6 and [RESEARCH.md](RESEARCH.md).

Exit criterion met: a placement table that says, per layout and GPU count, fits or does not, with margins.

### P1 — Host microbenchmarks (no model weights, no GPU contention except E4)

E1 DRAM bandwidth, E2 GPU memory bandwidth, E3 Engram-style random gather from DRAM and from both NVMe drives, E4 PCIe host-to-device bandwidth per GPU. These replace every *vendor spec* and *estimate* row in HARDWARE.md with a measurement and set the ceilings for Options A and B.

Exit criterion: HARDWARE.md has no "estimate" rows left; FEASIBILITY.md Option B ceiling is recomputed from measured PCIe bandwidth.

### P2 — Baseline

Run this repo's `bench-decode.py`, `bench-prefill.py` and `verify.py` against the already-qualified `deepseek-v4-flash-nvfp4-tp2` deployment so that every later number is directly comparable (same fixture, same method, same power condition).

Exit criterion: `docs/BENCHMARKS.md` baseline tables filled from `results/` JSON produced by this harness, not copied from the other repo.

### P3 — Boundary experiments (the "push" track)

- E5: load V4.1-Flash on 2 GPUs with generic CPU offload in SGLang/vLLM and document the exact failure mode or throughput. Expected: OOM or unusable. Purpose: evidence, not hope.
- E6: expert-streaming prototype. Measure how many tokens/s a 39%-offloaded expert set can sustain given E4's bandwidth, first as a synthetic torch loop (no model), then, if warranted, as an SGLang patch.
- E7: expert pruning feasibility. Collect routing statistics from the V4-Flash baseline (same MoE family) to estimate how non-uniform V4.1 routing is; decide whether a 384→232 pruned checkpoint is worth building.

Exit criterion: each experiment has a number and a one-paragraph verdict in BENCHMARKS.md, including negative results.

### P4 — 4-GPU qualification (blocked on hardware)

Install two more RTX PRO 6000; run `preflight.sh` (which will then pass), `serve.sh` with the SGLang `dev-dsv41` profile and Engram in DRAM, then the full verify + benchmark suite. Compare against 0xSero's published 4-GPU numbers and against the P2 baseline.

Exit criterion: README verdict flips from "does not fit" to a qualified profile with revision pins and measured tables.

## Innovation candidates (ranked by measurability)

1. **Engram fully in DRAM on a 256 GB host** (Option A variant): nobody has published a consumer-GPU V4.1 result with the whole FP8 Engram table resident in DRAM instead of NVMe-backed. This host can. Measurable as prefill tok/s and TTFT delta versus NVMe mode on the same 4-GPU box.
2. **MoE-aware expert streaming on SM120** (Option B): a real engineering gap in both engines. Ceiling ~60 tok/s decode is the honest expectation; the prototype's value is the measurement.
3. **Expert pruning for 2-GPU fit** (Option D): research-grade; only worth building if E7 shows routing skew.
4. **Baseline improvements** that need no new hardware: DSpark block size sweep, FP8 KV, `--enable-decoder-swa-bounded-replay` equivalents on V4-Flash, concurrency scaling to C8/C16 on the existing recipe.

## Non-goals

- Shipping a V4.1-Flash 2-GPU "recipe" that has not run. `preflight.sh` fails closed.
- Quantizing weights ourselves. Experts are already 4-bit; sub-4-bit has no SM120 kernels.
- Publishing any host identifiers. `scripts/ci.sh` scans for UUIDs, PCI IDs, home paths and LAN addresses.

Detailed experiments: [EXPERIMENTS.md](EXPERIMENTS.md). Task list with owners and status: [TASKS.md](TASKS.md). Sources and open questions: [RESEARCH.md](RESEARCH.md).

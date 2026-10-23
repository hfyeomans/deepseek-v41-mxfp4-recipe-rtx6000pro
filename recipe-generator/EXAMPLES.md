# EXAMPLES.md — the three worked examples

Each example is a budget JSON + hardware JSON pair run through
[`placement_calc.py`](../placement_calc.py). Regenerate with [`run-examples.sh`](run-examples.sh) (GPU-free:
headers only, no serve). The text files next to the inputs are the calculator's verbatim output.

The three cases deliberately exercise the calibration rule at its three trust levels:

| Example | Model | Hardware | Trust label | Why |
|---|---|---|---|---|
| 1 | DeepSeek-V4.1-Flash (475 GiB, **real**) | 2× RTX PRO 6000 + 246.8 GiB host (**real, measured**) | `calibrated` | must reproduce the repo's measured qualified deployment |
| 1b | DeepSeek-V4.1-Flash-**1017** (475 GiB, **real, second checkpoint**) | same real box | `calibrated` | Phase-0 validation: reproduce the measured 1017 deployment and emit a recipe that matches the serving recipe knob-for-knob |
| 2 | synthetic 211 GiB V4-class MoE (**synthetic**) | same real box | `calibrated-model (k carried)` | the owner's case; gpu_ms itself unvalidated |
| 3 | Qwen3-30B-A3B (60.5 GiB bf16, **real model**) | RTX 5090 / RTX 3090 (**published reference numbers**) | `calibrated` | published datapoint as anchor; weaker than local measurement |

---

## Example 1 — the real 475 GiB case (the calculator's own anchor)

Inputs: [`dsv41-flash/budget.json`](dsv41-flash/budget.json) (extends the repo's
`evidence/budget-upstream-bf16fp8.json`, adds the arch section) and
[`dsv41-flash/hardware.json`](dsv41-flash/hardware.json) (measured bandwidths + the qualified 1017
deployment as calibration). Output: [`dsv41-flash/result.txt`](dsv41-flash/result.txt).

What it must reproduce (measured, `evidence/t352/default-flip`): **219/165 experts per layer, el8 map with
15 % CPU share, D=4, chunk 8192, Engram on NVMe, KV 262K×4, C1 60.8, C4 124.0, prefill 4,344 tok/s cold,
TTFT 16K ≈ 3.7 s.** The calculator reproduces all of it from the budget + hardware JSONs alone — that
reproduction is what makes its projections for *new* rows trustworthy, and the self-test pins it.

Reading the plan: 268.9 GiB of routed experts cannot fit 176.5 GiB of usable VRAM, so the engine keeps 219
per layer resident and computes the other 165 on the CPU tier from DRAM; the fitted hot-expert map holds the
CPU route share to 15 % (checkpoint order would be 43 %); D=4 hides the CPU tier behind the GPU step; the
188.8 GiB Engram tables go to NVMe behind the page cache because DRAM cannot hold them beside the CPU
experts (−78 GiB); the cold-Engram chunk-0 tax (1.7–3.1 s) is reported separately.

## Example 1b — the 1017 checkpoint (Phase-0 validation of the whole generator)

Inputs: [`dsv41-1017/budget.json`](dsv41-1017/budget.json) (extends the upstream budget; the 1017
component bytes are identical to the pinned checkpoint — re-verified by `scripts/checkpoint-budget.py`
on the local snapshot) and [`dsv41-1017/hardware.json`](dsv41-1017/hardware.json) (calibration =
`evidence/t1017`: C1 61.91/55.88, C4 123.58/113.99, prefill 3,095/3,372 cold and 4,504/4,613 warm,
14.4 % CPU share). Output: [`dsv41-1017/result.txt`](dsv41-1017/result.txt).

This is the generator's second real checkpoint — the proof that the pipeline generalizes past the single
checkpoint it was built on. Three results: (1) the plan reproduces the measured 1017 deployment
(219/165, D=4, chunk 8192, C1 61.9, C4 123.6, prefill 3,095 cold, TTFT 16K 5.4 s vs measured 5.18 s);
(2) the emitted recipe's profile matches the serving recipe's `qualified.env` **knob-for-knob with zero
conflicts** — including `KT_EXPERT_MAP`, now emitted with the map file copied in (`--expert-map`);
(3) the only knob left host-side is `CUDA_GRAPH_MAX_BS`, a registry default. This is the Phase-0 exit
criterion of PROJECT-PLAN.md, pinned as self-test #9.

## Example 2 — the owner's case: a ~211 GiB model on 192 GiB of VRAM (SYNTHETIC)

Inputs: [`synthetic-211g/budget.json`](synthetic-211g/budget.json) — a scaled-down same-architecture cousin
of V4.1-Flash (133.5 GiB experts, 60 GiB Engram, everything else as V4.1). **Every model number is synthetic
and labeled; the hardware numbers are the real box's measurements.** Output:
[`synthetic-211g/result.txt`](synthetic-211g/result.txt).

What the plan shows — exactly the owner's intuition, quantified:

- **100 % of routed experts resident in VRAM** (384/384 per layer). No CPU tier at all: no deferral, no
  dispatch, `cpu_share 0 %`.
- **Engram tables move to DRAM** (60 GiB fits beside the OS reserve with 162.8 GiB host margin) — no NVMe
  tier needed.
- **Projected C1 73.8 tok/s** (vs 60.8 measured today on the 475 GiB model — the 475 GiB model's CPU tier is
  the difference), C4 150.6, prefill 5,270 tok/s cold, TTFT 16K 3.2 s.
- Margins: +24.6 GiB VRAM, +162.8 GiB host. The extra VRAM is *spendable*: longer context ladder, or Q6/FP8
  dense parts, or keeping DSpark resident.

Two honesty labels to read carefully: `gpu_ms_per_token: 13.55` is a *synthetic estimate* (scaled from the
measured 16.45 ms by the resident-work mix), and the k-factors are *carried* from example 1 (same host, same
engine family). Trust is `calibrated-model (k carried from another example)` — the plan is a strong prior for
a real 211 GiB model, and the **first serve of the real model is the anchor** that promotes it to
`calibrated`. That is the calibration rule working as designed: the calculator never pretends.

## Example 3 — one small model: Qwen3-30B-A3B on one card

Inputs: [`qwen3-30b-a3b/budget.json`](qwen3-30b-a3b/budget.json) (totals published; per-component split
derived from the published config.json and labeled; regenerate with `scripts/checkpoint-budget.py` before
trusting), [`hardware-rtx5090.json`](qwen3-30b-a3b/hardware-rtx5090.json) and
[`hardware-rtx3090.json`](qwen3-30b-a3b/hardware-rtx3090.json) (published Hardware Corner llama.cpp numbers
as the calibration anchors — weaker than local measurements; the methodology re-measures on your host in
P1/P2).

What the plans show:

- **RTX 5090 (32 GiB):** Q4_K_M weights 17.8 GiB + 32K KV 3.0 GiB fit with 7.8 GiB margin; all 128
  experts/layer resident; projected C1 226.1 tok/s and prefill 7,093 tok/s — reproducing the published
  anchor by construction (gpu_ms inverted from it). The k_prefill of 0.35 exposes the √chunk law's error on
  llama.cpp — a documented model gap, not a hidden one.
- **RTX 3090 (24 GiB):** same plan at 4K context; C1 153.6 / prefill 2,989 (published anchors); the 24 GiB
  card is KV-bound well before the 32K native context — the plan says so in its margins, which matches the
  published recipe's "keep context modest (4K–16K)".
- Contrast with example 1: a 60.5 GiB model on one 24 GiB card needs *no* tiering tricks — quantization
  (×0.307) does the whole job. The generator's job here is to prove "fits, with margins" and to stop the
  owner from over-engineering (no CPU tier, no Engram, no deferral — the plan emits none of them).

## What the three together demonstrate

1. **The engine generalizes across scales**: 475 GiB hybrid → 211 GiB VRAM-rich → 60 GiB single-card, one
   code path, no per-model special cases.
2. **The calibration rule does real work**: each example wears its trust label; only example 1's anchor is
   the repo's own measurement; the published anchors in example 3 are explicitly weaker; example 2 is honest
   about being synthetic.
3. **The lever settings come out, not in**: expert split, map, D, chunk, KV shape, Engram tier and prefetch
   mode are emitted per plan — the tuning mode (LEVERS.md) then lets the owner move any of them and watch
   the projections re-light.

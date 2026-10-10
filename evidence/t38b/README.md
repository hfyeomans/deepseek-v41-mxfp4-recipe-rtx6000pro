# T3.8b — per-layer expert remap at load (`--kt-expert-map`, el8 mix map)

Question: does putting each layer's hot experts (the el8 mix map from T3.8, 219 per layer) on
the GPU instead of the positional ids 0–218 cut the CPU tier's work and lift decode, at no cost
to prefill, memory, long context or quality?

Mechanism (overlay `c777da2d`, `python/sglang/srt/layers/moe/kt_remap.py`): the model loader
permutes each MoE layer's experts so the map's GPU set takes physical ids 0..218 (expert weight
names, the gate's logit rows and both correction biases move together); the compact kt loader
reads the 165 CPU experts in the same permuted order. Routing is an exact permutation (same
scores, same experts chosen, up to top-k tie order); what changes is which tier computes a given
expert, so outputs are not bit-identical (kt AMX vs FlashInfer CUTLASS rounding) — the NLL gate
is "inside the rounding band", not "0 moved". Map: `config/expert-map-el8-mix-219.json`
(40 × 219 ids, 57.4 % overlap with positional). Launcher knob `KT_EXPERT_MAP=<host path>`
(empty = positional, the prior behaviour; the overlay is byte-identical in behaviour with the
flag unset).

Profile: launcher defaults (219/165, D=4, chunk 8192, 64K × 4, `KT_CPUINFER=16`, `ahead`).
Three serves, 2026-10-10, one driver (`qual.sh`, console log `qual-log.txt`):

| serve | dir | what |
|---|---|---|
| 1 | `qual-el8-mix/` | map on: the T3.39 qualification set (16K / 32K prefill with `--warm-pass`, C1 ×6, C4 ×3, NLL corpus vs `t328/default-219-8192`, same-prompt C4, 4 × 62K stress + needles, verify 65,536) plus `scripts/kt-cpu-per-step.sh` |
| 2 | `histogram-el8-mix/` | map on + `--expert-distribution-recorder-mode stat`: the four T3.8 round-2 traffic sets → realised CPU share on the remapped ids (`analysis/routes-remapped.txt`, same `t38/round2/analysis/routes.py`; raw `.pt` dumps gitignored) |
| 3 | `kt-cpu-positional/` | map off, same overlay and box state: `kt-cpu-per-step.sh` + C1, the same-session comparator |

## Results

Comparator = serve 3 (same session) and the `99128a7e` qualification (`t339/qual-99128a7e/`).

| metric | positional (`99128a7e` qual / serve 3) | el8 mix map (serve 1) | change |
|---|---|---|---|
| C1 median tok/s (distinct fixtures, 512 tokens, 6 runs) | 42.41 / 42.15 (wall 40.43 / 40.16) | **57.97** (wall 56.18) | **+37 %** |
| C4 aggregate median (3 runs) | 55.75 / 56.72 (wall 53.63 / 50.69) | **123.91** (wall 118.84) | **+120 %** |
| kt CPU-seconds per decode step, C1 / C2 / C4 (serve 3 vs 1) | 0.428 / 0.725 / 1.246 | 0.313 / 0.429 / 0.697 | −27 / −41 / −44 % |
| CPU-tier share of routes, decode (HumanEval C4, distinct C1/C4) | 42.4 / 42.8 % (T3.8) | **15.1 / 14.7 %** | T3.8 predicted 15.3 / 14.5 |
| CPU-tier share, prefill (their prompts, NLL corpus, random words) | 43.8 / 43.4 / 43.6 / 46.4 % | 12.0 / 14.2 / 15.0 / 23.2 % | predicted 12.2 / 14.3 / 15.1 / 23.4 |
| HumanEval generate, 40 problems, C4 (recorder on, both) | decode median 11.3 tok/s per stream, wall 159 s | 24.2 tok/s, wall 83 s | 5,934 vs 5,911 completion tokens |
| 16K prefill tok/s (seed 777 / warm 1277) | 2,040 / 2,171 | 2,013 / 2,173 | unchanged (−1.3 / +0.1 %) |
| 32K prefill (seed 1277 / warm 1777) | 2,272 / 2,353 | 2,270 / 2,330 | unchanged |
| NLL vs `t328` corpus, 65,653 tokens | 0 moved (same tiers) | ΔNLL −0.00033 ± 0.00101 nats/tok, z +0.33 (cluster-robust z +0.26), worse/better > 0.5: 1,921 / 1,944 | inside the rounding band (chunk 4096 moved z +1.18 with the same symmetric spread) |
| same-prompt C4 consistency, before and after stress | PASS | PASS / PASS | — |
| 4 × 62K stress, wall / min free MiB | 124.0 s / 2,149 · 2,022 | 108.7 s / 2,189 · 2,062 | −12 % wall, memory unchanged |
| needles at 55K, verify 65,536 | 6/6, PASS | 6/6, PASS | — |
| VRAM after capture (MiB) | 92,778 / 92,837 | 92,778 / 92,909 | no VRAM cost |
| health scan (NaN / Xid / Traceback / OOM) | 0 | 0 | — |

Reading: the decode gain is exactly the CPU-tier reduction the counts predicted (2.5 → 0.9
CPU-expert reads per token). kt CPU-seconds/step fall less than the route share (−27 % at C1 vs
−65 % of routes) because the per-step fixed term (export, dispatch, thread wake) does not shrink;
C4 benefits most (−44 % CPU-seconds, 2.2× aggregate) because the CPU tier was the saturated
resource at batch 4. Prefill is unchanged as predicted: the 165 streamed experts ship whole
regardless of which ones they are. The stress wall drop (124 → 109 s) is the decode tail of the
256-token generations at C4.

The realised shares match the T3.8 held-out prediction to within 0.3 points on every set, which
also confirms the remap is wired correctly end to end (the recorder sees physical ids, so
"positional CPU share" on this serve is the live CPU-tier share).

## Caveats

- The map is fitted on el8's published counts (prose / code decode, text prefill) and checked on
  our in-house sets; random-word prefill keeps 23 % on the CPU. Traffic unlike either (other
  languages, long reasoning) should be re-measured with the recorder before trusting the 15 %.
- Not bit-identical to positional (tier rounding); greedy continuations can diverge on near-ties
  exactly as the chunk 4096 and D=4 profiles do.
- One session per side; the C1/C4 numbers come from the standard 6-/3-run benches with the usual
  ±1 tok/s spread, far below the measured gap.

Default flip (`KT_EXPERT_MAP` → the el8 mix map) is the owner's decision.

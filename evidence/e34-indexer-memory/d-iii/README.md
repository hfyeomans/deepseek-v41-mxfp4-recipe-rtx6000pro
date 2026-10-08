# E34d-iii — why chunk 8192 moved NLL by +0.0067 (one serve, 2026-10-08)

Serve A: overlay `d6811728`, **219 / 165 + `CHUNKED_PREFILL_SIZE=8192`**, 64K x 4, D=4, spec off
(`launch-a.txt`, `launch-container-a.txt`; ready with 5.56 / 5.50 GB free, same as E34d-ii).
Twenty-one `nll-score.py` runs, 4-9 s each, no other load. Health 0, EDAC 0/0.
Comparisons are paired per-token logprob deltas over the common token prefix (the script's
`delta_mean_nll` mixes token counts when the prompt is truncated; the numbers below are recomputed
offline over `[0, n-3)` to drop the truncation-boundary tokens). Positive = worse.

**Finding: the shift is deterministic, is a function of the prefill batch's row count M, flips at
M = 4,736 rows (37 x 128), and is not a consistent bias — on three texts the same regime change
gives +0.0067 / -0.0029 / +0.0053 nats/tok (pooled +0.0031 +/- 0.0017 over 14,087 tokens).
Chunk 8192 is not shown to be broken; the single 4,795-token fixture cannot resolve a +0.002 gate
for any config that changes a kernel's M. No overlay bug found; no code changed.**

## 1. Replicate — deterministic, not serve noise

`nll-a-full.*`: full fixture, M = 4,796 prompt tokens: mean NLL **1.49469**, +0.00674 vs t316c
(z -1.89), per-segment 2.04897 / 1.94119 / 1.19791 / 1.05393 / 1.10907 — **bit-identical to
E34d-ii** (`../d-ii/nll.json`). Hypothesis 1 of the d-ii README (one-serve noise) is out.

## 2. Truncation sweep on the fixture — a step at M = 4,736

`--max-chars` truncations of `scripts/fixtures/nll-corpus.txt`, each one prefill batch of M rows on
the chunk-8192 serve, compared with the 219 / 165 chunk-4096 control (`../d-ii/control-219-chunk4096/
nll.json`, whose first batch is M = 4,096) over tokens [0, 3000) — a span that is inside the first
batch in every run:

| prompt tokens M | file | dNLL vs control [0,3000) | z |
|---|---|---|---|
| 3,667 | `nll-a-trunc-3800` | +0.00045 | +0.08 |
| 4,270 | `nll-a-trunc-4300` | -0.00331 | -1.03 |
| 4,460 | `nll-a-trunc-c14000` | +0.00053 | +0.13 |
| 4,644 | `nll-a-trunc-c14500` | +0.00053 | +0.13 |
| 4,733 | `nll-a-trunc-c14800` | +0.00053 | +0.13 |
| **4,737** | `nll-a-trunc-c14810` | **+0.00811** | **+1.87** |
| 4,741 / 4,744 / 4,748 / 4,753 / 4,768 / 4,774 / 4,778 / 4,780 | `nll-a-trunc-c148xx`, `c149xx` | +0.00811 (all identical) | +1.87 |
| 4,796 (full) | `nll-a-full` | +0.00811 | +1.87 |

Tokens 0-2,999 are **bit-identical** across M = 4,460 / 4,644 / 4,733 (call it regime C) and
bit-identical across every M from 4,737 to 4,796 (regime D); M = 3,667 and 4,270 are two further
regimes. So the logprobs of the *first* 3,000 tokens depend on how many rows are in the batch,
through a small number of discrete regimes; the fixture's M = 4,796 lands in the one that reads
+0.008 on these 3,000 tokens. The step is at 4,736 = 37 x 128: the 38th 128-row tile changes some
batch-wide kernel's schedule. It is **not** "M > 4096": 4,270-4,733 are clean.

Candidates checked in the overlay: `_SM120_INDEXER_M_CHUNK = 4096`
(`layers/attention/dsv4/metadata.py`, `indexer.py` ~L975) chunks the paged indexer at 4,096 rows on
SM120 — an M > 4096 path the deployed chunk-4096 config never takes, but a second indexer chunk
cannot alter rows 0-2,999, so it is not the cause. The row-tiled dense indexer is one tile at these
sizes (256 MiB / (M x 4 B) ~ 14K rows). What remains is a batch-wide GEMM or fused-MoE tile /
split-K heuristic (cuBLAS / CUTLASS / FlashInfer MXFP4 MoE at M x topk rows) — **unidentified**;
identifying it needs kernel-selection logging and was not worth a serve for this decision.

## 3. Is regime D a bias? Two more texts, same serve

`text2-learning.txt` (first 17,500 printable chars of `docs/LEARNING.md`) and
`text3-experiments.txt` (E22 section of `plan/EXPERIMENTS.md`), each scored once in regime C
(M just under 4,736) and once in regime D (just over), compared over their common prefix:

| text | M_C / M_D | tokens | dNLL (D - C) | z | mean abs dlogp |
|---|---|---|---|---|---|
| fixture (`c14800` vs `c14950`) | 4,733 / 4,780 | 4,729 | **+0.00666** | +2.36 | 0.096 |
| text2 (`nll-a-text2-C/D`) | 4,680 / 4,853 | 4,676 | **-0.00290** | -1.04 | 0.095 |
| text3 (`nll-a-text3-C/D`) | 4,686 / 4,769 | 4,682 | **+0.00533** | +1.67 | 0.118 |
| **pooled** | | 14,087 | **+0.00305** | +1.79 (se 0.0017) | |

Signs disagree; the pooled estimate is +0.003 +/- 0.0017 — compatible with zero and with a small
bias, not with the +0.0067 the fixture alone suggested. The per-token jitter (mean |dlogp| ~ 0.10)
is the same size as every other path change on this stack (streamed vs CPU experts 0.125, D=4
reruns 0.13): with MXFP8 / MXFP4 block-scaled activations, any change in accumulation order flips
quantisation bins and the network amplifies it to ~0.1 nats per token, zero-mean to first order.
On one 4.8K-token text that jitter puts the standard error of the mean at ~0.0035 nats/tok, i.e.
**the +0.002 gate is below the fixture's resolution for any config that changes a kernel's M**;
it only works as a "0 moved" identity check for same-shape configs (placement, refactors).

## Conclusions

1. E34d-ii's NLL failure is a 2-sigma draw from M-regime-dependent numerics, reproducible to the
   bit, not evidence of a defect in chunk 8192 or the streamed-prefill path. Nothing to fix in the
   overlay from this experiment.
2. Chunk 8192 is neither quality-rejected nor quality-proven. Pooled over three texts the regime
   change is +0.003 +/- 0.0017; a decision at the +0.002 level needs ~4x the tokens (se 0.0017 at
   14K; ~60K tokens for se 0.001).
3. The NLL gate needs a shape-aware statement: "0 moved" for same-shape configs; for configs that
   change M (chunk size, batch composition, streamed threshold) a pooled multi-text estimate with
   its standard error, and a gate on the pooled mean. Registered as a methodology task (see
   TASKS T3.26 / T3.19 evidence-manifest work).
4. The deployed default (226 / 158, chunk 4096, M <= 4,096 per batch) is unaffected: it never
   crosses the 4,736-row step.

Files: `launch-a.txt`, `launch-container-a.txt`, `nll-a-full.*`, `nll-a-trunc-*.{json,txt}`
(fixture truncations; `prompt_tokens` in each JSON is M), `text2-learning.txt`,
`text3-experiments.txt`, `nll-a-text{2,3}-{C,D}.json`.

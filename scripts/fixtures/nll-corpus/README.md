# NLL corpus (T3.28 shape-aware gate)

Frozen texts scored by `scripts/nll-score.py --corpus scripts/fixtures/nll-corpus`.
Do not edit these files: every baseline JSON under `evidence/` is tied to their
exact bytes. Add a text as a new numbered file and re-baseline instead.

Why a corpus: one 4.8K-token fixture gives a per-token Δlogp standard error of
≈0.0035 nats under the ~0.1-nat zero-mean jitter that block-scaled FP8/FP4
activations show whenever a kernel schedule changes (E34d-iii), so the
+0.002 nats/token gate could only ever act as a same-shape identity check.
Pooling ≈60K tokens brings the se to ≈0.001 and lets a shape-changing config
(different prefill chunk, expert placement, GEMM tile) be judged on its mean.

Texts are in-house snapshots (same licence as the repo; no third-party prose).
Markdown runs at ≈3.1–3.4 chars/token; the varying lengths leave different
remainders after each prefill chunk, so several batch shapes are sampled.

| file | source (snapshot 2026-10-08) | bytes |
|---|---|---|
| `01-readme-early.txt` | byte-identical copy of `../nll-corpus.txt`, the historic single fixture (early README) | 15,140 |
| `02-learning.txt` | `docs/LEARNING.md` from the top, ~33K chars | 33,693 |
| `03-experiments.txt` | `plan/EXPERIMENTS.md` from byte ~120,000, ~33K chars | 33,823 |
| `04-benchmarks.txt` | `docs/BENCHMARKS.md` from byte ~40,000, ~33K chars | 33,706 |
| `05-troubleshooting.txt` | `docs/TROUBLESHOOTING.md` from the top, ~33K chars | 33,771 |
| `06-bios-guide.txt` | `docs/BIOS-OPTIMIZATION-GUIDE.md` from byte ~30,000, ~33K chars | 34,019 |
| `07-serve-hybrid-sh.txt` | `scripts/serve-hybrid.sh`, whole file (shell) | 16,856 |
| `08-bench-decode-py.txt` | `scripts/bench-decode.py`, whole file (Python) | 9,344 |

Excerpts were cut at line boundaries. Token counts per text are recorded in
each baseline's `pooled.json` (`texts[].tokens_scored`).

Gate statement (see `docs/BENCHMARKS.md`):

- same kernel shapes (same placement, chunk, kernels): `pooled.compare.tokens_changed == 0`;
- shape-changing: `pooled.compare.delta_mean_nll <= +0.002` nats/token, reported with `se` and `z_paired`.

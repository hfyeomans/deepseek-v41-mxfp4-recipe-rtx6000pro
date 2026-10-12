# T3.41 — decode-shape block-32 FP8 GEMV (E44), 2026-10-11/12

Question: the seven dense W8A8 `[32, 32]`-block-scaled projections of a TP2 rank run at decode
M ≤ 4 through SGLang's Triton tensor-core tiles (E20 configs) at 428–748 GB/s against a 1,470 GB/s
GDDR7 roof, ≈ 20 % of the 17.3 ms C1 step (T3.27 window 1). Does a plain FMA row-dot GEMV pay?

Answer: per call yes, in the served step no (+1–2 % C1, 0 at C4, inside the band). The kernel
ships opt-in (`SGLANG_DSV41_FP8_GEMV`, default 0, overlay `d00e8e30` = patch 0060). The control
serve found the real C4 lever instead — see `serve-stock/NOTES.md` and T3.52.

## `sweep/` — L2-cold, in-graph sweep on GPU 1 (`kernels/t341/fp8_gemv_block32.py`)

423 GEMV configs × 7 shapes × M = 1–4, median of 30 CUDA-graph replays per launch, every candidate
checked against the FP32 dequantised reference (rtol = atol = 1e-2, 2 seeds); acceptance = faster
than the E20 Triton winner by > 2 % **and** by more than the p10–p90 spread (T3.27 rule).
`full.txt` is the complete log, `results.json` the rows, `gemv_configs.json` the table the overlay
reads (copied into its `configs/gemv,device_name=…,dtype=fp8_w8a8,block_shape=[32, 32].json`).

| N × K | M=1 triton → gemv µs | M=2 | M=3 | M=4 |
|---|---|---|---|---|
| 16384 × 1280 | 26.87 → 14.49 (+46 %) | 25.87 → 14.84 (+43 %) | 26.28 → 15.32 (+42 %) | 24.80 → 16.29 (+34 %) |
| 25600 × 6144 | 201.5 → 99.5 (+51 %) | 194.3 → 101.3 (+48 %) | 195.2 → 104.0 (+47 %) | 199.4 → 109.1 (+45 %) |
| 5120 × 4096 | 20.47 → 14.69 (+28 %) | 20.74 → 15.23 (+27 %) | 20.87 → 16.16 (+23 %) | 20.76 → 17.20 (+17 %) |
| 4096 × 1280 | 6.72 → 4.79 (+29 %) | 6.64 → 5.48 (+17 %) | 6.79 → 6.31 (+7 %) | reject (−3 %) |
| 5120 × 1152 | 7.10 → 5.27 (+26 %) | 6.91 → 5.84 (+15 %) | reject (+1 %) | reject (−4 %) |
| 2304 × 5120 | 11.62 → 9.81 (+16 %) | 11.42 → 10.18 (+11 %) | 11.72 → 10.79 (+8 %) | reject (−2 %) |
| 1792 × 5120 | 9.31 → 8.14 (+13 %) | 9.36 → 8.48 (+10 %) | 9.40 → 9.10 (+3 %) | reject (−9 %) |

23/28 accepted; the small shapes sit on a launch floor and lose their edge at M = 4, where the
tensor-core tile amortises. Effective bandwidth of the winners 1.1–1.45 TB/s.

## `serve-gemv/` — launcher default + `SGLANG_DSV41_FP8_GEMV=1` (`run.sh`, `rerun-log.txt`)

C1 60.70 (6 runs) then 58.54; C2 86.79; C3 100.56 / 99.06; C4 108.24 (one run 123.6) / 107.77.
NLL vs `t328` and vs the identity anchor: 0 of 65,653 moved — expected, the GEMV never runs in
prefill so the corpus cannot see it. Same-prompt C4 PASS (`consistency.txt`). Decode numerics
(`spec-consistency.txt`, `--flush-cache --compare evidence/t39hb/a2-spec-off/consistency-cold.json`):
argmax agreement 1.000 / 1.000 / 0.964 / 0.974 vs the spec-off baseline 1.000 / 1.000 / 0.953 / 0.974,
flips > 0.5 nats 0 / 0 / 1 / 2 vs 0 / 0 / 1 / 1, same max |Δlogp| magnitude → clean.
`gemv-table-log.txt` shows the table loading on both ranks; `vram-after-capture.txt` unchanged.

## `serve-stock/` — `SGLANG_DSV41_FP8_GEMV=0` control, then the thread-affinity probe

C4 106.04 → 123.55 → 123.63 → 122.54, C1 59.48 — bimodal exactly like serve A, so the GEMV is not
the C4 lever. Pinning every non-kt thread to the SMT siblings: C4 125.05 / 125.29, C1 60.38.
Full story and the prefill A/B/A/B in `serve-stock/NOTES.md`.

Decision: keep `SGLANG_DSV41_FP8_GEMV=0`; flip only if a pinned, paired A/B reads ≥ +2 % C1.

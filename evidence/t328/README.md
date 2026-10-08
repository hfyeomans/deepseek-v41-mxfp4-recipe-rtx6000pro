# T3.28 — shape-aware NLL gate and the chunk-8192 adoption decision (2026-10-08)

Overlay `d6811728` (unchanged), image `dsv41-kt:recipe`, BIOS slot 3 (FCLK 2100), 300 W,
`CONTEXT_LENGTH=65536 MAX_RUNNING_REQUESTS=4 KT_NAN_PROBE_MIN_TOKENS=1`, D=4, spec off.
Corpus: `scripts/fixtures/nll-corpus/` (8 in-house texts, 65,653 scored tokens; see its README).

## Why

E34d-iii showed that one 4.8K-token fixture resolves a mean NLL shift only to se ≈ 0.0035
nats/token under the ~0.1-nat zero-mean jitter that block-scaled MXFP8/MXFP4 activations
produce whenever a kernel's M changes (prefill chunk, expert placement, tile). The +0.002 gate
was therefore only usable as a same-shape identity check. E34d-ii (219/165 + chunk 8192) read
+0.0067 on that fixture and could be neither accepted nor rejected.

## Serves and results

| dir | config | result |
|---|---|---|
| `baseline-226-4096/` | 226/158, chunk 4096 — the pre-change default | `identity-vs-t316c.json`: 0 / 4,795 tokens changed vs `evidence/t316-review-intake/nll-t316c.json` (same-shape identity holds). Corpus baseline: 65,653 tokens, token-weighted mean NLL 1.57616. |
| `baseline-226-4096-replicate/` | same serve, scored again | pooled Δ −0.00000, **0 / 65,653 tokens changed** (deterministic; identity floor for same-shape configs). |
| `cand-219-8192/` | 219/165, chunk 8192 (`KT_NUM_GPU_EXPERTS=219 CHUNKED_PREFILL_SIZE=8192 ALLOW_UNTESTED_FIT=1`) | **pooled ΔNLL +0.00075 ± 0.00098 nats/token (z −0.77; cluster-robust se 0.00165, z −0.45) over 65,653 tokens; 65,226 tokens changed; >0.5-nat worse/better 1,891 / 1,846.** Per text: +0.0067, −0.0030, −0.0069, +0.0024, +0.0047, +0.0032, +0.0018, +0.0024 (se 0.0024–0.0036 each). **Passes the ≤ +0.002 gate.** |
| `default-219-8192/` | plain `bash scripts/serve-hybrid.sh` after the default change (no overrides) | corpus **0 / 65,653 tokens changed vs `cand-219-8192/`** (the default reproduces the candidate bit-for-bit); needles 6/6 at 55,206 tokens (TTFT 34.7 s); prefill 16K 1,504 tok/s (TTFT 10.7 s); C1 41.34 summed / 39.39 wall tok/s; health 0; EDAC 0/0. This corpus is the new-default baseline for later gates. |
| `d0-219-8192/` | default + `KT_MAX_DEFERRED=0` | **0 / 65,653 tokens changed vs `default-219-8192/`.** The corpus is scored in chunks ≥ `KT_GPU_PREFILL_MIN_TOKENS` (1024), which take the streamed GPU prefill path; deferral only exists in kt-kernel's CPU `_prepare_forward_cpu_buffers`, so the gate was blind to D. The T3.15 D readings came from the 699-token tail of the 4,795-token fixture at chunk 4096. |
| `cpu-prefill-d4-219-8192/` | default + `KT_GPU_PREFILL_MIN_TOKENS=0` (all prefill through the CPU expert path), D=4 | vs `default-219-8192/`: pooled −0.00025 ± 0.00104 (cluster se 0.00132) — E22 streamed GPU prefill is NLL-neutral against the CPU path. |
| `cpu-prefill-d0-219-8192/` | same, `KT_MAX_DEFERRED=0` | vs `cpu-prefill-d4-219-8192/`: pooled −0.00066 ± 0.00103 (cluster se 0.00225; per-text −0.0116 … +0.0074), i.e. **D=4 costs +0.0007 ± 0.001 nats/token** on the only path where deferral acts — within the gate (≤ +0.005 at 95 %, cluster-robust). |

Per-text means scatter more than iid token noise predicts (Σz² ≈ 19 on 8 texts), so
`pooled.json` also reports a cluster-robust se with one cluster per text; the gate is on the
pooled point estimate, reported with both se values. The 95 % upper bound on the candidate's
shift is +0.004 nats/token (cluster-robust); halving that needs ~4× the corpus.

The near-balanced worse/better counts (1,891 / 1,846) and the sign-mixed per-text means are the
signature of accumulation-order jitter, not of a systematic degradation: a real −0.0067 regression
on text 01 would not come with −0.0069 on text 03 under the same kernels.

## Decision

219/165 + `CHUNKED_PREFILL_SIZE=8192` is adopted as the `serve-hybrid.sh` default (owner asked for
chunk 8192 if it qualified). Performance was measured in E34d-ii on this exact config
(`evidence/e34-indexer-memory/d-ii/`): prefill 1,438 / 1,526 tok/s at 16K / 32K (+38 / +47 %),
C1 41.39 summed / 39.45 wall tok/s (−1.9 %), C4 56.32 / 54.16, 4×62K stress min free
2,339 / 1,302 MiB, needles 6/6, health 0. The 128K / 262K / 512K × 1 ladder rungs were qualified at
226/4096 only; the launcher now requires those two values above 64K until they are re-qualified.

## Files

- `*/0N-*.json` — per-text `nll-score.py` output (token ids, logprobs, segments, `compare` block).
- `*/pooled.json` — per-text table plus pooled statistics (`compare.delta_mean_nll`, `se`, `z_paired`, `se_cluster`, `z_cluster`).
- `*/launch-container.txt` — container log for the serve (health grep: 0 hits each).

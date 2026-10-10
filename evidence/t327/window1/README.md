# T3.27 window 1 — entry-condition timeline on the qualified profile (2026-10-10)

One serve of the qualified profile (overlay `124ca1de`, image `dsv41-kt:recipe`, launcher defaults
64K × 4 / chunk 8192 / `KT_CPUINFER=16` / el8 mix map) with `KT_TIMELINE=1` (T3.19a's in-graph
per-layer buckets, tensor dumps off) under the standard C1 and C4 decode recipes, then one
torch-profiler trace (CPU+GPU, `num_steps=8`) of a C1 decode and one of a C4 decode (E20's method,
`scripts/step-profile.py`). Driver: `w1.sh`; log: `w1-log.txt`. Question: what share of the
*current* decode step is the dense W8A8 FP8 GEMMs (T3.27 target 1) and the resident-expert MXFP4
grouped GEMM (target 3), i.e. does T3.27's entry condition (≥ 10 % of the targeted path) hold after
T3.8b (el8 map) and T3.39 (streamed prefill) — and is there anything left for a launch-table sweep.

## Where the C1 / C4 decode step goes now (rank 0, `candidate_all` graphs)

T3.19a (2026-09-24, 226 / 158 positional, D=4) beside this serve (219 / 165 el8 map, D=4):

| bucket | T3.19a bs 1 | **now bs 1** | T3.19a bs 4 | **now bs 4** |
|---|---:|---:|---:|---:|
| export (D2H + kt submit host node, 40 layers) | 2.482 (10.5 %) | **2.513 (14.6 %)** | 2.890 (4.2 %) | **2.859 (8.6 %)** |
| gpu_experts (resident grouped GEMM) | 2.517 (10.7 %) | **2.829 (16.4 %)** | 5.111 (7.4 %) | **6.380 (19.2 %)** |
| cpu_wait (exposed kt wait) | 7.879 (33.4 %) | **2.188 (12.7 %)** | 48.588 (70.8 %) | **13.093 (39.3 %)** |
| h2d + combine | 0.459 (2.0 %) | 0.466 (2.7 %) | 0.494 (0.7 %) | 0.495 (1.5 %) |
| engram_break (2 gaps) | 1.822 (7.7 %) | **0.770 (4.5 %)** | 2.915 (4.2 %) | **1.483 (4.5 %)** |
| other_gpu (attention, dense, norms, all-reduce, lm head) | 8.073 (34.2 %) | **8.130 (47.1 %)** | 8.336 (12.1 %) | **8.405 (25.2 %)** |
| between_replays | 0.269 | 0.082 | 0.209 | 0.428 |
| **step (unsampled replay period)** | **23.589** | **17.255** | **68.659** | **33.305** |
| buckets / period | 98.9 % | 99.5 % | 99.7 % | 98.7 % |
| bench-decode wall tok/s, same serve | 39.15 | **53.96** (median 55.65) | 55.56 | **111.07** (median 115.63) |

Rank 1 (no kt work): `other_gpu` 13.36 ms of 17.18 at bs 1, 24.95 of 32.89 at bs 4 — still the
all-reduce wait for rank 0's CPU tier (T3.22's target), unchanged in kind.

What moved since T3.19a: `cpu_wait` −5.7 ms at bs 1 and −35.5 ms at bs 4 (the el8 map, T3.8b);
`engram_break` −1.05 / −1.43 ms (T3.20 ahead-of-use fetch). What did not move: `other_gpu`
(8.07 → 8.13 / 8.34 → 8.41 ms), `export` (2.48 → 2.51), `gpu_experts` (+0.31 ms at bs 1, +1.27 at
bs 4 — 219 instead of 226 resident experts but the per-count runner of window 1b and a different
tactic pick; noted, not chased). The step is 17.3 ms; the GPU-side buckets other than the CPU tier
(`other_gpu` + `gpu_experts` + `engram_break`) are 11.7 ms = 68 % of it.

The instrumented serve runs ≈ 3 % under the qualified numbers (C1 57.61 / 55.79 → 55.65 / 53.96,
C4 121.03 / 116.18 → 115.63 / 111.07): ≈ 240 event records per step plus a sync every 8th replay,
as in T3.19a. VRAM after capture 92,396 / 92,467 MiB (qualified 92,396 / 92,463).

## Splitting `other_gpu`: the torch-profiler traces

`prof-c1/step-profile-tp0.txt` (rank 0, 8 profiled steps, ms per step, GPU busy 17.05):
`moe_expert_gpu` 6.46 (includes the prompt's prefill MoE: ten 390 µs grouped-GEMM calls per
"step" are the prefill chunk amortised over the 8 steps), **`dense_fp8_gemm` 3.62** (`_w8a8_block_fp8_matmul_hopper`
161 calls / step at 13.0 µs = 2.10 ms, `_w8a8_block_fp8_matmul` 42 calls at 36.2 µs = 1.52 ms,
`_reduce_block_fp8_split_k` 0.25), `elementwise_index` 1.62, `bf16_gemm` 1.51 (prefill lm head /
indexer, 1 call), `attention_indexer` 1.34 (`sparse_mla_decode_dsv4` 0.71 + merge, `_hc_mix_stats`
0.78 + `_hc_mix_reduce_sinkhorn` 0.48 are the indexer), `collective` 0.69, `other` 2.07. The E20
M = 1 table gives the decode-only dense sum independently: the five per-layer calls are the
12.3–30.7 µs shapes (four of them in the 12–16 µs `_hopper` class, one in the 31 µs class), ≈ 86 µs
per layer × 40 = 3.4 ms, plus the per-step calls (the 217 µs N = 25600 × K = 6144 shape among them);
so **dense FP8 GEMMs ≈ 3.4–3.6 ms = 20 % of the 17.3 ms C1 step**, and the resident grouped GEMM
2.83 ms = 16 %. `prof-c4/step-profile-tp0-vs-c1.txt`: at C4 dense 4.55 ms (the M = 4 configs),
MoE 17.49 (prefill of four prompts amortised), collective 1.66.

The trace counts include the profiled requests' prefill forward (the bench's prompts), which is why
`moe_expert_gpu` and `bf16_gemm` exceed the timeline's decode-only buckets; the dense GEMM count per
step (203 calls = 40 × 5 + 3) matches the decode shape list, so its ms / step is the decode figure
to within the prefill chunk's share (E20's trace had the same structure: 3.40 ms).

## Verdict on the entry condition and on the sweep

Both targets clear the 10 % bar on paper: dense FP8 GEMMs ≈ 20 % and the resident grouped GEMM
≈ 16 % of the C1 step. **But the launch-table space T3.27 would sweep is already swept:**

* **Target 1 (dense W8A8 block-32 FP8).** E20 (`kernels/e20/tune_w8a8_block32.py`,
  `evidence/e20-dense-fp8-gemm/merged/results.json`) tried **480 configs per shape**
  (`BLOCK_M/N/K`, `GROUP_M`, warps, stages, `SWAP_AB`, `SPLIT_K`) at **14 M buckets (1 … 4096)
  for all seven live shapes**, 0 configs rejected numerically, and shipped the per-bucket winners in
  the overlay's config JSON (the SGLang kernel picks the nearest-M entry). The winners run the
  weights at 428–748 GB/s at M = 1–4 — 29–51 % of the measured 1,470 GB/s GDDR7 roof (E2). A
  bit-exact re-sweep over the same Triton space can only re-select among configs E20 already timed;
  its expected gain is inside the 1–2 % run spread. Not covered by E20: the M = 8192 bucket (the
  chunk; falls back to the 4096 config) — but prefill dense GEMM is 4 % of the prefill period
  (T3.21), so ≤ +2 % prefill even at 2×.
* **Target 3 (resident MXFP4 grouped GEMM).** FlashInfer's own autotuner already runs at every
  launch for this path — `[AutoTuner]: Tuning trtllm::fused_moe::gemm1 3/3`, `gemm2 3/3`,
  `sparse_mla_sm120_decode_dsv4 6/6` in `launch.txt` — and SGLang's tactic cache keys it by
  model / backend / TP / arch. The tactic space is three candidates per GEMM; the choice is already
  made by measurement on this GPU.
* **Target 2 (T3.21 large-M kernel)** was closed on its stop clause.

What is *not* exhausted is a different kernel, not a different launch table: a block-scaled FP8
GEMV for M ≤ 4 that reaches ≥ 1.2 TB/s would take the dense bucket from ≈ 3.5 to ≈ 2.0 ms —
**≤ +9 % C1 ceiling (1.5 of 17.3 ms), realistically +4–6 %**, and nothing at C4 or in prefill.
That is CUDA kernel work of the kind the owner set aside (T3.21 close, the 2026-10-10 kernel
research), registered as a low-priority candidate (T3.41), not started.

**Recommendation: close T3.27 on its stop clause** — the method's remaining value (recorder,
bit-exact gate, e2e gate, invalidation hash) is the right *way* to run any future sweep, but there
is no sweep left to run on this stack that could clear the 2 % + spread acceptance bar. This
timeline replaces T3.19a as the "where the decode step goes" baseline.

## Files

| file | content |
|---|---|
| `w1.sh`, `w1-log.txt` | driver and its log |
| `launch.txt`, `ready.txt`, `manifest.json`, `vram-after-capture.txt` | serve record (image / overlay / env), readiness (TTFT 18.47 s cold), VRAM after capture |
| `c1.json/.txt`, `c4.json/.txt` | C1 (6 × 512) and C4 (3 × 512) recipes with the timeline on |
| `timeline/timeline-rank{0,1}.jsonl`, `timeline/report.{txt,md,json}` | raw sampled records (255 steps at bs 1, 216 at bs 4 per rank) and `scripts/kt-timeline-report.py` output |
| `prof-c1/`, `prof-c4/` | torch-profiler traces (both ranks, 8 steps) + `step-profile-tp0*.txt/.json` |
| `prof-c1-bench.txt`, `prof-c4-bench.txt` | the profiled requests (profiler overhead: C1 47.28, C4 105.03 — not performance numbers) |
| `health-lines.txt` | the two `Traceback`s in the health grep are `/stop_profile` 500s (the profiler had already stopped itself after `num_steps=8`, so the explicit stop found nothing to stop; the traces were written); 0 NaN / Xid / OOM |

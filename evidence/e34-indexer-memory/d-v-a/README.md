# E34d-v (a) / (b) — 262K × 4: fails at chunk 8192, qualified at chunk 4096 (2026-10-09)

Overlay `1eeb48cd` (the 1M × 1 overlay, unchanged), image `dsv41-kt:recipe`, 219 / 165 experts, D=4,
spec off, `MEM_FRACTION_STATIC=0.96`, `ALLOW_UNTESTED_FIT=1` (the launcher guard refuses more than
one slot above 64K). Two serves, ~1 h 45 min. Comparators: `../d-v-i/262k-4slot/` (the same rung on
`2323052b`, OOM on the first chunk), `../../t329/262k-1slot/` (one slot on 219 / 8192: prefill
1,400 / 1,493, TTFT 193 s at 253K), `../../t328/default-219-8192/corpus/` (NLL baseline; chunk 4096
is a shape change, so the T3.28 pooled gate applies, not "0 moved").

Question (EXPERIMENTS Part d-v (a), owner's sequence): with ≈ 0.87 GiB of chunk-proportional
transient removed by d-v (ii), does 262K × 4 at the launcher default (219 / 8192) fit? If not,
(b) chunk 4096 for the rung, (c) 212 / 172 as the second fallback.

## (a) `262k-4slot/` — chunk 8192: fail

| | reading |
|---|---|
| pool | 1,064,960 tokens (same size as the 1M × 1 pool at 0.962; here 0.96 and bs ≤ 4 graphs) |
| after capture | **2.92 / 2.86 GB** available (the 1M × 1 serve had 3.25 / 3.18 — bs-4 graphs cost 0.33 GB) |
| stress | `stress-kv-full.py --concurrency 4 --prompt-tokens 260096 --seed 31`: **CUDA OOM on the third 8192-token chunk of request 1** (prefix ≈ 24K; the other three queued) |
| failing allocation | **320 MiB** at `kernels/ops/layernorm/mhc.py:1258` (`mhc_post` output on TP0 / GPU 1) — one of the 320 / 400 / 640 MiB retries the d-v (i) run had named by size |
| allocator state | 265.75 MiB free, 92.44 GiB allocated, 58 MiB graph pools, **1.05 GiB reserved-but-unallocated** |
| health before the OOM | 0 NaN / Xid / Traceback other than the OOM; EDAC 0/0 |

The reading that matters: 1.05 GiB of reserved-but-unallocated memory is larger than the 320 MiB
request that failed, and the OOM is on the *third* chunk, not the first — the peak is not a single
missing transient but the caching allocator failing to fold freed chunk-8192 blocks into one 320 MiB
piece. That is the first concrete trigger for T3.31 idea 8 (`expandable_segments` enabled after
capture); it stays a fallback, never a default (E34d: VMM segments cannot be CUDA-IPC exported by the
custom all-reduce at capture, so it has to be toggled after capture, which is an overlay change).

## (b) `262k-4slot-chunk4096/` — chunk 4096: every gate passed

`CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 CHUNKED_PREFILL_SIZE=4096 ALLOW_UNTESTED_FIT=1`.

| gate | measured | verdict |
|---|---|---|
| after capture | **3.65 / 3.59 GB** available (pool 1,064,960 tokens, +0.73 GB vs chunk 8192); `nvidia-smi` 93,752 / 93,813 MiB used (`vram-after-capture.txt`) | — |
| (b) stress 4 × ~253K (`--prompt-tokens 260096 --seed 31`) | all four finished, wall **1,082 s**; TTFT 289 / 810 / 1,064 / 550 s; 256 output tokens each; peak used 97,010 / 97,149 MiB, min free **877 / 738 MiB** | **pass** |
| (c) needles (`verify.py --context 262144`) | **6/6 at 221,756 tokens**, TTFT 216 s (one slot on 219 / 8192: 133 s); alias / ceiling / arithmetic / thinking toggle / tool call PASS | **pass** |
| (d) NLL corpus (shape change → T3.28 pooled gate ≤ +0.002) | pooled ΔNLL **−0.00114 ± 0.00097** nats/tok (z +1.18; cluster-robust z +0.87); 62,668 of 65,653 changed; worse / better by > 0.5 nat: 1,826 / 1,852; per-text max z +2.39 (05-troubleshooting), 08-bench-decode-py 0 moved (short text, same M) | **pass** |
| (f) prefill 16K / 32K (`bench-prefill.py --runs 3 --seed 777`) | **975.18 / 974.82 tok/s**, TTFT 16.47 / 32.83 s — vs 1,887 / 1,901 at chunk 8192 on the 64K × 4 default: **−48 %** | measured cost, see below |
| (f) decode C1 / C4 (`bench-decode.py --distinct-prompts --seed 4242`, 6 / 3 runs) | C1 **41.44 / 39.55 wall**, TTFT 0.618 s; C4 **56.23 / 54.07 wall**, TTFT 1.525 s | unchanged: **pass** |
| (e) health, EDAC | `launch-container.txt` grep (see `health-hits.txt`), EDAC 0/0 | **pass** |

Two extra C1 runs on this serve are the spec-off baselines for the T3.9h n-gram probe (same seed,
512 tokens, 6 runs): `c1-copy-code.*` (edit a ~1.2K-token module that is in the prompt) **42.38 /
39.27 wall**, TTFT 0.974 s; `c1-copy-table.*` (re-emit a 40-row table from the prompt) **42.16 /
41.70 wall**, TTFT 0.124 s (the TTFT is a radix-cache hit on the warmup run's prefix; the wall figure
is therefore closer to the summed one than on the other fixtures).

## What this settles

- **262K × 4 is a qualified rung at 219 / 165 + chunk 4096** (`CHUNKED_PREFILL_SIZE=4096` per serve,
  `ALLOW_UNTESTED_FIT=1`), the first multi-slot profile above 64K. T3.35 (code-review exercise) is
  unblocked at this shape.
- The price is prefill: 975 tok/s against 1,887 at chunk 8192, i.e. the chunk-4096 figure E22/E34b
  measured before the chunk flip (≈ 1,040), not a new regression. Decode and NLL do not move.
- Keeping chunk 8192 at 262K × 4 needs either idea 8 (compaction of the 1.05 GiB the allocator holds
  but cannot hand out) or (c) 212 / 172 (≈ 2.3 GB for 7 experts, −2 % decode). Owner decision; the
  rung as qualified needs neither.

## Files

- `262k-4slot/`: `launch.txt`, `launch-container.txt` (OOM traceback at line ~580), `stress-kv.txt`,
  `vram-after-capture.txt`, `edac.txt`, `health-hits.txt` (empty).
- `262k-4slot-chunk4096/`: `manifest.json`, `launch.txt`, `launch-container.txt`,
  `vram-after-capture.txt`, `stress-kv.{json,txt}`, `verify.{json,txt}`, `corpus/` + `corpus.txt`,
  `prefill-16k.*`, `prefill-32k.*`, `c1.*`, `c4.*`, `c1-copy-code.*`, `c1-copy-table.*`,
  `health-hits.txt`, `edac.txt`.

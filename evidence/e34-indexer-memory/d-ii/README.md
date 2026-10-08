# E34d-ii — placement trade 219 / 165 + prefill chunk 8192 (measurement serve, 2026-10-08)

Overlay `d6811728` (unchanged), D=4, spec off, 64K x 4 slots, `MEM_FRACTION_STATIC=0.96`.
Owner-agreed measurement-only serve: `KT_NUM_GPU_EXPERTS=219 CHUNKED_PREFILL_SIZE=8192
ALLOW_UNTESTED_FIT=1` (the launcher guard predates E34b). Resolved launch line in `launch.txt`:
`experts gpu=219 cpu=165 ... chunk=8192 mem_fraction=0.96 context=65536 slots=4`.
Control serve (this directory, `control-219-chunk4096/`): same placement, chunk 4096, NLL only.

**Result: the trade delivers the projected prefill gain at a smaller decode cost than projected,
but chunk 8192 fails the NLL gate (+0.0067 nats/tok vs the +0.002 gate) while the placement alone
passes (+0.0013). Not adopted; default stays 226 / 158 + chunk 4096. The NLL shift is registered
as E34d-iii (cause unknown).**

## Measurements (219 / 165 + chunk 8192) vs the deployment default (226 / 158 + chunk 4096, E34b-ii)

| metric | 226 / 158, chunk 4096 | **219 / 165, chunk 8192** | delta |
|---|---|---|---|
| free after capture (TP0 / TP1) | 3.48 / 3.42 GB | **5.56 / 5.51 GB** | +2.1 GB (7 experts x 0.34 GB = 2.4 GB, minus the larger chunk's activations) |
| prefill 16K / 32K (idle, `prefill-16k.*`, `prefill-32k.*`) | 1,040 / 1,040 tok/s | **1,438 / 1,526 tok/s** (TTFT 11.2 / 21.0 s) | **+38 % / +47 %** |
| C1 median / wall (`c1-off6.json`, 6 runs, 512 tok, distinct prompts) | 42.17 / 40.23 | 41.39 / **39.45** | **-1.9 % wall** (projected -3 to -4 %) |
| C4 median / wall (`c4-off6.json`, 3 runs) | 57.84-58.96 / 55.53-57.40 (E34a soak; T3.19b 57.86 / 55.51) | 56.32 / 54.16 | **-2.4 to -5.6 % wall** (vs the soak range) |
| 4 x 62K stress (`stress-kv.*`) | min free 1,541 / 1,484 MiB, wall 290.6 s | min free **2,339 / 1,302 MiB**, wall **175.5 s** | all finished; TP1 min free lower (larger-chunk activations land on TP1 too) |
| needles (`verify-64k.*`) | 6/6 | **6/6 at 55,206 tokens**, TTFT 32.8 s | tool-call + multi-needle all pass |
| NLL vs `nll-t316c.json` (`nll.*`) | 0 moved | **+0.00674 nats/tok (z -1.89)**, mean abs dlogp 0.122, 146 worse / 121 better (> 0.5 nats) | **fails the +0.002 gate** |
| health (container log) / EDAC | 0 / 0 0 | 0 / 0 0 | - |

The prefill figures reproduce E22e's chunk-8192 numbers (1,375 / 1,404 at 208 / 176) plus the
E34b indexer gain. Per-chunk host cost is still the ~4.6 s export + barriers (E22), so 8192 tokens
per chunk amortise it over twice the tokens.

## The NLL finding

`control-219-chunk4096/nll.*` — same 219 / 165 placement at chunk 4096 — scores **+0.00134
(z -0.35), 139 worse / 146 better**: the E22 streamed-prefill signature (the 7 experts that moved
to the GPU now run through the MXFP8-activation kernel), well inside the gate. So the +0.0067 is
the chunk, not the placement. Offline per-token comparison of the two 219 serves
(`nll.json` vs `control-219-chunk4096/nll.json`, paired logprob deltas):

| span | tokens | dNLL (8192 - 4096) | paired z | median dlogp | fraction better |
|---|---|---|---|---|---|
| [0, 4096) — one prefill chunk in **both** modes | 4,096 | **+0.0080** | **+2.22** | 0.000 | 0.48 |
| [4096, 4795) — second chunk at 4096 (699 tokens, below `KT_GPU_PREFILL_MIN_TOKENS=1024`, so the CPU kt path); part of the single chunk at 8192 (streamed GPU path) | 699 | -0.0098 | -1.06 | 0.000 | 0.54 |

No 512-token (`logprob_chunk`) pattern; the shift is spread over the first 4,096 tokens as a
loss-tail effect (median 0, slightly more tokens > 0.5 nats worse than better), not a per-token
bias. The tail difference has an obvious cause (CPU exact path vs streamed MXFP8 path) and is not
distinguishable from 0 at 699 tokens. The first-4,096 shift has no obvious cause: those tokens are
one prefill batch in both modes; only the batch's row count differs (4,096 vs 4,795).

**Chunk 8192 was never NLL-scored before this serve.** E22e (2026-09-22) ran prefill and needles at
chunk 8192, but its NLL file (`evidence/e22-layerwise-prefill/nll-stream.json`, label `e22-stream`)
came from the chunk-4096 serve; the E22 verdict's "NLL-neutral" covered 4096 only. Every NLL run
in the recipe (t316c baseline included) used chunk 4096, so all of them share whatever a 4,795-token
prompt does at M = 4,096 + 699, and this serve is the first with a single M = 4,795 batch.

Hypotheses, all **unverified**, for E34d-iii:
1. Serve-to-serve noise with a one-serve sample. Placement-only reruns sit at +/-0.0013, D=4
   reruns at 0; +0.008 at z 2.2 is 5x that, but there is no replicate at chunk 8192. First step
   of the diagnosis is a second chunk-8192 NLL on a fresh serve.
2. An M > 4096 assumption in the streamed-prefill path (E22; buffers, tiling or tactic selection
   in `fused_experts_none_to_flashinfer_mxfp4` at M = 4,795 rows) — a latent overlay bug that the
   deployed chunk-4096 config never exercises. Testable with `nll-score.py --max-chars` so the
   prompt is <= 4,096 tokens on a chunk-8192 serve: if the common-prefix delta vs the chunk-4096
   control is ~0 at M <= 4,096 and returns at M = 4,795, the shift is row-count-specific.
3. A chunk-boundary effect in the *baseline* (second-chunk tokens seeing fp8/compressed KV from the
   pool vs in-batch K/V) — would show up in the tail, not in the first 4,096 tokens, so it does
   not explain the main shift on its own.
4. Attention / indexer kernel configuration at M > 4096 (the row-tiled indexer test
   `test_indexer_row_tiling.py` is row-count-invariant by construction, so it cannot distinguish).
5. The 219 / 165 placement interacting with the larger chunk (different expert set streamed per
   chunk) — the control rules out placement alone but not the interaction.

`--compare` on a truncated prompt reports `delta_mean_nll` over different token counts; use the
common-prefix paired mean / z (printed) or recompute offline as above.

## Decision

Not adopted as the deployment default: the gate is the gate (owner rule: +0.002), and a config
whose NLL moved for an unknown reason is not qualified for a coding / code-review box regardless of
its +38-47 % prefill. The prefill and decode numbers stand as the measured cost/benefit of the
trade should E34d-iii clear the chunk. `scripts/wait-ready.py` gained a `--container` pool-cap
check (warns when the KV sizer logs `larger than the profiled value`) during this serve; it
reported no cap here (64K x 4 pool fits).

Files: `launch.txt`, `launch-container.txt` (TP logs), `prefill-16k.*`, `prefill-32k.*`,
`c1-off6.json` / `c1-bench.txt`, `c4-off6.json` / `c4-bench.txt`, `stress-kv.*`, `verify-64k.*`,
`nll.*`; `control-219-chunk4096/{launch.txt,launch-container.txt,nll.*}`.

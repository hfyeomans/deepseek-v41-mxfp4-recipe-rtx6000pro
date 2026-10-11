# T3.15e — where the 14,775-token greedy residual comes from (E42)

2026-10-11. Three developer-profile serves on the launcher default (219 / 165, chunk 8192, el8-mix, lever
on, both T3.15b determinism levers, exact top-k patch 0055) with the overlay's opt-in step-fingerprint
recorder (`python/sglang/srt/layers/attention/dsv4/step_fingerprint.py`, overlay commits `8b86a84f`,
`833dc954`, `3caa0182` = patches 0056 / 0057; `SGLANG_DSV41_STEP_FINGERPRINT_DIR`, off by default, zero cost
when unset), then offline replays of the captured row inside the image. Question (from E41): the residual
survives an exact top-k and bit-identical indexer logits — which step-109 state differs first, and why?

## Answer

**The residual is one exact fp32 score tie at the #512 / #513 boundary of layer 36's decode top-k, resolved
differently by the serving path on a minority of repeats.** At decode step 109 of the 14,775-token prompt
(compressed indexer length 14,884 on the ratio-1 layers), the layer-36 indexer row has positions **4971 and
10595 tied at −1.1060476** exactly in fp32 (only two tied positions; 511 strictly greater; the fp16 coarse bin
holds 16 values, so the kernel's ≤ 32-candidate warp tie path runs). The patch-0055 rule is `(value desc,
idx asc)` → 4971. Rank 1 chose 4971 on 15 / 15 repeats; rank 0 chose 10595 on 2 / 15 (the first and the
last). The two ranks then attend to different 512-sets for one layer, the all-reduce after layer 36 carries
the difference into every later stage (`fingerprint-diff`: first differing stage `sel` at layer 36 on the
flipping rank, `q` at layer 37 on the other), and the logprobs differ from token 109 on by ≤ 0.051 while
the argmax never moves. Everything upstream of the top-k — the indexer input `x`, the logits row (bit-identical
on all 15 repeats and both ranks), the KV slots — is identical.

**Offline, the same kernel on the same row never flips.** `offline/topk-v2-replay-probe.py` pushes the
captured row through `topk_transform_paged_v2` 12,800 times — both captured tails, window 131,072 / 16,384 /
14,912 / 14,884, output buffer pre-filled empty / with 4971 / with 10595 / −1, batch 1 and 4, synthetic
tails (+100, −inf, NaN, a third tie) — and selects 4971 exactly every time. The in-isolation kernel is
deterministic; the production flip needs a production-only factor (CUDA-graph replay with PDL on the
persistent-cluster build, concurrent side streams, or the SM120 plan path), and the code read of the
streaming path (`TopKStreaming`, `find_threshold`, `handle_tie`) shows no race. Left open; see Decision.

## Serves

| dir | overlay | trigger | what |
|---|---|---|---|
| `.` (serve 1) | `833dc954` | none | recorder on, 5 fresh repeats (`drift-fresh`: 5/5 bit-identical) then 15 (`drift-x15`: 1/15 differs, tokens identical, max \|Δlogp\| 0.0507). `fingerprint-diff-serve1.txt`: rank 1 repeat 11 differs first at **step 108, layer 36, stage `sel`**; rank 0 at step 108 layer 37 stage `q` — the flip was on rank 1 that time |
| `trigger-7442/` (serve 2) | `3caa0182` | 7,442 | trigger dump at the ratio-2 compressed length: only layers 2 / 8 / 14 fire at 7,442 and all were identical; no manifest (the drift JSON failed the one-serve-per-dir rule because serve 1's manifest already sat here — logs only) |
| `trigger/` (serve 3) | `3caa0182` | 14,884 | trigger dump at the ratio-1 compressed length: layers 20 / 24 / 28 / 32 / 36 fire; `drift.json` tokens identical, max \|Δlogp\| 0.0507; `fingerprint-diff.txt`: rank 0 flips at step 108 layer 36 `sel`; `offline/trigger-narrow-14884.txt` names the two tied positions |

The `.pt` dumps (58–88 MB per rank per serve) are kept out of git (`.gitignore`, copies under
`.amp/in/t315e/dumps/`); every number above is reproduced by the `offline/` scripts from them.

## Offline

| file | what |
|---|---|
| `offline/trigger-analyze.py`, `offline/trigger-narrow.py` | per-layer ring analysis: distinct selected sets, in-range boundary values, tail contents, stable-top-k check; `trigger-narrow-7442.txt` (all identical) and `trigger-narrow-14884.txt` (layer 36 rank 0, 2 / 15 pick 10595) |
| `offline/tie-bin-count.py`, `.txt` | the boundary: #511 −1.1055861, #512 = #513 −1.1060476, #514 −1.1072798; 2 exact ties, 16 values in fp16 bin 3014 |
| `offline/topk-v2-oob-probe.py`, `.txt` | rules out an out-of-range read: every candidate index is `< seq_len` on every repeat and rank (ALL OK) |
| `offline/topk-v2-replay-probe.py`, `.txt` | the captured row through the production kernel 12,800×: 4971 every time, zero flips |
| `scripts/fingerprint-diff.py` (recipe) | aligns repeats of one prompt in a dump and names the first differing (step, layer, stage) |

## Decision

Not a quality gate (tokens identical in every run of E38 / E39 / E41 / E42; the bound is one tied KV
position for one layer at one step) and not a performance item. Recorded as an open upstream-facing finding:
the DSV4 decode top-k can resolve an exact tie differently across TP ranks and repeats in production while
being deterministic in isolation — a candidate follow-up to sgl-project/sglang #35257 (owner call; the
secret gist already holds the #35257 draft). The recorder patches stay in the series (off by default) so the
capture is reproducible; `greedy-drift.py` now retries `/flush_cache` while a request is still retiring.
Queue continues with T3.9h-b (n-gram) and T3.41 (FP8 GEMV).

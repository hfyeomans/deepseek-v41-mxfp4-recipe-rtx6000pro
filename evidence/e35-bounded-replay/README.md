# E35 — DSV4.1 decoder SWA bounded replay on the hybrid (2026-10-08/09)

Question (T3.33): does the overlay's dormant `--enable-decoder-swa-bounded-replay` raise 16K / 32K prefill on the launcher default (219 / 165, chunk 8192, 64K × 4), net of the prefill CUDA graph it refuses, with quality intact?

Answer: **prefill +39 % at 16K (1,514 → 2,106 tok/s, TTFT 10.6 → 7.6 s) and +34 % at 32K, decode and memory unchanged — and a quality STOP by two orders of magnitude.** The replay runs layers 21–39 over only the last 128 rows of each extend with the SWA window floored at the tail start, so the late layers' local context is cut by construction (the last row's 19-layer SWA receptive field is ~2,432 rows, not 128). Paired teacher-forced next-token score vs the control: fresh prompts **+1.13 nats/token** (95 % CI +0.43…+1.83; greedy token differs at 34 / 158 positions; code texts +2.3…+2.8), extends **+0.055** (1–8-token extends +0.29, 129–256-token +0.02). Gate was ≤ +0.002. Not adopted; the encoder variant (C) was not served (see below).

## Serves

| dir | serve | flags on top of the launcher default | what ran |
|---|---|---|---|
| `A/` | control, manifest `8bdf1f56983c` | `--cuda-graph-backend-prefill=disabled` | prefill 16K / 32K × 3 cold + warm, C1 (off 6), C4 (off 0), needles 64K, corpus NLL vs `evidence/t328/default-219-8192/corpus`, teacher-forced scorer |
| `B-crash/` | first decoder-replay attempt on the committed overlay `bc73ef8d` | A's + `--enable-decoder-swa-bounded-replay` | crashed on the first 16K prefill: `KTContractError: layer 21: 128 tokens take the CPU path while a streamed forward is open` — the E22 streamed-prefill pipeline requires one token count per forward, the replay tail (128 < `KT_GPU_PREFILL_MIN_TOKENS` 1024) breaks it |
| `B/` | decoder replay, overlay `bc73ef8d` + the bind-mounted patch in `patches/` (manifest `6b02b1608fce`, rescrubbed from `4e06fdbd4ef0`) | A's + `--enable-decoder-swa-bounded-replay`; `KT_GPU_PREFILL_LOG=1` | prefill, C1, C4, needles; then the corpus NLL request killed the server (`crash-prompt-logprob-refusal.log`) |
| `B-serve2/` | same config relaunched (manifest `1a94f5c29b64`, rescrubbed from `3cb2eeb2c337`) | same | teacher-forced scorer, stress 4 × 62K. `attempt1-*` are the first pass of the same two steps, whose outputs the evidence-manifest stamp refused into `B/` (one directory holds one serve); the scorer's printed per-text means were identical to the second pass |

Prefill CUDA graph: already auto-disabled for DSV4 by the overlay's `cuda_graph_hook.py` breakable rule, so A is the launcher default (A corpus NLL: 0 of 65,653 tokens moved vs T3.28's default serve). The "graph-off cost" the protocol wanted to separate is zero.

## Numbers

| | A (control = default) | B (decoder replay) | Δ |
|---|---|---|---|
| prefill 16K cold / warm, tok/s | 1,514.5 / 1,650.3 | 2,106.3 / 1,990.6 | **+39 % / +21 %** |
| prefill 32K cold / warm, tok/s | 1,620.3 / 1,665.6 | 2,175.4 / 2,155.6 | **+34 % / +29 %** |
| TTFT 16K / 32K cold, s | 10.61 / 19.76 | 7.63 / 14.71 | −28 % / −26 % |
| C1 median / wall tok/s, TTFT | 41.35 / 39.46, 0.616 s | 41.46 / 39.55, 0.618 s | 0 |
| C4 median / wall tok/s | 56.13 / 53.94 | 56.15 / 54.04 | 0 |
| needles 64K (55,206 tok) | 6/6, TTFT 30.84 s | 6/6, TTFT 22.58 s | −27 % TTFT |
| stress 4 × 62K min free GPU0 / GPU1 | not run (A is the G1-exit config: 641 / 582 MiB, E34a) | 941 / 880 MiB, all finished, 129.7 s | — |
| corpus NLL (65,653 tok, prompt logprobs) | 0 moved vs T3.28 default | **refused — and the refusal is a scheduler `ValueError` that takes the server down** | — |
| teacher-forced Δ NLL, fresh (158 positions, 8 texts) | baseline | **+1.1286 ± 0.356 se** (worst single position −42.7 logp; 40 positions move > 0.5; greedy same 124 / 158) | STOP |
| teacher-forced Δ NLL, extends (2,400 positions) | baseline | **+0.0553 ± 0.008**; by extend length 1–8: +0.289, 9–32: +0.087, 33–128: +0.049, 129–256: +0.020 | STOP |
| health (NaN / Xid / Traceback / contract), EDAC | 0, 0/0 | 0 on the two measured serves (the two crashes are recorded separately), 0/0 | — |

B's first cold 16K run (8,785 tok/s, TTFT 1.8 s) was a radix hit on the seed-777 prompt the pre-battery smoke had already run; the medians are the two uncached runs. A's first run (576 tok/s) is the cold-Engram first forward.

## Why the quality moves (and why it had to)

`late_layer_tail_layout` (overlay `python/sglang/kernels/ops/attention/dsv4_attn_metadata_kernels.py`) selects the last `min(128, extend)` rows per request and floors every tail row's SWA window at the tail start; `_build_late_layer_tail_metadata` (`deepseek_v4_backend.py`) says why: "window KV before it is never written at those layers". So at layer 21 the first tail row attends to itself only, and the last row — the one that produces the next token — is exact at layer 21 but inherits truncated-window hidden states through layers 22–39. The flag's own wording ("deterministic for a fixed prompt and chunk size") promised determinism, not equivalence; the scorer reproduced the fresh-prompt means bit-for-bit across the two B serves, so it is deterministic and wrong by the same amount every time. The effect is largest where the window matters most: code (`serve-hybrid.sh` +2.30, `bench-decode.py` +2.81, EXPERIMENTS +2.72 nats) and 1–8-token extends (+0.29) — i.e. exactly the owner's multi-turn coding shape.

The prefill win is the same mechanism: ~19 of 40 layers run on 128 rows instead of 8,192, which is why it is +39 % and not the "~2×" the TensorFold recipes claim for their (different) CED replay.

## Two operational findings

1. **The E22 streamed-prefill contract vs a shrinking forward.** `PrefillPipeline` requires one token count per forward; the replay tail enters layer 21 with 128 tokens and the committed overlay raises. `patches/overlay-kt-prefill-abandon-forward.patch` (+74 / −5 lines in `kt_prefill.py` and `kt_ep_wrapper.py`) adds `PrefillPipeline.abandon()` / `export_discarded()` and `LayerwisePrefillCoordinator.abandon_forward()`: when a CPU-path layer meets an open forward *and the token count shrank*, rank 0 syncs the pending export, the buffer is released, the forward closes and the rest runs on the CPU path; the original raise stays for every other case. `patches/recipe-test-abandon.patch` adds `test_abandon_closes_early_when_tokens_shrink` (10/10 passed with the patch applied). Both were bind-mounted for the B serves (`EXTRA_DOCKER_ARGS`), logged as `forward closed at layer 21: 128 tokens (streamed layers ran at 8192)`, and reverted after the STOP: the overlay stays at `bc73ef8d`, the image unchanged. Reapply both patches for any successor.
2. **A model-level refusal that is a scheduler exception is a crash.** `_check_late_layer_tail_readers` raises `ValueError` inside the forward when a request asks for prompt logprobs (`logprob_start_len < prompt length`); the scheduler process dies and the tokenizer manager SIGQUITs the container. Any variant of this flag that ever reaches a default needs the check in the tokenizer manager (HTTP 400), not the model.

## Serve C (encoder variant) — not served, with the arithmetic

`--enable-encoder-swa-bounded-replay` stops caching SWA KV in the pool (per-request 128-row windows instead) and, on every prefix hit, re-runs the last 128 cached tokens with the window floored at the replay start (`model_executor/encoder_swa_replay.py`) — the same approximation class as B's extend shape, applied to every multi-turn turn. The memory it can free is bounded by the SWA pool, already capped on the default: `DSV4 SWA sizing: mode=cap, swa_tokens=25856 … swa_fixed=0.56 GB` of the 3.34 GB available to the pool per GPU, minus its own request windows and scratch. ≤ 0.5 GB/GPU does not move any rung of the T3.29 ladder (262K × 4 OOMs in the fused-MoE workspace, not the pool), and the quality gate would fail on the shape it targets. Owner can veto and ask for the measurement.

## Successor registered (not scheduled)

E35-ii in `plan/EXPERIMENTS.md`: an *exact* bounded replay whose tail is the late layers' receptive field (128 × (19 + 1) = 2,560 rows per extend, floor kept) so that the last 128 rows — everything decode can reach — are computed over their full windows at every late layer; expected prefill gain ≈ half of B's (late-layer rows per 8192 chunk 8192 → 2,560) and no gain for extends ≤ 2,560 tokens. Needs: a `tail_len` knob in the backend, the E22 contract change above *for tails ≥ 1024* (they take the streamed path at a different token count — a second contract case), the HTTP-level refusal, and this scorer as its gate.

## Files

- `A/`, `B/`, `B-serve2/`: `manifest.json` (hashed, `scripts/evidence_manifest.py`), `run.out`, `launch-flags.txt`, `gpu-during.csv`, `health-grep.txt`, the JSONs named above. `B/crash-prompt-logprob-refusal.log` is the scheduler traceback; `B/nll-corpus.out` records the refusal.
- `A/tf-score.json`, `B-serve2/tf-score.json`: the scorer's positions (`shapes.fresh|ext.<text>[]` with `k`, `true_id`, `logprob`, `greedy_id`, `cached`, `prompt_tokens`) and the comparison in `B-serve2/tf-score.json["compare"]`. Scorer and battery scripts are the thread's scratch tools (`.amp/in/e35-tf-score.py`, `.amp/in/e35-run.sh`), to be promoted into `scripts/` with T3.30 if a successor needs them.
- `patches/`: the two reverted diffs.
- `B/manifest.json` and `B-serve2/manifest.json` carry `rescrubbed_from`: the manifest writer scrubbed `$HOME` only at the start of a string, so the bind-mount paths inside `EXTRA_DOCKER_ARGS` (the patch mount) leaked the account name. `scripts/evidence_manifest.py rescrub <dir>` re-applied the scrub to the originals from `~/.cache/dsv41-recipe/manifests/`, re-hashed, and re-stamped the sibling JSONs (`B/`: prefill, c1, c4; `B-serve2/`: tf-score, stress-kv). `A/` had nothing to scrub. `verify-64k.json` (from `scripts/verify.py`) is not stamped by that script; it is treated as pre-cut-over.

# T3.19a — where the decode step goes: per-layer timeline at bs 1 and 4 (2026-09-24)

Stage 1 deliverable (a) of the migration roadmap (G1). Overlay commit `e66d44fb`
(`kt_timeline.py`, opt-in via `KT_TIMELINE=1`; behaviour-identical when unset).
Report: `scripts/kt-timeline-report.py`; tests: `scripts/tests/run-kt-timeline-tests.sh`
(58 checks, in-image, scripted clock, in `ci.sh`).

Server: the qualified D=4 / speculation-off / 226-GPU-expert profile, `scripts/serve-hybrid.sh`
with `EXTRA_DOCKER_ENV="KT_TIMELINE=1 KT_TIMELINE_DIR=/tmp/kt-timeline"` (sample every 8th replay,
dump layers 14 and 20). Traffic: `verify-batch-consistency.py` (PASS), `bench-decode.py` C1 then C4
(the standard recipes, 6 runs × 512 tokens, distinct prompts), `nll-score.py`.

## How the numbers are measured

Six `torch.cuda.Event(enable_timing=True, external=True)` per MoE layer are recorded *inside* the
captured graph (start → after kt `submit_forward` = **export** → after the resident-expert GEMM =
**gpu_experts** → after `cpu_infer.sync_with_cuda_stream` = **cpu_wait** → after
`copy_forward_output_to_device` = **h2d** → after `output + cpu_output` = **combine**). Segment
start/end events and host `perf_counter` clocks are recorded around each replay segment and each
eager break. Every replay writes a *tick*; every 8th replay syncs and writes the full record.

The step time compared against is the **replay period**: host wall between consecutive replays of
the same key, taken from *unsampled* ticks (no sync, no perturbation). 1000 / period is the decode
rate the scheduler can deliver for that batch size. `engram_break` = GPU idle between segments;
`other_gpu` = segment time not inside a MoE layer span (attention, dense, norms, all-reduce,
lm head); `between_replays` = period − GPU span (logits/sampler kernels outside the graph).

## Result — rank 0 (the kt rank), `decode:1` and `decode:4`, `candidate_all` graphs

| bucket (median over sampled steps) | bs 1 ms | share | bs 4 ms | share |
|---|---:|---:|---:|---:|
| export (D2H + kt submit host node, 40 layers) | 2.482 | 10.5 % | 2.890 | 4.2 % |
| gpu_experts (resident 226-expert GEMM, overlapped with CPU) | 2.517 | 10.7 % | 5.111 | 7.4 % |
| **cpu_wait** (exposed kt wait at the `sync` host node, 40 layers) | **7.879** | **33.4 %** | **48.588** | **70.8 %** |
| h2d (pinned output → device) | 0.231 | 1.0 % | 0.262 | 0.4 % |
| combine (`output + cpu_output`) | 0.228 | 1.0 % | 0.232 | 0.3 % |
| engram_break (2 eager `_engram_file_rows`, GPU idle) | 1.822 | 7.7 % | 2.915 | 4.2 % |
| **other_gpu** (attention, dense, norms, all-reduce, lm head) | **8.073** | **34.2 %** | 8.336 | 12.1 % |
| between_replays (outside the graph) | 0.269 | 1.1 % | 0.209 | 0.3 % |
| **step = unsampled replay period** | **23.589** | 100 % | **68.659** | 100 % |
| GPU-event buckets as share of the period | 23.320 | 98.9 % | 68.450 | 99.7 % |
| 1000 / period (steps/s → tok/s) | 42.4 tok/s | | 14.6 steps/s = 58.3 tok/s | |
| bench-decode wall tok/s (same serve) | 39.15 | | 55.56 (13.9 / stream) | |
| GPU backlog at replay start (host runs ahead) | 10.1 | | 41.0 | |
| cpu_total = sync − export, summed (CPU tier latency incl. overlapped part) | 10.4 (0.26 / layer) | | 53.7 (1.34 / layer) | |

Rank 1 (no kt work): `other_gpu` 18.9 ms of 23.3 at bs 1 and 60.2 of 68.5 at bs 4 — it spends the
difference to rank 0's real work (≈ 8 ms) waiting in the per-layer all-reduce for rank 0's CPU tier.
This is the "one GPU pegged, the other at 7–20 %" the owner sees in nvtop.

Segments and breaks (rank 0, bs 1): segment GPU time 0.54 / 6.97 / 14.12 ms; breaks
`_engram_file_rows` host 11.47 ms → GPU gap **1.09** ms, host 7.25 ms → GPU gap **0.73** ms. The host
time of a break is mostly the D2H sync waiting for the GPU to drain the preceding segment; the
exposed cost is the GPU gap. bs 4: gaps 1.47 + 1.43 ms.

Per-layer rows: `report.md`. bs 1 alternates 0.13 / 0.22 ms of `cpu_wait` per layer (deferral
phase); layer 39 (last MoE layer, D=0 rule) waits 0.39 ms with nothing to overlap. bs 4: 1.0–1.6 ms
per layer except layers 0, 1 and 14 (0.33–0.42 ms) — the first layer of each segment starts with an
empty deferred backlog because the CPU pool caught up while the GPU sat in the Engram gap.

## Gate (per-step, from `evidence/t319/baseline-pin.json`)

| check | pin | measured with timeline on | gate | result |
|---|---:|---:|---|---|
| C1 median / wall tok/s, TTFT | 41.92 / 39.86 / 0.609 s | 41.37 / **39.15** / 0.622 s | wall ≥ 39.0, TTFT ≤ 0.65 | pass (−1.8 % wall: ≈ 240 event records per step + a sync every 8th replay) |
| C4 aggregate / wall, per stream | 58.09 / 55.79 / 13.9 | 57.80 / 55.56 / **13.9** | ≥ 13.6 / stream | pass |
| NLL vs `nll-t316c.json` | | ΔNLL +0.00000, 0 / 0 of 4795 tokens moved | 0 tokens moved | pass |
| consistency `--same-prompt` C4 | | rows identical, prefix 519/519 | PASS | pass |
| NaN / Xid / Traceback / contract / seal lines | | 0 | 0 | pass |
| EDAC ce / ue | | 0 / 0 | 0 / 0 | pass |

The plain (KT_TIMELINE unset) path is untouched; these numbers are the *instrumented* serve, not a
new baseline.

## Recorded-tensor dumps (`dumps/`, 3.2 MB)

`dump-rank{r}-decode-{bs}-{variant}-layer{14,20}-{n}.pt`: graph-replay snapshots of
`hidden_states` (bf16 [bs, 5120]), `topk_ids` (int32 [bs, 6]), `topk_weights` (fp32), `kt_output`,
`moe_output` for bs 1..4 (n=0 is the post-capture check replay; n=1 is real traffic), plus
`dump-rank{r}-eager-layer{14,20}-0.pt` from the eager warm-up forward. Real routing (experts 1, 273,
375 recur), all values finite. These are the inputs for the T3.16e(3) graph-vs-eager parity test,
which is **not yet run**: it needs a harness that feeds a recorded `hidden_states`/`topk` through the
eager kt path in a loaded container and compares `kt_output` bit-for-bit; that harness is the first
consumer of the T3.19b `CpuExpertProvider` seam and is scheduled there.

## Reading

1. At bs 1 the step is **not** Python-launch-bound: the host is 10 ms ahead of the GPU when it
   launches a replay, and GPU-event buckets explain 98.9 % of the period. The migration roadmap's
   "Python launch cost" bucket is ≈ 0 exposed at bs 1 and 4.
2. Two buckets of equal size own the bs 1 step: the exposed CPU-tier wait (7.9 ms, 33 %) and the
   non-MoE GPU work (8.1 ms, 34 %). The CPU tier's total latency is 10.4 ms/step (0.26 ms/layer,
   consistent with E21's 0.146 ms per full expert read at ≈ 2.5 CPU experts per token-layer with
   partial overlap); D=4 hides 2.5 ms of it behind the resident GEMM.
3. `export` is 2.5 ms (10.5 %) at bs 1 — 62 µs per layer for the D2H + submit host node — and does
   not grow with bs (2.9 ms at bs 4). It is the third bucket, not the first; T3.21 (kt export) stays
   conditional as the ledger says.
4. Engram costs 1.8 ms/step exposed at bs 1 (7.7 %), below the 2.43 ms/step the ledger carried from
   E26 (a profiler-trace figure; E26 notes the profiler inflates spans). SGLang's bs 1 layer-14 KV
   prefetch stream is *not* active in this profile (it requires the FlashInfer CuteDSL MXFP8 dense
   backend; no "Engram layer 14 KV prefetch enabled" line in `launch.txt`). T3.20's "≥ 50 % of break
   time" target is therefore worth ≤ 0.9 ms ≈ 4 % of the step at bs 1 — the measured ceiling on that
   lever.
5. At bs 4 the CPU tier is 71 % of the step and grows super-linearly (cpu_wait 7.9 → 48.6 ms for
   4× tokens): more distinct experts per layer, each paid at per-expert latency (E21, E26 B). This is
   the concurrency knee, measured per layer for the first time.
6. `other_gpu` (8.1 ms) is the next thing to split — attention vs dense vs all-reduce — before
   deciding anything about the GPU side; that is a follow-up marker set, not part of this deliverable.

Files: `timeline-rank{0,1}.jsonl` (raw), `report.md` / `report.json` (this table + per-layer rows),
`c1-timeline.json`, `c4-timeline.json`, `c1-bench.txt`, `c4-bench.txt`, `consistency.json`,
`nll-t319a.json`, `launch.txt` (redacted server log), `dumps/`.

# T3.19b: the `CpuExpertProvider` / `BufferLease` seam, the C3 remainder, and the graph-vs-eager parity test (2026-10-07)

Stage 1 deliverable (b) of the migration roadmap (G1). Overlay commit `9fc14817` (on `e66d44fb`).
Behaviour-preserving: the plain decode path calls the same kt-kernel methods in the same order with
the same arguments as before; what is added is metadata-only bookkeeping that is safe under CUDA
graph capture (no device sync, no allocation). The B2 side path is untouched (retired at T3.19c).

## What the seam is

`python/sglang/srt/layers/moe/kt_provider.py` (new, 410 lines) names the four calls through which
`KTEPWrapperMethod` drove kt-kernel's `KTMoEWrapper` with implicit class-wide state:

| Name | What it is | Replaces |
|---|---|---|
| `CpuExpertProvider` (Protocol) | `reserve(CaptureContract)`, `validate(ExpertBatchShape)`, `submit(shape, hidden, topk_ids, topk_weights, …) -> ExpertSubmission`, `wait(submission, stream, mark=None)`, `check_forward_end()` | direct `submit_forward` / `sync_forward` calls plus the timed mirror |
| `KtKernelProvider` | the adapter around today's `KTMoEWrapper`; validates layer / hidden / top_k / D against the wrapper at construction | ad-hoc attribute reads in `kt_ep_wrapper.py` |
| `BufferLease` | the 14 `data_ptr`s of the pinned/device buffer set kt hands out for one token count, recorded when a graph captures that size and asserted equal (`check_same`) at every later Python-side submit of the same size | the unnamed static tuples the T3.16a tests guarded by hand |
| `ForwardLedger` | per-forward record of submissions: a deferred slot written at layer L is consumed at L+1 and nowhere else; no slot outstanding at forward end; must agree with kt's own `_layer_has_pending_deferred` dict at the last layer | review C3 7.5 / 7.6 (previously untested) |
| module registry | `providers()`, `ledger()`, `reset_registry()`, `reserve_capture_sizes(sizes)` | the two `KTMoEWrapper.set_capture_batch_sizes` calls in `cuda_graph_setup.py` and `decode_cuda_graph_runner.py` |

`kt_contract.py` gains `check_deferral_adjacency(outstanding_for_layer, layer_idx)` and
`check_forward_boundary(outstanding_for_layer, pending_flags, last_layer_idx)` beside the existing
`check_moe_boundary`, which stays in `apply` on every rank (rank 1 and the streamed-prefill path never
touch a provider). `kt_ep_wrapper.py`: provider built in `create_weights` after the compact selector
binds; `submit()` goes through `provider.submit(ExpertBatchShape.of(x, topk_ids), …)`; `sync()` is
`provider.wait(self._take_pending(), stream)`; `_sync_timed` delegates with `mark=` (the T3.19a timed
split mirror now lives in the provider); `apply` calls `provider.check_forward_end()` after the kt
sync of the last MoE layer (`num_layers - 1`).

Decisions taken without the owner, open to veto: (1) `ExpertSubmission` carries the `hidden_states`
reference so `wait()` can match the ledger entry; (2) the buffer-lease identity is checked at every
Python-side submit of a captured size, which guards the registry the replays were captured against,
not the replay itself (replays run no Python); (3) the two C3 negative tests live in
`test_kt_provider.py` rather than in `test_kt_deferral_adjacency.py` (unchanged, still 9).

## Tests (CPU-only, in `ci.sh`)

- `scripts/tests/test_kt_provider.py` via `run-kt-provider-tests.sh` (in-image, kt C++ MoE and
  `CPUInfer` replaced by recorders): **17/17**. Six conformance tests run twice, once against a
  `FakeProvider` and once against `KtKernelProvider`; five kt-only tests assert the recorded kt call
  sequence is unchanged versus the pre-seam direct path, the timed wait keeps kt's wait-then-H2D
  order, a lease detects a replaced buffer, forward-end sees kt's own pending flags, and wrapper
  validation rejects a mismatched layer / hidden / top_k / D. Includes the two C3 negative tests
  (`conf_adjacency_is_enforced`, `conf_forward_end_is_checked`).
- `scripts/tests/test_kt_timeline.py` updated to find the timed mirror in `KtKernelProvider.wait`
  and to assert `_sync_timed` delegates (59 checks, 0 failed).
- `scripts/tests/kt-parity-replay.py check` via `run-kt-parity-check.sh`: re-reads `parity/` and
  fails `ci.sh` if any pair is not bit-equal (exit 3 if the evidence is missing).
- Pre-existing suites unchanged and green: buffer lifetime 8/8, contract 14/14, deferral adjacency
  9/9, split seal 7/7, DSpark overlap gate 7/7. `ci.sh` rc = 0.

## T3.16e(3) parity: graph replay vs eager kt, bit-equal (measured)

The T3.19a dumps could not be replayed because layer L's output under D=4 depends on the deferred
experts of layer L-1, which the single-layer dump did not carry. This serve dumped **adjacent pairs**
(13, 14) and (19, 20) under `KT_TIMELINE=1 KT_TIMELINE_DUMP_LAYERS=13,14,19,20` (`dumps/`, 28 rank-0
files, 2.4 MB: `decode-{1..4}-candidate_all` n = 0 (post-capture check replay) and n = 1 (real
traffic, bs 1 and 4), plus the eager warm-up forward). `kt-parity-replay.py run` loads the 158 CPU
experts of each pair into a fresh `KTMoEWrapper` in the image (GPUs free, no server), feeds the
recorded layer L-1 `hidden_states` / `topk` so the deferred slots are populated exactly as they were
in the serve, then replays layer L and compares `kt_output` to the dump.

| | pairs | bit-equal | max abs diff |
|---|---:|---:|---:|
| `parity.json` (replay through L-1 then L) | 14 | **14 / 14** | 0.0 |
| `control.json` (layer L alone, L-1's deferred experts missing) | 14 | 1 / 14 | up to 0.100 (5,079 of 5,120 elements differ at bs 1) |

The one equal control case (`decode-3`, layer 20) had `prev_cpu_slots = 3` with every slot
protected, so nothing was deferred into layer 20 and the control is legitimately identical. The
other 13 differ, which is what makes the 14/14 result meaningful rather than vacuous. kt is
deterministic (E29), so the gate is bit-equality with no tolerance.

## Per-step gate on the plain serve (`KT_TIMELINE` unset)

Qualified profile, `scripts/serve-hybrid.sh` with no extra env: gpu = 226 cpu = 158 D = 4 cpuinfer
= 16 chunk = 4096 mem_fraction = 0.96 context = 32768 slots = 4 max_total_tokens = 147456
capture_bs = [1 2 3 4] dspark = 0 (`launch.txt`). GPU power cap 325 W, BIOS slot 3 (FCLK 2100).

| check | pin (`evidence/t319/baseline-pin.json`) | T3.19b measured | gate | result |
|---|---:|---:|---|---|
| C1 median / wall tok/s, TTFT (`c1-off6.json`) | 41.92 / 39.86 / 0.609 s | **42.04 / 40.15 / 0.594 s** | wall >= 39.0, TTFT <= 0.65 | pass (+0.7 % wall) |
| C4 aggregate / wall, per stream (`c4-off0.json`) | 58.09 / 55.79 / 13.9 | **57.86 / 55.51 / 13.88** | >= 13.6 / stream | pass |
| NLL vs `nll-t316c.json` (`nll-t319b.json`) | | **+0.00000, 0 / 0 of 4795 tokens moved** | 0 moved | pass |
| consistency `--same-prompt` C4 (`consistency.json`) | | PASS, 3 of 4 rows exact, min pairwise prefix 225 chars | PASS (>= 40) | pass (225 is inside the accepted C4 range 155..310 of every prior same-prompt C4 file; T3.19a's 519 was the outlier) |
| NaN / Xid / Traceback / `KTContract` / `Sealed` lines (`launch-gate-container.txt`) | | 0 | 0 | pass |
| EDAC ce / ue | | 0 / 0 | 0 / 0 | pass |

This is the same serve as the E33 context-ladder 32K rung (single slot and 4 slots at 32K are both
covered by the C1 and C4 rows above; the needle run at 32K is the T3.6 verify already on record).

The dump serve (`launch-dumpserve-container.txt`, `verify-dumpserve.json`: consistency PASS, 0
health hits) ran the full provider path with the timeline on; it is the parity input, not the gate.

## Reading

1. The seam costs nothing measurable at the step level: C1 wall 40.15 vs 39.86 pinned, C4
   13.88 vs 13.9 per stream, both inside the run-to-run band. The kt call sequence is pinned by
   test to be identical, so this is expected, not a result.
2. The deferral contract (C3) is now enforced from Python independently of kt's flag dict and the
   two must agree at the forward end; both negative tests fail without the asserts.
3. The graph-captured decode path and the eager kt path produce bit-identical expert output for
   the recorded steps at bs 1..4 (14/14), with a control that shows the comparison would catch a
   missing deferred slot. This closes T3.16e(3) and the "graph vs eager" item of the review.
4. Everything that T3.22 (multi-rank) would need to replace is now a named call on one object;
   T3.19c moves the registry and the remaining class-wide state onto the runner.

Files: `c1-off6.json`, `c4-off0.json`, `c1-bench.txt`, `c4-bench.txt`, `nll-t319b.json`,
`consistency.json`, `launch.txt` (redacted launcher log), `launch-gate-container.txt` (redacted
container log of the gate serve), `launch-dumpserve-container.txt`, `verify-dumpserve.json`,
`dumps/`, `parity/parity.json`, `parity/control.json`, `parity/replay-*.pt`.

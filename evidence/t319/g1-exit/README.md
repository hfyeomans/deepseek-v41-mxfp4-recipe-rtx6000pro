# G1 exit: Stage 1 (T3.19) exit gate on the final Stage 1 overlay commit (2026-10-08)

Closes the [gate ledger](../../../docs/research/MIGRATION-ROADMAP-PYTHON-TO-NATIVE-HYBRID.md#gate-ledger-added-2026-09-24-the-enforced-sequence)
G1 exit checklist: the stale GPU harnesses refreshed and re-run in the image (`harness/`), and one
C1..C4 soak block plus the full per-step gate on overlay `bc73ef8d` (T3.19d, the final Stage 1
commit) with launcher defaults (`soak/`). Review pass 2 is §11 of
[`MAINTAINABILITY-REVIEW-2026-09.md`](../../../docs/research/MAINTAINABILITY-REVIEW-2026-09.md).
No overlay change; recipe-side harness refresh only.

Serve: `scripts/serve-hybrid.sh` defaults, `resolved: experts gpu=219 cpu=165 deferral D=4 cpuinfer=16
chunk=8192 mem_fraction=0.96 context=65536 slots=4 max_total_tokens=278528 capture_bs=[1 2 3 4]
dspark=0` (`soak/launch.txt`), image `dsv41-kt:recipe` id `0ddb14aa…`, BIOS slot 3 (FCLK 2100,
HSMP: FCLK 2100 / DDR5-6400 on all 612 samples), GPU cap 300 W. Box state before the serve: owner ran
`drop_caches` + swap off/on (235 GB free, 1 GB cache, swap 0). Every JSON under `soak/` carries the
serve's manifest hash (`2d54b246…`, `soak/manifest.json`, container check `match`): this is the
first evidence set stamped end to end by T3.19f.

## harness/ — the three GPU harnesses off the removed compact-loader monkey-patch (R7)

T3.19c replaced `_install_kt_mxfp4_loader_compact` / `_KT_LOADER_GPU_MASK` with the explicit
`CompactMXFP4Loader`; `scripts/test-kt-one-layer.py`, `scripts/kt-export-bench.py` and
`scripts/tests/kt-parity-replay.py` (dump mode) now build their CPU experts through
`KtRuntime(tp_rank=0).load_cpu_experts(...)` and were re-run in the image at the current default
split (219 / 165, layer 5, 16 threads).

| harness | file | result | prior (`evidence/t36b-kt-one-layer/layer5-compact-gpu224.json`, 224 GPU experts, monkey-patched loader) |
|---|---|---|---:|
| one-layer kt vs torch reference, 8 tokens all-CPU routing (worst case) | `one-layer-219.json` | rel-L2 **0.00387**, max abs 0.0256, finite; T=1 **0.777 ms** / layer (0.723 min), T=4 0.907 ms/token, T=8 0.919 ms/token; RSS 11.0 GiB | rel-L2 0.00377, T=1 0.835 ms |
| compact export + H2D, compact experts 0 / 1 / 164 (global 219 / 220 / 383), both ranks | `export-bench-219.json` | **byte-identical** w13 / w2 weights and bf16 scales on both ranks (`byte_compare_all_identical: true`); streamed vs resident kernel output max abs diff **0.0**; export 54.3 GB/s (0.367 ms / expert), H2D 44.5 GiB/s per card, pipelined layer 39–41 GB/s at group 8–32 | identical, 51.9 GB/s (E22a `e22a-export-bench-t16.json`, 208 GPU experts) |

The one-layer rel-L2 moves 0.00377 → 0.00387 because the reference compares a different expert
set (165 compact experts instead of 160; the eight all-CPU tokens route to different ids), not
because the loader changed: the export check above is bit-identical against the resident tensors.

## soak/ — C1..C4 block + per-step gate, 141,312 decoded tokens, health 0, EDAC 0 / 0

`soak-g1.sh` ran unattended: same-prompt consistency at concurrency 1..4, the eight E27 Boot 12 /
E34a bench blocks (12 runs × 512 tokens each, distinct prompts, seed 4242), the NLL gate, prefill
16K cold + warm (T3.19f), 4 × 62K stress and the 64K needles. Health line (NaN / Xid / Traceback /
KTContract / Sealed in the container log) was 0 after every block and at the end; EDAC CE / UE 0 / 0.
GPU samples (2 s, 3,056 rows): peak 197 W, 78 °C, 95,955 MiB used (the 4 × 62K stress; 1.9 GiB
min free). HSMP: FCLK 2100 / MCLK 3200 every sample, DRAM up to 200 GB/s, socket 103–312 W.

| block | C | summed tok/s (min / max) | wall tok/s | per stream | TTFT s | E34a soak (same blocks, 219 / 165 chunk 4096, 2026-10-08) | Boot 12 soak (226 / 158, pin) |
|---|---:|---:|---:|---:|---:|---:|---:|
| c4-off0 | 4 | 55.15 (53.82 / 55.69) | 52.94 | 13.79 | 1.589 | 58.02 / 55.78 | 58.09 / 55.79 |
| c3-off2 | 3 | 53.89 (53.48 / 54.20) | 52.38 | 17.96 | 0.877 | 56.06 / 54.48 | 56.00 / 54.38 |
| c2-off4 | 2 | 50.75 (50.57 / 51.15) | 48.45 | 25.37 | 0.978 | 52.10 / 49.72 | 52.12 / 49.80 |
| c4-off8 | 4 | 55.69 (55.35 / 56.06) | 53.52 | 13.92 | 1.564 | 57.84 / 55.53 | 58.06 / 55.80 |
| **c1-off6** | 1 | **41.43** (41.35 / 41.64) | **39.51** | 41.43 | **0.618** | 41.89 / 39.94 | 42.02 / 40.08 |
| c4-off12 | 4 | 57.06 (56.76 / 57.30) | 55.58 | 14.27 | 1.030 | 58.96 / 57.40 | 58.97 / 57.39 |
| c3-off9 | 3 | 53.67 (53.18 / 54.21) | 52.22 | 17.89 | 0.847 | 55.80 / 54.30 | 55.63 / 54.14 |
| c2-off0 | 2 | 48.81 (48.38 / 49.31) | 46.30 | 24.41 | 1.165 | 50.47 / 47.92 | 50.27 / 47.73 |

The C2–C4 blocks sit 2.5–5 % under the E34a / Boot 12 columns; those two soaks ran at chunk 4096
(E34a) or 226 / 158 + chunk 4096 (Boot 12), i.e. before the T3.29 default change that traded seven
resident experts and a bigger chunk for prefill (T3.28: C4 56.3 / 54.2 on the new default). Against
the post-T3.29 same-profile reference (`evidence/t319/d/README.md`, serve 1: C1 41.35 / 39.41 /
0.636 s, C4 55.99 / 53.74 / 14.0 per stream) this soak is C1 +0.2 % / +0.3 % wall, C4 −1.5 % wall
at off0 and −0.4 % at off8 — inside the run-to-run band. Stage 1 promised no performance change
and delivered none.

### Per-step gate (`baseline-pin.json` `per_step_gate`)

| check | file | measured | gate | result |
|---|---|---|---|---|
| consistency `--same-prompt` C1..C4 | `consistency-c{1..4}.json` | PASS / PASS / PASS / PASS | PASS bs 1..4 | pass |
| C1 wall tok/s | `c1-off6.json` | **39.51** (pin 39.86 → −0.9 %; post-T3.29 ref 39.41 → +0.3 %) | ≥ 39.0; within 2 % of pin | pass |
| C4 per-stream tok/s | `c4-off0/8/12.json` | **13.79 / 13.92 / 14.27** (pin 13.9) | ≥ 13.6; within 2 % | pass |
| TTFT C1 | `c1-off6.json` | **0.618 s** (pin 0.609) | ≤ 0.65; within 0.05 s | pass |
| NLL, pooled 65,653-token corpus vs same-shape baseline `evidence/t328/default-219-8192/corpus/` | `corpus/pooled.json`, `nll-corpus.out` | **ΔNLL −0.00000 ± 0.00000, 0 tokens changed, max abs Δlogp 0.000** on all 8 texts | 0 moved | pass |
| NaN / Xid / Traceback / KTContract / Sealed; EDAC | `soak.out` | 0 after every block; 0 / 0 | 0; 0 / 0 | pass |
| prefill 16K cold / warm (T3.19f pair) | `prefill-16k.json` | cold (seed 777, shards 7.1 % resident) **1,482 tok/s** / 10.85 s; warm (seed 1277) **1,600 tok/s** / 9.99 s | within 5 % of the T3.28 curve (1,504 cold) | pass (−1.5 %) |
| 4 × 62K stress | `stress-kv.json` | 4 / 4 finished, 61.8–62.0K prompt tokens each, 256 completion, wall 172 s, TTFT 44–154 s (serialised prefill), peak 95,894 / 95,953 MiB | all finish, no OOM | pass |
| 64K needles + API checks | `verify-64k.json` | ALL PASS: models alias, context ceiling, arithmetic, thinking toggle, tool call, multi-needle 6 / 6 at 55,206 tokens (TTFT 31.4 s) | ALL PASS | pass |

**NLL comparator note.** `soak-g1.sh` first ran the pre-T3.28 single-fixture comparison
(`nll.json`: 4,795-token text vs `evidence/t316-review-intake/nll-t316c.json`, the 226 / 158 +
chunk 4096 pin). That reads +0.0067 nats/token with 4,776 of 4,795 tokens changed, which is the
documented shape-changing shift of the 219 / 165 + chunk 8192 default (E34d-iii, T3.28: +0.0067 on
this very text), not a Stage 1 regression, and it is not the gate. The gate since T3.28 is the
pooled corpus against the *same-shape* baseline, run on the same serve right after the soak
(`nll-corpus.out`): 0 of 65,653 tokens moved, as T3.19c and T3.19d also got. The pin file's `nll`
reference (`nll-t316c.json`) is superseded for same-shape checks by
`evidence/t328/default-219-8192/corpus/`; `baseline-pin.json` records this in the new confirmation
entry rather than by re-pinning (image and BIOS slot are unchanged).

## Verdict

Every G1 exit checkbox is evidenced: a..f have their evidence dirs (`evidence/t319/{a,b,c,d}/`,
T3.19e goldens in `scripts/tests/`, T3.19f manifests in this set); R1 / R5 re-measured in review
§11; the soak block on the final Stage 1 commit is clean at 141K decoded tokens; the pooled NLL is
bit-identical; C1 wall / C4 per-stream / TTFT are within the pin's 2 % / 0.05 s band. G1 is `done`;
T3.20 / T3.21 (G3) may start.

## Files

`harness/`: `one-layer-219.json`, `one-layer.log`, `export-bench-219.json`, `export.log`.
`soak/`: `soak-g1.sh`, `soak.out`, `launch.txt`, `manifest.json`, `consistency-c{1..4}.json`,
`c{1..4}-off*.json` (8), `nll.json` (superseded comparator, kept), `corpus/` (8 per-text JSON +
`pooled.json`) + `nll-corpus.out` (the gate), `prefill-16k.json`, `stress-kv.json`,
`verify-64k.json`, `gpu-during-soak.csv`, `hsmp-during-soak.txt`.

# T3.51 — #35257 upstream prep: the exact `topk_v2` was wrong on the cluster path (E45), 2026-10-12

Question (item (0) of T3.51): what does the "61/61 exact and stable" probe behind the #35257 draft
actually prove, and which kernel paths does it leave untested? The served kernel (patch 0055,
T3.15d) has two refinement paths: `refine_and_collect` (register / streaming templates) and
`cluster_refine` (thread-block clusters, DSMEM). Upstream compiles the cluster path on SM90 / SM100
only (`SUPPORT_CLUSTER`, excluded on SM120 for "poor cluster performance"), so every probe and serve
on this machine had exercised the first path alone.

Answer: the cluster path was broken in two ways, and forcing `SUPPORT_CLUSTER` on SM120 is a valid
gate for it — the stock kernel passes the whole upstream suite that way (288, `main-stock-forced-cluster.txt`),
so the exclusion is about speed, not correctness. The first patch version failed 156 of 302 cases
forced (`main-patched-forced-cluster.txt`); the fixed one passes 302 forced and 302 default
(`main-patched-forced-cluster-v2.txt`, `main-patched-v2.txt`) and the probe is 72/72 exact and
stable forced (`probe-forced-cluster-v2.txt`; stock forced 13/72, `probe-forced-cluster-stock.txt`).
Production on SM120 was never affected — the recipe never ran the cluster path — but the upstream
patch would have been wrong on H100 / B200.

## Root causes (printf instrumentation, `cluster-fix.diff`)

1. `cluster_refine` folded every warp's per-thread key AND / OR into one 64-bit word in rank 0's
   shared memory with `atomicAnd` / `atomicOr` over DSMEM from all warps of all ranks. Updates were
   lost (the AND returned bits no candidate shared, e.g. `…ffff0400` vs the correct `…ffff0000`), so
   rounds could not narrow the key, the collect emptied, and `handle_tie` padded with indices 0…511
   or dropped / duplicated the lowest-index survivors. Fix: each rank folds locally (local atomics
   into its own `tie_handle.key_and/or`, reset per round) and thread 0 writes one slot of a per-rank
   array in rank 0's smem; rank 0 reduces the N slots once after the cluster barrier.
2. Peers read rank 0's final `count_gt` / `count_eq` after the third barrier to decide whether to
   refine, while rank 0 could already have exited (small-batch INDICES mode — the production path,
   `scoring.py:449`) or started its next item (persistent kernel); DSMEM of an exited CTA →
   "unspecified launch failure" in `test_topk_v2_ragged[False-64-persistent-*]`. Fix: `forward()`
   zeroes the counts after the first barrier wait instead of before the arrive, and every cluster
   kernel ends on a cluster barrier (`cluster.sync()` in all modes of the small-batch kernel,
   `this_cluster().sync()` in the persistent one).

## Cost (forced cluster, SM120, µs per call, median of 50; `probe-forced-cluster-*.txt`)

| row | stock | patched |
|---|---|---|
| `randn` (tie-free) L = 262,147 / 65,537 / 20,000 | 25.2 / 22.1 / 12.5 | 27.3 / 24.2 / 12.8 |
| zero tie (ReLU, < k positives) 262,147 / 65,537 / 20,000 | 19–20 / 16–17 / 15–17 | 79–80 / 62–67 / 27–30 |
| one value × 2,049…60,000 at 262,147 | 23–25 | 79–81 |
| dense bin (> 2,048 distinct in one fp16 bin) 262,147 / 65,537 | 19.5 / 16.6 | 101.9 / 73.3 |

The streaming refinement at 262,147 costs ≈ 200 µs for the same overflow rows (E41 / the #35257 draft),
so the cluster path is the cheaper exact one where it compiles. Stock is faster on overflow rows
because it answers them wrongly (59 of 72 probe cases are not the exact set or move between repeats).

## Files

- `main-stock-forced-cluster.txt` — upstream main `e6072935`, stock kernel, `SUPPORT_CLUSTER` forced on SM120: 288 passed (the gate is valid).
- `main-patched-forced-cluster.txt` — first patch version, forced: 156 failed / 147 passed.
- `main-patched-forced-cluster-v2.txt`, `main-patched-v2.txt` — fixed patch, forced and default: 302 passed, 19 skipped each (288 upstream + 15 added − 1 skip).
- `main-stock-regression.txt` — the added `test_topk_v2_exact_under_ties` on the stock kernel: 14 failed, 1 skipped (it does catch the defect).
- `overlay-fixed.txt`, `overlay-fixed-ties.txt` — the overlay tree (`deec7a6f`, patch 0061) on the overlay's test file: 288 + 14 passed.
- `probe-forced-cluster-stock.txt`, `probe-forced-cluster-v2.txt` — `topk-v2-exact-probe.py` (72 cases; L = 20,000 added vs the 61-case run behind the first draft), forced cluster.
- `cluster-fix.diff` — the kernel fix on top of the first patch version (the overlay commit `deec7a6f` and patch 0061 carry the same change plus the regression test).

Harness: `docker run --rm --gpus '"device=0"' --entrypoint python3 -v <sglang python dir>:/sgl-workspace/sglang/python/sglang:ro -v <test dir>:/tmp/test:ro dsv41-kt:recipe -m pytest -q /tmp/test/registered/kernels/ops/attention/test_topk_v2.py` (JIT recompiles per run, ≈ 30 s). Forcing the path = `#define SUPPORT_CLUSTER (SGL_CUDA_ARCH >= 900)` in `topk_v2.cuh`; the device probe then reports `SGL_TOPK_V2_MAX_C8_OCC2=46`, `MAX_C16_OCC1=10`.

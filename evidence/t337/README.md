# T3.37 — the "per-chunk prefill drops ≈ 17 % at the 1M-token pool" gap (2026-10-11)

Registered 2026-10-09 from the lever soak's per-chunk reading (`evidence/t331/64k4-lever-soak/`):
on the same seed-777 32K prompts the steady chunks ran 1,939–2,020 tok/s at the 278,528-token pool
(64K × 4) and 1,519–1,638 at the 1,064,960-token pool (262K × 4), so "something on the chunk path
scales with `max_total_tokens`". The task was to profile one chunk at both pools, name the kernel
that scales with the pool, then make the smallest overlay change. Go bar: steady chunks ≥ 1,850
tok/s at the 1M pool with NLL 0 moved.

**Result: the gap no longer exists on the current overlay (`05f9e88b`, 52 patches), and the go bar
is passed by 2.4×.** Three serves, same prompts, same day, launcher defaults everywhere except
`CONTEXT_LENGTH=262144` on the 262K × 4 serves:

| serve | pool (tokens) | 32K prefill, fresh serve (seed 777 warm, then seed 1277 × 2) | steady chunks, seed-1277 32K (container log) |
|---|---|---|---|
| 64K × 4, `64k4/` | 278,528 | 3,481 / **3,293** tok/s | 3,500–4,200 |
| 262K × 4, `262k4/` | 1,064,960 | 3,496 / **3,290** tok/s | 3,500–4,200 |
| 262K × 4 after the pool was filled (stress 4 × 260K), `262k4-filled/` | 1,064,960 | **4,507 / 4,147** (seed 1277), 4,147 (seed 2277); after `/flush_cache` 4,055 | **4,400–4,960** |

The two pools are identical chunk for chunk (3,293 vs 3,290 tok/s on the same seed), and the GPU
traces of the profiled seed-1777 pass are identical to the kernel: `analysis/kernels.py` over the
two rank-0 traces gives span 9,174 vs 9,120 ms, GPU busy 13,981 vs 13,954 ms (sum over streams),
and no kernel whose total differs by more than 34 ms (the all-reduce, −0.7 %). Nothing on the chunk
path scales with the pool: `req_to_token`, the DSA indexer, the KV-write scatter and the
post-capture KV placement all show the same time at 278K and 1,064,960 tokens.

## Where the 17 % went

The 2026-10-09 reading came from the idea-8 serve (`evidence/e34-indexer-memory/d-v-a/262k-4slot-idea8/`,
overlay `30e8c412`), whose container log holds **114 `expandable_segments: memory mapping failed with
OOM` warnings with `free:` 1–20 MiB on device 1 during the prefill** — the allocator was on its
retry path on every chunk. The 64K × 4 serve the same day had 0 such warnings and 1.2 GiB more
headroom. Since then T3.39 window 1b (`99128a7e`, 2026-10-10) gave the hybrid one FlashInfer runner
per expert count, which ended the per-call MoE-workspace recompute *and* re-allocation that was
driving the near-full pool into those retries; today's 262K × 4 serves log **0** mapping failures
at the same `MEM_FRACTION_STATIC`, and the 4 × 260K stress that took 678 s on 2026-10-09 takes 268 s.
So the gap was the allocator's near-OOM fallback at the large pool, exposed by the workspace churn,
not a pool-sized kernel. Caveat: the pre-fix pair was not re-run on the old overlay; the attribution
rests on the 114 → 0 warning count and the identical traces, not on a bisect.

## Second reading: the first requests of a serve are 15–25 % slower than a warm box

Fresh serve → 3,290–3,500 tok/s at 32K; after ~1M tokens of traffic → 4,150–4,500 on the same
prompts, and still 4,055 after `/flush_cache` (so it is not the radix cache). The level persists
for the serve, so `release-smoke.sh --full` (stress first, benches after) measures the warm box;
`bench-prefill.py` on a fresh serve under-reads it by ~20 %. BENCHMARKS numbers are warm-box numbers
by construction; a one-off prefill bench on a fresh serve is not comparable to them.

## 262K × 4 qualification on the current overlay

`262k4-smoke/`: `release-smoke.sh serve --context 262144 --slots 4 --full` on the `262k4-filled`
serve (first full-gate run of the rung since `99128a7e` and patch 0052). See the table below.

| gate | 262K × 4 today (`262k4-smoke/`) | 64K × 4 launcher default (`../t343b/64k4/`, same overlay family) |
|---|---|---|
| stress 4 × 260,096 tokens | wall **241 s**, all slots filled; peak 97,222 / 97,291 MiB, min free 665 / 596 MiB; **0** `memory mapping failed` (the 2026-10-09 serve: 678 s, 114 failures) | n/a (64K stress) |
| same-prompt C4 | pass | pass |
| NLL corpus (4 texts, 65,653 tokens) vs `t343b/64k4/corpus` | **identical: 0 tokens moved**, max\|Δlogp\| 0.000; vs the 219/8192 anchor \|z\| ≤ 1.86 | identity baseline |
| needles at 262,144 | **6 / 6** at 255,388 prompt tokens, TTFT 59.5 s; arithmetic / thinking / tool call pass | 6 / 6 at 64K, TTFT 14.1 s |
| decode C1 (6 runs × 512, distinct prompts) | **57.21** tok/s median (55.99–57.82), TTFT 0.33 s | 59.08 (56.06–59.34), TTFT 0.30 s |
| decode C4 (3 runs × 512) | **116.78** (116.24–120.52), TTFT 0.77 s | 114.54 (113.25–119.84), TTFT 0.75 s |
| prefill 16K / 32K (seed 777, warm box) | **4,101 / 4,452** tok/s, TTFT 3.90 / 7.19 s | 3,792 / 3,981, TTFT 4.24 / 8.05 s |
| power (500 ms sampler) | decode C1 422 W total busy-median, SM 2,280–2,325 MHz, max 83 °C, 0 throttle samples | — |
| health | 0 error lines, Xid 0 | 0 |

Every gate passes; C1 is −3 % and C4 +2 % against the 64K × 4 run two days earlier (run-to-run band),
prefill is +8 / +12 %, NLL is bit-identical, and the rung sits 600 MiB above OOM at the stress peak
with the allocator never on its retry path. `262k4-soak/`: the E27-shape soak (8 blocks × 12 runs ×
512 tokens at C4 / C3 / C2 / C1, distinct fixture offsets) on the same serve — the owner's
precondition for a default change.

## 262k4-soak (E27 shape, same serve, 08:22–08:49)

8 blocks × 12 runs × 512 tokens, distinct fixture prompts, `warmup 0`, health check after every block
(`analysis/soak.sh`). Summed median tok/s (per-stream rates are identical within a batch):

| Block | C | summed median | min | max | wall tok/s | TTFT s |
|---|---|---|---|---|---|---|
| c4-off0 | 4 | **110.09** | 104.62 | 114.29 | 105.36 | 0.85 |
| c3-off2 | 3 | 88.68 | 86.87 | 99.15 | 86.18 | 0.52 |
| c2-off4 | 2 | 70.44 | 67.86 | 71.93 | 67.90 | 0.55 |
| c4-off8 | 4 | **109.75** | 106.86 | 113.63 | 104.97 | 0.85 |
| c1-off6 | 1 | **56.84** | 55.16 | 58.32 | 54.86 | 0.34 |
| c4-off12 | 4 | 95.78 | 88.83 | 103.26 | 93.13 | 0.63 |
| c3-off9 | 3 | 94.44 | 88.14 | 103.11 | 92.27 | 0.42 |
| c2-off0 | 2 | 95.73 | 87.71 | 98.31 | 90.55 | 0.55 |

141,312 tokens decoded; `health-hits.txt` 0 × 8, `mapping-failures.txt` 0, every stream finished by
`length`. The spread between blocks at the same concurrency (C4 95.8–110.1, C2 70.4–95.7) follows the
prompt set, not the clock (the fast C2 block ran last): under the hot-expert map a batch's step time
depends on how many of its tokens route to CPU-tier experts, where the positional-map lever soak
(`evidence/t331/64k4-lever-soak/`, same offsets) spread < 3 % at every C. Every block is above that
soak's figures (C4 56–58, C2 49–51, C1 41.6). Recorded for T3.8c: a per-prompt route recorder would
show which prompts pay, and whether a refit narrows the spread.


## Files

* `analysis/profile.sh` — the two profiling serves (warm → 2 unprofiled → CPU+GPU trace of one 32K pass, `num_steps` 6).
* `analysis/kernels.py` — per-kernel GPU-time diff of two traces.
* `analysis/filled-pool.sh` — the fresh → fill → measure → flush → measure serve.
* `64k4/`, `262k4/` — launch, ready, bench JSON / txt, traces (`*-TP-{0,1}.trace.json.gz`, 17 MB each, kept locally and git-ignored; `analysis/kernels.py` output is the committed summary), prefill log, manifest.
* `262k4-filled/` — same plus `stress-kv.*`, `prefill-32k-{fresh,filled,filled-newseed,flushed}.*`, `flush.txt`.
* `262k4-smoke/` — the release-smoke evidence directory.

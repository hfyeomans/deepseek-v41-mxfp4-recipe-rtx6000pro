# T3.41 stock control serve (SGLANG_DSV41_FP8_GEMV=0) and CPU-thread pinning probe

Launcher default (219/165, chunk 8192, el8-mix, 262K×4, KT_CPUINFER=16), `PROFILE=developer`,
same image/overlay as `../serve-gemv/` (manifest hash identical). Decode = `bench-decode.py`
512 tokens, 3 runs + 1 warmup, distinct prompts, seed 4242; prefill = `bench-prefill.py` 16K, 3 runs.

## Decode (stock, no pinning)

| file | C | median tok/s | note |
|---|---|---|---|
| `c4-212904.json` | 4 | 106.04 | slow mode |
| `c1-213022.json` | 1 | 59.48 | |
| `c4-213058.json` | 4 | 123.55 | fast mode |
| `pre-pin-c4-1.json` / `-2.json` | 4 | 123.63 / 122.54 | fast mode |

C4 on this launcher is bimodal (≈106 or ≈123 tok/s) **independent of the GEMV flag**; the
GEMV serve showed the same two modes (108/123). Root cause is below.

## Root cause: floating host threads collide with the pinned kt-kernel workers

kt-kernel pins its 16 workers (`numa_0_t_1..15`, `numa_0_m_0`) to logical CPUs 0–15 (one per
physical core, hwloc singlify). Every other container thread (TP0 scheduler, the ~100 % spinning
TP1 main thread, tokenizer, HTTP) keeps mask `0xffffffff` and floats; whenever one lands on CPUs
0–15 it time-shares with a worker and the CPU-tier fork-join stalls.

* `forced-slow-c4.{json,txt}`: TP0 main pinned to cpu5 and TP1 main to cpu2 (both inside the
  worker set) → C4 **33.80 tok/s**.
* `pinned-log.txt`, `pinned-c4-{1,2}.json`, `pinned-c1.json`: all non-`numa_*` threads pinned to
  `0xffff0000` (the SMT siblings, CPUs 16–31) → C4 **125.05 / 125.29**, C1 **60.38**
  (min 60.19, max 60.54). Fast mode became the only mode.

Pin helper used (host side, no container rebuild):
`docker exec` → for every `/proc/<pid>/task/*` whose comm is not `numa_*`, `taskset -p <mask> <tid>`.

## Prefill 16K (fresh seed per pass; 3 runs, cold)

| file | threads | seed | median tok/s |
|---|---|---|---|
| `pinned2-prefill-16k.json` | pinned 16–31 | 5777 | 3,711 |
| `unpinned2-prefill-16k.json` | floating | 6777 | 3,789 |
| `pinned3-prefill-16k.json` | pinned 16–31 | 7777 | 3,812 |
| `unpinned3-prefill-16k.json` | floating | 8777 | 3,859 |

Prefill is neutral to −2 % with pinning; the sequence drifts upward monotonically
(Engram residency 19 → 24 % across the four passes), so the gap is within the warm-up drift.

**`pinned-prefill-*.json` / `unpinned-prefill-*.json` are INVALID as an A/B**: they reuse the
same seeds and sizes inside one serve, so the second pass hit the radix cache (4,620 tok/s is a
cache hit, not prefill). Kept only as the record of the mistake; use the `*2`/`*3` files.

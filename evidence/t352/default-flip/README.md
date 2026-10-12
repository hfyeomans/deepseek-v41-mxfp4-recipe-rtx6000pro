# T3.52 default flip — the plain launcher after `HOST_THREAD_CPUS` defaulted to `auto`

2026-10-12, owner approval ("flip default to auto") after the `evidence/t352/` qualification. One
qualified-profile serve with **no overrides** (`resolved: … host_thread_cpus=16-31 … profile=qualified
overrides=[]`), overlay `deec7a6f`, image `dsv41-kt:recipe`, manifest `manifest.json`; the ladder
rule's one-default-serve-per-flip. This directory is the headline row of `docs/BENCHMARKS.md`
(`config/benchmarks-headline.json`), replacing `evidence/t315d/serve-a`.

Affinity (`thread-affinity.txt`): every non-kt thread on `16-31`, the 16 `numa_0_*` workers on one
CPU each of 0–15 — the launcher default now does what the qualification serve did with the override.

## Release gate: 12 / 12 PASS (`smoke-log.txt`)

stress 4 × 253,428 tokens (wall 278 s, all finished, min free 653 MiB), same-prompt C4, NLL
ΔNLL −0.00033 ± 0.00101 (z +0.33) vs the anchor and **0 of 65,653 tokens moved** vs
`evidence/t352/corpus`, needles 6/6 at 255,388 (TTFT 59.5 s), decode C1 / C4, prefill 16K / 32K,
power summary, container log, health 0 (`health.txt`).

| metric | default flip (this serve) | qualification `evidence/t352/` (override) | previous headline `t315d/serve-a` (floating) |
|---|---|---|---|
| C1 decode tok/s | **60.75** (58.41–61.84); rerun 59.89 | 60.69 / 60.30 | 60.1 |
| C4 decode tok/s | **123.99** (123.84–125.97); rerun 123.96 (123.39–127.85) | 126.73 / 123.02 | 124.8 (106 possible on a bad run, E44) |
| prefill 16K tok/s (TTFT) | **4,344** (3.69 s) | 4,329 | 4,361 (3.7 s) |
| prefill 32K tok/s (TTFT) | **4,495** (7.12 s) | 4,210 | 4,513 (7.1 s) |
| power busy-median total | C1 441 W, C4 394 W, 16K 496 W, 32K 508 W; max 85 °C; 0 throttle samples | | |

Six C4 runs 123.4–127.9, no 106 mode. Prefill is within 0.4 % of the floating headline serve on
both sizes, so the −1…−2 % paired cost measured in `evidence/t352/` sits inside the serve-to-serve
band at the headline level (the paired A/B/A/B remains the honest number for the knob itself).

## Files

`launch.log`, `wait.txt`, `smoke-args.txt`, `manifest.json` (unredacted by rule), `vram-after-capture.txt`,
`thread-affinity.txt`, `smoke-log.txt`, `smoke-rc.txt`, `stress-kv.*`, `same-prompt-c4.*`, `corpus/`,
`corpus.txt`, `verify.*`, `c1.*`, `c4.*`, `prefill-16k.*`, `prefill-32k.*`, `power-*.csv`,
`power-summary.json`, `power.txt`, `health.txt`, `c1-rerun.*`, `c4-rerun.*`, `run.sh`, `run-log.txt`.

# T3.52 — host-thread pinning (`HOST_THREAD_CPUS=auto`) qualification serve

One qualified-profile serve on the launcher default (219 / 165 el8-mix map, chunk 8192, D=4,
262K × 4, FUSED_FINALIZE=0 + SORT_DECODE_TOPK=1, FP8 GEMV off) with the single override
`HOST_THREAD_CPUS=auto`, which resolves on this host to `16-31` — the SMT siblings of the 16
physical cores kt-kernel's workers own. 2026-10-12, overlay `deec7a6f` (patch 0061 tree),
image `dsv41-kt:recipe`, manifest `manifest.json`.

Mechanism (E44 / `evidence/t341/serve-stock/NOTES.md`): kt-kernel pins its 16 workers to logical
CPUs 0–15 with hwloc regardless of the inherited mask, and every other container thread floated on
0–31; whenever the TP0 scheduler or the spinning TP1 main thread landed on a worker's CPU the
fork-join stalled and C4 read ≈ 106 instead of ≈ 123 tok/s. The knob starts the server under
`taskset -c 16-31 python3 …` (`scripts/serve-hybrid.sh`), so the inherited mask keeps every
non-kt thread on the siblings while kt re-pins its workers to 0–15 itself.

## Affinity, verified in the container (`thread-affinity.txt`)

All 335+ python3 / `sglang::scheduler` / gloo / NCCL / ZMQ / CUDA threads on `16-31`; the 16
`numa_0_*` kt workers on one CPU each of 0–15. The mechanism works as designed with no code in the
model path.

## Release gate (`smoke-log.txt`, `release-smoke.sh serve --context 262144 --slots 4 --full`)

Every gate PASS: stress 4 × 260,096 (4 × ~253.4K tokens, wall 273.9 s, all finished, min free
646 MiB), same-prompt C4, NLL corpus (pooled ΔNLL −0.00033 ± 0.00101 nats/tok vs the quality
anchor, z +0.33; identity **0 of 65,653 tokens moved** vs `evidence/t341/serve-gemv/corpus`),
needles 6/6 at 255,388 tokens (TTFT 59.5 s), decode C1 / C4, prefill 16K / 32K, power summary,
container log readable, health 0 error lines (`health.txt`).

| metric | this serve (pinned) | t315c default-flip (floating) | headline default (t315d, floating) | E44 pinned-from-host |
|---|---|---|---|---|
| C1 decode tok/s (6 runs, median) | **60.69** (59.78–61.55); rerun 60.30 (57.94–61.59) | 57.98 | 60.1 | 60.38 |
| C4 decode tok/s (3 runs, median) | **126.73** (125.49–127.46); rerun 123.02 (122.60–123.98) | 121.08 | 124.8 | 125.05 / 125.29 |
| prefill 16K tok/s (TTFT) | 4,329 (3.71 s) | 3,311 | 4,361 (3.7 s) | 3,711 / 3,812 |
| prefill 32K tok/s (TTFT) | 4,210 (7.61 s) | 3,480 | 4,513 (7.1 s) | — |
| power, busy-median total | C1 441 W, C4 398 W, 16K 495 W, 32K 496 W; max 85 °C; 0 thermal-throttle samples | | | |

No 106 mode in any of the 6 C4 runs (two passes): 125.49–127.46 and 122.60–123.98. The first C4
pass is the highest C4 median on the launcher-default shape under the distinct-prompt harness
(previous high: E44's pinned-from-host 125.29; older 131–147 figures in BENCHMARKS are same-prompt
or pre-219/165 serves).

`spec-consistency.py` (`spec-consistency.txt`, compared with `evidence/t341/serve-gemv/`):
copy-code 384/384, copy-table 384/384 (max |Δlogp| 0.00015), review 366/384 with 1 flip (0.625),
long12000 374/384 with 1 flip (0.75) — the same figures as the E43 spec-off baseline, so decode
numerics are untouched.

## The open question: is the pinned prefill −1…−2 % real? (`prefill-aba.sh`, `prefill-aba-log.txt`)

Paired A/B/A/B on this serve after the smoke (Engram warm), every non-`numa_*` thread re-pinned
from the host between passes, one fresh seed per pass (seeds 10 apart — see the note below), 3
runs each:

| pass | threads | tokens | median prefill tok/s | TTFT s |
|---|---|---|---|---|
| `aba-pinned1-16k` | 16–31 | 16,020 | 4,144 | 3.88 |
| `aba-floating1-16k` | 0–31 | 15,970 | 4,175 | 3.83 |
| `aba-pinned2-16k` | 16–31 | 15,994 | 4,140 | 3.87 |
| `aba-floating2-16k` | 0–31 | 15,955 | 4,190 | 3.80 |
| `aba-pinned1-32k` | 16–31 | 32,002 | 4,273 | 7.49 |
| `aba-floating1-32k` | 0–31 | 31,970 | 4,356 | 7.36 |

**Yes, it is real and small: −0.7 / −1.2 % at 16K, −1.9 % at 32K**, floating faster in all three
pairs with no drift to hide behind (the two pinned passes agree within 0.1 %). Mechanism (reading,
not measured): during prefill the kt workers do the expert export but the Engram gather, the
scheduler and the tokenizer are Python threads that now share 16 logical CPUs instead of 32; the
decode step has no such parallel host work, which is why decode only gains.

`prefill-aba-INVALID-log.txt` / `aba-INVALID-*.txt`: the first attempt used seeds 5101…5104.
`bench-prefill.py` seeds run *i* of a size as `seed + 1000 × size + i`, so consecutive seeds share
two of three prompts with the previous pass and the "20,000 tok/s" readings are radix-cache hits.
Kept as the record of the mistake; seeds for passes inside one serve must differ by more than
`--runs`.

## Verdict

Qualified. Pinning removes the C4 slow mode (≈ −14 % whenever it struck) for a C1 that is unchanged
(60.7 / 60.3 vs 60.1–60.7) and a prefill cost of 1–2 %. Recommendation to the owner: flip
`HOST_THREAD_CPUS` to `auto` as the launcher default.

## Files

- `launch.log`, `wait.txt`, `smoke-args.txt`, `manifest.json` (unredacted by rule), `vram-after-capture.txt`
- `thread-affinity.txt` — in-container `Cpus_allowed_list` per thread comm
- `smoke-log.txt`, `stress-kv.*`, `same-prompt-c4.*`, `corpus/`, `corpus.txt`, `verify.*`, `c1.*`, `c4.*`, `prefill-16k.*`, `prefill-32k.*`, `power-*.csv`, `power-summary.json`, `power.txt`, `health.txt`, `container-log.txt`
- `c1-rerun.*`, `c4-rerun.*` — second decode pass; `spec-consistency.*` — decode-numerics gate
- `prefill-aba.sh`, `prefill-aba-log.txt`, `aba-*.{json,txt}` — the pinned / floating prefill A/B/A/B
- `run.sh`, `run-log.txt` (truncated to its first line by the smoke's fd redirection; the gates are in `smoke-log.txt`)

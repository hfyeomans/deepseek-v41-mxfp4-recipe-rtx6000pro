# T3.15c default flip — one serve on the plain launcher (no overrides)

2026-10-11, after the owner approved flipping both determinism levers. `bash scripts/serve-hybrid.sh`
with nothing set: the resolved line reads `… expandable_after_capture=1 fused_finalize=0
sort_decode_topk=1 …` and the container env carries `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0`
`SGLANG_DSV41_SORT_DECODE_TOPK=1` (`launch.log`). Overlay `743f8fa9` (patch 0054), image
`dsv41-kt:recipe` (`314749b5a268`), 219 / 165 experts on the el8 mix map, chunk 8192, D=4,
KT_CPUINFER=16, 262K x 4, docker 29.9.0 / containerd v2.4.1. Ready in 15.3 s TTFT after load.

`IDENTITY_CORPUS=evidence/t315c/serve-2/corpus bash scripts/release-smoke.sh serve
--output-dir evidence/t315c/default-flip --context 262144 --slots 4 --full` (`smoke-log.txt`, rc 0):

| Gate | Result | Band / anchor |
|---|---|---|
| stress 4 x 260,096 | PASS, wall 320.4 s, min free 749 / 682 MiB | all slots finished |
| same-prompt C4 | PASS | |
| NLL corpus vs 219/8192 anchor | dNLL -0.00033 +/- 0.00101, z +0.33, cluster z +0.26, moved 65,226 | \|z\| <= 2 |
| identity vs T3.15c serve 2 | **0 of 65,653 tokens moved** | the registry flip is the EXTRA_DOCKER_ENV serve |
| needles at 262,144 | 6/6 at 255,388 tokens, TTFT 61.3 s | 6/6 |
| decode C1 | 57.98 tok/s (wall 56.17), TTFT 0.30 s | headline 57.2 / 55.3 |
| decode C4 per stream | 121.08 (wall 116.21) | headline 116.8 / 112.1 |
| prefill 16K | 3,311 tok/s, TTFT 4.85 s (runs 3,177 / 3,313 / 3,311) | see below |
| prefill 32K | 3,480, 9.20 s (3,480 / 3,523 / 3,327) | see below |
| power (busy median, both GPUs) | C1 430 W, prefill 16K 198 + 206 W, 32K 210 + 216 W; 0 thermal / 2–5 cap samples | 300 W Max-Q |
| health / Xid | 0 / 0 | |

Greedy drift on the default serve (`drift.txt`, 5 repeats each, cache flushed, compared with
serve 2's `drift-fresh.json`): bit-identical at 23–28, 5,172 and 20,554 prompt tokens
(`bitwise=True` everywhere; the T3.15d tie residual did not show in these 30 runs).

## Prefill: the smoke's figure is a serve-state figure (T3.47)

`rerun/` — the same serve a few minutes after the smoke, fresh seeded prompts (seed 778):
16K **4,520 tok/s** (TTFT 3.55 s), 32K **4,633** (6.90 s); seed 777 again (the smoke's prompts,
radix cache possibly warm) 4,580. Across four 262K x 4 serves on this shape the smoke's prefill
16K reads 3,311 (this) / 3,327 (serve 1) / 3,996 (serve 2) / 4,101 (t337) and the first-traffic
stress 320 / 328 / 286 / 241 s; the slow serves draw ~200 W busy per GPU where the fast ones draw
~235 W, at the same 2,150–2,165 MHz SM clock (`power-summary.json` here and in the other three
dirs), so the GPUs are waiting on the host, not throttling. The levers are not the cause: serve 2
ran them and matched t337 without them. Page-cache warmth of the Engram file after the ~160 GB
GPU-weight load is the leading candidate (`free -g` after this smoke: 2 GiB free, 222 GiB
buff/cache, 136 GiB shared). Registered as T3.47. The headline row carries the smoke's numbers
as measured; `rerun/` is not stamped into it.

## Files

`launch.log` (launcher + resolved line), `wait-ready.log`, `smoke-log.txt`, `smoke-args.txt`,
`manifest.json`, `stress-kv.*`, `same-prompt-c4.*`, `corpus/` + `corpus.txt` (the new
`IDENTITY_CORPUS` anchor of `release-smoke.sh`), `verify.*`, `c1.*`, `c4.*`, `prefill-16k.*`,
`prefill-32k.*`, `power-*.csv`, `power-summary.json`, `health.txt`, `vram-after-capture.txt`,
`drift.*`, `rerun/prefill-{16k,32k}-seed778.json`, `rerun/prefill-16k-seed777.json`.

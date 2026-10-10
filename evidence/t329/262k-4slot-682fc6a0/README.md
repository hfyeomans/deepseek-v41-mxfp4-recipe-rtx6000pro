# 262K × 4 re-qualified on overlay `682fc6a0` (T3.29 follow-up, 2026-10-10)

One serve, ~22 min of gates. Launch 19:13:46Z:
`CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 ALLOW_UNTESTED_FIT=1 bash scripts/serve-hybrid.sh`
on the launcher defaults (219 / 165 by the el8 mix map, D=4, chunk 8192, fraction 0.96, spec off,
`SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1`, Engram prefetch `ahead` + chunk-ahead 1,
`KT_REJECT_NAN_ROWS=1`), overlay `682fc6a0` (C10), image `dsv41-kt:recipe`, BIOS slot 3 (FCLK 2100),
GPU power limit 325 W per the manifest. The resolved
line is the first on this ladder to carry the T3.24a profile fields:
`profile=qualified overrides=[CONTEXT_LENGTH MAX_RUNNING_REQUESTS ALLOW_UNTESTED_FIT]`
(`manifest.json`, content hash `bc60d843…`). **This launch predates T3.24b** (landed the same
afternoon): since then `ALLOW_UNTESTED_FIT=1` under `qualified` is refused and the same command
needs `PROFILE=developer` — or nothing at all, because this pass relaxed the guard (below).

## Why

The only 262K × 4 pass was the T3.31 idea-8 serve on overlay `30e8c412`
(`../../e34-indexer-memory/d-v-a/262k-4slot-idea8/`), six overlay changes ago (T3.20 ahead /
chunk-ahead Engram, T3.39 window 1b and 2, T3.8b remap-at-load, C10). The launcher guard still
said "above 64K only one slot fits". This serve re-runs the rung on the current overlay and doubles
as the full gate set for `682fc6a0` (C10 had only the instrumented NaN-injection serve).

## Gates

| gate | measured | idea-8 serve (`30e8c412`) | 64K × 4 headline (`124ca1de`) | verdict |
|---|---|---|---|---|
| after capture | `Memory pool end` 5.38 / 5.30 GB (pool 1,064,960 tokens); bs-4 capture at 3.54 GB; lever log reserved 92,482 → 92,482 MiB, allocated 92,198; `nvidia-smi` 94,118 / 94,187 MiB used (3.77 / 3.70 GB free) | 5.01 / 4.93 GB; 3.16 GB; 93,126 / 92,836; 94,498 / 94,559 MiB | 5.50 GB (pool 278,528) | **+380 / +372 MiB** vs idea-8 |
| (b) stress 4 × ~253K (`--prompt-tokens 260096 --seed 31`) | all four finished (`length`), wall **349.9 s**; TTFT 342 / 180 / 264 / 100 s; peak 97,156 / 97,225 MiB; `nvidia-smi` min free 731 / 662 MiB; **0** `expandable_segments … mapping failed` retries | 678 s; TTFT 346 / 506 / 184 / 660; min free 553 / 552; 114 retries at 16–20 MiB free | 4 × 62K: 80.5 s | **pass**, −48 % wall |
| (c) needles (`verify.py --context 262144`) | **6/6 at 221,756 tokens, TTFT 54.1 s**; arithmetic, thinking toggle, tool call PASS | 6/6, 123 s | 6/6 at 55K | **pass**, −56 % TTFT |
| (d) NLL corpus vs `../../t328/default-219-8192/corpus` | pooled **ΔNLL −0.00033 ± 0.00101 nats/tok** (z +0.33; cluster-robust z +0.26), 65,226 of 65,653 changed, worse/better > 0.5: 1,921 / 1,944 | 0 moved (positional placement = the comparator's tiers) | T3.8b qualification: −0.00033 ± 0.00101, z +0.33, 1,921 / 1,944 | **pass** — digit-for-digit the T3.8b reading: the map moved the tiers, this rung moved nothing |
| same-prompt C4 (`verify-batch-consistency.py --same-prompt --concurrency 4`) | PASS; rows not identical (near-tie flips), min pairwise prefix 224 chars (gate ≥ 40) | — | PASS | **pass** |
| (f) decode C1 / C4 (`--distinct-prompts --seed 4242 --fixture-offset 6`, 6 / 3 runs) | C1 **57.86 median / 56.07 wall**, TTFT 0.300 s; C4 **123.06 / 118.05 wall**, TTFT 0.745 s | 41.31 / 39.42; 56.25 / 54.07 | 58.0 / 56.2; 123.9 / 118.8 | unchanged vs headline (−0.2 / −0.6 %): **pass** |
| (f) prefill 16K / 32K (cold pass, `--seed 777`) | **3,191 / 3,115 tok/s**, TTFT 5.01 / 10.27 s | 1,532 / 1,541 | 2,934 / 3,532 cold | 16K +9 %, 32K −12 % vs headline (single cold pass each, different Engram residency; not a regression call) |
| (e) health, Xid, EDAC | container log: only the two benign `[sm120-split] sealed` lines, 0 NaN / `NonFiniteLogits` / Traceback; `kern.log` Xid 0; EDAC ce/ue 0/0 (sysfs) | same | same | **pass** |

## What the numbers say

- **The rung passes again, and faster everywhere the six commits touched.** Stress wall 678 → 350 s
  and needle TTFT 123 → 54 s are the T3.39 window-2 prefill (no host export) and the T3.20 Engram
  path at 253K-token prompts; decode is the 64K × 4 headline to within noise, so a 1.06M-token
  pool costs C1 / C4 nothing. The NLL reading is the T3.8b map's, digit for digit.
- **Still at the VRAM ceiling.** `nvidia-smi`'s 1 s sampler saw min free 731 / 662 MiB, but the
  container log carries **130 `Triton kernel '…' device-loaded after serving started` warnings**
  (idea-8: 82) with `free device mem` **0.11 / 0.18 GiB** on 116 of them (0.16 / 0.23 on the rest):
  118 of them during the stress (19:20–19:26Z) and the last 12 at 19:30Z during the hand re-run of the
  NLL corpus (after the C1 / C4 / prefill benches, which loaded nothing new), kernels
  `_reduce_block_fp8_split_k` (46), `_w8a8_block_fp8_matmul*` (62), `_expand_prefill_causally_kernel`
  (6), the allocator kernels and the hC / Engram hashers. These are late JIT loads at new shapes
  (the TROUBLESHOOTING entry), not failures — but they say the real trough under 4 × 253K is
  ~110–180 MiB, the same ceiling the idea-8 serve's 114 mapping retries revealed. The allocator did
  not need its retry path this time (0 `mapping failed` lines, where idea-8 had 114), which is the
  +380 MiB after capture.
- **So: qualified as a per-rung override, not a default.** 262K × 4 at 219 / 8192 with the lever
  passes every gate twice (two overlays, one day apart). It has ~0.1–0.2 GB of margin under the
  worst case it is qualified for; the 64K × 4 default keeps 5.5 GB. The guard now accepts it
  (below); BENCHMARKS and CONFIG say what the margin is.

## Launcher change

`serve_config.py guards()` (T3.29 rule "above 64K only single-slot rungs") now accepts
`CONTEXT_LENGTH <= 262144` with `MAX_RUNNING_REQUESTS <= 4` at 219 / 8192 **only with
`SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1`** (the lever-off serve OOMed on the third chunk,
idea-8 (a)); 512K / 1M with more than one slot were never run and stay refused. The refusal names
the rung and the override (`PROFILE=developer ALLOW_UNTESTED_FIT=1`, T3.24b).
`scripts/tests/test-serve-config.sh`: 262K × 4 accepted, lever-off 262K × 4 and 512K × 2 refused
(18 cases).

## Files

- `launch.txt`, `launch-container.txt` (includes the hook, KT_BUF and lever lines, the 130
  late-load warnings), `manifest.json`, `wait.txt`, `vram-after-capture.txt`.
- `stress-kv.{txt,json}`, `verify.{txt,json}`, `corpus.txt` + `corpus/*.json` (re-run by hand after
  the gate script's `nll-score.py --model` typo — the same slip the idea-8 README records; the server
  was still up, nothing else had run in between), `same-prompt-c4.{txt,json}`, `c1.{txt,json}`,
  `c4.{txt,json}`, `prefill-16k.{txt,json}`, `prefill-32k.{txt,json}`, `xid-count.txt`, `edac.txt`.

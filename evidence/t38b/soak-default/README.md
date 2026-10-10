# T3.8b soak of the new default (`KT_EXPERT_MAP` = el8 mix map) on coding + long-reasoning traffic

Question (owner, after the default flip): the el8 mix map was fitted on el8's prose / code
counts and checked on our short sets; does it hold on real coding traffic (the full HumanEval)
and on long reasoning (thinking on, thousands of reasoning tokens per request), and does a
~30-minute served session drift anything (C1 / C4 at start vs end, memory, health, EDAC)?

Profile: launcher defaults (overlay `c777da2d`, 219 / 165 with the map, D=4, chunk 8192, 64K × 4,
`KT_CPUINFER=16`, `ahead`) plus `--expert-distribution-recorder-mode stat`, so every traffic set
also yields the realised CPU-tier route share (`../analysis/routes-soak.txt`, the T3.8
`routes.py`; raw `.pt` dumps gitignored, 11 GB). Two serves, 2026-10-10, drivers `soak.sh`
(serve 1, console `soak-log.txt`, 02:54–03:15) and `soak2.sh` (serve 2, `soak2-log.txt`,
03:19–03:29; the long-reasoning sets with the new `scripts/bench-reasoning.py`).

## Traffic sets (recorder on)

| set | dir | traffic | decode tok/s per stream | pass / answers | CPU-tier share decode / prefill |
|---|---|---|---|---|---|
| HumanEval full, C4, thinking off | `humaneval-full-c4/` | 164 problems, 30,585 completion tokens, wall 334 s | median 25.86 (TTFT median 0.52 s) | **159/164** (see below) | 15.0 % / 11.8 % |
| HumanEval, C4, thinking on | `humaneval-think-c4/` | 32 problems, 13,388 tokens, median reasoning 536 chars | 32.59 | 32/32 | 12.5 % / 12.4 % |
| HumanEval, C1, thinking on | `humaneval-think-c1/` | 6 problems, 2,172 tokens, median reasoning 1,045 chars | 55.82 | 6/6 | 12.8 % / 12.6 % |
| long reasoning, C4, thinking on | `reasoning-c4/` | 8 hard prompts, 17,894 tokens (max 7,297 / request; reasoning up to 22,820 chars), wall 185 s | median 39.23, aggregate 96.6 | 6/6 known answers found | **10.7 % / 13.0 %** |
| long reasoning, C1, thinking on | `reasoning-c1/` | 4 prompts, 4,427 tokens (max 2,772) | 56.71 | 4/4 | **9.6 % / 12.5 %** |
| distinct fixtures C1 / C4 | `decode-distinct/` | 256 tokens × 3 runs | C1 56.09, C4 123.05 aggregate (wall 50.93 / 112.94) | — | 13.8 % / 14.2 % |
| NLL corpus (prefill text) | `prefill-text/` | 65,653 tokens, 8 texts | — | ΔNLL −0.00033 ± 0.00101, z +0.33 (= the qualification run, bit for bit) | 24.9 % (8 tokens) / 15.0 % |

Decode tokens recorded this session (passes × tokens/pass in `routes-soak.txt`; routes are
per layer, 6 × 40 per token): 31.8K (HumanEval C4) + 15.9K (thinking) + 22.2K (reasoning) +
3.1K (distinct) ≈ 73K, plus ≈ 13K from the T3.8b histogram serve — short of the ≥ 100K-per-phase
bar the T3.8b row set for a refit on our own counts, and the dumps are kept (gitignored) so a
later serve can top them up (T3.8c note in `plan/TASKS.md`).

## Gates (serve 1, after the traffic sets)

| gate | result | comparator (T3.8b qual `../qual-el8-mix/`) |
|---|---|---|
| C1 start / end (fixture offset 6, 512 tokens, 3 runs) | 55.83 / 53.45 | 57.97 (6 runs) |
| C4 end | 116.99 aggregate (wall 112.42) | 123.91 |
| 16K prefill, seed 777 / warm 1277 | 2,052.5 / 2,067.2 tok/s | 2,013 / 2,173 |
| same-prompt C4 consistency | PASS (min pairwise row prefix 323 chars, gate ≥ 40) | PASS |
| 4 × 62K stress | wall 122.3 s, min free 2,003 / 1,918 MiB, all finished | 108.7 s, 2,189 / 2,062 |
| needles at 55K / verify 65,536 | 6/6 / ALL PASS | 6/6 / PASS |
| VRAM after capture (MiB) | 92,888 / 92,975 | 92,778 / 92,909 |
| health scan (NaN / Xid / Traceback / OOM), both serves | 0 (`health-hits*.txt` = the two benign idea-8 `expandable_segments` lines) | 0 |
| EDAC counters | unchanged (`edac-before.txt` = `edac-after.txt`) | — |
| ready TTFT (serve 1 / 2) | 18.6 s / 12.8 s | — |

## Reading

- **The map holds on long reasoning, and better than on prose/code:** 10.7 % (C4) and 9.6 %
  (C1) of decode routes go to the CPU tier on thousands-of-token reasoning traces, versus 15 % on
  HumanEval decode and 12–15 % on prefill text. Reasoning routes concentrate harder (50 % of
  routes in 13 experts at C1 vs 22–33 on the other sets) and those experts are in the map's hot
  set (oracle top-219 on GPU 98.9 %). Per-stream decode on reasoning is 56.7 tok/s at C1 and
  39.2 at C4 (aggregate 96.6 over 8 requests of unequal length — not a steady-state C4 number;
  the distinct-fixture C4 is 123 aggregate).
- **HumanEval 159/164 vs E32's 161/164 on checkpoint order** (ids 132, 140, 145 fail on both;
  50 and 103 are new). Both new failures are greedy-tie divergences, not systematic: 103 shares
  546 of 590 characters with the E32 sample and differs only in `round(total / count)` vs
  `int(avg + 0.5)`; 50 diverges at character 14 (the model answered with `decode_shift` only
  instead of restating `encode_shift` first). This is the expected not-bit-identical behaviour of
  a tier change (`../README.md` caveats) and inside the day-to-day spread of greedy coding runs
  (the thinking-on sets are 32/32 and 6/6). Wall for the 164 problems: 334 s at C4 vs E32's 962 s
  at C1 on `1eeb48cd`.
- **Session drift:** C1 55.83 → 53.45 and C4 123.05 → 116.99 from the start of serve 1 to its
  end (after 2.6 M decode tokens, the stress and the needles). Within the day's spread (the qual
  C1 runs span 56–58, the T3.39 serves 40–42 on positional) and not a leak (VRAM, min free and
  health unchanged); noted, not acted on. The reasoning C1 on serve 2 is 56.7.
- Prefill, NLL (bit-identical to the qualification), consistency, stress, needles, VRAM, health
  and EDAC are all as qualified.

## Files

`soak.sh` / `soak2.sh` the drivers; `soak-log.txt` / `soak2-log.txt` the consoles; per set
`start.txt` / `dump.txt` / `stop.txt` from the recorder endpoints and the bench output
(`bench/` for HumanEval with `report.json`, `samples.jsonl`, `programs/`, `score.txt`;
`bench.json` + `bench.txt` for the reasoning sets — stamped after the run from `manifest2.json`, since the first `bench-reasoning.py` lacked the manifest stamp and CI caught it; `container_check` is therefore `unavailable`, the container id matches `manifest2.json`); `../analysis/routes-soak.txt` the route
analysis over all seven sets; `manifest.json` / `manifest2.json` the launch manifests;
`launch*.txt`, `ready*.txt`, `warm*.txt`, `vram-after-capture.txt`, `remap-lines.txt`,
`recorder-log.txt`, `pre-serve-state.txt`, `edac-*.txt`, `health-hits*.txt`.

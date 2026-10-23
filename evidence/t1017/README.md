# T1017 — DeepSeek-V4.1-Flash-1017 first serve and qualification battery (2026-10-23)

Serve: `HF_MODEL_DIR=/models_engrams/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017 SNAPSHOT=1bae0e1088993b9b82de08f59de5be5a34e42170 PROFILE=developer EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" bash scripts/serve-hybrid.sh` — manifest `20261023T145716Z-dsv41-kt.json` (the evidence scripts copied it beside every stamped JSON below). Same shape as the qualified profile (219/165 el8 map, D=4, chunk 8192, 262K x 4, `HOST_THREAD_CPUS=16-31`, image `dsv41-kt:recipe`, overlay `deec7a6f`); `PROFILE=developer` is only for the recorder server arg.

Checkpoint facts (verified 2026-10-23 against the pinned `dba1be0a…`): `config.json` byte-identical; weight_map identical (96,085 tensors, 48 shards, total_size 510,286,023,000 B); `tokenizer.json` / `tokenizer_config.json` identical (md5); LICENSE identical (md5); reasoning-effort tiers identical. Only the weights differ (every shard blob hash differs). The engine, the 61-patch overlay, the image and the el8 map therefore apply unchanged; the serve ran with zero serve-path code changes.

## Results (first serve; Engram page cache starts cold on a new checkpoint)

| Metric | 1017 | pinned headline (`evidence/t352/default-flip`) |
|---|---|---|
| C1 median / wall tok/s (256 tok, distinct) | 61.91 / 55.88 (recorded set 61.97 / 55.78) | 60.8 / 58.8 |
| C4 aggregate / wall tok/s | 123.58 / 113.99 (first pass 113.20 / 103.97) | 124.0 / 118.9 |
| Prefill 16K / 32K tok/s, Engram cold (TTFT) | 3,095 / 3,372 (5.18 / 9.50 s) | no cold-box datapoint |
| Prefill 16K / 32K tok/s, Engram warm (TTFT) | 4,504 / 4,613 (3.56 / 6.94 s); random-words 16K 4,583 | 4,344 / 4,495 |
| NLL corpus (65,653 tok) | **1.55385 nats/tok, bit-stable across 3 runs** | 1.57616 (1017 is 0.022 lower = better) |
| HumanEval pass@1 greedy (164) | **158/164 = 0.9634**; failed {32, 83, 103, 132, 140, 145}; passes 163 of mlx-stream's six failures | 161/164 = 0.9817 (failed {132, 140, 145}) |
| verify 16K all-pass + needles | ALL PASS at `--context 18000` (6/6 needles, 17,149 tok); at `--context 16384` the needle prompt lands 14 tokens under the >=95 % fixture bar (15,550 / 15,564) — fixture sizing, not the model | 6/6 |
| health | 0 NaN / 0 Xid across the whole battery | 0 |

HumanEval caveat: the harness has documented +-2 run noise from greedy near-ties (E32 / T3.8b saw 161 <-> 159 on the old model alone); a confirmation re-run is queued before any headline claim. The model card's gains are on agentic / SWE harnesses (DeepSWE 74.2 -> 77.0, ProgramBench 20.3 -> 29.1, Terminal-Bench 31.2 -> 42.3), not HumanEval pass@1.

## Expert map on the new gates (T3.8-style recorder)

Sets `recorder/{decode-distinct,prefill-random,prefill-text,humaneval-c4}` (dumps gitignored; analysis `recorder/routes.txt` + `recorder/maps.json`). The recorder sees post-remap logical ids, so the analysis's `positional` row IS the served el8 map's realized CPU share:

| set / phase | el8-map CPU share (1017) | old model |
|---|---|---|
| humaneval-c4 decode / prefill | 14.4 % / 11.7 % | 12-15 % |
| decode-distinct decode / prefill | 14.5 % / 14.3 % | 12-15 % |
| prefill-text (real text) | 15.0 % | 12-15 % |
| prefill-random (random words) | 22.9 % | 23 % |
| per-set oracle floor | 2.2-2.8 % decode / 0-4 % prefill | 1-3 % |

**The el8 mix map transfers to the 1017 gates unchanged - keep it.** The static-map floor and the T3.8d runtime-placement case (GitHub #6) are unchanged on 1017. `evidence/t1017/nll/baseline/` is the new-model corpus baseline (identity repeat: `nll/repeat/`).

## What changed / what did not

No serve-path code changed. Registry defaults, the `qualified` profile and every gate baseline still point at `dba1be0a…`; flipping them (`serve_config.py` defaults, `load-env.sh` pins + golden, CONFIG regen, PROVENANCE row) is a follow-up owner decision after the HumanEval confirmation and the ladder / needle re-qualification on the 1017 defaults.

## Next steps (registered T3.56 / E47)

1. HumanEval confirmation re-run; if 158 holds, read the three new failures (32 / 83 / 103) for near-tie flips.
2. Full release-smoke + ladder on 1017 defaults (262K x 4, 512K / 1M x 1), identity baseline vs `evidence/t1017/nll/baseline`.
3. Owner flip (defaults + pins + docs), then the T3.49 close-out queue resumes on 1017.

## Ladder qualification (2026-10-23, release-smoke.sh serve --full per rung; BASELINE/IDENTITY = evidence/t1017/nll/baseline)

| rung | stress wall (peak VRAM MiB) | C1 median/wall tok/s | prefill 16K/32K tok/s (TTFT s) | NLL vs baseline | needles |
|---|---|---|---|---|
| 262K x 4 (recorder serve, dev profile) | 268.0 s (97333 / 97214) | 59.67 / 57.74 | 3248.2 / 3996.7 (4.92 / 8.01) | dNLL -0.0, 0 moved | 6/6 @ 255,388 tok, TTFT 60.0 s |
| 512K x 1 (qualified, clean) | 172.7 s (95945 / 95650) | 60.15 / 58.21 | 3381.3 / 3323.2 (4.75 / 9.61) | dNLL -0.0, 0 moved | 6/6 @ 511,145 tok, TTFT 142.1 s |
| 1M x 1 (qualified, clean) | 496.6 s (97145 / 96990) | 59.05 / 57.24 | 2666.7 / 3009.1 (6.03 / 10.61) | dNLL -0.0, 0 moved | 6/6 @ 1,022,456 tok, TTFT 447.8 s |

C4 on the 262K x 4 rung: 121.37 aggregate / 114.3 wall tok/s. Every rung passed every release-smoke gate (same-prompt, NLL identity vs the new baseline, needles, health, stress; --full benches + power summary). HumanEval confirmation: 158/164 at C1 and 158/164 at C4 (stable fails 32, 103, 132, 140, 145 + one of 50/83).


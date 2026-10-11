# T3.15c — qualify the two determinism levers as the launcher default (E39)

2026-10-11. Two serves on the launcher defaults (219 / 165, chunk 8192, el8 mix map, lever on,
262K x 4, D=4, KT_CPUINFER=16) with both T3.15b levers,
`PROFILE=developer EXTRA_DOCKER_ENV="SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 SGLANG_DSV41_SORT_DECODE_TOPK=1"`.
Image `dsv41-kt:recipe` (`314749b5a268`); host runtime docker 29.9.0 / containerd v2.4.1 from this
serve on (the owner's `apt upgrade` during T3.15b serve A). Serve 1 = overlay `e07379c6` (patch
0053, sorts physical KV slots); serve 2 = overlay `743f8fa9` (patch 0054, sorts logical positions
before the slot map).

**Answer.** Both levers pass every release gate on the launcher shape at zero measured cost, twice;
the corpus NLL is bit-stable across three serves on the unfused path (0 of 65,653 values moved
serve C -> 1 -> 2). Greedy decode is bit-identical ×5 and across serves at 23 … 3,715 prompt tokens;
at 14,775 tokens every run emits the same tokens but 6 of 20 runs (three serves) differ in logprobs
from decode token 109 on (max |Δlogp| 0.012–0.055). That residual is the top-k kernel's
arrival-order tie-break (`offline/README.md`, T3.15d, upstream sgl-project/sglang #35257), not the
MoE finalize, not the KV allocator (serve 2 removed that dependence and the residual stayed), and
not the levers. Recommendation: flip both levers to the launcher default; keep T3.15d for the tie. **Flipped 2026-10-11 (owner approval): `default-flip/` is the re-measure on the plain launcher — every gate passes, corpus bit-identical to serve 2, greedy bit-identical through 20,554 tokens; it is the new headline row and identity anchor.**

## Release smoke, `--context 262144 --slots 4 --full`, `NLL_REQUIRE_IDENTICAL=0` (`serve-*/smoke-log.txt`)

| Gate | Serve 1 (`e07379c6`) | Serve 2 (`743f8fa9`) | Band / anchor |
|---|---|---|---|
| stress 4 × 260,096 | PASS, wall 327.8 s | PASS, wall 286.4 s | all slots finished |
| same-prompt C4 | PASS | PASS | |
| NLL corpus vs 219/8192 anchor | Δ −0.00033 ± 0.00101, z +0.33, moved 65,226 | identical to serve 1 (0 of 65,653 values moved) | \|z\| <= 2 |
| needles at 262,144 | 6/6 at 255,388 tokens, TTFT 60.7 s | 6/6, TTFT 59.7 s | 6/6 |
| decode C1 (after the stress, headline method) | 58.05 tok/s | 57.49 | headline 57.2 / 55.3 |
| decode C4 per stream | 121.9 | 121.1 | headline 116.8 / 112.1 |
| prefill 16K | 3,327 tok/s, TTFT 4.81 s | 3,996, 4.02 s | headline 4,101 |
| prefill 32K | 3,415, 9.35 s | 4,215, 7.59 s | headline 4,452 |
| power (busy W / max °C) C1, C4, 16K, 32K | 424 / 84, 381 / 79, 412 / 79, 426 / 80 | 425 / 84, 381 / 79, 470 / 80, 486 / 82 | |
| health / Xid | 0 / 0 | 0 / 0 | |

Serve 1's prefill ran slower than serve 2 on the same traffic order; both are inside the day's band
and serve 2 matches the headline, so the logical-first route (one extra `filter_topk_pages` kernel
per decode layer when the lever is on) costs nothing measurable at C1 / C4 either.

## Greedy drift (`scripts/greedy-drift.py`, 5 repeats, cache flushed, temperature 0)

| Run | 23–28-token prompts | 3,715 tokens | 14,775 tokens |
|---|---|---|---|
| T3.15b serve C (`e07379c6`, fresh) | bit-identical | bit-identical | bit-identical (5/5 the modal outcome) |
| serve 1, after the full smoke (`serve-1/drift.*`) | bit-identical, same bits as serve C | same | tokens identical; 2/5 runs differ in logprobs from token 109 |
| serve 2, fresh (`serve-2/drift-fresh.*`) | same bits | same | tokens identical; 1/5 differs from token 109 |
| serve 2, after the full smoke (`serve-2/drift-after-traffic.*`) | same bits | same | tokens identical; 2/5 differ from token 109 |

Across the 20 long-prompt runs there are four distinct logprob vectors; 14 runs share the modal one
(bit-identical across serves and overlays), and every deviation starts at decode token 109
(position 14,884) — one decode step whose 512th-best indexer score is tied, resolved by the kernel's
`atomicAdd` arrival order. `offline/topk-v2-ties-probe.py` reproduces the mechanism on synthetic
scores (set moves 29/29 when the boundary sits in a tie; the > 2,048-candidate bin is also inexact).

## Files

`serve-1/`, `serve-2/`: `smoke-log.txt`, `smoke-args.txt`, bench JSON + `.txt`, `corpus/`,
`verify.{json,txt}`, `stress-kv.*`, `same-prompt-c4.*`, `power-*.csv`, `power-summary.json`,
`health.txt`, `xid-count.txt`, `vram-after-capture.txt`, `manifest.json`, `drift*.{json,txt}`.
`offline/`: the tie probe and its output.

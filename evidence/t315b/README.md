# T3.15b — source of the run-to-run greedy drift at bs=1 (E38)

2026-10-11. Overlay `05f9e88b` (serves A, B) and `e07379c6` (serve C, adds the sort lever), image
`dsv41-kt:recipe` (`314749b5a268`), launcher defaults (219 / 165, chunk 8192, el8 mix map, lever on,
262K x 4, D=4, KT_CPUINFER=16), `PROFILE=developer` only to pass `EXTRA_DOCKER_ENV`.

**Answer.** The drift has exactly two sources, both GPU-side and both order-of-reduction effects;
neither is kt-kernel, NCCL, Engram or the deferral. Turning both off makes bs=1 greedy decode
bit-reproducible (5 repeats x 256 tokens, prompts of 23 to 14,775 tokens) at no measured cost.

| Source | Where | Shows up | Lever |
|---|---|---|---|
| FlashInfer cutlass MXFP4 fused-MoE **fused finalize** (the top-k expert contributions are summed in the GEMM epilogue, order run-dependent) | `flashinfer_cutlass.py:675` | every decode step, any context length (short prompts diverge at token 16-52) | `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0` (upstream env, default 1) |
| DSV4 indexer **top-k order**: `topk_v2` emits the same 512 slots in a run-dependent order once the request is longer than 512 tokens, and the SM120 sparse decode kernel's split-K reduction is order-sensitive | `deepseek_v4_backend.py` `_low_ratio_index_topk_decode` | decode of any request past 512 tokens of context | `SGLANG_DSV41_SORT_DECODE_TOPK=1` (new overlay env, default 0: descending sort of each row's slots, padding stays last) |

## Offline probes (`offline/`, `kt-repeat/`)

See `offline/README.md`. kt-kernel CPU tier: bit-identical on every repeat, 8 threads == 16 threads.
`topk_v2`: set stable, order differs 29/29 repeats above 512. SM120 sparse decode MLA: bit-identical
for a fixed index list; the same set in another order changes ~2,300 of 32,768 outputs. NCCL excluded
at TP2 (a+b is commutative). `--enable-deterministic-inference` cannot start (`dsv4` is not in the
backend's allowed list) and the dsv4 backend ignores it anyway.

## Serves (`scripts/greedy-drift.py`, 5 repeats, cache flushed before every request, temperature 0)

| Serve | Env | short prompts (23-28 tok, 256 out) | 3,715-tok prompt (128 out) | 14,775-tok prompt |
|---|---|---|---|---|
| A `a-baseline` | defaults | diverge at token 16 / 32 / 41 / 52; max abs dlogp on shared prefix 0.27-0.62 | diverges at 57 | - |
| A, `--no-flush` | defaults, radix cache kept | diverge at 4 / 17 / 27 / 34 (prefix-cache hits do not help: the drift is in decode) | - | - |
| B `b-finalize0` | `FUSED_FINALIZE=0` | **bit-identical x5, all four** | diverges at 63 | - |
| C `c-finalize0-sorted` | `FUSED_FINALIZE=0` + `SORT_DECODE_TOPK=1` | **bit-identical** | **bit-identical** | **bit-identical** |

Cross-serve: B and C produce the same tokens on the short prompts (the sort is a no-op under 512
slots). A vs B/C first differ at tokens 64 / 32 / 41 / 52 / 63, inside A's own repeat-to-repeat spread
(16-64 / 32-46 / 41-86 / 52-53 / 57-90): the levers remove a run-dependent reduction order, they do
not move the numerics beyond it.

## Cost and quality (serve-level, same window; today's four E37 serves on defaults read C1 59.7-61.3, C4 118.7-121.4, prefill 16K 3,150-3,683 / 32K 3,400-4,292)

| Serve | C1 tok/s | C4 per-stream | prefill 16K tok/s (TTFT) | 32K | needles 64K | NLL corpus vs 219/8192 anchor | health |
|---|---|---|---|---|---|---|---|
| B | 60.9 | 116.9 | 907 cold / 3,007 (17.6 / 5.3 s; no warm run first, Engram 2.3 % resident) | 3,221 / 3,408 (9.9 / 9.4 s) | - | - | - |
| C | 61.0 | 123.5 | 3,183 / 3,192 (5.0 s) | 3,406 / 3,489 (9.4 / 9.2 s) | 6/6 at 63,563 tok, TTFT 14.5 s | **-0.00033 +- 0.00101 nats/tok (z +0.33, cluster z +0.26), 65,226 of 65,653 tokens changed** | 0 nan/xid/traceback hits; 97.1 / 97.2 GiB |

Reading: decode C1 unchanged, C4 at the top of the day's band, prefill inside the band (B's cold 16K
first run is the usual idle-box Engram effect, not the lever). The NLL gate passes as a kernel change
(pooled delta well inside +0.002), but `tokens_changed` is 65,226, so `FUSED_FINALIZE=0` moves the
prefill bits too: it is a different reduction, not the same one made deterministic. Whether the
unfused prefill is itself bit-stable across serves (the fused one is, per BENCHMARKS) was not measured
here; a second serve replicating `corpus/` is part of any default flip.

## Decision

Diagnosis closed. Both levers are off by default; `SGLANG_DSV41_SORT_DECODE_TOPK` is overlay commit
`e07379c6`. Flipping both on by default (bit-reproducible greedy decode at temperature 0 for every
user of the recipe) is registered as T3.15c: one qualification serve on the launcher defaults with
both levers (262K needles, 4 x 62K stress, 64K x 4 soak, corpus replicate for bit-stability, C1/C4),
then the owner decides. Nothing in this task changes the served defaults.

Files: `a-baseline/` (also `launch-killed-1556Z.txt`: the first serve A was killed by an external
`docker kill -s 15` at 15:56:46Z, not from this work; relaunched), `b-finalize0/`, `c-finalize0-sorted/`
(bench JSON + `corpus/` + `verify-64k.json` + `manifest.json`), `offline/`, `kt-repeat/`.
Tools kept: `scripts/greedy-drift.py` (serve-level drift measurement), `scripts/test-kt-determinism.py`
(kt one-layer repeatability, runs inside the image like `test-kt-one-layer.py`).

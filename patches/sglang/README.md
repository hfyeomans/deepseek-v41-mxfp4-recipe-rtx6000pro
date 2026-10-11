# SGLang overlay as a patch series (T3.24c)

Everything this recipe changes in SGLang, as 50 `git format-patch` files on top of one upstream commit. With this directory, the public `dev-dsv41` image and the model snapshot, the exact serving stack can be rebuilt without access to the local overlay clone. The kt-kernel side of the hybrid is a separate, much smaller series in [`config/kt-kernel-patches/`](../../config/kt-kernel-patches/) (three patches applied by `config/Dockerfile.dsv41-kt`).

| | |
|---|---|
| Upstream base | `sgl-project/sglang` commit `da64c5cbb8cf6bfd39be19da43573fdfd484c43a` — "[DSV4.1] Support raw-index output in TopK v2 (port of #33672) (#39098)", the commit the `lmsysorg/sglang:dev-dsv41` image was built from (`ai.sglang.build.commit` label; see `PROVENANCE.md`) |
| Overlay branch | `dsv41-kt-hybrid` in the local clone `~/dev/src/sglang-dsv41-overlay` (no fork, no push) |
| Exported head | commit `90ae890e33f33cce57dee98ba4cec27516318d15`, tree `90df91d06413bebf1612abc9a3b68a0f6837cbb3`, 51 patches (the `recipe-2026-10` tag stays at `682fc6a0`, patch 0050; 0051 is T3.35's request-admission clamp) (`SERIES.env` is the machine-readable copy) |
| Commits as written | no history rewrite: author, dates and messages are the overlay's. A themed rebase into ≈ 8 reviewable units is **T3.24c-ii**, an owner decision |

## Apply

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout --detach da64c5cbb8cf6bfd39be19da43573fdfd484c43a
git am /path/to/recipe/patches/sglang/*.patch
git rev-parse 'HEAD^{tree}'        # must print 90df91d06413bebf1612abc9a3b68a0f6837cbb3
```

Only `python/sglang/` changes. The runtime image never contains these patches: [`scripts/serve-hybrid.sh`](../../scripts/serve-hybrid.sh) bind-mounts `$OVERLAY_DIR/python/sglang` read-only over `/sgl-workspace/sglang/python/sglang` inside `dsv41-kt:recipe`, so a checkout produced by the commands above can serve by setting `OVERLAY_DIR` to it. `config/Dockerfile.dsv41-kt` builds the image (base image digest + pruned kt-kernel wheel + the kt patches); the overlay is applied at run time, not build time.

## Keep in sync

- `bash scripts/export-sglang-series.sh` re-exports the branch (deletes the old `*.patch`, rewrites `SERIES.env`). Run it after every overlay commit that a recipe commit depends on.
- `scripts/tests/check-sglang-patch-series.sh` (in `scripts/ci.sh`) applies the series onto a clean `da64c5cb` checkout in a temporary shared clone (≈ 1.5 s) and fails when the tree hash differs from `SERIES.env`, or when the overlay branch has moved without a re-export. It exits 3 (skipped) on a machine without a git checkout that holds the base commit.
- The ci.sh identifier scan covers these files; the overlay commits contain no home-directory paths or host identifiers.

## Index

Recipe tasks and experiments are in `plan/TASKS.md` and `plan/ROADMAP.md`; the upstream-candidacy column is the unit in [`docs/research/OPEN-SOURCE-PLAN.md`](../../docs/research/OPEN-SOURCE-PLAN.md) §3 (PR n = numbered candidate, *diag* = opt-in diagnostics, *RFC* = "later, as design discussions", *local* = recipe-only, not meant for upstream, — = landed after the plan's last classification pass).

| # | Overlay commit | Change | Recipe task / experiment | OPEN-SOURCE-PLAN |
|---|---|---|---|---|
| 0001 | `3805730e` | Compact CPU-expert view for kt-kernel EP; kt-kernel 0.7 API; `fuse_routed_scaling_factor_in_topk=False` | E17 | RFC |
| 0002 | `a1ca246a` | Engram `layout=file`: tables stay in the checkpoint shard, rows fetched per step | E18 | RFC |
| 0003 | `cd55166a` | Breakable-graph (BCG) dataclass outputs; `extra` KV pools re-paged to 64 rows for FlashInfer SM120 sparse-MLA prefill | E19 | early fix |
| 0004 | `32a25f37` | Seven tuned W8A8 block-FP8 `[32,32]` Triton configs for RTX PRO 6000 Max-Q; SPLIT_K/SWAP_AB kernel allowed on SM120 | E20 | PR 1 |
| 0005 | `1d6a98d3` | Zero-fill the SM120 page-split scratch (stale bytes decoded to FP8 NaN) | T3.6d2 | PR 3 |
| 0006 | `11e7f745` | kt expert deferral with the compact CPU view | E21 | RFC |
| 0007 | `0138c6ad` | Layerwise streamed GPU prefill coordinator for CPU experts (`KT_GPU_PREFILL_MIN_TOKENS`) | E22b | RFC |
| 0008 | `670994ac` | `KT_NAN_PROBE_MIN_TOKENS` per-layer non-finite probe | E22d | diag (not ready) |
| 0009 | `6032dd44` | Prefill-graph breaks kept out of the speculative TARGET_VERIFY step | E23 | early fix |
| 0010 | `3b1fedce` | Low-ratio source-projection break is prefill-graph only | E23 | early fix |
| 0011 | `71645529` | BCG dataclass outputs keep per-field row counts | E23 | rides with PR 4 |
| 0012 | `95f2fe3d` | BCG output cap scales with `captured_req_width` | E23 | rides with PR 4 |
| 0013 | `b457c4e6` | `SGLANG_BCG_DEBUG_SYNC` fault attribution; nested breaks run eagerly under `--debug-cuda-graph` | E23 R1 | diag |
| 0014 | `4685955e` | `SGLANG_BCG_DEBUG_EAGER=draft\|target`; verify skips the mm-embedding path | E23 R1 | diag |
| 0015 | `838490a6` | SM120 page-split never frees graph-captured buffers; pre-sized from pool geometry | E23 R3 | PR 4 |
| 0016 | `07f4aef0` | Debug sync before segment 0 (fault inherited from the preceding eager step) | E23 R3 | diag |
| 0017 | `cad73914` | NaN probe skipped while a CUDA graph is capturing | E22d / E23 | diag |
| 0018 | `8ad514ed` | kt capture sizes registered before the eager autotune forward; captured ⊆ registered ⊆ cached assertion | T3.16c | SGLang half of PR 5 |
| 0019 | `7e41317e` | B2 direct-UVA experts (`KT_B2_UVA_EXPERTS`) — superseded by 0029 | T3.18 / E30 | local (retired) |
| 0020 | `b93cb915` | `kt_contract.py`: executable SGLang→kt boundary and rank-topology contract | T3.16d | RFC |
| 0021 | `5b531732` | Multi-stream overlap forced off for DSpark on SM120; explicit `=1` refused (Bug A) | review C4 | overlap-gate PR |
| 0022 | `9ecd54a6` | `reserve_split_buffers()` seals the page-split buffers; a later grow raises | review C5 | follow-up to PR 4 |
| 0023 | `e66d44fb` | `KT_TIMELINE=1` per-layer decode timeline + recorded-tensor dump | T3.19a | diag |
| 0024 | `9fc14817` | `CpuExpertProvider` / `BufferLease` / `ForwardLedger` seam around `KTMoEWrapper`; C3 remainder asserts | T3.19b | RFC |
| 0025 | `1953a92e` | `_TORCH_INDEXER_SCORE_BUDGET_BYTES` 1 GiB → 256 MiB | E34a | PR 6 rider |
| 0026 | `c639cbee` | Dense fp4 indexer prefill scoring tiled over query rows under the score budget | E34b | PR 6 |
| 0027 | `d6811728` | Prefill candidate masks published at block granularity | E34b-ii | PR 6 |
| 0028 | `095789db` | Runner-owned kt state (`KtRuntime`), explicit `CompactMXFP4Loader`, no module registry | T3.19c | RFC |
| 0029 | `b6c76738` | Remove the B2 direct-UVA residue (ledger G2) | T3.16j | local |
| 0030 | `bc73ef8d` | Streamed prefill as `PrefillPipeline` + `LayerwisePrefillCoordinator` (`kt_prefill.py`) | T3.19d | RFC |
| 0031 | `b6dc5173` | Prefill candidates published as block-ID lists | E34e | PR 6 |
| 0032 | `2323052b` | Persistent FlashInfer CUTLASS MXFP4 workspace per device | E34d-v (i) | — |
| 0033 | `e5f60f54` | Indexer pool gathers fp4 key rows by page/offset | E34d-v (ii) | — |
| 0034 | `f18dc961` | bf16 hc stream passed to `hc_combine` at the model tail | E34d-v (ii) | — |
| 0035 | `1eeb48cd` | Streamed expert scales converted to E8M0 in place on the landing zone | E34d-v (ii) | — |
| 0036 | `16260ad3` | Overlap gate also forces multi-stream overlap off for NGRAM on SM120 | T3.9h | overlap-gate PR |
| 0037 | `84cd3ec4` | `deepseek_v4_hook`: NGRAM speculation admitted on probation | T3.9h (n-gram rejected) | local |
| 0038 | `eeb5320c` | V4.1 feature guard admits NGRAM beside DSpark | T3.9h (n-gram rejected) | local |
| 0039 | `30e8c412` | Optional `expandable_segments` after CUDA-graph capture (`SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE`, the "lever") | T3.31 idea 8 | — |
| 0040 | `ff516964` | Engram file layout: ahead-of-use row fetch for the second Engram layer (`SGLANG_DSV41_ENGRAM_FILE_PREFETCH`) | T3.20 / E36 | RFC (Engram) |
| 0041 | `53714776` | One post-hasher break starts every Engram layer's fetch; layer breaks only join | T3.20 round 2 | RFC (Engram) |
| 0042 | `88b2f081` | Engram pinned staging allocated on the launch thread; worker CPU-only (capture-safe) | T3.20 | RFC (Engram) |
| 0043 | `ad973440` | One ahead worker shared by every Engram table | T3.20 | RFC (Engram) |
| 0044 | `99128a7e` | One C++ `FusedMoeRunner` per routed-expert count | T3.8b (qualification serve) | — |
| 0045 | `755fe3f0` | `vl_routing` reports routed ids to the expert-distribution recorder | T3.8b | — |
| 0046 | `c777da2d` | `kt_remap.py`: per-layer expert remap at load (`--kt-expert-map`), compact loader explicit order | T3.8b | — |
| 0047 | `9d481b0b` | Engram: hash and fetch the chunked request's next prefill chunk from the post-hasher break | T3.20 (c) | RFC (Engram) |
| 0048 | `a8537fe7` | `kt_prefill`: shared weight arenas replace the kt export (pairs with kt patch 0003) | T3.39 window 2 K2 | — |
| 0049 | `124ca1de` | `kt_prefill`: down scales staged as whole rows in one copy, sliced on device | T3.39 window 2 qual | — |
| 0050 | `682fc6a0` | Sampler/scheduler: a NaN/Inf logits row aborts that request, not the server | T3.13 / C10 | — |
| 0051 | `90ae890e` | `tokenizer_manager`: clamp `max_new_tokens` to the remaining context instead of HTTP 400 (`SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS`) | T3.35 | upstream candidate |

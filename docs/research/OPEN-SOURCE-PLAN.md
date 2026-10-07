# Open-source and community plan for the SGLang overlay

Written 2026-09-22. The 2-GPU hybrid runs on a locally maintained SGLang fork ("the overlay"). This note records how to preserve it reproducibly, which pieces are candidates for upstream, with what evidence, and what is not ready. The discussion thread's assessment, reproduced here as its assessment, was: publish the complete reproducible overlay in this repository first, then upstream the generally useful pieces as small evidence-backed PRs; the SM120 FP8 tuning and tuner fixes are the strongest first candidates; the architectural pieces "are more architectural and likely require design discussion before a PR". Labels per [README.md](README.md).

## 1. State of the overlay (measured, 2026-09-22)

| Item | Value |
|---|---|
| Location | `~/dev/src/sglang-dsv41-overlay`, branch `dsv41-kt-hybrid`, working tree clean |
| Remote | `origin` = `https://github.com/sgl-project/sglang` (fetch only in practice; **nothing pushed**, no fork, no tag) |
| Upstream base | `da64c5cb` — "[DSV4.1] Support raw-index output in TopK v2 (port of #33672) (#39098)"; the same commit the `lmsysorg/sglang:dev-dsv41` image was built from (image label `ai.sglang.build.commit`, `PROVENANCE.md`) |
| Runtime image | `dsv41-kt:recipe` from `config/Dockerfile.dsv41-kt` = `dev-dsv41@sha256:4a5d132a…` + `kt-kernel==0.7.0.post4` pruned to its own distribution (`config/prune_wheel.py`), version guards in `config/kt_smoke.py` |
| How it is applied | bind-mounted read-only over the image's `sglang` package by `scripts/serve-hybrid.sh` (`OVERLAY_DIR`), so no image rebuild per change |

Commit series on top of `da64c5cb`, oldest first (`git log --oneline da64c5cb..HEAD`):

| SHA | Change | Experiment | Upstream candidate |
|---|---|---|---|
| `3805730e` | kt-ep hybrid: compact CPU-expert view, kt-kernel 0.7 API, V4.1-Flash MXFP4 fixes (two upstream bugs: double routed scaling; a 0.6-era `KTMoEWrapper` call missing `swiglu_limit`) | E17 | later RFC (compact CPU experts) |
| `a1ca246a` | Engram `file` layout (A-min): table stays in the checkpoint shard, mmap + WILLNEED | E18 | later RFC (file-backed Engram) |
| `cd55166a` | breakable CUDA graph (BCG): dataclass outputs; re-page c1/c2 extra KV pools to 64 for FlashInfer SM120 sparse-MLA prefill | E19 | bug fix, could go early (needs a test) |
| `32a25f37` | seven tuned W8A8 block-[32,32] FP8 configs for `NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition`; `is_sm90` → `is_sm90 or is_sm120` gate | E20 | **PR 1** |
| `1d6a98d3` | zero-fill SM120 page-split scratch buffers (`flash_mla_sm120.py::_split_kv_pages_to_64`) | T3.6d2 | **PR 3** |
| `11e7f745` | kt expert deferral works with the compact CPU view (`-1`-tolerant selector) | E21 | later RFC (deferral) |
| `0138c6ad` | layerwise streamed GPU prefill coordinator for CPU experts (`KT_GPU_PREFILL_MIN_TOKENS`) | E22b | later RFC (streamed prefill) |
| `670994ac` | `KT_NAN_PROBE_MIN_TOKENS` per-layer non-finite probe + per-forward log | E22d | debugging aid; not for upstream as-is |
| `6032dd44`, `3b1fedce` | prefill-graph breaks (engram hash, low-ratio sources, attention) gated on `is_extend_without_speculative()` so `TARGET_VERIFY` under the decode graph does not hit them | E23 | bug fix; upstream main has the same gates as of 2026-09-22 (its TBO gate already uses the speculative-aware predicate) |
| `71645529` | BCG dataclass outputs keep per-field row counts | E23 | harmless; superseded in effect by `95f2fe3d` |
| `95f2fe3d` | BCG output cap = ShapeKey.size × `captured_req_width` (speculative verify returns bs × width rows) | E23 | bug fix once E23's IMA is resolved |

The tuner and validation scripts live in the recipe repo (`kernels/e20/tune_w8a8_block32.py`, `merge_configs.py`, `check_sglang_w8a8_path.py`), not in the overlay.

## 2. Preserve the overlay now (immediate, no upstream contact needed)

Goal: someone with the recipe repo, the public `dev-dsv41` image and the model snapshot can rebuild the exact serving stack. Mentioning local SHAs is not enough — they must be retrievable.

1. **Export the patch series into this repo**: `git -C ~/dev/src/sglang-dsv41-overlay format-patch da64c5cb..HEAD -o patches/sglang/` (12 patches), committed here with a `patches/sglang/README.md` stating the base SHA, the apply command (`git apply` or `git am` on a checkout of `da64c5cb`), and the image digest. This is the durable form even if the fork is never published; the recipe repo already carries the licences (§4).
2. **Optionally a public fork** with a durable branch (`dsv41-kt-hybrid` on a fork of `sgl-project/sglang`) and a tag per recipe milestone (`e20`, `e22`, …). A pinned submodule + patches is the third option. Any of these requires the owner's explicit instruction to push; none has been given.
3. **Keep `PROVENANCE.md` in sync**: base SHA, full commit list (§1), image digest, kt-kernel wheel version and prune manifest, snapshot revision.
4. **Build/run instructions** are `config/Dockerfile.dsv41-kt` (build) and `scripts/serve-hybrid.sh --print` (run, with every env knob and its default); `scripts/verify.py` and `scripts/nll-score.py --compare` are the acceptance checks. Reference them from the patch README rather than duplicating.
5. **Tests to keep with the patches**: `kernels/e20/check_sglang_w8a8_path.py` (70 cases), `scripts/test-kt-one-layer.py` and `scripts/kt-export-bench.py --check-tokens` (CPU tier and streamer), the E18 standalone TP2-shaped Engram test, and the `verify.py` 16K suite.

## 3. Candidate upstream PRs, in reviewable units

Do not submit the hybrid runtime as one PR. Each unit below lists the change, the evidence that exists, the tests a reviewer will ask for, and scope limits.

### PR 1 — SM120 W8A8 block-FP8 configuration support (`32a25f37`)

- **Change**: allow the tuned split-K / swap-AB Triton path on `is_sm120` (today `is_sm90` only); add the seven `NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition … block_shape=[32, 32].json` files.
- **Evidence in hand (measured)**: per-kernel 5.6–12.1× at M=1; 70/70 finite through the real call path at max rel. error 0.0037–0.0043; server C1 13.98 → 31.60 tok/s; details in [BENCHMARK-EVOLUTION-AND-E20-FP8.md](BENCHMARK-EVOLUTION-AND-E20-FP8.md) §2.
- **Tests to add**: a unit test that loads each shipped config and checks `w8a8_block_fp8_matmul` against the FP32 reference at the tuned M keys on SM120; a skip on other archs.
- **Scope limits**: configs are for TP=2 shapes of V4.1-Flash on this exact device name; other TP degrees or other SM120 cards (RTX PRO 6000 non-Max-Q, RTX 5090) need their own files. The gate change is arch-wide and is the part reviewers will scrutinise — the kernel is plain Triton, so the argument is "the restriction was where it was tuned, not what it needs".

### PR 2 — block-32 tuner improvements (recipe `kernels/e20/`, not yet in overlay form)

- **Change**: let `benchmark/kernels/quantization/tuning_block_wise_kernel.py` search `BLOCK_SIZE_K=32` for `[32,32]` scale blocks (its current `{64,128}` filter yields an empty space), add split-K and swap-AB to the search, make the device-name/config generation reusable.
- **Evidence**: the pass-2 intermediate-M cliff (`1792×5120` M=32 68 → 27 µs after adding split-K), the winner patterns, the merge-best-per-M procedure.
- **Tests**: tuner smoke run on one small shape producing a valid JSON; the config-loader test from PR 1 consuming it.
- **Scope limits**: search cost grows with split-K × swap-AB; provide defaults that keep the stock run time.

### PR 3 — FP8 scratch-buffer correctness fix (`1d6a98d3`)

- **Change**: `torch.empty` → `torch.zeros` for the SM120 page-split scratch in `flash_mla_sm120.py::_split_kv_pages_to_64` (destination pages the touched-page mask never marks otherwise hold stale VRAM bytes; e4m3 `0x7F`/`0xFF` decode to NaN).
- **Evidence (measured)**: the non-deterministic first-mixed-batch `probability tensor contains inf/nan` went from 2 of 3 attempts to 0 in a 35-minute / 96-mixed-batch soak; bs=4 greedy logprobs match bs=1.
- **Tests**: a minimal reproducer that fills the scratch with `0xFF` before the split and asserts finite output; ideally a bisected root cause (open — "stale bytes" is the explanation that fits, not a proven mechanism). State that honestly in the PR.
- **Scope limits**: one allocation; if upstream prefers a targeted mask write over zero-fill, the reproducer still applies.

### PR 4 — SM120 page-split scratch must not be freed under a captured CUDA graph (`838490a6`, measured 2026-09-23)

- **Change**: `flash_mla_sm120.py::_split_kv_pages_to_64` kept its int8 page mask and split-output buffers as grow-only tensors in `get_resources().buffers` and dropped the old tensor on growth. Any CUDA graph captured earlier (DSpark verify, plain decode) holds the old pointer in a memset and two Triton launches and keeps writing it at every replay. The fix retires old tensors instead of freeing them, adds `reserve_split_buffers(pools, device)` so `DeepseekV4AttnBackend` pre-sizes from the real pool geometry before capture (keyed on the indexed device — `torch.device("cuda")` and `"cuda:0"` stringify differently), logs a grow-after-capture at ERROR, and gives `_page_mark_kernel` an upper bound.
- **Evidence (measured)**: DSpark K=2 in graph mode died with `cudaErrorIllegalAddress`/Xid 31 in layer 0 on the second request, every run (attempts 5–7, diagnostic runs B–I); the graph-input dump showed verify `positions`/`out_cache_loc` overwritten with page-mark bytes (`0x00010100`); with the fix, probe/3-needle verify pass and a C1/C3/C4 512-token matrix runs with zero faults. Eager mode never faulted, which is the signature of a captured-pointer bug.
- **Tests**: (a) unit test that captures a graph around `_split_kv_pages_to_64` on a small pool, then calls it eagerly on a larger pool, replays the graph and asserts the retired buffer is still alive and the replay output unchanged (modelled on vLLM's `d2983f2` padded-row regression test); (b) assert `reserve_split_buffers` leaves no grow during a DSV4.1 launch (grep the ERROR line). Both are cheap and CPU/GPU-small.
- **Scope limits**: one module plus a 30-line backend hook. Upstream may prefer sizing the scratch from `max_total_num_tokens` in the pool constructor; the retire list is still the right safety net for any later caller.
- **Related upstream state (checked via GitHub, 2026-09-23)**: no fix after base `da64c5cb`; nearest is open issue #34384 (padded slot geometry). Prior art for "graph padding rows must point at real memory": vLLM `d2983f2`, `6fbbcf2`, `d8f483dc30`; TRT-LLM `f1f773f`.

### PR 5 — kt-kernel (KTransformers, not SGLang): `KExpertsCPUBuffer` temp buffer must not be baked into a CUDA graph or freed under in-flight async copies (`config/kt-kernel-patches/0001-experts-base-temp-buffer-lifetime.patch`, measured 2026-09-23)

- **Change**: `kt_kernel/experts_base.py KExpertsCPUBuffer.get_buffer` (kt-kernel 0.7.0.post4; `main` still identical on 2026-09-23) keys pinned host buffers on the flattened token count; sizes in `capture_bs` are permanent, every other size shares one class-level temp tuple that is replaced — and freed — on the next different size. The lookup order is `capture_buffers → temp → allocate`, so a size first seen *before* `set_capture_batch_sizes` (SGLang's flashinfer autotune dummy forward runs the max speculative shape 26 s before the verify runner exists) is served from the temp slot during capture and then freed by the next capture. Fix 1: promote the temp tuple into `capture_buffers` when its size is registered. Fix 2: retire replaced temps behind a `torch.cuda.Event` (the D2H copy, host-callback expert compute and H2D are all stream-asynchronous and the C++ tasks hold raw `data_ptr()`s the caching host allocator cannot see).
- **Evidence (measured)**: DSpark K=5 with four concurrent requests died on the first bs=4 verify, every run — host SIGSEGV inside `cuGraphLaunch` (R1) or a copy-engine MMU write fault at address 0 (R2, `KT_MAX_DEFERRED=0`, so deferral is not a factor). Instrumented launch (`evidence/e23-spec-decode/r3/k5-d0-nofix-c4-segfault.txt`): `alloc temp buffer bs=24 … capture_bs=[]` at autotune, capture of bs=4 finds 24 in the temp slot, bs=3 capture replaces it, final registered set `[6, 12, 18]` — 24 missing. With the patch: C4 probes 3/3 and the full K=5 D=4 C1/C3/C4 matrix, zero Xid.
- **Broader correctness impact (measured 2026-09-23)**: this is also a silent correctness bug for plain speculation-off decode, and in general for **any captured batch size first requested before `set_capture_batch_sizes`**. The pristine bs=4 graph baked the reusable temp pinned buffer; after the first prefill freed it, only row 0 received correct CPU-expert output and rows 1–3 used stale/aliased storage. bs=3 pads into the same graph and was likewise corrupted; bs=2 was correct because it had a permanent buffer. The patched image logs `KT_BUF promote temp buffer bs=4 into capture_buffers` and produces byte-identical same-prompt rows apart from staggered admission.
- **Upstream reports this fixes**: kvcache-ai/ktransformers#1815 (two RTX PRO 6000, four running requests, SIGSEGV with speculative decoding; maintainers replied "we do not support speculative decoding currently") and #2009 (MTP SIGSEGV in graph replay). Both are this lifetime bug, not a missing feature.
- **Tests**: (a) call `get_buffer(24)` before `set_capture_batch_sizes([6, 12, 18, 24])`, then `get_buffer(24)` under `torch.cuda.graph` capture followed by `get_buffer(18)`; assert the tuple returned during capture is the one in `capture_buffers` and is still alive. (b) allocate temp A, launch a non-blocking copy into it, request temp B of another size, assert A is retired (event pending) and freed only after `event.query()`.
- **Upstream regression test addition**: run four identical greedy prompts in one batch at temperature 0 (the `scripts/verify-batch-consistency.py --same-prompt` protocol), compare every row byte-for-byte and compare row 0 with a bs=1 reference. Repeat at every captured batch size, including padded bs=3 → bs=4; this catches silent wrong-row computation that crash-only and row-0 logprob tests miss.
- **Scope limits**: ~40 lines in one Python file; no C++ change. Licence for kt-kernel must be recorded (§4) before the patch is posted.
- **Tests now exist (2026-09-24, T3.16a)**: `scripts/tests/test_kt_buffer_lifetime.py` — eight deterministic, GPU-free cases that run inside the image with `is_current_stream_capturing` replaced by a flag. Discrimination checked: stock kt-kernel 1/8 (the bug-C case `test_promoted_temp_is_never_retired` fails with `KeyError: 24` — capture got the temp tuple), 0001 only 3/8, 0001 + 0002 8/8. The companion patch `0002-experts-base-capture-assertions.patch` (capture-time assertions + merging `set_capture_batch_sizes`) is the second half of the same PR: the fix plus the fail-fast that would have caught the bug at capture.

### PR 6 — DSV4 indexer: tile the dense prefill scoring so the transient is independent of prefix length (E34b, registered 2026-10-07, not yet written)

- **Change (planned)**: `deepseek_v4_backend.py::_low_ratio_index_topk_dense` builds fp32 logits `[chunk_tokens, lc]` for every prefill chunk, with `lc` = prefix length at compress ratio 1, then `select_candidate_blocks` pads a copy of it in slices bounded by `_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30`. Score over key-block tiles with a running top-k instead, so the transient is O(chunk × tile) rather than O(chunk × prefix). Upstream code, model-level (not SM120-specific).
- **Evidence (measured, E33)**: on 96 GB cards with ≈ 4 GB free after capture, 128K / 1 slot and 64K / 4 slots die with `torch.OutOfMemoryError` in `F.pad` inside `select_candidate_blocks` at ≈ 60–65K prefix; allocation sizes are exactly 4096 × prefix × 4 B; the KV pool for the same contexts costs ≈ 0.1–0.2 GB per 65K tokens. The transient is 2 GiB at 128K and 16 GiB at 1M, so it binds on any card below ~120 GB at 262K+.
- **Tests**: CPU test that the tiled top-k block masks are bit-equal to the dense path on random `[chunk, lc]` inputs and on the E33 prefix shapes; a serve-level check that the 4K NLL corpus is bit-identical (0 tokens moved); the E33 ladder resumed from 128K.
- **Scope limits**: the scorer only; no change to the KV layout, candidate block size or top-k semantics. E34a (lowering the budget constant) is a probe, not the PR.
- **Related upstream state**: not yet checked; do so before writing the PR (the constant's comment "transients run ~3x this" suggests upstream knows the bound and sized it for 141–192 GB cards).
- **E34a (measured, overlay `1953a92e`)**: the 256 MiB budget qualifies 64K × 4 slots and moves the 128K OOM from ≈ 65K to ≈ 110K prefix; the residual is the `[4096, lc]` fp32 logits plus the `[4096, lc]` bool from `masked_fill_(j >= lens)`. A rider PR for the budget alone is defensible (one constant, no semantics); the tiled scorer is the real one.

### Diagnostics worth offering as opt-in flags (from the same bisect)

- `SGLANG_BCG_DEBUG_LAYER_BREAKS=all|"0-9"|"0,4"` — per-layer breakable-graph segments so a replay fault is attributed to a layer; `SGLANG_BCG_DEBUG_SYNC=1` syncs after each segment **and, since 2026-09-23, before segment 0** (logs "fault surfaced BEFORE segment 0" so a fault inherited from the preceding eager prefill/draft step is not blamed on the graph — that discriminator is what pointed E23 at `cuGraphLaunch` itself).
- `SGLANG_SM120_IDX_GUARD=1` — masks attention token indices outside `[0, pool)` to −1 and accumulates in-graph OOB stats (`idx_oob_stats_buffer`, int64[5]) that the layer breaks read; FlashInfer's SM120 sparse-MLA kernel has no upper-bound check on positive indices, so this is the only way to see one.
- `SGLANG_BCG_DEBUG_INPUTS=1` — dumps the verify graph inputs (`req_pool_indices`, `seq_lens`, `out_cache_loc`, `positions` with pointers) at the first layer break.

### Bug still carried as an env workaround (not PR-ready)

- Multi-stream overlap (`alt_streams`) inside the captured DSpark verify graph produces wrong tokens (accept 0.26–0.57 vs 0.77 eager; garbled tool calls). `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` fixes it. The PR-able change is to disable the overlap for speculative draft/verify forwards only (or fix the stream ordering at capture); needs the R0 cost of the global switch measured first.

### Early bug fixes that could ride along (need tests first)

- `cd55166a` BCG dataclass outputs and the `extra_page_block_size ∈ {64, 2}` re-page for FlashInfer SM120 prefill.
- `6032dd44`/`3b1fedce` speculative-aware prefill-graph-break gates — check whether upstream main has since changed these; if not, a small PR with the E23 attempt-2 traceback as the reproducer.
- `95f2fe3d` BCG output cap for speculative widths — E23's illegal memory access (bug B, PR 4) and the C4 segfault (bug C, PR 5) are both resolved as of 2026-09-23, so this can ride with PR 4.

### Later, as RFCs / design discussions before any PR

| Piece | Overlay | Why it needs discussion first |
|---|---|---|
| Compact CPU-expert view for kt-kernel EP | `3805730e` | changes kt-kernel's memory model (8 GiB/layer for all experts → 3.4 GiB for the CPU subset); touches the loader contract and kt-kernel 0.7's API |
| File-backed Engram (`layout=file`) | `a1ca246a` | new env surface, thread pool inside the model forward, graph-break requirement (`breakable` decode); the upstream host-table path assumes `cudaHostRegister` |
| Layerwise streamed prefill of CPU experts | `0138c6ad` | POSIX-shm pinned buffer shared by TP ranks, per-layer export from kt-kernel, double-buffered DMA; a scheduler-adjacent feature with a known open NaN sighting (E22c/E24) |
| Expert deferral with the compact view | `11e7f745` | depends on kt-kernel's `--kt-max-deferred-experts-per-token`; ~~D ≥ 5 NaN is unresolved~~ the D≥5 NaN was kt-kernel bug C (T3.15, 2026-09-23); D=5 is rejected on ΔNLL +0.00287 and greedy-consistency grounds, so the upstreamable claim is "D ≤ 4 is quality-neutral"; quality argued by teacher-forced NLL, which upstream may want as a test |

## 4. Licences and provenance

| Component | Licence | Where recorded |
|---|---|---|
| This repository (scripts, docs, patches once exported) | Apache-2.0 | `LICENSE` |
| SGLang and the overlay commits (derivative work) | Apache-2.0 | `THIRD_PARTY_NOTICES.md`; patches must keep upstream headers |
| FlashInfer | Apache-2.0 | `THIRD_PARTY_NOTICES.md` |
| DeepSeek-V4.1-Flash weights | MIT | `THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt` |
| kt-kernel 0.7.0.post4 (KTransformers) | **to record** — read the wheel's `METADATA` `License` field and add a row to `THIRD_PARTY_NOTICES.md` before publishing patches that call its API | `PROVENANCE.md` has the version and prune manifest only |
| E20 tuned configs | generated data, Apache-2.0 with the repo | `kernels/e20/`, overlay configs dir |

No component is under a non-commercial or no-derivatives licence (`THIRD_PARTY_NOTICES.md`).

## 5. What is not upstream-ready, and the current gap

- **Not ready**: everything in the "Later" table; the NaN probe (`670994ac`); any change while E24's three NaN sightings are open — a PR that ships a known instability will be rejected and should be. (E23's two graph-mode faults are closed: PR 4 and PR 5.)
- **Runtime image drift (measured 2026-09-23)**: `config/Dockerfile.dsv41-kt` now applies `config/kt-kernel-patches/` after the kt-kernel install; until `dsv41-kt:recipe` is rebuilt, the patched `experts_base.py` is bind-mounted over the wheel with `EXTRA_DOCKER_ARGS` (see `scripts/serve-hybrid.sh`). `PROVENANCE.md` must list the patch series next to the wheel version.
- **Reproducibility gap (measured state)**: both the recipe repo and the overlay are local-only git repositories; no fork, tag or patch export exists; container state is ephemeral. Pushing only the recipe repo would not preserve the implementation because the overlay is a separate repository — the patch export in §2 item 1 is what closes that gap without a push.
- **Owner policy**: commit locally; **push only on an explicit instruction**, each time.

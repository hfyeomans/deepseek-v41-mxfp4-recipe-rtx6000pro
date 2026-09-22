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

### Early bug fixes that could ride along (need tests first)

- `cd55166a` BCG dataclass outputs and the `extra_page_block_size ∈ {64, 2}` re-page for FlashInfer SM120 prefill.
- `6032dd44`/`3b1fedce` speculative-aware prefill-graph-break gates — check whether upstream main has since changed these; if not, a small PR with the E23 attempt-2 traceback as the reproducer.
- `95f2fe3d` BCG output cap for speculative widths — **only after** E23's illegal memory access is resolved, so the PR is not carrying a live bug.

### Later, as RFCs / design discussions before any PR

| Piece | Overlay | Why it needs discussion first |
|---|---|---|
| Compact CPU-expert view for kt-kernel EP | `3805730e` | changes kt-kernel's memory model (8 GiB/layer for all experts → 3.4 GiB for the CPU subset); touches the loader contract and kt-kernel 0.7's API |
| File-backed Engram (`layout=file`) | `a1ca246a` | new env surface, thread pool inside the model forward, graph-break requirement (`breakable` decode); the upstream host-table path assumes `cudaHostRegister` |
| Layerwise streamed prefill of CPU experts | `0138c6ad` | POSIX-shm pinned buffer shared by TP ranks, per-layer export from kt-kernel, double-buffered DMA; a scheduler-adjacent feature with a known open NaN sighting (E22c/E24) |
| Expert deferral with the compact view | `11e7f745` | depends on kt-kernel's `--kt-max-deferred-experts-per-token`; D ≥ 5 NaN is unresolved; quality argued by teacher-forced NLL, which upstream may want as a test |

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

- **Not ready**: everything in the "Later" table; the NaN probe (`670994ac`); any change while E23's IMA and E24's three NaN sightings are open — a PR that ships a known instability will be rejected and should be.
- **Reproducibility gap (measured state)**: both the recipe repo and the overlay are local-only git repositories; no fork, tag or patch export exists; container state is ephemeral. Pushing only the recipe repo would not preserve the implementation because the overlay is a separate repository — the patch export in §2 item 1 is what closes that gap without a push.
- **Owner policy**: commit locally; **push only on an explicit instruction**, each time.

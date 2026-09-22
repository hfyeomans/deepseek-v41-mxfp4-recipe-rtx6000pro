# Troubleshooting

## `preflight.sh` fails with "checkpoint does not fit"

This is the expected outcome on two RTX PRO 6000. It is not a configuration problem. The calculator output above the failure line shows the margin (about −112 GiB with Engram in host DRAM). Options: see [FEASIBILITY.md §4](FEASIBILITY.md). To run experiment E5 anyway, set `ALLOW_UNQUALIFIED=1` and expect the container to die during weight load with a CUDA OOM at roughly 140 GiB demanded per GPU.

## Hugepages warning from `preflight.sh`

SGLang's Engram host table (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`) allocates the ~189 GiB FP8 tables in hugepage-backed memory. Reserve at runtime (reversible; fails if memory is fragmented, so do it soon after boot or while nothing large is running):

```bash
# 2 MiB pages: 190 GiB = 97,280 pages
sudo sysctl -w vm.nr_hugepages=97280
grep HugePages_ /proc/meminfo
```

Or at boot in the kernel command line: `hugepagesz=1G hugepages=190`. This host has 246.8 GiB; after 190 GiB of hugepages about 56 GiB remains for the OS and the serving process, which is enough for SGLang's host-side buffers but not for a second large model. Revert with `vm.nr_hugepages=0`.

If the GLM container is running, `docker stop glm53-flash-2x-rtxpro6000` first; it pins host memory.

## SM120 pitfalls reported by the community for DeepSeek-V4 / V4.1

From vLLM issues #47266 and #56700 and the 0xSero SGLang recipe. Not all reproduce on every build; check before applying.

| Symptom | Cause | Mitigation |
|---|---|---|
| Server hangs at startup with no log progress | DeepGEMM / FlashInfer autotune on SM120 | Disable autotune for the affected backend; SGLang cookbook says leave backends on auto |
| Crash during CUDA graph capture | Capture pool exceeds memory or unsupported kernel in graph | `--enforce-eager` (vLLM) / `--disable-cuda-graph` (SGLang); expect lower decode throughput |
| Wrong answers only on prompts ≥ 100K tokens | FP8 KV scale layout bug | Use `KV_CACHE_DTYPE=auto` (BF16) or a build that includes the fix; `verify.py --context 262144` catches it |
| MXFP4 experts run slowly | Marlin fallback instead of native FP4 grouped GEMM (vLLM #30135) | Check the kernel selection in the server log; compare against SGLang's FlashInfer MXFP4 path |
| All-reduce errors or NCCL hang on PCIe boxes | Custom all-reduce on non-NVLink topology | `--disable-custom-all-reduce` (vLLM); confirm `nvidia-smi topo -p2p r` shows OK |
| Sparse MLA dispatch error | FlashInfer SM120 dispatch gap for some head configurations | Use the SGLang preview image, which the 4-GPU recipe proved on SM120 |

## `dsv41-kt` dies at decode graph capture with `ImportError: libnvrtc.so.12`

Seen on the first T3.6d serve (2026-09-21). The traceback ends in `deep_gemm/__init__.py`, reached from `deepseek_v4_backend._fp4_paged_mqa_logits`. The base image (`dev-dsv41`, CUDA 13) is fine; the culprit is `pip install --no-deps kt-kernel==0.7.0.post4`. That wheel vendors ~1,000 files that are not kt-kernel at top level — a cu12 `deep_gemm` (832 files, 139-line `__init__` without `fp8_fp4_paged_mqa_logits`), `sgl_kernel` **0.3.21** (replacing 0.4.6.post1 and its `flashmla_ops`/`spatial_ops` `.so`), `triton_kernels`, `flash_attn_origin`, mscclpp — and `--no-deps` unpacks all of them over the base copies. `kt_kernel` itself imports none of them.

Fix (in `config/Dockerfile.dsv41-kt` since this date): download the wheel, run `config/prune_wheel.py` to keep only `kt_kernel/`, `kt_kernel.libs/` and the dist-info, install the pruned wheel. `config/kt_smoke.py` now fails the build if `sgl_kernel/version.py` ≠ 0.4.6.post1, if `deep_gemm/__init__.py` lacks the function, or if any `deep_gemm/*.so` links `libcudart.so.12`/`libnvrtc.so.12`. Rule for any future `--no-deps` install: list `site-packages` top-level entries before and after and diff them.

Do not "fix" this by installing `sgl-deep-gemm==0.2.0` or by moving the image to CUDA 12 — the base already has the right cu13 deep_gemm; the job is to stop overwriting it.

Runs made on the clobbered image (E17 one-layer test, E18 Engram bench) did not touch `sgl_kernel` or `deep_gemm`, so their numbers stand.

## `dsv41-kt` dies in `deep_gemm.fp8_fp4_paged_mqa_logits` with `deep_gemm_metadata=None`

Second T3.6d blocker (2026-09-21). On SM120 SGLang defaults `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1` and `SGLANG_OPT_USE_TILELANG_INDEXER=1` (the torch/TileLang fallbacks for the indexer), but V4.1-Flash's low-ratio indexer calls the FP4 DeepGEMM kernel unconditionally and expects the metadata those fallbacks never build.

Fix: launch with `--enable-deepseek-v4-fp4-indexer` **and** `-e SGLANG_OPT_USE_TILELANG_INDEXER=0 -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0`. The cu13 `deep_gemm` in `dsv41-kt:recipe` runs the FP4 indexer on SM120 fine once it is actually asked to.

## `Unsupported BCG output type: LogitsProcessorOutput` from the breakable decode graph

Third T3.6d blocker. `--cuda-graph-backend-decode breakable` (needed because the Engram `file` lookup is a graph break) copies outputs between graph segments and only knew tensors and containers, not the `LogitsProcessorOutput` dataclass the model returns.

Fix: overlay commit `cd55166a` (`breakable_cuda_graph_backend.py`) recurses into dataclasses. Upstream instead moved logits out of the graph (#40222) and no longer needs `breakable` for Engram; our `file` layout still does, so keep the overlay until the layout is ported onto that design.

## `sparse_mla_sm120_prefill.cu dispatch_dsv4_dual` aborts during the server's warmup prefill

Fourth T3.6d blocker. FlashInfer 0.6.18 compiles the SM120 DSv4 dual-pool prefill kernel only for `extra_page_block_size ∈ {64, 2}`. SGLang pages each compressed KV pool at `page_size // compress_ratio`, so with the default 256-token page the V4.1 pools come out at c1 = 256, c2 = 128, c4 = 64, c128 = 2 — and V4.1's `compress_ratios` (5×0, 20×1, 18×2) means 20 layers hit the unsupported 256 and 18 more the unsupported 128. The decode kernel takes `pbs_extra` at runtime and is unaffected; only prefills with more than 64 query rows (the warmup, every real prompt) crash.

Fix: overlay `cd55166a` in `flash_mla_sm120.py` — when the extra pool's page size is not 64 or 2 and divides by 64, re-page the touched pages to 64 through `_split_kv_pages_to_64(..., buffer_tag="extra")` (the tag keeps a separate cache buffer from the SWA split). Cost: ≈ 2–3 GB per card out of the ≈ 7 GB "available" pool; GPU used went 90 → 93.1 GB per card. A real fix is a FlashInfer build that instantiates 128/256, or running SGLang with `--page-size 64`.

## `probability tensor contains either inf, nan or element < 0` after minutes of clean serving

Fifth T3.6d blocker (attempt 5, 12 minutes in). Device-side assert in sampling on the first decode after a 4-sequence prefill with **distinct** prompts; the async error surfaces wherever the next sync happens (`engram.py _engram_file_rows` inside the breakable replay), which is misleading. Not reproducible on demand: every prompt alone, C3/C4 distinct at 64 tokens, and the original c2→c4 256-token sequence all passed on attempt 6.

If it fires: save `docker logs dsv41-kt` (the file is the only evidence; the crash dump interleaves both ranks), `docker rm -f dsv41-kt`, relaunch. A GPU left at "100 % utilization, 2 MiB, ~95 W" with no process after the crash is the dead context's kernel state; it clears when the next process creates a context, so no `nvidia-smi -r` is needed. Suspects and probes are in E19 item 8.

## Thinking mode surprises

DeepSeek-V4.1 reasoning is on by default at effort 50 in both engines. Every client in this repo sends `chat_template_kwargs: {"thinking": false}` unless asked otherwise. If your own client sees long, slow answers, that is why. The recipe's smoke test is `17*19` → `323`.

## Weight load is slow

475 GiB from NVMe at ~14 GB/s is at least 33 s of pure I/O; the engine's dequant and layout passes dominate, and the Engram host-table copy adds a pass over 189 GiB. Ten to thirty minutes is normal on the 4-GPU profile. `wait-ready.py` defaults to a one-hour timeout.

## `download-model.sh` refuses to start

It needs ~500 GiB free at `MODEL_DIR`. On this host nvme0 has 1.7 TB free; make sure `MODEL_DIR` is on it and not on a smaller volume. The checkpoint interleaves vision, DSpark and text tensors across all 48 shards, so a partial download cannot save space.

## `ci.sh` reports "possible identifier or secret"

The scan catches home paths, LAN addresses, GPU UUIDs, PCI bus IDs and token-shaped strings in tracked files. Scrub the file (replace with placeholders) rather than whitelisting it; evidence JSON is meant to be publishable.

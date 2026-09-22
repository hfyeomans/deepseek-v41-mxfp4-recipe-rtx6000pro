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

## DSpark on: `illegal memory access` in `dspark_verify.py … breakable_cuda_graph.py replay → seg.replay()` on the first decode after a long prompt

E23 attempts 5–7. Short prompts (≤ 512 tokens) speculate, but with a low accept rate (0.26–0.57) and garbled tool calls; the first target-verify step after a 5,376- or 13,463-token prompt dies with `cudaErrorIllegalAddress` inside the breakable graph replay. `CUDA_LAUNCH_BLOCKING=1` names no kernel because the fault surfaces at graph replay. **Eager mode runs the same prompt correctly** (needles 3/3, accept rate 0.77, tool call parsed), so the graph replay, not the kernels, is wrong.

**Status: open — root cause not found.** Hypothesis tested and falsified (attempt 8): `DeepseekV4AttnBackend.shared_read_ends` returns `IN_REPLAY` for DSpark target-verify on the assumption that "DSPARK verify replays one full (non-breakable) graph", while our recipe runs `--cuda-graph-backend-decode breakable` (the Engram `file` lookup is a graph break), so later segments re-read scheduler-shared buffers (`req_to_token`, KV metadata) after the in-graph marker and the overlap scheduler could rewrite them for the next batch mid-verify. The non-DSpark branch of the same function says `POST_REPLAY` for that reason, so the overlay was changed to return `IN_REPLAY` only when `check_cuda_graph_backend(Phase.DECODE, Backend.BREAKABLE)` is false. **The IMA reproduced unchanged with that fix in place** (attempt 8 log, local `evidence/e23-spec-decode/crash/attempt8-shared-read-fix-still-ima.log`; `*.log` is gitignored), so the early marker is at most a latent race, not this fault. The change is kept because the non-breakable assumption is still wrong for this recipe, but it is not the fix.

Next bisect steps (T3.9, not yet run): `EXTRA_SERVER_ARGS="--debug-cuda-graph"` (every op eager, but through the same segmented capture/replay path — separates "kernel inside a replayed segment" from "segment plumbing"); then the target-verify metadata path — the decode runner's `init_forward_metadata_out_graph` / `init_forward_metadata_in_graph` and the verify branch of `deepseek_v4_backend.py` — checking that the per-segment metadata (`seq_lens`, KV indices, the re-paged extra pool from the `_split_kv_pages_to_64` fix) is sized for `bs × (K+1)` tokens after a long prefix, since short prompts survive and long ones fault.

Two traps met on the way:
- `--disable-cuda-graph` is `no_cli` in this fork and, in `arg_groups/cuda_graph_hook.py`, the explicit `--cuda-graph-backend-decode breakable` that `serve-hybrid.sh` always passes wins over both `--disable-cuda-graph` and `--disable-decode-cuda-graph`. To run eager, append `EXTRA_SERVER_ARGS="--cuda-graph-backend-decode disabled"` (argparse last-wins). `--debug-cuda-graph` is the middle ground: every op eager but still through the breakable capture/replay path.
- Eager target-verify on the vision-capable DSV4 model entered `_prepare_mm_embeddings` (TARGET_VERIFY is `is_extend()` and text-only requests still carry a `MultimodalInputs` object) and died with `TypeError: 'NoneType' object is not iterable` in `_count_mm_tokens_in_extend` because a verify batch has no `extend_prefix_lens_cpu`. Overlay fix in `models/deepseek_v4.py`: skip the mm-embedding branch for `is_decode_or_idle()` and `is_target_verify()`, matching the vision check two lines below. Graph mode never saw it because the captured dummy batch has `mm_inputs=None`.

## `probability tensor contains either inf, nan or element < 0` after minutes of clean serving

Fifth T3.6d blocker (attempt 5, 12 minutes in). Device-side assert in sampling on the first decode after a 4-sequence prefill with **distinct** prompts; the async error surfaces wherever the next sync happens (`engram.py _engram_file_rows` inside the breakable replay), which is misleading. Not reproducible on demand: every prompt alone, C3/C4 distinct at 64 tokens, and the original c2→c4 256-token sequence all passed on attempt 6.

If it fires: save `docker logs dsv41-kt` (the file is the only evidence; the crash dump interleaves both ranks), `docker rm -f dsv41-kt`, relaunch with `bash scripts/serve-hybrid.sh`. A GPU left at "100 % utilization, 2 MiB, ~95 W" with no process after the crash is a faulted channel the driver still counts as busy — see the dedicated entry below; **correction 2026-09-22:** creating a new context does *not* clear it (a full GEMM/copy test on GPU0 ran at full speed and the card still read 100 % / P0 afterwards); only `nvidia-smi --gpu-reset` or a reboot does. Suspects and probes are in E19 item 8.

Third sighting on attempt 7 (E20, 2026-09-22): first decode after a 1-running + 3-new distinct-prompt prefill, assert on both TP ranks. Not the tuned E20 GEMM — `kernels/e20/check_sglang_w8a8_path.py` ran the real SGLang path at M 1–1,025 for all seven shapes, 70/70 finite. Probe applied in overlay `1d6a98d3`: the page-split scratch buffer in `flash_mla_sm120.py::_split_kv_pages_to_64` was `torch.empty`, so any destination page the touched-page mask never marked held stale VRAM bytes — and in FP8 e4m3 `0x7F`/`0xFF` are NaN, which fits a non-deterministic, first-mixed-batch, both-ranks failure. Now `torch.zeros`. Attempt 8 passed the exact reproducer; the soak result is in E20 item 6 and `evidence/t36d-first-serve/attempt8-soak/`. If it fires again after this, the next probe is `SGLANG_SM120_FLASHMLA_BACKEND=triton` to move the SM120 prefill kernel out of the picture.

**Deterministic variant (E21, 2026-09-22):** with `KT_MAX_DEFERRED=5` or `6` (`--kt-max-deferred-experts-per-token`) the same assert fires on the *first* bs=4 decode step, every time, and D=6 also costs +0.23 nats on `nll-score.py`. This is a different bug from the zero-fill one above — it reproduces on demand and is tied to deferring ≥ 5 of the CPU-tier experts per token. Leave the default `KT_MAX_DEFERRED=4` (E21 default; +22 % C1, ΔNLL −0.006, soak-clean at bs 1–4). Log of the D=5 crash: `evidence/e21-kt-fixed-floor/serve-d5-crash.log`; root cause is open (T3.6d3).

**Multi-sequence D=4 sighting (E22c, 2026-09-22; first read as a streamed-prefill fault):** with `KT_GPU_PREFILL_MIN_TOKENS=1024` resident, the assert fired on the first bs=4 decode after a **4-sequence prefill batch the scheduler logged as `#new-token: 1024`**. E22d showed the batch had 664 real tokens (4 × the 166-token bench fixture; the scheduler pads `#new-token` per sequence to a multiple of 256, and `Prefill batch` lines print *after* the forward), so it ran the **regular kt-kernel D=4 CPU path, not the streamed one**. The streamed kernel is finite offline at T = 1,024 / 1,175 / 4,096 and in-server at 1,978–8,192 tokens, and 20 same-shape 4-sequence batches (14 HTTP C4 runs + 6 forced 4-prompt `/generate` batches) ran clean with the per-layer probe on. Circumstances that did not recur: 0.98 GiB free on TP1 (after 16K + 32K prefill plus NLL logprobs) and three Triton kernels lazily loading in the same second. Log: `evidence/e22-layerwise-prefill/serve-e22c-crash.log`. Streamed prefill is the default since 2026-09-22; this sighting is tracked with the D≥5 case under E24 (NaN as a release defect): no retry/clamp workarounds; run soaks with `KT_NAN_PROBE_MIN_TOKENS=1` so the next occurrence names the layer and whether the MoE *input* was already non-finite (attention/dense side) or only its output (CPU path).

## GPU0 reads 100 % utilisation, P0, ~94 W with no model loaded (and it keeps coming back)

Seen after every server crash that ends in a device-side assert (the NaN entries above). `journalctl -k | grep Xid` shows one `NVRM: Xid (PCI:0000:01:00): 43` (GPU0) and one for `c1:00` (GPU1) per crash — 43 is "GPU stopped processing, user-level fault". GPU1 recovers to P8; GPU0 is left with a faulted channel the driver still samples as busy: `nvidia-smi` shows 100 % utilisation, 2 MiB used, P0, SM clock 2,325 MHz, ~94 W, and `--query-compute-apps` lists nothing. It is a phantom for throughput — a fresh context on GPU0 measured 305 TFLOP/s bf16 GEMM and 1,365 GiB/s D2D against 308 / 1,367 on GPU1 — but the clocks stay pinned (≈ 68 W wasted at idle) and nvtop/nvidia-smi are useless until it clears. It "happens more often" exactly as often as the server asserts.

Clear it (no model running):

```
sudo nvidia-smi --gpu-reset -i 0
nvidia-smi --query-gpu=index,utilization.gpu,power.draw,pstate --format=csv   # expect 0 %, ~25 W, P8
```

The reset is refused with "GPU is in use" if any process holds `/dev/nvidia0`; today that is only the display kiosk (`cage` opens both `/dev/nvidia0` and `/dev/nvidia1` even though Chrome runs with `--disable-gpu` on `renderD129` = GPU1). Stop the kiosk, reset, restart it; a dedicated small display GPU removes the dependency. Creating a new CUDA context does **not** clear it (tested 2026-09-22). The real fix is upstream of this entry: zero NaN crashes (E24).

## `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!` (and the server is slow, not broken)

Not a crash — one WARNING line per dense shape at startup, easy to miss, and worth ~2× of the decode step (E20). SGLang's Triton block-FP8 GEMM looks for `kernels/ops/quantization/configs/N=…,K=…,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[32, 32].json`. Upstream ships those files only for H200 (and for `[128, 128]` on H100/B200). Without one, the default config runs the seven V4.1-Flash dense shapes at 60–80 GB/s on a 1,600 GB/s card: 40 ms of a 70 ms bs=1 step (T3.6d3 profile).

Fix: `kernels/e20/tune_w8a8_block32.py` (run inside the container, see its docstring) writes the seven JSONs; copy them into the overlay's `python/sglang/kernels/ops/quantization/configs/` (bind-mounted, so no image rebuild) and make sure the overlay carries the `is_sm120` gate lift in `fp8_kernel.py::w8a8_block_fp8_matmul_triton` — the tuned configs use `SPLIT_K`/`SWAP_AB`, which stock SGLang only honours on SM90. Check the fix took: the warning disappears and `Using configuration from … for W8A8 Block FP8 kernel.` appears for each shape. If you change `--tp`, the shapes change (N is sharded) and the files must be re-tuned; the warning tells you the exact missing shape.

## Thinking mode surprises

DeepSeek-V4.1 reasoning is on by default at effort 50 in both engines. Every client in this repo sends `chat_template_kwargs: {"thinking": false}` unless asked otherwise. If your own client sees long, slow answers, that is why. The recipe's smoke test is `17*19` → `323`.

## Weight load is slow

475 GiB from NVMe at ~14 GB/s is at least 33 s of pure I/O; the engine's dequant and layout passes dominate, and the Engram host-table copy adds a pass over 189 GiB. Ten to thirty minutes is normal on the 4-GPU profile. `wait-ready.py` defaults to a one-hour timeout.

## `download-model.sh` refuses to start

It needs ~500 GiB free at `MODEL_DIR`. On this host nvme0 has 1.7 TB free; make sure `MODEL_DIR` is on it and not on a smaller volume. The checkpoint interleaves vision, DSpark and text tensors across all 48 shards, so a partial download cannot save space.

## `ci.sh` reports "possible identifier or secret"

The scan catches home paths, LAN addresses, GPU UUIDs, PCI bus IDs and token-shaped strings in tracked files. Scrub the file (replace with placeholders) rather than whitelisting it; evidence JSON is meant to be publishable.

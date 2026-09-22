# Frozen provenance

Every artifact this plan reasons about is identified by an immutable revision. Byte totals were read from the safetensors shard headers at these revisions on 2026-09-20 by [`scripts/checkpoint-budget.py`](scripts/checkpoint-budget.py) and are committed in [`evidence/`](evidence/).

| Layer | Immutable identity | Bytes |
|---|---|---:|
| Target model (pinned in `scripts/load-env.sh`) | `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277` | 510,286,023,000 (475.24 GiB, 48 shards, 96,085 tensors) |
| NVIDIA NVFP4 re-quantization (evaluated, not selected) | `nvidia/DeepSeek-V4.1-Flash-NVFP4@3431dde3247c13b5957f682b1e3c6fcae2566079` | 491.06 GiB (188,245 tensors) |
| LibertAI FP4-Engram variant (evaluated, not selected) | `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4@dfce15b92ed1fa76e80e2a46ba847e5b5451f12c` | 399.90 GiB |
| Baseline model (sibling recipe, fits 2 GPUs) | `nvidia/DeepSeek-V4-Flash-0731-NVFP4@f1caa71142bd0be02f728c79f75042ac1e461579` | ≈ 164 GB on disk |
| Baseline source | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | ≈ 158 GB on disk |
| Runtime image (4-GPU profile; base for the 2-GPU hybrid) | `lmsysorg/sglang:dev-dsv41@sha256:4a5d132a06a77c8331e15845f2e925adc788b00105097ad55409afa3f4fa4860` — image label `ai.sglang.build.commit=da64c5cbb8cf6bfd39be19da43573fdfd484c43a` (SGLang main 2026-09-11, "[DSV4.1] Support raw-index output in TopK v2 #39098"), built 2026-09-11T18:28Z; py 3.12, torch 2.13.0+cu130, FlashInfer 0.6.18, triton 3.7.1, nvidia-cutlass-dsl 4.6.2; no `kt_kernel` | 51.7 GB |
| CPU-expert reference image (E6 stage 1) | `approachingai/ktransformers:DSV4-specific@sha256:9f0747de8d10d5d36bfc38da98bf6bfe8dfeb43a59699a527c7cd303f5d743fd` — labels: sglang-kt on fork `yyj6666667/sglang@5d861db9aa5af9de44172494c8c73b234c94295f`, ktransformers `3a19fd6d`, CUDA 12.8.1, FlashInfer 0.6.15.post1, SM 80–120, layerwise prefill default `enabled-threshold-2048-swa-0.4`; installed `sglang-kt 0.7.0.post1`, `kt-kernel 0.7.0.post1`, torch 2.9.1+cu128, py 3.11 | 37.1 GB |
| CPU expert kernel (for the 2-GPU hybrid image) | PyPI `kt-kernel==0.7.0.post4` (`cp312-manylinux_2_35_x86_64`; metadata pins `torch==2.9.1` but the extension links only gomp/hwloc/numa — installs with `--no-deps` into `dev-dsv41` and imports, variant `avx512_bf16`) | — |
| 2-GPU hybrid runtime image (T3.6a, built locally; rebuilt 2026-09-21 after T3.6d) | `dsv41-kt:recipe` = `config/Dockerfile.dsv41-kt` on the `dev-dsv41` digest above + `kt-kernel==0.7.0.post4` **pruned** to its own `kt_kernel/`, `kt_kernel.libs/`, dist-info (`config/prune_wheel.py`) and installed `--no-deps`. The unpruned wheel vendors a cu12 `deep_gemm` and `sgl_kernel` 0.3.21 that overwrote the base's cu13 deep_gemm and sgl-kernel 0.4.6.post1 (first image id `sha256:362c5d58…`; TROUBLESHOOTING.md). Build-time `config/kt_smoke.py` checks the `avx512_bf16` extension loads under torch 2.13.0+cu130, the `[layers, experts]` placement mask helper, `sgl_kernel/version.py` = 0.4.6.post1, `deep_gemm/__init__.py` has `fp8_fp4_paged_mqa_logits`, no cu12 `.so` under `deep_gemm/`. Current image id `sha256:d6526ff3a0ec…`. Rebuild: `docker build -f config/Dockerfile.dsv41-kt -t dsv41-kt:recipe config/` | 51.8 GB |
| 4-GPU reference recipe | `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000` (MIT); its SGLang base `sha256:c4ca6511…`, model rev `fb2764a5…` as published by its author | — |
| Target license text | `THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt`, sha256 `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` | 1,084 |

Why the upstream checkpoint and not an NVFP4 re-quantization: experts in the upstream checkpoint are already FP4 (MXFP4, UE8M0 block-32 scales). NVIDIA's NVFP4 is 15.8 GiB larger (g16 FP8 scales) and offers kernel compatibility rather than size; LibertAI's FP4 Engram is lossy and unevaluated, and Engram size is not the constraint on this host (246.8 GiB DRAM holds the FP8 tables). The upstream checkpoint is also the one SGLang's `dev-dsv41` cookbook targets.

## Checkpoint identity checks

`scripts/preflight.sh` verifies `RECIPE_PIN.txt` equals the pin above, that the index maps to exactly 48 shards under safe relative paths, that shard bytes equal the pinned tensor total plus header overhead, and that `config.json` and `tokenizer_config.json` exist. `scripts/download-model.sh` writes the pin after `hf download --revision`.

`config.json` at the pinned revision declares `quantization_config: {quant_method: fp8, weight_block_size: [32, 32], scale_fmt: ue8m0, expert_dtype: fp4}`, 40 layers, 384 routed experts top-6, Engram layers [1, 14] with 384,006,168 and 384,016,682 rows, `max_position_embeddings 1,048,576` (YaRN ×16 over 65,536).

## Host condition under which evidence was gathered

Recorded in [`evidence/hardware-inventory.json`](evidence/hardware-inventory.json): ASUS Pro WS WRX90E-SAGE SE BIOS 1317; Threadripper PRO 9955WX; 246.8 GiB DDR5; 2 × RTX PRO 6000 Blackwell Max-Q 97,887 MiB at 300 W; driver 615.71.09; Ubuntu 26.04.1, kernel 7.0.0-31. DIMM part numbers pending root access.

## Edits made to sibling recipes from this work

- `~/dev/src/deepseek-v4-flash-nvfp4-tp2` commit `42e1f61` (local): `example.env`/`.env` profile → `GPU_MEMORY_UTILIZATION=0.9651 MAX_NUM_SEQS=16 MAX_CUDAGRAPH_CAPTURE_SIZE=96` (E15/T2.4 measurements), `docs/running.md` profile table, README summary line, `tests/test_operator_config.py` pins.
- `~/dev/src/glm-5.3-flash-2x-rtx-pro-6000-blackwell` commit `324f478` (local): `scripts/serve.sh` capture size derived from `MAX_NUM_SEQS × (DFLASH_TOKENS + 1)` (was hard-coded 96), `scripts/validate-config.sh` accepts up to 32 slots, `config/example.env`/`.env` comment recording the ≈ 20-sequence ceiling (E16). Profile values unchanged.

## Overlay commits (`~/dev/src/sglang-dsv41-overlay`, branch `dsv41-kt-hybrid`, local only)

- `3805730e` compact CPU expert view in the kt-kernel loader (E17).
- `a1ca246a` Engram `file` layout (E18).
- `cd55166a` breakable-graph dataclass outputs + `extra` KV pool re-page to 64-row pages for FlashInfer's SM120 prefill (E19 blockers 3–4).
- `32a25f37` seven tuned `w8a8 block_shape=[32, 32]` Triton configs for `NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition` and the `is_sm90` → `is_sm90 or is_sm120` gate in `fp8_kernel.py` (E20).
- `1d6a98d3` zero-fill the SM120 page-split scratch buffers in `flash_mla_sm120.py` (T3.6d2 NaN probe).

## Qualification status

**No V4.1-Flash profile in this repository is qualified.** The 4-GPU profile in `serve.sh` is assembled from the SGLang cookbook and the 0xSero recipe and has not run on this host. The README verdict flips only when experiment E9 in [`plan/EXPERIMENTS.md`](plan/EXPERIMENTS.md) produces the tables in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

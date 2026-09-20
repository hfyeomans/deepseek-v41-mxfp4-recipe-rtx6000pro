# Frozen provenance

Every artifact this plan reasons about is identified by an immutable revision. Byte totals were read from the safetensors shard headers at these revisions on 2026-09-20 by [`scripts/checkpoint-budget.py`](scripts/checkpoint-budget.py) and are committed in [`evidence/`](evidence/).

| Layer | Immutable identity | Bytes |
|---|---|---:|
| Target model (pinned in `scripts/load-env.sh`) | `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277` | 510,286,023,000 (475.24 GiB, 48 shards, 96,085 tensors) |
| NVIDIA NVFP4 re-quantization (evaluated, not selected) | `nvidia/DeepSeek-V4.1-Flash-NVFP4@3431dde3247c13b5957f682b1e3c6fcae2566079` | 491.06 GiB (188,245 tensors) |
| LibertAI FP4-Engram variant (evaluated, not selected) | `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4@dfce15b92ed1fa76e80e2a46ba847e5b5451f12c` | 399.90 GiB |
| Baseline model (sibling recipe, fits 2 GPUs) | `nvidia/DeepSeek-V4-Flash-0731-NVFP4@f1caa71142bd0be02f728c79f75042ac1e461579` | ≈ 164 GB on disk |
| Baseline source | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` | ≈ 158 GB on disk |
| Runtime image (4-GPU profile) | `lmsysorg/sglang:dev-dsv41` — **digest not yet pinned**; pull, then record `sha256:` here and in `.env` | — |
| 4-GPU reference recipe | `0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000` (MIT); its SGLang base `sha256:c4ca6511…`, model rev `fb2764a5…` as published by its author | — |
| Target license text | `THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt`, sha256 `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` | 1,084 |

Why the upstream checkpoint and not an NVFP4 re-quantization: experts in the upstream checkpoint are already FP4 (MXFP4, UE8M0 block-32 scales). NVIDIA's NVFP4 is 15.8 GiB larger (g16 FP8 scales) and offers kernel compatibility rather than size; LibertAI's FP4 Engram is lossy and unevaluated, and Engram size is not the constraint on this host (246.8 GiB DRAM holds the FP8 tables). The upstream checkpoint is also the one SGLang's `dev-dsv41` cookbook targets.

## Checkpoint identity checks

`scripts/preflight.sh` verifies `RECIPE_PIN.txt` equals the pin above, that the index maps to exactly 48 shards under safe relative paths, that shard bytes equal the pinned tensor total plus header overhead, and that `config.json` and `tokenizer_config.json` exist. `scripts/download-model.sh` writes the pin after `hf download --revision`.

`config.json` at the pinned revision declares `quantization_config: {quant_method: fp8, weight_block_size: [32, 32], scale_fmt: ue8m0, expert_dtype: fp4}`, 40 layers, 384 routed experts top-6, Engram layers [1, 14] with 384,006,168 and 384,016,682 rows, `max_position_embeddings 1,048,576` (YaRN ×16 over 65,536).

## Host condition under which evidence was gathered

Recorded in [`evidence/hardware-inventory.json`](evidence/hardware-inventory.json): ASUS Pro WS WRX90E-SAGE SE BIOS 1317; Threadripper PRO 9955WX; 246.8 GiB DDR5; 2 × RTX PRO 6000 Blackwell Max-Q 97,887 MiB at 300 W; driver 615.71.09; Ubuntu 26.04.1, kernel 7.0.0-31. DIMM part numbers pending root access.

## Qualification status

**No V4.1-Flash profile in this repository is qualified.** The 4-GPU profile in `serve.sh` is assembled from the SGLang cookbook and the 0xSero recipe and has not run on this host. The README verdict flips only when experiment E9 in [`plan/EXPERIMENTS.md`](plan/EXPERIMENTS.md) produces the tables in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

# Attribution

This repository is a feasibility study and integration plan. It does not claim authorship of DeepSeek-V4.1-Flash, its quantizations, SGLang, vLLM, FlashInfer, CUDA, Docker or the NVIDIA container stack.

## Models

- **DeepSeek-AI** created and published DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, MIT) and DeepSeek-V4-Flash-0731. Their license text is retained in [`THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt`](THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt). The architecture facts in this plan (CSA2, Engram, DSpark, hyper-connections) are theirs, read from the published `config.json` and model card.
- **NVIDIA** published `nvidia/DeepSeek-V4.1-Flash-NVFP4` and `nvidia/DeepSeek-V4-Flash-0731-NVFP4` (MIT), the latter being the baseline this host already serves.
- **LibertAI** published `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4` (MIT) with the FP4 Engram experiment and a clear statement of the residency constraint, which this plan confirms numerically.

## Runtimes and references

- **SGLang / LMSYS contributors** built the `dev-dsv41` preview, the DeepSeek-V4.1 cookbook, the Engram host-table path and the DSpark integration that the 4-GPU profile uses.
- **vLLM contributors** built DeepSeek-V4.1 support (PR #56228, tag `deepseekv41-flash-0909`) and the recipe whose memory table this plan cross-checks; community members in issues #47266 and #56700 documented the SM120 pitfalls summarised in `docs/TROUBLESHOOTING.md`.
- **0xSero** published `deepseek-v4.1-flash-4x-rtx-pro-6000` (MIT), the only consumer-GPU V4.1 deployment with numbers; it is the calibration point for `scripts/feasibility-calc.py` and the template for `serve.sh`.
- **hermia-ai, lastloop-ai, jacklarmer** published earlier DeepSeek-V4 SM120 recipes that informed the pitfalls list.
- **T.J. Purtell and Samuel Cardillo**'s GLM-5.3 Flash dual-RTX-PRO-6000 recipe provided the repository structure, the fail-closed preflight pattern, the benchmark method and the identifier-scan CI that this repository copies.

## Ecosystem

Hugging Face (hub, `hf` CLI, safetensors format), PyTorch, NVIDIA (CUDA, drivers, container toolkit), FlashInfer, Docker/Moby and their contributors, under their respective licenses.

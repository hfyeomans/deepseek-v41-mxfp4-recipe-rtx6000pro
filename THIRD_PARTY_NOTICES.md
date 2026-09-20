# Third-party notices

This repository is licensed under Apache-2.0 (see [LICENSE](LICENSE)). It contains no model weights, container layers or third-party source code. The components it depends on or refers to are distributed separately under their own terms:

| Component | License | Notes |
|---|---|---|
| DeepSeek-V4.1-Flash, DeepSeek-V4-Flash-0731 (DeepSeek-AI) | MIT | Text retained in `THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt` |
| DeepSeek-V4.1-Flash-NVFP4, DeepSeek-V4-Flash-0731-NVFP4 (NVIDIA) | MIT | As declared on the model cards |
| DeepSeek-V4.1-Flash-NVFP4 (LibertAI) | MIT | As declared on the model card |
| SGLang (`lmsysorg/sglang:dev-dsv41`) | Apache-2.0 | Container image pulled by the operator, not redistributed |
| vLLM | Apache-2.0 | Referenced for the alternative profile and pitfalls |
| FlashInfer | Apache-2.0 | Inside the runtime image |
| PyTorch | BSD-3-Clause | Inside the runtime image; used by `bench-pcie-h2d.py` |
| NVIDIA driver, CUDA, NVIDIA Container Toolkit | NVIDIA EULA / Apache-2.0 (toolkit) | Installed by the operator |
| Docker / Moby | Apache-2.0 | Installed by the operator |
| Hugging Face Hub client (`hf`) | Apache-2.0 | Used by `download-model.sh` |
| 0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000 | MIT | Design reference; no code copied |
| GLM-5.3 Flash 2× RTX PRO 6000 recipe (Cardillo / Purtell) | Apache-2.0 | Repository structure and CI pattern adapted |

No component here is under a non-commercial or no-derivatives license. If a future DSpark drafter or quantization with such terms is introduced, it must be gated the way the sibling recipe gates its DFlash2 draft.

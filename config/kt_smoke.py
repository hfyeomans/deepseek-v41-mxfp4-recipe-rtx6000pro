"""Import smoke test for the dsv41-kt image (T3.6a).

Runs at build time and on demand. Fails loudly if kt-kernel is missing, picked
the wrong SIMD variant for this host class, or cannot see the MXFP4 pieces the
2-GPU hybrid depends on (E6-DESIGN §4.1). Does not touch a GPU.
"""

import importlib.metadata
import sys

import torch

import kt_kernel
from kt_kernel.utils import loader

EXPECTED_VARIANT = "avx512_bf16"  # Zen 5 (Threadripper PRO 9955WX); see HARDWARE.md
V41_MOE_LAYERS, V41_EXPERTS = 45, 384  # V4.1-Flash shapes used by the mask check


def main() -> int:
    ext_file = getattr(kt_kernel.kt_kernel_ext, "__file__", "") or ""
    print(f"kt-kernel {importlib.metadata.version('kt-kernel')}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    print(f"extension {ext_file}")

    problems = []
    if EXPECTED_VARIANT not in ext_file:
        problems.append(f"extension is not the {EXPECTED_VARIANT} build: {ext_file}")
    for name in ("KTMoEWrapper", "generate_gpu_experts_masks"):
        if not hasattr(kt_kernel, name):
            problems.append(f"kt_kernel lacks {name}")
    if not hasattr(loader, "MXFP4SafeTensorLoader"):
        problems.append("kt_kernel.utils.loader lacks MXFP4SafeTensorLoader")

    # --kt-num-gpu-experts N becomes a [layers, experts] bool mask holding
    # layers*N experts chosen by *global* top-k of activation frequency
    # (uniform frequencies -> ties, still exactly layers*N set).
    freq = torch.ones(V41_MOE_LAYERS, V41_EXPERTS)
    per_layer = 224
    mask = kt_kernel.generate_gpu_experts_masks(freq, per_layer * V41_MOE_LAYERS)
    if mask.shape != (V41_MOE_LAYERS, V41_EXPERTS) or int(mask.sum()) != per_layer * V41_MOE_LAYERS:
        problems.append(f"gpu mask: shape {tuple(mask.shape)}, {int(mask.sum())} set")

    for p in problems:
        print("FAIL:", p)
    if problems:
        return 1
    print("ok: MXFP4 CPU expert path importable, placement mask helper works")
    return 0


if __name__ == "__main__":
    sys.exit(main())

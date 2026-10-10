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
V41_MOE_LAYERS, V41_EXPERTS = 40, 384  # V4.1-Flash shapes used by the mask check
# Base-image packages that kt-kernel's unpruned wheel overwrote (Dockerfile note).
BASE_SGL_KERNEL = "0.4.6.post1"
BASE_DEEP_GEMM_FN = "fp8_fp4_paged_mqa_logits"  # required by deepseek_v4_backend


def base_image_intact() -> list[str]:
    """File-level checks: the build stage has no GPU driver, so sgl_kernel and
    deep_gemm cannot be imported here (both dlopen libcuda.so.1)."""
    import importlib.util
    import re
    from pathlib import Path

    problems = []
    site = Path(importlib.util.find_spec("kt_kernel").origin).parent.parent

    ver_file = site / "sgl_kernel" / "version.py"
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", ver_file.read_text()) if ver_file.exists() else None
    sgl_ver = m.group(1) if m else "missing"
    if sgl_ver != BASE_SGL_KERNEL:
        problems.append(f"sgl_kernel/version.py says {sgl_ver}, base image has {BASE_SGL_KERNEL}")

    dg_init = site / "deep_gemm" / "__init__.py"
    if not dg_init.exists() or BASE_DEEP_GEMM_FN not in dg_init.read_text():
        problems.append(f"deep_gemm/__init__.py lacks {BASE_DEEP_GEMM_FN} (vendored cu12 copy?)")
    cu12_libs = [
        p.name
        for p in (site / "deep_gemm").rglob("*.so")
        if re.search(rb"lib(nvrtc|cudart)\.so\.12", p.read_bytes())
    ]
    if cu12_libs:
        problems.append(f"deep_gemm .so links CUDA 12: {cu12_libs}")

    if torch.version.cuda and not torch.version.cuda.startswith("13"):
        problems.append(f"torch cuda is {torch.version.cuda}, expected 13.x")
    print(f"sgl_kernel {sgl_ver}; deep_gemm {dg_init} ({len(dg_init.read_text().splitlines()) if dg_init.exists() else 0} lines)")
    return problems


def main() -> int:
    ext_file = getattr(kt_kernel.kt_kernel_ext, "__file__", "") or ""
    print(f"kt-kernel {importlib.metadata.version('kt-kernel')}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda}")
    print(f"extension {ext_file}")

    problems = base_image_intact()
    if EXPECTED_VARIANT not in ext_file:
        problems.append(f"extension is not the {EXPECTED_VARIANT} build: {ext_file}")
    for name in ("KTMoEWrapper", "generate_gpu_experts_masks"):
        if not hasattr(kt_kernel, name):
            problems.append(f"kt_kernel lacks {name}")
    if not hasattr(loader, "MXFP4SafeTensorLoader"):
        problems.append("kt_kernel.utils.loader lacks MXFP4SafeTensorLoader")

    # kt patch 0003 (T3.39 window 2): the avx512_bf16 extension is rebuilt from
    # source with the caller-owned weight arena; the stock wheel lacks these.
    moe = kt_kernel.kt_kernel_ext.moe
    fp4 = getattr(moe, "AMXFP4_KGroup_MOE", None)
    if fp4 is None:
        problems.append("kt_kernel_ext.moe lacks AMXFP4_KGroup_MOE")
    elif not (hasattr(fp4, "weight_arena_bytes") and hasattr(fp4, "weight_arena_layout")):
        problems.append("AMXFP4_KGroup_MOE lacks weight_arena_bytes/weight_arena_layout (patch 0003 not compiled in)")
    cfg = moe.MOEConfig(1, 1, 64, 64)
    cfg.weight_arenas = [64]
    if getattr(cfg, "weight_arenas", None) != [64]:
        problems.append(f"MOEConfig.weight_arenas does not round-trip: {getattr(cfg, 'weight_arenas', None)}")
    # The rebuilt extension must keep the wheel's CUDA host-stream bridge
    # (cudaLaunchHostFunc behind submit/sync_with_cuda_stream): the hybrid's
    # decode graphs and layerwise prefill submit through it. A CPUINFER_USE_CUDA=0
    # build compiles these bindings out (KTRANSFORMERS_CPU_ONLY) and only fails
    # at CUDA-graph warmup (window 2 qualification attempt 1).
    cpu_infer = kt_kernel.kt_kernel_ext.CPUInfer
    for name in ("submit_with_cuda_stream", "sync_with_cuda_stream"):
        if not hasattr(cpu_infer, name):
            problems.append(f"CPUInfer lacks {name} (extension built without CPUINFER_USE_CUDA=1)")

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

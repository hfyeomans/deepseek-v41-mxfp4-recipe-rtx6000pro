#!/usr/bin/env bash
# Build the E13 stage-1 grouped GEMM (MXFP4 x MXFP4 -> BF16) for SM120 (RTX PRO 6000 Blackwell).
# Compile is GPU-free; it runs nvcc from the CUDA 13.0 devel image against an external CUTLASS checkout.
#
# Usage:
#   kernels/e13/build.sh                # W4A4: A = MXFP4, B = MXFP4, tile 128x128x128
#   E13_A8=1 kernels/e13/build.sh       # W4A8: A = MXFP8 (E4M3), B = MXFP4 (kind::mxf8f6f4 path)
#   TILE_M=64 TILE_N=128 TILE_K=128 kernels/e13/build.sh   # alternate tile (K3); decode wants a small M tile
#
# Why sm_120a and not sm_120f: PTX ISA says mma.sync .kind::mxf4nvf4 / .kind::mxf4 are only available on
# sm_120a / sm_121a ("architecture-specific" features), whereas .kind::mxf8f6f4 is on sm_120f too.
# W4A4 therefore needs compute_120a/sm_120a; W4A8 would build on either.  See plan/RESEARCH.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

CUTLASS_DIR="${CUTLASS_DIR:-$HOME/dev/src/cutlass}"
IMAGE="${IMAGE:-dsv4-sm120:cuda-devel-https}"
TILE_M="${TILE_M:-128}"
TILE_N="${TILE_N:-128}"
TILE_K="${TILE_K:-128}"
OUT="kernels/e13/mxf4_grouped_gemm"
DEFS=(-DE13_TILE_M="$TILE_M" -DE13_TILE_N="$TILE_N" -DE13_TILE_K="$TILE_K")
if [[ "${E13_A8:-0}" == "1" ]]; then
  DEFS+=(-DE13_A8)
  OUT="${OUT}_a8"
fi
OUT="${OUT}_m${TILE_M}n${TILE_N}k${TILE_K}"

[[ -d "$CUTLASS_DIR/include/cutlass" ]] || { echo "CUTLASS not found at $CUTLASS_DIR (git clone -b v4.7.1 https://github.com/NVIDIA/cutlass \$CUTLASS_DIR)"; exit 1; }

echo "nvcc -> $OUT  (defs: ${DEFS[*]})"
docker run --rm \
  -v "$CUTLASS_DIR:/cutlass:ro" \
  -v "$PWD:/w" -w /w \
  --entrypoint nvcc "$IMAGE" \
  -std=c++17 -O3 --expt-relaxed-constexpr \
  -gencode arch=compute_120a,code=sm_120a \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 \
  "${DEFS[@]}" \
  -I/cutlass/include -I/cutlass/tools/util/include -I/cutlass/examples/common \
  kernels/e13/mxf4_grouped_gemm.cu -o "$OUT" "$@"
echo "built $OUT"

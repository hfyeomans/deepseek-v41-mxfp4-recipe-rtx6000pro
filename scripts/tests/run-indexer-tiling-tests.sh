#!/usr/bin/env bash
# Run the row-tiled dense fp4 indexer bit-equality test (recipe E34b) inside the recipe
# image with the overlay's python/sglang mounted read-only over the installed package.
# Needs a GPU for a few seconds (deep_gemm fp8_fp4_mqa_logits, fused top-k, block select);
# do not run while a server is capturing graphs on the same GPU.
# Exit 0 = all pass; 3 = docker, the image, the overlay checkout or a GPU is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
SGLANG_SITE=/opt/sglang/lib/python3.12/site-packages/sglang

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "indexer row tiling tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$OVERLAY_DIR/python/sglang/srt/layers/attention/deepseek_v4_backend.py" ]]; then
  echo "indexer row tiling tests: overlay not found at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

docker run --rm --gpus "${TEST_GPUS:-device=0}" --entrypoint python3 \
  -v "$OVERLAY_DIR/python/sglang:$SGLANG_SITE:ro" \
  -v "$ROOT/scripts/tests/test_indexer_row_tiling.py:/t/test_indexer_row_tiling.py:ro" \
  "$IMAGE" /t/test_indexer_row_tiling.py

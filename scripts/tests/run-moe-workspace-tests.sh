#!/usr/bin/env bash
# Run the persistent FlashInfer MoE workspace test (recipe E34d-v (i)) inside the recipe
# image with the overlay's python/sglang mounted read-only over the installed package.
# Needs a GPU for a few seconds (the SM120 cutlass_fused_moe MXFP4 kernel);
# do not run while a server is capturing graphs on the same GPU.
# Exit 0 = all pass; 3 = docker, the image, the overlay checkout or a GPU is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
SGLANG_SITE=/opt/sglang/lib/python3.12/site-packages/sglang

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "moe workspace tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$OVERLAY_DIR/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py" ]]; then
  echo "moe workspace tests: overlay not found at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

docker run --rm --gpus "${TEST_GPUS:-device=0}" --entrypoint python3 \
  -v "$OVERLAY_DIR/python/sglang:$SGLANG_SITE:ro" \
  -v "$ROOT/scripts/tests/test_moe_workspace.py:/t/test_moe_workspace.py:ro" \
  "$IMAGE" /t/test_moe_workspace.py

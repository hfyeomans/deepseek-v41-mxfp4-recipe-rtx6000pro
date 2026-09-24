#!/usr/bin/env bash
# Run the SM120 page-split buffer seal tests (review C5) inside the recipe image with the
# overlay's python/sglang mounted read-only over the installed package. No GPU is requested
# (the test stubs the capture query and the buffer registry); safe while a server is running.
# Exit 0 = all pass; 3 = docker, the image or the overlay checkout is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
SGLANG_SITE=/opt/sglang/lib/python3.12/site-packages/sglang

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "sm120 split seal tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$OVERLAY_DIR/python/sglang/kernels/ops/attention/flash_mla_sm120.py" ]]; then
  echo "sm120 split seal tests: overlay not found at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

docker run --rm --entrypoint python3 \
  -v "$OVERLAY_DIR/python/sglang:$SGLANG_SITE:ro" \
  -v "$ROOT/scripts/tests/test_sm120_split_seal.py:/t/test_sm120_split_seal.py:ro" \
  "$IMAGE" /t/test_sm120_split_seal.py

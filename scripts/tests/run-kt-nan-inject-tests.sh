#!/usr/bin/env bash
# Run the T3.13 / C10 NaN-rejection unit tests inside the recipe image with the overlay's
# python/sglang mounted read-only over the installed package: the kt_nan_inject.py state
# machine, the sampler's flag-then-sanitize step and the scheduler's per-row abort, all on
# CPU torch. No GPU is requested; safe while a server is running.
# Exit 0 = all pass; 3 = docker, the image or the overlay checkout is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
SGLANG_SITE=/opt/sglang/lib/python3.12/site-packages/sglang

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt nan-inject tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$OVERLAY_DIR/python/sglang/srt/layers/moe/kt_nan_inject.py" ]]; then
  echo "kt nan-inject tests: overlay not found at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

docker run --rm --entrypoint python3 \
  -v "$OVERLAY_DIR/python/sglang:$SGLANG_SITE:ro" \
  -v "$ROOT/scripts/tests/test_kt_nan_inject.py:/t/test_kt_nan_inject.py:ro" \
  "$IMAGE" /t/test_kt_nan_inject.py

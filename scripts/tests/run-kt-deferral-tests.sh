#!/usr/bin/env bash
# Run the kt-kernel deferral-adjacency tests (T3.16e) inside the recipe image, against the
# installed experts_base.py with every patch in config/kt-kernel-patches/ applied, plus the
# overlay's kt_ep_wrapper.py mounted read-only so the compact deferral selector can be
# loaded without importing sglang. No GPU is requested; safe while a server is running.
# Exit 0 = all tests pass; 3 = docker, the image, or the overlay is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
KT_PATH=/opt/sglang/lib/python3.12/site-packages/kt_kernel/experts_base.py
WRAPPER="$OVERLAY_DIR/python/sglang/srt/layers/moe/kt_ep_wrapper.py"

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt deferral tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$WRAPPER" ]]; then
  echo "kt deferral tests: overlay kt_ep_wrapper.py not found at $WRAPPER; skipped" >&2
  exit 3
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kt_kernel"
docker run --rm --entrypoint cat "$IMAGE" "$KT_PATH" > "$work/kt_kernel/experts_base.py"
# Python patches only; 0003+ are C++ (compiled into the image's extension, T3.39).
for p in "$ROOT"/config/kt-kernel-patches/000[12]-*.patch; do
  (cd "$work" && patch -p1 -N -s -r - < "$p" >/dev/null) || true
done
python3 -m py_compile "$work/kt_kernel/experts_base.py"

docker run --rm --entrypoint python3 \
  -v "$work/kt_kernel/experts_base.py:$KT_PATH:ro" \
  -v "$WRAPPER:/overlay/kt_ep_wrapper.py:ro" \
  -v "$ROOT/scripts/tests/test_kt_deferral_adjacency.py:/t/test_kt_deferral_adjacency.py:ro" \
  "$IMAGE" /t/test_kt_deferral_adjacency.py

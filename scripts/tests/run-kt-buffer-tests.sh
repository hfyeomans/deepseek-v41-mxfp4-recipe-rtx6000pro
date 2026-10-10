#!/usr/bin/env bash
# Run the kt-kernel buffer-lifetime tests inside the recipe image, against the
# installed experts_base.py with every patch in config/kt-kernel-patches/ applied
# (the image already carries the ones it was built with; `patch --forward` skips
# those and applies the rest). No GPU is requested, so this is safe while a server
# is running. Exit 0 = all tests pass; 3 = docker or the image is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
KT_PATH=/opt/sglang/lib/python3.12/site-packages/kt_kernel/experts_base.py

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt buffer tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kt_kernel"
docker run --rm --entrypoint cat "$IMAGE" "$KT_PATH" > "$work/kt_kernel/experts_base.py"
# Python patches only; 0003+ are C++ (compiled into the image's extension, T3.39).
for p in "$ROOT"/config/kt-kernel-patches/000[12]-*.patch; do
  # --forward + -N: already-applied patches (baked into the image) are skipped silently.
  (cd "$work" && patch -p1 -N -s -r - < "$p" >/dev/null) || true
done
python3 -m py_compile "$work/kt_kernel/experts_base.py"

docker run --rm --entrypoint python3 \
  -v "$work/kt_kernel/experts_base.py:$KT_PATH:ro" \
  -v "$ROOT/scripts/tests/test_kt_buffer_lifetime.py:/t/test_kt_buffer_lifetime.py:ro" \
  "$IMAGE" /t/test_kt_buffer_lifetime.py

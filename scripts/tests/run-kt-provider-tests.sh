#!/usr/bin/env bash
# Run the CpuExpertProvider seam conformance tests (T3.19b) inside the recipe image:
# the overlay's kt_contract.py + kt_provider.py through a stub `sglang` package, the
# installed experts_base.py with every recipe patch applied, and the deferral test's
# recorders (test_kt_deferral_adjacency.py) as the fake C++ side. No GPU is requested;
# safe while a server is running. Exit 0 = all pass; 3 = docker/image/overlay unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
KT_PATH=/opt/sglang/lib/python3.12/site-packages/kt_kernel/experts_base.py
MOE="$OVERLAY_DIR/python/sglang/srt/layers/moe"

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt provider tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
for f in kt_contract.py kt_provider.py kt_ep_wrapper.py; do
  if [[ ! -f "$MOE/$f" ]]; then
    echo "kt provider tests: overlay $MOE/$f not found (set OVERLAY_DIR); skipped" >&2
    exit 3
  fi
done

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/kt_kernel" "$work/stub/sglang/srt/layers/moe"
docker run --rm --entrypoint cat "$IMAGE" "$KT_PATH" > "$work/kt_kernel/experts_base.py"
for p in "$ROOT"/config/kt-kernel-patches/*.patch; do
  (cd "$work" && patch -p1 -N -s -r - < "$p" >/dev/null) || true
done
python3 -m py_compile "$work/kt_kernel/experts_base.py"
touch "$work/stub/sglang/__init__.py" "$work/stub/sglang/srt/__init__.py" \
  "$work/stub/sglang/srt/layers/__init__.py" "$work/stub/sglang/srt/layers/moe/__init__.py"
cp "$MOE/kt_contract.py" "$MOE/kt_provider.py" "$work/stub/sglang/srt/layers/moe/"

docker run --rm --entrypoint python3 \
  -e PYTHONPATH=/stub \
  -v "$work/stub:/stub:ro" \
  -v "$work/kt_kernel/experts_base.py:$KT_PATH:ro" \
  -v "$MOE/kt_ep_wrapper.py:/overlay/kt_ep_wrapper.py:ro" \
  -v "$ROOT/scripts/tests/test_kt_deferral_adjacency.py:/t/test_kt_deferral_adjacency.py:ro" \
  -v "$ROOT/scripts/tests/test_kt_provider.py:/t/test_kt_provider.py:ro" \
  "$IMAGE" /t/test_kt_provider.py

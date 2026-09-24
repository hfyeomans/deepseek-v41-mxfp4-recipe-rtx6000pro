#!/usr/bin/env bash
# Run the SGLang -> kt-kernel boundary-contract tests (T3.16d) inside the recipe image,
# against the overlay's kt_contract.py (torch-only module, CPU tensors, no GPU requested,
# so this is safe while a server is running). Exit 0 = all pass; 3 = docker, the image
# or the overlay checkout is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
CONTRACT="$OVERLAY_DIR/python/sglang/srt/layers/moe/kt_contract.py"

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt contract tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$CONTRACT" ]]; then
  echo "kt contract tests: $CONTRACT not found (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

# Import kt_contract through a stub package so the test does not pull in the rest of
# sglang (whose import needs the model config and CUDA).
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/sglang/srt/layers/moe"
touch "$work/sglang/__init__.py" "$work/sglang/srt/__init__.py" \
  "$work/sglang/srt/layers/__init__.py" "$work/sglang/srt/layers/moe/__init__.py"
cp "$CONTRACT" "$work/sglang/srt/layers/moe/kt_contract.py"

docker run --rm --entrypoint python3 \
  -e PYTHONPATH=/stub \
  -v "$work:/stub:ro" \
  -v "$ROOT/scripts/tests/test_kt_contract.py:/t/test_kt_contract.py:ro" \
  "$IMAGE" /t/test_kt_contract.py

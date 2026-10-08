#!/usr/bin/env bash
# Run the streamed-prefill state-machine tests (T3.19d) inside the recipe image, against
# the overlay's kt_prefill.py. PrefillPipeline is pure Python (the module imports torch
# but the pipeline never touches a device), so this is safe while a server is running.
# Exit 0 = all pass; 3 = docker, the image or the overlay checkout is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
MOE="$OVERLAY_DIR/python/sglang/srt/layers/moe"

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt prefill tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$MOE/kt_prefill.py" || ! -f "$MOE/kt_contract.py" ]]; then
  echo "kt prefill tests: $MOE/kt_prefill.py or kt_contract.py not found (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

# Stub package: kt_prefill imports only kt_contract from sglang, so the rest of sglang
# (model config, CUDA) stays out of the import graph.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/sglang/srt/layers/moe"
touch "$work/sglang/__init__.py" "$work/sglang/srt/__init__.py" \
  "$work/sglang/srt/layers/__init__.py" "$work/sglang/srt/layers/moe/__init__.py"
cp "$MOE/kt_contract.py" "$MOE/kt_prefill.py" "$work/sglang/srt/layers/moe/"

docker run --rm --entrypoint python3 \
  -e PYTHONPATH=/stub \
  -v "$work:/stub:ro" \
  -v "$ROOT/scripts/tests/test_kt_prefill.py:/t/test_kt_prefill.py:ro" \
  "$IMAGE" /t/test_kt_prefill.py

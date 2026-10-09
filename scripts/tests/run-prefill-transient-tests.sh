#!/usr/bin/env bash
# Run the recipe E34d-v (ii) prefill-transient bit-equality tests inside the recipe image
# with the overlay's python/sglang mounted read-only over the installed package:
#   test_indexer_k_gather.py   fp4 indexer key-row gather by page/offset == column grids
#   test_hc_combine_input.py   hc_combine on bf16 input == on its fp32 copy
# Needs a GPU for a few seconds; do not run while a server is capturing graphs on the
# same GPU.
# Exit 0 = all pass; 3 = docker, the image, the overlay checkout or a GPU is unavailable.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
SGLANG_SITE=/opt/sglang/lib/python3.12/site-packages/sglang

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "prefill transient tests: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$OVERLAY_DIR/python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py" ]]; then
  echo "prefill transient tests: overlay not found at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

rc=0
for t in test_indexer_k_gather test_hc_combine_input; do
  docker run --rm --gpus "${TEST_GPUS:-device=0}" --entrypoint python3 \
    -v "$OVERLAY_DIR/python/sglang:$SGLANG_SITE:ro" \
    -v "$ROOT/scripts/tests/$t.py:/t/$t.py:ro" \
    "$IMAGE" /t/$t.py || rc=$?
done
exit $rc

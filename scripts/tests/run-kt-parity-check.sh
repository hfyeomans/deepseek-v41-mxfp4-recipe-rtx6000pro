#!/usr/bin/env bash
# Re-check the recorded T3.19b parity evidence (evidence/t319/b/) inside the recipe image:
# every standalone-kt replay output must still equal the recorded serving-path kt_output
# bit-for-bit. CPU-only (torch.equal on bf16 tensors), no GPU requested, safe while a
# server is running. Exit 0 = all pairs bit-equal; 1 = mismatch; 3 = docker/image or
# evidence unavailable (ci.sh treats 3 as skipped). Producing the evidence needs the
# GPUs and the model: see the docstring of scripts/tests/kt-parity-replay.py (run mode).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-dsv41-kt:recipe}"
EVID="$ROOT/evidence/t319/b"

if ! command -v docker >/dev/null || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "kt parity check: docker or image $IMAGE unavailable; skipped" >&2
  exit 3
fi
if [[ ! -f "$EVID/parity/parity.json" ]]; then
  echo "kt parity check: $EVID/parity/parity.json absent; skipped" >&2
  exit 3
fi

docker run --rm --entrypoint python3 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$EVID:/evid:ro" \
  -v "$ROOT/scripts/tests/kt-parity-replay.py:/t/kt-parity-replay.py:ro" \
  "$IMAGE" /t/kt-parity-replay.py check --dumps /evid/dumps --out /evid/parity

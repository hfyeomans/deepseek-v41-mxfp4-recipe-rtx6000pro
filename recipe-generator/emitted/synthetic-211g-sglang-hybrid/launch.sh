#!/usr/bin/env bash
# Generated launcher (sglang-hybrid engine shape). Set the pins first:
#   HF_MODEL_DIR=<models dir> SNAPSHOT=<checkpoint revision>
#   OVERLAY_DIR=<checkout with the patch series applied>  (see engine-notes.md)
#   IMAGE=dsv41-kt:recipe (built by config/Dockerfile.dsv41-kt in the parent recipe)
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PROFILE:-qualified}"
# The parent recipe's typed registry refuses unknown/mistyped/removed/unsafe knobs
# before launch. Point this at its scripts/serve_config.py to get that protection:
if [[ -n "${SERVE_CONFIG:-}" ]]; then
  eval "$(python3 "$SERVE_CONFIG" shell)"
fi
echo "resolved profile=$PROFILE - see config/profiles/$PROFILE.env"
echo "launch: docker run (image from PROVENANCE.md) with the overlay bind-mounted"
exit 1  # replace with the parent recipe's serve-hybrid.sh invocation for this host

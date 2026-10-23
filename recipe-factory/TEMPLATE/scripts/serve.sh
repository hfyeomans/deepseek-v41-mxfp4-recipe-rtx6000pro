#!/usr/bin/env bash
# Serve launcher template for a new recipe. Pattern lifted from the reference recipe's
# scripts/serve-hybrid.sh: the launcher carries NO ${VAR:-default} lines of its own — it evals the
# typed registry (serve_config.py), which resolves, validates and guards every knob, and prints the
# one resolved: line that names the configuration. DRY_RUN=1 / --print probe without launching.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

eval "$(python3 "$ROOT/scripts/serve_config.py" shell)"   # KEY='value' lines, rc 2 on any refusal

echo "$RESOLVED_LINE" >&2
if [[ "${DRY_RUN:-0}" == "1" || "${1:-}" == "--print" ]]; then
    # --print continues to echo the launch command; DRY_RUN stops here.
    [[ "${DRY_RUN:-0}" == "1" ]] && exit 0
fi

# --- preflight (fail closed: pins, shard bytes, GPU class/count, fit gates) ----------------
"$ROOT/scripts/preflight.sh" || { echo "preflight failed; refusing to launch" >&2; exit 1; }

# --- launch --------------------------------------------------------------------------------
# Adapt to your engine/runtime. The reference recipe runs a pinned container with the overlay
# bind-mounted; a bare-metal engine build works the same way. Every flag comes from the registry
# (SERVER_ARG sinks), every env from the registry's CONTAINER_ENV list — never hand-written.
# The manifest (evidence_manifest.py) is written by the launcher before the engine starts and
# stamped by every evidence script afterwards.

echo "launching $SERVED_MODEL_NAME on $BIND_ADDRESS:$PORT (profile=$PROFILE)" >&2
# docker run ...  (the real command, assembled from the resolved values)
exit 0

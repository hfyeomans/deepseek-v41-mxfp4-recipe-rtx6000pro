#!/usr/bin/env bash
# Stop whichever serve is up: the qualified 2-GPU hybrid (scripts/serve-hybrid.sh, container
# named by scripts/serve_config.py CONTAINER_NAME, default dsv41-kt) or the legacy all-GPU
# chain (scripts/serve.sh, CONTAINER_NAME from scripts/load-env.sh, default
# dsv41-flash-rtxpro6000). Before 2026-10-11 only the legacy name was removed and the hybrid
# kept both GPUs while this script printed "no container" (T3.38 / R2).
#
# `docker rm -f` is retried once: a server killed mid-step can leave a zombie that makes the
# first removal fail ("PID ... is zombie and can not be killed", seen twice on 2026-10-11).
# Exit 0 when nothing of either name remains, 1 when a removal failed twice.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

names=()
# Hybrid: the name this environment resolves to; the registry default when the shell's env
# would not resolve (a probe of a running serve must not fail on an unrelated knob).
hybrid="$(python3 "$ROOT/scripts/serve_config.py" endpoint 2>/dev/null | sed -n 's/^CONTAINER=//p')"
[[ -n "$hybrid" ]] || hybrid="$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import serve_config as sc; print(sc.BY_NAME["CONTAINER_NAME"].default)' "$ROOT/scripts")"
names+=("$hybrid")
# Legacy: load-env.sh in a subshell so its exports do not leak into this one.
legacy="$(ENV_FILE="${ENV_FILE:-$ROOT/.env}" bash -c 'source "$1" >/dev/null 2>&1 && printf %s "$CONTAINER_NAME"' _ "$ROOT/scripts/load-env.sh")"
[[ -n "$legacy" && "$legacy" != "$hybrid" ]] && names+=("$legacy")

rc=0; removed=0
existing="$(docker ps -a --format '{{.Names}}')"
for name in "${names[@]}"; do
  grep -qx "$name" <<<"$existing" || continue
  if docker rm -f "$name" >/dev/null 2>&1 || { sleep 2; docker rm -f "$name" >/dev/null; }; then
    echo "removed $name"; removed=$((removed + 1))
  else
    echo "FAIL could not remove $name (twice); docker ps -a --filter name=$name" >&2; rc=1
  fi
done
[[ $removed -eq 0 && $rc -eq 0 ]] && echo "no container named ${names[*]}"
exit $rc

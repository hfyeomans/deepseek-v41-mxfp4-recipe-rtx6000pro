#!/usr/bin/env bash
# Export the SGLang overlay branch as a patch series into patches/sglang/ (recipe T3.24c).
#
# The overlay (OVERLAY_DIR, default ~/dev/src/sglang-dsv41-overlay) is a local-only clone of
# sgl-project/sglang whose branch dsv41-kt-hybrid carries every change this recipe serves with;
# scripts/serve-hybrid.sh bind-mounts its python/sglang over the image's copy. This script writes
# the branch as `git format-patch` files plus patches/sglang/SERIES.env (base, head, tree, count) so
# the recipe repository alone reproduces the overlay. scripts/tests/check-sglang-patch-series.sh
# (run by ci.sh) re-applies the series on a clean base checkout and compares tree hashes.
#
# Usage: bash scripts/export-sglang-series.sh            # exports OVERLAY_BRANCH (default dsv41-kt-hybrid)
#        OVERLAY_DIR=... OVERLAY_BRANCH=... bash scripts/export-sglang-series.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
OVERLAY_BRANCH="${OVERLAY_BRANCH:-dsv41-kt-hybrid}"
BASE="${OVERLAY_BASE:-da64c5cbb8cf6bfd39be19da43573fdfd484c43a}"   # upstream commit the dev-dsv41 image was built from
OUT="$ROOT/patches/sglang"

if [[ ! -d "$OVERLAY_DIR/.git" ]]; then
  echo "export-sglang-series: $OVERLAY_DIR is not a git checkout (set OVERLAY_DIR)" >&2; exit 1
fi
if [[ -n "$(git -C "$OVERLAY_DIR" status --porcelain)" ]]; then
  echo "export-sglang-series: overlay working tree is dirty; commit or stash first" >&2; exit 1
fi
head="$(git -C "$OVERLAY_DIR" rev-parse "$OVERLAY_BRANCH")"
tree="$(git -C "$OVERLAY_DIR" rev-parse "$OVERLAY_BRANCH^{tree}")"
if ! git -C "$OVERLAY_DIR" merge-base --is-ancestor "$BASE" "$head"; then
  echo "export-sglang-series: $BASE is not an ancestor of $OVERLAY_BRANCH" >&2; exit 1
fi

mkdir -p "$OUT"
rm -f "$OUT"/*.patch
# --no-signature keeps the files identical across git versions; the "From <sha>" header line keeps
# the overlay commit id as the index key used by patches/sglang/README.md.
git -C "$OVERLAY_DIR" format-patch --no-signature -o "$OUT" "$BASE..$head" >/dev/null
count="$(find "$OUT" -maxdepth 1 -name '*.patch' | wc -l)"
cat >"$OUT/SERIES.env" <<EOF
# Written by scripts/export-sglang-series.sh; read by scripts/tests/check-sglang-patch-series.sh.
OVERLAY_BASE=$BASE
OVERLAY_HEAD=$head
OVERLAY_TREE=$tree
PATCH_COUNT=$count
EOF
# The ci.sh identifier scan refuses home-directory paths anywhere in the repository.
if grep -lE '/(home|media)/[A-Za-z0-9._-]+/' "$OUT"/*.patch; then
  echo "export-sglang-series: the files above contain home-directory paths; redact the overlay commits first" >&2; exit 1
fi
echo "exported $count patches ($BASE..${head:0:8}, tree ${tree:0:8}) to patches/sglang/"

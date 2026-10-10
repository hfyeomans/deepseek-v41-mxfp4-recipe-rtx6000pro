#!/usr/bin/env bash
# T3.24c: the patch series under patches/sglang/ must rebuild the overlay exactly.
# Applies every patch with `git am` onto a clean checkout of the recorded base commit in a temp
# clone (object store shared with the overlay, no network) and compares the resulting tree hash
# with patches/sglang/SERIES.env; then checks that the overlay branch itself still has that tree,
# so an overlay commit that was not re-exported fails CI with the re-export command.
# Exit 0 = series reproduces the overlay; 3 = no git or no overlay checkout holding the base commit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"
OVERLAY_BRANCH="${OVERLAY_BRANCH:-dsv41-kt-hybrid}"
SERIES="$ROOT/patches/sglang"

[[ -f "$SERIES/SERIES.env" ]] || { echo "FAIL $SERIES/SERIES.env missing (run scripts/export-sglang-series.sh)"; exit 1; }
OVERLAY_BASE=""; OVERLAY_HEAD=""; OVERLAY_TREE=""; PATCH_COUNT=""
# shellcheck disable=SC1091
source "$SERIES/SERIES.env"
[[ -n "$OVERLAY_BASE" && -n "$OVERLAY_HEAD" && -n "$OVERLAY_TREE" && -n "$PATCH_COUNT" ]] || { echo "FAIL SERIES.env incomplete"; exit 1; }

mapfile -t patches < <(find "$SERIES" -maxdepth 1 -name '*.patch' | sort)
if [[ ${#patches[@]} -ne $PATCH_COUNT ]]; then
  echo "FAIL patches/sglang/ has ${#patches[@]} patches, SERIES.env says $PATCH_COUNT"; exit 1
fi

if ! command -v git >/dev/null || [[ ! -d "$OVERLAY_DIR/.git" ]] \
   || ! git -C "$OVERLAY_DIR" cat-file -e "$OVERLAY_BASE^{commit}" 2>/dev/null; then
  echo "sglang patch series check: no git checkout holding base $OVERLAY_BASE at $OVERLAY_DIR (set OVERLAY_DIR); skipped" >&2
  exit 3
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone -q --shared --no-checkout "$OVERLAY_DIR" "$work/src"
git -C "$work/src" checkout -q --detach "$OVERLAY_BASE"
if ! git -C "$work/src" -c user.name=ci -c user.email=ci@localhost am -q "${patches[@]}"; then
  echo "FAIL patches/sglang/ does not apply cleanly onto $OVERLAY_BASE"; exit 1
fi
got="$(git -C "$work/src" rev-parse 'HEAD^{tree}')"
if [[ "$got" != "$OVERLAY_TREE" ]]; then
  echo "FAIL applied series tree $got != SERIES.env tree $OVERLAY_TREE"; exit 1
fi

# Drift: the overlay branch moved without a re-export (the served code and the recorded series differ).
if git -C "$OVERLAY_DIR" rev-parse -q --verify "$OVERLAY_BRANCH^{commit}" >/dev/null; then
  branch_tree="$(git -C "$OVERLAY_DIR" rev-parse "$OVERLAY_BRANCH^{tree}")"
  if [[ "$branch_tree" != "$OVERLAY_TREE" ]]; then
    echo "FAIL overlay $OVERLAY_BRANCH tree $branch_tree != exported series tree $OVERLAY_TREE; re-export with: bash scripts/export-sglang-series.sh"; exit 1
  fi
fi
echo "sglang patch series: $PATCH_COUNT patches rebuild tree ${OVERLAY_TREE:0:12} from base ${OVERLAY_BASE:0:12}"

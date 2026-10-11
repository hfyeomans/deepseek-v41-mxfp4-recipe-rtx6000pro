#!/usr/bin/env bash
# Download the pinned DeepSeek-V4.1-Flash snapshot (475.24 GiB) into MODEL_DIR
# and write RECIPE_PIN.txt. Refuses to start without ~500 GiB free.
# Pass --language-only to skip vision.* shards is NOT possible: the checkpoint
# interleaves components across all 48 shards, so the whole thing is needed.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/load-env.sh
source "$ROOT/scripts/load-env.sh"

command -v hf >/dev/null || { echo "FAIL missing hf CLI (pip install -U huggingface_hub)"; exit 1; }
[[ -n "$MODEL_DIR" ]] || { echo "FAIL MODEL_DIR not set"; exit 1; }
mkdir -p "$MODEL_DIR"
free_gib="$(df -BG --output=avail "$MODEL_DIR" | tail -1 | tr -dc '0-9')"
have_gib="$(du -sBG "$MODEL_DIR" 2>/dev/null | cut -f1 | tr -dc '0-9')"
need_gib=$(( 500 - ${have_gib:-0} ))   # a resumed / re-pinned download only needs what is still missing
if (( need_gib > 0 && free_gib < need_gib )); then
  echo "FAIL only ${free_gib} GiB free at MODEL_DIR (${have_gib:-0} GiB already there); need ~${need_gib} GiB more for the 475.24 GiB checkpoint"; exit 1
fi

echo "downloading $MODEL_PIN to $MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" \
  hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
printf '%s\n' "$MODEL_PIN" >"$MODEL_DIR/RECIPE_PIN.txt"
echo "wrote $MODEL_DIR/RECIPE_PIN.txt"
echo "next: scripts/preflight.sh"

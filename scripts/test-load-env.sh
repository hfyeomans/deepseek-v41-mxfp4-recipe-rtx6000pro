#!/usr/bin/env bash
# load-env.sh must accept KEY=VALUE, ignore comments, and reject anything that
# could execute. serve.sh --print must reflect the profile without Docker.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/good.env" <<'EOF'
# comment
PORT=9999
TP_SIZE=4
ENGRAM_PLACEMENT=dram
EOF
out="$(ENV_FILE="$tmp/good.env" bash -c 'source "$1"; echo "$PORT $TP_SIZE $MODEL_PIN $(gpu_count)"' _ "$ROOT/scripts/load-env.sh")"
[[ "$out" == "9999 4 deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277 4" ]] || { echo "FAIL good env: $out"; exit 1; }

# shellcheck disable=SC2016  # these strings are intentionally literal; they must NOT expand
for bad in 'PORT=$(rm -rf /)' 'PORT=1; echo pwned' 'echo hi' 'PORT=`id`'; do
  printf '%s\n' "$bad" >"$tmp/bad.env"
  if ENV_FILE="$tmp/bad.env" bash -c 'source "$1"' _ "$ROOT/scripts/load-env.sh" 2>/dev/null; then
    echo "FAIL accepted dangerous line: $bad"; exit 1
  fi
done

cat >"$tmp/print.env" <<'EOF'
MODEL_DIR=/srv/models/dsv41
HF_CACHE_DIR=/srv/hf
GPU_DEVICES=0,1,2,3
TP_SIZE=4
PROFILE=4gpu-sglang
ENGRAM_PLACEMENT=dram
PORT=8010
EOF
cmd="$(ENV_FILE="$tmp/print.env" "$ROOT/scripts/serve.sh" --print)"
for needle in 'SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1' '--tp 4' '--speculative-algorithm DSPARK' '--reasoning-parser deepseek-v41' \
              '/dev/hugepages:/dev/hugepages' '127.0.0.1:8010:30000' 'CUDA_VISIBLE_DEVICES=0,1,2,3' '--context-length 262144'; do
  grep -qF -- "$needle" <<<"$cmd" || { echo "FAIL serve.sh --print missing: $needle"; echo "$cmd"; exit 1; }
done

# 2-GPU attempt is refused without ALLOW_UNQUALIFIED=1
printf 'MODEL_DIR=/srv/m\nGPU_DEVICES=0,1\nTP_SIZE=2\nPROFILE=2gpu-attempt\n' >"$tmp/two.env"
if ENV_FILE="$tmp/two.env" "$ROOT/scripts/serve.sh" --print >/dev/null 2>&1; then
  echo "FAIL 2gpu-attempt accepted without ALLOW_UNQUALIFIED"; exit 1
fi
printf 'ALLOW_UNQUALIFIED=1\n' >>"$tmp/two.env"
ENV_FILE="$tmp/two.env" "$ROOT/scripts/serve.sh" --print | grep -qF -- '--tp 2' || { echo "FAIL 2gpu-attempt with ALLOW_UNQUALIFIED"; exit 1; }

# gpu placement drops the Engram env and hugepage mount
printf 'MODEL_DIR=/srv/m\nGPU_DEVICES=0,1,2,3\nTP_SIZE=4\nENGRAM_PLACEMENT=gpu\n' >"$tmp/gpu.env"
cmd="$(ENV_FILE="$tmp/gpu.env" "$ROOT/scripts/serve.sh" --print)"
if grep -qF 'ENGRAM_HOST_TABLE' <<<"$cmd" || grep -qF '/dev/hugepages' <<<"$cmd"; then
  echo "FAIL gpu placement still references host table"; exit 1
fi
echo "ok load-env and serve --print"

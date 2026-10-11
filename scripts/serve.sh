#!/usr/bin/env bash
# Launch SGLang (dev-dsv41 preview) for DeepSeek-V4.1-Flash in Docker.
#
# PROFILE=4gpu-sglang  Option A from docs/FEASIBILITY.md. UNQUALIFIED until
#                      experiment E9 runs on four RTX PRO 6000. Flags follow
#                      the SGLang DeepSeek-V4.1 cookbook and 0xSero's 4-GPU
#                      recipe; backends are left to auto-select as the cookbook
#                      instructs.
# PROFILE=2gpu-attempt Experiment E5 only. Requires ALLOW_UNQUALIFIED=1 and is
#                      expected to fail during weight load.
#
# Engram placement (ENGRAM_PLACEMENT):
#   dram  SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 with the full FP8 table in
#         hugepage-backed host memory (this host: 246.8 GiB, enough).
#   gpu   tables resident in VRAM (needs headroom this host lacks on 4 GPUs).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/load-env.sh
source "$ROOT/scripts/load-env.sh"

print_only=0
for arg in "$@"; do
  case "$arg" in
    --print) print_only=1 ;;
    --profile=*) PROFILE="${arg#*=}" ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

case "$PROFILE" in
  4gpu-sglang) [[ "$TP_SIZE" -eq 4 ]] || { echo "FAIL profile 4gpu-sglang needs TP_SIZE=4"; exit 1; } ;;
  2gpu-attempt)
    [[ "$ALLOW_UNQUALIFIED" == "1" ]] || { echo "FAIL 2gpu-attempt is experiment E5; set ALLOW_UNQUALIFIED=1"; exit 1; }
    [[ "$TP_SIZE" -eq 2 ]] || { echo "FAIL profile 2gpu-attempt needs TP_SIZE=2"; exit 1; } ;;
  *) echo "FAIL unknown PROFILE=$PROFILE"; exit 1 ;;
esac

if [[ $print_only -eq 0 ]]; then
  "$ROOT/scripts/preflight.sh"
fi

engram_opts=()
if [[ "$ENGRAM_PLACEMENT" == "dram" ]]; then
  engram_opts+=(-e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 -e "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=${ENGRAM_HOST_TABLE_LAYOUT}"
                -v /dev/hugepages:/dev/hugepages)
fi

server_args=(
  python3 -m sglang.launch_server
  --model-path /model
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 30000
  --tp "$TP_SIZE"
  --trust-remote-code
  --context-length "$MAX_MODEL_LEN"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --reasoning-parser deepseek-v41
  --tool-call-parser deepseekv41
  --speculative-algorithm DSPARK
  --speculative-dspark-block-size "$DSPARK_K"
)

docker_cmd=(
  docker run -d --name "$CONTAINER_NAME"
  --gpus "\"device=${GPU_DEVICES}\""
  --ipc=host --shm-size 32g
  --ulimit memlock=-1 --ulimit stack=67108864
  -p "${BIND_ADDRESS}:${PORT}:30000"
  -v "${MODEL_DIR}:/model:ro"
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface"
  -e "NCCL_DEBUG=${NCCL_DEBUG}"
  -e "CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((TP_SIZE - 1)))"
  "${engram_opts[@]}"
  --restart unless-stopped
  "$RUNTIME_IMAGE"
  "${server_args[@]}"
)

if [[ $print_only -eq 1 ]]; then
  printf '%s ' "${docker_cmd[@]}"; echo
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "FAIL container $CONTAINER_NAME already exists; run scripts/stop.sh first"; exit 1
fi
echo "profile $PROFILE is UNQUALIFIED on this host (see README verdict); launching for experiment purposes"
"${docker_cmd[@]}"
echo "started $CONTAINER_NAME; follow with: docker logs -f $CONTAINER_NAME"
echo "then: scripts/wait-ready.py --url http://${BIND_ADDRESS}:${PORT} --container ${CONTAINER_NAME} && scripts/verify.py --url http://${BIND_ADDRESS}:${PORT}"

#!/usr/bin/env bash
# Launch the 2-GPU three-tier hybrid: DeepSeek-V4.1-Flash on two RTX PRO 6000
# (SM120) with kt-kernel CPU experts and the Engram tables read from the
# checkpoint shard on NVMe (E6 stage 2, first served in E19).
#
# This is the exact configuration that serves (attempt 5/6 in
# evidence/t36d-first-serve/); every value can be overridden in the environment.
# It does not go through preflight.sh, whose feasibility gate is written for the
# all-GPU profiles and would refuse two cards.
#
#   scripts/serve-hybrid.sh            # launch
#   scripts/serve-hybrid.sh --print    # show the docker command only
#   docker rm -f dsv41-kt              # stop
set -euo pipefail

print_only=0
for arg in "$@"; do
  case "$arg" in
    --print) print_only=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Every knob (type, default, evidence, where it goes) is declared once in
# scripts/serve_config.py (T3.19e). It validates the environment, refuses the
# combinations the evidence condemned (T3.16b / E34d / T3.29; each refusal names its
# override), rejects unknown KT_* / DSPARK* / NGRAM* variables and removed knobs, and emits the
# resolved values this script evals. `python3 scripts/serve_config.py markdown` prints
# the knob table (docs/CONFIG.md).
cfg="$(python3 "$(dirname -- "${BASH_SOURCE[0]}")/serve_config.py" shell)" || exit $?
# Assigned by the eval (listed for shellcheck): the knobs this script reads directly plus
# RESOLVED_LINE, CONTAINER_ENV (fixed + forwarded -e pairs) and DIAG_ENV (set-only diagnostics).
declare SNAPSHOT SERVED_MODEL_NAME KT_NUM_GPU_EXPERTS KT_CPUINFER KT_MAX_DEFERRED DSPARK DSPARK_K NGRAM NGRAM_DRAFT_TOKENS NGRAM_BREADTH \
  MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE CONTEXT_LENGTH MAX_RUNNING_REQUESTS CUDA_GRAPH_MAX_BS \
  CUDA_GRAPH_BS MAX_TOTAL_TOKENS EXTRA_SERVER_ARGS EXTRA_DOCKER_ENV EXTRA_DOCKER_ARGS DRY_RUN \
  CONTAINER_NAME BIND_ADDRESS PORT HF_MODEL_DIR OVERLAY_DIR IMAGE MANIFEST_DIR RESOLVED_LINE CONTAINER_ENV DIAG_ENV
eval "$cfg"

snapshot_dir="/hfm/snapshots/${SNAPSHOT}"

server_args=(
  -m sglang.launch_server
  --model-path "$snapshot_dir"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0 --port 30000
  --tp 2
  --trust-remote-code
  --attention-backend dsv4
  --moe-runner-backend flashinfer_mxfp4
  --enable-deepseek-v4-fp4-indexer
  --kt-weight-path "$snapshot_dir"
  --kt-method MXFP4
  --kt-num-gpu-experts "$KT_NUM_GPU_EXPERTS"
  --kt-cpuinfer "$KT_CPUINFER"
  --kt-max-deferred-experts-per-token "$KT_MAX_DEFERRED"
  --kt-threadpool-count 1
  --cuda-graph-backend-decode breakable
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --context-length "$CONTEXT_LENGTH"
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
  --reasoning-parser deepseek-v41
  --tool-call-parser deepseekv41
)
if [[ "$DSPARK" == "1" ]]; then
  server_args+=(--speculative-algorithm DSPARK --speculative-dspark-block-size "$DSPARK_K")
fi
if [[ "$NGRAM" == "1" ]]; then
  server_args+=(--speculative-algorithm NGRAM --speculative-num-draft-tokens "$NGRAM_DRAFT_TOKENS"
                --speculative-ngram-max-bfs-breadth "$NGRAM_BREADTH")
fi
if [[ -n "$CUDA_GRAPH_BS" ]]; then
  # shellcheck disable=SC2206  # intentional word splitting of the batch-size list
  server_args+=(--cuda-graph-bs-decode $CUDA_GRAPH_BS)
fi
if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  server_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi

# Resolved configuration, one line, so a launch log names every knob that decides the
# result (the container log prints SGLang's own server_args; kt logs its capture sizes
# as 'KT_BUF capture sizes registered').
echo "$RESOLVED_LINE"
# DRY_RUN=1: print the resolved line (after every guard) and stop before any docker
# call. For probing the guards and for the config tests; added after a guard probe on
# 2026-10-08 launched a real server.
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi
# shellcheck disable=SC2206  # intentional word splitting of the override string
server_args+=($EXTRA_SERVER_ARGS)

# Container env from the registry (fixed Engram/indexer values, forwarded knobs), then
# diagnostics the caller set (KT_TIMELINE*, SGLANG_BCG_DEBUG_*, ...), then EXTRA_DOCKER_ENV.
container_env_flags=()
for kv in $CONTAINER_ENV; do
  container_env_flags+=(-e "$kv")
done
extra_env_flags=()
for kv in $DIAG_ENV $EXTRA_DOCKER_ENV; do
  extra_env_flags+=(-e "$kv")
done
# shellcheck disable=SC2206  # intentional word splitting of the override string
extra_docker_args=($EXTRA_DOCKER_ARGS)

docker_cmd=(
  docker run -d --name "$CONTAINER_NAME"
  --gpus all
  --ipc=host --shm-size 32g
  --ulimit memlock=-1 --ulimit stack=67108864
  --security-opt label=disable
  --entrypoint python3
  -p "${BIND_ADDRESS}:${PORT}:30000"
  -v "${HF_MODEL_DIR}:/hfm:ro"
  -v "${OVERLAY_DIR}/python/sglang:/sgl-workspace/sglang/python/sglang:ro"
  "${container_env_flags[@]}"
  "${extra_env_flags[@]}"
  "${extra_docker_args[@]}"
  "$IMAGE"
  "${server_args[@]}"
)

if [[ $print_only -eq 1 ]]; then
  printf '%q ' "${docker_cmd[@]}"; echo
  exit 0
fi

for path in "$HF_MODEL_DIR/snapshots/$SNAPSHOT" "$OVERLAY_DIR/python/sglang"; do
  [[ -d "$path" ]] || { echo "FAIL missing $path"; exit 1; }
done
if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "FAIL container $CONTAINER_NAME already exists; docker rm -f $CONTAINER_NAME first"; exit 1
fi
if ss -ltn "( sport = :${PORT} )" | grep -q LISTEN; then
  echo "FAIL port ${PORT} is in use (GLM or V4-Flash still up?)"; exit 1
fi

"${docker_cmd[@]}"
# Evidence manifest (T3.19f): what served, on what, in what box state, with a content hash the
# evidence scripts stamp into every JSON they write. A serve without one cannot produce evidence,
# so a write failure stops the container rather than leaving an unrecorded server up.
if ! manifest="$(python3 "$(dirname -- "${BASH_SOURCE[0]}")/evidence_manifest.py" write --container "$CONTAINER_NAME" --out-dir "$MANIFEST_DIR")"; then
  echo "FAIL evidence manifest not written; stopping $CONTAINER_NAME"; docker rm -f "$CONTAINER_NAME" >/dev/null; exit 1
fi
echo "manifest $manifest"
echo "started $CONTAINER_NAME (about 4 min to ready); follow with: docker logs -f $CONTAINER_NAME"
echo "then: scripts/wait-ready.py --url http://${BIND_ADDRESS}:${PORT} && scripts/verify.py --url http://${BIND_ADDRESS}:${PORT} --model ${SERVED_MODEL_NAME}"

#!/usr/bin/env bash
# Source this. Loads $ENV_FILE (default: repo .env) if present, then defaults.
# Only KEY=VALUE lines are accepted; anything else is rejected so a stray
# command in .env cannot execute. A trailing unquoted " # comment" is dropped
# and a fully "quoted"/'quoted' value is unquoted (R21); nothing is expanded.
ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*=[^\`\$\(\)\;\&\|]*$ ]]; then
      key="${line%%=*}"; val="${line#*=}"
      # A fully quoted value keeps everything inside the quotes (including '#');
      # otherwise a trailing unquoted " # comment" and trailing blanks are dropped.
      if [[ "$val" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ ]] || [[ "$val" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
        val="${BASH_REMATCH[1]}"
      else
        val="${val%%[[:space:]]#*}"
        val="${val%"${val##*[![:space:]]}"}"
      fi
      export "${key?}=${val}"
    else
      echo "FAIL invalid line in $ENV_FILE: $line" >&2
      # shellcheck disable=SC2317  # exit is reached when this file is executed rather than sourced
      return 1 2>/dev/null || exit 1
    fi
  done <"$ENV_FILE"
fi

: "${MODEL_DIR:=}"
: "${HF_CACHE_DIR:=${HOME}/.cache/huggingface}"
: "${GPU_DEVICES:=0,1,2,3}"
: "${BIND_ADDRESS:=127.0.0.1}"
: "${PORT:=8010}"
: "${CONTAINER_NAME:=dsv41-flash-rtxpro6000}"
: "${SERVED_MODEL_NAME:=deepseek-v4.1-flash-local}"
: "${RUNTIME_IMAGE:=lmsysorg/sglang:dev-dsv41}"
: "${PROFILE:=4gpu-sglang}"
: "${TP_SIZE:=4}"
: "${MAX_MODEL_LEN:=262144}"
: "${MEM_FRACTION_STATIC:=0.90}"
: "${CHUNKED_PREFILL_SIZE:=8192}"
: "${MAX_RUNNING_REQUESTS:=8}"
: "${DSPARK_K:=${DSPARK_BLOCK_SIZE:-5}}"  # DSPARK_BLOCK_SIZE is the pre-R21 name; DSPARK_K matches the hybrid knob
unset DSPARK_BLOCK_SIZE
: "${KV_CACHE_DTYPE:=auto}"
: "${ENGRAM_PLACEMENT:=dram}"
: "${ENGRAM_HOST_TABLE_LAYOUT:=shared}"
: "${GPU_RESERVE_GIB:=4}"
: "${HOST_RESERVE_GIB:=24}"
: "${ALLOW_UNQUALIFIED:=0}"
: "${NCCL_DEBUG:=WARN}"
export MODEL_DIR HF_CACHE_DIR GPU_DEVICES BIND_ADDRESS PORT CONTAINER_NAME SERVED_MODEL_NAME RUNTIME_IMAGE \
  PROFILE TP_SIZE MAX_MODEL_LEN MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE MAX_RUNNING_REQUESTS DSPARK_K \
  KV_CACHE_DTYPE ENGRAM_PLACEMENT ENGRAM_HOST_TABLE_LAYOUT GPU_RESERVE_GIB HOST_RESERVE_GIB ALLOW_UNQUALIFIED NCCL_DEBUG

# Pinned identities. Change only with a PROVENANCE.md update.
# 2026-10-23 (E47 / T3.56): flipped to the 1017 checkpoint; the previous pin
# was deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a40aa45a94ad051997016db3960a90277.
export MODEL_REPO="deepseek-ai/DeepSeek-V4.1-Flash-1017"
export MODEL_REVISION="1bae0e1088993b9b82de08f59de5be5a34e42170"
export MODEL_PIN="${MODEL_REPO}@${MODEL_REVISION}"
export MODEL_SHARDS=48
export MODEL_TOTAL_BYTES=510286023000  # 475.2408927306533 GiB; index metadata and shard headers agree

gpu_count() { local IFS=,; local -a a; read -r -a a <<<"$GPU_DEVICES"; echo "${#a[@]}"; }

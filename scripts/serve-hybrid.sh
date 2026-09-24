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

# Image built by config/Dockerfile.dsv41-kt (dev-dsv41 + pruned kt-kernel).
IMAGE="${IMAGE:-dsv41-kt:recipe}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-kt}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsv41-flash-hybrid}"
BIND_ADDRESS="${BIND_ADDRESS:-127.0.0.1}"
PORT="${PORT:-8000}"

# Checkpoint: the HF hub cache directory is mounted whole so the Engram `file`
# layout can mmap the shard that holds the tables.
HF_MODEL_DIR="${HF_MODEL_DIR:-$HOME/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash}"
SNAPSHOT="${SNAPSHOT:-dba1be0a40aa45a94ad051997016db3960a90277}"
# SGLang overlay (branch dsv41-kt-hybrid) bind-mounted over the image's package.
OVERLAY_DIR="${OVERLAY_DIR:-$HOME/dev/src/sglang-dsv41-overlay}"

# Expert placement and CPU tier (E6-DESIGN, E17, E19).
# Of 384 per layer; GPU experts are the contiguous IDs 0..N-1, the rest run on the CPU.
# T3.6g (2026-09-23): 208 -> 226 once the KV pool was capped (MAX_TOTAL_TOKENS below);
# each expert costs ~0.33 GB per rank. 226 leaves 779 MiB on the busier GPU with all
# four slots at ~30K and decoding (scripts/stress-kv-full.py) - the ceiling for
# 32K x 4 slots with CHUNKED_PREFILL_SIZE=4096; 228 is projected not to fit. C1 wall
# 33.9 -> 36.1 tok/s, C4 47.5 -> 51.9, NLL neutral. 222 is the conservative fallback
# (~2.9 GB free in the same worst case). Qualified by the E27 served soak 2026-09-23
# (141K decoded tokens at C1-C4 on 226/158 + capture list 1 2 3 4, 0 NaN, NLL identical).
# 230 was tried (E29, 2026-09-24): needs CHUNKED_PREFILL_SIZE=2048 + MEM_FRACTION_STATIC=0.975
# + SGLANG_LOGPROB_CHUNK_SIZE=512 to fit, gives +1.6 % C1 wall for -40 % prefill and
# fails the NLL gate (+0.0088) - rejected. 226 is the ceiling at this VRAM.
KT_NUM_GPU_EXPERTS="${KT_NUM_GPU_EXPERTS:-226}"
# One kt worker per physical core (E17; E26 A1 on the 16-core 9955WX: 14 and 16
# threads are a plateau at C1 ≈ 33.5 wall tok/s, 24 costs −12 %, 30 collapses to
# 5 tok/s). kt-kernel pins each worker to a distinct hwloc core and busy-spins
# 50 ms before sleeping; workers beyond the core count float on SMT siblings and
# starve the two TP-rank launch threads. Never set this above the physical cores.
KT_CPUINFER="${KT_CPUINFER:-16}"
PHYS_CORES="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l || true)"
if [[ "${PHYS_CORES:-0}" -gt 0 && "$KT_CPUINFER" -gt "$PHYS_CORES" ]]; then
  echo "WARNING: KT_CPUINFER=$KT_CPUINFER exceeds the $PHYS_CORES physical cores; E26 A1 measured −12 % at 24 and a 7× collapse at 30 threads" >&2
fi
# Expert deferral (E21): each token's D lowest-scored routed experts are computed
# during the next layer and added one layer late, hiding most of the CPU-tier wait.
# D=4 keeps every token's top-2 on time: +22 % C1 / +17 % C2 / +22 % C4 over D=0,
# teacher-forced NLL unchanged (ΔNLL −0.006, z 1.4). D=5 no longer crashes on the
# patched image (E24/T3.15, 2026-09-23: the bs=4 NaN was kt bug C) and is +4.7 % C1
# wall, but ΔNLL +0.00287 breaks the 0.002 gate and greedy bs=1 output diverges
# run-to-run after 9-39 chars (D=4: 225-365) -- rejected on quality. D=6 loses
# 0.23 nats/token. 0 disables.
KT_MAX_DEFERRED="${KT_MAX_DEFERRED:-4}"
# Layerwise GPU prefill (E22): prefill chunks with at least this many tokens
# stream each layer's CPU experts to the GPUs (kt-kernel export -> shared pinned
# buffer -> per-rank H2D -> FlashInfer MoE) instead of computing them on the CPU.
# 0 keeps every chunk on the CPU path. KT_GPU_PREFILL_LOG=1 logs per-chunk timing.
# Default 1024 since E22 (2026-09-22): 837 tok/s at 16K with the 4096 chunk,
# 1.4K tok/s with CHUNKED_PREFILL_SIZE=8192 (leaves ~1.2 GiB VRAM; no room for
# DSpark drafts). Decode is unchanged (E22d: C1 38.4 / C2 47.8 / C4 145).
KT_GPU_PREFILL_MIN_TOKENS="${KT_GPU_PREFILL_MIN_TOKENS:-1024}"
KT_GPU_PREFILL_LOG="${KT_GPU_PREFILL_LOG:-0}"
# E22d debugging: per-layer non-finite check on forwards of >= N MoE tokens (0 = off).
KT_NAN_PROBE_MIN_TOKENS="${KT_NAN_PROBE_MIN_TOKENS:-0}"
# Speculative decoding (E23 / T3.9): DSPARK=1 adds the three MTP draft stages
# (--speculative-algorithm DSPARK, block size DSPARK_K). Off by default: E6-DESIGN
# 4.6 predicts K=5 loses under a CPU-bound MoE; E23 measures off / K=2 / K=5.
# Draft weights are ~3.7 GiB per card, so check free VRAM after graph capture.
DSPARK="${DSPARK:-0}"
DSPARK_K="${DSPARK_K:-5}"
# 0.92 -> 0.96 with T3.6g: the KV pool is capped by MAX_TOTAL_TOKENS, so the fraction
# no longer sizes it; it only has to leave room for prefill activations and capture.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.96}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
# 4096 kept ~3.3 GiB VRAM free after a 32K prefill at 208 experts (E26 A4); that
# headroom is now spent on experts. 8192 (1.4K tok/s prefill) needed ~2.2 GB more
# and does not fit beside 226 experts (projected from E22's 1.2 GiB at 208); use it
# only with KT_NUM_GPU_EXPERTS=208. 16384 never fit. 2048 frees ~1.6 / 1.2 GB per
# rank of worst-case headroom (E29) but costs 40 % of prefill throughput (16K: 860 ->
# 514 tok/s; the per-chunk CPU sync + Engram gather dominates) and is neutral on
# decode - only worth it for a deployment whose prompts are all under ~1K tokens.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
SGLANG_LOGPROB_CHUNK_SIZE="${SGLANG_LOGPROB_CHUNK_SIZE:-512}"
# KV pool cap (T3.6g). Left to the fraction, SGLang sized the pool at 1,933,056 tokens
# (~3.3 GB/rank) for a server that admits MAX_RUNNING_REQUESTS x CONTEXT_LENGTH =
# 131,072 - 15x oversized (E26 A4). Derived here as slots x context + one 16K chunk
# of headroom (147,456 at the defaults; ~0.6 GB/rank). The freed ~2.7 GB/rank holds
# 18 more experts per layer. Set MAX_TOTAL_TOKENS="" to let the fraction size it
# again (needed if CONTEXT_LENGTH grows past what the experts leave room for).
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS-$(( MAX_RUNNING_REQUESTS * CONTEXT_LENGTH + 16384 ))}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-4}"
# Explicit decode capture list. SGLang derives [1, 2, 4] from --cuda-graph-max-bs-decode 4 and
# pads a 3-row batch to 4, which cost C3 22 % (38.3 vs 49.0 summed tok/s, E26 B1). Capturing
# bs=3 too costs ~nothing (6.2 GB free before capture) and leaves bs=1 untouched (NLL gate
# ΔNLL −0.001, same-prompt gate PASS at C2/C3/C4). Qualified by the E27 served soak
# 2026-09-23 (see KT_NUM_GPU_EXPERTS above). Set CUDA_GRAPH_BS="" to fall back to the derived list.
CUDA_GRAPH_BS="${CUDA_GRAPH_BS-1 2 3 4}"
ENGRAM_FILE_THREADS="${ENGRAM_FILE_THREADS:-16}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"          # appended verbatim (word-split)
# Space-separated KEY=VALUE pairs passed into the container as -e flags, e.g.
# EXTRA_DOCKER_ENV="CUDA_LAUNCH_BLOCKING=1" to make async CUDA faults report
# at the kernel that raised them (much slower; diagnosis only).
EXTRA_DOCKER_ENV="${EXTRA_DOCKER_ENV:-}"
# Extra `docker run` flags, word-split. Diagnosis only: e.g. bind-mount an
# instrumented copy of a kt-kernel module over the image's file (E23 R3) with
# EXTRA_DOCKER_ARGS="-v /tmp/x/experts_base.py:/opt/sglang/lib/python3.12/site-packages/kt_kernel/experts_base.py:ro"
EXTRA_DOCKER_ARGS="${EXTRA_DOCKER_ARGS:-}"

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
if [[ -n "$CUDA_GRAPH_BS" ]]; then
  # shellcheck disable=SC2206  # intentional word splitting of the batch-size list
  server_args+=(--cuda-graph-bs-decode $CUDA_GRAPH_BS)
fi
if [[ -n "$MAX_TOTAL_TOKENS" ]]; then
  server_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
fi
# shellcheck disable=SC2206  # intentional word splitting of the override string
server_args+=($EXTRA_SERVER_ARGS)

extra_env_flags=()
for kv in $EXTRA_DOCKER_ENV; do
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
  # Engram tables stay in the checkpoint shard on NVMe (E18, overlay a1ca246a).
  -e SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
  -e SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file
  -e "SGLANG_DSV41_ENGRAM_FILE_DIR=${snapshot_dir}"
  -e "SGLANG_DSV41_ENGRAM_FILE_THREADS=${ENGRAM_FILE_THREADS}"
  # SM120 indexer defaults pick paths V4.1 cannot take (TROUBLESHOOTING).
  -e SGLANG_OPT_USE_TILELANG_INDEXER=0
  -e SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
  # A return_logprob request materialises chunk x 129,280-vocab fp32 logits: 1.06 GB at
  # SGLang's default 2048, the largest single allocation the server makes, and a CUDA
  # OOM there kills the server, not the request (E29). 512 is bit-identical (NLL delta 0.0
  # on every token) and no slower end to end.
  -e "SGLANG_LOGPROB_CHUNK_SIZE=${SGLANG_LOGPROB_CHUNK_SIZE}"
  -e "KT_GPU_PREFILL_MIN_TOKENS=${KT_GPU_PREFILL_MIN_TOKENS}"
  -e "KT_GPU_PREFILL_LOG=${KT_GPU_PREFILL_LOG}"
  -e "KT_NAN_PROBE_MIN_TOKENS=${KT_NAN_PROBE_MIN_TOKENS}"
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
echo "started $CONTAINER_NAME (about 4 min to ready); follow with: docker logs -f $CONTAINER_NAME"
echo "then: scripts/wait-ready.py --url http://${BIND_ADDRESS}:${PORT} && scripts/verify.py --url http://${BIND_ADDRESS}:${PORT} --model ${SERVED_MODEL_NAME}"

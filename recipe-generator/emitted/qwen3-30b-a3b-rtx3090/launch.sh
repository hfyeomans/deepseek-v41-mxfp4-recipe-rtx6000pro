#!/usr/bin/env bash
# Generated llama.cpp launcher. Model: Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M
set -euo pipefail
MODEL="${1:-model.gguf}"
exec llama-server -m "$MODEL" \
  -ngl 99 -fa 1 -c ${CONTEXT_LENGTH:-4096} --host 127.0.0.1 --port 8000
# Offload variants if the full model does not fit (see engine-notes.md):
#   -ot exps=CPU                  regex expert offload to CPU
#   --n-cpu-moe N / --cpu-moe      put N MoE experts (or all) on CPU
#   --moe-expert-cache N          LRU VRAM cache over host experts (draft PR)

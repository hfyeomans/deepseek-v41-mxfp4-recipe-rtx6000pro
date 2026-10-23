#!/usr/bin/env bash
# Regenerate the worked-example outputs and re-run the calculator self-tests.
# GPU-free: reads headers/budgets only, never launches a serve. Usage: bash examples/run-examples.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CALC="$ROOT/placement_calc.py"

python3 "$CALC" "$ROOT/examples/dsv41-flash/budget.json" \
  --hardware "$ROOT/examples/dsv41-flash/hardware.json" > "$ROOT/examples/dsv41-flash/result.txt"
python3 "$CALC" "$ROOT/examples/dsv41-1017/budget.json" \
  --hardware "$ROOT/examples/dsv41-1017/hardware.json" --engine sglang-hybrid > "$ROOT/examples/dsv41-1017/result.txt"
python3 "$CALC" "$ROOT/examples/synthetic-211g/budget.json" \
  --hardware "$ROOT/examples/synthetic-211g/hardware.json" > "$ROOT/examples/synthetic-211g/result.txt"
python3 "$CALC" "$ROOT/examples/qwen3-30b-a3b/budget.json" \
  --hardware "$ROOT/examples/qwen3-30b-a3b/hardware-rtx5090.json" \
  --quant gguf-q4_k_m --context-tokens 32768 --sequences 1 > "$ROOT/examples/qwen3-30b-a3b/result-rtx5090.txt"
python3 "$CALC" "$ROOT/examples/qwen3-30b-a3b/budget.json" \
  --hardware "$ROOT/examples/qwen3-30b-a3b/hardware-rtx3090.json" \
  --quant gguf-q4_k_m --context-tokens 4096 --sequences 1 > "$ROOT/examples/qwen3-30b-a3b/result-rtx3090.txt"

python3 "$ROOT/tests/test-placement-calc.py"
echo "examples regenerated; self-tests green"

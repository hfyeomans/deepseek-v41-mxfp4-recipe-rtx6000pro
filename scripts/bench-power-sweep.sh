#!/usr/bin/env bash
# E8b: tok/s-per-watt curve. Repeats the C1 decode and 32K prefill rows at each
# power cap, sampling nvidia-smi at 500 ms, and restores the default limit.
#
# Prerequisite: the server under test must be up and healthy at --url. For the E8
# baseline that is the sibling V4-Flash deployment:
#   (cd ~/dev/src/deepseek-v4-flash-nvfp4-tp2 && bash recipe.sh)   # ~5 min to healthy
# Needs root for `nvidia-smi -pl` (no passwordless sudo on this host), so run:
#   sudo bash scripts/bench-power-sweep.sh --url http://127.0.0.1:8000 --model dsv4-nvfp4
# Caps below power.min_limit (250 W on RTX PRO 6000 Max-Q) are rejected by the driver.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

URL="http://127.0.0.1:8000"
MODEL=""
CAPS="300,275,250"
OUT="$ROOT/evidence/e8b-power-sweep"
RUNS=3
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --caps) CAPS="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] || { echo "--model required" >&2; exit 2; }
if ! curl -sf -m 5 "$URL/health" >/dev/null; then
  echo "no healthy server at $URL — start the deployment first (see header), wait for /health, then rerun" >&2
  exit 3
fi
mkdir -p "$OUT"

DEFAULT_PL="$(nvidia-smi -i 0 --query-gpu=power.default_limit --format=csv,noheader,nounits | cut -d. -f1)"
GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
restore() { for g in $GPUS; do nvidia-smi -i "$g" -pl "$DEFAULT_PL" >/dev/null || true; done; echo "restored -pl $DEFAULT_PL"; }
trap restore EXIT

sample() { # $1 = csv path; prints sampler pid
  nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu \
    --format=csv,nounits -lms 500 > "$1" &
  echo $!
}

IFS=, read -r -a caps <<< "$CAPS"
for cap in "${caps[@]}"; do
  echo "== power cap ${cap} W"
  for g in $GPUS; do nvidia-smi -i "$g" -pl "$cap" >/dev/null; done
  sleep 3
  pid="$(sample "$OUT/power-decode-${cap}w.csv")"
  python3 "$ROOT/scripts/bench-decode.py" --url "$URL" --model "$MODEL" --concurrency 1 --runs "$RUNS" --warmup 1 --max-tokens 512 \
    --output "$OUT/decode-c1-${cap}w.json" | tail -1
  kill "$pid" || true
  pid="$(sample "$OUT/power-prefill-${cap}w.csv")"
  python3 "$ROOT/scripts/bench-prefill.py" --url "$URL" --model "$MODEL" --tokens 32768 --runs "$RUNS" \
    --output "$OUT/prefill-32k-${cap}w.json" | tail -1
  kill "$pid" || true
done
python3 "$ROOT/scripts/summarize-power.py" "$OUT"

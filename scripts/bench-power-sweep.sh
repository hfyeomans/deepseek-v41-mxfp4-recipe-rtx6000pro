#!/usr/bin/env bash
# E8b: tok/s-per-watt curve. Repeats the decode (C1 by default) and 32K prefill
# rows at each power cap, sampling nvidia-smi at 500 ms, and restores the default limit.
# --concurrency N runs the decode row at N streams (files become decode-cN-*w.json /
# power-decode-cN-*w.csv); the summary reports wall tok/s for N > 1. Use it for the
# 325 W question: single-stream decode never reaches the cap, batched decode and
# prefill do. Caps above power.default_limit (300 W) are allowed up to power.max_limit
# (325 W on this card); watch temp_max_c in the summary.
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
PHASES="decode,prefill"
CONC=1
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --caps) CAPS="$2"; shift 2 ;;
    --output) OUT="$2"; shift 2 ;;
    --runs) RUNS="$2"; shift 2 ;;
    --phases) PHASES="$2"; shift 2 ;;   # e.g. --phases prefill to redo one row
    --concurrency) CONC="$2"; shift 2 ;;  # decode streams per run (single integer)
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
SAMPLER_PID=""
restore() {
  # Kill the sampler first: a bench that dies mid-cap (first E8b run: server not up)
  # otherwise leaves a root nvidia-smi loop that keeps appending to the CSV forever and
  # NUL-fills any later run's file of the same name.
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true
  for g in $GPUS; do nvidia-smi -i "$g" -pl "$DEFAULT_PL" >/dev/null || true; done
  echo "restored -pl $DEFAULT_PL"
  # Running under sudo leaves root-owned evidence; hand it back to the invoking user.
  [ -n "${SUDO_USER:-}" ] && chown -R "$SUDO_USER" "$OUT" || true
}
trap restore EXIT

sample() { # $1 = csv path; sets SAMPLER_PID
  nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu \
    --format=csv,nounits -lms 500 > "$1" &
  SAMPLER_PID=$!
}
stop_sample() { kill "$SAMPLER_PID" 2>/dev/null || true; SAMPLER_PID=""; }

IFS=, read -r -a caps <<< "$CAPS"
for cap in "${caps[@]}"; do
  echo "== power cap ${cap} W"
  for g in $GPUS; do nvidia-smi -i "$g" -pl "$cap" >/dev/null; done
  sleep 3
  case ",$PHASES," in *,decode,*)
    # C1 keeps the historical names (power-decode-300w.csv, decode-c1-300w.json).
    suffix=""; [ "$CONC" != 1 ] && suffix="-c${CONC}"
    sample "$OUT/power-decode${suffix}-${cap}w.csv"
    python3 "$ROOT/scripts/bench-decode.py" --url "$URL" --model "$MODEL" --concurrency "$CONC" --runs "$RUNS" --warmup 1 --max-tokens 512 \
      --output "$OUT/decode-c${CONC}-${cap}w.json" | tail -1
    stop_sample ;;
  esac
  case ",$PHASES," in *,prefill,*)
    # Seed per cap: bench-prefill derives prompts from the seed, and the server's prefix
    # cache would otherwise serve caps 2..n from cap 1's KV (32K tokens in 0.14 s).
    sample "$OUT/power-prefill-${cap}w.csv"
    python3 "$ROOT/scripts/bench-prefill.py" --url "$URL" --model "$MODEL" --tokens 32768 --runs "$RUNS" --seed "$((777 + cap))" \
      --output "$OUT/prefill-32k-${cap}w.json" | tail -1
    stop_sample ;;
  esac
done
python3 "$ROOT/scripts/summarize-power.py" "$OUT"

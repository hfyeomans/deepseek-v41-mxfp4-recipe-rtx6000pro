#!/usr/bin/env bash
# CPU-seconds burned by kt-kernel's worker threads (numa_*_t_*/numa_*_m_*) in
# the TP0 scheduler process per decode step, at several concurrencies.
# Separates "kt does more work per step" from "kt waits longer per step".
#
#   bash scripts/kt-cpu-per-step.sh [--url URL] [--model NAME] [--concurrency 1,2,4]
#                                   [--max-tokens 200] [--output DIR]
set -euo pipefail
URL=http://127.0.0.1:8000
MODEL=dsv41-flash-hybrid
CONC=1,2,4
MAXTOK=200
OUT=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL=$2; shift 2;;
    --model) MODEL=$2; shift 2;;
    --concurrency) CONC=$2; shift 2;;
    --max-tokens) MAXTOK=$2; shift 2;;
    --output) OUT=$2; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
CPID=$(docker inspect -f '{{.State.Pid}}' dsv41-kt)
# TP0 = the child of the container entrypoint that owns the numa_* threads
TP0=$(for p in $(pgrep -P "$CPID"; pgrep -P "$(pgrep -P "$CPID" | head -1)" || true); do
        if grep -qs '^numa_' /proc/"$p"/task/*/comm 2>/dev/null; then echo "$p"; break; fi; done)
[[ -n "$TP0" ]] || { echo "could not find the TP0 process with numa_* threads" >&2; exit 1; }
tick=$(getconf CLK_TCK)

cpu_kt() {  # CPU-seconds of kt threads + main scheduler thread
  local kt=0 main=0
  for t in /proc/"$TP0"/task/*; do
    local comm; comm=$(cat "$t"/comm)
    local j; j=$(awk '{print $14+$15}' "$t"/stat)
    case "$comm" in
      numa_*) kt=$((kt + j));;
    esac
    [[ "$(basename "$t")" == "$TP0" ]] && main=$j
  done
  echo "$kt $main"
}

[[ -n "$OUT" ]] && mkdir -p "$OUT"
printf "%3s %8s %10s %12s %12s %12s %12s\n" C steps "wall s" "kt cpu-s" "kt cpu/step" "main cpu/step" "kt threads busy"
IFS=, read -ra CS <<<"$CONC"
for c in "${CS[@]}"; do
  read -r kt0 m0 <<<"$(cpu_kt)"
  t0=$(date +%s.%N)
  python3 "$HERE/bench-decode.py" --url "$URL" --model "$MODEL" --concurrency "$c" \
    --max-tokens "$MAXTOK" --runs 1 --warmup 0 --distinct-prompts --fixture-offset $((c * 3)) \
    --seed $((100 + c)) > "${OUT:-/tmp}/kt-cpu-c$c.bench.out" 2>&1
  t1=$(date +%s.%N)
  read -r kt1 m1 <<<"$(cpu_kt)"
  wall=$(echo "$t1 - $t0" | bc -l)
  ktsec=$(echo "($kt1 - $kt0) / $tick" | bc -l)
  msec=$(echo "($m1 - $m0) / $tick" | bc -l)
  # steps ~= max_tokens (one step per token per stream, streams share steps);
  # prefill steps are few and ignored here
  printf "%3d %8d %10.1f %12.2f %12.4f %12.4f %12.2f\n" "$c" "$MAXTOK" "$wall" "$ktsec" \
    "$(echo "$ktsec / $MAXTOK" | bc -l)" "$(echo "$msec / $MAXTOK" | bc -l)" "$(echo "$ktsec / $wall" | bc -l)"
  grep -E "^\s+[0-9]+\s" "${OUT:-/tmp}/kt-cpu-c$c.bench.out" | sed 's/^/      bench: /'
done

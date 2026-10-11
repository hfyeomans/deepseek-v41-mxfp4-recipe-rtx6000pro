#!/usr/bin/env bash
# gpu-power-sample.sh <csv> -- <command...>
#
# Run <command> while nvidia-smi samples both cards at 500 ms into <csv> (power.draw,
# SM / memory clock, temperature, utilization, fan, clock-event bitmask: the same columns
# bench-power-sweep.sh wrote for E8b / T2.9, so `summarize-power.py <csv>` reads it).
# The sampler is killed when the command exits, whatever the exit status, and the
# command's status is returned. T3.30 recorded that no serve after E23 had a power or
# temperature trace; release-smoke.sh --full now wraps its four benches in this.
#
#   bash scripts/gpu-power-sample.sh evidence/x/power-decode-c4.csv -- \
#     python3 scripts/bench-decode.py --url http://127.0.0.1:8000 --concurrency 4 ...
set -uo pipefail
[[ $# -ge 3 && "$2" == "--" ]] || { sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2; exit 2; }
csv="$1"; shift 2
mkdir -p "$(dirname "$csv")"
nvidia-smi --query-gpu=timestamp,index,power.draw,clocks.current.sm,clocks.current.memory,temperature.gpu,utilization.gpu,fan.speed,clocks_event_reasons.active \
  --format=csv,nounits -lms 500 > "$csv" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
"$@"
rc=$?
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
# The trace is the point of this wrapper: a command that succeeded under a sampler that wrote no
# rows (nvidia-smi missing, killed early) is a failed measurement, not a pass (R10).
rows="$(grep -c '^[0-9]\{4\}/[0-9]\{2\}/[0-9]\{2\} ' "$csv" 2>/dev/null || true)"
if [[ "$rc" -eq 0 && "${rows:-0}" -lt 1 ]]; then
  echo "gpu-power-sample: $csv has no nvidia-smi sample rows; the command passed but the power trace is empty" >&2
  exit 1
fi
exit "$rc"

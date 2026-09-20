#!/usr/bin/env bash
# Fail-closed launch gate. Checks tooling, GPUs, the pinned checkpoint, and
# runs the memory placement calculator against the actual GPU count and host
# RAM. Refuses to continue when the model does not fit unless ALLOW_UNQUALIFIED=1
# (intended only for experiment E5, which documents the failure mode).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/load-env.sh
source "$ROOT/scripts/load-env.sh"

fail() { echo "FAIL $*" >&2; exit 1; }
ok() { echo "ok   $*"; }

for c in docker nvidia-smi python3; do command -v "$c" >/dev/null || fail "missing command: $c"; done
docker info >/dev/null 2>&1 || fail "Docker daemon unavailable"
ok "tooling"

n_gpus="$(gpu_count)"
[[ "$n_gpus" -eq "$TP_SIZE" ]] || fail "GPU_DEVICES lists $n_gpus GPUs but TP_SIZE=$TP_SIZE"
mapfile -t mem_mib < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_DEVICES" 2>/dev/null | tr -d ' ')
[[ "${#mem_mib[@]}" -eq "$n_gpus" ]] || fail "nvidia-smi did not report all of GPU_DEVICES=$GPU_DEVICES"
min_mib="${mem_mib[0]}"
for m in "${mem_mib[@]}"; do (( m < min_mib )) && min_mib="$m"; done
gpu_gib="$(awk -v m="$min_mib" 'BEGIN {printf "%.3f", m / 1024}')"
ok "$n_gpus GPUs, smallest ${gpu_gib} GiB"

if [[ "$n_gpus" -gt 1 ]]; then
  if nvidia-smi topo -p2p r -i "$GPU_DEVICES" 2>/dev/null | grep -Eq '\b(NS|CNS|GNS|TNS|DR)\b'; then
    fail "P2P read is not OK between all selected GPUs"
  fi
  ok "P2P read matrix OK"
fi

host_gib="$(awk '/MemTotal/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"

[[ -n "$MODEL_DIR" ]] || fail "MODEL_DIR is not set"
[[ -f "$MODEL_DIR/RECIPE_PIN.txt" ]] || fail "missing $MODEL_DIR/RECIPE_PIN.txt (run scripts/download-model.sh)"
[[ "$(<"$MODEL_DIR/RECIPE_PIN.txt")" == "$MODEL_PIN" ]] || fail "RECIPE_PIN.txt does not match $MODEL_PIN"
python3 - "$MODEL_DIR" "$MODEL_SHARDS" "$MODEL_TOTAL_BYTES" <<'PY'
import json, sys
from pathlib import Path, PurePosixPath
root, shards_expected, total_expected = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
try:
    index = json.loads((root / "model.safetensors.index.json").read_text())
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"FAIL checkpoint index unreadable: {exc}")
shards = sorted(set(index["weight_map"].values()))
if len(shards) != shards_expected:
    raise SystemExit(f"FAIL expected {shards_expected} shards, found {len(shards)}")
size = 0
for name in shards:
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise SystemExit(f"FAIL unsafe shard path {name!r}")
    p = root / rel
    if not p.is_file():
        raise SystemExit(f"FAIL shard missing: {name}")
    size += p.stat().st_size
header_overhead = size - total_expected
if header_overhead < 0 or header_overhead > 64 * 1024 * 1024:
    raise SystemExit(f"FAIL shard bytes {size} inconsistent with pinned tensor total {total_expected}")
for req in ("config.json", "tokenizer_config.json"):
    if not (root / req).is_file():
        raise SystemExit(f"FAIL missing {req}")
print("ok   checkpoint shards and pin")
PY

budget="$ROOT/evidence/budget-upstream-bf16fp8.json"
[[ -f "$budget" ]] || fail "missing $budget"
layout="engram-host"; [[ "$ENGRAM_PLACEMENT" == "gpu" ]] && layout="gpu-only"
set +e
python3 "$ROOT/scripts/feasibility-calc.py" "$budget" --gpus "$n_gpus" --gpu-gib "$gpu_gib" --host-gib "$host_gib" \
  --util "$MEM_FRACTION_STATIC" --gpu-reserve-gib "$GPU_RESERVE_GIB" --host-reserve-gib "$HOST_RESERVE_GIB" \
  --context-tokens "$MAX_MODEL_LEN" --sequences 1 --layout "$layout"
rc=$?
set -e
if [[ $rc -eq 3 ]]; then
  if [[ "$ALLOW_UNQUALIFIED" == "1" ]]; then
    echo "WARN checkpoint does not fit this layout; continuing because ALLOW_UNQUALIFIED=1 (experiment mode)" >&2
  else
    fail "checkpoint does not fit $n_gpus GPU(s) with layout $layout. See docs/FEASIBILITY.md. Set ALLOW_UNQUALIFIED=1 only for experiment E5."
  fi
elif [[ $rc -ne 0 ]]; then
  fail "feasibility-calc.py error ($rc)"
else
  ok "memory placement fits ($layout, $n_gpus GPUs)"
fi

if [[ "$ENGRAM_PLACEMENT" == "dram" ]]; then
  hp="$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)"
  hp_kb="$(awk '/Hugepagesize/ {print $2}' /proc/meminfo)"
  hp_gib="$(awk -v n="$hp" -v k="$hp_kb" 'BEGIN {printf "%.1f", n * k / 1048576}')"
  if awk -v g="$hp_gib" 'BEGIN {exit !(g < 190)}'; then
    echo "WARN only ${hp_gib} GiB of hugepages reserved; SGLang Engram host table wants ~190 GiB (see docs/TROUBLESHOOTING.md)" >&2
  else
    ok "hugepages ${hp_gib} GiB"
  fi
fi

docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1 || fail "runtime image not present: $RUNTIME_IMAGE (docker pull it, then pin the digest in .env)"
ok "runtime image present"
echo "preflight passed for profile $PROFILE"

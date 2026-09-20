#!/usr/bin/env bash
# Print the hardware facts that decide feasibility, in a form safe to paste
# into public evidence: no GPU UUIDs, PCI bus IDs, serial numbers, hostnames
# or home paths. Add --json for machine-readable output.
#
# DIMM part numbers and configured speed need root (dmidecode). When run
# without root the script says so instead of guessing.
set -euo pipefail

json=0
[[ "${1:-}" == "--json" ]] && json=1

read_or() { cat "$1" 2>/dev/null || echo "$2"; }

board="$(read_or /sys/devices/virtual/dmi/id/board_vendor unknown) $(read_or /sys/devices/virtual/dmi/id/board_name unknown)"
bios="$(read_or /sys/devices/virtual/dmi/id/bios_version unknown) ($(read_or /sys/devices/virtual/dmi/id/bios_date unknown))"
cpu_model="$(lscpu | awk -F: '/Model name/ {gsub(/^ +/, "", $2); print $2; exit}')"
cpu_cores="$(lscpu | awk -F: '/^Core\(s\) per socket/ {gsub(/ /, "", $2); print $2}')"
cpu_threads="$(nproc)"
cpu_flags="$(lscpu | awk -F: '/^Flags/ {print $2}')"
has() { grep -qw "$1" <<<"$cpu_flags" && echo yes || echo no; }
numa_nodes="$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')"
mem_total_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
mem_total_gib="$(awk -v k="$mem_total_kb" 'BEGIN {printf "%.1f", k / 1048576}')"
hugepages="$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)"
thp="$(read_or /sys/kernel/mm/transparent_hugepage/enabled unknown)"
kernel="$(uname -r)"
# shellcheck disable=SC1091  # system file, not part of the repo
os="$(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || echo unknown)"

dimms="not read (requires root: sudo dmidecode -t memory)"
if command -v dmidecode >/dev/null && [[ $EUID -eq 0 ]]; then
  dimms="$(dmidecode -t memory | awk '
    /^Memory Device/ {inblk=1; size=""; speed=""; cfg=""; part=""; type=""}
    inblk && /^\tSize:/ {size=$2" "$3}
    inblk && /^\tType:/ {type=$2}
    inblk && /^\tSpeed:/ {speed=$2" "$3}
    inblk && /^\tConfigured Memory Speed:/ {cfg=$4" "$5}
    inblk && /^\tPart Number:/ {part=$3}
    inblk && /^$/ && size != "" && size != "No Module Installed" {printf "%s %s rated %s configured %s part %s; ", size, type, speed, cfg, part; inblk=0}
  ')"
fi

gpus=""
if command -v nvidia-smi >/dev/null; then
  gpus="$(nvidia-smi --query-gpu=name,memory.total,power.limit,power.max_limit,pcie.link.gen.max,pcie.link.width.max,clocks.max.sm,clocks.max.memory,driver_version --format=csv,noheader)"
fi
p2p="unknown"
if command -v nvidia-smi >/dev/null; then
  p2p="$(nvidia-smi topo -p2p r 2>/dev/null | awk '/^ *GPU[0-9]+[[:space:]]/ {$1=$1; printf "%s; ", $0}' || echo unknown)"
fi

nvmes=""
for d in /sys/class/nvme/nvme*; do
  [[ -e "$d" ]] || continue
  model="$(read_or "$d/model" unknown | xargs)"
  fw="$(read_or "$d/firmware_rev" unknown | xargs)"
  speed="$(read_or "$d/device/current_link_speed" unknown)"
  width="$(read_or "$d/device/current_link_width" unknown)"
  nvmes+="${model} fw ${fw} ${speed} x${width}; "
done

if [[ $json -eq 1 ]]; then
  python3 - "$board" "$bios" "$cpu_model" "$cpu_cores" "$cpu_threads" "$(has avx512f)" "$(has avx512_bf16)" "$(has amx_tile)" "$numa_nodes" "$mem_total_kb" "$hugepages" "$thp" "$dimms" "$gpus" "$p2p" "$nvmes" "$kernel" "$os" <<'PY'
import json, sys
a = sys.argv[1:]
gpus = []
for line in a[13].splitlines():
    f = [x.strip() for x in line.split(",")]
    if len(f) >= 9:
        gpus.append({"name": f[0], "memory_total_mib": int(f[1].split()[0]), "power_limit_w": float(f[2].split()[0]),
                     "power_max_limit_w": float(f[3].split()[0]), "pcie_gen_max": int(f[4]), "pcie_width_max": int(f[5]),
                     "sm_clock_max_mhz": int(f[6].split()[0]), "mem_clock_max_mhz": int(f[7].split()[0]), "driver": f[8]})
print(json.dumps({
    "board": a[0], "bios": a[1],
    "cpu": {"model": a[2], "cores": int(a[3]), "threads": int(a[4]), "avx512f": a[5], "avx512_bf16": a[6], "amx": a[7], "numa_nodes": int(a[8])},
    "memory": {"total_kib": int(a[9]), "total_gib": round(int(a[9]) / 1048576, 1), "hugepages_total": int(a[10]), "thp": a[11], "dimms": a[12]},
    "gpus": gpus, "p2p_read_matrix": a[14],
    "nvme": [s.strip() for s in a[15].split(";") if s.strip()],
    "kernel": a[16], "os": a[17],
}, indent=2))
PY
  exit 0
fi

cat <<EOF
Board            : $board
BIOS             : $bios
OS / kernel      : $os / $kernel
CPU              : $cpu_model ($cpu_cores cores / $cpu_threads threads; avx512f=$(has avx512f) avx512_bf16=$(has avx512_bf16) amx=$(has amx_tile); NUMA nodes=$numa_nodes)
Memory           : ${mem_total_gib} GiB total (MemTotal ${mem_total_kb} kB); HugePages_Total=$hugepages; THP=$thp
DIMMs            : $dimms
GPUs             :
                   ${gpus//$'\n'/$'\n'                   }
P2P read matrix  : $p2p
NVMe             : $nvmes
EOF

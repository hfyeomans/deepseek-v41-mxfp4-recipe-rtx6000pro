#!/usr/bin/env bash
# Read-only firmware / platform inventory for the BIOS optimisation map (docs/BIOS-OPTIMIZATION-GUIDE.md).
# Unprivileged: reads sysfs/procfs only. Never changes anything. Serial numbers, UUIDs and asset tags are
# not read (the sysfs files that hold them are root-only anyway) — safe to commit the output.
#
#   bash scripts/bios-inventory.sh > evidence/e25-dram-revalidation/bios-inventory.txt
#
# Re-run after every BIOS change and diff against the committed copy.
set -u

section() { printf '\n== %s\n' "$1"; }
show() { # show <label> <path>
    if [[ -r "$2" ]]; then printf '%-28s %s\n' "$1" "$(tr -d '\n' < "$2")"; else printf '%-28s (unreadable)\n' "$1"; fi
}

section "firmware (sysfs DMI)"
for f in sys_vendor product_name product_family board_vendor board_name board_version bios_vendor bios_version bios_date bios_release ec_firmware_release; do
    show "$f" /sys/class/dmi/id/$f
done

section "cpu"
grep -m1 'model name' /proc/cpuinfo | sed 's/.*: /model                        /'
grep -m1 'microcode' /proc/cpuinfo | sed 's/.*: /microcode                    /'
printf '%-28s %s\n' "cores/threads" "$(grep -c '^processor' /proc/cpuinfo) threads, $(lscpu | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}') cores"
printf '%-28s %s\n' "L3 instances (=CCDs)" "$(for c in /sys/devices/system/cpu/cpu[0-9]*/cache/index3/id; do cat "$c"; done | sort -u | wc -l) x $(cat /sys/devices/system/cpu/cpu0/cache/index3/size)"
printf '%-28s %s\n' "NUMA nodes (NPS)" "$(find /sys/devices/system/node -maxdepth 1 -name "node[0-9]*" | wc -l)"
printf '%-28s %s\n' "SMT" "$(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"
show "cpufreq driver" /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
show "governor" /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
show "EPP preference" /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
show "boost" /sys/devices/system/cpu/cpufreq/boost
show "amd_pstate status" /sys/devices/system/cpu/amd_pstate/status
show "amd_pstate prefcore" /sys/devices/system/cpu/amd_pstate/prefcore
show "max freq kHz" /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq
printf '%-28s %s\n' "cpuidle driver" "$(cat /sys/devices/system/cpu/cpuidle/current_driver 2>/dev/null)"
printf '%-28s %s\n' "C-states" "$(for s in /sys/devices/system/cpu/cpu0/cpuidle/state*; do printf '%s(%s) ' "$(cat "$s/name")" "$([[ $(cat "$s/disable") == 1 ]] && echo off || echo on)"; done)"
printf '%-28s %s\n' "kernel cmdline" "$(sed -E 's/UUID=[0-9a-fA-F-]+/UUID=<scrubbed>/g; s/BOOT_IMAGE=[^ ]+ //' /proc/cmdline)"

section "memory"
grep -E 'MemTotal|HugePages_Total|Hugepagesize' /proc/meminfo | sed 's/: */                  /'
show "THP enabled" /sys/kernel/mm/transparent_hugepage/enabled
show "THP defrag" /sys/kernel/mm/transparent_hugepage/defrag
printf '%-28s %s\n' "numactl -H" "$(numactl -H 2>/dev/null | grep -E 'available|node 0 size' | tr '\n' ';')"
echo "DIMM speed/part numbers need 'sudo dmidecode -t memory' (owner-run); transcribed copy in evidence/hardware-inventory.json"

section "ECC / EDAC / RAS"
if [[ -d /sys/devices/system/edac/mc ]]; then
    for mc in /sys/devices/system/edac/mc/mc[0-9]*; do
        printf '%-28s name=%s size=%s MiB ce=%s ue=%s\n' "$(basename "$mc")" "$(cat "$mc/mc_name" 2>/dev/null)" "$(cat "$mc/size_mb" 2>/dev/null)" "$(cat "$mc/ce_count" 2>/dev/null)" "$(cat "$mc/ue_count" 2>/dev/null)"
        for d in "$mc"/dimm*; do [[ -d $d ]] && printf '  %-26s %s %s ce=%s ue=%s\n' "$(basename "$d")" "$(cat "$d/dimm_label" 2>/dev/null)" "$(cat "$d/dimm_mem_type" 2>/dev/null)" "$(cat "$d/dimm_ce_count" 2>/dev/null)" "$(cat "$d/dimm_ue_count" 2>/dev/null)"; done
    done
else
    echo "no EDAC memory controller registered (amd64_edac not loaded, or ECC reporting off in BIOS)"
fi
printf '%-28s %s\n' "edac modules" "$(lsmod | awk '/edac/{print $1}' | tr '\n' ' ')"
printf '%-28s %s\n' "mce log entries" "$(find /sys/devices/system/machinecheck/machinecheck0/ -maxdepth 1 -mindepth 1 2>/dev/null | wc -l) sysfs attrs; use 'sudo journalctl -k | grep -iE \"mce|edac|ecc\"' (owner-run)"

section "PCIe / IOMMU"
printf '%-28s %s\n' "IOMMU groups" "$(find /sys/kernel/iommu_groups -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)"
printf '%-28s %s\n' "IOMMU devices" "$(find /sys/class/iommu -maxdepth 1 -mindepth 1 -printf "%f " 2>/dev/null)"
for g in $(lspci -Dn -d 10de: 2>/dev/null | awk '$2 ~ /^03/{print $1}'); do
    # BDF is machine-private (ci.sh rejects it); print ordinal only
    i=$((${i:-0}))
    printf 'GPU%-25s link %s x%s (max %s x%s); BAR1 %s\n' "$i" \
        "$(cat /sys/bus/pci/devices/"$g"/current_link_speed 2>/dev/null)" "$(cat /sys/bus/pci/devices/"$g"/current_link_width 2>/dev/null)" \
        "$(cat /sys/bus/pci/devices/"$g"/max_link_speed 2>/dev/null)" "$(cat /sys/bus/pci/devices/"$g"/max_link_width 2>/dev/null)" \
        "$(lspci -vvs "$g" 2>/dev/null | awk '/Region 1:/{print $NF}')"
    i=$((i+1))
done
printf '%-28s %s\n' "NVMe link" "$(for n in /sys/class/nvme/nvme*/device; do printf '%s x%s; ' "$(cat "$n/current_link_speed")" "$(cat "$n/current_link_width")"; done)"
printf '%-28s %s\n' "ReBAR" "BAR1 >= 96G VRAM above means Resizable BAR and Above-4G decoding are enabled"

section "power / thermal sensors"
for h in /sys/class/hwmon/hwmon*; do
    n=$(cat "$h/name" 2>/dev/null)
    case "$n" in
        k10temp) printf '%-28s Tctl %s mC\n' "$n" "$(cat "$h/temp1_input")";;
        amd_hsmp_hwmon) printf '%-28s socket power %s uW, cap %s uW\n' "$n" "$(cat "$h/power1_input")" "$(cat "$h/power1_cap" 2>/dev/null)";;
        corsairpsu) printf '%-28s PSU in %s uW\n' "$n" "$(cat "$h/power1_input" 2>/dev/null)";;
    esac
done
printf '%-28s %s\n' "GPU power limits" "$(nvidia-smi --query-gpu=power.limit,power.max_limit --format=csv,noheader 2>/dev/null | tr '\n' ';')"

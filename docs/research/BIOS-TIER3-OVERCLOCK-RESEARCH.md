# BIOS Tier 3 clock-domain and overclock research

**Target:** Ryzen Threadripper PRO 9955WX (Shimada Peak, Zen 5), ASUS Pro WS WRX90E-SAGE SE, BIOS 1317 / AGESA 1.0.0.1e, eight DDR5 ECC RDIMMs, two RTX PRO 6000 Blackwell Max-Q GPUs.  
**Purpose:** determine which clock limits CPU-originated DRAM reads before adopting any daily overclock.  
**Date:** 2026-09-23.

> **Integration note (recipe repo, 2026-09-23, after this document was written):** T3-0 was satisfied the same day *without a boot* — the running kernel's `hsmp_acpi` driver exposes `/dev/hsmp` world-readable and `scripts/hsmp-clocks.py` reads **FCLK 2000 MHz, MCLK 3200 MHz** at EXPO 6400, flat across idle and the host suite, plus a controller-side peak of 174 GB/s of 410 (BIOS guide §6, E27, `evidence/bios-trials/2026-09-23-boot3-expo-restored/`). §2.3's `amd_hsmp` row is therefore confirmed for this box (the sysfs tree has only `mclk_input`; the FCLK comes from the `GET_FCLK_MCLK` mailbox message). §5's 28.57 tok/s baseline is the rounded 35 ms step used in the briefing, not the qualified C1 (34.0 wall / 39.2 summed tok/s, E23 R3); carry its percentages, not its absolute tok/s. Everything else stands as written by the research thread; the recipe's protocol of record is E27 in `plan/EXPERIMENTS.md`.
>
> **Integration note 2 (2026-09-23, Boots 4–5):** the first Tier 3 boot recommended in §7 (FCLK 2100, DDR5-6400, UCLK Auto, voltages Auto) **failed**: POST passed, then the OS boot logged ≈ 500 corrected MCEs/s, all from MC bank 16 *Global Memory Interconnect PCS Unit* (Ext. Error Code 6, status `0x9820000000060150`), with DRAM EDAC at 0. The owner reverted to FCLK Auto (verified 2000 MHz, EDAC 0/0, 0 MCE). Per §4.3 the tier stops on any MCE; FCLK 2200 is struck and no voltage will be added to rescue 2100. This does not test H-GMI's bandwidth claim (nothing was measured at 2100), but it does say the stock GMI link has little clock margin on this sample — the single-system TR 9960X report in §2 that called 2100 "a conservative practical ceiling" at stock voltage (labelled **[C, one system]**) did not transfer to this 9955WX — consistent with §7 item 6 (no reproducible TR 9000 maximum-stable-FCLK distribution exists; results are sample-specific and sometimes nonmonotonic). Only the MEMCLK arm (§7 steps 6600 → 6800 at fixed FCLK) remains, at the owner's discretion. Evidence: `evidence/bios-trials/2026-09-23-boot4-t3f1-fclk2100/`; BIOS guide §6–§7; E27.
>
> **Integration note 3 (2026-09-23, Boot 6):** the fixed-FCLK MCLK arm (§7 step 5, T3-M1 = DDR5-6600, FCLK Auto, UCLK Auto, voltages Auto) **ran and returned the H-GMI null**: HSMP held FCLK at 2000 MHz while MCLK read 3300; per-CCD read stayed 62.7 GB/s (0 %), socket 125.6, kt one-layer T=1 0.603 → 0.589 ms (noise); latency-bound rows (1-thread read, split ×2, copy, triad) fell a reproducible 7–10 % — consistent with looser Auto timings and/or a 1:2 UCLK divider above the profile, neither observable **[M-owner; cause inferred]**. EDAC 0/0, zero MCEs. Not adopted (fails ≥ +4 % per CCD); T3-M2 (6800) struck; owner reverts to slot 2. Tier 3 is closed under the no-added-voltage rule. This also resolves the §2 Boot 2 ambiguity in favour of the fabric-clock reading: −25 % MEMCLK moved the ceiling −11 % through the Auto-derived fabric clock, +3 % MEMCLK at fixed FCLK moved it 0 %. Evidence and detail: BIOS guide §6–§7, E27, `evidence/bios-trials/2026-09-23-boot6-t3m1-memclk6600/`.

## Evidence notation and scope

Every quantitative statement below is tagged:

- **[M-owner]** measured on this machine by the owner; this document did not run those tests.
- **[M-source]** measured by a cited third party.
- **[V]** vendor-documented (AMD, ASUS, Lenovo, or DIMM vendor).
- **[C]** community-reported or community-inferred; not a platform guarantee.
- **[D]** derived arithmetically from labeled inputs.
- **[P]** projected, not measured.

The BIOS setting names and availability in this document come from the photographed BIOS 1317 inventory in `BIOS-OPTIMIZATION-GUIDE.md`. That inventory is the ground truth for this firmware. A setting not in it is marked **verify present**. EPYC 9005 material is the closest public Zen 5 IOD documentation, not proof of an ASUS WRX90 default. No public AMD or ASUS document found states the Auto FCLK, live UCLK, UCLK ratio, daily-safe manual voltage ceilings, or GMI link allocation of a 9955WX.

## Executive conclusion

1. The old working model—Auto FCLK is fixed near 2000 MHz and Auto UCLK switches from 1:1 to 1:2 above roughly DDR5-6000—is **Zen 4 community guidance, not verified for Shimada Peak**. Do not use it as a Zen 5 premise.
2. The owner measured EXPO 6400 → JEDEC 4800, with the same visible fabric-policy settings, changing per-CCD read from 63.1 to 56.4 GB/s (−11%) and kt one-layer T=1 from 0.603 to 0.681 ms (+13%) **[M-owner]**. Thus at least one hidden/Auto-resolved IOD or controller behavior follows memory configuration; raw DIMM bandwidth cannot explain the 114 GB/s socket ceiling because DDR5-4800 across eight 64-bit channels is 307.2 GB/s theoretical **[D]**.
3. Public Zen 5 evidence does **not** identify which clock moved. Linux HSMP can expose live FCLK and MCLK when supported, but its documented interface has no UCLK field. The owner's live-clock tool should settle the first-order question on the next EXPO boot.
4. A Shimada Peak owner published AMD engineering correspondence saying a selectable 1:1 UCLK:MCLK mode is not architecturally supported on a TR 9960X **[C, indirect vendor correspondence]**. That is stronger and more relevant than Zen 4 folklore, but it is neither a public AMD specification nor a direct 9955WX measurement. Treat the BIOS `UCLK DIV1 MODE` control as **verify present and verify effective**, not as an assured experiment.
5. The 63.1 GB/s per-CCD ceiling **[M-owner]** closely matches a narrow GMI read path of 32 B/clock × about 2.0 GHz = about 64 GB/s **[D/C]**. It does not resemble the roughly 99.8 GB/s per CCD that Chips and Cheese measured on a Turin EPYC 9355P using GMI-Wide **[M-source]**. This is strong evidence that the 9955WX uses one narrow GMI interface per active CCD, but no public OPN table was found to prove it.
6. **Recommended first Tier 3 boot:** after restoring EXPO 6400, change nothing. Cold boot and record live FCLK/UCLK/MCLK both idle and under the host-bandwidth load, then reproduce the short bandwidth and kt baselines. Only then perform the first overclock boot: FCLK explicitly 2100 MHz, MCLK 6400, existing UCLK behavior untouched, and all voltages Auto. This avoids confounding clock discovery with an overclock.

## 1. Shimada Peak clock domains on WRX90

### 1.1 Domain map

```diagram
┌──────────────────────┐  per-CCD GMI  ┌─────────────────────────────┐
│ Zen 5 CCD 0          │──────────────▶│                             │
│ 8 cores + 32 MiB L3 │◀──────────────│ Zen 5 I/O die (IOD)         │
└──────────────────────┘               │                             │
                                       │ DF / memory controllers     │
┌──────────────────────┐  per-CCD GMI  │ FCLK: fabric clock          │
│ Zen 5 CCD 1          │──────────────▶│ UCLK: controller clock      │
│ 8 cores + 32 MiB L3 │◀──────────────│ MCLK: DRAM clock            │
└──────────────────────┘               └──────────────┬──────────────┘
                                                      │ 8 channels
                                                      ▼
                                       ┌─────────────────────────────┐
                                       │ DDR5 RDIMMs                 │
                                       │ DDR rate = 2 × MCLK         │
                                       └─────────────────────────────┘
```

| Domain | Meaning | What Auto is known to do on this machine / generation | Status |
|---|---|---|---|
| **MCLK** | Physical DRAM clock. DDR5-6400 means MCLK 3200 MHz and DDR5-4800 means MCLK 2400 MHz **[D]**. | Ai Overclock Tuner `EXPO` trains the rated DDR5-6400 profile; `Auto` trained JEDEC DDR5-4800 **[M-owner]**. AMD says EPYC `Memory speed = AUTO` trains the correct speed for DIMM population/rank **[V]**. | Known from configured rate; confirm live with the owner's tool or HSMP. |
| **UCLK** | Unified memory-controller clock inside the IOD. Its ratio to MCLK affects controller throughput and latency. | **Unknown for 9955WX.** Zen 4 reports of 1:1 through about DDR5-6000 and 1:2 above it are Zen 4-only **[C]**. Shimada Peak correspondence instead says the platform has a fixed divider topology and no selectable 1:1 mode **[C]**. It does not publish the actual ratio. | Must be measured. `UCLK DIV1 MODE` is **verify present/effective** on BIOS 1317. |
| **FCLK** | Data Fabric clock for the IOD fabric and CCD↔IOD path. It is distinct from MCLK and UCLK even when firmware coordinates them. | **Unknown for 9955WX Auto.** No AMD/ASUS source found saying Auto is fixed at 2000 MHz on TR 9000. BIOS exposes `FCLK Frequency` / `Infinity Fabric Frequency and Dividers`, Auto with selectable 800–3000 MHz **[M-owner BIOS inventory]**; the running Auto value was not shown. | Read live before changing it. |
| **GMI** | On-package Global Memory Interconnect between each CCD and the IOD. This is not xGMI, the socket-to-socket link. | AMD documents a dedicated GMI connection per CCD and says selected “wide” EPYC OPNs use two interfaces per CCD **[V]**. No public 9955WX link-allocation table found. Owner scaling identifies a separate per-CCD ceiling **[M-owner]**. | Narrow on 9955WX is a strong inference, not vendor documentation. |
| **DF P-state / APB policy** | Runtime power/frequency policy for fabric and memory controllers; not another physical link. | On a Zen 5 EPYC platform, Auto DF P-states dynamically change; P0 plus APBDIS=1 holds fabric/controllers at full power **[V-platform analog]**. Boot 1 already set APBDIS=1, DF C-states Disabled, and Maximum IO Performance **[M-owner]**. | Keep fixed throughout clock trials. |

**Sources:** [AMD EPYC 9005 architecture overview](https://docs.amd.com/api/khub/documents/iZ5eCtQ5v6PyPN7wEd8HQg/content), [AMD EPYC 9005 HPC tuning guide](https://docs.amd.com/api/khub/documents/v4Qq7nkUvpSfjc3bpPycCw/content), [Lenovo Zen 5 EPYC UEFI guide](https://lenovopress.lenovo.com/lp2210-tuning-uefi-settings-5th-gen-amd-epyc-processor-servers), [Shimada Peak UCLK discussion and quoted AMD correspondence](https://forum.level1techs.com/t/threadripper-9000-series-missing-uclk-div1-1-1-mode-but-why/239170), [AMD Shimada Peak platform white paper](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen-threadripper/shimada-peak-me-vertical-white-paper.pdf).

### 1.2 GMI width, clock, and the observed ceiling

Chips and Cheese reports these Zen 5 paths:

| Configuration | Read path | Write path | Example |
|---|---:|---:|---|
| GMI-Narrow | 32 B/clock **[M-source/inferred]** | 16 B/clock **[M-source/inferred]** | 32 B/clock × 2.0 GHz = 64 GB/s read; 16 B/clock × 2.0 GHz = 32 GB/s write **[D]** |
| GMI-Wide | 64 B/clock **[M-source]** | 64 B/clock **[M-source]** | 99.8 GB/s read measured on EPYC 9355P **[M-source]**, below its link's arithmetic maximum |

Sources: [Chips and Cheese, EPYC 9355P](https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core) and [Chips and Cheese, Turin UMA](https://chipsandcheese.com/p/evaluating-uniform-memory-access).

For a narrow read link:

\[
B_{read,CCD}=32\ \text{B/clock}\times FCLK
\]

At FCLK 2000/2100/2200 MHz this is 64.0/67.2/70.4 GB/s **[D]**. The owner's 63.1 GB/s at DDR5-6400 **[M-owner]** is 98.6% of the 2000 MHz narrow-link arithmetic ceiling **[D]**. Two independent CCDs then give 128 GB/s arithmetic, close to the measured 125–126 GB/s socket read **[M-owner]**. This fit is unusually strong evidence for narrow GMI near 2 GHz, but it does **not** prove that the Auto FCLK register actually reads 2000 MHz or that every transaction uses this simple rate.

Could the 9955WX use GMI-Wide? AMD says wide OPNs consume two of the IOD's GMI interfaces per CCD **[V]**, and Chips and Cheese measured almost 100 GB/s per wide CCD **[M-source]**. A wide path running at a much lower effective rate is logically possible, but the owner's exact 32 B/clock-shaped ceiling makes it unlikely. No AMD or ASUS source found names the 9955WX as wide or narrow; therefore **“9955WX is GMI-Narrow” remains community/measurement-inferred**.

### 1.3 What is known about Auto and maximum stable FCLK

- **Auto FCLK fixed at 2000 MHz? Not established.** That description comes from Zen 4 community experience **[C]**. No Zen 5 TR 9000, WRX90, AGESA 1.0.0.1x primary source found confirms it.
- A TR 9960X owner describes 2100 MHz as a conservative practical ceiling and reports DDR5-7200 CL34/FCLK 2100 at stock voltages increasing an MLC read result from about 176.6 to 213.6 GB/s versus DDR5-6400 **[C, one system]**. This is not a stress-tested platform limit. [Source](https://forum.level1techs.com/t/tr-9960x-cores-and-memory-bandwidth-questions/249797).
- A TR PRO 9985WX owner reports 6400/FCLK 2133 and 7200/FCLK 2167 performing similarly, while 7200/FCLK 2200 was about 40 GB/s slower **[C, one system]**. This warns that higher FCLK can produce divider/phase interactions and regress rather than scale monotonically. [Source](https://forum.level1techs.com/t/threadripper-9000-series-missing-uclk-div1-1-1-mode-but-why/239170/14).
- **No credible public TR 9000 or EPYC 9005 database was found establishing a maximum stable FCLK or typical required VSOC/VDDG.** The responsible starting point is 2100 MHz with Auto voltage, followed by 2200 MHz only after the complete gate. “Boots” is not “stable.”

## 2. Discriminating cold-boot experiment plan

### 2.1 Controls and common validation gate

Hold these constant in every arm: EXPO profile and primary timings unless the row explicitly changes MCLK; APBDIS=1; DF C-states Disabled; Maximum IO Performance; Power Down Disabled; RAS settings from Boot 1; NPS1; SMT on; PBO/Curve Optimizer/scalar Auto; all manual voltages Auto; same OS, affinity, benchmark sizes, GPU power limits, and ambient/cooling state.

Every row is a **separate cold boot**. Save a profile and photograph the changed page before exit. After POST:

1. Confirm requested FCLK/MCLK/UCLK live, memory configured rate, NUMA topology, and that inventory differs only as intended.
2. Require zero new EDAC corrected errors (CE), uncorrectable errors (UE), MCE/WHEA-equivalent events, GPU Xids, or training retries/failures **[project gate]**.
3. For a clock or memory change, run at least 200 GiB of `memtester`, or `stressapptest -M 200000 -s 1800` for 30 minutes **[project protocol]**, then re-read EDAC.
4. Run the host-bandwidth suite, including per-CCD, split-CCD, patterns, expert-chunk/rows, and concurrent GPU DMA; then the kt one-layer bench with the same layer/expert/thread arguments.
5. For a candidate worth keeping, run serving/32K prefill, NLL (absolute delta under 0.002 **[project threshold]**), needle verification, and a 20,000-token mixed 20K NaN soak **[project protocol]**.
6. **Stop and revert immediately** on any CE or UE, MCE, failed/repeated training, crash/assert/Xid, NLL or needle failure, NaN/nonfinite output, thermal limit/throttle, or more than 5% regression in any protected performance path **[project criterion]**. Do not “fix” the same boot by adding voltage.

### 2.2 Boot sequence and falsifiable predictions

| Boot | Exact change | Observation that supports FCLK/GMI binding | Observation that supports UCLK/UMC or memory-clock binding | Decision |
|---|---|---|---|---|
| **T3-0: live-clock baseline (first)** | Restore Boot 1 profile: EXPO DDR5-6400; **no BIOS changes**. Cold boot. Read FCLK/UCLK/MCLK idle and during per-CCD/split read. Repeat short suite + kt baseline. | FCLK is near 2000 MHz and changes with load/configuration; 63.1 GB/s approximates 32 B/clock × FCLK **[P]**. | UCLK value/ratio is visible and explains 6400 vs 4800; or FCLK is invariant while UCLK differs **[P]**. | This boot establishes facts and catches baseline drift. Do not overclock without it. |
| **T3-F1** | FCLK explicit **2100 MHz**, MCLK 3200 MHz / DDR5-6400, observed UCLK mode unchanged, all voltage Auto. | Per-CCD read rises about 5% toward 66 GB/s and socket/expert reads follow; kt T=1 falls roughly in proportion **[P]**. One such result falsifies “FCLK irrelevant.” | FCLK reads 2100 but ceilings remain within run spread (about ±2%) **[P]**. That falsifies a simple FCLK-limited model at this operating point. | Full gate. Keep only for the next experiment, not yet daily. |
| **T3-F2** | FCLK explicit **2200 MHz**, otherwise identical to T3-F1. Attempt only after T3-F1 fully passes. | Per-CCD read approaches 69–70 GB/s, about 10% over a 2000 MHz baseline **[P/D]**. | No gain or nonmonotonic regression indicates another divider/UMC limit or unfavorable clock interaction **[P]**. | Full gate. Stop rather than add voltage. |
| **T3-U1** | At DDR5-6400 and fixed winning/baseline FCLK, inspect `UCLK DIV1 MODE`. If present, force the current ratio first as a control, then force 1:1 and 1:2 in separate cold boots. | Changing UCLK ratio has little effect on saturated per-CCD read while FCLK controls it; latency may move **[P]**. | Per-CCD ceiling and kt latency change materially with UCLK while FCLK stays fixed **[P]**. | **Do not run if absent, rejected, or live UCLK does not change.** Shimada Peak evidence says selectable 1:1 may not exist. |
| **T3-M1** | If UCLK forcing is unavailable, hold FCLK at the established value; EXPO+ DDR5-6600, profile timings/voltage only. | Fixed FCLK leaves per-CCD ceiling flat even if socket copy/triad move **[P]**. | Per-CCD/expert ceiling rises with MCLK despite fixed FCLK **[P]**; this points to UCLK/UMC or another memory-derived divider. | Run only after FCLK arm identifies no adequate gain and only if DIMM/board firmware offers this rate. Full gate. |
| **T3-M2** | Fixed FCLK; DDR5-6800. Attempt only after T3-M1 fully passes. | Same as T3-M1. | Further monotonic gain supports a memory-derived controller bottleneck **[P]**; nonmonotonic loss exposes a bad divider/phase point. | Stop rather than tighten timings or add voltage. |

Interpretation is deliberately two-dimensional:

| Result | Conclusion |
|---|---|
| FCLK +5% produces about +5% per-CCD read; fixed-FCLK MCLK does not | Narrow GMI/FCLK is the dominant ceiling. |
| FCLK changes little; fixed-FCLK MCLK moves per-CCD/expert reads | UCLK/UMC or a memory-derived IOD clock is dominant. |
| Both move it | Mixed bottleneck; select the safer, more efficient gain rather than stacking immediately. |
| Neither moves it despite verified live clocks | The 6400→4800 result involved another Auto policy/divider; stop manual clocking and compare complete firmware-resolved state. |

### 2.3 Reading clocks from Linux

| Tool/path | Can read on Zen 5 TR? | Limitation / recommended use |
|---|---|---|
| **Owner's live-clock tool** | Owner says it reads FCLK/UCLK; use as the primary observation. | Record tool/version and idle/load values. Confirm that displayed “memory clock” distinguishes MCLK from effective DDR rate. |
| `amd_hsmp` sysfs | Current kernel documentation lists `fclk_input` and `mclk_input` for supported Family 1Ah EPYC telemetry **[V]**. Check `/sys/devices/platform/AMDI0097:0*/` and `/sys/bus/platform/drivers/amd_hsmp/`. | No documented `uclk_input`. Threadripper exposure is firmware/kernel-dependent even if HSMP power telemetry exists. Read-only observation is appropriate. [Kernel documentation](https://www.kernel.org/doc/html/next/arch/x86/amd_hsmp.html). |
| AMD E-SMI / `/dev/hsmp` | Can query documented HSMP telemetry where the platform exposes the protocol **[V]**. | Public table layout depends on protocol/PPR; not a shortcut to UCLK unless the table explicitly labels it. Do not infer UCLK from an unknown field. |
| `dmidecode -t 17` | Reads configured effective memory speed and SMBIOS voltage metadata. | Not live FCLK, UCLK, or MCLK; voltage may be nominal SPD rather than active EXPO voltage. |
| `ryzen_smu` | Upstream support list ends at older parts including Milan; no Shimada Peak/Family 1Ah support **[project documentation]**. | Model-specific PM-table layouts; unsafe to install merely for this test. [Project](https://github.com/leogx9r/ryzen_smu). |
| `ryzen_monitor` | Project advertises older Ryzen generations, not Zen 5 TR **[project documentation]**. | Depends on known PM-table layouts; do not treat a plausible number as valid. [Project](https://github.com/hattedsquirrel/ryzen_monitor). |
| `zenpower` / `zenmonitor` | Reads temperature, SVI2 voltage/current/power where supported **[project documentation]**. | Does not claim FCLK/UCLK/MCLK; no demonstrated Shimada Peak support. Replaces/conflicts with `k10temp`, so do not install for clocks. [Project](https://github.com/ocerman/zenpower/blob/master/README.md). |

## 3. From-Auto cleanup matrix

“Explicit no-op” means a useful documentation choice only if the resolved value is confirmed; it is not expected to increase throughput. Effects are relative to the present Boot 1 policy. `0` means no expected material effect; `±` means workload-dependent or small; arrows are projected direction, not measured facts.

### 3.1 CPU, boost, and power policy

| Group / setting | Auto resolution / explicit value | Per-CCD ceiling | kt layer latency | PCIe DMA / NCCL | Stability, ECC, thermals | Classification |
|---|---|---:|---:|---:|---|---|
| **PBO limits: PPT/TDC/EDC** | Current PBO Auto; package class 350 W **[M-owner/V]**. Leave Auto/stock. The host suite peaked near 144 W and 57°C **[M-owner]**. | 0 | 0 unless cores actually power-limit | 0; raising CPU limits can reduce shared package/IOD headroom **[P]** | Raised limits increase socket/VRM heat; no DRAM ECC benefit. Liquid cooling does not cool the IOD, VRM, or RDIMMs directly. | **Auto is right.** Not a memory-bandwidth lever. |
| **Curve Optimizer / Curve Shaper** | Auto/off; photographed magnitude 0 is a no-op **[M-owner]**. Explicit off. | 0 | At most core-compute side; not GMI | 0 | Undervolt instability can be silent; unrelated to ECC. | **Explicit no-op; leave off.** |
| **PBO scalar / boost override** | Auto/stock. Explicit Auto or Disabled/stock. | 0 | 0 for the measured bandwidth-bound path | 0 | More voltage residency/heat for no demonstrated limit. | **Auto is right.** |
| **CPPC** | OS proves enabled via `amd-pstate-epp`; Zen 5 EPYC analog defaults enabled **[M-owner/V-analog]**. Explicit Enabled if documenting. | 0 | Keeps responsive core boost; disabling may regress | 0/± interrupt service | No ECC risk; disabling can raise latency. | **Documenting no-op.** |
| **CPPC Preferred Cores** | Effective enabled (`prefcore`) **[M-owner]**. Explicit Enabled. | 0 | ± scheduler placement; leave on | ± NCCL/helper threads | No stability risk. | **Documenting no-op.** |
| **Determinism** | ASUS/TR Auto not publicly documented. Zen 5 EPYC vendor analog usually resolves Auto to Performance determinism; the existing guide's older claim that TR Auto is Power remains unverified. | 0 | Power may improve this individual CPU's boost; Performance reduces sample variance **[V-EPYC]** | 0 | Power can run nearer package thermal limit under compute; current workload is far below it. | **Leave Auto; resolve before making explicit.** Not a bandwidth lever. |
| **SMT** | Auto = Enabled, confirmed 32 threads **[M-owner]**. | 0 or ↓ under oversubscription | Existing recipe uses sibling threads; disabling likely regresses orchestration **[P]** | Could hurt NCCL/pollers | No ECC risk; small thermal/power redistribution. | **Keep Auto/Enabled.** Workload lever, not clock cleanup. |
| **Global C-states** | Auto = Enabled, C1/C2 visible **[M-owner]**. AMD says retain core C-states and disable C2 from Linux only for latency-sensitive networking **[V]**. | 0 steady-state | Possible first-burst jitter; kt threads spin, so expected 0 **[P]** | Can affect short-message/network latency | Disabled raises idle power/heat without clock stability benefit. | **Auto is right.** Test OS C2 first if needed. |

### 3.2 Fabric, SoC, and memory policy

| Group / setting | Auto resolution / explicit value | Per-CCD ceiling | kt layer latency | PCIe DMA / NCCL | Stability, ECC, thermals | Classification |
|---|---|---:|---:|---:|---|---|
| **DF C-states** | Boot 1 already explicit Disabled. AMD recommends Disabled for HPC to remove fabric wake latency **[V]**. | 0 steady-state | ↓ first-transaction jitter; serving result was neutral **[M-owner]** | May reduce burst DMA jitter | Higher IOD idle power; no ECC risk. | **Keep explicit Disabled as policy.** |
| **APBDIS** | Boot 1 already explicit `1`. Zen 5 EPYC analog: APBDIS=1 with DF P0 holds fabric/controllers full-power **[V-analog]**. | 0 once saturated | Could reduce ramp latency; serving neutral **[M-owner]** | Could reduce burst jitter | Higher idle SoC power; no clock overvoltage. | **Keep constant during Tier 3. Real latency lever, already exhausted.** |
| **SoC/Uncore OC Mode** | Auto. Help text says Enabled forces IF/memory maximum specified frequency **[M-owner BIOS]**. It overlaps APBDIS policy. | 0 if APBDIS already effective | Duplicate/no expected gain | Duplicate | Higher idle power; confounds the experiment. | **Leave Auto. Do not stack with APBDIS.** |
| **Power Profile** | Already Maximum IO Performance **[M-owner]**. | May preserve top fabric policy; no additional cleanup | ± | Best vendor-profile direction for IOD/DMA **[V-analog]** | Can trade core headroom for IOD headroom; thermals observed acceptable. | **Keep explicit; real policy lever already applied.** |
| **Memory Power Down** | Boot 1 already Disabled; Zen 5 EPYC platform default is commonly Enabled **[V-platform analog]**. | 0 sustained | Possible lower wake latency; serving neutral **[M-owner]** | 0 | More RDIMM idle power/temperature, no intrinsic ECC loss. | **Keep fixed during trials; optional revert if idle thermals matter.** |
| **MCLK / EXPO** | Auto = JEDEC 4800 on this board; EXPO = 6400 32-39-39-80 at 1.40 V **[M-owner]**. | 4800 caused −11% per CCD **[M-owner]**; 6600/6800 unmeasured. | 4800 caused +13% T=1 **[M-owner]** | Copy/triad fell 18% **[M-owner]**; DMA effect needs measurement | Above profile is OC and retraining risk; ECC detects DRAM faults but does not make OC safe. | **EXPO 6400 is the measured baseline. Higher MCLK is Tier 3 only.** |
| **UCLK ratio** | Unknown. Zen 5 Shimada Peak community evidence says selectable 1:1 is not supported **[C]**. | Potentially real if ratio can change | Real latency/controller lever | Could affect host DMA service | Wrong ratio may fail training or silently corrupt. | **Verify present/effective. Do not force blindly.** |
| **FCLK** | Auto running value unknown. | Principal GMI hypothesis | Should track expert reads if binding | Potential IOD/DMA effects | Non-ECC fabric errors possible; clock OC. | **Real Tier 3 lever; isolate first.** |
| **Primaries 32-39-39-39-80** | EXPO-resolved **[M-owner]**. Keep profile values. | Little expected change for long sequential streams | Small latency effect | Little | Tightening increases DRAM error/training risk. | **Auto/profile is right.** |
| **tRFC / secondary/tertiary timings** | Auto; actual trained values not recorded. | Refresh overhead may affect socket patterns, unlikely to lift narrow GMI | Small latency effect | Possible copy/triad effect | Tight tRFC is temperature-sensitive and can cause CE/UE. | **Auto is right. Record, do not tune.** |
| **GDM / Gear Down Mode** | Not established in the photographed inventory: **verify present**. | 0 | Minor command-latency/stability trade | 0 | Disabling can reduce command margin. | **Leave Auto; not justified.** |

### 3.3 Voltage cleanup

| Rail | Known Auto / profile state | Explicit recommendation | Performance effect | Risk / classification |
|---|---|---|---|---|
| **VSOC** | 0.924 V in BIOS monitor at the baseline **[M-owner]**. | **Auto** for 2100 and 2200 FCLK trials. | Voltage alone adds no bandwidth. | Manual VSOC stresses IOD/PHY; no TR 9000 daily-safe vendor ceiling found. |
| **VDDG CCD/IOD** | CCD high/low 0.903/0.863 V and IOD high/low 0.903/0.830 V **[M-owner]**. | **Auto**. Record resolved values after each boot. | Margin rail, not performance. | Fabric errors may be silent and outside DRAM ECC coverage. |
| **CLDO VDDP** | Auto; live value not recorded. | **Auto**. | Training/PHY margin only. | Poor values cause training failure or memory errors. |
| **VDDIO / MC** | Monitor showed VDDIO 1.092 V and VDD_11_S3/MC 1.312 V **[M-owner]**; naming/telemetry is board-specific. | **Auto/profile**. Do not equate displayed VDDIO with DRAM VDD/VDDQ. | Margin only. | Extra controller heat and degradation risk. |
| **DRAM VDD/VDDQ** | EXPO profile 1.40 V **[V profile/M-owner BIOS]**; SMBIOS 1.1 V is nominal metadata. | Keep the profile's 1.40 V at 6400; for 6600/6800, do not exceed the validated profile merely to pass. | Enables rate/timings; does not directly increase GMI. | RDIMM heat and aging; require thermal monitoring and zero CE. |

**Cleanup verdict:** most Auto fields should remain Auto. Make only already-proven policy choices explicit (DF C-states Disabled, APBDIS=1, Maximum IO Performance, RAS hygiene) and keep them fixed. Explicitly setting dozens of resolved defaults creates configuration debt without performance. The only untested real levers in scope are live clock resolution, FCLK, and—if Shimada Peak actually allows it—UCLK/MCLK behavior.

## 4. Safety, degradation signals, and rollback

### 4.1 Voltage boundaries: evidence is weak

AMD does not publish daily-safe manual VSOC, VDDG CCD/IOD, VDDP, or VDDIO/MC ceilings for Threadripper 9000 in the sources found. ASUS exposes the controls but its board manual does not specify safe values. Consequently the only defensible plan for this machine is **leave all these rails Auto and stop when clock-only testing fails**.

For context—not guarantees for Shimada Peak—AM5/Zen 5 overclocking communities commonly treat the following as conservative daily *discussion ranges*: VSOC at or below 1.20 V preferred, with 1.30 V often cited as an upper community limit; VDDG CCD/IOD at or below 1.00 V preferred and never above 1.10 V; CLDO VDDP around 1.00–1.10 V; VDDIO/MC no higher than 1.40 V when the rated memory profile itself requires it **[C, AM5-derived, not validated for TR 9000]**. These values are included only because a ceiling was requested; **they are not authorization to set them and not evidence that 9955WX tolerates them daily**. The project's stricter operational ceiling is the current Auto-resolved voltage; if 2100/2200 or 6600/6800 needs more, abandon that point.

Potential IOD/fabric degradation or insufficient margin appears as progressively lower formerly-stable FCLK, new cold-boot training retries, EDAC CE accumulation at previously clean memory settings, MCE/bus/interconnect errors, unexplained application NaNs, PCIe/AER/Xid events, or a need for rising voltage to reproduce the same clock **[C/engineering symptom list]**. A CE proves the configuration is not clean; it does not prove which rail or component caused it. A UE, persistent MCE, or loss of prior stability requires immediate return to baseline and retest—not compensation with voltage.

### 4.2 Rollback on this board

1. Before each trial, save the known-good Boot 1 EXPO profile and the trial profile separately; photograph only non-sensitive settings.
2. On an ordinary regression, load the known-good profile and cold boot.
3. If POST fails but setup remains reachable, use the motherboard **Safe Boot** button. ASUS documents that it temporarily applies safe settings while retaining overclocked settings so they can be repaired **[V]**.
4. To clear CMOS, ASUS documents: turn the computer off, unplug power, press the rear-I/O **CLR_CMOS** button, reconnect power, start, hold Delete, and re-enter BIOS settings **[V]**. If that fails, remove the onboard coin-cell battery, press CLR_CMOS again, then reinstall the battery **[V]**. The manual documents a rear button; no separate user CLR_CMOS header was found.
5. Never press CLR_CMOS during BIOS FlashBack. After a clear, load the baseline deliberately—do not assume any Auto resolves exactly as before.

Source: [ASUS Pro WS WRX90E-SAGE SE manual, sections 1.2, 2.8, and 2.9](https://dlcdnets.asus.com/pub/ASUS/mb/SocketsTR5/Pro_WS_WRX90E-SAGE_SE/E25489_Pro_WS_WRX90E-SAGE_SE_EM_V3_WEB.pdf).

### 4.3 Non-negotiable stop criteria

- Any new EDAC CE or UE, MCE, fabric/bus error, PCIe AER, or NVIDIA Xid **[project criterion]**.
- Any failed/repeated memory training, boot loop, or changed clock that does not read back as requested **[project criterion]**.
- Any NLL delta at or above 0.002, needle failure, NaN/nonfinite value, assertion, crash, or 20K-soak anomaly **[project criterion]**.
- Any thermal throttle/limit event or unexpected sustained rise in IOD, VRM, RDIMM, CPU, or GPU temperature **[project criterion]**. CPU liquid cooling is not evidence that RDIMMs/IOD/VRM are cool.
- More than 5% regression in per-CCD/socket bandwidth, kt one-layer latency, serving throughput, prefill, or PCIe/NCCL behavior **[project criterion]**. Nonmonotonic clock behavior is a rejection, not a prompt to stack more settings.

## 5. Expected payoff ceiling

The batch-1 baseline is a 35 ms decode step, of which the CPU expert tier is about 17 ms and the remainder about 18 ms **[M-owner]**. Assuming bandwidth is the only CPU-tier limiter, a fractional bandwidth gain \(g\) projects:

\[
t_{step}(g)=18+\frac{17}{1+g}\ \text{ms},\qquad
R(g)=\frac{1000}{t_{step}(g)}\ \text{tokens/s}
\]

| Bandwidth gain | Projected CPU tier | Projected decode step | Projected decode rate | Projected end-to-end gain |
|---:|---:|---:|---:|---:|
| Baseline | 17.00 ms **[M-owner]** | 35.00 ms **[M-owner]** | 28.57 tok/s **[D]** | — |
| +5% **[P]** | 16.19 ms **[P]** | 34.19 ms **[P]** | 29.25 tok/s **[P]** | +2.4% **[P]** |
| +10% **[P]** | 15.45 ms **[P]** | 33.45 ms **[P]** | 29.89 tok/s **[P]** | +4.6% **[P]** |
| +15% **[P]** | 14.78 ms **[P]** | 32.78 ms **[P]** | 30.50 tok/s **[P]** | +6.8% **[P]** |

This is an optimistic ceiling. It assumes perfect inverse scaling of the entire 17 ms CPU tier and no change in GPU work, scheduling, synchronization, power, or errors. A 5% bandwidth improvement is worth only about 0.68 tok/s projected; a 10% improvement is worth about 1.32 tok/s projected **[P]**. That bounded payoff argues for clock-only trials at Auto voltage and against long voltage/timing campaigns.

## 6. Recommended order and decision rule

1. **T3-0 measurement-only at EXPO 6400.** Record live FCLK/UCLK/MCLK idle and loaded, then refresh the short baseline.
2. **T3-F1: FCLK 2100, all voltages Auto.** This is the cleanest single-variable test of the narrow-GMI hypothesis.
3. If and only if T3-F1 passes the complete gate, run **T3-F2: FCLK 2200**, still Auto voltage.
4. Inspect UCLK controls. If a selectable ratio exists and live readback proves it changes, run the UCLK arm in separate boots. If it is absent or ignored—as the Zen 5 report predicts—do not fight the firmware.
5. Only if FCLK does not explain the ceiling, run fixed-FCLK **MCLK 6600**, then **6800**, one cold boot each, keeping profile timings/voltages and stopping rather than adding voltage.
6. Do not stack the winning FCLK and MCLK changes until each independently passes and the projected end-to-end value exceeds the operational cost. Preserve the lower-risk single lever when gains overlap.

## 7. Claims that could not be sourced conclusively

1. The exact Auto FCLK on a 9955WX under BIOS 1317 / AGESA 1.0.0.1e, and whether it changes between DDR5-4800 and DDR5-6400.
2. The exact UCLK and UCLK:MCLK ratio at DDR5-4800 and DDR5-6400 on a 9955WX.
3. Public AMD confirmation that Shimada Peak lacks selectable 1:1 UCLK:MCLK. The available evidence is community-posted AMD support/engineering correspondence about a 9960X, not an AMD publication.
4. Whether BIOS 1317 actually exposes `UCLK DIV1 MODE` for this CPU, and whether selecting it changes hardware. The photographed inventory only establishes the parent `Infinity Fabric Frequency and Dividers` menu; treat UCLK selection as **verify present**.
5. A public AMD/ASUS OPN table proving whether the 9955WX has GMI-Narrow or GMI-Wide. Narrow is inferred strongly from owner measurements.
6. A reproducible maximum stable FCLK distribution for TR 9000 or EPYC 9005. Community examples around 2100–2200 MHz are sparse, workload-specific, and sometimes nonmonotonic.
7. AMD-approved daily-safe manual ceilings for VSOC, VDDG CCD/IOD, CLDO VDDP, and VDDIO/MC on Threadripper 9000. The context ranges above are AM5-community-derived and must not be treated as Shimada Peak specifications.
8. The ASUS-specific Auto resolution of Determinism on this TR PRO firmware. EPYC OEM guidance is inconsistent with the existing guide's unsourced TR assumption, so Auto should remain Auto until observed.
9. A vendor-documented mapping between FCLK and per-CCD GMI transfer rate for the 9955WX. The 32 B/clock formula comes from third-party measurement/architecture analysis, albeit with an excellent fit to this machine.

## References

- AMD, [5th Gen EPYC Processor Architecture Overview](https://docs.amd.com/api/khub/documents/iZ5eCtQ5v6PyPN7wEd8HQg/content).
- AMD, [EPYC 9005 HPC Tuning Guide](https://docs.amd.com/api/khub/documents/v4Qq7nkUvpSfjc3bpPycCw/content).
- AMD, [EPYC 9005 BIOS & Workload Tuning Guide](https://docs.amd.com/v/u/en-US/58467_amd-epyc-9005-tg-bios-and-workload).
- AMD, [Shimada Peak platform white paper](https://www.amd.com/content/dam/amd/en/documents/products/processors/ryzen-threadripper/shimada-peak-me-vertical-white-paper.pdf).
- ASUS, [Pro WS WRX90E-SAGE SE manual, revised edition V3](https://dlcdnets.asus.com/pub/ASUS/mb/SocketsTR5/Pro_WS_WRX90E-SAGE_SE/E25489_Pro_WS_WRX90E-SAGE_SE_EM_V3_WEB.pdf).
- Linux kernel, [AMD HSMP interface](https://www.kernel.org/doc/html/next/arch/x86/amd_hsmp.html).
- Lenovo, [Tuning UEFI Settings for 5th Gen AMD EPYC](https://lenovopress.lenovo.com/lp2210-tuning-uefi-settings-5th-gen-amd-epyc-processor-servers) (vendor-platform analog; not ASUS defaults).
- Chips and Cheese, [AMD's EPYC 9355P: Inside a 32 Core Zen 5 Server Chip](https://chipsandcheese.com/p/amds-epyc-9355p-inside-a-32-core).
- Chips and Cheese, [Evaluating Uniform Memory Access](https://chipsandcheese.com/p/evaluating-uniform-memory-access).
- Level1Techs, [Threadripper 9000 missing UCLK DIV1](https://forum.level1techs.com/t/threadripper-9000-series-missing-uclk-div1-1-1-mode-but-why/239170) (community, including quoted private AMD correspondence).
- Level1Techs, [TR 9960X cores and memory bandwidth](https://forum.level1techs.com/t/tr-9960x-cores-and-memory-bandwidth-questions/249797) (community measurements).
- [`ryzen_smu`](https://github.com/leogx9r/ryzen_smu), [`ryzen_monitor`](https://github.com/hattedsquirrel/ryzen_monitor), and [`zenpower`](https://github.com/ocerman/zenpower) project documentation.

# BIOS / firmware optimisation map — ASUS Pro WS WRX90E-SAGE SE, Threadripper PRO 9955WX, 8 × DDR5-6400 RDIMM

Status: **map and protocol only. No firmware setting has been changed.** Every change below needs a separate, explicit owner approval for the exact change set, a saved profile, and the checklist in §5. Written 2026-09-22 from the E25 host-memory revalidation ([plan/EXPERIMENTS.md E25](../plan/EXPERIMENTS.md), evidence in `evidence/e25-dram-revalidation/`) and a full read of the ASUS TR5-series BIOS manual (§0).

Why this document exists: E25 confirmed directly that this host's CPU cores read DRAM at **125 GB/s of the 409.6 GB/s the eight DDR5-6400 channels can deliver**, and that the ceiling is **≈ 62–63 GB/s per CCD** — two threads on one CCD already saturate it, four threads split across the two CCDs already reach the socket ceiling. The DIMMs, the channel count and the memory clock are not the limit; the CCD→IO-die fabric is. That reframes which BIOS knobs can matter for the V4.1 hybrid (CPU expert tier, PCIe DMA, NCCL) and which cannot.

## 0. Sources

| Source | What it is | URL |
|---|---|---|
| ASUS E22761, *AMD TR5 Series BIOS Manual* (covers Pro WS WRX90E-SAGE SE) | The menu items, option lists and vendor descriptions every row below is mapped against; read in full | `https://dlcdnets.asus.com/pub/ASUS/mb/SocketsTR5/Pro_WS_WRX90E-SAGE_SE/E22761_AMD_TR5_Series_BIOS_manual_EM_WEB.pdf` |
| ASUS BIOS download / release notes page for this board | Firmware versions and their notes (BIOS 1317 dated 2026-02-03 is installed) | `https://www.asus.com/uk/supportonly/pro%20ws%20wrx90e-sage%20se/helpdesk_bios` |
| Level1Techs forum, "PSA: ASUS Pro WS WRX90E-SAGE SE BIOS 1317 errata" | Community report that `Refresh Interval` (tREFI) resets to 12643 on POST under 1317; **community report, unverified here** | `https://forum.level1techs.com/t/psa-asus-pro-ws-wrx90e-sage-se-bios-1317-errata/246753` |
| `scripts/bios-inventory.sh` → `evidence/e25-dram-revalidation/bios-inventory.txt` | Read-only sysfs/procfs inventory of what the OS sees (this is the "current value" column) | this repo |
| `docs/HARDWARE.md` § System memory | DIMM part numbers, SPD vs configured speed, channel population (from the owner-run `dmidecode`) | this repo |

Menu paths below are the manual's names. Where the manual lists an item under both *Ai Tweaker* and *Advanced → AMD Overclocking* / *AMD CBS*, the CBS path is the one AGESA reads last; set a knob in one place only.

## 1. Inventory (read-only, 2026-09-22)

No serials, UUIDs, asset tags or MAC addresses are recorded anywhere in this repo; `bios-inventory.sh` does not read them.

| Item | Observed | How |
|---|---|---|
| Board | ASUSTeK Pro WS WRX90E-SAGE SE, `Rev 1.xx` | sysfs DMI |
| BIOS | AMI, version **1317**, 2026-02-03 (release 13.17); EC firmware not readable unprivileged | sysfs DMI |
| CPU | AMD Ryzen Threadripper PRO 9955WX, 16 cores / 32 threads, **2 L3 instances = 2 CCDs** (cores 0–7 CCD0, 8–15 CCD1, SMT siblings 16–31), microcode `0xb008121`, max 5,452 MHz | `/proc/cpuinfo`, sysfs cache ids |
| NUMA | **1 node** → NPS1 in effect | `numactl -H`, sysfs |
| SMT | on | sysfs |
| Frequency policy | `amd-pstate-epp` active, governor `performance`, EPP `performance`, boost on, `prefcore` enabled | cpufreq sysfs |
| Idle | `acpi_idle`, C1 and C2 exposed (Global C-state control is therefore enabled in firmware) | cpuidle sysfs |
| Memory | 258,790,840 kB (≈ 246.8 GiB usable); 8 × Kingston KF564R32-32 32 GB DDR5 RDIMM 2R ECC, one per channel A–H; SPD 4800 MT/s, **configured 6400 MT/s** (an EXPO/XMP profile is active); running voltage not exposed by SMBIOS | HARDWARE.md (`dmidecode`) |
| Hugepages | `HugePages_Total=0`; THP `madvise`, defrag `madvise` | sysfs |
| ECC / EDAC | `mc0` `F1Ah_M08h` 256 GiB, **ce=0 ue=0**; `amd64_edac` + `edac_mce_amd` loaded | sysfs EDAC |
| IOMMU | on, 59 groups (`ivhd0–3`) | sysfs |
| GPUs | 2 × PCIe 5.0 x16 capable (idle at 2.5 GT/s — ASPM/dynamic speed), **BAR1 128 GiB each → Resizable BAR + Above-4G decoding enabled** | `lspci` via sysfs |
| NVMe | 2 × PCIe 5.0 x4 | sysfs |
| Socket power | `amd_hsmp` cap **350 W**; 73–144 W observed during E25 | hwmon |
| GPU power | 300 W default, 325 W max, 325 W set system-wide (T2.9) | `nvidia-smi` |
| Kernel perf access | `perf_event_paranoid=4` (uncore/DF counters unreadable without root) | sysfs |

Not observable from Linux without root MSR/SMU access: FCLK, UCLK:MEMCLK mode, DRAM timings and voltages, PBO/PPT/TDC/EDC values, Determinism mode, Power Profile Selection, APBDIS, prefetcher state, interleave/hash modes, patrol scrubber. §1.1 fills in what the owner's Boot 0 screen survey showed; the rest stays blank until the AMD CBS screens are photographed.

### 1.1 Boot 0 screen survey (owner photographs, 2026-09-22; photos kept outside git)

Read from 17 BIOS screen photographs (Main, Ai Tweaker and its sub-menus, Advanced, Tool, Server Mgmt). Values below are what the screens *display as set*; the AI Features prediction screen is excluded because it shows what AI Overclocking *would* apply, not what is applied.

| Screen | Setting | Value shown | Note |
|---|---|---|---|
| Main | BIOS / AGESA | **1317**, build 02/03/2026; **ComboSP6PI 1.0.0.1e**; EC MBEC-X690-0135 | confirms sysfs DMI |
| Main | CPU speed / memory | 5450 MHz (fused boost limit); 256 GB DDR5, Memory Frequency **6400 MHz** | |
| Ai Tweaker | Ai Overclock Tuner | **[EXPO]** — profile `DDR5-6400 32-39-39-80 1.40 V / 1.40 V` | resolves the "unverified 1.4 V" row in §3.1: DRAM VDD/VDDQ **1.40 V**, i.e. EXPO, not JEDEC 1.1 V |
| Ai Tweaker | Target CPU / DRAM | 4500 MHz / 6400 MHz; BCLK1 100.0000; PCIE Frequency Auto | |
| Ai Tweaker | **FCLK Frequency** | **[Auto]** (option range shown 800–3000 MHz) | Auto is the *policy*, not the running clock. Under AGESA Auto with MEMCLK 3200 (UCLK=MEMCLK) the fabric typically settles at 2000 MHz — **inferred, not measured**; §7 row 1 records it as unknown until read from a monitoring page or `ryzen_smu` |
| Ai Tweaker | CPU Core Ratio / Per-CCX / Core VID 0–1 / CCX0 Ratio (Rail0/Rail1) / Dynamic OC Switcher | all **Auto** | no manual core OC |
| Ai Tweaker | Performance Bias / AI Features / DIGI+VRM / CPU Core0, Core1, VSOC voltage | all **Auto** | no manual voltage |
| DRAM Timing Control | Primaries | **32-39-39-39-80** (tCL 32, tRCD 39/39, tRP 39, tRAS 80); every secondary/tertiary timing Auto | exactly the EXPO profile; tREFI not displayed on the photographed page, so the community "1317 resets tREFI" report stays unverified |
| Tweaker's Paradise | Spread spectrum, skews, RComp | all Auto | |
| Precision Boost Overdrive | PBO | **[Auto]**; Medium Load Boostit Auto; Peak Current Control Auto; Disable Current Limiter Auto; Per Core Power Offset **Disabled**; PBO Scalar Auto; CPU Fused Boost Clock Limit **5450**; Boost Clock Override Auto; Per-Core Boost Clock Limit Auto; Platform Thermal Throttle Limit Auto | stock limits; §3.5 "Current" updated |
| Curve Shaper | Min Frequency – Low Temperature | Enabled, sign +, magnitude **0**; all other 14 regions Auto | a zero offset — effectively off; nothing to revert |
| Advanced | — | menu only (USB4 item highlighted); **no AMD CBS values captured** | see gaps |
| Tool | BIOS Image Rollback | Disabled; ASUS User Profile menu present | slot 1 save **not confirmed** |
| Server Mgmt | BMC | firmware 2.01.30; OS Watchdog Timer Disabled | |

**Verdict:** the board is running one vendor profile (EXPO DDR5-6400 @ 1.40 V) on top of factory-Auto everything else. There is no manual overclock, undervolt, fabric or timing change to unwind before Tier 1 — the "baseline" is the EXPO profile plus AGESA Auto.

**Gaps that block Boot 1 (T1-d + T1-b) from being logged against a known "before" state** — none of the AMD CBS pages were photographed:

- Advanced → AMD CBS → **DF Common Options**: NUMA nodes per socket (NPS), Memory interleaving, ACPI SRAT L3 as NUMA domain.
- **SMU Common Options**: APBDIS, Power Profile Selection, DF PState Latency/Frequency Optimizer, DF Cstates, Determinism Control, TDP/PPT/TDC/EDC.
- **UMC Common Options → DDR Options**: Power Down Enable, **DDR RAS** (Patrol Scrubber, Redirect Scrubber, Corrected Error Counter), ECC mode, memory interleave/hash.
- **NBIO Common Options**: PCIe Speed PWM Control, IOMMU, ACS.
- **CPU Common Options**: SMT, Global C-state Control, prefetchers.
- The **actual FCLK** (Ai Tweaker only shows the Auto policy).
- Confirmation that **User Profile slot 1** was saved (Tool → ASUS User Profile).

The owner's photo folder also contains non-BIOS shots (terminal, `nvtop` under load, a Wi‑Fi module label with MAC/serial, case interior). That folder is listed in `.gitignore` and must never be committed.

## 2. What E25 says a BIOS change can and cannot buy

| Tier of bandwidth | Value | Source | A BIOS knob that moves it |
|---|---|---:|---|
| Theoretical DIMM/channel | 8 × 8 B × 6,400 MT/s = **409.6 GB/s** | arithmetic | Memory Frequency (but see the next rows — irrelevant while the CPU cannot reach it) |
| Socket-sustainable (all requesters) | **≥ 235 GB/s** demonstrated (CPU 121 + GPU DMA 57 + 57 at once, E25 gpu-dma); true ceiling unknown, only the GPUs reach past 125 | E25 / E4b | Memory Frequency, interleaving, Power Profile → IO |
| Reachable from the CPU cores | **125 GB/s read**, ≈ 78 write, ≈ 97 write-nt, ≈ 136 copy-nt; **62–63 GB/s per CCD** | E25 scaling / patterns | **FCLK** (per-CCD link runs at FCLK), APBDIS / DF P-states (fabric clock floor), Power Profile Selection |
| Effective for kt-kernel CPU experts | **≈ 120–127 GB/s** (E17 T=1 127; E25 expert-shaped 120–125 with per-layer sync) | E17 / E25 expert | same as the row above — kt is already at the CPU ceiling |

Consequences for the knob map:

- **Memory frequency and timings cannot raise CPU-side bandwidth here.** The socket has ≥ 110 GB/s of DIMM headroom above what the cores can pull (the GPUs prove it). Going 6400 → 4800 (option H2) may cost nothing CPU-side and buys ECC/thermal margin; it *would* cost the GPU-DMA + CPU combined figure only if 235 GB/s ever approached the DIMM limit (it does not at 4800: 307 GB/s theoretical).
- **The one knob with a mechanism to raise the CPU ceiling is the fabric clock (FCLK)** — each CCD's GMI link moves a fixed number of bytes per fabric clock. The measured 62–63 GB/s per CCD is consistent with a 32 B/clk read path at ≈ 2.0 GHz (*inferred*; FCLK is not readable from Linux here, and Threadripper/EPYC CCD link widths are not documented publicly in bytes/clk). If FCLK is currently 2000 MHz and runs stably at 2200, the read ceiling would scale ≈ +10 % (≈ 137 GB/s), which is a ≈ 5 tok/s change in the CPU floor. **This is overclocking (Tier 3, approval-gated), and the gain is bounded.**
- **The structural fix stays H1 (a CPU with more CCDs)**; no firmware setting adds CCDs.
- Latency-side knobs (C-states, DF P-states, APBDIS, Power Profile, Determinism) act on the **≈ 0.45 ms per-layer wait** that E21 measured, not on bandwidth. E25 ran with cores at 2.8–5.2 GHz and the socket at ≤ 144 W — power was never the limiter during a bandwidth run; whether the fabric down-clocks between kt bursts during real decode is unmeasured and is what Tier 1(b) tests.
- PCIe knobs act on the E22 streamed-prefill path (57 GB/s per card H2D, already at the Gen5 x16 practical limit) and on NCCL TP2 traffic; there is little to gain and a Gen4 fallback would cost 2× on prefill export.

## 3. Setting matrix

Legend — **Tier 1**: topology / performance-policy settings, no voltage or frequency change, safe to benchmark. **Tier 2**: memory frequency / timing changes *within the DIMM vendor's rated profiles* (JEDEC 4800 or the EXPO 6400 already in use). **Tier 3**: manual frequencies or voltages beyond a vendor profile (FCLK, timings, VDD/VDDQ/SoC) — **overclocking, explicitly approval-gated, ECC-relevant.** "BW" = generic DRAM bandwidth (E25 suite); "kt" = kt-kernel per-layer latency / CPU floor (E17/E21); "PCIe" = GPU DMA + NCCL. "Current" is blank where Linux cannot observe it.

### 3.1 Memory frequency, timings, voltage, training

Manual: Ai Tweaker (Ai Overclock Tuner, Memory Frequency, DRAM Timing Control, DRAM voltages); Advanced → AMD CBS → UMC Common Options → DDR Options.

| Setting (path) | Current | Vendor default | Candidates | Mechanism / expected effect (BW · kt · PCIe) | Risk · ECC · rollback | Tier |
|---|---|---|---|---|---|---|
| Ai Overclock Tuner (Ai Tweaker) | **[EXPO]** — DDR5-6400 32-39-39-80 1.40 V (screen, §1.1) | Auto (= JEDEC SPD, 4800) | Auto; EXPO I; EXPO II; Manual | Selects which SPD profile programs frequency, primary timings and DRAM voltage. **BW: none expected CPU-side** (E25: cores pinned at 125 by the fabric); socket-level combined CPU+DMA falls only if 235 GB/s were near the DIMM limit (it is not at 4800). kt: none. PCIe: none | Auto → 1.1 V JEDEC, cooler DIMMs, more ECC margin; requires retraining (one long POST). Rollback: reload saved profile | 2 (H2 test) |
| Memory Frequency (Ai Tweaker) | DDR5-6400 (via EXPO; screen) | Auto | DDR5-4800; DDR5-5600; DDR5-6400 | Same as above with the profile's timings kept; 5600 is an intermediate point if 4800 shows a CPU-side drop (would falsify "fabric-bound" — E25 says it will not) | Retraining; a frequency the DIMM is not rated for is Tier 3 | 2 |
| UCLK DIV1 MODE (Ai Tweaker) | — (not on the photographed pages) | Auto | Auto; UCLK=MEMCLK; UCLK=MEMCLK/2 | Memory-controller clock vs DRAM clock. UCLK=MEMCLK (1:1, 3200 MHz at 6400) is the low-latency mode; 1:2 halves controller clock. **kt: latency-side**, could move the per-layer wait a few %; BW: no CPU-side change expected (fabric-bound) | Auto already picks 1:1 when the SoC can hold it; forcing 1:1 at 6400 may need more SoC voltage → training failures. Rollback: profile | 3 |
| FCLK Frequency (Ai Tweaker) / Infinity Fabric Frequency and Dividers (AMD Overclocking) | **[Auto]** (screen; range 800–3000); running clock unknown, likely 2000 (inferred) | Auto (manual: Auto = FCLK follows MCLK rules) | 2000; 2100; 2200 | **The one knob with a mechanism to lift the 62–63 GB/s per-CCD ceiling** (bytes per fabric clock × FCLK). Expected +5 % per +100 MHz on CPU read BW and on kt's T=1 latency if the link, not the DIMMs, is the wall — E25 says it is. PCIe: DMA also crosses the fabric; may rise slightly | **Overclocking.** Instability shows as MCE/WHEA-equivalent, silent data corruption risk before a crash; ECC covers DRAM only, not fabric transfers. Needs the full §5 gate incl. NLL and NaN soak. Rollback: profile / Clear CMOS | **3** |
| DRAM Timing Control: tCL/tRCD/tRP/tRAS, Trfc1/Trfc2/Trfcsb, Refresh Interval (tREFI) | 32-39-39-39-80 primaries, all else Auto (screen); tREFI not shown | Auto (profile) | Auto | Timings change latency, not the fabric ceiling; kt's m=1 GEMV does long sequential reads where timings matter little. **Do not hand-tune.** Note the community report that BIOS 1317 resets tREFI to 12643 on POST (unverified here) — if the profile's value differs, that is a firmware bug to check on the Ai Tweaker screen, not a knob to tune | Tighter timings + ECC = corrected-error storms then UE. Tier 3 by definition | 3 |
| DRAM Refresh Rate (UMC: 3.9 µs / 1.95 µs) | — | Auto | Auto | Higher refresh rate halves the refresh interval for hot DIMMs — a *reliability* knob; costs a few % of BW | Set 1.95 µs only if DIMM temperatures exceed vendor limits (not measured; RDIMM SPD-hub thermal sensors are readable with root `dmidecode`/`i2c` — not done) | 1 (reliability) |
| DRAM VDD / VDDQ / VPP, VDDIO, SoC/Uncore voltage (Ai Tweaker) | VDD/VDDQ **1.40 V** from the EXPO profile (screen); VSOC, Core0/Core1 Auto; SMBIOS still reports the JEDEC 1.1 V nominal | Auto (profile) | Auto | Voltage does not move bandwidth; it is what a profile needs for stability | Manual voltage = overclocking, thermal and DIMM-life risk. **Never manual without approval** | 3 |
| Memory Context Restore (Ai Tweaker / CBS) | — | Auto | Enabled | Skips full DRAM training on warm boot → shorter POST. No runtime effect | Known on some AGESA versions to cause boot loops with 4 DPC / high-capacity kits; with 1 DPC generally fine. Rollback: disable | 1 (convenience) |
| DDR Training Runtime Reduction; DDR5 Nitro Mode; M_ORDERING | — | Auto | Auto | Training-time and controller-ordering knobs from the desktop OC menu. No known mechanism for our sequential-read workload | Nitro Mode trades latency for stability at high MT/s; leave Auto | 3 |
| Power Down Enable (UMC / Ai Tweaker) | — | Auto (enabled) | Disabled | DRAM power-down between accesses adds exit latency on the first access after idle. **kt: latency-side** — the CPU tier idles between the ≈ 40 per-layer bursts; disabling could shave µs per burst. BW: none | Slightly higher idle DRAM power; safe. Rollback: Auto | 1 |
| Memory Healing BIST; Memory Clear (UMC) | — | Disabled / Auto | leave | Boot-time repair / clear; no runtime effect | Long POST if enabled | — |

### 3.2 Channel population and interleaving

Manual: Advanced → AMD CBS → DF Common Options → Memory Addressing (Memory interleaving; ACPI SRAT L3 Cache as NUMA Domain); UMC Common Options (Chipselect Interleaving; Address Hash Bank / CS / Rm / Subchannel; BankSwapMode).

| Setting (path) | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| Physical population | 8 channels × 1 DIMM (1 DPC, 2R) — **already the full-bandwidth topology** | — | none | All eight channels interleave; there is nothing to add. E25's 235 GB/s combined shows the channels have headroom | — | — |
| Memory interleaving (DF) | — (Auto → channel interleaving across all 8 under NPS1) | Auto | Auto; Disabled | Disabled = each channel's address range contiguous → a single stream hits one channel (8× worse). **Never disable.** Recorded so it is not toggled by mistake | — | — |
| Chipselect Interleaving (UMC) | — | Auto | Auto | Interleaves the two ranks of each DIMM; helps bank-level parallelism for sequential streams | Leave Auto | — |
| Address Hash Bank / CS / Rm / Subchannel; BankSwapMode (UMC) | — | Auto | Auto | Hash functions that spread strided access across banks/subchannels. kt reads 21.8 MiB contiguous experts; hashing already handles it (E25 expert-chunk 124 GB/s = the ceiling) | Changing can create pathological strides; leave Auto | — |
| ACPI SRAT L3 Cache as NUMA Domain (DF) | disabled (1 NUMA node) | Disabled | Enabled | Exposes each CCD as a NUMA node so `numactl`/OpenMP can place threads per CCD **without changing memory interleaving**. Pure topology hint; memory stays interleaved. Lets kt / the suite bind "8 threads on CCD0" by node instead of by CPU list. Effect on BW: none by itself | Some schedulers treat NUMA nodes as memory-locality domains and may migrate less; harmless. Rollback: disable | 1 |

### 3.3 NUMA / NPS

Manual: Advanced → AMD CBS → DF Common Options → NUMA nodes per socket [NPS0 / NPS1 / NPS2 / NPS4 / Auto]; Advanced → AMD Overclocking → NPS Mode [Auto/NPS1/NPS2/NPS4].

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| NUMA nodes per socket | **NPS1** (1 node) | Auto (= NPS1) | NPS2; NPS4 | Splits the 8 channels into 2 or 4 interleave groups, each its own NUMA node. **Mechanism for us:** with NPS2, a thread pinned to CCD *n* reading memory on node *n* takes the shorter IO-die path and does not compete with the other CCD's traffic in the same interleave set. On EPYC this raises per-CCD local bandwidth and lowers latency a few %; on this 2-CCD part the CCD→IOD link is still the bottleneck, so **BW gain expected ≤ 5 %, possibly 0**. kt: kt-kernel is not NUMA-aware — its compact expert arena (136 GiB) is one allocation; under NPS2 half the experts would be remote to each thread unless the arena is interleaved (`numactl --interleave=all`), which returns to NPS1 behaviour. **PCIe: each GPU becomes local to one node; H2D from remote memory crosses the IOD — E4b-style additivity should be re-measured.** Engram page cache is allocated where the faulting thread runs | Software that assumes one node (our scripts, kt-kernel, SGLang's pinned staging buffers) may lose bandwidth if placement is wrong; NPS4 with 2 CCDs leaves nodes with no cores. Reversible, no training. Rollback: Auto | **1 (first trial)** — measure, expect to revert |

### 3.4 CCD / CCX / core topology and SMT

Manual: Advanced → AMD Overclocking → SMT Control, CCD Bit Map Down Core; Advanced → AMD CBS → CPU Common Options → SMT Control, Core Performance Boost, Global C-state Control, Prefetchers, Streaming Stores Control, AVX512.

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| SMT Control | **on** (32 threads) | Auto (on) | Disabled | E25: SMT siblings never help bandwidth (smt-all×32 118.6 vs 125; expert-rows 91.8 vs 124) and E17/E6 s0 show 32 kt threads are *slower*. But the recipe places SGLang's Python loops, NCCL polling and the Engram WILLNEED threads on the siblings while the 16 kt threads hold the cores (E21: 18–21 runnable threads). **Disabling SMT removes 16 hardware threads the engine uses** → likely a regression, not a gain. Test only with `KT_CPUINFER=14` and Engram threads reduced | Reversible, no training. Rollback: Auto | 1 (low priority; expect negative) |
| CCD Bit Map Down Core | all cores | Auto | none | Disables cores per CCD. No use case: bandwidth is per CCD, and fewer cores per CCD only lowers compute | — | — |
| Core Performance Boost | on (boost=1) | Auto (on) | keep | Off pins cores at base clock → slower export (E22: core-bound) | — | — |
| L1 Stream / L1 Stride / L1 Region / L2 Stream / L2 Up-Down prefetchers | — (on) | Auto (on) | keep on | Hardware prefetch is what lets 2 threads saturate a CCD link (E25 ccd0×2 = 62 GB/s). Turning any off cuts sequential BW | — | — |
| Streaming Stores Control | — (on) | Auto | keep | Non-temporal stores gave E25 write 78 → 97 GB/s and copy 115 → 136; kt's output writes are small, so mostly irrelevant, but never disable | — | — |
| AVX512 | on | Auto (on) | keep | kt-kernel's MXFP4 CPU kernels use AVX-512 (HARDWARE.md) | — | — |

### 3.5 Determinism, power and boost limits (PBO / PPT / TDC / EDC / TjMax)

Manual: Ai Tweaker → Precision Boost Overdrive [Auto/Disabled/Enabled/Manual/Enhancement] with PPT/TDC/EDC Limit, Platform Thermal Throttle Limit, Curve Optimizer; Advanced → AMD CBS → SMU Common Options → TDP Control, PPT Control, TjMax Control, Determinism Control [Auto/Power/Performance].

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| Determinism Control (SMU) | — | Auto (= Power determinism on TR PRO, i.e., run to the PPT) | Performance | Performance determinism clamps every part to the same guaranteed frequency curve (lower variance, lower peak); Power determinism lets this specific chip run to its power limit. **For a throughput box Power (default) is right**; Performance only lowers run-to-run variance at a cost | Reversible; no training | 1 (benchmark variance only; not recommended as a default) |
| PPT / TDC / EDC limits (PBO Manual or SMU) | socket cap **350 W** (hsmp); E25 peak 144 W; E21 decode rank-0 process ≈ 19 threads busy — socket power under decode not yet logged (T1.11 adds it) | Auto (350 W PPT class) | keep Auto | Raising limits above stock is PBO (Tier 3). The workload never approached 350 W in any host bench; raising buys nothing. **Lowering** (e.g., 280 W) is a thermal/PSU option if the 1,500 W budget ever binds with 2 × 325 W GPUs (HARDWARE.md power table) | Lower PPT = lower sustained clocks → slower export (core-bound). Reversible | 1 (only as a power-budget tool) |
| Precision Boost Overdrive | **Auto** (screen); Fused Boost Clock Limit 5450; Per Core Power Offset Disabled; Boost Override, Scalar, Thermal Throttle Limit Auto | Auto | Disabled (= stock limits) or Auto | PBO Enabled/Enhancement raises PPT/TDC/EDC and boost beyond warranty limits | **Tier 3.** Not needed: no host bench was power-limited | 3 |
| Curve Optimizer | Curve Shaper: one region Enabled with magnitude 0 (= no offset), rest Auto (screen) | Auto (off) | off | Per-core voltage-frequency offset; undervolting = OC territory; instability shows as silent errors before crashes | 3 | 3 |
| Platform Thermal Throttle Limit / TjMax Control | — | Auto (95 °C class) | keep | Lowering caps sustained clocks earlier; Tctl peaked 57 °C in E25 — thermal headroom is not a problem after the owner's three case fans | — | — |

### 3.6 CPPC, P-states, C-states, fabric and SoC power management

Manual: Advanced → AMD CBS → SMU Common Options → CPPC, CPPC Preferred Cores, APBDIS [Auto/0/1], Power Profile Selection [High Performance / Efficiency / Maximum IO Performance], DF PState Frequency Optimizer, DF PState Latency Optimizer, DF Cstates, BoostFmax, HSMP Support, xGMI (n/a single socket); CPU Common Options → Global C-state Control; NBIO → PCIe Speed PWM Control.

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| CPPC / CPPC Preferred Cores | enabled (`amd-pstate-epp` active, `prefcore` enabled) | Auto (on) | keep | Lets the kernel driver steer frequency; `performance` EPP already selected in the OS. Disabling would drop to `acpi-cpufreq` — worse control | — | — |
| Global C-state Control | enabled (C1/C2 exposed) | Auto (on) | Disabled | Deep core C-states add wake-up latency (tens of µs) to the first instruction after idle. **kt: the 40 per-layer waits are ≈ 450 µs each and kt threads spin, so the cores are not idle** — E21 measured run-queue wait ≤ 0.22 ms/step. Expected gain ≈ 0; the OS-side equivalent (`cpuidle.off=1` or `idle=poll`) can be tested on the kernel command line **without a BIOS change** and is preferred | Disabled = higher idle power (≈ +20–40 W socket), no stability risk. Reversible | 1 (test the kernel-cmdline equivalent first) |
| **APBDIS** (SMU) | — | Auto (= 0, Algorithmic Performance Boost enabled: the fabric/uncore down-clocks under light load) | **1** (disables fabric down-clocking → fixed high DF P-state) | **The knob most likely to touch the E21 per-layer latency without overclocking.** Decode alternates ≈ 40 short CPU bursts with GPU phases; if the data fabric drops a P-state between bursts, each burst starts at a lower fabric clock and pays a ramp. APBDIS=1 (+ DF PState Latency Optimizer) holds the fabric at its top P-state. Expected: kt per-layer wait −0 to −10 %; BW at steady state unchanged (E25 was steady-state); PCIe DMA latency also benefits | Higher idle SoC power (order 10–20 W); AMD documents this as the HPC/low-latency setting on EPYC. No training. Rollback: Auto | **1 (first trial)** |
| Power Profile Selection (SMU) | — | High Performance (Auto) | Maximum IO Performance | AMD's IO-optimised profile biases fabric clocks for PCIe/DMA traffic. **PCIe: H2D 57 GB/s per card is already at Gen5 x16 practical; gain likely ≤ 2 %.** May combine with APBDIS | Reversible | 1 (pair with APBDIS in the same trial group) |
| DF PState Frequency Optimizer / DF PState Latency Optimizer | — | Auto | Latency Optimizer enabled | Same mechanism as APBDIS from the other side: prefer the low-latency DF P-state table | Reversible | 1 (same group) |
| DF Cstates | — | Auto (on) | Disabled | Data-fabric C-states: fabric power gating when idle; adds wake latency to the first transaction | Reversible; higher idle power | 1 (same group) |
| BoostFmax | — | Auto | keep | Caps max boost; only useful to *lower* clocks for power | — | — |
| PCIe Speed PWM Control (NBIO) | — (GPUs idle at 2.5 GT/s → Dynamic) | Auto (Dynamic) | Static GEN5 | Dynamic link speed drops idle links to Gen1 and retrains on demand; a retrain is 10s–100s of µs on the first DMA after idle. **kt/E22: the streamed-prefill export and NCCL wake from idle every layer.** Static GEN5 removes the retrain latency; bandwidth unchanged | Slightly higher idle link power; no training; the same effect is available in the OS via `pcie_aspm=off` / `powersave` policy, so test the OS knob first | 1 (pair with the group above) |
| HSMP Support | on (`amd_hsmp` present) | Auto | keep | Exposes socket power/cap to Linux (`hwmon`); we read it | — | — |

### 3.7 PCIe Gen5, bifurcation, IOMMU, ACS, Above-4G, Resizable BAR, SR-IOV

Manual: Advanced → Onboard Devices Configuration → PCIEX16(G5) slot configuration [x16 / x8x8 / x4x4x4x4 (RAID)], PCIE Speed Control, PCIEX16/M.2/Chipset Link Mode [Auto/GEN1–GEN5]; Advanced → PCI Subsystem Settings → Above 4G Decoding, Re-Size BAR Support, SR-IOV Support; Advanced → AMD CBS → NBIO Common Options → IOMMU, ACS Enable, PCIe ARI Support, Ten Bit Tag, AER Cap.

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| Slot link mode / speed for the two GPU slots | Gen5 x16 capable on both (max 32 GT/s x16) | Auto | keep Auto (Gen5) | E4/E25: 57 GB/s H2D per card, 114 both — the Gen5 x16 practical ceiling. Any forced lower generation halves E22's export rate | Forcing GEN4 is a diagnostic only (if link errors appear). Reversible | — |
| Slot bifurcation (PCIEX16 Configuration) | x16 on both GPU slots | x16 | keep | Bifurcating a GPU slot would cut its lanes; only for riser/NVMe cards | — | — |
| Above 4G Decoding | **enabled** (BAR1 128 GiB mapped) | Enabled | keep | Required for 96 GB VRAM BARs and Resizable BAR | Disabling = GPUs fail to map | — |
| Re-Size BAR Support | **enabled** (BAR1 = 128 GiB ≥ VRAM) | Auto | keep | Full-VRAM BAR: lets the CPU and the peer GPU address all of VRAM; NCCL P2P over PCIe uses it. Also what makes the E22 pinned-staging path map directly | Disabling reverts to 256 MiB windows → NCCL falls back to host-staged copies (slower TP2) | — |
| IOMMU (NBIO) | **on**, 59 groups | Auto (on) | keep on; test `iommu=pt` in the OS | The IOMMU adds an address-translation step to every DMA. **PCIe: with the default DMA-API mode, large pinned transfers pay IOTLB misses; `iommu=pt` (pass-through, a kernel-cmdline change, not BIOS) gives identity mapping for host drivers while keeping the IOMMU on for VFIO.** E4's 57 GB/s suggests translation is not the bottleneck today, but NCCL's many small P2P writes are the case to measure | Turning the IOMMU off in BIOS removes DMA isolation and breaks `nvidia-peermem`-style checks; keep it on and test `iommu=pt` in the OS first (reversible via GRUB) | 1 (OS test first; BIOS unchanged) |
| ACS Enable (NBIO) | — (P2P works between the two GPUs today) | Auto | keep | Access Control Services force peer-to-peer TLPs through the root complex for isolation. With ACS enforced on the GPU root ports, GPU↔GPU P2P (NCCL TP2 all-reduce) is redirected; NVIDIA recommends ACS disabled on the P2P path. **Current NCCL P2P status was not captured — do it before touching this** (`nvidia-smi topo -p2p r` and `NCCL_DEBUG=INFO` in the next serve log) | Disabling weakens VFIO isolation (irrelevant: no VMs). Reversible | 1 (verify first) |
| PCIe ARI Support; Ten Bit Tag; AER Cap | — | Auto | Auto | Ten-Bit Tag raises outstanding-request tags → sustained Gen5 x16 throughput on long transfers; AER needed for ACS/error reporting. Auto enables what the endpoint supports | Leave Auto | — |
| SR-IOV Support | — | Disabled | leave | Only for virtual functions (NICs, GPU vGPU); the recipe runs bare-metal containers | Enabling costs BAR space; no benefit | — |

### 3.8 Thermal and fan control

Manual: Monitor → Q-Fan Configuration (per-header curves, CPU/chassis, Optimise All), Chassis Intrusion, temperature sources.

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| Q-Fan curves for the case headers | owner's (three fans added 2026-09-21; GPU idle 42 °C, sustained GLM 89 °C at 325 W) | Standard | Turbo / custom by a case-airflow sensor | GPU fan speed is the cards' own controller (T2.9: 34–73 %); case fans govern intake temperature. A curve keyed to the motherboard/PCH sensor rises with GPU exhaust heat | Noise only. Reversible | 1 (operational) |
| CPU fan curve | — | Standard | keep | Tctl ≤ 57 °C in E25 and ≤ 144 W socket; no CPU thermal issue | — | — |

### 3.9 ECC, RAS, scrubbing

Manual: Advanced → AMD CBS → UMC Common Options → DDR RAS: DRAM ECC Enable [Auto=Enabled], Data Poisoning, RCD Parity, Write CRC, Read CRC, UECC Retry, Memory Clear, Redirect Scrubber Control, Patrol Scrubber Control [Auto=Disabled], Corrected Error Counter; TSME (Transparent Secure Memory Encryption).

| Setting | Current | Vendor default | Candidates | Mechanism / expected effect | Risk · rollback | Tier |
|---|---|---|---|---|---|---|
| DRAM ECC Enable | **enabled** (EDAC `mc0` reports ce/ue counters) | Auto (Enabled) | keep | Single-bit correction; the guard for every Tier 2/3 memory trial. Never disable | — | — |
| Patrol Scrubber Control | — (Auto = Disabled per manual) | Auto/Disabled | **Enabled** | Background read-correct-write sweep of all DRAM; catches accumulated soft errors before they pair into a UE. Cost: a small steady background bandwidth draw (AMD default scrub rate is low, order 1 % or less) | Safe. Recommended **on** for a 247 GiB box that holds 136 GiB of expert weights for hours; the only reason to keep it off is to protect a bandwidth benchmark's last 1 %. Reversible | 1 (reliability; recommended) |
| Redirect Scrubber Control | — | Auto | Enabled | Re-writes a corrected line at the moment a CE occurs (demand scrub) | Safe | 1 (with patrol) |
| Corrected Error Counter | — | Auto | Enabled | Exposes a per-channel CE threshold counter; the OS EDAC `ce_count` we watch is the software view | Safe | 1 |
| Data Poisoning / UECC Retry | — | Auto | Auto | Poisoning marks an uncorrectable line so the consuming process dies rather than the whole system; standard server RAS | Leave Auto | — |
| Write CRC / Read CRC / RCD Parity | — | Auto | Auto | Link-level CRC on the DIMM bus and register parity; costs a little bandwidth when on. Auto follows the profile | Leave Auto unless CEs appear at 6400 (then enable as a diagnostic) | 1 (diagnostic) |
| TSME | — | Disabled | leave off | Memory encryption engine adds latency to every access | Off is right for this workload | — |

### 3.10 Workstation / HPC-specific knobs already covered elsewhere

| Knob | Where | Note |
|---|---|---|
| NPS mode, ACPI SRAT L3-as-NUMA | §3.2–3.3 | topology |
| APBDIS, DF P-state optimizers, Power Profile Selection, Determinism | §3.5–3.6 | the AMD "HPC tuning guide" set |
| PCIe Speed PWM Control static Gen5, Ten Bit Tag, ACS | §3.6–3.7 | I/O latency |
| Memory Context Restore | §3.1 | POST time on a 256 GB box |
| Q-Fan | §3.8 | airflow |
| BIOS Image Rollback, ASUS User Profile 1–8, EzFlash, CrashFree BIOS 3 | Tool menu | recovery — §5 |

Not applicable on this board/CPU (single socket): xGMI link settings, cross-socket interleave, IBS/preferred-IO socket selection.

## 4. What is *not* a BIOS problem (so it is not in the matrix)

- Per-CCD read bandwidth of 62–63 GB/s: silicon (2 CCDs). Only H1 (a 4-, 8- or 12-CCD SKU on this board) changes it.
- The E21 per-layer wait: kt-kernel's m=1 GEMV latency and the SGLang↔kt handshake; deferral (D=4) hides it. Firmware knobs in §3.6 are at most a second-order help.
- E22 export at 51.8 GB/s: core-bound (16 cores × ≈ 3.2 GB/s each). Clocks are already at boost; only more cores help.
- THP / hugepages: OS-side (`madvise` today; T1.6). E25 measured ≤ 1 % between 4 KiB and 2 MiB pages for streaming reads, so hugepages matter for TLB-heavy random access (Engram, page tables of a 136 GiB arena), not for bandwidth.

## 5. Conservative experimental protocol

Applies to **every** firmware change, including Tier 1. The owner performs every BIOS step (the agent never reboots or enters setup).

1. **Preserve the current state.** Tool → ASUS User Profile → save to slot 1, named with the date and "baseline-1317". Photograph every screen under Ai Tweaker, Advanced → AMD CBS (CPU, DF, UMC, NBIO, SMU) and AMD Overclocking; store the photos outside the repo (they may show serials). Run `bash scripts/bios-inventory.sh > evidence/bios-trials/<date>-baseline.txt` and `sudo dmidecode -t memory` (owner) for the DIMM configured speed and voltage line.
2. **Baselines on the unchanged system** (once, already partly done): `python3 scripts/bench-dram-suite.py --only scaling,patterns,expert` (E25 = the baseline); `--gpu-dma 0,1` for the DMA-concurrency row; `scripts/test-kt-one-layer.py --bench-cold` T=1; a hybrid serve with C1/C3/C4 (`bench-decode.py --distinct-prompts`) and the E22 32K prefill; `nll-score.py` against `nll-d0.json`; `verify.py --context 16384 --needles 3`; `journalctl -k | grep -iE 'mce|edac|ecc|xid'` count; EDAC `ce_count`/`ue_count`; `nvidia-smi topo -p2p r`.
3. **One logical group per trial.** Change only the settings listed for that trial; write them to a new User Profile slot (2–8) *before* saving and exiting; cold boot (power off, 10 s, on) when the change touches memory frequency, timings, voltage, FCLK/UCLK, NPS or interleaving (training and address maps are rebuilt); a warm reboot is enough for SMT, C-states, APBDIS, Power Profile, PCIe link policy, scrubber, fans.
4. **Immediately after POST:** confirm the machine trained (POST completed; `dmidecode` shows the intended configured speed), `bios-inventory.sh` diff against baseline shows only the intended deltas, `journalctl -k -b` has zero new `mce`/`edac`/`ecc` lines, EDAC `ce_count`/`ue_count` unchanged, `numactl -H` shows the intended node count.
5. **Memory integrity/stress before any application run** for Tier 2/3: `memtester` on ≥ 200 GiB or `stressapptest -M 200000 -s 1800` (30 min), then re-read EDAC. A corrected-error count > 0 that was 0 before is a **stop** for that trial (Tier 2/3) or a **note** (Tier 1 — nothing in Tier 1 should touch the DRAM path; if CEs appear, the change is not what it seemed).
6. **Application gates**, all of them, in this order: E25 suite (`--only scaling,expert`, 3 iterations) → kt one-layer cold T=1 → serve + C1/C3/C4 + 32K prefill → `nll-score.py` (|ΔNLL| < 0.002 vs `nll-d0.json`) → `verify.py` 16K → ≥ 20K-token mixed C1–C4 distinct-prompt soak with zero NaN/assert/Xid (the E24 release gate). Log socket power (`hwmon amd_hsmp`), Tctl, per-core clocks (the suite samples them) and GPU power/temperature (`bench-power-sweep.sh` sampler) throughout.
7. **Record** in `evidence/bios-trials/<date>-<group>/` (suite JSON, kt bench, bench-decode JSON, nll JSON, inventory diff, journal excerpt) and one row in the trial log in §7. Update the E25 tiers table only if a number moves outside its E25 min–max spread.
8. **Rollback.** Tool → ASUS User Profile → load slot 1 → save and exit → cold boot → step 4. If the board does not POST after a memory/fabric change: wait for the automatic retrain / safe-mode POST (the board retries with defaults after failed training), otherwise **Clear CMOS** (the `CLR_CMOS` header/button per the board manual) and reload slot 1 (profiles survive a CMOS clear only if stored in the BIOS flash region — re-enter by hand from the photographs if not). A corrupted BIOS image is recovered with **CrashFree BIOS 3** (USB stick with the renamed BIOS file per the manual) or **BIOS Image Rollback** (Tool menu) to the previous version.

### Stop criteria (abort the trial, roll back, record the reason)

| Signal | Threshold | Where to look |
|---|---|---|
| Corrected ECC errors | any increase in `ce_count` during Tier 2/3 stress or the soak | `/sys/devices/system/edac/mc/mc0/ce_count`, `journalctl -k` |
| Uncorrected ECC / MCE | **any** (also a reason to stop experimenting until root-caused) | `ue_count`, `mcelog`/`rasdaemon`, `journalctl -k -p err` |
| Failed training / no POST / safe-mode POST | once | POST screen, `dmidecode` speed ≠ intended |
| Thermal throttling | CPU Tctl ≥ 90 °C, or sustained package clock below base under load; GPU thermal flag samples > 0 in `bench-power-sweep.sh` | suite clock column, `nvidia-smi -q -d PERFORMANCE` |
| Power | PSU input approaching 1,400 W (`corsairpsu` hwmon) | hwmon |
| NaN / assert / Xid | any, in any gate run | server log, `journalctl -k \| grep -i xid` |
| Quality | \|ΔNLL\| ≥ 0.002 or any needle miss | `nll-score.py --compare`, `verify.py` |
| Performance regression | E25 suite read median below 121 GB/s (E25 min was 121.4) on the split×16 row, or C1 wall tok/s down > 3 % vs the same-day baseline | suite table, bench-decode JSON |

## 6. Staged tiers and the first recommended trial set

**Tier 1 — topology / NUMA and performance-policy settings. No voltage, no frequency. Safe to run after §5 step 1–2.**

First trial set (recommended order; each is one profile slot and one boot):

| Trial | Settings (exact) | Expected | Falsifies / confirms | Gate subset |
|---|---|---|---|---|
| **T1-a** NPS2 topology | AMD CBS → DF Common → NUMA nodes per socket = **NPS2**; also ACPI SRAT L3 Cache as NUMA Domain = **Enabled** (cold boot) | `numactl -H` shows 2 nodes with 8 cores each; E25 suite run twice: default placement, then `numactl --cpunodebind=0 --membind=0` for the ccd0 rows and `--interleave=all` for split rows. CPU read ceiling: 125 → 125–131 GB/s (≤ 5 %); per-CCD local read 63 → 63–66 | Whether NUMA-local access moves the CCD ceiling at all; whether kt-kernel (not NUMA-aware) loses bandwidth under NPS2 without `--interleave=all` (then NPS2 is rejected unless kt gets NUMA placement) | steps 4, 6 (suite, kt one-layer, C1/C4 serve) — **skip the 20K soak; NPS2 is not a release candidate until it wins** |
| **T1-b** fabric / IO latency policy | SMU Common → **APBDIS = 1**, **DF PState Latency Optimizer = Enabled**, **DF Cstates = Disabled**, **Power Profile Selection = Maximum IO Performance**; UMC → **Power Down Enable = Disabled** (warm reboot) | Steady-state BW unchanged (125 ± 2); kt T=1 cold latency 1.08 ms → 1.0–1.08; E21 per-layer wait median 470 µs → 420–470; C1 wall 36.0 → 36–39; socket idle power +10–30 W | Whether fabric/DRAM P-state ramps contribute to the per-layer wait at all. If C1 and the step profile do not move, the wait is entirely kt/handshake and this group is reverted (idle power) | steps 4, 6 incl. `step-profile.py`/`gap-context.py` on a trace, NLL, 16K verify; soak only if adopted |
| **T1-c** PCIe idle policy | NBIO → **PCIe Speed PWM Control = Static GEN5**; leave IOMMU and ACS as they are (warm reboot). *Precondition:* first try the OS equivalent (`pcie_aspm=off` / `powersupersave` off) via GRUB and measure; only do the BIOS change if the OS knob cannot hold Gen5 at idle | `lspci` shows 32 GT/s at idle; E22 32K prefill TTFT and per-chunk export time unchanged or −1–2 %; NCCL all-reduce latency in a `nccl-tests`/serve trace slightly lower | Whether link retrain latency is visible in the E22 per-chunk fixed cost (≈ 4.6 s — almost certainly not) | E22 prefill bench, C1/C4 |
| **T1-d** RAS hygiene (adopt regardless of performance) | UMC DDR RAS → **Patrol Scrubber Control = Enabled**, **Redirect Scrubber Control = Enabled**, **Corrected Error Counter = Enabled** (warm reboot) | E25 split×16 read within spread (≥ 121); EDAC counters stay 0 | Costs ≤ 1 % BW; makes the ECC signal in every later trial trustworthy | suite, step 4 |

Not in the first set (Tier 1 but expected negative or already testable in the OS): SMT off, Global C-states off (`cpuidle.off=1` first), Determinism = Performance, IOMMU pass-through (`iommu=pt` in GRUB first), ACS off (capture `nvidia-smi topo -p2p r` first).

**Tier 2 — memory frequency/timing within vendor-rated profiles** (after Tier 1 is recorded): **T2-a** Ai Overclock Tuner = Auto (JEDEC 4800, 1.1 V) — the H2 test. Prediction from E25: CPU read stays 125 ± 2, per-CCD 63, kt T=1 unchanged, combined CPU + 2-GPU DMA stays ≈ 235 (well under 307 GB/s theoretical at 4800). If confirmed, 4800 becomes a legitimate default (cooler DIMMs, JEDEC voltage, ECC margin) with **no performance cost**; if CPU read drops, FCLK follows MCLK on this AGESA and the fabric coupling matters — which would make T3-a more interesting, not less. Cold boot, full §5 incl. 30-min stress.

**Tier 3 — manual frequencies/voltages. Overclocking. Requires a separate explicit approval naming the exact values.** **T3-a** Infinity Fabric Frequency and Dividers = 2100, then 2200 MHz (UCLK/MEMCLK Auto, voltages Auto), each with the full §5 protocol including 30-min memory stress and the 20K soak. Expected ceiling change +5 % per +100 MHz (≈ 131 / 137 GB/s read) **if** the per-CCD link is the wall (E25 says so) and the SoC holds the clock without added voltage. Any CE, MCE or training failure ends the tier. Never combine with manual DRAM timings or voltages. Not recommended as a standing default even if stable: the gain (≈ 5 tok/s on the CPU floor) is smaller than the H1 upgrade and lives on a fabric with no ECC.

## 7. Trial log

| Date | Trial | Profile slot | Settings changed | Result | Kept? | Evidence |
|---|---|---|---|---|---|---|
| 2026-09-22 | baseline (no change) — Boot 0 survey | 1 (save **not yet confirmed**) | — | BIOS 1317 / AGESA 1.0.0.1e; EXPO DDR5-6400 32-39-39-80 @ 1.40 V; FCLK Auto (running clock unknown); PBO Auto, all core/SoC voltages and timings Auto; AMD CBS pages not yet photographed. E25 numbers; EDAC ce=0 ue=0 | — | `evidence/e25-dram-revalidation/`; §1.1 |

## 8. Owner checklist before the first trial

- [x] Photograph the Ai Tweaker screens (§1.1, done 2026-09-22).
- [ ] Save User Profile slot 1 and confirm it (Tool → ASUS User Profile).
- [ ] Photograph Advanced → AMD CBS: DF Common, SMU Common, UMC → DDR Options (incl. DDR RAS), NBIO Common, CPU Common — these are the pages Boot 1 changes.
- [ ] Read the running FCLK (a BIOS monitoring page, or `ryzen_smu` with root) so T3-a has a measured starting point.
- [ ] Run the owner-only baselines: `sudo dmidecode -t memory` (configured speed + voltage line), `sudo journalctl -k | grep -icE 'mce|edac|ecc|xid'`, `nvidia-smi topo -p2p r`.
- [ ] Confirm nothing is serving (`docker ps`), GPUs idle, no active benchmark.
- [ ] Approve the exact trial (T1-a … T1-d as written, or an edited list). The agent then prepares the measurement commands; the owner performs the BIOS change and reboot.

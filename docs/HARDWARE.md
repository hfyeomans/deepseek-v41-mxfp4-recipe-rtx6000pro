# Host hardware and the numbers that matter for feasibility

Everything marked *measured* comes from [`scripts/hardware-inventory.sh`](../scripts/hardware-inventory.sh) on this host and is committed as [`evidence/hardware-inventory.json`](../evidence/hardware-inventory.json). Everything marked *vendor spec* is from the manufacturer's datasheet and has **not** been measured here yet; the microbenchmarks in [`plan/EXPERIMENTS.md`](../plan/EXPERIMENTS.md) (E1–E4) exist to replace them with measurements. *Derived* means arithmetic from the two.

## Platform

| Item | Value | Source |
|---|---|---|
| Motherboard | ASUS Pro WS WRX90E-SAGE SE | measured |
| BIOS | 1317 (2026-02-03) | measured |
| Chipset / socket | AMD WRX90, sTR5 | vendor spec |
| PCIe topology | 7 × PCIe 5.0 x16 slots (128 CPU lanes); 4 × M.2 PCIe 5.0 x4; SlimSAS | vendor spec |
| Memory topology | 8 DIMM slots, 8-channel DDR5 RDIMM (ECC) | vendor spec |
| OS / kernel | Ubuntu 26.04.1 LTS / 7.0.0-31-generic | measured |
| Container runtime | Docker 29.8.1 with CDI GPU devices | measured |

Why it matters: the board has enough PCIe 5.0 x16 slots and CPU lanes to take **four** RTX PRO 6000 cards at full x16 without a switch, which is what Option A in [FEASIBILITY.md](FEASIBILITY.md) needs. Each GPU's own x16 link is the ceiling for the PCIe-streaming fallback and for layerwise GPU prefill in the hybrid (Option B).

## CPU

| Item | Value | Source |
|---|---|---|
| Model | AMD Ryzen Threadripper PRO 9955WX, Zen 5 | measured |
| Cores / threads | 16 / 32 | measured |
| Max clock | 5,452 MHz | measured |
| L3 | 64 MiB (2 × 32 MiB) | measured |
| ISA | AVX-512 (F, BF16, VNNI); **no AMX** | measured |
| NUMA | 1 node (Threadripper PRO presents a single node) | measured |

Why it matters: in the 2-GPU three-tier layout these 16 cores **compute 42% of the routed experts** (≈101 expert activations, 3.6 GMAC per decoded token) from DRAM, plus the Engram host-table gathers. KTransformers' `MXFP4` CPU method needs only AVX2, so the lack of AMX rules out KT's AMX-optimised INT8/BF16 paths but not the hybrid itself. Peak FP32 with two 512-bit FMA units per core is roughly 2 TFLOP/s across 16 cores. **Measured (E1b, E6 s0): these 16 cores read DRAM at only ≈125 GB/s** (30 % of the 409.6 GB/s the eight channels can deliver), and a bf16 expert GEMV sustains 94 GB/s of that — bandwidth, not FLOPs, binds the CPU side. The cause is structural: 16 Zen 5 cores are **two CCDs**, and each CCD's link to the IO die caps reads at roughly 60–65 GB/s. E4b confirmed the DIMMs themselves have headroom (GPU DMA adds another 114 GB/s on top). **E25 (2026-09-22) measured the per-CCD limit directly with a pinned C/OpenMP suite: 62–63 GB/s from one CCD in every access pattern (reached with 2 threads, flat to 8), 125 GB/s with threads split across both CCDs, 235 GB/s socket total with both GPUs' DMA engines reading concurrently; no clock, power or thermal limiting during the runs** ([`evidence/e25-dram-revalidation/`](../evidence/e25-dram-revalidation/), [BENCHMARKS](BENCHMARKS.md) "E25: the four bandwidth tiers"). For the hybrid this means the CPU upgrade that matters is **CCD count** (9975WX 32-core = 4 CCDs, 9985WX 64-core = 8), not memory speed. The only firmware lever on the per-CCD link is Infinity Fabric clock (an overclocking setting; approval-gated Tier 3 in [BIOS-OPTIMIZATION-GUIDE](BIOS-OPTIMIZATION-GUIDE.md)).

## System memory

| Item | Value | Source |
|---|---|---|
| Installed | 246.8 GiB usable (`MemTotal` 258,790,840 kB) ≈ 256 GB nominal | measured |
| Kit | 8 × Kingston FURY Renegade Pro **KF564R32-32**, 32 GB DDR5 RDIMM, dual-rank (2R), 80-bit (ECC, multi-bit), 1.1 V configured; one DIMM in each of channels A–H (`P0 CHANNEL A`…`H`, all `DIMM 0`) | measured (`sudo dmidecode -t memory`, 2026-09-20) |
| SPD / JEDEC speed | 4,800 MT/s (`Speed: 4800 MT/s`) | measured |
| **Configured speed** | **6,400 MT/s** (`Configured Memory Speed: 6400 MT/s`) — the EXPO/XMP 6400 profile is active, not the JEDEC default | measured |
| Rated profile | Kingston: JEDEC DDR5-4800 CL40-39-39 @ 1.1 V; EXPO/XMP profile DDR5-6400 CL32-39-39 @ 1.4 V | vendor spec — CAS latency is not exposed by SMBIOS; SMBIOS reports the JEDEC 1.1 V, so the running voltage is unverified |
| Channels | 8 (WRX90 octa-channel), all populated, one DIMM per channel (1DPC — the configuration that allows 6400) | measured |
| Theoretical bandwidth | 8 ch × 64 bit × 6,400 MT/s ÷ 8 = **409.6 GB/s** | derived |
| Achieved copy bandwidth (E1) | **146 GB/s** read+write aggregate with 32 processes; 140 GB/s with 16; 60 GB/s single process. 36 % of theoretical; a Python-memcpy lower bound, not STREAM | measured — [`evidence/e1-dram-bandwidth.json`](../evidence/e1-dram-bandwidth.json) |
| STREAM-style suite (E25, C/OpenMP, pinned) | read **125.0**, write 77.8, write-nt 96.8, copy 115.1, copy-nt 136.4, triad 121.7 GB/s at 16 threads split across CCDs; **one CCD 62–63 GB/s** in every pattern; CPU + both GPUs' DMA **235 GB/s** concurrently | measured — [`evidence/e25-dram-revalidation/`](../evidence/e25-dram-revalidation/) |
| Fabric clocks and controller-side bandwidth (HSMP, Boot 3) | **FCLK 2000 MHz, MCLK 3200 MHz (DDR5-6400)**, constant idle and under load; memory-controller counter **174 GB/s peak** (42 % of 410) during the copy phase while cores read ≤ 125 → the 125 GB/s CPU ceiling is the two CCD↔IOD links (2 × 32 B/clk × 2.0 GHz = 128), not the DIMMs. JEDEC 4800 (Boot 2) cost −11 % per CCD; EXPO 6400 is the standing setting. UCLK not observable | measured — `scripts/hsmp-clocks.py`, [`evidence/bios-trials/2026-09-23-boot3-expo-restored/`](../evidence/bios-trials/2026-09-23-boot3-expo-restored/) |
| Random 264 B row gather (E3) | **4.15 M rows/s** from 16 Python threads on a 4.9 GiB table; 48-row batch p99 0.015 ms | measured — [`evidence/e3-engram-gather-dram.json`](../evidence/e3-engram-gather-dram.json) |
| Huge pages | `HugePages_Total=0`; THP `madvise` | measured |

Why it matters:

- The full FP8 Engram tables (188.83 GiB) fit in DRAM with 34 GiB to spare after a 24 GiB OS reserve. On the 128 GB box that produced the only 4-GPU result, they did not; that recipe had to read Engram from NVMe with a 64 GiB DRAM cache. This host can run SGLang's `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` path with the *whole* table in DRAM — the one place this machine is better positioned than the published reference.
- SGLang's host-table path wants huge pages. Reserving ~190 GiB of 2 MiB or 1 GiB hugepages is a boot-time (`hugepagesz=1G hugepages=190`) or runtime (`vm.nr_hugepages`) change; it is reversible and is task T2.2. **Read from the `dev-dsv41` image (2026-09-21):** the host table does *not* need reserved hugetlb pages — `_HostTable` uses `madvise(MADV_HUGEPAGE)` on an anonymous mapping (`private` layout) or a memfd (`shared`), i.e. transparent huge pages. `choose_layout("auto")` picks `shared` only when `/sys/kernel/mm/transparent_hugepage/shmem_enabled` allows THP; on this host it is `never` while `enabled` is `madvise`, so auto → `private`, which does get THP. Leave `shmem_enabled` alone unless a multi-process layout needs `shared`; then `echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled` (runtime, reversible). Either way the table's THP allocation competes with page cache — drop the download's cache first (`echo 1 | sudo tee /proc/sys/vm/drop_caches`) or set `SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE=1`.
- Decode Engram traffic is trivial: 48 random 264 B rows per token ≈ 12.4 KB. Prefill of 64K tokens is ≈ 3.1 M random row reads; at DRAM random-access latency (~80–120 ns, parallel across cores) that is well under a second. E3 measures it.
- The 2-GPU three-tier layout (Option B) puts 112 GiB of routed experts in DRAM for CPU compute. With the full FP8 Engram also in DRAM that is 325 GiB — does not fit; with Engram on NVMe behind a 64 GiB row cache it is 200 GiB against 246.8 (+47 GiB), with FP4 tables 233 GiB (+13). **DRAM read bandwidth is the decode bottleneck of this layout**: 1.75 GiB of expert weights per token, 15.0 ms at the 125 GB/s the CPU cores can read (E1b), 20 ms at the 94 GB/s they sustain while computing (E6 s0), 4.6 ms at the 409.6 GB/s theoretical. The CPU cannot reach the DIMMs' full rate (two-CCD fabric limit) but the GPUs' DMA engines can add 114 GB/s on top of the CPU's reads (E4b), so the hybrid should feed DRAM-resident experts through both paths (`feasibility-calc.py` `split ceiling`).

## GPUs

| Item | Value | Source |
|---|---|---|
| Model | 2 × NVIDIA RTX PRO 6000 Blackwell **Max-Q** Workstation Edition | measured |
| Architecture | Blackwell GB202, SM 12.0 (compute capability 12.0) | vendor spec |
| VRAM | 97,887 MiB = 95.59 GiB each; 191.19 GiB total | measured |
| Memory bus | 512-bit GDDR7 ECC; `clocks.max.memory` 14,001 MHz → 28 Gbps PAM3 → 1,792 GB/s theoretical per card; **≈1,470 GB/s measured D2D copy (E2, read+write counted), 82 %** | measured |
| Power | 300 W limit, 325 W max (Max-Q); the full Workstation Edition is 600 W | measured |
| Max SM clock | 3,090 MHz | measured |
| PCIe | Gen 5 x16 both cards; **pinned H2D 53.1 / 52.9 GiB/s (57 GB/s) per card, 106.1 GiB/s (114 GB/s) both concurrently** (E4); **P2P read: OK** both directions; topology NODE (same CPU, no NVLink) | measured |
| Driver / CUDA | 615.71.09 / CUDA 13.4 UMD | measured |
| FP4 tensor throughput | ~2 PFLOPS dense FP4 spec for the 600 W card; **1,060 TFLOPS measured** on a 4,096×4608×5120 MXFP4 grouped GEMM at the 300 W Max-Q cap (E13 s1) | vendor spec / measured |

Why it matters:

- Weight capacity is the whole story: 95.6 GiB per card versus ~140 GiB per card needed for a 2-GPU split of the non-Engram weights. See the placement table in FEASIBILITY.md.
- PCIe 5.0 x16 is 63 GB/s theoretical per direction; measured pinned host-to-device is 53 GiB/s per card and 106 GiB/s for both together (E4). That sets the ~46 tok/s ceiling of the PCIe-streaming fallback (versus ~50 tok/s for CPU compute at the 125 GB/s the cores can read) and ≈1.06 s per 4,096-token chunk of layerwise GPU prefill in the hybrid. Because GPU DMA and CPU reads of host DRAM were measured as additive (E4b, 234 GB/s combined), the two paths can serve DRAM-resident experts simultaneously (Option B2, ~78 tok/s ceiling).
- SM120 is consumer/prosumer Blackwell: block-scaled FP4 `mma.sync` exists, but there is no TMEM/tcgen05 and shared memory is 99 KiB per block (228 KiB on B200), so datacenter FP4 kernels do not run here and every engine's default MoE path currently falls back to Marlin W4A16 dequant on these cards (RESEARCH.md, SM120 kernel status). Custom SM120 kernels are the kernel track in the plan (E12/E13).
- P2P OK means TP2 all-reduce runs over PCIe peer copies rather than bouncing through host memory. vLLM SM120 reports still sometimes need `--disable-custom-all-reduce`.
- Max-Q at 300 W: every published number from 600 W cards (or 0xSero's 275 W-capped cards) is a different power condition. Do not transfer throughput claims between machines.

## Storage

| Item | Value | Source |
|---|---|---|
| nvme0 (root, HF cache) | Samsung SSD 9100 PRO 4 TB, firmware 0B2QNXH7, PCIe 5.0 x4 (32 GT/s) | measured |
| nvme0 free | 1.7 TB of 3.6 TB | measured |
| 9100 PRO 4 TB sequential | 14,800 MB/s read / 13,400 MB/s write | vendor spec |
| 9100 PRO 4 TB random 4K | 2,200K IOPS read / 2,600K IOPS write (QD32 T16) | vendor spec |
| nvme1 | Crucial T705 4 TB (CT4000T705SSD3), firmware PACR5111, PCIe 5.0 x4 | measured |
| T705 4 TB | 14,500 MB/s read / 12,700 MB/s write; 1,550K / 1,800K IOPS | vendor spec |
| 9100 PRO Engram-style gather (E3) | **403K** random 4 KiB O_DIRECT reads/s from 16 Python threads (QD 16); ≈ 40 µs per read; 1.90 ms per 48-read token batch done serially | measured — [`evidence/e3-engram-gather-nvme0.json`](../evidence/e3-engram-gather-nvme0.json) |
| T705 Engram-style gather (E3) | **347K** reads/s under the same harness; ≈ 46 µs per read; 2.20 ms per serial 48-read batch | measured — [`evidence/e3-engram-gather-nvme1.json`](../evidence/e3-engram-gather-nvme1.json) |

Why it matters:

- The upstream V4.1-Flash checkpoint is 475.24 GiB; NVIDIA's NVFP4 is 491.06 GiB. Either fits on nvme0's free space alone, alongside the 322 GB of V4-Flash checkpoints already cached.
- The 0xSero Engram-from-NVMe path does 48 random reads per decoded token. At 2.2 M IOPS the drive can serve ~45K tokens/s of decode lookups; latency per token is bounded by one queue depth of ~50–80 µs. Prefill's 3.1 M reads per 64K tokens would take ~1.4 s at saturation. E3 measures both drives with O_DIRECT random 4 KiB reads so Option A can be planned with either "Engram in DRAM" or "Engram on NVMe + cache".
- Sequential 14.8 GB/s means a cold 475 GiB load takes ≥ 33 s from disk; in practice the engine's loader and dequant dominate.

## Power and cooling (planning inputs for Option A)

| Item | Value | Source |
|---|---|---|
| PSU | **1,500 W** (owner statement; model and 12V-2x6 connector count not recorded) | owner |
| Current GPU power limit | 2 × 300 W (325 W max, **250 W minimum** — `power.min_limit`; the driver rejects `-pl 225`) | measured (`nvidia-smi --query-gpu=power.min_limit`) |
| Measured GPU draw, V4-Flash TP2 C1 decode | 242 / 249 W per card at 2,272 / 2,310 MHz SM — not touching the cap | measured (E8, `evidence/baseline-v4-flash/power-decode-c1-300w.csv`) |
| Measured GPU draw, 32K prefill | 300 / 300 W (at the cap), SM throttled to 1,950 MHz | measured (E8, `power-prefill-32k-300w.csv`) |
| Sustained: ~1M-token prefill, 8 min | 300 / 300 W for 489 s; SM 1,867 / 1,815 MHz (−6 % vs the 32K burst); **88 / 87 °C peak** — the thermal ceiling to watch with four cards in the chassis | measured (E8, `power-prefill-1m-300w.csv`) |
| Power-cap sweep 325 / 300 / 275 / 250 W | decode 216–229 tok/s at every cap (never reaches the cap; 228–251 W/card, spread is temperature); prefill pinned at each cap: **8,770 / 8,428 / 7,991 / 7,482 tok/s** at 2.09 / 1.98 / 1.86 / 1.72 GHz → 13.6 / 14.0 / 14.5 / 15.0 tok/s/W | measured (E8b, `evidence/e8b-power-sweep/summary.json`) |
| CPU | Threadripper PRO 9955WX, 350 W TDP | vendor spec |
| Platform (board, 8 RDIMMs at 1.4 V EXPO, 2 NVMe, fans) | ~80–120 W | estimate |
| Today (2 GPUs) | 600 + 350 + ~100 ≈ **1,050 W DC worst case** — fits with ~30 % margin | derived |
| Today at the 325 W maximum cap | 650 + 350 + ~100 ≈ 1,100 W — fits; **adopted (T2.9)**: GLM +9 % C16 decode / +5 % prefill, V4-Flash 0 % / +1 %; sustained 89 °C, 0 throttle events | derived + measured |
| Four Max-Q at 300 W | 1,200 + 350 + ~100 ≈ **1,650 W DC** — **exceeds the 1,500 W PSU** | derived |
| Four Max-Q at 250 W | 1,000 + 350 + ~100 ≈ 1,450 W — inside the label, ~3 % margin; not acceptable for sustained load | derived |
| Four Max-Q at 250 W + CPU PPT capped at 200 W | 1,000 + 200 + ~100 ≈ 1,300 W — 87 % of the PSU; above the usual 80 % continuous-load ceiling | derived |
| Four Max-Q at 225 W | **not settable**: 250 W is the driver's floor on these cards | measured |

### Power cap: what it is, how to set it, what it buys (T2.9, 2026-09-21)

The cap is a per-GPU driver setting (`nvidia-smi -pl`), not a per-recipe or per-container option: the serving containers run unprivileged and cannot change it, and it applies to whichever model is up. It resets to the 300 W default on reboot.

| Action | Command |
|---|---|
| Read current caps | `nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.max_limit,power.min_limit --format=csv` |
| Set 325 W now (both cards, until reboot) | `sudo nvidia-smi -pl 325` |
| Back to the 300 W default now | `sudo nvidia-smi -pl 300` |
| Persist 325 W across reboots | `sudo cp systemd/gpu-power-limit.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now gpu-power-limit` |
| Remove the persistent setting | `sudo systemctl disable --now gpu-power-limit` (its `ExecStop` restores 300 W) |
| Measure the effect on the running server | `sudo bash scripts/bench-power-sweep.sh --url http://127.0.0.1:8000 --model <served-name> --caps 325,300 --concurrency 16 --runs 5 --output evidence/<dir>` |

Per-recipe caps would need a sudoers rule for exactly `/usr/bin/nvidia-smi -pl *` and a call from each `serve.sh`; not built — GLM is the daily driver and V4-Flash is indifferent to the cap.

**What 325 W buys over 300 W** (same runs, same fixtures; [`evidence/t29-*`](../evidence/)):

| Phase | GLM-5.3-Flash (EXL3) | DSV4-Flash (NVFP4) |
|---|---|---|
| C16 batched decode | **+9.1 %** (589 → 642 wall tok/s); sustained 15 runs 607 | **0 %** (1,213 → 1,206; run noise 1,088–1,338) |
| 32K prefill | **+5.1 %** (4,215 → 4,429) | +1.3 % (8,256 → 8,360; noise) |
| C1 decode | cap not reached at C1 (256–259 W) | 0 % (216–229 tok/s at every cap, E8b) |
| SM clock under load | +10 % (1.87 → 2.06 GHz) | unchanged (2.19 GHz at both caps) |
| Decode tok/s per W | flat (0.98 → 0.99) | **−8 %** (2.02 → 1.85) |
| Sustained heat | 89 °C, fans 73 %, 0 throttle events | 86 °C, fans 57 %, 0 throttle events |

Why they differ: GLM's EXL3 trellis weights are dequantised on the SMs before every matmul, so it turns watts into clock. V4-Flash's NVFP4 GEMMs run on tensor cores and wait on memory bandwidth, which the cap does not touch — it drew the full 650 W at the same clock, i.e. leakage on hotter silicon. Expect any V4.1 engine (NVFP4/MXFP4) to behave like V4-Flash. Throttle point per the T.Limit readout is ≈ 92 °C; the sweep summary prints `thermal-throttle samples` per row, and it has been 0 in every run so far.

**Decision:** 325 W system-wide, because the daily driver gains 9 % and the other recipe loses nothing but efficiency. If V4-Flash becomes primary, drop to 300 W. 250 W is retired — it was a four-card PSU budget, and the owner has ruled out planning for four cards.

**Option A needs more power than this machine has.** Four RTX PRO 6000 Max-Q at their 300 W limit plus the 350 W CPU do not fit a 1,500 W supply, even before transient spikes. The choices, in order of preference:

1. **Replace the PSU with a 2,000 W-class unit (or add a second PSU for two of the cards).** At 2,000 W the 300 W/card profile has ~18 % margin. A 2,000 W supply at 120 V draws ~17–18 A at the wall, above the 16 A continuous rating of a 20 A circuit, so it also needs a 240 V or dedicated 30 A circuit; at 240 V it is ~9 A. Record the circuit before ordering.
2. **Keep the 1,500 W PSU and cap the cards at 250 W** (`nvidia-smi -pl 250`, the lowest the driver accepts) with the CPU PPT limited in BIOS. **E8b measured the cost: 250 W costs decode 0 % (226 vs 216–229 tok/s at the other caps; the cards draw 228–251 W uncapped) and prefill 11 % (7,482 vs 8,428 tok/s at 300 W), for 17 % less GPU power.** Efficiency rises monotonically as the cap drops (13.6 tok/s/W at 325 W → 15.0 at 250 W), so more cards at a lower cap beats fewer cards at a higher one. Even so, four cards at 250 W plus a 200 W CPU sit at ~1,300 W, 87 % of the supply — workable for experiments, not for continuous serving.
3. Not an option: full 600 W Workstation-edition cards. Four of them are 2,400 W of GPU alone.

Measure, don't assume: GPU draw under the E8 baseline is now recorded above. Still missing: whole-system draw at the wall (no meter), and the CPU's real serving draw (usually well under TDP, but Engram host-table lookups load all 16 cores during prefill) — needed to replace the "~100 W platform" and 350 W CPU estimates.

## Software already on the host relevant to this plan

- `hf` CLI 1.29.0; HF cache holds `deepseek-ai/DeepSeek-V4-Flash-0731` (158 GB), `nvidia/DeepSeek-V4-Flash-0731-NVFP4` (164 GB), `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`, and GLM-5.3 checkpoints.
- Docker images: several `dsv4-sm120` / `dsv4-nvfp4` builds (28.8 GB) from the V4-Flash recipe.
- Host Python 3.14.7 (Homebrew) without torch/numpy; GPU-side benchmarks run inside a container.
- A GLM-5.3 Flash container (`glm53-flash-2x-rtxpro6000`) is normally running on both GPUs. Any GPU experiment here must stop it first and restore it afterwards; the plan's tasks say so explicitly.

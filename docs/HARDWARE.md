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

Why it matters: the board has enough PCIe 5.0 x16 slots and CPU lanes to take **four** RTX PRO 6000 cards at full x16 without a switch, which is what Option A in [FEASIBILITY.md](FEASIBILITY.md) needs. Each GPU's own x16 link is also the ceiling for Option B (expert streaming).

## CPU

| Item | Value | Source |
|---|---|---|
| Model | AMD Ryzen Threadripper PRO 9955WX, Zen 5 | measured |
| Cores / threads | 16 / 32 | measured |
| Max clock | 5,452 MHz | measured |
| L3 | 64 MiB (2 × 32 MiB) | measured |
| ISA | AVX-512 (F, BF16, VNNI); **no AMX** | measured |
| NUMA | 1 node (Threadripper PRO presents a single node) | measured |

Why it matters: Engram host-table gathers and any host-side expert staging run on these 16 cores. No AMX means KTransformers-style CPU expert compute (built around Intel AMX) is not a path here. AVX-512 BF16 is enough for gather/scatter and dequant-free FP8 copies, which is all the Engram host path requires.

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
| Random 264 B row gather (E3) | **4.15 M rows/s** from 16 Python threads on a 4.9 GiB table; 48-row batch p99 0.015 ms | measured — [`evidence/e3-engram-gather-dram.json`](../evidence/e3-engram-gather-dram.json) |
| Huge pages | `HugePages_Total=0`; THP `madvise` | measured |

Why it matters:

- The full FP8 Engram tables (188.83 GiB) fit in DRAM with 34 GiB to spare after a 24 GiB OS reserve. On the 128 GB box that produced the only 4-GPU result, they did not; that recipe had to read Engram from NVMe with a 64 GiB DRAM cache. This host can run SGLang's `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` path with the *whole* table in DRAM — the one place this machine is better positioned than the published reference.
- SGLang's host-table path wants huge pages. Reserving ~190 GiB of 2 MiB or 1 GiB hugepages is a boot-time (`hugepagesz=1G hugepages=190`) or runtime (`vm.nr_hugepages`) change; it is reversible and is task T2.2.
- Decode Engram traffic is trivial: 48 random 264 B rows per token ≈ 12.4 KB. Prefill of 64K tokens is ≈ 3.1 M random row reads; at DRAM random-access latency (~80–120 ns, parallel across cores) that is well under a second. E3 measures it.
- Option B (expert streaming) would need 105 GiB of experts pinned in DRAM alongside 97 GiB of FP4 Engram: 226 GiB against 246.8 GiB. It fits, barely. DRAM bandwidth (409.6 GB/s theoretical) is not the bottleneck; PCIe is.

## GPUs

| Item | Value | Source |
|---|---|---|
| Model | 2 × NVIDIA RTX PRO 6000 Blackwell **Max-Q** Workstation Edition | measured |
| Architecture | Blackwell GB202, SM 12.0 (compute capability 12.0) | vendor spec |
| VRAM | 97,887 MiB = 95.59 GiB each; 191.19 GiB total | measured |
| Memory bus | 512-bit GDDR7 ECC; `clocks.max.memory` 14,001 MHz → 28 Gbps PAM3 → **1,792 GB/s** theoretical per card (NVIDIA lists 1.6 TB/s) | measured clock, derived bandwidth — **measure in E2** |
| Power | 300 W limit, 325 W max (Max-Q); the full Workstation Edition is 600 W | measured |
| Max SM clock | 3,090 MHz | measured |
| PCIe | Gen 5 x16 both cards; **P2P read: OK** both directions; topology NODE (same CPU, no NVLink) | measured |
| Driver / CUDA | 615.71.09 / CUDA 13.4 UMD | measured |
| FP4 tensor throughput | ~2 PFLOPS dense NVFP4 (Max-Q throttles below the 600 W card) | vendor spec |

Why it matters:

- Weight capacity is the whole story: 95.6 GiB per card versus ~140 GiB per card needed for a 2-GPU split of the non-Engram weights. See the placement table in FEASIBILITY.md.
- PCIe 5.0 x16 is 63 GB/s theoretical per direction; realistic pinned host-to-device copies land around 50 GiB/s. That number sets the ~60 tok/s ceiling on Option B and is the first thing E4 measures.
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
| Current GPU power limit | 2 × 300 W (325 W max) | measured |
| CPU | Threadripper PRO 9955WX, 350 W TDP | vendor spec |
| Platform (board, 8 RDIMMs at 1.4 V EXPO, 2 NVMe, fans) | ~80–120 W | estimate |
| Today (2 GPUs) | 600 + 350 + ~100 ≈ **1,050 W DC worst case** — fits with ~30 % margin | derived |
| Four Max-Q at 300 W | 1,200 + 350 + ~100 ≈ **1,650 W DC** — **exceeds the 1,500 W PSU** | derived |
| Four Max-Q at 250 W | 1,000 + 350 + ~100 ≈ 1,450 W — inside the label, ~3 % margin; not acceptable for sustained load | derived |
| Four Max-Q at 225 W + CPU PPT capped at 200 W | 900 + 200 + ~100 ≈ 1,200 W — 80 % of the PSU, the usual continuous-load ceiling | derived |

**Option A needs more power than this machine has.** Four RTX PRO 6000 Max-Q at their 300 W limit plus the 350 W CPU do not fit a 1,500 W supply, even before transient spikes. The choices, in order of preference:

1. **Replace the PSU with a 2,000 W-class unit (or add a second PSU for two of the cards).** At 2,000 W the 300 W/card profile has ~18 % margin. A 2,000 W supply at 120 V draws ~17–18 A at the wall, above the 16 A continuous rating of a 20 A circuit, so it also needs a 240 V or dedicated 30 A circuit; at 240 V it is ~9 A. Record the circuit before ordering.
2. **Keep the 1,500 W PSU and cap the cards at 225 W** (`nvidia-smi -pl 225`) with the CPU PPT limited in BIOS. 0xSero's 4-GPU numbers were at 275 W; decode is memory-bandwidth-bound so 225 W costs less than the proportional 25 %, but prefill (compute-bound) will slow. This is the interim experiment plan: E8b's power sweep (300 → 275 → 250 → 225 W) on the existing two cards measures the actual cost per watt before committing.
3. Not an option: full 600 W Workstation-edition cards. Four of them are 2,400 W of GPU alone.

Measure, don't assume: the E2/E4 GPU window should also record whole-system draw at the wall if a meter is available, and `nvidia-smi --query-gpu=power.draw` under the E8 baseline load, so the "~100 W platform" and the CPU's real serving draw (usually well under TDP, but Engram host-table lookups load all 16 cores during prefill) replace the estimates above.

## Software already on the host relevant to this plan

- `hf` CLI 1.29.0; HF cache holds `deepseek-ai/DeepSeek-V4-Flash-0731` (158 GB), `nvidia/DeepSeek-V4-Flash-0731-NVFP4` (164 GB), `webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`, and GLM-5.3 checkpoints.
- Docker images: several `dsv4-sm120` / `dsv4-nvfp4` builds (28.8 GB) from the V4-Flash recipe.
- Host Python 3.14.7 (Homebrew) without torch/numpy; GPU-side benchmarks run inside a container.
- A GLM-5.3 Flash container (`glm53-flash-2x-rtxpro6000`) is normally running on both GPUs. Any GPU experiment here must stop it first and restore it afterwards; the plan's tasks say so explicitly.

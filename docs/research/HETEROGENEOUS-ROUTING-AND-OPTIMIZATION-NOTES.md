# Heterogeneous routing and optimisation notes

Written 2026-09-22. Covers five discussions that produced decisions but no experiment yet: pairing a 128 GB Mac with the 2× RTX PRO 6000 box (§1), adding a third RTX PRO 6000 (§2), routing work between hosted interactive models and the local V4.1 hybrid (§3), the speculative-decoding program (§4), and the risk/follow-up list (§5). Labels follow [README.md](README.md): **measured / projected / inferred / pending / failed**.

Fixed facts used throughout (all **measured** or read from the checkpoint unless marked): 40 MoE layers; hidden size 5,120; 384 routed experts, top-6; one native MXFP4 expert ≈ 17.9 MiB; placement 208 GPU + 176 CPU experts per layer (`KT_NUM_GPU_EXPERTS=208`); CPU-tier read ceiling ≈ 125 GB/s (E1b); Engram 48 rows × 264 B = 12,672 B per token (E18); full Engram tables 188.83 GiB (2 × 94.4 GiB); host RAM ≈ 247 GiB; C1 decode 38.5 tok/s (E21), 13.98 tok/s before E20.

---

## 1. Exo / MLX / macOS 128 GB next to the Linux CUDA box

### 1.1 Verdict

**Inferred, no run:** do not plan on a Mac improving single-stream (C1), token-synchronous V4.1 decode by serving routed experts over a direct cable. The parts exist — IP over Thunderbolt works Linux↔macOS, and MLX has real packed-MXFP4 grouped-matmul kernels for Metal — but the intersection needed is missing:

- no maintained runtime joins SGLang/CUDA V4.1 to a Metal expert worker;
- Mac↔Linux Thunderbolt RDMA is not documented by Apple;
- a 128 GB Mac cannot hold all 176 CPU-resident native experts (§1.4);
- a remote-expert design adds ≈ 40 sequential request/response dependencies per token, one per MoE layer.

The link's **small-message round trip**, not its 80/120 Gbit/s headline, decides the outcome. The idea earns exactly one bounded microbenchmark (§1.6) because replacing 125 GB/s host-CPU expert reads with 546–614 GB/s Apple unified memory could help prefill and batched decode.

Mac roles ranked by expected value (§1.5): independent/asynchronous services; a compatible draft model synchronised per draft block instead of per layer; an experimental prefill / high-concurrency expert worker; whole-layer pipeline placement for capacity, not speed. Engram-only remote placement is unattractive.

### 1.2 Physical interconnect

**Direct cable**

- **[documented]** Linux implements Apple's ThunderboltIP host-to-host protocol in `thunderbolt-net`; with macOS it appears as a virtual Ethernet interface. The credible cross-OS route is static link-local IP, then TCP/UDP/RPC. Source: kernel admin guide, "Networking over Thunderbolt cable" ([1](#sources)).
- **[documented]** Linux has `USB4STREAM` raw streams, but macOS exposes no matching public cross-platform peer API. Not a design basis.
- **[documented]** Thunderbolt 5 is 80 Gbit/s per direction; the 120 Gbit/s mode is asymmetric 120/40 and intended for display traffic. Intel specifies **64 Gbit/s for Thunderbolt networking/PCIe data** before protocol and software overhead ([2](#sources)).
- **[measured elsewhere, Mac↔Mac, not this topology]** MLX/JACCL moved large buffers at about **3.5 GB/s (≈ 28 Gbit/s)**, ≈ 37 % of the 80 Gbit/s physical rate ([3](#sources)). A TB4 Linux example measured **13.7 Gbit/s TCP** and **≈ 1.09 ms RTT** ([4](#sources)).
- **[measured elsewhere, different link]** Metal + CUDA llama.cpp RPC over direct 10 GbE: **9.41 Gbit/s**, **≈ 0.17 ms RTT**; decode slowed **47–50 %** while some prefills improved ([5](#sources)). Directional evidence only.
- **[unknown]** No primary benchmark establishes sustained throughput, 10–20 KiB request RTT, p99 jitter or CPU cost for this exact Mac / Linux / controller / cable pairing.

**RDMA, DMA, RoCE, peer-to-peer**

- **[documented]** Apple's Thunderbolt RDMA requires Apple silicon + TB5 + macOS 26.2+, is enabled from Recovery, and is documented as peer-to-peer **between Mac computers** ([6](#sources)). It is two-sided send/receive only (UC queue pairs, pre-posted matching receives, maximum message 16,773,120 bytes, IOMMU-registered local memory); completion does not acknowledge remote receipt. JACCL/MLX is Mac-only and does not build on Linux.
- **[inferred]** Treat Mac↔Linux as TCP over ThunderboltIP. Linux verbs/RoCE and Apple's Thunderbolt verbs are separate providers with no documented interoperability; llama.cpp likewise lists Linux RoCE NICs and macOS TB5 RDMA as separate transports ([7](#sources)).
- Neither OS exposes CUDA↔Metal peer GPU memory. Apple unified memory helps local Metal↔CPU movement only; the Linux side pays CUDA D2H/H2D staging on every exchange. GPUDirect RDMA needs a supported NIC/provider, not `thunderbolt-net`. There is no cross-host CUDA P2P, shared BAR mapping, or "Mac as CUDA eGPU" path.
- Optical Ethernet improves distance and bulk capacity but not runtime compatibility, staging, or the ≈ 40 dependent per-token exchanges; it does not change the RTT-dominated C1 economics by itself.

**Hardware requirements if it is ever tried**

- Certified **TB5 80 Gbit/s** cable, ≤ 1 m to start; not charge-only or USB 3. TB5 is backwards compatible, so a 40 Gbit/s USB4/TB4 endpoint silently halves the ceiling.
- The Linux host needs a host-capable USB4/TB controller with XDomain/networking support; a USB-C port alone is insufficient. TB5 add-in-card support is board/BIOS/header/lane dependent — verify on the WRX90E-SAGE SE with `lspci`, `boltctl`, `/sys/bus/thunderbolt`, negotiated generation and `thunderbolt_net`.
- Setup: direct host-to-host cable, macOS Thunderbolt Bridge, Linux `thunderbolt0`, static `/30`, persistent route, no dock/hub; keep Ethernet for management.
- An external PCIe enclosure is not analogous: a Mac host does not become a PCIe endpoint, its unified memory is not a BAR, and its Metal GPU cannot be a CUDA peer.
- Apple lists the 128 GB M4 Max at **546 GB/s** and the 128 GB M5 Max at **614 GB/s**; both have TB5 ([8](#sources), [9](#sources)).

### 1.3 Model and runtime compatibility

| Runtime | Status for V4.1 experts on Metal | Label |
|---|---|---|
| MLX | Has Metal QMV/QMM/gather-QMM for packed group-32 MXFP4 (E2M1 + UE8M0) and MXFP8 (E4M3 + UE8M0); weights stay packed and dequantise tile-wise ([10](#sources), [11](#sources)); `mlx-lm`'s `QuantizedSwitchLinear` uses indexed `gather_qmm` ([12](#sources)). **Primitive exists; integration does not.** | documented |
| MLX-LM | No DeepSeek V4.1 graph or Engram; issue reports `Model type deepseek_v4 not supported` ([13](#sources)). Older DeepSeek FP8 loader dequantises block FP8 at load; generic MXFP8 cannot be assumed compatible with V4.1's dense FP8. | documented |
| PyTorch MPS | Unsuitable for the production FP8/MXFP4 grouped-MoE route; float8 incomplete ([14](#sources)). CPU fallback is not Metal evidence. | documented |
| llama.cpp RPC | Remote CUDA and Metal in one graph with coarse layer/tensor splits; proof-of-concept, fragile, insecure, no exact V4.1/Engram support, cannot shard expert IDs inside stacked expert tensors ([7](#sources)). | documented |
| ds4 | Strongest V4.1 Metal reference: dedicated V4.1 Metal/CUDA kernels, Engram handling, exact-format tests, two-Mac / two-DGX-Spark expert TP, TCP pipeline modes ([15](#sources), [16](#sources)). Beta, fast-moving; network TP is homogeneous (Mac+Mac or Spark+Spark), not Mac+CUDA host; artefacts are calibrated Q2/Q4 conversions, not this native SGLang checkpoint. Reports V4.1 speculative decoding unsupported. | documented |
| SGLang EP | dispatch → permute → grouped GEMM → combine with CUDA/ROCm DeepEP/Mooncake/NIXL/PPLX/MORI backends ([17](#sources)). Its MLX backend is separate, not a mixed CUDA+Metal EP rank. | documented |
| Exo | Active for Apple clusters; Linux CUDA planned, not tier-one. No production mixed CUDA/Metal expert-parallel V4.1 runtime. | documented |

A prototype would need a new SGLang MoE runner/dispatcher or a sidecar RPC at the V4.1 MoE boundary that batches **all** Mac-owned experts selected in a layer into one persistent request and returns one weighted/reduced partial hidden vector, overlapping local work. Per-expert RPC is rejected. The wire ABI must fix shapes/strides, wire dtype (BF16/FP16/FP32), endianness, IDs, route weights, UE8M0 scale layout, E2M1 nibble order, activation rounding, SwiGLU order, partial-sum dtype and TP2 all-reduce placement; buffers preallocated and pinned; no Python serialisation on the hot path. The existing breakable CUDA graphs already break for CPU and Engram work, but a network worker adds **40 external synchronisation points**; rank-0-only ownership preserves today's asymmetry, both ranks issuing RPCs doubles traffic — one rank should own dispatch and broadcast/reduce, or the work must be deliberately partitioned.

**Inferred conclusion:** the intersection "SGLang V4.1 + mixed CUDA/Metal + remote expert routing + maintained production transport" is empty today.

### 1.4 Transport / RTT model (**projected**)

**Mac expert capacity.** All 176 CPU experts: 176 × 17.9 MiB × 40 = **123.1 GiB** before runtime, OS, Metal heaps and staging — not safe in 128 GB unified memory.

| Experts per layer on the Mac | Weight bytes |
|---:|---:|
| 140 | 97.9 GiB |
| 150 | 104.9 GiB |
| 160 | 111.9 GiB |
| 176 | 123.1 GiB |

Practical initial limit ≈ **140–150 experts/layer**, leaving some CPU-resident experts on the Linux host.

**C1 decode.** For R Mac-owned experts the expected number selected per token per layer is 6R/384; at R = 176 that is **2.75**. Mac weight reads per token: 2.75 × 17.9 MiB × 40 = **1.92 GiB**.

- Apple-memory lower bound: ≈ **3.8 ms** at 546 GB/s, **3.4 ms** at 614 GB/s.
- At 30–70 % effective bandwidth plus gate/up/down kernels: ≈ **6–15+ ms/token**, pending measurement.
- Host-CPU streaming lower bound for the same bytes: ≈ **16.5 ms** at 125 GB/s, before compute/launch/synchronisation. (E21 measured the real CPU tier differently — 0.19 ms per expert, ≈ 40 waits/step hidden by deferral — so the 16.5 ms floor must be replaced by the measured per-step CPU time before any go decision.)

One RPC per layer carrying one BF16 5,120-vector out and one reduced BF16 vector back is ≈ **20 KiB/layer = 800 KiB/token** (1.56 MiB at FP32); per-selected-expert transfer would be ≈ 2.15 MiB/token and is to be avoided. At 2–3.5 GB/s the serialisation cost is **0.23–0.40 ms/token** — negligible next to 40 dependent round trips:

| RTT per layer | Added latency per token |
|---:|---:|
| 50 µs | 2 ms |
| 0.2 ms | 8 ms |
| 0.5 ms | 20 ms |
| 1.0 ms | 40 ms |

Break-even: T_Mac + 40 × RTT + staging < T_CPU.

- Mac execution 8 ms against a 16.5 ms CPU floor → RTT must be **< ≈ 0.21 ms** before staging.
- Ideal 3.8 ms Mac execution → RTT **< ≈ 0.32 ms**.
- Safe gate: **p99 application RTT ≤ 0.15–0.20 ms** with kernel margin to spare.

**Higher concurrency.** Expected unique Mac experts per layer for batch B: R·[1 − (1 − 6/384)^B]. At R = 176:

| Decode batch B | Unique Mac experts / layer | Mac weight bytes per token |
|---:|---:|---:|
| 1 | 2.75 | 1.92 GiB |
| 4 | 10.7 | 1.88 GiB |
| 16 | 39.2 | 1.71 GiB |
| 64 | 111.8 | 1.22 GiB |

One RPC per layer per batch amortises RTT and larger Metal matrices raise utilisation, so **C4/C16 is more promising than C1** — but each request still crosses all 40 round trips. Compare with distinct prompts; same-prompt route sharing overstates the benefit (E19 measured that distinction on the hybrid).

**Prefill and other placements.**

- Large chunks touch nearly all R experts: at R = 176 the lower-bound expert traffic is 123 GiB per 40-layer chunk, ≈ 30.8 MiB/token at a 4,096-token chunk. Aggregated BF16 wire traffic stays ≈ 800 KiB/token; at 215.6 tok/s (E19 prefill) that is ≈ 168 MiB/s. **Bandwidth is not the prefill blocker**; batched Metal gather-QMM efficiency and integration are.
- Whole-layer pipeline: one boundary RTT per token, but a layer carries ≈ 6.9 GiB of routed experts plus dense/attention weights, so perhaps **12–14 layers** fit with headroom. Capacity-positive, C1-speed-negative.
- Engram: 188.83 GiB cannot fit. Indices are small and raw rows are 12.7 KiB/token, but the two Engram graph breaks would become network dependencies. Local warm Engram is ≈ 0.003 ms/token and cold-prefetched ≈ 1.02 ms (E18, measured); remote TCP + SSD/cache is unlikely to win.
- A Mac speculative draft exchanges token blocks rather than layer activations: negligible wire cost, and verifier batching can help CUDA utilisation. Acceptance and draft speed must be measured; ds4 reports V4.1 speculative decoding unsupported.

### 1.5 Candidate Mac roles, ranked

1. **Independent / asynchronous service** — embeddings, reranking, retrieval, preprocessing, prompt compression, a separate medium-model queue. Highest confidence and return.
2. **Compatible speculative/draft model** — one synchronisation per draft block, not 40 per-layer RTTs; needs a draft faster than the target and useful acceptance.
3. **Remote routed experts for prefill and C ≥ 4** — research-grade; native MXFP4 primitives and better batching, but requires a custom exact SGLang + Metal RPC.
4. **Contiguous whole-layer pipeline** — capacity, not speed.
5. **Engram hot cache / partial placement** — low value.
6. **Per-expert RPC / cross-host tensor parallel on every op** — reject.

### 1.6 Bounded experiment plan with stop/go thresholds (**pending**, not registered as an E-number)

**Phase 0 — inventory.** Record Mac model/SoC, macOS version (26.2+ only if testing RDMA), free unified memory, Linux kernel, board, TB/USB4 controller firmware, negotiated generation, IOMMU state, certified cable. Build ThunderboltIP with a static `/30`; keep Ethernet for management; test reconnect after reboot/sleep; disable Mac sleep. If the Linux controller lacks support, use direct 10 GbE as the control rather than buying an unverified add-in card.

**Phase 1 — transport gate.** Bidirectional `iperf3` (one and several streams), then a persistent request/reply loop at **4, 10, 20, 256 KiB and 1 MiB**. Record p50/p95/p99/max RTT, jitter, GB/s, retransmits and CPU over **30–60 minutes**. Compare TB vs 10 GbE, default vs validated larger MTU, `TCP_NODELAY`, pinned buffers, socket tuning.

Stop all C1 expert work if any of: p99 20 KiB RTT **> 0.25 ms**; frequent **> 1 ms** outliers; sustained payload **< 2 GB/s**; unreliable reconnect; excessive host CPU. Go signal: p99 **≤ 0.15–0.20 ms**. Ping is not evidence.

**Phase 2 — exact Metal expert microbenchmark.** Before any SGLang work, load **16–64 real V4.1 MXFP4/UE8M0 expert tensors** and implement exact gate/up/SwiGLU/down through MLX `gather_qmm` or ds4's Metal kernels. Benchmark B = 1, 4, 16, 64 and prefill M = 128, 512, 4,096 at 5,120 ↔ 2,304. Measure kernel time, effective weight GB/s, launches, memory, CPU, power (Metal System Trace / counters, `powermetrics`); reject CPU fallback. Validate packing, UE8M0, E2M1, rounding, per-expert outputs, router weighting and the reduced vector against kt-kernel on identical IDs/routes.

Go only if Mac B = 1 compute is **≈ 2× faster** than the *measured* local CPU-expert time, **or** T_mac + 40 × RTT_p99 + staging ≤ 0.85 × T_cpu. For prefill, require **> 20 % predicted end-to-end improvement**.

**Phase 3 — synthetic loop.** C++/Rust persistent sidecar, preallocated buffers, one batched request and one reduced result per layer; all 40 sequential exchanges with the exact kernels, before SGLang. Test delay, disconnect, malformed size, restart. No Python/pickle/HTTP on the hot path.

**Phase 4 — staged integration.** 1 layer → 4 → 40, overlapping local shared/CUDA experts. Profile route, D2H, send, queue, Metal, receive, H2D, combine and TP sync with Nsight and Metal signposts. Baselines: hybrid before/after dense FP8 tuning; local CPU experts with identical routes; loopback/no-compute; local-Mac/no-network; 10 GbE vs TB; C1/C4/C16 distinct prompts; 16K/long prefill; warm/cold Engram. Report tok/s, aggregate tok/s, TTFT, inter-token p50/p95/p99, useful GPU utilisation, CPU, memory, errors, J/token. Validate logits/layers, the 16K arithmetic/thinking/tool/needle suite (`scripts/verify.py`), deterministic and stochastic output, long context, mixed batches, reconnect/replay; **≥ 2 h soak**.

Full-system go: **≥ 15 % C1** or **≥ 20 % distinct-prompt aggregate throughput** after dense tuning; TTFT/p99 regression **< 10 %**; no correctness delta; stable **2 h**; acceptable J/token. A 5–10 % gain does not justify another OS/runtime/network failure domain. Capacity-only work needs its own criterion, e.g. **≥ 70 %** of current throughput while freeing enough DRAM/VRAM for the required model/context.

### Sources

1. https://docs.kernel.org/admin-guide/thunderbolt.html#networking-over-thunderbolt-cable
2. https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2023-09/thunderbolt-5-technology-brief.pdf
3. https://github.com/ml-explore/mlx/issues/3207
4. https://gist.github.com/geosp/80fbd39e617b7d1d9421683df4ea224a
5. https://github.com/kjaiswal/llama-cpp-distributed-benchmarks
6. https://developer.apple.com/documentation/technotes/tn3205-low-latency-communication-with-rdma-over-thunderbolt
7. https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
8. https://support.apple.com/en-us/122211
9. https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows
10. https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/fp_quantized.metal
11. https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/quantized.cpp
12. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py
13. https://github.com/ml-explore/mlx-lm/issues/1233
14. https://github.com/pytorch/pytorch/issues/132624
15. https://github.com/antirez/ds4/blob/main/docs/MODELS.md
16. https://github.com/antirez/ds4/blob/main/docs/DISTRIBUTED.md
17. https://docs.sglang.io/docs/advanced_features/expert_parallelism
18. https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

---

## 2. Third RTX PRO 6000 — back-of-envelope

Everything in this section is **projected** unless labelled measured. No 3-GPU build or benchmark exists; the owner's standing decision is no 4-GPU future-proofing, and the 3-GPU layout is information for a later purchase.

### 2.1 Why not TP3

V4.1-Flash has **64 attention heads**; 64 does not divide by 3, so plain TP3 is invalid. EP3 with replicated attention is possible in principle but replicates dense/attention state on every card and adds routing/collective complexity. Pipeline parallelism is feasible but worsens C1 latency across 40 layers; it is a batch/prefill option.

### 2.2 Proposed topology: TP2 + asymmetric expert rank

Keep GPUs 0–1 as today's TP2 dense/attention path. The third card holds **complete, not TP-sharded, routed experts**; routed activations are sent to it and its expert outputs are returned and reduced into the TP2 residual stream. It is an EP/remote-expert group, not a TP peer. (The discussion labelled the card "GPU 2" in the diagram and "GPU3" in prose; both mean the third card.)

```diagram
                    ┌───────────────────────────────┐
                    │ Dense, attention, shared      │
                    │ components: TP2               │
                    └───────────────┬───────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                │                                       │
        ┌───────▼───────┐                       ┌───────▼───────┐
        │ GPU 0         │◀────── TP/NCCL ──────▶│ GPU 1         │
        │ dense +       │                       │ dense +       │
        │ experts       │                       │ experts       │
        └───────┬───────┘                       └───────┬───────┘
                │           routed activations          │
                └────────────────┬──────────────────────┘
                                 ▼
                         ┌───────────────┐
                         │ third card    │
                         │ dedicated EP  │
                         │ expert rank   │
                         └───────┬───────┘
                                 │ residual overflow
                                 ▼
                         ┌───────────────┐
                         │ CPU experts   │
                         └───────────────┘
```

Capacity: **≈ 90–120 complete experts/layer** on the third card depending on replicated router/dense metadata, CUDA graphs, workspace and staging; conservative first target **≈ 100/layer** (≈ 70 GiB of weights across 40 layers, leaving useful but not luxurious margin).

| Placement | Experts per layer | GPU-resident share |
|---|---:|---:|
| Current (**measured** config, E28 2026-09-23) | 226 GPU / 158 CPU | 59 % |
| E19–E26 config (superseded) | 208 GPU / 176 CPU | 54 % |
| Third card at 100 experts | 326 GPU / 58 CPU (from 226; was 308 / 76 from 208) | 85 % |
| Third card at 90–120 | 316–346 GPU / 38–68 CPU | 82–90 % |

A static ≈ 100-expert partition is the recommended first implementation; frequency-based placement waits for V4.1 routing statistics (kt-kernel's `frequency` strategy selects a global top-N across layers, T3.6a). An earlier "91 % resident" figure assumed a generic EP3/PP hybrid and was not an implementation result.

### 2.3 Memory and context implications

Measured or planning facts today: ≈ 93.1 GB/card at 32K (E19); streamed-prefill staging 2.27 GiB per rank on GPU plus 6.53 GiB shared pinned host (E22); free VRAM after a 32K prefill ≈ 3.7 GiB with 4K chunks, ≈ 1.2 GiB with 8K chunks, 16K chunks do not fit (E22); KV ≈ 890 B/token by plan, so 262K is not a KV-capacity problem — weights, graph/workspace reserve and prefill activations/staging are the constraints.

Projected with ≈ 100 CPU experts moved to the third card: CPU-expert staging shrinks roughly proportionally, freeing perhaps **1–1.5 GiB per TP rank**; that does not guarantee 16K chunks but makes larger chunks or DSpark draft coexistence less implausible. 32K × C1–C4 should remain easy; 262K × C1–C4 is feasible in principle but is an engineering/soak target — 262K is unqualified and the NaN/Xid history (E24) is unresolved.

Engram stays on NVMe/page cache: the third card reduces host pressure but does not make a fully pinned/DRAM Engram safe (see [CURRENT-ARCHITECTURE-AND-ENGRAM.md](CURRENT-ARCHITECTURE-AND-ENGRAM.md) §2).

### 2.4 PCIe / NVLink / rank balance

Current hardware: PCIe Gen5 x16 per card, same node, P2P-capable, **no NVLink** (**measured**, `docs/HARDWARE.md`; E4: 53 GiB/s per card, 106 GiB/s both).

Without NVLink, TP2 + EP adds a per-layer activation dispatch and an output return/reduce. Volume is manageable next to expert weights; the risk is the **40 layerwise synchronisation points** and small-message latency. Expected: GPUDirect P2P adequate for batch/prefill; C1 loses a meaningful fraction of its theoretical CPU-work savings to dispatch/reduce latency; C4 benefits more than C1 if dispatch amortises. An NVLink-class fabric would improve dispatch/return latency and bandwidth, collective overhead and overlap stability — but would not remove custom placement/routing, the residual CPU tier, the Engram graph breaks or the NaN gate. These cards have no NVLink.

Rank balance (**measured**, E19): rank 0 runs ≈ 1,880 % CPU vs rank 1 ≈ 120 % because CPU experts and Engram live on rank 0; GPU1's 97 % "utilisation" is largely NCCL spin-wait. Design goals for a third card: distribute remote-expert dispatch and residual CPU work across ranks, assign the third card's experts statically, keep a reproducible uniform fallback.

### 2.5 Projected performance

Architectural estimates anchored to E21/E22 measurements; **not benchmark results**.

| Metric | Current TP2 hybrid (**measured**) | 3-GPU TP2 + EP, PCIe-only (**projected**) | With an NVLink-class fabric (**projected**, not this hardware) |
|---|---:|---:|---:|
| C1 decode | 38.5 tok/s | 42–60 | 50–70 |
| C3 / C4 aggregate | 112 / 147 tok/s | 140–210 | 170–250 |
| Prefill, 4K chunks | ≈ 0.84K tok/s | same to modestly better | same to modestly better |
| Prefill, 8K chunks | ≈ 1.38–1.40K tok/s | 1.5–2.0K if freed staging makes 8K+ chunks reliable | perhaps 1.8–2.5K |
| 32K TTFT | ≈ 19–23 s (streamed prefill) | modest improvement, mainly via larger chunks | similar |
| 262K TTFT | unqualified | minutes, not interactive | still minutes |

**Erratum 2026-09-23:** the current C3/C4 figures and the projected C3/C4 ranges above rely on kt-kernel bug C: the unpatched bs=4 graph computed correct CPU-expert output only for row 0. The qualified patched measurements are C3 **38.3 summed / 35.8 wall tok/s (12.8 per stream)** and C4 **50.8 summed / 47.0 wall tok/s (12.7 per stream)**. The three-GPU C3/C4 projections multiplied the artifact and are **pending recomputation**; do not use those ranges. See `docs/BENCHMARKS.md` “Erratum” and `docs/TROUBLESHOOTING.md` bug C.

Decode gains are bounded: the third card removes CPU-expert work but adds a GPU dispatch/reduce every layer, and the Engram graph breaks and a residual CPU tier remain. At 1.5–2K tok/s a 262K fill takes **2 min 55 s – 2 min 11 s** (projected) against ≈ 5 min at the measured 32K rate of 811 tok/s extrapolated linearly (an optimistic extrapolation, not a 262K measurement); none of it is comfortable interactively without prefix/KV caching.

### 2.6 Third card vs remote Metal

A third local GPU is much higher value for the V4.1 forward path than Mac remote experts: it keeps every exchange on PCIe inside one CUDA process instead of ≈ 40 dependent network round trips per token (≈ 8 ms at 0.2 ms RTT, ≈ 20 ms at 0.5 ms, before staging and Metal execution), and needs no new runtime. Recommendation: prefer the third card; prototype TP2 + a dedicated EP rank with a static 100-expert partition; **measure PCIe dispatch latency before assuming any gain**.

---

## 3. Router: hosted interactive models and the local hybrid

### 3.1 Routing rules

| Tier | Send here |
|---|---|
| Interactive — Claude, Codex or GLM (GLM-5.3-Flash local, 132 tok/s C1 **measured** E14) | coding loops, user-facing chat, tool use, short-latency agent turns, anything where first-token responsiveness matters |
| Local DeepSeek V4.1 hybrid | offline long-document analysis, large queued jobs, bulk extraction/classification, privacy-sensitive local work, codebase-wide background analysis, long contexts where minutes of TTFT are acceptable, batch/concurrent work where aggregate throughput matters more than C1 |

A third GPU would strengthen the offline/batch tier and moderate concurrency; it is not expected to make V4.1 the default interactive peer of the hosted models.

### 3.2 Four-agent behaviour (**measured**, E23 R0 and E22)

Distinct prompts, 512-token outputs, D=4, speculation off (`evidence/e23-spec-decode/r0-baseline-d4-specoff.json`); C2 from E22's 256-token streamer-resident run because R0 did not include C2.

| Concurrent agents | Aggregate decode (summed streams) | Per agent | End-to-end wall aggregate | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 38.73 tok/s | 38.73 | **36.01 tok/s** | 1.02 s |
| 2 | 47.81 tok/s (E22, 256 tok) | **23.9** | not recorded | — |
| 3 | 111.71 tok/s | **37.2** | **98.24 tok/s** | 1.45 s |
| 4 | 148.33 tok/s | **37.0** | **131.89 tok/s** | 1.63 s |

"Four users each get ≈ 37–38 tok/s" holds for C1/C3/C4: the fixed per-step CPU-tier synchronisation cost is shared by three or four active streams (E19: distinct routing is free at bs=4). Wall aggregate is 7–11 % below the summed stream rate because it includes TTFT and ramp; wall is the number to quote for real use.

**Erratum 2026-09-23:** the C3/C4 rows and the four-user conclusion are artifacts of kt-kernel bug C; the unpatched bs=4 graph computed CPU-expert output only for row 0, while bs=3 padded into that graph. With the patched image, distinct-prompt C4 is **50.8 summed / 47.0 wall tok/s (12.7 per stream)** and C3 is **38.3 / 35.8 (12.8 per stream)**. C4 buys only ≈ **+30% aggregate** over C1 while cutting per-stream speed by about 3×. Same-prompt C4 is 56.2 summed / 49.6 wall, only +11% over distinct, so decode is per-row CPU compute/latency-bound rather than free route sharing. See `docs/BENCHMARKS.md` “Erratum” and `docs/TROUBLESHOOTING.md` bug C.

**C2 anomaly (measured, CLOSED 2026-09-23):** per-stream 23.9 tok/s at C2 against 38.7 / 37.2 / 37.0 at C1 / C3 / C4 — present since E20 ("the bs=2 dip"). Resolved by the BENCHMARKS Erratum plus the E27 soak: the C3/C4 per-stream figures it was compared against were unpatched-kt artifacts; at the qualified defaults C2 = 50–52 summed (25–26 per stream) vs C1 42 / C3 56 (18.7) / C4 58 (14.6) — monotone, no dip.

**Erratum 2026-09-23:** C2 was correct on both builds; it is no longer the anomaly. Patched C3 is worse than C2 because bs=3 pads into the bs=4 graph; capturing bs=3 is an open follow-up.

### 3.3 Queue and concurrency policy (**inferred**, needs measurement before production use)

The server runs **four** requests (`MAX_RUNNING_REQUESTS=4`, `CUDA_GRAPH_MAX_BS=4`); more queue. Start one local job immediately; when independent offline jobs are available, batch three or four; cap active local work at four; queue beyond that. With exactly two jobs active, briefly hold for a third, use dummy batching only if shown safe and worthwhile, or accept the lower per-stream rate.

**Erratum 2026-09-23:** this queue advice relied on artifact near-linear C3/C4 scaling. Pending a new latency/throughput policy measurement, prefer C1 for interactive work and treat C4 as a throughput trade: ≈ +30% aggregate at 12.7 tok/s per stream, not four C1-speed agents.

### 3.4 Context and prefill caveats (**measured**, E22)

Short-prompt TTFT does not carry to long contexts. Streamed prefill: **837 tok/s at 16K (TTFT 19.1 s), 811 at 32K (38.4 s)** with the default 4K chunk; **1,375 / 1,404 tok/s (TTFT 11.6 / 22.8 s)** with 8K chunks, which leave ≈ 1.2 GiB VRAM and cannot coexist with DSpark drafts. Four already-prefilled or short-context agents do well; four simultaneous long-context jobs pay full prefill latency and tighten KV pressure. DSpark draft weights cost 3.85 GB per rank (E23 fit, measured) and at `MEM_FRACTION_STATIC=0.92` left no KV budget; 0.935 fits with reduced headroom. A 16K chunk does not fit. The non-speculative KV pool is ≈ 3.4 GB.

---

## 4. Speculative decoding / MTP roadmap (E23, T3.9)

Protocol and status live in `plan/EXPERIMENTS.md` E23 and `docs/BENCHMARKS.md`; this section records the reasoning and the state on 2026-09-22.

**Why it might work here (hypothesis).** E6-DESIGN §4.6 predicted that K=5 speculation *hurts* a CPU-bound MoE because a 6-token verify step touches up to 6× the distinct CPU experts per layer while accepting ≈ 3.3. E21 changed the shape of the cost: the per-layer CPU wait is kt-kernel's fixed m=1 latency (0.19 ms/expert, ≈ 40 waits/step) hidden by deferral, not DRAM bytes. A multi-token verify step pays that fixed synchronisation once per layer for K+1 tokens — it may **amortise the ≈ 40 boundaries** — or it may expand CPU work faster than acceptance pays back. E15 on V4-Flash (eager + DSpark 49.5 vs eager 15.9 tok/s, 3.1×) shows the amortisation effect exists when the step is overhead-bound. Speculation never removes target-side expert work; it changes how many tokens each fixed cost serves.

**Trial matrix (iterative, not one A/B).** R0 D=4 spec-off reference (**done**); R1 D=4 with K=2 and K=5 at C1/C3/C4 (**pending**); R2 D=0 with off/K=2/K=5 (deferral × verification interaction: deferred experts of a rejected token are wasted); R3 K ∈ {1,2,3} knee; R4 1,024-token outputs and an 8K-prompt variant (acceptance by workload); R5+ retune from the measured bottleneck (capture sizes, `MAX_RUNNING_REQUESTS`, scheduling, `KT_CPUINFER`). Never `KT_MAX_DEFERRED ≥ 5` until E21's D=5 NaN has a root cause (E24).

**Per cell:** wall tok/s per stream and aggregate (not summed stream rates), TTFT, ITL p50/p95, accepted tokens per verification and acceptance %, CPU-tier wait count and ms per *accepted* token, host DRAM GB/s, Engram critical-path ms/step, GPU memory/power/SM utilisation per card, distinct-CPU-expert count per verify step (routing reuse). **Gates for any candidate default:** `nll-score.py --compare evidence/e21-kt-fixed-floor/nll-d0.json` with |ΔNLL| < 0.002 (speculation is lossless by construction, so drift is a bug); `verify.py --context 16384 --needles 3` all pass; ≥ 20K-token mixed C1–C4 soak with zero NaN/assert/Xid. **Go:** ≥ 10 % C1 wall gain over R0 (≥ 39.6 tok/s), no material C3/C4 per-stream regression, acceptance ≈ ≥ 55 % as a health signal, NLL in gate, zero NaN. **Stop/retune:** CPU wait per accepted token does not fall; draft memory forces a worse expert/KV placement whose cost exceeds the gain; any instability.

**Graph and memory constraints (measured, E23 fit):** the draft (`DeepseekV4ForCausalLMDSpark`, three MTP stages) is 3.85 GB per rank and all-GPU — kt clamps `num_gpu_experts` to the draft's 128 experts, so the draft never touches the CPU tier. `MEM_FRACTION_STATIC=0.92` leaves no KV (main 82.97 + draft 3.85 GB consumed the static budget); **0.935** fits: KV pool 279,808 tokens, 5.74 GB free after capture (E22 non-spec: ≈ 7.0). The breakable decode graph captures target-verify at bs 1–4 × 3 tokens for K=2 (gamma 2, `speculative_num_draft_tokens=3`); the 208/176 placement is unchanged.

**Status (2026-09-22):** R0 done; K=2 reached speculation for ≈ 500 tokens (attempt 5, **measured**): accept length 1.8–2.15 of 3, acceptance 0.41–0.57, ≈ 30–34 tok/s — *not* faster than the 36 tok/s wall baseline at that point — then a CUDA illegal memory access in target-verify graph replay. Three overlay fixes were needed to get that far (`6032dd44`/`3b1fedce` prefill-graph breaks gated on `is_extend_without_speculative()`, `71645529` per-field dataclass rows, `95f2fe3d` output cap = bs × captured_req_width); the IMA is open (**failed**, diagnosis in progress; eager repro launched). K=5 and D=0 rounds have no result. The reliable production number remains ≈ 37 tok/s per user at C3/C4 with speculation off.

**Erratum 2026-09-23:** the final C3/C4 production claim is invalid because of bug C. The corrected speculation-off rates are 12.8 and 12.7 tok/s per stream respectively; the per-row CPU cost strengthens the conclusion that target verification, which multiplies rows, is unlikely to win.

---

## 5. Risks and follow-up experiments

| Item | Status | What is known | Next |
|---|---|---|---|
| NaN / E24 release gate | open, standing | Three sightings without root cause: E21 deterministic D≥5 assert on the first bs=4 step; E22c assert after a 4-seq batch (E22d: 664 real tokens, regular D=4 path, not reproduced in 20 same-shape batches); E19 non-deterministic first-mixed-batch assert removed by zero-fill (`1d6a98d3`) but explained only as stale FP8 bytes. Each leaves Xid 43 and a phantom-busy GPU0 | Bisect to a deterministic reproducer before changing code; no retry/clamp; ≥ 20K-token soak with zero Xid before any config becomes default; make a NaN fail the request, not the server |
| E23 K=2 illegal memory access | failed, diagnosing | First verify step after a 5,376-token needle prompt inside `seg.replay()`; hypothesis: verify window crossing a new KV page (hybrid FULL+SWA pools) | Eager repro (`CUDA_LAUNCH_BLOCKING=1 --disable-cuda-graph`), read the faulting kernel, fix in overlay, then R1 |
| ~~bs=2 / C2 anomaly~~ | **closed 2026-09-23** (E27 soak) | was an artifact of the pre-E26 C3/C4 comparators; qualified C2 = 50–52 summed, monotone C1 < C2 < C3 < C4 | — |
| 262K qualification | unqualified | 811 tok/s is a 32K measurement; "5 min" for 262K is linear extrapolation | Needs a fit at 262K (KV ≈ 890 B/token is not the limit; activations/staging are) and a soak |
| Prefix / KV caching | not tested on the hybrid | mitigations for long-context agent use: prefix/KV reuse across turns, incremental ingestion, prompt caching between agent steps, background prefill, long-lived resident sessions | Measure hit behaviour with `bench-decode.py` seeds (the V4-Flash sweep already showed prefix-cache hits distort prefill numbers) |
| Third-GPU prototype | projected only | §2 | Measure PCIe dispatch latency first |
| Mac transport / Metal kernel microbench | pending, unregistered | §1.6 phases 1–2 | Only if a TB5-capable Linux controller is confirmed |
| Reproducibility / push gap | open | Recipe repo and overlay are both local-only; container state is ephemeral | [OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md) |

**Erratum 2026-09-23:** the bs=2 row's comparison baseline is an artifact; C2 is correct, while patched C3/C4 are 12.8/12.7 tok/s per stream. The third-GPU projection is pending recomputation from corrected batching measurements.

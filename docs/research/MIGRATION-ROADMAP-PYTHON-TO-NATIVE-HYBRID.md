# Migration roadmap: Python-coordinated hybrid to a production native hybrid

Written 2026-09-23. This is a staged engineering roadmap, not an experiment result. It starts from the qualified Python/SGLang + C++/CUDA hybrid and answers one sequencing question: **which native work should precede multi-rank CPU-expert/Engram ownership, which should follow it, and which should wait for a stable architecture?** Sources are the repository evidence listed in [Evidence base](#evidence-base).

**Gate ledger (2026-09-24):** the stage prose below is bound to task rows, closure criteria, exit checklists, owners and a completion definition in the [Gate ledger](#gate-ledger-added-2026-09-24-the-enforced-sequence) at the end of this file. When the ledger and the prose disagree, the ledger wins; the prose is kept as written for provenance.

## Labels and decision rule

| Label | Meaning |
|---|---|
| **measured** | Recorded in repository evidence, benchmark tables, or experiment results; preliminary measurements say so explicitly |
| **projected** | Target or arithmetic estimate from measured inputs; not yet demonstrated end to end |
| **inferred** | Conclusion from code, architecture, or documentation rather than a run |
| **pending** | Required experiment or gate with no result yet |

The sequencing rule is:

1. **Before multi-rank:** native work whose contract does not depend on ownership topology: large-M dense kernels, fixed-address buffers, asynchronous prefetch/staging, and narrowly proven export hot paths.
2. **At multi-rank:** prototype ownership, dispatch, reduction, graph, failure, and affinity semantics in Python around stable native kernels; measure the contract.
3. **After multi-rank:** move only measured hot coordination loops and synchronisation into C++/CUDA, preserving a Python policy layer.
4. **After architecture stabilises:** ABI hardening, optional hardware transports, broad scheduler integration, packaging, and upstream-facing refactors.

This order avoids implementing rank-0 assumptions twice. Low-coupling kernel/dataflow work survives an ownership redesign; dispatch and coordination code usually does not.

## Evidence base

- [CURRENT-ARCHITECTURE-AND-ENGRAM.md](CURRENT-ARCHITECTURE-AND-ENGRAM.md): **measured** current TP2 placement, rank-0 kt-kernel path, streamed prefill, breakable graphs, Engram mmap/page cache, bottlenecks, and RAM accounting.
- [BENCHMARK-EVOLUTION-AND-E20-FP8.md](BENCHMARK-EVOLUTION-AND-E20-FP8.md): **measured** E19→E23 R0 decode/prefill evolution and E20 dense W8A8 FP8 tuning.
- [HETEROGENEOUS-ROUTING-AND-OPTIMIZATION-NOTES.md](HETEROGENEOUS-ROUTING-AND-OPTIMIZATION-NOTES.md): **projected/inferred** third-card, remote-Mac, routing, and speculative-decode trade-offs.
- [OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md): **measured** overlay base `da64c5cb`, patch series, reproducibility gap, and upstream candidates.
- [ROADMAP.md](../../plan/ROADMAP.md), [TASKS.md](../../plan/TASKS.md), and [EXPERIMENTS.md](../../plan/EXPERIMENTS.md): **measured/pending** E22, E23, E24, T3.6d3, T3.12, long-context, caching, and release gates.
- [E6-DESIGN.md](../../plan/E6-DESIGN.md), [FEASIBILITY.md](../FEASIBILITY.md), [BENCHMARKS.md](../BENCHMARKS.md), and [LEARNING.md](../LEARNING.md): **measured/projected/inferred** original design, memory model, benchmark method, and lessons.
- `evidence/e23-spec-decode/r0-baseline-d4-specoff.json`: **measured** qualified R0 reference.

## Executive sequencing answer

> **Sequencing amendment (2026-09-24, owner decision):** the Stage 1 Python-level refactor (adapters, runner-owned state, prefill coordinator, typed flags — `plan/TASKS.md` T3.19) runs *before* Stage 3, immediately after the B2 prototype's verdict (T3.18), not after the architecture stabilises. The original "do not refactor what multi-rank may invalidate" argument applies to *native* code (Stage 4, T3.23), which stays deferred; Python seams are what Stage 3 consumes. The E24 precondition on refactoring is dropped. Stage rows: T3.19 (1), T3.20/T3.21 (2), T3.22 (3), T3.23 (4), T4.1b (5).

**Do before changing ownership across TP ranks:**

- **Projected work:** replace the large-M block-scaled FP8 prefill path with a CUTLASS/DeepGEMM implementation behind the existing matmul contract. E20 measured only about **15% of FP8 peak** at `M=4096`; ownership does not change the GEMM mathematics.
- **Projected work:** allocate fixed-address Engram gather/prefetch/staging buffers, prefetch layer-14 rows while layer 1 proceeds, and make graph-compatible copies consume those buffers. Keep file-backed mmap/page-cache ownership unchanged.
- **Conditional projected work:** optimise kt export only if Stage 1 tracing shows conversion/copy dominates its **measured 90–117 ms per layer** rather than barriers or contention. Preserve the exported layout contract.
- **Measured and rejected (E30, 2026-09-24, T3.18):** the B2 prototype below was built as a symmetric-TP direct-UVA path (pinned-host copy of the last |S| CPU experts, stock SM120 CUTLASS grouped GEMM on UVA pointers, forked on a per-device side stream beside kt, gated to bs ≤ `KT_B2_MAX_TOKENS`) and served: **C1 wall 39.9 → 40.7 tok/s (+2 %) at |S| = 40, 38.7 (−3 %) at |S| = 60; C4 unchanged; consistency PASS; 0 NaN.** The gate was +10 %. The premise that failed is time-additivity, not bandwidth-additivity: 0.21 ms per streamed expert on one x16 link vs 0.146 ms per kt expert, with the kt path already mostly hidden by D=4. Kept off by default (overlay `7e41317e`); the fork/join side-stream pattern carries into item 1 (Engram staging). Evidence: [`evidence/t318-b2-prototype/README.md`](../../evidence/t318-b2-prototype/README.md). Original projection kept below for the record.
- ~~**Projected work (added 2026-09-23, Option B2 — the largest projected single-user lever):**~~ a rank-0-only *prototype* that streams a fixed subset of the DRAM-resident experts over PCIe into fixed device buffers each decode step and computes them with the E13 SM120 grouped GEMM, while kt-kernel computes the rest on the cores. This validates the dataflow (DMA overlapped with the forward, byte accounting, E13 dispatch from the served path) and the one measured premise it rests on — E4b's **measured** additivity of CCD reads and PCIe DMA (121 + 114 = 235 GB/s) held for synthetic reads, not yet under a real forward. Calculator ceiling at the slot-3 bandwidth: **52 → 79 tok/s (projected)**. Stage 2 item 0 below; ROADMAP layout B2 and kernel-track K7 point here.
- **Do not do yet (B2's other half):** the *placement policy* — which experts go to VRAM, CPU compute or PCIe streaming, per rank — and the PCIe-root/NUMA affinity of the streaming buffers. Both depend on rank ownership and belong to Stage 3.
- **Do not do:** a native scheduler, native rank-0 dispatcher, or broad C++ rewrite of SGLang coordination.

**Do at the multi-rank architecture stage, in Python first:**

- Define a static, deterministic rank partition for CPU experts and Engram work; specify dispatch, partial-output reduction, ordering, timeout, cancellation, and fallback.
- Validate fixed buffer addresses under CUDA graph capture/replay and batch-size changes.
- Measure per-rank CPU work, barriers, H2D, Engram graph breaks, p99 inter-token latency, and failure containment.
- Decide duplicate versus shared host state and pin each rank's workers/buffers to the appropriate CCD/NUMA locality.
- Decide the B2 placement policy per rank (VRAM / CPU compute / PCIe stream) and each rank's streaming-buffer affinity to its own PCIe root; the Stage 2 prototype's fixed subset becomes a policy input here.

**Do after the Python prototype passes:**

- Move only hot, proven dispatch/reduction loops, queue operations, and synchronisation into a C++/CUDA extension.
- Freeze an ABI for tensor shapes, strides, ownership maps, stream/event hand-off, errors, and configuration.
- Re-profile; multi-rank work removes rank asymmetry and may expose dense kernels, collectives, Engram, or Python launch as the next bottleneck.

**Wait until the architecture is stable:**

- A broad scheduler rewrite, a general Rust control plane, remote transports, dynamic expert placement, third-card dispatch abstractions, and public/upstream API commitments.

## As-is architecture

All values in this diagram are **measured** unless marked otherwise.

```diagram
 request
   │
   ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Python / SGLang scheduler and model coordination                    │
 │ rank 0 owns request policy, CPU-expert submission, Engram lookup    │
 │ D=4 deferral; speculation off; streamed prefill threshold=1024      │
 └───────────────────────┬──────────────────────────────────────────────┘
                         │ TP2 / NCCL over PCIe
              ┌──────────┴──────────┐
              ▼                     ▼
 ┌─────────────────────────┐   ┌─────────────────────────┐
 │ GPU rank 0              │   │ GPU rank 1              │
 │ dense FP8 + attention   │   │ dense FP8 + attention   │
 │ 208 GPU experts/layer   │   │ 208 GPU experts/layer   │
 │ breakable graph owner   │   │ waits at TP boundaries  │
 └────────────┬────────────┘   └────────────┬────────────┘
              │ pinned D2H/H2D              │ TP collectives
              ▼                             │
 ┌──────────────────────────────────────────┴───────────────────────────┐
 │ rank-0 kt-kernel: 176 CPU experts/layer, 16 worker threads          │
 │ compact MXFP4 view; submit/sync per MoE layer; fixed m=1 floor      │
 └───────────────────────┬──────────────────────────────────────────────┘
                         ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Engram: shard mmap + Linux page cache; WILLNEED + gather + H2D      │
 │ layers 1 and 14; eager gaps inside breakable decode graphs           │
 └──────────────────────────────────────────────────────────────────────┘
```

The current host imbalance is **measured**: rank 0 uses about **1,880% CPU**, rank 1 about **120%**, while rank 1's high GPU utilisation is largely NCCL spin. The CPU tier contributes about **40 waits per decode step**; Engram contributes **measured 2.4–3 ms per step** across two graph breaks. Streamed prefill has a **measured ≈4.6 s fixed cost per chunk**, composed of **measured 40 layers × export ≈90–117 ms plus two barriers per layer**.

## Target architecture

The target is **projected**. It is deliberately hybrid: native kernels and hot transport loops, Python scheduling and policy.

```diagram
 request
   │
   ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ Python policy plane                                                 │
 │ admission, batching, placement policy, experiments, rollback mode   │
 └───────────────────────┬──────────────────────────────────────────────┘
                         │ versioned ownership + buffer contract
              ┌──────────┴──────────┐
              ▼                     ▼
 ┌─────────────────────────┐   ┌─────────────────────────┐
 │ TP rank 0               │   │ TP rank 1               │
 │ dense/attention kernels │   │ dense/attention kernels │
 │ native local dispatch   │   │ native local dispatch   │
 │ CPU/Engram partition A  │   │ CPU/Engram partition B  │
 └────────────┬────────────┘   └────────────┬────────────┘
              │ fixed buffers + events      │ fixed buffers + events
              ▼                             ▼
 ┌─────────────────────────┐   ┌─────────────────────────┐
 │ CPU worker group A      │   │ CPU worker group B      │
 │ CCD/NUMA-local weights  │   │ CCD/NUMA-local weights  │
 │ deterministic partials  │   │ deterministic partials  │
 └────────────┬────────────┘   └────────────┬────────────┘
              │                             │
              └──────────┬──────────────────┘
                         ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ deterministic combine/reduction; capture-safe replay; bounded error │
 │ Engram mmap/page-cache retained; prefetched into fixed device slots │
 └──────────────────────────────────────────────────────────────────────┘
```

The target does **not** require Python to disappear. It requires Python to stop being on any measured hot per-layer loop while remaining the safest place for policy changes.

## Before/after component ownership

| Component | As-is owner | Target owner | Migration timing |
|---|---|---|---|
| Request admission and batching | Python scheduler on rank 0 (**measured**) | Python policy plane (**projected**) | Keep throughout |
| Dense FP8 and attention | CUDA/Triton per TP rank (**measured**) | CUDA/Triton per TP rank (**projected**) | Kernel improvements before multi-rank |
| GPU routed experts | FlashInfer CUTLASS per TP rank (**measured**) | Native GPU kernel per TP/EP placement (**projected**) | Keep unless profiling changes priority |
| CPU-expert ownership | rank-0 kt-kernel only (**measured**) | static rank-balanced partitions first (**projected**) | Python prototype in Stage 3 |
| CPU-expert dispatch | Python/kt wrapper on rank 0 (**measured**) | Python-defined contract; native hot loop later (**projected**) | Native implementation after Stage 3 |
| CPU partial reduction | rank-0 result enters existing TP reduction (**inferred**) | deterministic, explicitly specified cross-rank combine (**projected**) | Define/test in Stage 3 |
| Engram file state | rank-0 mmap/page cache (**measured**) | duplicate mappings or shared state by measured trade-off (**projected**) | Decide in Stage 3 |
| Engram prefetch/staging | Python thread pool and eager graph gaps (**measured**) | fixed host/device slots and event hand-off (**projected**) | Build buffers before; assign owners in Stage 3 |
| Streamed-prefill export | rank-0 kt export + shared buffer + barriers (**measured**) | partitioned export or retained single exporter, selected by profile (**projected**) | Optimise layout before; ownership after |
| Placement policy | environment flags/Python (**measured**) | Python with versioned maps (**projected**) | Keep in Python |
| Failure and rollback | process-level failure; E24 open (**measured/pending**) | per-request error, mode fallback, bounded timeout (**projected**) | Semantics in Stage 3; production in Stage 6 |

## Language and implementation decision matrix

### Criteria

Choose the least native layer that meets all required properties:

- **Latency sensitivity:** whether Python launch/dispatch appears on the measured critical path.
- **Data locality:** whether code directly transforms GPU/host buffers.
- **Capture safety:** stable addresses, streams, events, and replay semantics.
- **Change rate:** policy and experimental logic should remain easy to alter.
- **Correctness surface:** distributed ownership and failure semantics should stay inspectable until validated.
- **Portability:** avoid tying stable policy to one device generation or one transport.

| Component | Python | C++ | CUDA/Triton | Rust | Decision |
|---|---|---|---|---|---|
| Admission, batching, placement policy | Best fit | Possible, low value | Not applicable | Possible later | **Python**: high change rate, not yet measured hot |
| Ownership-map construction | Best for prototype | Best after stable if hot | Not applicable | Possible | **Python first**, C++ only after Stage 3 |
| Rank-local dispatch queue | Prototype/control | Best for proven hot loop | CUDA only for device queueing | Good for sidecar, extra ABI | **C++ after measurement** |
| CPU MXFP4 expert compute | Wrapper only | Existing native library | Not applicable | No advantage | **C++/kt-kernel**, already native |
| Dense large-M block FP8 | Orchestration only | CUTLASS host glue | Best fit | No advantage | **CUDA/CUTLASS or DeepGEMM** before Stage 3 |
| Small-M tuned dense FP8 | Orchestration only | Possible | Existing Triton path | No advantage | **Triton**, retain measured E20 path |
| Engram ID/hash policy | Best fit | If measured hot | Fused kernel if graph-safe | No advantage | **Python/CUDA split**, contract first |
| Engram gather/dequant/copy | Too launch-heavy if per row | Host prefetch and mmap | Gather/dequant/copy | No advantage | **C++ + CUDA/Triton** behind fixed buffers |
| Deterministic partial combine | Reference implementation | Host coordination | Best for device reduction | No advantage | **Python reference, then CUDA/C++** |
| CUDA graph capture/replay glue | Policy only | Stream/event lifetime | Device kernels | No advantage | **C++/CUDA**, after semantics fixed |
| Remote sidecar transport | Avoid hot-path serialization | Strong fit | Endpoint staging only | Strong safety fit | **Wait**; Rust is optional only for a stable remote protocol |
| Metrics/config/experiments | Best fit | Export counters | Kernel timestamps | Possible | **Python**, native counters exposed upward |

Rust is not a default migration destination. It becomes reasonable only for a separate, stable, failure-isolated transport/service process; introducing it into the in-process CUDA path would add an ABI without removing a measured bottleneck.

## Dependency graph

```diagram
 ┌─────────────────────────────────────┐
 │ Stage 0: preserve qualified baseline│
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐
 │ Stage 1: profiles + contracts       │
 └──────────────┬──────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
 ┌────────────────┐  ┌─────────────────────────┐
 │ Stage 2:       │  │ E24 + E23 correctness   │
 │ low-coupling   │  │ gates remain blocking   │
 │ native wins    │  └────────────┬────────────┘
 └───────┬────────┘               │
         └──────────────┬─────────┘
                        ▼
 ┌─────────────────────────────────────┐
 │ Stage 3: Python multi-rank prototype│
 └──────────────────┬──────────────────┘
                    ▼
 ┌─────────────────────────────────────┐
 │ Stage 4: native consolidation       │
 └──────────────┬──────────────────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
 ┌────────────────┐  ┌─────────────────────────┐
 │ Stage 5:       │  │ Stage 6: production and │
 │ optional HW    │  │ focused upstream work   │
 └───────┬────────┘  └────────────┬────────────┘
         └──────────────┬─────────┘
                        ▼
             ┌────────────────────┐
             │ qualified releases │
             └────────────────────┘
```

Stage 2 can proceed beside E23/E24 diagnosis, but Stage 3 cannot become the default until correctness gates pass. Inside Stage 2 the B2 prototype (item 0) is independent of items 1–3 and is the first to run once Stage 1's attribution exists; its placement half is a Stage 3 input, not a Stage 2 deliverable. Stage 5 depends on stable Stage 3/4 contracts; it must not dictate them prematurely.

## Stage 0 — Preserve and stabilise the current qualified baseline

### Architecture boundary

Freeze the current behavioural reference: TP2, **measured 208 GPU / 176 CPU experts per layer**, rank-0 CPU ownership, file-backed Engram, D=4, speculation off, streamed prefill enabled with **measured default chunk 4096** and **measured optional chunk 8192**. Preserve the overlay lineage from **measured base `da64c5cb`** and exact applied patches.

### Entry criteria

- Existing R0 evidence and overlay commits are available (**measured**).
- No architecture migration begins from an unrecorded working tree (**inferred requirement**).

### Acceptance gates

- Reproduce **measured** R0 within a **projected ±5% tolerance**: C1 **38.7 tok/s per-stream summed / 36.0 wall**, C3 **111.7 aggregate / 98.2 wall**, C4 **148.3 aggregate / 131.9 wall**, with **measured 512-token outputs** and **measured TTFT about 1.0–1.6 s**.
- **Erratum 2026-09-23:** the C3/C4 R0 gates above are artifacts of kt-kernel bug C (only row 0 received correct CPU-expert output in the bs=4 graph; bs=3 padded into it). Replace them with the patched-image references C3 **38.3 summed / 35.8 wall tok/s (12.8 per stream)** and C4 **50.8 / 47.0 (12.7 per stream)**. See `docs/BENCHMARKS.md` “Erratum” and `docs/TROUBLESHOOTING.md` bug C.
- Preserve E22 warm prefill: **measured 1,375 tok/s at 16K** and **measured 1,404 tok/s at 32K** with chunk 8192; keep chunk 4096 as the current default.
- Resolve E24's remaining NaN causes and prove per-request rejection rather than server death (**pending**).
- Resolve E23 target-verify correctness before treating speculation as a migration accelerator (**pending**). Preliminary **measured 2026-09-23** evidence: K=2 graph mode first crashed because SM120 page-split scratch buffers grew after capture and freed tensors remained graph targets; overlay commit `838490a6` uses retire-not-free plus pool-geometry pre-sizing. A second captured-graph multi-stream corruption is **measured preliminary** and currently avoided with `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`.
- Pass NLL, 16K needles, and the **pending ≥20K-token mixed soak** with zero NaN/assert/new device error.

### Rollback

One launch-mode switch returns to D=4, speculation off, rank-0 ownership, file-backed Engram, and the known overlay revision. Keep evidence and config snapshots for each candidate.

### Expected benefit and cost

- **Projected benefit:** a trustworthy control prevents performance work from hiding correctness regressions.
- **Projected cost:** no direct throughput gain; engineering time goes to reproducibility and defect closure.

## Stage 1 — Profile and formalise the Python↔native contract

### Architecture boundary

Python continues to coordinate. Native kernels continue to compute. Instrument the boundary without redesigning ownership.

Define a versioned per-layer contract containing: tensor shapes, dtypes, strides, expert-id namespace, placement-map version, route weights, output accumulation dtype, fixed buffer addresses, producer/consumer streams, event ownership, batch/verify width, timeout, cancellation, and error reporting.

### Entry criteria

- Stage 0 reference is reproducible (**pending gate**).
- Existing step-profile tooling can attribute graph segments and CPU gaps (**measured capability**).

### Acceptance gates

- Produce per-layer and per-step timelines that separately report CPU expert compute, queue/submit, export conversion, host copy, barrier, H2D/D2H, GPU expert compute, Engram prefetch, Engram gather/dequant, graph breaks, TP collective, and Python launch (**pending**).
- Split CPU-expert cost into per-token bytes/work and per-step/per-layer fixed cost. This is required because preliminary K=2 results are **measured 2026-09-23** at C1 **38.9 median / 35.5 wall** but C3 only **47.8 aggregate / 42.1 wall**, versus **measured R0 C3 111.7 aggregate**. The **inferred** cause is speculative verification multiplying CPU-expert work per step on this CPU-bound hybrid.
- Add contract tests for shape/stride errors, stale ownership maps, batch-size transitions, graph replay, cancellation, and injected native errors (**projected test suite**).
- Establish p50/p95/p99 timing, not only averages (**pending**).
- Measure CPU/CCD affinity and memory locality for each process and worker group (**pending**); retain the **measured 124.7–125 GB/s** host read ceiling and **measured two-CCD topology** as context.

### Rollback

Instrumentation must be removable or disabled with one flag. Contract wrappers retain the old direct call path until their equivalence tests pass.

### Expected benefit and cost

- **Projected benefit:** prevents optimising export when barriers dominate, or rewriting Python when native compute dominates.
- **Projected cost:** added tracing perturbation and temporary duplicate adapters; report overhead and use sampling outside qualification runs.

## Stage 2 — Low-coupling native wins before multi-rank

### Architecture boundary

Keep rank-0 ownership and Python coordination. Change only kernels and dataflow components with contracts that remain valid under any later rank partition.

### Work ordered by durability

0. **Option B2 decode prototype — PCIe-streamed experts beside CPU-computed experts (added 2026-09-23). Measured E30 2026-09-24: +2 % at |S| = 40, −3 % at |S| = 60 → stop condition "C1 does not move at |S| ≈ 40" hit; B2 rejected as a decode lever, code retained off by default.** As built it was symmetric TP (each rank its own slice), direct UVA compute rather than staged DMA, and the stock FlashInfer SM120 grouped GEMM rather than E13 — none of which changes the verdict, because the measured per-expert link cost (0.21 ms) exceeds kt's (0.146 ms). Original design text follows. Rank 0 only; kt-kernel path unchanged for the experts it keeps. Each decode step: DMA a fixed subset *S* of the CPU-resident experts' MXFP4 weights (chosen once at launch, e.g. the ≈ 40–60 highest-frequency CPU-tier experts per layer from the E7/E28 placement census) from pinned or page-cache host memory into a fixed device buffer on a side stream, overlapped with attention and the GPU-resident experts; compute them with the E13 kernel (`kernels/e13/`, **measured 1,397 GB/s at M=1**); merge with the kt outputs at the existing combine point. Why it is low-coupling: the buffer, the DMA schedule and the E13 dispatch have no rank in their contract — a later rank split changes *which* experts stream from *which* rank, not how. Why it is first: it is the only remaining lever whose ceiling the byte model puts above 60 tok/s on this box (**projected 79 tok/s split vs 52 CPU-only**, `feasibility-calc.py --dram-gbps 131`), and every environment knob is exhausted (E26). Prerequisites: Stage 1's per-step timeline (E26 A3's GPU-only step profile is a partial substitute: **measured** gpu_busy 15.4 ms of a ≈ 27 ms C1 step, every large gap a wait for the CPU tier); a pinned-host copy or page-cache-resident view of the streamed experts' weights (kt packs all experts per layer into host RAM already — **measured 8 GiB/layer**, so no new host footprint if the view aliases kt's buffer; a separate pinned copy costs |S| × 17.9 MiB × 40 layers). Go/stop gate: ≥ +10 % C1 wall at a fixed 226/158 placement with |S| ≈ 40 (**projected** +12–15 % if additivity holds: ≈ 0.7 GB/step leaves the CCD path), zero NaNs on the streamed experts (`verify-batch-consistency` + NLL gate), and the HSMP DRAM counter reading ≈ CCD + PCIe bytes rather than contention; stop if C1 does not move at |S| ≈ 40 (additivity fails under a real forward — record it, it kills B2 outright) or if the side-stream DMA extends the step (overlap fails — retune before abandoning). Rollback: `|S| = 0` is today's path. Tracks: ROADMAP layout B2, K7 (unblocked by this item), TASKS T3.6e–g tail.
1. **Large-M block-scaled FP8 prefill kernel.** Keep E20's small-M Triton configs; introduce CUTLASS/DeepGEMM for large-M shapes behind shape-based dispatch. E20 measured **about 15% of FP8 peak at M=4096** and measured **2.2–3.0×** gains over the old default at that shape, leaving a native-kernel opportunity independent of ownership.
2. **Fixed Engram buffers and early prefetch.** Keep mmap/page cache. Allocate fixed host/device slots; issue layer-14 requests while layer 1 runs; use residency-aware prefetch; copy into graph-consumable device addresses. Current Engram lookup is **measured 1.02 ms cold per token**, **measured about 3 µs warm**, but the two eager graph breaks are **measured 2.4–3 ms per step**.
3. **Export optimisation only when proven.** Current streamed-prefill export is **measured 51.8 GB/s** in isolation and **measured core-bound**; server export is **measured about 90–117 ms per layer**. Optimise conversion/copy loops only if Stage 1 attributes a dominant fraction there. If barriers dominate, change buffering/barrier granularity without embedding rank ownership.
4. **Tests and references.** Preserve byte-exact layout checks, FP32/dequant references, finite checks, and graph-address lifetime tests.

### Entry criteria

- Stage 1 has trustworthy attribution for the candidate path (**pending**).
- The candidate interface does not encode rank 0 or a fixed number of ranks (**projected design gate**).

### Acceptance gates

- Kernel correctness matches existing tolerances and all outputs are finite (**pending**).
- End-to-end prefill tok/s and TTFT improve without C1/C4 decode regression beyond a **projected 5% tolerance**.
- Fixed-buffer replay survives all captured C1–C4 and verify widths in the candidate configuration (**pending**).
- Engram graph-break time decreases from the **measured 2.4–3 ms per step** toward a **projected ≤1 ms per step target** without changing lookup results.
- Export changes reduce measured server-side export or barrier time by a **projected ≥15%** before they are retained.
- B2 prototype (item 0): ≥ +10 % C1 wall at fixed placement with ≈ 40 streamed experts per layer, zero NaN, NLL within the standing gate (|Δ| < 0.002), no C3/C4 per-stream regression beyond the **projected 5 % tolerance** — **failed (measured E30: +2 % at 40, −3 % at 60; zero NaN and no C4 regression did hold)**.

### Rollback

Per-component backend switches select existing Triton, current Engram eager gaps, and current exporter. No checkpoint conversion or irreversible state migration.

### Expected benefit and cost

- **Projected benefit:** faster prefill/TTFT, fewer graph breaks, and reusable fixed-buffer primitives for Stage 3.
- **Projected cost:** kernel maintenance across SM/device versions and extra VRAM/pinned-memory pressure; all buffer sizes require explicit accounting.

## Stage 3 — Multi-rank CPU-expert and Engram architecture

### Architecture boundary

Redesign ownership in Python first while reusing Stage 2 native primitives. The initial implementation is static and explicit, not dynamic or adaptive.

### Static partition first

- Partition CPU expert IDs deterministically between TP ranks; record a placement-map generation with every dispatch (**projected**).
- Assign Engram prefetch/gather ownership independently from CPU experts. Test duplicate mmap mappings, shared file/page-cache state, and one-owner-plus-broadcast; choose by measured p99 and memory pressure (**pending**).
- Each rank produces a deterministic partial output; define reduction order and accumulation dtype. Compare rank-split outputs against the rank-0 reference at layer boundaries and final logits (**pending**).
- Pin workers and buffers by CCD/NUMA locality. The host has **measured two CCDs** and **measured about 62–63 GB/s per CCD**; a static one-rank-per-local-worker-group design is the first hypothesis, not a result.
- Specify failure semantics: timeout, rank loss, stale map, cancellation, buffer overrun, partial-result rejection, and fallback to rank-0 mode.
- Capture/replay requires fixed addresses and stable ownership for every captured batch/verify width. No allocation, free, or map mutation may occur inside replay.

### Duplicate versus shared state

| Choice | Benefit | Cost/risk | Gate |
|---|---|---|---|
| Duplicate CPU metadata, partitioned weights | Simple rank isolation (**projected**) | More metadata and load complexity (**projected**) | Host-memory accounting + equivalence test (**pending**) |
| Shared CPU weights with rank-local queues | No duplicate packed weights (**projected**) | Contention and lifetime coupling (**projected**) | p99 and fault-isolation test (**pending**) |
| Duplicate Engram mmap mappings | Simple ownership; Linux page cache still shares file pages (**inferred**) | Duplicate virtual mappings and per-rank staging (**projected**) | RSS/page-cache and p99 measurement (**pending**) |
| One Engram owner + broadcast | One gather/dequant path (**projected**) | New per-layer rank dependency (**projected**) | Must beat local duplicate lookup at p99 (**pending**) |

### Entry criteria

- Stage 0 correctness gates pass (**pending**).
- Stage 1 contract and traces exist (**pending**).
- Fixed buffers needed for graph replay have passed Stage 2 address-lifetime tests (**pending**).

### Acceptance gates

- Layer-output and final-logit equivalence against rank-0 ownership within the established reference tolerance (**pending**).
- Deterministic reduction under repeated C1–C4 and captured verify shapes (**pending**).
- No rank waits indefinitely; injected timeout/cancellation produces a bounded request error and the next request succeeds (**pending**).
- CPU work and wait time are balanced by measured rank timelines; a **projected target** is no rank carrying more than **60%** of CPU-expert work under uniform routing.
- C1 and C4 wall throughput do not regress beyond a **projected 5% tolerance** before optimisation; retain only if CPU-wait-per-token or p99 has a measured path to improvement.
- NLL, 16K needle, mixed-batch, and **pending ≥20K-token soak** gates pass.
- CUDA graph capture and replay pass all supported batch sizes without buffer growth or multi-stream corruption (**pending**).

### Native-decision checkpoint

Only now answer whether dispatch belongs in a C++/CUDA extension. If Python dispatch/queue/reduction consumes a material measured fraction of p99 or CPU wait, move that loop in Stage 4. If native compute and collectives dominate, keep coordination in Python.

### Rollback

Retain a runtime ownership mode: `rank0`, `static-split`, and later `native-split` (**projected names**). A request may fall back only before execution starts; mid-step partial fallback risks double application and is rejected.

### Expected benefit and cost

- **Projected benefit:** use both TP ranks' host-side capacity, reduce rank-1 spin, improve overlap, and establish reusable EP/third-card semantics.
- **Projected cost:** distributed failure modes, additional reduction/dispatch points, duplicated staging, stricter graph lifetime, and more difficult diagnosis.

## Stage 4 — Post-migration native consolidation

### Architecture boundary

Preserve Stage 3 semantics exactly. Replace implementation, not policy.

Candidate native units are: rank-local queue submission, placement-map lookup, compact ID remap, fixed-buffer state transitions, event/barrier fan-in, deterministic partial combine, and low-overhead timestamp/counter export. Keep admission, workload policy, placement experiments, fallback selection, and release controls in Python.

### Entry criteria

- Python static-split mode passes Stage 3 correctness, failure, graph, and soak gates (**pending**).
- Stage 1/3 traces identify specific Python loops or synchronisation as hot (**pending**).
- The ownership and error contract has stopped changing across at least one qualification cycle (**projected criterion**).

### Acceptance gates

- Native and Python split modes pass the same conformance suite and produce equivalent layer/logit results (**pending**).
- Each native move has a measured end-to-end benefit; a **projected retention bar** is **≥5% wall throughput**, **≥10% CPU-wait reduction**, or a material p99 reduction with no correctness loss.
- ABI/config are versioned; incompatible map or buffer versions fail before capture (**pending**).
- Re-profile the full step after every consolidation because bottlenecks move (**required**).
- Python remains able to select rank-0 mode and run experiments without rebuilding native code (**projected operability gate**).

### Rollback

Select Python static-split or rank-0 mode at launch. Native extensions must not mutate checkpoint files or persistent caches.

### Expected benefit and cost

- **Projected benefit:** lower launch, queue, and synchronisation overhead while retaining inspectable policy.
- **Projected cost:** ABI maintenance, native debugging, compiler/device matrix, and the risk of optimising a bottleneck that Stage 3 already moved.

## Stage 5 — Optional hardware and topology extensions

### Architecture boundary

Extend the stable ownership/dispatch contract; do not redesign it around unqualified hardware.

### Candidates

1. **Third GPU, TP2 + asymmetric EP.** Plain TP3 is invalid because **measured model fact: 64 attention heads do not divide by 3**. A third card holding **projected about 90–120 complete experts per layer** uses the Stage 3 dispatcher and deterministic combine. Measure PCIe latency before purchase/commitment.
2. **DSpark/MTP.** Treat speculation as workload multiplication, not a free GPU optimisation. Preliminary **measured 2026-09-23** clean K=2 results are C1 **38.9 median / 35.5 wall** (R0 parity) and C3 **47.8 aggregate / 42.1 wall**, a **measured preliminary about 2.3× per-stream regression** versus R0 C3 aggregate **111.7**. The lever is CPU-expert cost per verified token and fixed cost per step, both separated in Stage 1.
3. **262K qualification.** Capacity is not qualification. Require context-length curves, chunk memory, correctness, cache behaviour, p99, and soak (**pending**).
4. **Prefix/KV caching.** Add only after benchmark methodology guarantees unique prompts for cold-prefill measurements. Existing evidence shows prefix hits can invalidate prefill numbers (**measured**).
5. **Remote Mac sidecar.** Wait for transport gates: **projected p99 application RTT ≤0.15–0.20 ms** and measured Metal expert advantage sufficient to beat local CPU plus **measured model fact: 40 dependent MoE layers**. Do not build remote hot-path integration before those gates.

### Entry criteria

- Stage 3 ownership protocol and Stage 4 ABI are stable (**pending**).
- Each extension has an independent stop/go microbenchmark (**pending**).

### Acceptance gates

- Third card: measured dispatch/reduce latency and end-to-end gain; existing **projected C1 range 42–60 tok/s** remains only a planning estimate.
- DSpark: meet E23 wall-throughput, per-stream, NLL, needle, and soak gates (**pending**).
- 262K: complete correctness and stability suite with reported TTFT and memory headroom (**pending**).
- Remote sidecar: transport, exact-kernel, synthetic-loop, and failure-recovery gates from the heterogeneous-routing note (**pending**).

### Rollback

Every extension is a launch profile. Core TP2 rank-0 and static-split modes remain available; external devices are never required to read the checkpoint or serve a request.

### Expected benefit and cost

- **Projected benefit:** fewer CPU experts, longer qualified context, or workload-specific throughput.
- **Projected cost:** hardware, PCIe/network synchronisation, new failure domains, memory pressure, and larger qualification matrices.

## Stage 6 — Production, reproducibility, and upstreaming

### Architecture boundary

Turn qualified modes into versioned artifacts without merging experimental policy into native internals.

### Work

- Export/package the overlay or fork from **measured base `da64c5cb`** with an explicit patch series, build inputs, licences, and provenance.
- Build unit, contract, graph-replay, multi-rank, failure-injection, NLL, needle, long-context, and soak CI tiers. GPU-required tests remain separately labelled (**projected CI design**).
- Produce versioned release artifacts and configuration schemas; reject unknown ABI/config combinations.
- Expose per-rank queue time, CPU compute, export, H2D/D2H, barriers, Engram residency/I/O, graph breaks, collectives, p50/p95/p99 inter-token latency, and error counters.
- Ship rollback modes and a release runbook.
- Upstream focused units: small-M SM120 config support, tuner improvements, correctness fixes, then stable generic interfaces. Do not upstream the entire hybrid as one PR.

### Entry criteria

- At least one multi-rank mode passes Stage 3/4 gates (**pending**).
- E24 release-defect gate passes (**pending**).
- Licences and the overlay's durable source form are complete (**pending** per open-source plan).

### Acceptance gates

- A clean checkout can reproduce the selected mode from pinned artifacts (**pending**).
- CI catches stale ownership maps, graph buffer lifetime errors, reduction nondeterminism, and per-request failure regressions (**pending**).
- A release candidate passes the milestone matrix below and rollback is exercised, not merely documented (**pending**).
- Observability distinguishes page-cache residency from actual Engram I/O and per-token from per-step CPU cost (**pending**).

### Rollback

Keep the previous release artifact and rank-0 profile; configuration migration is additive and versioned. Failed native-extension load falls back only at process start.

### Expected benefit and cost

- **Projected benefit:** reproducible deployment, diagnosable regressions, reviewable upstream changes, and bounded operational risk.
- **Projected cost:** packaging and CI maintenance across CPU, CUDA, graph, and topology combinations.

## Milestone metrics and gates

Every candidate records all rows; an optimisation cannot hide a regression by reporting only its target metric.

| Metric | Current reference | Stage 2 gate | Stage 3 gate | Stage 4/6 release gate |
|---|---:|---:|---:|---:|
| C1 decode | **measured 38.7 tok/s summed; 36.0 wall** | **projected ≥95% of reference** | **projected ≥95% before tuning; then measurable gain** | **projected no regression; target set from Stage 3 best** |
| C4 decode | **measured 148.3 aggregate; 131.9 wall** | **projected ≥95% of reference** | **projected ≥95% before tuning; balanced rank work** | **projected no per-stream regression** |
| Prefill, 16K/32K | **measured 1,375/1,404 tok/s warm at chunk 8192** | **projected improve or preserve** | **projected ≥95% of Stage 2** | **projected qualified cold/warm curves** |
| TTFT | **measured about 1.0–1.6 s for R0 decode fixtures; 11.6 s at 16K and 22.8 s at 32K with chunk 8192** | **projected no >5% regression** | **projected no >5% regression** | **projected p50/p95/p99 SLOs from measured candidate** |
| p99 inter-token latency | **pending baseline** | **pending, must record** | **projected no regression; reduction required for a latency claim** | **projected release SLO from measured Stage 3/4** |
| CPU wait per accepted token | **pending unified counter; measured about 40 waits/step in non-spec decode** | **pending record** | **projected reduce and balance across ranks** | **projected alert threshold from qualified distribution** |
| Engram graph-break time | **measured 2.4–3 ms/step** | **projected ≤1 ms/step target** | **projected preserve or improve** | **projected SLO from measured result** |
| Correctness | **measured 16K verify pass at baseline** | same outputs/tolerances (**pending**) | layer/logit equivalence + deterministic reduction (**pending**) | full conformance suite (**pending**) |
| NLL | **measured D=4 ΔNLL −0.0061 nats/token vs D=0; E22 ΔNLL −0.0013** | no unexplained drift (**pending**) | no ownership-induced drift (**pending**) | candidate-specific gate; speculation requires **pending absolute ΔNLL <0.002** |
| Needle | **measured 16K pass** | 16K pass (**pending**) | 16K pass and long-context candidate pass (**pending**) | 16K plus qualified 262K if claimed (**pending**) |
| Soak | **pending E24 release gate despite prior clean runs** | zero new failure in focused soak (**pending**) | **pending ≥20K-token mixed soak** | **pending ≥20K-token minimum plus production-duration target** |

**Erratum 2026-09-23:** the historical C4 reference row is retained for provenance but is a bug-C artifact. Apply the percentage gates to the qualified **50.8 summed / 47.0 wall tok/s (12.7 per stream)** reference. Any migration or rank-balancing projection that assumed C4 ≈ 4 × C1 is **pending recomputation**.

The speculation NLL threshold above is a gate, not a result.

## Why not native-rewrite now

A native rewrite now would encode today's topology precisely where Stage 3 is expected to change it. The following code/assumptions may be invalidated:

- **Rank-0-only CPU wrapper logic:** construction, compact expert-ID remapping, submit/sync ownership, and the assumption that other ranks contribute zero CPU output.
- **Current placement map:** contiguous **measured 208/176 split**, one compact CPU prefix/view, and one rank's physical→logical namespace.
- **Streamed-prefill coordinator:** one rank exporting, one POSIX shared buffer, per-rank slices, and **measured two barriers per layer**. A rank partition may remove, duplicate, or reorder all of these.
- **Engram eager-gap ownership:** rank 0 issues mmap/WILLNEED/gather/dequant/H2D. Multi-rank may choose duplicate mappings, partitioned lookup, or one owner with broadcast.
- **Reduction placement:** today's CPU partial enters the existing TP flow from one rank. Split ownership requires an explicit deterministic combine and may change collective ordering.
- **Graph capture assumptions:** captured addresses and widths currently follow one owner's buffers. Per-rank pools, cancellation, verify width, and replay lifetime alter the state machine.
- **Failure handling:** one owner failing currently kills the process path. Split ownership needs timeout, partial-result rejection, request cancellation, rank recovery policy, and safe restart.
- **Affinity and memory allocation:** one **measured 16-thread** kt pool uses both CCDs. Static rank-balanced ownership may use CCD-local pools and partitioned packed weights.
- **Metrics:** current timing aggregates compute, export, queue, barriers, and copy in places. A native rewrite before Stage 1 would make attribution harder, not easier.
- **Speculative verification assumptions:** preliminary **measured 2026-09-23** C3 regression shows that verify changes CPU work shape. A native dispatcher designed around one-token decode could be immediately obsolete.
- **Future EP/third-card transport:** a rank-0-specific native API would not naturally express TP2 + asymmetric EP, while a measured ownership contract can.

Native code is harder to unwind because buffer lifetime, graph capture, stream ordering, and ABI become coupled. The correct sequence is to make semantics explicit in Python, validate and measure them, then lower only stable hot loops.

## Cross-stage rollback modes

| Mode | Purpose | Status |
|---|---|---|
| `rank0` | Current qualified architecture | **measured baseline** |
| `rank0-fixed-buffers` | Stage 2 kernels/staging with old ownership | **projected** |
| `static-split-python` | Stage 3 semantic reference | **projected** |
| `static-split-native` | Stage 4 production candidate | **projected** |
| `extension-profile` | Third card, DSpark, long-context, or remote sidecar | **projected** |

Rollback is selected at launch. Persistent model files and Engram shards remain unchanged in every mode.

## Stop conditions

- Stop Stage 2 work if profiling does not show the candidate on the critical path (**projected policy**).
- Stop Stage 3 promotion on any unexplained NaN, graph corruption, nondeterministic reduction, deadlock, or ownership-induced NLL/needle failure (**pending gates**).
- Stop native consolidation if the Python prototype is not semantically stable or if a native unit misses its retention bar (**projected policy**).
- Stop optional topology work when transport/dispatch overhead consumes the projected compute saving (**pending measurement**).
- Stop upstream work that combines architecture, performance, and correctness changes into one unreviewable patch (**inferred maintainability rule**).

## Recommended implementation order

1. Close E24 and E23 correctness while preserving the R0 launch profile (**pending**).
2. Add the Stage 1 contract and timelines, including per-token versus per-step CPU costs (**projected work**).
3. Implement and qualify large-M FP8 dispatch and fixed Engram staging independently (**projected work**).
4. Optimise kt export only if the trace says conversion/copy dominates (**conditional projected work**).
5. Build static rank-balanced CPU-expert ownership in Python with deterministic reference reduction (**projected work**).
6. Add Engram ownership alternatives and select by measured p99/memory/failure behaviour (**pending decision**).
7. Qualify capture/replay, NLL, needles, mixed soak, failures, and rollback in `static-split-python` (**pending**).
8. Lower only measured hot loops into C++/CUDA and freeze the first ABI (**projected work**).
9. Re-profile and set measured production SLOs (**pending**).
10. Extend to third-card EP, DSpark, 262K, caching, or remote transport only through independent stop/go gates (**pending**).
11. Package the overlay/fork and submit focused upstream changes after the architecture has a qualified reference (**projected work**).

The durable conclusion is simple: **native kernels and fixed dataflow first; multi-rank semantics in Python second; native coordination third; optional topology and broad production integration last.**

## Gate ledger (added 2026-09-24; the enforced sequence)

This section is the single place where the stage prose above is bound to task rows in [`plan/TASKS.md`](../../plan/TASKS.md) and to debt items in [`MAINTAINABILITY-REVIEW-2026-09.md`](MAINTAINABILITY-REVIEW-2026-09.md). A task in the ledger is *done* only when its exit checklist is fully ticked here; the T3.16i review pass that closes each stage is what flips the checkbox. Numbers quoted as "current reference" are the 2026-09-24 qualified baseline (T3.16d serve gate: C1 41.92 summed / 39.86 wall tok/s, TTFT 0.609 s, `evidence/t316-review-intake/c1-off6-t316c.json`; C4 E29 control 58.09 summed / 55.79 wall = 13.9 per stream). Everything in the "gate" columns is **projected policy** until the row's evidence link is filled.

Owners: **owner** = hyeomans (approvals, BIOS, reboots, sudo, GPU windows); **agent** = the Amp session executing the row. Status values: `open`, `in progress`, `done <date> <sha>`, `closed-by-decision <date>` (Stage 4 only, see the decision record).

### Enforced order

```diagram
G0 correctness closure ─▶ G1 T3.19 Stage 1 seams ─▶ G2 T3.16j B2 retirement ─▶ G3 T3.20 / T3.21 Stage 2 ─▶ G4 T3.22 Stage 3 (owner approval) ─▶ G5 T3.23 Stage 4 or decision record ─▶ G6 T3.24 Stage 6 packaging
  (C4, C5 before;            (a..f, each step          (trigger: T3.19c            (stop/go per lever;             (Python prototype;               (only loops T3.22               (completion definition;
   C3-rest, T3.16e(3)         bit-identical NLL,        commit; keep only what      independent of each             rollback mode `rank0`             traces prove hot)              unsafe flags retired)
   inside T3.19a/b;           consistency PASS,         T3.20 has adopted)          other; may run in                stays launchable)
   C10 before G4)             C1 wall in band)                                      parallel)
                                    │
                                    └── T3.16i review pass 2 closes G1; pass 3 closes G4; a pass is a stage's exit condition, not an afterthought
```

Rules that make the sequence non-skippable:

1. **No Stage 2 productionisation, no T3.22 design approval request, and no native code** is committed to the overlay while any G1 checkbox is open. `ci.sh` cannot enforce this; the T3.16i pass and this ledger do, and `plan/TASKS.md` rows T3.20 to T3.23 carry the dependency in their status column.
2. Each G1 deliverable lands as **one behaviour-preserving overlay commit** with the per-step gate below recorded in `evidence/t319/<step>/`. A deliverable without its evidence directory is not done.
3. **Rollback stays launchable at every gate**: `scripts/serve-hybrid.sh` on the previous overlay commit must still serve the qualified profile (226/158, D=4, spec off, capture 1 2 3 4). If a step makes the previous commit un-launchable (image change, kt patch), the step is split.
4. A stage's **exit is the review pass**, not the last commit: T3.16i pass N refreshes the hotspot map and debt register, updates this ledger, and only then is the next stage's entry gate evaluated.

### G0 — correctness closure feeding T3.19

| Debt | Closure criterion (what makes it `done`) | Where it closes | Owner | Status |
|---|---|---|---|---|
| **T3.16e (3)** graph-vs-eager parity on one recorded layer | One serve dumps `hidden_states`/`topk_ids`/`topk_weights`/kt output for MoE layer 14 (the Engram layer) and layer 20 at bs 1..4 in eager and in graph replay; a CPU test replays the recorded inputs through the overlay's Python path against the dump; parity = bit-equal kt output (kt is deterministic per E29), documented tolerance only if a measured source of nondeterminism is named | inside **T3.19a** (the same recorded-tensor dump is the timeline's input) | agent | open |
| **C3 remainder** (review 7.5, 7.6): forward-boundary emptiness, adjacent-layer identity | `kt_contract.py` gains `check_forward_boundary` (no deferred slot outstanding when the forward ends; every deferred slot written at layer L is consumed at L+1 and nowhere else), metadata-only, graph-safe like `check_moe_boundary`; `test_kt_deferral_adjacency.py` gains the two negative tests; one serve with consistency PASS bs 1..4 | inside **T3.19b** (the asserts move into the `CpuExpertProvider` adapter as they are written) | agent | open |
| **C4** `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` launcher workaround, no forward-mode gate | The overlay refuses (or forces off, with one log line) multi-stream overlap whenever a DSpark verify graph is captured, in code, independent of the launcher env; the launcher line becomes redundant and is deleted; one CPU test asserts the refusal; TROUBLESHOOTING Bug A entry points at the code gate | **before G1 entry** (one overlay commit, no serve needed beyond the CPU test; spec is off by default so no GPU regression risk) | agent | **done 2026-09-24** overlay `5b531732`; `test_dspark_overlap_gate.py` 7/7; both branches seen live (`evidence/t316-review-intake/c4-overlap-gate/`) |
| **C5** post-capture growth survived, not rejected | kt patch 0002 / SM120 persistent buffers raise `KTContractError` on any request larger than the captured/registered size instead of growing; `run-kt-buffer-tests.sh` gains the negative test (currently 8/8); one serve confirms capture list 1 2 3 4 still passes consistency | **before G1 entry** | agent | **done 2026-09-24** overlay `9ecd54a6` (`SM120SplitBufferSealedError`, own type rather than `KTContractError`: it is the attention page-split buffer, not the kt boundary); `test_sm120_split_seal.py` 7/7 mutation-checked; serve: seal logged on both ranks, consistency PASS, C1 39.64 wall (`evidence/t316-review-intake/c5-split-seal/`) |
| **C10 / T3.13** E24 per-request NaN rejection unproven | Injected-NaN test (one request's logits forced non-finite under the breakable decode graph) shows that request fails with an error while the other three streams of a C4 batch complete; E22c 1,024-token-batch assert bisected to a cause or a pinned reproducer; recorded in `evidence/e24-nan/` and TROUBLESHOOTING | **before G4 entry** (T3.22 failure-propagation semantics presuppose per-request rejection) and the Stage 4/6 release gate; does **not** block G1 (owner decision 2026-09-24: E24 no longer gates the Python seams) | agent | open (E22c bisect is the current item) |

### G1 — T3.19 Stage 1: profile, contract, Python seams

**Entry gate** (all required):

- [x] T3.16d contract asserts and 14 CPU tests in `ci.sh` (`26a52d5`, overlay `b93cb915`)
- [x] T3.16e parts 1 and 2 in `ci.sh` (8 buffer tests, 9 deferral tests; `96e15b6`)
- [x] T3.16i pass 1 recorded (review §10, `77398b1`)
- [x] T3.18 verdict recorded (B2 rejected as decode lever, E30)
- [x] C4 closed (code gate for multi-stream overlap; overlay `5b531732`, 2026-09-24)
- [x] C5 closed (post-capture growth rejected; overlay `9ecd54a6`, 2026-09-24)
- [x] Baseline snapshot pinned for the whole stage (2026-09-24): `evidence/t319/baseline-pin.json` records sha256 of `c1-off6-t316c.json` (C1 41.92 / 39.86 / 0.609), Boot 12 `c4-off0.json` (C4 58.09 / 55.79), `nll-t316c.json` and `verify-c4-t316c.json`, plus the per-step gate numbers; re-taken only if the image or BIOS slot changes
- [x] No GPU experiment mid-flight (GPUs idle after the C5 serve gate teardown, 2026-09-24)

**Deliverables, acceptance tests, per-step gate.** Each row is one overlay commit (plus recipe evidence commit). Per-step gate for every row: `ci.sh` green (all CPU test suites), one serve on the qualified profile, `verify-batch-consistency --same-prompt` PASS at concurrency 1..4, NLL **bit-identical** to `nll-t316c.json` (0 tokens moved; T3.16d showed this is achievable for behaviour-preserving changes), C1 wall **≥ 39.0 tok/s** (−2 % band around 39.86; E29/T3.16d run-to-run spread was < 1 %), C4 per-stream **≥ 13.6** (−2 %), TTFT ≤ 0.65 s, 0 NaN/Xid/`KTContractError`, EDAC 0/0. Evidence in `evidence/t319/<letter>/` with the launch log redacted.

| Step | Deliverable | Acceptance test (beyond the per-step gate) | Status |
|---|---|---|---|
| T3.19a | Per-layer decode timeline: CPU expert compute / kt export / `sync` barrier / H2D / Engram break / Python launch, per layer and per step, at bs 1 and 4; recorded-tensor dump for T3.16e(3) | Timeline JSON with the six buckets summing to the measured step time within 5 %; parity test from T3.16e(3) passes; the two Engram breaks and the ~40 per-layer waits are visible as separate rows | **done 2026-09-24** (overlay `e66d44fb`, `KT_TIMELINE=1`; [`evidence/t319/a/`](../../evidence/t319/a/README.md)): GPU-event buckets explain **98.9 % (bs 1) / 99.7 % (bs 4)** of the unsampled replay period on both ranks; 40 per-layer rows + 2 break rows per step; per-step gate passed (C1 39.15 wall, C4 13.9/stream, NLL 0 moved, consistency PASS, 0 NaN/Xid). bs 1 = 23.6 ms: cpu_wait 7.9 (33 %), other_gpu 8.1 (34 %), export 2.5 (10.5 %), gpu_experts 2.5, Engram 1.8 (7.7 %); Python launch not exposed (host 10 ms ahead of the GPU). bs 4 = 68.7 ms: cpu_wait 48.6 (71 %). **Parity harness (T3.16e(3)) not yet run:** the recorded dumps exist (layers 14/20, bs 1..4, graph replay + eager), the replay-through-eager-kt comparison needs a callable seam and is the first consumer of T3.19b's `CpuExpertProvider` |
| T3.19b | `CpuExpertProvider` (what rank 0 asks kt for: `submit(layer, hidden, topk_ids, topk_weights) -> handle`, `sync(handle)`, `topology()`) and `BufferLease` (captured buffer identity + lifetime, replacing ad-hoc static tuples) as adapters around today's code; `kt_contract.py` asserts and the C3 remainder move inside | Existing 14 + 9 + 8 tests pass unchanged against the adapters; a fake provider passes a conformance test; C3 remainder negative tests pass | open |
| T3.19c | Runner-owned state: no module/class globals in `kt_ep_wrapper.py` or the deferral path; loader monkey-patching replaced by explicit loader arguments; `_b2_stream`/fork/join and `_uva_view` live here if kept | Two runner instances in one process share no state (CPU test); graph capture asserts unchanged (T3.16a tests) | open |
| T3.19d | Prefill coordinator: streamed prefill (`KT_GPU_PREFILL_MIN_TOKENS`, POSIX shared buffer, two barriers per layer) under one object with explicit buffer states and begin/end-forward lifecycle | State-machine unit test (every legal transition, every illegal one raises); 16K verify pass and E22a-style prefill tok/s within 5 % of the qualified curve | open |
| T3.19e | Typed config registry: every `KT_*`, `SGLANG_*`, `DSPARK_*`, `MEM_FRACTION_STATIC`, capture list and `EXTRA_*` knob declared once (type, default, aliases, `unsafe` flag, evidence link); `serve-hybrid.sh`'s `resolved:` line and `fail_config` checks generated from it | Unknown knob rejected; every current env alias still accepted; `fail_config` refusals from T3.16b reproduced by tests; the `resolved:` line is byte-identical for the qualified profile | open |
| T3.19f | Immutable evidence manifests: each serve writes `manifest.json` (recipe SHA, overlay SHA, image digest, resolved config, BIOS slot, GPU power cap, kernel) and a content hash; `bench-decode.py`, `nll-score.py`, `verify-batch-consistency.py` embed the manifest hash; `ci.sh` refuses evidence JSON without one after the cut-over date | Manifest round-trips; a benchmark file from before the cut-over is still accepted (dated exemption) | open |

**Exit gate** (all required, then T3.16i pass 2):

- [ ] a..f done with evidence directories
- [ ] Review R1 (`KTEPWrapperMethod` size) and R5 (single-source config) re-measured and updated; hotspot map refreshed
- [ ] One C1..C4 soak block (E27 style, ≥ 20K decoded tokens, 0 NaN/Xid) on the final Stage 1 overlay commit
- [ ] T3.16i **pass 2** recorded in review §11; this ledger's G1 status set to `done`
- [ ] ROADMAP change-log entry; LEARNING §11 Surprise row

Performance non-regression across the stage: final C1 wall and C4 per-stream within the −2 % band of the pinned baseline; TTFT within 0.05 s. Stage 1 promises **no** performance gain; a measured gain is recorded but is not a pass criterion.

### G2 — T3.16j B2 residue retirement

- **Trigger:** the T3.19c commit (runner-owned stream state) is merged into `dsv41-kt-hybrid`. Not earlier (the stream/UVA primitives would have nowhere to live), not later than the T3.19 exit (the exit review must not count B2 residue as hotspot 16 again).
- **Remove:** `_B2HostPool`, `_B2UvaExperts.run`, `remap_topk_ids_for_b2`, the `use_b2` branch, `submit(num_kt_experts)`, launcher knobs `KT_B2_UVA_EXPERTS` / `KT_B2_MAX_TOKENS` and their `fail_config` checks and the `resolved:` fields.
- **Keep only if measured-reusable:** `_b2_stream` + fork/join and `_uva_view`, and only if the T3.20 design note (written at G3 entry) names them. If T3.20 does not adopt them within its first commit, they go too. E30's evidence (bit-equal UVA grouped GEMM, graph-replayed) is the record; the code is not.
- **Gate:** `run-kt-buffer-tests.sh` green, one serve, consistency PASS bs 1..4, C1 wall in band. One overlay commit.

### G3 — T3.20 / T3.21 Stage 2 low-coupling native wins

Both may start after G1 exit; they are independent of each other and may run in parallel GPU windows. Neither touches ownership.

| Task | Depends on | Go (continue / promote) | Stop (record and park) |
|---|---|---|---|
| **T3.20** Engram fixed-buffer prefetch, graph-compatible staging | T3.19c (runner-owned state), T3.19a (measured break time per step: ~~E26 A3 reference 2.43 ms/step~~ **T3.19a measured 1.82 ms/step exposed GPU gap at bs 1 = 7.7 % of the 23.6 ms step, 2.9 ms at bs 4**; a 50 % cut is worth ≤ 0.9 ms ≈ 4 % C1), T3.16j (stream primitives placed) | Measured Engram break time per step drops by ≥ 50 % **and** C1 wall improves ≥ +3 % with NLL bit-identical and consistency PASS; then promote to the qualified profile | After one tuning round: < +1 % C1 wall, or the graph-break count cannot drop below 2, or fixed staging forces a worse expert/KV placement (fewer than 226 GPU experts) — park with the measured numbers |
| **T3.21** large-M block-scaled FP8 prefill GEMM (CUTLASS / DeepGEMM) behind the matmul contract | T3.19b (contract), E20 tuned-Triton reference (**measured** ≈ 15 % of FP8 peak at M=4096); bounded spike of two GPU windows to prove an SM120 block-scaled FP8 kernel exists and links in the image | 16K prefill tok/s ≥ +20 % over the qualified E22a curve **and** TTFT at 16K/32K improves, decode untouched (C1 wall in band), NLL bit-identical on the frozen corpus | No SM120 kernel after the two-window spike; or the kernel changes prefill numerics beyond the NLL gate; or the gain is < +10 % at 16K — park |

Neither task may add an env knob outside the T3.19e registry, and neither may change rank ownership; if either needs to, it is T3.22 work and waits.

### G4 — T3.22 Stage 3 multi-rank ownership, Python prototype (owner approval required)

**Entry gate:** G1 exit; G2 done; C10 closed; a written design record (`docs/research/T322-DESIGN.md`, projected name) covering partition, reduction order, failure semantics, NUMA ownership and rollback, **approved by the owner** before the first overlay commit. This is the largest architecture change in the project and the only one that is approval-gated by name.

**Python-prototype gates** (all measured before any native work):

| Gate | Criterion |
|---|---|
| Correctness | Layer/logit equivalence against `rank0` mode on recorded tensors (T3.19a dump) at bs 1..4; NLL on the frozen corpus within the T3.6g3-calibrated threshold (bit-identity is *not* required because the combine order changes; any difference must be explained by the reduction order and be **run-to-run bit-identical**) |
| Deterministic reductions | Fixed-order combine of per-rank CPU partials; 100 repeated decodes of the same prompt bit-identical; `verify-batch-consistency` PASS at every captured width |
| Failure propagation | Injected per-rank timeout and injected NaN each fail the affected request(s) only; the server keeps serving the other streams; process-level fault falls back to a clean exit with a named cause |
| NUMA / CCD ownership | Per-rank kt thread pool pinned to one CCD; HSMP DRAM counters show both CCDs drawing; CPU wait per token recorded per rank and balanced within 20 % |
| Graph capture / replay | Capture list 1 2 3 4 captured and replayed with per-rank fixed buffers; T3.16a/c asserts extended per rank; no allocation during capture |
| Rollback | `--mode rank0` (the current architecture) stays launchable from the same overlay commit and passes the same gate; switching modes is a launch-time choice only |
| Performance (promotion, not entry) | C1 wall ≥ baseline; C4 wall ≥ baseline with per-stream no regression; p99 ITL recorded; the second GPU's sampled utilisation rises from the observed 7–20 % (owner observation, nvtop) — measured with `nvidia-smi --query-gpu=utilization.gpu` at 1 Hz over a C1 block |
| Soak | 20K-token mixed C1..C4 soak, 0 NaN/Xid, EDAC 0/0 |

Exit = T3.16i **pass 3** recorded, with a per-layer timeline (T3.19a tooling) of the `static-split-python` mode that names every Python coordination loop and its share of the step.

### G5 — T3.23 Stage 4 native consolidation, or a decision record

T3.23 is **mandatory only for loops the G4 timeline proves hot**: a Python-side dispatch/reduction/synchronisation loop costing ≥ 5 % of the C1 step wall, or ≥ 1 ms/step, sustained across bs 1..4. Each such loop moves to C++/CUDA as one unit with a retention bar (the unit must keep ≥ 80 % of the Python-measured saving after re-profiling) and the ABI is frozen only after two units land or the decision record closes the stage.

If the G4 timeline shows **no** loop over the threshold, Stage 4 is closed by decision, not skipped silently. The record goes in this file under the ledger and in `plan/TASKS.md` row T3.23 as `closed-by-decision <date>`:

```
Decision record — no native migration justified (template)
Date / recipe SHA / overlay SHA:
Timeline evidence: evidence/t322/<run>/timeline.json (bs 1..4)
Hottest Python coordination loop: <name>, <ms/step>, <% of step>
Threshold: 5 % of step wall or 1 ms/step
Verdict: below threshold; native consolidation not justified at this topology
What would reopen it: <a measured change: e.g. third-card EP (T4.1b) adds a dispatch owner; DSpark verify changes CPU work shape>
Reviewer: owner
```

### G6 — T3.24 Stage 6 packaging, upstream cleanup, completion definition

Entry: G4 exit and G5 done or closed-by-decision. Work items, each with a checkbox in the TASKS row:

- Stable interfaces frozen and documented: `CpuExpertProvider`, `BufferLease`, prefill coordinator, config registry (typed profiles as files, e.g. `config/profiles/qualified-226-158-d4.toml`, replacing the env-var launch)
- Patch/fork organisation: overlay rebased into a reviewable series (SM120 W8A8 config/gate; block-32 tuner; FP8 scratch-buffer fix; kt capture asserts; contracts; seams; multi-rank) per [OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md); base SHA, branch, and build/run instructions reproducible from a clean clone
- Tests/CI: every CPU suite in `ci.sh`; one GPU smoke (`verify-batch-consistency` on the qualified profile) documented as the pre-release step
- Provenance: `PROVENANCE.md` and evidence manifests (T3.19f) cover every artefact the release claims
- **Deprecation and removal of unsafe flags and workarounds:** `KT_ALLOW_UNSAFE_DEFERRAL` (D ≥ 5 fails the NLL gate; remove the escape hatch once T3.6g3 fixes the threshold), `ALLOW_UNTESTED_FIT` (replaced by registry-declared fit rules), `EXTRA_SERVER_ARGS` / `EXTRA_DOCKER_ENV` / `EXTRA_DOCKER_ARGS` (kept only as a documented developer profile, never in a release profile), the `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` launcher line (gone at C4), the B2 knobs (gone at G2), `KT_NAN_PROBE_MIN_TOKENS` as a launch knob (becomes a registry field once C10's rejection path exists)
- Release artefacts: image digest, overlay tag, profile file, manifest, BENCHMARKS table regenerated from manifests, TROUBLESHOOTING entries for every retired workaround

**Completion definition — the project has left experimental-overlay form when all of the following are true:**

1. Every G0..G5 row in this ledger is `done` or `closed-by-decision` with an evidence link, and T3.16i has a recorded pass after G1, G4, and G6.
2. The qualified profile launches from a typed profile file with zero ad-hoc env vars, and every remaining knob is registry-declared with an evidence link.
3. No unsafe flag or launcher workaround from the list above remains in a release profile.
4. `rank0` and the promoted mode both launch from the same overlay commit and pass the same gate (rollback is a launch flag).
5. The overlay is a reviewable patch series on a pinned upstream SHA, reproducible from a clean clone, with licences and provenance recorded.
6. A release manifest exists whose C1/C4/prefill/TTFT/NLL/needle/soak rows were all produced by scripts that embed the manifest hash.

Until then the repository is an engineering recipe with an experimental overlay, and README/BENCHMARKS say so.

### Ledger status table

| Gate | Task rows | Owner | Status | Evidence |
|---|---|---|---|---|
| G0 | T3.16e(3), C3-rest, C4, C5, C10/T3.13 | agent | open (C4 and C5 done 2026-09-24; C3-rest and T3.16e(3) ride on T3.19a/b; C10 before G4) | review §10a, §10f |
| G1 | T3.19 a..f | agent | **entered 2026-09-24** (8/8); **T3.19a done 2026-09-24**; T3.19b next (carries the T3.16e(3) parity harness and the C3 remainder) | `evidence/t319/baseline-pin.json`, `evidence/t319/a/` |
| G2 | T3.16j | agent | waits for T3.19c | review §10c |
| G3 | T3.20, T3.21 | agent | waits for G1 exit | E26 A3, E20 |
| G4 | T3.22 | **owner approval** + agent | waits for G1 exit, G2, C10, design record | `evidence/t322/` (to be created) |
| G5 | T3.23 | agent | waits for G4 timeline | decision record or units |
| G6 | T3.24 | owner + agent | waits for G4 exit and G5 | OPEN-SOURCE-PLAN |

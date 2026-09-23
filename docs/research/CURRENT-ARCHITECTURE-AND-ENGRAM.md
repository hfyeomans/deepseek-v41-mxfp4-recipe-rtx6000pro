# Current architecture of the V4.1-Flash hybrid, and the Engram placement decision

> **Update 2026-09-23 (E28):** expert placement moved from 208 GPU / 176 CPU experts per layer to **226 / 158**, with the KV pool capped at 147,456 tokens (`--max-total-tokens`; ≈ 0.6 GB/card instead of 3.3) and `MEM_FRACTION_STATIC=0.96`. The diagram and RAM table below carry the new numbers; prose that says "176 CPU experts" describes the E19–E26 configuration and its measurements. C1 is now 41.2 summed / 36.1 wall tok/s (BENCHMARKS "E28").

Written 2026-09-22. §1 describes what `scripts/serve-hybrid.sh` actually runs today (overlay `dsv41-kt-hybrid` at `95f2fe3d`, image `dsv41-kt:recipe`), so that later changes can be diffed against it. §2 records the Engram decision (T3.12) with its measurements and RAM accounting. Labels per [README.md](README.md).

## 1. Current architecture

### 1.1 Hardware (measured, `docs/HARDWARE.md`, `evidence/hardware-inventory.json`)

| Part | Detail | Relevant measured limit |
|---|---|---|
| GPUs | 2 × RTX PRO 6000 Blackwell Max-Q Workstation Edition, SM120, 96 GB GDDR7 each, PCIe 5.0 x16, no NVLink, 300 W default / 325 W max cap | ≈ 1,470 GB/s GDDR7 copy (E2); 53 GiB/s H2D per card, 106 GiB/s both (E4) |
| CPU | AMD Threadripper PRO 9955WX, 16 cores / 32 threads, 2 CCDs | DRAM read 125 GB/s of 410 GB/s spec (30 %) at 16 threads; 94 GB/s while computing (E1b, E6 s0) |
| RAM | 8 × Kingston KF564R32-32 DDR5 RDIMM, configured 6,400 MT/s, ≈ 247 GiB usable | fabric-bound, not DIMM-bound: faster DIMMs would not help, more CCDs would (E1, E17) |
| NVMe | Samsung 9100 PRO, Crucial T705 | random-row rate keeps up with decode-shaped Engram gathers (E3, E18) |
| PSU | 1,500 W | both cards at 325 W + host fit; 4 cards would need 250 W caps |
| Display | a kiosk (`cage`) holds `/dev/nvidia0` and `/dev/nvidia1`; it blocks `nvidia-smi --gpu-reset` until stopped; a small dedicated display GPU is planned | — |

### 1.2 Model and weight formats

DeepSeek-V4.1-Flash from the Hugging Face snapshot (`dba1be0a…`), 48 safetensors shards. Three formats coexist in one forward pass:

| Component | Format in checkpoint | Kernel on this box |
|---|---|---|
| Dense: attention projections, shared expert, dense-layer MLPs | W8A8 block-FP8, `block_shape=[32,32]` | SGLang Triton `_w8a8_block_fp8_matmul` with the E20 SM120 configs (overlay `32a25f37`) |
| Routed experts (384 per layer, top-6) | MXFP4 (E2M1 + UE8M0 scales) | GPU tier: FlashInfer `flashinfer_mxfp4` fused MoE; CPU tier: kt-kernel MXFP4 AMX/AVX path, compact view (overlay `3805730e`) |
| Engram tables, layers 1 and 14 | `weight` F8_E4M3 `[384,006,168, 256]` + `scale` F8_E8M0 `[384,006,168, 8]`, 264 B/row, one contiguous tensor per table in shards 47 and 48 | file layout: mmap of the shard + two offsets, WILLNEED prefetch, `index_select`, dequant, H2D (overlay `a1ca246a`) |
| DSpark draft (three MTP stages, 128 experts each) | same as main | all-GPU (kt clamps `num_gpu_experts` ≥ 128); loaded only with `DSPARK=1` |

The E20 work is about the **dense W8A8 FP8** GEMM; the routed experts' MXFP4 kernels were already at the roofline (E12/E13) and were not changed. Keep the two apart when reading any "FP8" line.

### 1.3 Placement and data flow

```diagram
 request ──▶ SGLang scheduler (rank 0) ── MAX_RUNNING_REQUESTS=4, chunked prefill 4096 (8192 optional)
                     │
                     ▼
 ┌─────────────────────────────── TP2 (NCCL over PCIe) ───────────────────────────────┐
 │  GPU0 / rank 0                                  GPU1 / rank 1                      │
 │  dense FP8 (N sharded)        ◀── all-reduce ──▶ dense FP8 (N sharded)             │
 │  attention: MLA + sliding-window, KV pool ≈ 0.6 GB/card, page 256 (re-paged 64)    │
 │  226 GPU experts/layer (MXFP4)                  226 GPU experts/layer (MXFP4)       │
 │  Engram rows (file layout, graph break)         [waits: NCCL spin ≈ 97 % "util"]   │
 │  kt-kernel submit/sync  ─┐                                                          │
 └──────────────────────────┼──────────────────────────────────────────────────────────┘
                            │ pinned H2D/D2H per layer, one submit + one sync
                            ▼
              host CPU tier (rank 0 process, 16 kt threads, ≈ 1,880 % CPU)
              158 CPU experts/layer, compact MXFP4 view: 21.8 MiB/expert, ≈ 135 GiB
              deferral D=4: the 4 lowest-scored experts of a token run one layer late
                            │
                            ▼  (page cache, WILLNEED prefetch on 16 threads)
              NVMe: Engram shards 47/48 (188.83 GiB), never pinned, never copied
```

Per decode layer (measured shape, E19/E21): the router picks 6 experts per token; the ≤ 208 GPU-resident ones run in FlashInfer on both ranks; the CPU-resident ones are submitted to kt-kernel on rank 0 as one batch, the GPU tier and the next layer's work proceed, and the sync happens one layer later for the D=4 deferred experts. ≈ 40 CPU-tier waits per step, median ≈ 470 µs at bs=1 before deferral hid them (E21). Layers 1 and 14 additionally gather 48 Engram rows per token from the page cache: two graph breaks costing 2.4–3 ms per step (E19).

### 1.4 CUDA graphs

- Decode runs `--cuda-graph-backend-decode breakable` (BCG): the step is captured as graph segments with eager gaps where kt-kernel and the Engram lookup run. Capture sizes bs 1–4 (`CUDA_GRAPH_MAX_BS=4`). Eager mode costs 4–6× on the sibling V4-Flash (E15), so the graph is load-bearing.
- Prefill graph is disabled by the engine itself ("Breakable CUDA graph is incompatible with DeepSeek-V4"); prefill runs eager with three explicit graph breaks (Engram hash ids, low-ratio sources, attention-with-output) that E23 had to gate on `is_extend_without_speculative()` because `TARGET_VERIFY` also satisfies `is_extend()`.
- Overlay fixes to BCG: dataclass outputs (`cd55166a`), per-field row counts (`71645529`), output cap = bs × captured_req_width (`95f2fe3d`) — the last two exist because a speculative verify body returns bs × (K+1) rows while the runner keys graphs by bs.

### 1.5 Attention and KV

MLA with compressed KV plus a sliding-window pool (hybrid FULL + SWA). Compressed pools are paged at `page_size // ratio` (256-token page → c1 256, c2 128, c4 64, c128 2); FlashInfer's SM120 sparse-MLA prefill only compiles page sizes 64 and 2, so the c1/c2 extra pools are re-paged to 64 (overlay `cd55166a`, ≈ 3 GB/card). KV pool ≈ 3.4 GB/card non-speculative at 32K context; with the DSpark draft resident and `MEM_FRACTION_STATIC=0.935` the pool is 279,808 tokens. Free VRAM after a 32K prefill: ≈ 3.7 GiB (4K chunks) / ≈ 1.2 GiB (8K chunks); 16K chunks do not fit. ≈ 93.1 GB/card resident at 32K.

### 1.6 Streamed prefill (E22, default since 2026-09-22)

For every prefill chunk ≥ `KT_GPU_PREFILL_MIN_TOKENS=1024`, the coordinator (overlay `0138c6ad`) exports the 176 CPU experts of a layer from kt-kernel into a 6.5 GiB POSIX-shm pinned buffer shared by both ranks, DMAs each rank's slice into 2.27 GiB of GPU staging, and runs the FlashInfer SM120 fused MoE on them, layer by layer, double-buffered. Export runs at 51.8 GB/s on 16 threads (core-bound); host time per chunk is nearly flat in tokens (4.7–5.1 s at 4,096, 5.4–5.8 s at 8,192), so chunk size is the lever until VRAM runs out. Prefill 216 → 837 tok/s (4K chunk) → 1,375–1,404 (8K). Decode is unchanged (still on the CPU tier).

### 1.7 Request flow, end to end

1. HTTP request → scheduler on rank 0; prompt tokenised; prefix cache checked (the V4-Flash sweep showed cache hits invalidate prefill numbers — seeds differ per run).
2. Prefill in chunks of 4,096 (or 8,192): dense FP8 + attention on both ranks; routed experts on GPU; CPU-resident experts streamed to the GPUs per layer (§1.6); Engram rows prefetched from page cache (0.50 s per 4K chunk cold, E18).
3. Decode: one breakable-graph replay per step at bs = running requests; kt-kernel CPU tier for the 176 CPU experts with D=4 deferral; two Engram breaks; sampling inside the graph (the `probability tensor contains inf/nan` assert fires here).
4. Streaming response; `bench-decode.py` reports summed stream tok/s, wall tok/s, TTFT and (with DSpark) acceptance.

### 1.8 Bottlenecks, in order of measured weight

| Bottleneck | Evidence | Lever |
|---|---|---|
| CPU-tier fixed latency, ≈ 40 waits/step | E21: 0.19 ms per CPU expert ≈ 120 GB/s; run-queue wait ≤ 0.22 ms/step (not scheduling); 12 vs 16 threads no change | overlap (deferral D=4 done; speculation E23 pending); fewer CPU experts (third GPU) |
| Rank asymmetry | rank 0 ≈ 1,880 % CPU, rank 1 ≈ 120 %; GPU1 spins in NCCL | split dispatch across ranks (design item) |
| Engram graph breaks | 2.4–3 ms/step, two breaks | warm-cache measurement, A-full prefetch of layer 14 at layer 1, graph-compatible staging (§2.4) |
| Prefill export | 4.6 s fixed per chunk, core-bound | bigger chunks (VRAM-limited); more cores would help, more DRAM bandwidth would not |
| Dense FP8 GEMM | was 57 % of the step; now 3.4 ms/step after E20 | prefill side still Triton at ≈ 15 % of FP8 peak → CUTLASS/DeepGEMM later |
| KV/VRAM headroom | 3.4 GB pool; 1.2–5.7 GB free depending on chunk/draft | fewer GPU experts (costs CPU bytes), lower fraction, third GPU |

### 1.9 Persistent vs ephemeral state

| Persistent (survives a relaunch) | Ephemeral (lost at `docker rm -f dsv41-kt`) |
|---|---|
| Recipe repo (scripts, plan, evidence, docs) — local git, unpushed | Container filesystem, including anything pip-installed at runtime |
| Overlay repo `~/dev/src/sglang-dsv41-overlay`, branch `dsv41-kt-hybrid` — local git, unpushed; bind-mounted read-only into the container | Captured CUDA graphs (≈ 4 min to rebuild), KV pool, prefix cache |
| Image `dsv41-kt:recipe` (`config/Dockerfile.dsv41-kt`, kt-kernel wheel pruned by `config/prune_wheel.py`) | Page-cache residency of the Engram shards (survives a relaunch, not a reboot or `drop_caches`) |
| HF snapshot and E20 config JSONs (`kernels/e20/`, copied into the overlay) | Phantom-busy GPU0 after an assert (cleared only by `nvidia-smi --gpu-reset` or reboot) |
| `systemd/gpu-power-limit.service` (325 W) | `/tmp/serve-*.log` launch logs |

## 2. Engram placement decision (T3.12)

### 2.1 Decision

Keep the full FP8 Engram file-backed through `mmap` and the Linux page cache (the `file` layout, overlay `a1ca246a`). Do **not** move the 188.83 GiB of tables to tmpfs, `mlock` them, or copy them into pinned anonymous RAM. Prioritise: a representative warm-cache measurement under real text, early / A-full-style prefetch (issue layer-14 rows when layer 1 runs), graph-compatible staging, and reducing the two Engram graph breaks. Record warm and cold timings separately and distinguish page-cache residency from actual disk I/O.

### 2.2 Measurements (E18, measured, cold page cache after reboot, `evidence/e18-engram-file/`)

| Shape | `torch` (serial faults) | `preadv-16` | `willneed-16` | warm (`torch`) |
|---|---:|---:|---:|---:|
| T=1 (48 rows) | 6.1 ms | 1.67 ms | **1.02 ms** (21 µs/row) | 0.003 ms |
| T=16 (768 rows) | 58 ms | — | **5.0 ms** | 0.056 ms |
| T=4,096 chunk (196,608 rows) | 1.41 s | 1.17 s | **0.50 s** (2.5 µs/row) | 2.9 ms |

64 threads are slower than 16 at every shape (thread spawn exceeds the I/O). Rows are bit-exact against `safetensors`. The warm path adds 0.7 ms of fixed thread cost at T=1 if the prefetch is issued anyway — the `mincore`-gated prefetch is the open item. In the server the two Engram graph breaks cost 2.4–3 ms per decode step (E19 step profile); that is graph-break overhead, not I/O.

### 2.3 Why tmpfs / mlock / pinned RAM is unsafe here (RAM accounting)

| Consumer | GiB | Label |
|---|---:|---|
| Host RAM usable | ≈ 247 | measured |
| CPU experts, compact view, 158/layer × 40 × 21.8 MiB (176/layer × 40 = 150 GiB before E28) | ≈ 135 | measured (E17: 21.8 MiB/expert; 136 GiB at 160/layer) |
| Rank-0 process, kt-kernel buffers, engine, pinned H2D/D2H | ≈ 5–10 | inferred from the observed ≈ 155 GiB process footprint |
| Streamed-prefill shared pinned buffer | 6.5 | measured (E22) |
| Remaining for page cache, OS, kiosk | ≈ 80–85 | projected |
| Full Engram tables | 188.83 | measured |

A resident Engram needs 188.83 GiB; the box has ≈ 80–85 GiB left once the CPU tier is loaded. tmpfs and `mlock` both make the pages unevictable, so either the loader fails or the kernel reclaims the process's anonymous memory and swaps or OOM-kills the rank-0 process mid-serve. Pinned anonymous RAM (`cudaHostRegister`, the upstream host-table path) is the same 188.83 GiB plus a copy. Even a quantised table in DRAM (`feasibility-calc.py --engram dram-fp4|dram-q3k`: LibertAI FP4 97.3 GiB, lossy and unevaluated; q3_K 78.7 GiB) would take all of the ≈ 80 GiB margin — those options fit only against the design-time 112.8 GiB expert budget at 224/160, not the 208/176 placement that actually serves — for a lookup that is already 1 ms cold and 3 µs warm. The third GPU (notes §2) lowers the CPU-expert footprint by ≈ 100 experts/layer ≈ 85 GiB but still does not make 188.83 GiB pinned safe.

### 2.4 Linux page-cache semantics that the decision relies on

- `mmap` + `MADV_WILLNEED` asks the kernel to read the named byte ranges asynchronously; the kernel coalesces the 96 row reads (48 weight + 48 scale) into device queue depth without a Python thread per row — the reason `willneed-16` beats `preadv-16`.
- Page-cache residency is opportunistic: pages stay until memory pressure evicts them. The CPU-expert load is anonymous memory and wins under pressure, so residency of the Engram working set degrades gracefully to NVMe reads (1.02 ms/token) rather than to a crash.
- `drop_checkpoint_page_cache` runs only from `_HostTable.finish_load`, so the `file` layout keeps whatever the load warmed. A reboot or `echo 3 > /proc/sys/vm/drop_caches` resets it — every "warm" number must say which it is.
- Residency can be read without I/O with `mincore`; disk I/O can be read from `/proc/diskstats` or `iostat` on the shard's device during a bench. A warm-cache run whose `iostat` shows zero reads is page-cache residency; anything else is disk.

### 2.5 Follow-up experiments on Engram (pending)

1. **Warm-cache measurement under real text** — E11's question: hit rate of the 48-row working set across a 20K-token mixed soak, with `mincore` residency and `iostat` reads logged per step; result decides whether the cold 1 ms ever appears in steady state.
2. **A-full prefetch** — issue layer-14 rows at layer 1 (E6-DESIGN §4.3); at 1.02 ms cold that is ≈ 2 ms/token, ≈ 4 % of a 55 tok/s step — not first-serve critical, but it removes the second cold miss.
3. **Graph-compatible staging** — pre-gather the rows into a fixed device buffer before the segment so the lookup becomes a copy inside the graph, collapsing the two breaks; target ≤ 1 ms/step of the current 2.4–3.
4. **`mincore`-gated prefetch** — skip the 0.7 ms thread cost when all rows are resident.

Decision fed: T3.12 closed as "file layout stays"; the four items above are the Engram entries on the roadmap.

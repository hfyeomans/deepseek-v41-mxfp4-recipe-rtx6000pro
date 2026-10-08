# Binder research (T3.31): patchable approaches to the three memory binders and the prefill ceiling

**Written 2026-10-08** by a research thread ([T-01a11d4e](https://ampcode.com/threads/T-01a11d4e-102d-7023-bd56-f484ce21882b)) from upstream source, commits, PRs, issues and primary papers; saved here verbatim below the fold. Labels follow [README](README.md): everything in the ranked list is **inferred** or **projected** unless it cites a file under `evidence/`.

**Addendum 2026-10-08 (after the thread ran).** The `dair-academy` MCP was authorised and searched (three questions — bounded MoE/indexer prefill memory, expert prefetch/placement/offload, block-scaled FP8/MXFP4 Blackwell kernels — twelve keyword searches, ~30 distinct papers). The collection is a curated *papers-of-the-week* set, not an arXiv index: results were dominated by surveys and model reports. Two on-topic hits, neither registrable: *Lookahead Sparse Attention / FlashMemory-DeepSeek-V4* ([2606.09079](https://arxiv.org/abs/2606.09079)) trains a new indexer that evicts KV chunks from GPU memory on the DSV4 architecture — a model change, outside a quantisation recipe; *SonicMoE* ([2512.14080](https://arxiv.org/abs/2512.14080)) is training-side IO-aware grouped-GEMM tiling whose "token rounding" alters routing and so cannot pass the zero-NLL gate (its IO-overlap kernel idea is already what FlashInfer's CUTLASS path does). Eliseev & Mazur ([2312.17238](https://arxiv.org/abs/2312.17238)) is already cited by E31 and §12. **No new registrations from DAIR.** The owner's Obsidian vault `Bookmarks_wiki` lives on the Mac, not this box, and is dropped from T3.31 by the owner (2026-10-08).

## Verification against the pinned stack (2026-10-08, this box)

Checked before registering anything, because each idea names the condition that falsifies it:

| Idea | Falsifier | What the overlay / image actually has | Verdict |
|---|---|---|---|
| 1 — pack the DSA candidate mask | "already packed; one bounded copy" | `select_candidate_block_keep` (`indexer.py:1132`) returns `torch.bool` (one byte per block), and a second helper expands it to a *per-position* bool mask `[rows, width]` — a larger prefix-scaled copy, not a smaller one | **live**; it is E34e hypothesis (a) with a concrete representation |
| 2 — caller-owned fused-MoE workspace + row tile | "pinned FlashInfer predates PR #4057" | image `dsv41-kt:recipe` ships FlashInfer **0.6.18** and `cutlass_fused_moe` already has `workspace_buffer`; the overlay's `moe_runner/flashinfer_cutlass.py` never passes it (no `workspace` in the file) | **live, cheaper than projected** — no backport, one call-site change plus a tile loop |
| 3 — fixed DSA arena | "`memory_allocated` grows in lockstep with `memory_reserved`" | not yet measured: this is exactly the E34e snapshot protocol's first step | **live, decided by the E34e trace** |
| 5/6/7 — cuTile `(1,32,32)` bench, SM120 CUTLASS instantiation, MXFP8 repack | — | not checked here; they reshape T3.21's spike (bench cuTile before writing a kernel) | folded into T3.21 as its first window |
| 10/11/12 — placement, exact-route overlap, staging slot | — | E31 already measured the L→L+1 predictability; 11 is the cheapest decode lever listed and needs only timestamps the T3.19a timeline already records | parked under T3.20/T3.22 planning, not registered |

**Registered (EXPERIMENTS):** E34e part e-i (idea 1), E34d-v part (i) (idea 2), E34e part e-ii (idea 3). **Struck for now:** 4 (audit folded into e-ii's trace), 8 (allocator toggling: only after e/d-v say the failing allocations are outside the graph pool), 9 (only if the e trace points at the sparse-prefill gather). Ideas 10–12 are not struck but wait for Stage 2/3 (T3.20/T3.22).

---

# Binder Research: DeepSeek-V4.1-Flash on 2× RTX PRO 6000 Blackwell

**Research date:** 2026-10-08  
**Target path:** SGLang overlay + kt-kernel, TP=2, 219 GPU / 165 CPU routed experts per layer, D=4, NVMe Engram tier  
**Measured binders:** E34d-v, E34e, T3.21

> **Source-access note:** the `dair-academy` MCP server was not connected; tool discovery exposed only Amp's own module. This report therefore uses repository source, commits, PRs/issues, and primary papers.

## Acceptance gates

| Gate | Acceptance |
|---|---|
| Prefill | 16K and 32K prompt throughput, compared at identical expert placement |
| Numerical, same quantization shape | 65K-token corpus, zero NLL movement |
| Numerical, changed quantization shape | Pooled NLL increase ≤0.002 nats/token |
| Capacity | Minimum free VRAM during a context-minus-2K stress prompt |
| Decode | Single-user wall-clock tok/s; baseline 39.5 tok/s |
| Safety | No NaN and no Xid |
| Graph compatibility | Capture and repeated replay at every supported graph shape |

## Ranked patch list

Ranking is by expected benefit divided by implementation and correctness risk, not by absolute upside.

### 1. Pack the DSA candidate mask or replace it with block-ID lists

**Binder:** E34e  
**Expected benefit / risk:** Very high / medium

**Evidence**

- TensorRT-LLM represents sparse routes as either BSR (`block_indptr`, `block_indices`) or packed `exact_block_bits`, rather than byte-per-candidate matrices: [sparse params](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/attention/backends/sparse/params.py#L110-L149).
- FlashInfer's task-scheduled sparse attention consumes logical block IDs and builds routes without a full-context bitmap or scan: [README](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/attention/prims_ts/README.md#L111-L164), [PR #4996 / commit `964a79d`](https://github.com/flashinfer-ai/flashinfer/commit/964a79d788a89bb3c3dd3fe894b8080d50eaa490).
- SGLang's earlier dense indexer-logit problem showed the same O(TL) failure mode: [issue #35201](https://github.com/sgl-project/sglang/issues/35201), fixed by [commit `7b67a966`](https://github.com/sgl-project/sglang/commit/7b67a966403e3e959aea6262b72b75a274339f85).

**Smallest overlay test**

Instrument the published `[rows, ceil(prefix/8)]` candidate tensor to record shape, dtype, density, allocation address, and consumer kernels. Add two representations behind a flag:

1. packed `uint32`/`uint64` bits consumed directly by candidate selection; and
2. compact block-ID lists when selected-block density makes IDs smaller than a bitset.

Do not unpack either representation into a persistent byte mask. A temporary unpacked tile is acceptable only if bounded independently of prefix length.

**Projection**

At 8,192 rows and 1M tokens, the current byte-per-8-token mask is about 0.95 GiB per live copy. Bit packing reduces it to about 0.12 GiB, saving approximately **0.83 GiB per copy**. Eliminating two or three concurrent/cached copies would plausibly recover **1.7–2.5 GiB**, enough to cover the projected 1M×1 shortfall. This is a projection; live-copy count must be measured.

**Deciding gates**

- Context-minus-2K minimum free VRAM at 512K and 1M.
- Same-shape zero-movement NLL.
- 16K/32K prefill throughput.
- No NaN/Xid.

**Falsified if**

The published mask is already packed in physical storage, only one bounded copy exists, or its consumer immediately expands to an equally large prefix-dependent tensor.

---

### 2. Row-tile FlashInfer fused MoE and reuse one caller-owned workspace

**Binder:** E34d-v  
**Expected benefit / risk:** High / medium

**Evidence**

FlashInfer current main at commit [`0caca03`](https://github.com/flashinfer-ai/flashinfer/tree/0caca03ce677d210d663c41431cb20e5ef8035c7) exposes:

- `cutlass_fused_moe_workspace_size(...)`: [source](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/fused_moe/core.py#L1187-L1247).
- `cutlass_fused_moe(..., workspace_buffer: Optional[torch.Tensor] = None)`: [signature](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/fused_moe/core.py#L1261-L1301).
- Documentation that caller reuse avoids repeated allocations potentially reaching 10–20 GiB: [documentation](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/fused_moe/core.py#L1453-L1456).
- The binding retains ownership separately from the raw workspace pointer: [binding](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu#L851-L856).

The API was introduced by [FlashInfer PR #4057](https://github.com/flashinfer-ai/flashinfer/pull/4057), commit [`d25d2d6`](https://github.com/flashinfer-ai/flashinfer/commit/d25d2d64f03fe00bcded80d0d1fe663e71d2cc63).

Current SGLang calls `cutlass_fused_moe` with an explicit output but no workspace: [SGLang source at `d1ba943`](https://github.com/sgl-project/sglang/blob/d1ba943016cb93fe874b924bbd293214ff086413/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py#L499-L533).

TensorRT-LLM already chunks MoE at framework level and reuses workspace by chunk: [scheduler](https://github.com/NVIDIA/TensorRT-LLM/blob/45c7060ab1392e18d072f8b90a40cc705ff264f3/tensorrt_llm/_torch/moe/fused_moe/moe_scheduler.py#L139-L305). [PR #18322](https://github.com/NVIDIA/TensorRT-LLM/pull/18322) reports reducing a 65,536-row workspace from about 14 GiB to four approximately 3.5 GiB chunks.

**Smallest overlay test**

First verify whether the pinned FlashInfer includes PR #4057; backport that commit if not.

In `fused_experts_none_to_flashinfer_mxfp4`:

1. Set an independent MoE row tile, initially 8,192.
2. Query workspace size for that tile.
3. Allocate one persistent 1-D CUDA byte tensor per device and concurrently active stream before graph capture.
4. Slice every row-indexed input consistently and write each result into the corresponding output slice.
5. Pass the persistent tensor through `workspace_buffer=`.
6. Treat attempted growth after graph capture as a recapture error.

Initially restrict the experiment to the local standard-dispatch/finalized-output path. Do not alias the FlashInfer A2A workspace.

**Projection**

An 8,192-row tile should cap the measured 16,384 workspace at roughly **980 MiB instead of 1.86 GiB**, recovering about **0.88 GiB** and removing late allocator growth. It may preserve most of the measured +22% 16K prefill gain while using a 16,384 outer chunk. It will not by itself reduce the other transients that make the total rise from 3.2 to 8 GiB.

A 4,096-row trial should approximately halve workspace again but adds more launches and sorting/finalization overhead.

**Deciding gates**

- 16K and 32K prefill tok/s.
- Same-shape zero-movement NLL.
- Context-minus-2K minimum free VRAM.
- Decode must remain within measurement noise.
- Graph capture/replay and 0 NaN/Xid.

**Falsified if**

Routing, expert sorting, scale generation, or fused finalization has cross-row state that changes results across tile boundaries, or the non-MoE transients still force removal of the same 14 experts.

---

### 3. Give every prefix-growing DSA tensor a fixed arena and explicit output buffers

**Binder:** E34e  
**Expected benefit / risk:** High / low-medium

**Evidence**

- SGLang `SparsePrefillWorkspace` grows by replacing its cached tensor when a larger request arrives: [source](https://github.com/sgl-project/sglang/blob/d1ba943016cb93fe874b924bbd293214ff086413/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py#L78-L105). This can transiently require old plus new capacity and leave caching-allocator high-water marks.
- Large unbounded sparse-prefill gathers have produced 15.4–27.2 GiB allocation attempts: [SGLang issue #41076](https://github.com/sgl-project/sglang/issues/41076).
- FlashInfer top-k supports caller-provided row-state/output buffers suitable for stable graph-visible storage: [top-k source](https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/topk.py#L65-L116).
- TensorRT-LLM retains one owning tensor per stream and grows it explicitly: [source](https://github.com/NVIDIA/TensorRT-LLM/blob/45c7060ab1392e18d072f8b90a40cc705ff264f3/cpp/tensorrt_llm/thop/moe/cutlass/moeOp.cpp#L1299-L1336).

**Smallest overlay test**

Add allocation tracing around indexer scores, candidate masks, top-k state, sparse-prefill gather indices, and candidate publication. For each allocation record:

- requested and capacity bytes;
- pointer;
- stream and layer;
- prefix and query rows;
- whether another generation remains live.

Then replace grow-on-demand tensors with a startup-profiled arena partitioned into reusable buffers. Tile work when a requested shape exceeds its partition rather than replacing the tensor.

**Projection**

If the observed staircase is allocator high-water retention, an immutable arena should make free VRAM flat after startup and recover the difference between live bytes and the largest historical capacities—plausibly **0.5–1.5 GiB** from the measured staircase. This estimate is a projection pending a memory snapshot.

**Deciding gates**

- Free-VRAM trace from 250K through 1M, not merely end-state memory.
- 16K/32K prefill throughput.
- Same-shape NLL and graph replay.
- 0 NaN/Xid.

**Falsified if**

`memory_allocated` grows in lockstep with `memory_reserved`, with no dead or superseded buffers, showing that the staircase is genuine persistent state rather than allocator retention.

---

### 4. Stop expanding request/block metadata per query token

**Binder:** E34e  
**Expected benefit / risk:** High if present / low-medium

**Evidence**

vLLM's sparse-MLA metadata builder duplicates an entire block-table row for each expanded token: [kernel](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/indexer.py#L98-L145). [Issue #60450](https://github.com/vllm-project/vllm/issues/60450) reports a persistent:

```text
[max_num_batched_tokens, max_model_len / block_size] int32
```

allocation—1 GiB/GPU at 16,384 rows, 1M context, and block size 64.

TensorRT-LLM instead builds an O(T) token-to-request map and retains one table per request: [DSA metadata](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/attention/backends/sparse/dsa/metadata.py#L64-L135).

**Smallest overlay test**

Audit all tensors whose dimensions include both query rows and prefix blocks. If found, pass:

- one request ID per query row;
- one block table per request;
- row-local position or valid length.

Resolve the table inside the kernel. If a flat table is unavoidable, expand only a bounded row tile into reusable scratch.

**Projection**

Removing one `[8192, 1M/64] int32` table saves about **488 MiB per rank**. Multiple metadata tables or smaller blocks could explain a substantial part of the 1.5–2.3 GiB projected shortfall.

**Deciding gates**

Minimum free VRAM at 512K and 1M, same-shape NLL, 16K/32K prefill, graph replay, 0 NaN/Xid.

**Falsified if**

The overlay already stores one table per request and all token-level metadata is O(T), not O(TL/B).

---

### 5. Benchmark FlashInfer cuTile FP8 with exact float scales before writing a kernel

**Binder:** T3.21  
**Expected benefit / risk:** Medium-high / medium

**Evidence**

The current FlashInfer cuTile API accepts configurable groupwise FP8 scales:

- API: [`gemm_fp8_nt_groupwise_cutile`](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/kernels/cutile/gemm_fp8_nt_groupwise_cutile.py#L467-L475).
- It requires activation scale granularity M_g=1, but permits N_g=32, K_g=32: [validation](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/kernels/cutile/gemm_fp8_nt_groupwise_cutile.py#L491-L517).
- For `(1,32,32)`, A scales are `[M, ceil(K/32)]` and B scales are `[ceil(N/32), ceil(K/32)]`: [layout](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/kernels/cutile/gemm_fp8_nt_groupwise_cutile.py#L544-L555).
- FlashInfer's existing benchmark already includes M=4096 and 8192: [benchmark](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/benchmarks/bench_groupwise_gemm_fp8_blackwell.py#L163-L202).

Thus the existing `[N/32,K/32]` weight scales can remain unchanged. Activation scales must be repeated over each 32-row group; the FP8 payload need not be requantized.

**Smallest overlay test**

Create a standalone shape manifest from the attention projections, shared expert, and dense indexer. For each shape at M=4096, 8192, 16384:

1. Run the current Triton kernel.
2. Expand A's `[M/32,K/32]` scale grid to `[M,K/32]`.
3. Call cuTile with `(1,32,32)`.
4. Measure scale-expansion time separately and combined.
5. Compare BF16 output error and corpus NLL.

Integrate only the shapes that win including scale expansion.

**Projection**

A mature tensor-core path could plausibly deliver **1.5–3× GEMM speedup** over a kernel measured at 15% of peak. End-to-end prefill improvement is likely smaller—approximately **15–40%**, depending on the GEMM share—so it must be established by profiling rather than inferred from kernel TFLOP/s.

**Deciding gates**

16K/32K prefill, same-shape zero-movement NLL, no decode regression, graph replay, 0 NaN/Xid.

**Falsified if**

cuTile is not faster after scale expansion, cannot graph-capture reliably on the pinned CUDA stack, or changed accumulation order violates the strict same-shape NLL gate.

---

### 6. Add an SM120 `(1,32,32)` float-scale CUTLASS instantiation

**Binder:** T3.21  
**Expected benefit / risk:** High / high

**Evidence**

FlashInfer's current SM120 CUTLASS binding carries explicit M/N/K scale granularities: [binding](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/csrc/gemm_sm120_binding.cu#L18-L23). However, it instantiates only:

- `(1,128,128)`;
- `(128,128,128)`.

Other granularities are rejected: [SM120 implementation](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/csrc/gemm_groupwise_sm120.cu#L30-L58).

CUTLASS has a single-CTA SM120 Stream-K implementation using `OpClassBlockScaledTensorOp` and `ClusterShape<1,1,1>`: [unit configuration](https://github.com/NVIDIA/cutlass/blob/main/test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu#L58-L120). It does not require the datacenter-only 2SM `tcgen05` path.

**Smallest overlay test**

Fork FlashInfer's SM120 groupwise implementation and instantiate `(1,32,32)` for one representative projection shape. Keep FP32 scales and current FP8 data. First expose it through the existing FFI, without changing SGLang dispatch globally.

Use the cuTile result from rank 5 as the correctness reference and performance floor.

**Projection**

If `(1,32,32)` maps efficiently to the SM120 kernel, it could remove scale-expansion overhead and exceed cuTile, potentially reaching **2–4× current Triton GEMM throughput**. End-to-end prefill remains projected at **20–50%**, bounded by attention, MoE, and memory traffic.

**Deciding gates**

16K/32K prefill, same-shape NLL, graph replay, 0 NaN/Xid. Reject if compilation materially increases startup beyond the overlay's acceptable cache-warm path.

**Falsified if**

CUTLASS's SM120 collective cannot efficiently apply arbitrary FP32 32×32 scales, or scale handling dominates enough that Triton/cuTile remains faster.

---

### 7. Trial an MXFP8-repacked SM120 path as an explicitly shape-changing experiment

**Binder:** T3.21  
**Expected benefit / risk:** High / high

**Evidence**

FlashInfer commit `0caca03` exposes:

```python
mm_mxfp8(a, b, a_descale, b_descale, out=None,
          out_dtype=torch.bfloat16,
          use_8x4_sf_layout=False, backend="auto")
```

[API](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/gemm_base.py#L6817-L6853).

The SM120 path requires per-row K32 UE8M0 scales in the 128×4 swizzle, not the existing 32×32 FP32 grid: [validation/layout](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/gemm_base.py#L6109-L6136), [CuTe DSL implementation](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/flashinfer/gemm/gemm_mm_mxfp8_cute_dsl.py#L140-L168).

The kernel landed in [commit `2febce55`](https://github.com/flashinfer-ai/flashinfer/commit/2febce559f466aa2b0adaca3f78c172c22066321).

**Smallest overlay test**

Offline-repack one weight matrix:

1. expand each 32×32 weight scale over its 32 rows;
2. encode or requantize into UE8M0-compatible scales;
3. store the required swizzle;
4. dynamically quantize/repack activations;
5. benchmark M=4096, 8192, 16384, including activation conversion.

Treat this as a changed quantization format, not a drop-in backend.

**Projection**

The warp-level SM120 MXFP8 kernel could produce the largest GEMM gain, but conversion may erase it. A successful integrated path could yield **20–50% projected prefill improvement**.

**Deciding gates**

Shape-change NLL increase ≤0.002 nats/token, 16K/32K prefill, graph replay, no decode regression, 0 NaN/Xid.

**Falsified if**

Original scales are not accurately representable in UE8M0, activation repacking consumes the saved time, or pooled NLL exceeds +0.002.

---

### 8. Enable expandable segments outside graph capture, but force graph allocations back to legacy-exportable storage

**Binder:** E34d-v and E34e  
**Expected benefit / risk:** Medium / medium-high

**Evidence**

The custom-all-reduce incompatibility is independently reproduced in [vLLM issue #42609](https://github.com/vllm-project/vllm/issues/42609): expandable segments use VMM ranges that legacy `cudaIpcGetMemHandle` cannot export as one allocation.

[vLLM PR #43923](https://github.com/vllm-project/vllm/pull/43923) proposes disabling expandable segments around capture so graph allocations remain IPC-exportable. [PR #57982](https://github.com/vllm-project/vllm/pull/57982), commit [`e55d076`](https://github.com/vllm-project/vllm/commit/e55d076f89fd01a0538a3e496d8ff20bf7980100), preserves the complete allocator configuration while toggling allocator behavior.

vLLM's custom all-reduce obtains allocation bases and exports them through CUDA IPC: [C++ source](https://github.com/vllm-project/vllm/blob/240785b82c299c6c08bc5239e997cf62d67cf7bb/csrc/custom_all_reduce.cuh#L93-L112), [registration](https://github.com/vllm-project/vllm/blob/240785b82c299c6c08bc5239e997cf62d67cf7bb/csrc/custom_all_reduce.cuh#L200-L225).

**Smallest overlay test**

Mirror PR #43923 around SGLang graph capture:

1. preserve the full allocator setting string;
2. disable expandable segments before capture allocations;
3. capture and register custom-AR IPC buffers;
4. restore the exact previous configuration;
5. assert captured pointers remain unchanged across replay.

Do not merely set `expandable_segments:False/True`, which can discard unrelated allocator settings.

**Projection**

This may recover fragmentation outside graph pools and remove late contiguous-allocation failures without paying NCCL's decode tax. Benefit is unquantifiable until allocator snapshots are collected; a reasonable test target is **≥0.5 GiB extra minimum free VRAM** or survival beyond the current ~190K failure point.

**Deciding gates**

Context-minus-2K stress, graph capture/replay, single-user decode, custom-AR enabled, 0 NaN/Xid.

**Falsified if**

The failing allocations are all graph-pool allocations, toggling does not improve fragmentation, or any exported pointer still belongs to a multi-handle VMM segment.

---

### 9. Tile sparse-prefill history gathers by gathered rows, not only new-token rows

**Binder:** E34e  
**Expected benefit / risk:** Medium-high / medium

**Evidence**

SGLang's sparse-prefill workspace is shaped `(num_tokens, 1, 512)` BF16 and is not bounded by `--chunked-prefill-size`: [source](https://github.com/sgl-project/sglang/blob/d1ba943016cb93fe874b924bbd293214ff086413/python/sglang/srt/layers/attention/dsv4/sparse_prefill_utils.py#L78-L142). [Issue #41076](https://github.com/sgl-project/sglang/issues/41076) reports that gathered compressed-history plus SWA rows can greatly exceed newly admitted prompt rows.

**Smallest overlay test**

Set a fixed gathered-row budget. For each query tile:

1. gather only its selected compressed/SWA KV rows;
2. execute sparse attention;
3. discard or reuse that gather buffer;
4. advance to the next tile.

Admission should account for projected gathered rows, not just `CHUNKED_PREFILL_SIZE`.

**Projection**

Peak sparse-attention gather memory should become fixed by the configured tile rather than prefix/batch history. Depending on whether this path is active in the overlay, projected recovery ranges from negligible to **multiple GiB**.

**Deciding gates**

512K/1M minimum free VRAM, 16K/32K prefill, same-shape NLL, 0 NaN/Xid.

**Falsified if**

The observed prefix slope remains unchanged while this gather workspace is disabled or held at constant capacity.

---

### 10. Make 219-expert placement conservative and route-trace-driven

**Binder:** Other—decode CPU expert pressure  
**Expected benefit / risk:** Medium / low-medium

**Evidence**

kt-kernel supports arbitrary per-layer placement masks and frequency-based or dynamic updates:

- mask and CPU skip test: [common.hpp](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/operators/common.hpp#L230-L258);
- frequency mask generation: [experts_base.py](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L162-L213);
- placement controls: [tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md#L96-L140);
- implementation provenance: [PR #1796](https://github.com/kvcache-ai/ktransformers/pull/1796), [commit `027832c`](https://github.com/kvcache-ai/ktransformers/commit/027832c590c1b08e228ed60987eb22ae89283602).

MoE-Infinity uses request-level activation traces for expert prediction and cache placement: [paper](https://arxiv.org/abs/2401.14361), [predictor source](https://github.com/EfficientMoE/MoE-Infinity/blob/main/moe_infinity/memory/expert_predictor.py#L23-L73).

**Smallest overlay test**

Compare exactly 219 slots per layer under:

1. current placement;
2. global route frequency;
3. per-layer minimum quota plus globally allocated remaining slots;
4. conservative dynamic replacement of only 4–8 slots at decode boundaries.

Require a large frequency margin over several windows before replacement and never evict an in-flight expert. Record migration bytes, hit rate, and CPU expert bytes.

**Projection**

If the current static set misses sequence-local hot experts, conservative adaptation could reduce CPU expert work and improve wall decode by **5–15%**. The gain will be near zero if routing is already balanced or 219 slots capture almost all weighted traffic.

**Deciding gates**

Single-user decode tok/s, same-shape NLL, CPU memory bandwidth, migration bytes, and 0 NaN/Xid.

**Falsified if**

Weighted GPU hit rate is already saturated, or migration cost exceeds the CPU work avoided.

---

### 11. Validate exact-route deferred overlap before adding another predictor

**Binder:** Other—decode scheduling  
**Expected benefit / risk:** Low-medium / low

**Evidence**

D=4 already divides top-8 routes into immediate and deferred CPU work:

- selection and `protected_k=top_k-D`: [source](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L492-L520);
- preparation: [source](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L542-L589);
- deferred submission and merge: [source](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L730-L817);
- `sync(allow_pending=1)`: [source](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L835-L879);
- provenance: [PR #1545](https://github.com/kvcache-ai/ktransformers/pull/1545), [commit `9bc00e5`](https://github.com/kvcache-ai/ktransformers/commit/9bc00e587ba57442b1e3398ce0abd5a5675ac55a).

The current prefill implementation constructs a batch-wide union of protected expert IDs, so D=4 is exact for batch-one decode but may defer fewer than four experts per token in prefill: [selection logic](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L507-L518).

**Smallest overlay test**

Add timestamps for router completion, immediate enqueue/start/end, deferred enqueue/start/end, GPU expert execution, and merge. Sweep D ∈ {0,1,2,3,4}.

Only if the trace shows deferred work waiting unnecessarily, enqueue known deferred routes sooner and order work as GPU-resident hits → immediate CPU → deferred CPU. This uses exact router output and introduces no prediction error.

**Projection**

Better overlap could improve single-user decode by **2–10%** without moving expert weights or changing outputs. If existing overlap is complete, no benefit is expected.

**Deciding gates**

Single-user decode tok/s and p95 TPOT, zero NLL movement, no NaN/Xid.

**Falsified if**

The deferred queue is already continuously busy and merge waits are dominated by DRAM bandwidth rather than scheduling gaps.

---

### 12. Prototype one staging slot for high-confidence layer-ahead route hits

**Binder:** Other—expert prefetch  
**Expected benefit / risk:** Low-medium / high

**Evidence**

The measured layer-L to layer-L+1 predictability provides a usable signal without adopting the rejected router-before-attention predictor. Relevant systems separate prediction from bounded cache movement:

- ProMoE predictor/prefetch architecture: [paper](https://arxiv.org/abs/2410.22134), [source](https://github.com/promoe-opensource/promoe).
- Klotski executes resident/hot experts while cold transfers proceed: [paper](https://arxiv.org/abs/2502.06888).
- HOBBIT prioritizes misses using router scores: [paper](https://arxiv.org/abs/2411.01433).

**Smallest overlay test**

Reserve one or two GPU staging slots outside the fixed 219-slot mask. From layer L's route trace, prefetch only the highest-confidence predicted CPU expert for L+1, asynchronously and cancellably. Do not alter routing; a miss simply uses the CPU path. Promote the staged expert only after the real router confirms it.

**Projection**

Benefit is bounded by PCIe copy time and prediction precision. With only one-layer lookahead, likely gain is **0–5% decode** unless expert payload transfer overlaps almost completely. This should remain below placement and exact-route scheduling in priority.

**Deciding gates**

Wall decode tok/s, staged bytes, useful-prefetch ratio, wasted bytes, and unchanged NLL.

**Falsified if**

A full expert cannot transfer within the available layer interval, useful-prefetch rate is low, or staging displaces enough resident experts to reduce the 219-slot hit rate.

## Rejected on inspection

1. **Preallocate one workspace sized for the full 16,384 rows, without tiling.**  
   Caller ownership removes churn and late allocation but does not reduce the 1.86 GiB algorithmic workspace or the 8 GiB total transient peak. FlashInfer tracks true workspace reduction separately in [issue #3364](https://github.com/flashinfer-ai/flashinfer/issues/3364).

2. **Assume current SGLang already forwards a FlashInfer workspace.**  
   It does not in the inspected CUTLASS MXFP4 call: [source](https://github.com/sgl-project/sglang/blob/d1ba943016cb93fe874b924bbd293214ff086413/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py#L499-L533).

3. **Treat FlashInfer `mm_mxfp8` as a drop-in for `[32,32]` FP8 scales.**  
   It requires per-row K32 UE8M0 scales in a swizzled physical layout, so it is a quantization-format change.

4. **Treat FlashInfer SM120 `gemm_fp8_nt_groupwise` as supporting `(1,32,32)`.**  
   Current source only instantiates `(1,128,128)` and `(128,128,128)`: [source](https://github.com/flashinfer-ai/flashinfer/blob/0caca03ce677d210d663c41431cb20e5ef8035c7/csrc/gemm_groupwise_sm120.cu#L30-L58).

5. **Use DeepGEMM as an already-available dense SM120 replacement.**  
   The inspected dense dispatcher at commit [`057ca596`](https://github.com/deepseek-ai/DeepGEMM/tree/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7) has SM90 and SM100 branches, then rejects other architectures: [dispatcher](https://github.com/deepseek-ai/DeepGEMM/blob/057ca5964aae0879ff2e0eb71ee05a3cb0ba3df7/csrc/apis/gemm.hpp#L112-L132). SM120 grouped/graph issues also remain: [#444](https://github.com/deepseek-ai/DeepGEMM/issues/444), [#372](https://github.com/deepseek-ai/DeepGEMM/issues/372).

6. **Port SM100/SM103 `tcgen05` or 2SM kernels.**  
   RTX SM120 lacks the required datacenter Blackwell TMEM/2CTA feature set. Use warp-level `mma.sync`, `ClusterShape<1,1,1>`, or software-scaled kernels instead.

7. **Enable expandable segments globally and retain the current custom all-reduce unchanged.**  
   Legacy CUDA IPC export fails for the VMM-backed graph allocations: [vLLM issue #42609](https://github.com/vllm-project/vllm/issues/42609).

8. **Use NCCL fallback as the allocator fix.**  
   It avoids the IPC failure but taxes every decode step, conflicting with the measured single-user decode priority.

9. **Use a full token-level bitmap merely because it is bit-packed.**  
   At 8,192 rows and 1M context it is still about 0.95 GiB. Pack the existing block mask or retain sparse block IDs.

10. **Adopt mlx-stream/router-before-attention prediction.**  
    It is already rejected for this GPU-resident design and introduces a second routing computation. Exact-route overlap and measured route-trace placement are lower risk.

11. **Direct UVA execution of DRAM-resident experts.**  
    The measured 100–125 GB/s host pipe prices every decode token by expert bytes. Current kt-kernel only uses pinned staging for activations/routes/results, not mapped expert GEMM: [pinned buffers](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/python/experts_base.py#L241-L267), [explicit H2D transport](https://github.com/kvcache-ai/ktransformers/blob/main/kt-kernel/fp8_layerwise_transport.cpp#L209-L255).

12. **Add NVMe streaming for CPU experts before proving DRAM capacity pressure.**  
    kt-kernel's experimental MESH path is unmerged [PR #2003](https://github.com/kvcache-ai/ktransformers/pull/2003). NVMe should remain the Engram tier unless the 165 CPU experts do not fit in DRAM; otherwise it only adds latency and cache complexity.

## Recommended experiment order

1. Trace E34e allocation shapes, pointers, densities, and live-copy count.
2. Pack the candidate mask or preserve sparse block IDs.
3. Add the persistent DSA arena and eliminate token-expanded metadata.
4. Backport/use FlashInfer caller-owned MoE workspace and add 8,192-row tiling.
5. Test outer prefill 16,384 with MoE tile 8,192, then 4,096.
6. Benchmark cuTile `(1,32,32)` at real DSV4 shapes.
7. Write a CUTLASS `(1,32,32)` kernel only if cuTile proves the format and end-to-end opportunity.
8. Trial MXFP8 repacking only under the +0.002 nats/token shape-change gate.
9. Sweep D and placement before implementing speculative expert movement.

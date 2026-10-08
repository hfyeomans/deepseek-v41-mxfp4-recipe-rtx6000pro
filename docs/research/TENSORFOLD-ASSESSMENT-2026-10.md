# TensorFold assessment (T3.32): engine switch or kernel inspiration?

**Written 2026-10-08** by a research thread ([T-01a11d77](https://ampcode.com/threads/T-01a11d77-0ed2-7761-a9b6-bde842688ac9)) from the TensorFold tree at v1.0.2, the frozen Python 0.6 branch, the three community DeepSeek/GLM recipes and the issue tracker; saved here verbatim below the fold. Labels follow [README](README.md): every TensorFold number is **inferred** from the pinned sources (none was run on this box); every number on our side is **measured** and cites the evidence dir it comes from.

**Owner decision (2026-10-08):** TensorFold is used as *kernel inspiration and experimentation*, not as an engine. The three recommendations in §5 are incorporated as follows.

## Verification against the pinned stack (2026-10-08, this box)

Checked before registering anything, because each recommendation names something the overlay either already has or already does differently:

| Recommendation | What it assumes | What the overlay / launcher actually has | Where it lands |
|---|---|---|---|
| 1 — shape autotune *method* (record real shapes, constrained candidates, byte-identical at both bucket ends with several seeds, L2-cold rotated timing, ≥ 2 % over spread, device-keyed JSON with an invalidation hash, mandatory end-to-end gate) | no tuner exists here | T3.27 was already registered 2026-10-07 with the bit-exact / L2-flush / > 2 % rules and the same Aevonix reference; it lacked shape *recording* from live calls, multi-seed + sentinel-fill comparison, rotated candidate order, the invalidation signature and — the important one — the **end-to-end acceptance gate** (the Aevonix port report says swapping only the saved table gave no consistent e2e gain) | **T3.27 row amended** with the six method items; exit now requires a C1 / C4 / 16K–32K prefill re-measurement, not just the kernel table |
| 2 — Engram row I/O ahead of use (hash early, dedup, O_DIRECT / io_uring reads issued before the consuming layer) vs page cache | Engram rows are demand-faulted | `_FileTable` (overlay `engram.py:675`) maps the shard `MAP_SHARED` read-only and `gather()` calls `prefetch()` = `MADV_WILLNEED` from a small thread pool **at lookup time**, i.e. one queue-depth-N round trip per layer break, not ahead of use; row ids come from `EngramHasher` on input ids only, so layer-14 ids are computable before layer 0 runs — which is exactly T3.20's "prefetch layer-14 rows during the early layers" | **T3.20 row amended** with a part (b): measure page faults / NVMe latency / page-cache churn / DRAM-fabric counters on the current path, then A/B deduplicated early-issued `MADV_WILLNEED` vs O_DIRECT + io_uring into a pinned staging buffer; adopt only if C1/C4 tails or long-prefill wall improve without stealing bandwidth from the kt CPU tier (registered as **E36**) |
| 3 — "CED bounded replay" (decoder-side prompt work halved) | absent in our stack | present but unlaunched: `--enable-encoder-swa-bounded-replay` and `--enable-decoder-swa-bounded-replay` (`server_args.py:3639–3649`, validated in `arg_groups/deepseek_v4_hook.py:232–347`). Neither is in `scripts/serve_config.py` or `serve-hybrid.sh`. The **decoder** variant ("after the last kv_source layer, run the remaining layers over only the last window_size tokens of a prefill; main and indexer KV stay exact; deterministic for a fixed prompt and chunk size") refuses the prefill CUDA graph and DP attention — and the prefill graph default on CUDA is `breakable` (`cuda_graph_config.py:117`), so the A/B must pay for prefill-graph-off too. The **encoder** variant needs explicit `--max-running-requests` (we set it) and chunk ≥ 128, and refuses mixed chunk / radix sessions / unified memory / PD | **new task T3.33 + E35**: one serve per variant on the default profile; gates = NLL 0 moved on the 65K corpus (T3.28), needles 6/6 at 64K, prefill 16K / 32K tok/s, C1 wall, VRAM min-free at 4 × 62K; the recipes claim ≈ 2× decoder-side prompt work, so the go bar is +25 % prefill at 16K with the other gates flat |
| benchmark discipline — distinct prompts for every concurrency number (issue #479: same-prompt overstated MoE batching 2–3×) | — | `bench-decode.py --distinct-prompts` has been the rule since E26 Part B (2026-09-23); `--same-prompt` is used only for the C2–C4 consistency gate and E23's "full expert reuse" row, both labelled | **BENCHMARKS Method** bullet made explicit |

**Not adopted, with the reason, so the same leads are not re-researched:** engine switch (native CUDA registry is Nemotron-only; no CPU/DRAM expert tier; DeepSeek forks are 2.9–3.0 bpw EXL3 — the quality point already rejected in T3.25; issue #479's same-model TP4 build stays 12–24 % behind its SGLang baseline after three days of tuning); the Mac SSD expert pool (streams weights and stalls, 62.8–73.6 → 8.9–10.5 tok/s); copying Aevonix launch tables or GLM-specific kernels (different model, different checkpoint format; the reusable asset is the search/verification framework); DSpark via a TensorFold fork (speculation is settled off here — E23, T3.25; if revisited it is SGLang-native, E23 R6 n-gram first).

**Re-open the engine question only if all six triggers in §5 hold** — above all upstream native CUDA DeepSeek-V4.1 on SM120 *and* a CPU/DRAM expert tier compatible with 219 / 165 placement *and* a same-weights, same-box benchmark beating C1 ≈ 40 / C4 ≈ 58 / 1,000–1,500 prefill with 64K × 4 and 512K × 1 intact.

---

# TensorFold assessment for our DeepSeek-V4.1-Flash server

**Assessment date:** 2026-10-08. **Source baseline:** TensorFold v1.0.2, commit [f8fe17d](https://github.com/ashhart/TensorFold/tree/f8fe17d24629aedabf90bbf78279dd776e6d62e7), plus the frozen Python 0.6 branch at [ed78d6f](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1) and the community recipes named below. I cloned and read the code, patches, recipe harnesses, and relevant issues; repository counts are a snapshot and will move.

## Bottom line

**Recommendation: (b) borrow selected methods; do not switch engines.**

The part worth adopting is the **Aevonix recipe's shape-autotuning method**, not TensorFold itself: record real shapes, enumerate constrained launch variants, reject anything that is not bit-identical at bucket boundaries, time L2-cold in rotated order, and persist a device-keyed, auditable table with a strong invalidation signature. Also prototype the community DeepSeek recipes' **ahead-of-use Engram row I/O** against our current page-cache path, and verify whether our overlay already has **CED bounded replay** before doing any implementation.

A switch would remove the capability that makes our deployment fit: TensorFold has **no CUDA CPU-expert/DRAM execution tier comparable to kt-kernel**. Upstream v1.0.2 has no native CUDA DeepSeek-V4.1 backend at all; native CUDA registers only Nemotron. The DeepSeek results are large custom forks on old Python TensorFold and mostly 2.9–3.0 bpw EXL3, which we have already rejected for quality. The strongest same-model comparison is TensorFold issue [#479](https://github.com/ashhart/TensorFold/issues/479): after three days of tuning on four GB10s, final native-weight TensorFold remained 12–24% slower in real-traffic wall time than that team's own SGLang deployment.

## Comparison

TensorFold evidence in this table is from the pinned sources; our side is from the owner notes supplied for this assessment.

| Dimension | TensorFold v1.0.2 / community forks | Our SGLang + kt-kernel deployment |
|---|---|---|
| Engine shape | Family-specialized exact-inference stack, migrating from frozen Python/PyTorch to a native Zig server/runtime | SGLang serving stack plus our overlay and kt-kernel |
| DeepSeek-V4.1 CUDA | **Not upstream-supported.** Open issue [#299](https://github.com/ashhart/TensorFold/issues/299); community families are patch stacks/forks | In production on 2× SM120 |
| Weight formats | Family-specific MLX affine 4/6/8-bit, NVFP4, EXL3; some MXFP4 family kernels; no GGUF. Native CUDA currently accepts only Nemotron MLX q4g64 | Native-quality MXFP4 routed experts; accepted quality point |
| Scheduler | Lane round loop: admit/prefill, synchronous shared verify forward, independent commit; adaptive speculative depth/tree/copy grafts priced by measured expected tokens/ms | Current qualified SGLang scheduler and our operational ladder |
| CUDA graphs | Yes, per family/shape; native graph wrapper exists. Current correctness/perf issues include [#336](https://github.com/ashhart/TensorFold/issues/336) and [#512](https://github.com/ashhart/TensorFold/issues/512) | Existing production path; no engine migration needed |
| Speculation | MTP chains, DFlash2, DSpark in forks, prompt/copy grafts; exactness target is drafted = serial and concurrent = solo | Not specified in the notes; evaluate independently if absent |
| Multi-GPU | Generic Zig planner models TP/PP/EP, and exact toy tests exist; qualified production support is family/backend-specific. Python CUDA is mostly 1–2 ranks; DeepSeek TP2/TP4 lives in custom forks | Working two-GPU placement with 219 experts/layer on GPU and 165 in CPU/DRAM |
| CPU/DRAM experts | **None on CUDA** | kt-kernel executes CPU-resident routed experts; socket fabric is the known limit |
| NVMe | Frozen Python/MLX can stream experts from SSD; community DeepSeek forks fetch Engram rows with O_DIRECT/AIO/io_uring. Native CUDA has direct-read loading utilities, not an execution tier | Engram shards file-backed on NVMe through page cache |
| Kernels | Own CUDA and Metal kernels; ahead-of-time captured Triton cubins; cuBLASLt and NCCL loaded dynamically. No CUTLASS or cuTile implementation found | SGLang plus kt-kernel and our overlay |
| Autotune | No general upstream autotuner. A strong per-shape tuner exists in the Aevonix GLM recipe | Open task; best immediate borrowing opportunity |
| Maturity | Apache-2.0; very active but young and concentrated; native rewrite still incomplete, no GitHub Actions workflows at the inspected commit | Qualified production baseline with known C1/C4/prefill/context behavior |
| Relevant measured evidence | DeepSeek forks are fast on 2× GB10 at 2.9 bpw; issue #479's native-weight TP4 build still loses to its local SGLang baseline | C1 ~40 wall tok/s; C4 ~58 aggregate; 1,000–1,500 prefill tok/s; 64K×4 and 512K×1 qualified |

## 1. What TensorFold actually is

### Architecture and scheduler

TensorFold is better described as a **collection of tightly specialized model-family engines behind an OpenAI-compatible server** than as a broad drop-in peer to SGLang. The old path uses Python, PyTorch, Triton/JIT CUDA extensions, or MLX. Version 1.0 introduces a native Zig control plane, checkpoint loader, tokenizer/server, CUDA/Metal runtime, kernels, and lane scheduler; the old Python engine is frozen.

The native lane engine is explicit in [zig/src/core/lanes/engine.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/core/lanes/engine.zig): a request is prefetched/prefilled, joins the live set, and each shared round verifies all chosen streams in one forward before each stream independently commits accepted rows. [depth.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/core/lanes/depth.zig) chooses draft depth from measured acceptance and round cost; [plan_lanes.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/core/lanes/plan_lanes.zig) prices MTP trees/chains and copied continuations by expected tokens per millisecond. Tests assert drafted output equals one-token decoding and shared rounds equal solo rounds in [engine_test.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/core/lanes/engine_test.zig).

This is thoughtful scheduling work, especially for speculative exactness, but concurrency remains family-specific. The frozen CUDA documentation says only Qwen3.8-27B and Flash Next share rounds; Nemotron, GLM, and Qwen3.6 remain serialized in that path ([CUDA recipe](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/docs/recipes/cuda.md)).

### Models, formats, graphs, speculation, multi-GPU

There are three different support matrices that should not be conflated:

1. **Native Zig CUDA v1.0.2:** the registry is literally `const registry = .{nemotron.native};` in [zig/src/native/cuda.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/native/cuda.zig#L12-L18). It accepts Nemotron 3.5 Lightning's MLX affine 4-bit/group-64 checkpoint. DeepSeek-V4.1, GLM, and Flash Next are not native CUDA families.
2. **Native Zig Metal:** Nemotron, Qwen Flash Next, GLM-5.3-Flash, and Qwen3.5-2B are registered in [zig/src/native/metal.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/native/metal.zig#L13-L17). DeepSeek's documented family is still an MLX lane-engine path; its own recipe says CUDA is not present ([deepseek-v4-flash.md](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/docs/recipes/deepseek-v4-flash.md)).
3. **Frozen Python CUDA:** Qwen3.8-27B, Qwen3.8 Flash Next, GLM-5.3-Flash, Qwen3.6-35B-A3B, and Nemotron, with family-specific NVFP4, EXL3, and MLX 4-bit support. EXL3 supports 3inst/mcg/mul1 codebooks and widths 1–8. TensorFold reads no GGUF ([CUDA checkpoint matrix](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/docs/recipes/cuda.md#checkpoints)).

CUDA graphs exist both in the Python families and the native wrapper [zig/src/cuda/graph.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/cuda/graph.zig). Speculation includes MTP chains, DFlash2 trees, copy/prompt-lookup grafts, and DSpark in DeepSeek forks. These features are real, but they are not uniformly available across families/backends.

The new cluster planner can deterministically place pipeline × tensor ranks and either spread or split experts ([zig/src/cluster/plan.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/cluster/plan.zig)); [backend_test.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/cluster/backend_test.zig) checks exact toy-model rounds at 1–4 nodes. This is credible architectural groundwork, not evidence that arbitrary CUDA DeepSeek TP/EP is production-ready. Native GLM CUDA TP/EP/DCP is itself still open as [#509](https://github.com/ashhart/TensorFold/issues/509).

### Kernel stack

The native runtime states its boundary directly: "our kernels and Triton's, no Python" in [zig/src/cuda/root.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/cuda/root.zig). It has owned CUDA kernels for experts, q4 prefill/GEMM, GDN, sampling, attention, and model glue under [zig/kernels/cuda](https://github.com/ashhart/TensorFold/tree/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/kernels/cuda). Triton kernels are captured ahead of time and loaded as cubins with exact launch metadata in [zig/src/cuda/aot.zig](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/zig/src/cuda/aot.zig). cuBLASLt and NCCL are runtime-loaded. I found no CUTLASS or NVIDIA cuTile implementation in the inspected tree.

### Maturity, license, cadence, contributors, issues

- **License:** Apache-2.0 ([LICENSE](https://github.com/ashhart/TensorFold/blob/f8fe17d24629aedabf90bbf78279dd776e6d62e7/LICENSE)). Model/drafter licenses remain separate; notably Aevonix's default DFlash2 is non-commercial.
- **Cadence:** 26 tags from v0.2.0 on 2026-09-25 to v1.0.2 on 2026-10-07, often several releases per day. This is exceptional velocity and also migration risk.
- **Scale/concentration:** at inspection: 791 commits, 67 Git authors; GitHub listed 57 code contributors. The top two accounts had 492 and 92 contributions. Repository metadata showed about 1.1K stars, 192 forks, and 112 open issues/PRs. There were no GitHub Actions workflows in the commit.
- **Operationally relevant open issues:** DeepSeek CUDA [#299](https://github.com/ashhart/TensorFold/issues/299); graph recapture causing 25–60% single-stream loss [#336](https://github.com/ashhart/TensorFold/issues/336); GLM snapshot clone memory spikes [#420](https://github.com/ashhart/TensorFold/issues/420); EXL3 byte-fallback mojibake [#442](https://github.com/ashhart/TensorFold/issues/442); cold prefill 1.4–1.5× behind vLLM [#447](https://github.com/ashhart/TensorFold/issues/447); prefix-cache failure at 3+ requests and 60% loss [#459](https://github.com/ashhart/TensorFold/issues/459); TP2 unbounded wait [#512](https://github.com/ashhart/TensorFold/issues/512).

**Maturity judgment:** impressive, high-signal engineering, but not a stable general replacement for our qualified stack. The rewrite, family matrix, release rate, absent CI workflow, and issue profile all argue for selective borrowing.

## 2. CPU/DRAM/NVMe offload

### What exists

The frozen Python/MLX engine has genuine **SSD expert streaming**. With `--ssd-experts`, GLM leaves 159.5 GiB of routed experts in the checkpoint and fills an LRU GPU pool after the router selects experts. A Metal shared event hands picks to the host; reads bypass the page cache; the resident expert kernel runs with a changed weight address. On an M3 Ultra, a 64 GiB pool reduced peak footprint from 184.0 to 87.6 GiB, but decode fell from 62.8–73.6 to 8.9–10.5 tok/s and 2K prefill from 451 to 144 tok/s ([GLM recipe, Experts on SSD](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/docs/recipes/glm-5.3-flash.md#experts-on-ssd)). Flash Next also supports `--ple-on-ssd` and expert pools; a 24 GiB expert pool ran at 31–39% of resident decode speed ([Flash Next recipe](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/docs/recipes/qwen3.8-flash-next.md)).

That path is **Mac/MLX-only, in the frozen Python engine, and not a CPU compute tier**. Current native CUDA has direct-I/O primitives and direct checkpoint loading, but no router split that executes selected experts on CPU. The community DeepSeek families' O_DIRECT/AIO/io_uring code fetches **Engram rows**, not routed-expert weights or activations.

### Fit for our system and cost to add

A kt-kernel-equivalent TensorFold port would require all of the following: a GPU/CPU expert placement contract; split routing and deterministic merge; CPU MXFP4 expert kernels; pinned activation/result staging; overlap with GPU experts; NUMA/CCD-aware work assignment; backpressure/admission; graph boundaries and failure handling; and long-context/concurrency qualification. Our bottleneck is the two CCD-to-IOD links, so merely wiring a generic CPU GEMM is not enough—the scheduler must preserve locality and overlap around roughly 132 GB/s aggregate fabric.

TensorFold's family-local row contracts and insistence on exact serial equivalence make kernel experiments clear. Its current transition, narrow CUDA registry, family forks, and lack of a stable heterogeneous placement API make product integration **harder**, not easier, than extending the SGLang+kt-kernel stack that already owns this boundary.

**Estimate:** several engineer-months for a minimally credible CUDA CPU-expert tier, plus hardware soak/quality work. It is not justified as a prerequisite to serving the same model we already run.

## 3. The autotuner

The tuner we previously examined is **not upstream TensorFold**. It is Aevonix recipe code at [scripts/glm_autotune.py](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/blob/a3fca2f7dcb3f9004908661b5a01cda60a204957/scripts/glm_autotune.py), enabled by recipe patches [0064](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/blob/a3fca2f7dcb3f9004908661b5a01cda60a204957/patches/aevonix/0064-glm-launch-tables.patch) and [0076](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/blob/a3fca2f7dcb3f9004908661b5a01cda60a204957/patches/aevonix/0076-glm-prefill-3.patch).

### Shape enumeration and coverage

- Shapes are derived from `config.json` plus TP/rank layout, or—better—from exactly observed engine calls written by `TF_GLM_TUNE_RECORD`.
- Decode buckets are upper bounds 1, 2, 4, 8, 16, 32, 63; each candidate is checked at both low and high rows. Prefill can use explicit rows such as 256, 512, 1024, 2048, 4096.
- Implemented tuners cover EXL3 routed-expert decode, q4 decode, q4 prefill (12 stock plus 24 added variants), FP8 decode, BF16 decode, KDA step, EXL3 prompt experts, and sparse-index score grids. l2pf only writes a heuristic wave size; it is not measured.
- The shipped preparation command actually tunes exl3_dec, kda_step, q4_prefill, and exl3_prompt ([scripts/prepare.sh](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/blob/a3fca2f7dcb3f9004908661b5a01cda60a204957/scripts/prepare.sh#L190-L209)). This is not a universal attention/MoE/server autotuner.

### Selection, correctness, timing, persistence

- Every candidate runs through the engine's actual call path with a launch-table override.
- Outputs are sentinel-filled first, then compared **byte for byte**, so signed zero, NaN payload, and unwritten bytes matter.
- Multiple fixed random seeds and both bucket boundaries must pass; one mismatch rejects that candidate for the whole shape.
- Independent references exist where practical: grouped EXL3 and fused KDA paths. q4 uses the current production tile as its oracle, so it proves bit-equivalence to current behavior, not independent mathematical correctness.
- Timing flushes at least 2× L2 (minimum 256 MiB), rewarms activations, uses CUDA events, rotates candidate order, and compares medians with p10–p90 spread. A winner needs at least 2% gain and must clear observed spread.
- JSON output is keyed by GPU name, SM count, capability, kernel, shape, and row bucket, and keeps measurements/settings as an audit trail. The recipe hashes patches, TP, GPU fingerprint, dense/load mode, and the tuner script into `.autotune-ready` before reuse.

The important caveat is in the recipe's own port report: **swapping only the saved tune table did not consistently improve end-to-end performance**, so defaults were restored ([PORT-v0.6.5.md](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/blob/a3fca2f7dcb3f9004908661b5a01cda60a204957/docs/PORT-v0.6.5.md#L101-L103)). The 5–77% figures are per-shape microkernel wins, not demonstrated server-level gains.

### What to reuse

Reuse the **method and artifact contract**, not GLM/EXL3 tile values:

1. Instrument actual SGLang/kt calls to record (kernel, M, N, K, dtype/format, rows, device, TP placement), including asymmetric low/high bucket boundaries.
2. Keep a constrained candidate registry for our SM120 q4/MXFP4 prefill and kt expert kernels.
3. Require independent-reference or current-qualified-path equality before timing.
4. Time L2-cold and rotate candidates; retain defaults unless the gain clears noise.
5. Persist measurements and invalidate on code, compiler, GPU, model layout, TP, chunk size, and kt placement changes.
6. Add an **end-to-end acceptance gate** (C1, C4, 16K/32K prefill, and CPU-fabric counters), because kernel wins can move bottlenecks or hurt overlap.

**Effort:** about **1–2 engineer-weeks** for shape recording plus a GPU prefill tuner and auditable cache; **2–4 additional weeks** for kt CPU-expert batch shapes, CCD/NUMA-aware buckets, invalidation, and end-to-end/soak integration.

## 4. Community recipes and numbers

These are not apples-to-apples benchmarks. Hardware, model, quantization, speculation, context, prompt mix, and whether prefill uses CED all differ.

| Build | Hardware / format / context | Reported result | What it adds beyond the supplied project description |
|---|---|---|---|
| [Aevonix GLM-5.3 recipe](https://github.com/Aevonix/GLM-5.3-Flash-EXL3-2x-RTX-PRO-6000-TensorFold/tree/a3fca2f7dcb3f9004908661b5a01cda60a204957) | 2× RTX PRO 6000, 300 W; 4 bpw EXL3 routed experts, q4 dense, FP8 KV; 512K/request, 4 slots; TensorFold 0.6.5 + 86 patches | C1 prose/code 238.3/350.8 tok/s; C4 aggregate 448.7/584.5; cold prefill 5,729/6,307/6,884/6,908/6,730/6,372 tok/s at 8K/16K/32K/64K/131K/262K | Per-shape launch tables, extensive GLM-specific fused kernels, PCIe overlap, DFlash2, shared KV pool. Different model and checkpoint quality; DFlash2 weights are non-commercial |
| [jayleaton DeepSeek recipe](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/tree/4b235ad21f73604d6ac5cc1f1225b5cceda6c005) | 2× DGX Spark GB10; DeepSeek-V4.1 EXL3 2.9 bpw, FP8 KV, DSpark, 4 slots; Engram on NVMe; 299K tested | C1 code/prose 84.7/46.8; C4 aggregate 101.1, 136 steady; CED replay prefill 2,028/2,310/2,294/2,118 at 8K/32K/64K/128K; full prefill 1,561/2,025/2,197/2,152 | CED bounded replay, exact DSpark integration, ahead-of-use Engram reads, grouped EXL3 paths, extensive exactness/soak gates |
| [Bertholomus DeepSeek recipe](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tree/890b929e648c95a915d446f1a606925d9054db31) | 2× GB10; Mia 2.9 bpw EXL3, DSpark, original Engram shards; 1,048,576 context with parallel=4 qualified | C1 code/prose/structured 100.5/62.5/140; C4 112.4 aggregate, 125.1 sustained; prefill 2,125/2,189/2,123/2,004 at 8K/32K/64K/128K; decode after 128K 128.7 | O_DIRECT/Linux AIO Engram read-ahead, long-context graph/indexer work, exact grouped EXL3 kernels |
| TensorFold issue [#479](https://github.com/ashhart/TensorFold/issues/479) | 4× GB10 TP4; native weights and a 3-bit hybrid tested; private real-traffic replay | Final native TensorFold wall c1/c2/c4 184/112/269 s vs local SGLang 162/88/238; diverse prose 43.5/62.9/80.1/115.1 vs 52.5/79.5/106.4/152.1; code 84.9/116.7/146.0/181.2 vs 104.3/147.1/215.4/274.3; prefill 3,162/3,540/3,668/3,678 vs 3,794/3,835/3,400/3,564 at 16K–128K | Best direct evidence: long prefill can catch/pass SGLang, but mixed serving and decode remained behind after substantial tuning |

The DeepSeek recipes' most relevant differences from our notes are:

- **CED bounded replay**, roughly halving decoder-side prompt work. First confirm whether our SGLang overlay already enables the V4.1 path and with what exactness/quality contract.
- **DSpark speculative decoding** with model-specific confidence/depth handling. If absent in our stack, evaluate an SGLang-native implementation; do not transplant a TensorFold fork wholesale.
- **Engram row-address I/O**: hash early, deduplicate, and issue O_DIRECT/AIO/io_uring reads before the consuming layer, rather than relying solely on demand faults/page-cache residency of two ~100 GB shards.
- Extensive exactness gates: drafted = serial, concurrent = solo, resumed = fresh, and row/chunk invariance.

They do **not** solve our defining routed-expert problem: their model weights fit through aggressive 2.9 bpw EXL3 across unified-memory GB10 nodes. None combines native-quality MXFP4 with 165 CPU/DRAM experts per layer on our one-box topology.

## 5. Recommendation and decision triggers

### Adopt now

1. **Build our shape tuner using the Aevonix methodology.** Start with SM120 prefill q4/MXFP4 shapes and only then add kt CPU-expert batch/CCD shapes. Estimated 3–6 weeks total in two independently useful stages.
2. **Run a bounded Engram I/O experiment** (1–2 engineer-weeks): record page faults, NVMe latency/read amplification, page-cache churn, and CPU-expert fabric pressure; compare current mmap/page cache with deduplicated row prefetch via io_uring/O_DIRECT. Adopt only if C1/C4 tails or long-prefill wall time improve without stealing DRAM bandwidth from kt.
3. **Audit CED bounded replay in our overlay**. If absent, prototype it as a model-level SGLang feature, not a TensorFold migration. Budget roughly 2–4 weeks plus quality/long-context qualification; its reported prefill gain is large enough to justify the check.
4. Reuse the recipes' **qualification invariants and benchmark discipline**, especially diverse prompts; issue #479 found same-prompt concurrency overstated MoE batching by 2–3×.

### Do not adopt now

- Do not switch serving engines.
- Do not port TensorFold's Mac SSD expert pool; it streams weights and stalls, rather than executing experts from DRAM, and its measured speed penalty is severe.
- Do not adopt the EXL3 2.9–3.0 bpw recipes as performance evidence for our accepted quality point.
- Do not copy saved launch tables or GLM-specific kernels; the reusable asset is the search/verification framework.

### Evidence that would change the verdict

Reconsider an engine switch only after all of these exist:

1. Upstream native Zig **DeepSeek-V4.1 CUDA** support on SM120, not a private Python 0.6 fork.
2. Qualified CUDA TP2/EP plus a **CPU/DRAM expert tier** and NVMe Engram path compatible with our placement.
3. Native MXFP4/no-quality-loss checkpoint support at our model fidelity.
4. Reproducible same-model, same-weights, same 2× RTX PRO 6000 benchmark showing a material gain over our C1 ~40, C4 ~58, 16K–32K prefill 1,000–1,500, and equal or better 64K×4 / 512K×1 behavior.
5. Multi-hour stability with graph recapture, prefix-cache, memory-spike, and bounded-wait issues closed or demonstrably irrelevant.
6. Autotune evidence that improves **end-to-end serving**, not only isolated kernel timings.

Until then, TensorFold is a useful source of kernel/scheduler experiments and unusually rigorous exactness practices, but SGLang+kt-kernel remains the only evaluated architecture here that satisfies our quality, heterogeneous-memory, and topology constraints.

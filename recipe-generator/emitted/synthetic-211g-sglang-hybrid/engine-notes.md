# Engine notes - sglang-hybrid (the reference recipe's engine)

## The advanced patch inventory (what this project changed in SGLang)

| group | patches | what | portability |
|---|---|---|---|
| placement / expert map | 0001, 0044-0046, 0061 | compact CPU-expert view, per-expert runner, per-layer expert remap at load, cluster-path fix | RFC + local map file (the el8 map is model-specific) |
| Engram file layout + prefetch | 0002, 0040-0043, 0047 | tables stay in the checkpoint shard; ahead-of-use row fetch, one hasher break, pinned staging, chunked hashing | recipe-local (model-specific lookup tables); the prefetch *pattern* ports |
| breakable CUDA graphs | 0003, 0009-0017 | BCG dataclass outputs, per-field row counts, capture-width scaling, fault attribution, page-split never-free | upstream candidates PR 3/4 + diagnostics |
| W8A8 block-FP8 [32,32] configs | 0004 | seven tuned Triton configs for RTX PRO 6000 Max-Q; 2.26x C1 (E20) | upstream candidate PR 1 |
| page-split seals | 0005, 0015, 0022 | zero-fill scratch (stale FP8 NaN), reserve-split seal | upstream candidate PR 3/4 |
| kt deferral | 0006 | expert deferral with the compact CPU view (D=4, +22% C1) | RFC |
| streamed prefill | 0007, 0030, 0035, 0048-0049 | layerwise GPU prefill coordinator, PrefillPipeline, E8M0 scale staging, shared weight arenas (+34-41% prefill) | RFC |
| B2 direct-UVA experts | 0019, 0029 | built, measured +2%, retired | local (retired - do not port) |
| kt boundary contract | 0020, 0024, 0028 | executable SGLang-kt contract, CpuExpertProvider seam, KtRuntime ownership | RFC |
| DSpark overlap gate | 0021 | multi-stream overlap forced off for DSpark on SM120 (Bug A) | upstream candidate |
| timeline / fingerprints | 0023, 0056-0058 | per-layer decode timeline recorder + step fingerprints (opt-in diagnostics) | diagnostics (opt-in) |
| indexer tiling + score budget | 0025-0027, 0031, 0033 | 256 MiB score budget, tiled dense fp4 scoring, block-id candidate masks, page-offset gathers (~2x prefill) | upstream candidate PR 6 |
| expandable segments after capture | 0039 | allocator switch kills 262Kx4 stress fragmentation (678 vs 1,082 s) | upstream candidate |
| robustness | 0050-0052 | per-request NaN row rejection, max_new_tokens clamp, API-key file | upstream candidates |
| deterministic decode | 0053-0055 | sorted decode top-k + exact deterministic ties (bit-reproducible greedy) | upstream candidate (sglang #35257 post ready) |
| NGRAM chain-only verify | 0036-0038, 0059 | speculation for pools without move_kv_cache; inert unless NGRAM=1 (E43 found unsafe) | local, off by default |
| decode-shape FP8 GEMV | 0060 | opt-in +1-2% C1 served; stays 0 until a pinned A/B | local, opt-in |

Base `da64c5cb` + 61-patch series (`patches/sglang/` in the parent recipe; exported head `deec7a6f`, tree `adfc1b0f`); kt-kernel side is 3 patches in `config/kt-kernel-patches/`; kernels `e13` (MXFP4 grouped GEMM), `e20` (W8A8 FP8 configs), `t321`, `t341`. The image never contains the patches: they are bind-mounted at run time.

## The deep-lever surface

The full deep-lever surface a generated sglang-hybrid recipe inherits (each with the
measured number behind its default; complete table in registry-surface.json):

- KV cache: CONTEXT_LENGTH x MAX_RUNNING_REQUESTS (262K x 4 default; ladder 512K/1M needs
  MEM_FRACTION_STATIC=0.962 - at 0.96 the pool silently caps at 505,856 tokens), MAX_TOTAL_TOKENS
  (engine default sized the pool 15x too large, E26), SGLANG_LOGPROB_CHUNK_SIZE=512 (a logprob
  request otherwise materializes 1.06 GB of fp32 logits and a CUDA OOM there kills the server).
- Expert placement: KT_NUM_GPU_EXPERTS + KT_EXPERT_MAP (219/165 + el8 map: CPU share 42->15%,
  C1 +37%, C4 +120%), KT_CPUINFER=16 (24 -> -12%; 30 -> 5 tok/s collapse), KT_MAX_DEFERRED=4
  (+22% C1; D=5 fails NLL - guarded), KT_GPU_PREFILL_MIN_TOKENS=1024 (prefill 216->1,375 tok/s).
- Prefill: CHUNKED_PREFILL_SIZE=8192 (16384: +22% prefill for -4.4% C1; guarded beside >219 experts).
- Engram/NVMe: ENGRAM_FILE_PREFETCH=ahead (+16-26% prefill), ENGRAM_CHUNK_AHEAD=1 (+7.6-11%),
  ENGRAM_ADMISSION_PREFETCH (off; -0.2-0.4 s cold TTFT when on), page-cache hygiene at ready.
- CUDA graphs: CUDA_GRAPH_BS="1 2 3 4" (eager is 4.4-5.6x slower; bs=3 capture +28% C3).
- Determinism: SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 + SGLANG_DSV41_SORT_DECODE_TOPK=1
  (bit-reproducible greedy at zero cost), exact top-k patch 0055.
- Robustness: KT_REJECT_NAN_ROWS=1 (one 500, server up), SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS=1.
- Allocator: SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1 (stress 678 vs 1,082 s).
- Host: HOST_THREAD_CPUS=auto (kills the C4 106/123 bimodality), BIOS FCLK 2100 slot 3 (+3.1% C1),
  power sweep protocol (decode cap-insensitive, prefill pins to the cap - per-model).
- Kernel: KERNEL_CACHE_DIR (kills the +14-19 s first-request Triton compile), SGLANG_DSV41_FP8_GEMV
  (opt-in +1-2% C1), REASONING_EFFORT (prompt-prefix only).
- Speculation: DSPARK=0 / NGRAM=0 with reopen triggers (CPU-tier change; spec-consistency.py gate).
- Guards/escape hatches: ALLOW_UNTESTED_FIT, KT_ALLOW_UNSAFE_DEFERRAL, ALLOW_UNKNOWN_KNOBS,
  EXTRA_* - exist so an experiment is *recorded*, refused everywhere except PROFILE=developer.


## What ports vs stays

- Ports to a new engine/model: the W8A8 FP8 configs (PR 1), page-split seals (PR 3/4), indexer tiling (PR 6), the overlap gate, the robustness clamps, the deterministic top-k (upstream post ready), the expandable-segments switch.
- Stays recipe-local: the Engram file layout + prefetch (model-specific lookup tables), the el8 hot-expert map file (fitted to this model's gates), the B2 UVA residue (retired), the FP8 GEMV opt-in (until a pinned A/B clears +2%).
- Diagnostics (opt-in, inert by default): BCG debug, timelines, step fingerprints.

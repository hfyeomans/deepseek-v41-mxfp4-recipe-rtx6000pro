# LEVERS.md — the lever dictionary (tuning mode)

Every lever the generator can turn, in the order they paid off on the reference build. Format per lever:
**what it moves · direction of effect · cost · when it helps · measured numbers · generator knob**.
The measured numbers are this host + DeepSeek-V4.1-Flash (2× RTX PRO 6000); they are calibration values for
the tuning model, never universal claims. Full citations in `recipe-factory/KNOBS-AND-PROFILES.md` §2 and
`recipe-factory/TRIED-OPTIONS-LEDGER.md`; the machine-readable form is [`levers.json`](levers.json).

## 1. Placement family (the biggest lever)

### 1.1 GPU-expert count (`KT_NUM_GPU_EXPERTS`) — `placement_calc.py --json → plan.gpu_experts`
- **Moves:** how much of the routed-expert traffic stays in VRAM vs streams from DRAM on the CPU tier.
- **Direction:** more GPU experts → less CPU traffic → higher C1/C4, but each expert costs VRAM that KV or
  chunk size could use.
- **Cost:** VRAM bytes; nothing else. Same-bytes placement paid the full byte-model gain (E28: +6.3 % C1,
  +9.2 % C4 for 226 vs 208).
- **Measured:** 208→226: C1 +6.3 % wall; 226→230: +2.5 % C1 wall but **−40 % prefill and NLL +0.0088 (fails the
  gate)** — the 230 row is why the NLL gate exists. Chunk 8192 forces ≤219 experts (fused-MoE workspace).
- **Generator:** the calc picks the split from capacity; the tuning mode lets you trade experts against chunk
  and see both projections move.

### 1.2 Hot-expert map (`KT_EXPERT_MAP`) — `--map {checkpoint_order,fitted_static,runtime_oracle}`
- **Moves:** which of the CPU-resident experts the router actually hits: CPU route share.
- **Direction:** checkpoint order = random split (**42–46 %** CPU share); a fitted static map (el8-style)
  = **15 %**; runtime replacement = 1–3 %.
- **Cost:** zero VRAM — a permutation is routing-exact, not bit-exact (NLL moved −0.00033 ± 0.00101, inside the
  band).
- **Measured:** el8 map: **C1 42.4 → 58.0 (+37 %), C4 +120 %** (T3.8b). Own-counts refits overfit (T3.8c):
  hot sets are content-specific, so a **static map's floor is 10–15 %** CPU share. Runtime placement is the
  only way past the floor (PARKED, GitHub #6; watch SGLang Paged Experts PR #42557 and llama.cpp #27861's LRU
  finding).
- **Generator:** `plan.cpu_share`; the three map qualities are the modeled endpoints.

## 2. CPU tier family

### 2.1 Deferral depth (`KT_MAX_DEFERRED`, D) — `--deferral`
- **Moves:** how many tokens the CPU expert pass may lag behind the GPU pass.
- **Cost:** nothing in memory; the cost is quality at the edge: **D=5 fails NLL** (+0.00287) and diverges at
  9–39 chars; D=6 loses 0.23 nats/token.
- **Measured:** D=4 → **+22 % C1** over D=0, NLL +0.00066 ± 0.00103 (passes).
- **Generator:** the decode model's `penalty(D)`; guard refuses D>4.

### 2.2 CPU worker threads (`KT_CPUINFER`) — not yet modeled; fixed at 16
- **Moves:** CPU-tier GEMV throughput on a bandwidth-bound tier.
- **Measured:** 14/16 = plateau; 24 → −12 %; 30 → collapses to 5 tok/s (SMT starvation). Never above physical
  cores. The tier is bandwidth-bound: DRAM read ceiling 131 GB/s (2 CCDs), 94 GB/s while computing (E6/E25).

### 2.3 Host thread pinning (`HOST_THREAD_CPUS=auto`) — not modeled; standing guard
- **Moves:** scheduler/tokenizer threads off the worker cores.
- **Measured:** floating threads gave **C4 106 or 123 on the same serve** (bimodality); `auto` → 126.7/123.0,
  no 106 mode. A list overlapping a worker CPU is refused (forced collision C4 33.8).

## 3. Memory / context / prefill family

### 3.1 KV context × slots (`CONTEXT_LENGTH`, `MAX_RUNNING_REQUESTS`) — `--context-tokens --sequences`
- **Moves:** the VRAM share of KV; the served concurrency shape.
- **Cost:** `kv_bytes_per_token × context × slots`; the ladder 262K×4 → 512K×1 → 1M×1 needs fraction 0.962
  (at 0.96 the pool silently caps at 505,856 tokens — guarded).
- **Measured:** 262K×4 costs ~no prefill/decode against 64K×4 (T3.39 overlay); >262K requires 1 slot.
- **Generator:** `plan.kv_gib`, `plan.kv_fraction`; the ladder rungs are the profiles' knobs.

### 3.2 Prefill chunk (`CHUNKED_PREFILL_SIZE`) — `--chunk-size`
- **Moves:** TTFT vs decode rate vs expert capacity (fused-MoE workspace scales with the chunk).
- **Measured:** 4096→8192: first-turn TTFT down, prefill **+38/+47 %** at 16K/32K; 16384: +22 % prefill **for
  −4.4 % C1 and 14 experts** (rejected); 2048: frees ~1.5 GB/rank **but −40 % prefill**.
- **Generator:** §4.4 of PLACEMENT.md — the chunk↔expert fitted line; auto picks the largest preferred chunk
  that fits.

### 3.3 CUDA graph capture (`CUDA_GRAPH_BS="1 2 3 4"`) — not modeled; standing default
- **Measured:** eager = 4.4–5.6× slower; derived [1,2,4] pads bs=3 → C3 −22 %; explicit bs=3 costs ~nothing.

## 4. Engram / NVMe tier family

### 4.1 Engram tier (`ENGRAM_FILE_PREFETCH` etc.) — `plan.engram_tier`
- **Moves:** where the giant lookup tables live and how their rows get fetched.
- **Measured:** DRAM as-shipped does **not** fit this host (−78 GiB); NVMe + page cache sustains decode-rate
  gathers (403K rows/s, E3); re-quantized tables **cannot shrink the page-cache footprint** (~190 KiB per
  first-touched token, E40) — the real fix was cache hygiene: dropping ~73 GiB of stale shard pages took
  prefill 16K 3,311 → 4,248 tok/s and stress 320 → 257 s.
- **Generator:** tier decision in §3; row-gather budget in §4.5; `--admission-prefetch` models the T3.49
  knob (−0.2–0.4 s per cold TTFT).

### 4.2 Row cache (`ENGRAM_ROW_CACHE_MB`) — `plan.engram.row_cache_gib`
- **Measured:** user-space row cache (264 B rows) is ~22× denser than the page cache; parked (T3.48), the
  coding-agent profile is its home. The generator recommends it only when the NVMe row budget fails.

## 5. Speculation family (off, and why — reopen triggers recorded)

- DSpark K=2: parity C1, loses C3/C4; K=5: **−16 % C1, −69…−77 % C3/C4** (every verify step routes bs×(K+1)
  CPU-expert rows). n-gram done properly: +60/+76 % C1 on copy prompts, **−26 % review, −63 % C4**, and a
  decode-state defect under rejected drafts (one token at −20.5 nats — `spec-consistency.py` is now the gate).
- **Reopen only after a CPU-tier change** (third GPU, more CCDs), and any speculative path must pass
  `spec-consistency.py` cold. The generator defaults speculation off and records the reopen trigger.

## 6. Quantization family — `--quant` (see `quantization.json`)

| Option | Size vs budget baseline | Quality | Kernel reality (SM120) | Verdict source |
|---|---|---|---|---|
| mxfp4 (as shipped) | 1.0 | trained-on | native CUTLASS, roofline at decode | ADOPT |
| nvfp4 | +5.9 % experts | re-quant | better engine support; larger | REJ for size |
| fp8-dense | dense ×2 | quality-neutral (E20) | the measured hot spot; 2.26× C1 when tuned | ADOPT where VRAM allows |
| exl3 2.9–3.0 bpw | ×0.65 | unknown on FP4-trained | bespoke forks only | REJ here |
| gguf-q4_k_m (bf16 budget) | ×0.307 | published ladder | llama.cpp −ot/−ngl offload | PARKED route (T3.7) |

Rule from the ledger: quantization buys *size*, never quality-neutrality for free — every format change that
alters the math re-runs the NLL identity gate.

## 7. How the tuning mode consumes this

Each lever row in [`levers.json`](levers.json) carries: id, family, knob names, what-it-moves, direction,
cost, when-it-helps, measured (list of {setting, result, evidence}), generator mapping (CLI flag or plan
field), and reopen trigger where one exists. The UI's tuning panel renders these rows; the calculator's CLI
accepts the generator-mapped flags; the profile files ([PROFILES.md](PROFILES.md)) are deltas over this
dictionary, so "why this preset" always points at a measured row.

## 8. The complete registry surface (the deep-lever coverage)

The dictionary above is the *modeled* levers; the full surface a generated recipe inherits is the parent
recipe's 70-knob typed registry — the crown jewel (`scripts/serve_config.py`): every knob declared once with
type, default, sink, evidence, aliases, guards; unknown/mistyped/removed/unsafe values **refused at launch**.
The generator ships it machine-readable as [`registry-surface.json`](registry-surface.json) (name/type/default,
extracted 2026-10-23) and emits it into every generated recipe; the UI's knob panel renders it, so the UI can
never invent a knob the registry would refuse. Beyond the families above, the surface includes: `MAX_TOTAL_TOKENS`
and the pool-cap guard, `SGLANG_LOGPROB_CHUNK_SIZE`, `KT_GPU_PREFILL_LOG` / `KT_NAN_PROBE_MIN_TOKENS`
(diagnostics), `SGLANG_BCG_DEBUG_*` and the step-fingerprint family (opt-in diagnostics from the breakable-graph
work), `KT_TIMELINE*` (per-layer decode timeline), `API_KEY_FILE` / `BIND_ADDRESS` / `PORT` (deployment),
`BIOS_SLOT`, `KERNEL_CACHE_DIR`, `MANIFEST_DIR`, `DRY_RUN`, the `EXTRA_*` escape hatches and the guard bypasses
(`ALLOW_UNTESTED_FIT`, `KT_ALLOW_UNSAFE_DEFERRAL`, `ALLOW_UNKNOWN_KNOBS` — developer profile only), and the
removed knobs that refuse with the experiment that removed them (`KT_B2_UVA_EXPERTS` — B2 rejected at E30).

The **advanced patch inventory** behind those knobs — the 61-patch SGLang overlay, the 3 kt-kernel patches,
and the kernel artifacts — is documented with a portability verdict per group in
[RECIPE-EMISSION.md §2.2](RECIPE-EMISSION.md) and emitted into every generated recipe's `engine-notes.md`.

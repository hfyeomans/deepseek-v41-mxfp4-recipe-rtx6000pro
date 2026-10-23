# Recipe Factory — KNOBS AND PROFILES

The knob taxonomy generalized from `scripts/serve_config.py` (the parent repo's typed knob registry — the
"crown jewel": every knob declared once with type, default, sink, and the evidence that set the default), plus
the measured trade-off numbers from this project for each family, plus ready-made profile presets.

**Two layers of truth:** the *taxonomy* (which knob families exist and how they interact) transfers to any
model/hardware. The *numbers* are this host's measurements for DeepSeek-V4.1-Flash on 2× RTX PRO 6000 — they
are calibration values to start from, never to copy. Every number cites its evidence.

---

## 1. The knob taxonomy

| Family | What it controls | This project's knobs | Why it exists |
|---|---|---|---|
| **Memory fraction** | how much VRAM the engine may take for weights+KV | `MEM_FRACTION_STATIC` | the residual VRAM decides everything else: pool size, expert count, chunk size |
| **Expert placement** | which routed experts are GPU-resident | `KT_NUM_GPU_EXPERTS`, `KT_EXPERT_MAP` | the single biggest decode lever on a CPU-hybrid MoE |
| **CPU tier** | worker threads, deferral depth, prefill handoff | `KT_CPUINFER`, `KT_MAX_DEFERRED`, `KT_GPU_PREFILL_MIN_TOKENS` | the CPU tier is bandwidth-bound; these trade latency for overlap |
| **Speculation** | draft-token mechanisms | `DSPARK`, `DSPARK_K`, `NGRAM`, `NGRAM_DRAFT_TOKENS`, `NGRAM_BREADTH` | verified draft rows multiply CPU-expert traffic; acceptance decides |
| **KV cache** | pool sizing, context, slots | `CONTEXT_LENGTH`, `MAX_RUNNING_REQUESTS`, `MAX_TOTAL_TOKENS` | pool = slots × context + one chunk; leaving it to the engine sized it 15× too large |
| **Prefill chunking** | prefill granularity | `CHUNKED_PREFILL_SIZE` | TTFT vs memory: bigger chunks prefill faster but cost workspace and experts |
| **Offload tiers / Engram** | where the giant lookup tables live and how they are fetched | `ENGRAM_FILE_PREFETCH`, `ENGRAM_FILE_THREADS`, `ENGRAM_CHUNK_AHEAD`, `ENGRAM_ADMISSION_PREFETCH` | page-cache discipline: ahead-prefetch overlaps, admission-prefetch hides chunk 0 |
| **CUDA graphs** | which decode batches are captured | `CUDA_GRAPH_BS`, `CUDA_GRAPH_MAX_BS` | eager is 4.4–5.6× slower; derived capture lists miss bs=3 |
| **Determinism** | bit-reproducible greedy decode | `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE`, `SGLANG_DSV41_SORT_DECODE_TOPK` | fused finalize + unsorted top-k = run-dependent output at temperature 0 |
| **Allocator levers** | segment strategy around capture | `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE` | chunk-sized prefill temporaries fragment around the graph pools |
| **Robustness** | per-request fault containment | `KT_REJECT_NAN_ROWS`, `SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS`, `SGLANG_LOGPROB_CHUNK_SIZE` | a NaN or a logprob OOM must kill one request, not the server |
| **Host threads** | which CPUs the server's non-worker threads use | `HOST_THREAD_CPUS` | floating threads landing on CPU-tier worker cores cause bimodal C4 |
| **Reasoning effort** | the default reasoning budget rendered for thinking requests | `REASONING_EFFORT` | only the prompt prefix changes — a pure quality/latency knob, no kernel effect |
| **Kernel selection** | opt-in kernel paths | `SGLANG_DSV41_FP8_GEMV` | ships opt-in until a pinned A/B shows a real served gain |
| **Escape hatches / guards** | deliberate guard bypasses | `ALLOW_UNTESTED_FIT`, `KT_ALLOW_UNSAFE_DEFERRAL`, `ALLOW_UNKNOWN_KNOBS`, `EXTRA_*` | exist so an experiment is *recorded*, and refused everywhere except `PROFILE=developer` |

Registry mechanics every new recipe should copy (from `scripts/serve_config.py`):

- **One declaration per knob**: type (`int|float|str|flag|list`), default, doc, sink (`server-arg` /
  `container-env` / `diagnostic-env` / `launcher` / `guard`), evidence citation, aliases, `unsafe` flag,
  `removed` message. The launcher evals the registry output instead of carrying its own `${VAR:-default}` lines.
- **Owned name families**: an env var in an owned prefix (`KT_*`, `DSPARK*`, `ENGRAM_*`, …) that is not a
  declared knob is **refused** — a mistyped override cannot launch the default while the log claims otherwise.
- **Removed knobs refuse** with the experiment that removed them (`KT_B2_UVA_EXPERTS: removed at T3.16j, B2 rejected`).
- **Guards encode condemned combinations** with measured messages: D≥5 refused (fails the NLL gate), chunk 8192
  beside >219 experts refused (fused-MoE workspace OOM), >262K at fraction 0.96 refused (pool silently caps at
  505,856 tokens). Each names its override.
- **Profiles layer**: `registry default < profile file < environment`, so an env override is an experiment on
  top of a named profile, and the resolved line + manifest record the profile and every override.
- **The resolved line** (`resolved: experts gpu=219 cpu=165 deferral D=4 ... profile=qualified overrides=[...]`)
  is the one line a launch log carries so every knob that decides a result is named.

---

## 2. The measured trade-offs (this project, 2× RTX PRO 6000, V4.1-Flash hybrid)

Format: knob → default → what moving it costs/buys, measured. Full citations in the registry and
`docs/BENCHMARKS.md`.

### 2.1 Expert placement (the biggest lever)

| Setting | Result | Evidence |
|---|---|---|
| 208 GPU / 176 CPU experts | the first-serve placement | E19 |
| **226 / 158** (checkpoint order) | C1 +6.3% wall, C4 +9.2% over 208; worst case 779 MiB free | E28/T3.6g |
| 230 / 154 | +2.5% C1 wall **for −40% prefill**, and NLL +0.0088 **fails the gate** (needs chunk 2048 + fraction 0.975) | E29 — rejected |
| **219 / 165 + chunk 8192** | prefill 16K/32K +38/+47%, C1 wall −1.9% vs 226/4096; pooled NLL +0.00075 ± 0.00098 (inside the tier-rounding band) | T3.28/E34d-ii — the default |
| 219 / 165, **checkpoint order** (positional 0–218) | 42–46% of routes land on the CPU tier — a random split | T3.8 |
| 219 / 165, **el8 hot-expert map** (`KT_EXPERT_MAP`) | CPU route share 42 → **15%** (12–15% on real text, 23% on random words); **C1 42.4 → 58.0 (+37%), C4 55.8 → 123.9 (+120%)**; NLL inside the band (a permutation is routing-exact, not bit-exact); prefill unchanged; zero VRAM | T3.8b — the default since 2026-10-10 |
| own-counts map refit | loses to the el8 map at every blend weight on held-out families (13.3% vs 16.4% decode CPU share); hot sets are **content-specific** (per-set oracle 1–3%, cross-family 20–30%) → **a static map's floor is 10–15% CPU share** | T3.8c — no refit |
| runtime placement (T3.8d) | the only documented way past the floor; blocked upstream (MXFP4 dynamic update is a no-op; no resident-weight pointer) | GitHub #6 |

Rule: placement beats kernels — same-bytes placement paid the full byte-model gain; a permutation costs no
VRAM. The map transfers to a weight-refreshed checkpoint (1017) unchanged: 14.4% CPU share re-measured
(`evidence/t1017/`).

### 2.2 CPU tier (latency vs overlap)

| Knob | Default | Trade-off measured |
|---|---|---|
| `KT_MAX_DEFERRED` (D) | **4** | D=4: **+22% C1** over D=0, NLL +0.00066 ± 0.00103 (gate passes). D=5: **fails NLL** (+0.00287) and greedy consistency (diverges at 9–39 chars). D=6: −0.23 nats/token. D costs nothing in memory |
| `KT_CPUINFER` | **16** | 14 and 16 are a plateau; 24 → **−12%**; 30 → **collapses to 5 tok/s** (workers beyond physical cores float on SMT siblings and starve the TP launch threads). Never above physical cores |
| `KT_GPU_PREFILL_MIN_TOKENS` | **1024** | layerwise GPU prefill: prefill 216 → 837 (4K) → 1,375/1,404 (8K); NLL-neutral vs the CPU path (−0.00025 ± 0.00104). 0 keeps every chunk on the CPU path |
| `HOST_THREAD_CPUS` | **auto** (SMT siblings of worker cores) | floating host threads gave **C4 106 or 123 on the same serve** (bimodality); `auto` → 126.7/123.0 with no 106 mode, C1 60.7/60.3, NLL bit-identical, prefill −0.7…−1.9% (fewer logical CPUs for the gather/scheduler/tokenizer). A list overlapping a worker CPU is refused (forced collision C4 33.8) |

### 2.3 Speculation (off, and why)

| Setting | Result | Evidence |
|---|---|---|
| DSpark K=2 | parity at C1, loses at C3/C4 | E23 |
| DSpark K=5 | **−16% C1, −69…−77% C3/C4** — every verify step routes bs×(K+1) rows of CPU experts on a bandwidth-bound tier | E23 — `DSPARK=0` |
| n-gram (defaults) | first request crashes both ranks (DSV4 pool lacks `move_kv_cache`) | T3.9h |
| n-gram (chain-only verify, done properly) | +60/+76% C1 on copy prompts, **−26% review, −63% C4**, and a decode-state defect under rejected drafts (one emitted token scored −20.5 nats by the model itself; `spec-consistency.py` is now the gate) | T3.9h-b/E43 — `NGRAM=0`, not recommended |
| eager (no graphs) | 4.4–5.6× slower than captured | E15 |

Rule: reopen speculation only after a CPU-tier change (third GPU, more CCDs), and any speculative path must
pass `spec-consistency.py` cold.

### 2.4 Memory, context, prefill chunk

| Knob | Default | Trade-off measured |
|---|---|---|
| `MEM_FRACTION_STATIC` | **0.96** | 0.92→0.96 at T3.6g (pool capped by `MAX_TOTAL_TOKENS`). **0.962 is required for the 512K/1M rungs** — at 0.96 the pool silently caps at 505,856 tokens and long prompts 400. Guarded: >262K context refuses any other fraction |
| `MAX_TOTAL_TOKENS` | slots × context + 16,384 | leaving the pool to the engine sized it 15× too large (E26). Empty restores the fraction-derived size |
| `CONTEXT_LENGTH` | **262,144** | ladder: 32K×4 → 64K×4 → 262K×4 (default) → 512K×1 → 1M×1. Values >1,048,576 refused (the ladder stops there). 262K×4 costs ~no prefill/decode against 64K×4 on the T3.39 overlay |
| `MAX_RUNNING_REQUESTS` | **4** | 4 at 262K and below; **1 above 262K** (multi-slot >262K never ran). 262K×4 needs the expandable-segments lever |
| `CHUNKED_PREFILL_SIZE` | **8192** | 4096→8192: first-turn TTFT down (chunk is what buys long-session TTFT). 16384: +22% prefill **but −4.4% C1 and costs 14 experts**. 2048: frees ~1.5 GB/rank but **−40% prefill**. The fused-MoE workspace scales with the chunk, so 8192 fits only beside ≤219 experts (guarded) |
| `CUDA_GRAPH_BS` | **"1 2 3 4"** | SGLang derives [1,2,4] and pads bs=3 to 4 → C3 −22%; capturing 3 costs ~nothing (E26). Eager = 4.4–5.6× slower (E15) |
| `SGLANG_LOGPROB_CHUNK_SIZE` | **512** | a return_logprob request materializes chunk × 129,280-vocab fp32 logits (1.06 GB at the engine default) and a CUDA OOM there **kills the server, not the request**. 512 is bit-identical and no slower |

### 2.5 Engram / offload tiers (the NVMe + page-cache discipline)

| Knob | Default | Trade-off measured |
|---|---|---|
| layout | `file` (mmap + page cache, in-place from the checkpoint shards) | full FP8 tables in DRAM do **not** fit this host (−78 GiB); FP4/q3_K re-quantized tables fit but **cannot shrink the page-cache footprint** (real text touches ~1 row per 4 KiB page ≈ 190 KiB per first-touched token) — rejected on footprint grounds (E11/E40) |
| `ENGRAM_FILE_PREFETCH` | **ahead** (layer-14 rows fetched from layer-1's break) | prefill 16K/32K **+16/+20%** cold, +24/+26% warm; Engram break −24%; C1 +1.7% (T3.20) |
| `ENGRAM_CHUNK_AHEAD` | **1** | next chunk's rows hashed and fetched during the current chunk: prefill 16K/32K **+7.6/+11%** (T3.20c) |
| `ENGRAM_ADMISSION_PREFETCH` | **0** (default off) | fetch chunk 0's rows at admission: **−0.2…−0.4 s (3–9%) on a cold 16K TTFT**; null on an idle cold TTFT (t349). Advisory-only; stays off until a C4-heavy admission case exists. Chunk-ahead provably cannot hide chunk 0 — the cold request pays a flat **+1.7–3.1 s** (+24–66% of warm TTFT at 16K–32K) |
| page-cache hygiene | `wait-ready.py` drops ~73 GiB of stale non-Engram shard pages after load | prefill 16K **3,311 → 4,248 tok/s**, stress **320 → 257 s** — the real Engram fix was cache eviction, not quantization (E40/T3.10) |
| kernel cache | `KERNEL_CACHE_DIR` bind-mounted | the first request's extra ~19 s was serving-time Triton compile; a persistent cache keyed by source hash removes it (T3.47/E40) |

### 2.6 Robustness, determinism, quality

| Knob | Default | Trade-off measured |
|---|---|---|
| `KT_REJECT_NAN_ROWS` | **1** | an injected NaN → one HTTP 500, 3 rows complete, server up; well-formed rows bit-identical. 0 = stock: sanitize/finish silently |
| `SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS` | **1** | coding agents send a fixed ceiling (128K) which otherwise turns a 262K server into a 134K one via HTTP 400; clamping is the scheduler's own behavior, moved to the API |
| `SGLANG_FLASHINFER_MOE_FUSED_FINALIZE` + `SGLANG_DSV41_SORT_DECODE_TOPK` | **0 / 1** | the two sources of run-dependent greedy: fused finalize sums top-k in run-dependent order; indexer top-k order past 512 tokens is run-dependent. Both levers → **temperature-0 output bit-identical run to run at zero cost** (T3.15c) |
| `SGLANG_DSV41_FP8_GEMV` | **0** (opt-in) | decode-shape FP8 GEMV: +3…+51% per call at M≤4, but only **+1–2% C1 served**, C4 flat — stays 0 until a pinned A/B shows ≥+2% (T3.41/E44) |
| `REASONING_EFFORT` | **high** (=50) | SGLang's tiers (25/50/75/100) are a lower scale than DeepSeek's reference (50/75/100, benchmarks at 100). Only the prompt prefix changes — no kernel/memory/gate effect. The default is an owner-quality-vs-latency call (T3.54) |
| `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE` | **1** | after capture, switch the allocator to expandable segments so chunk-sized prefill temporaries stop fragmenting: 262K×4 stress **678 vs 1,082 s** at chunk 4096; NLL 0 moved, C1/C4/prefill within noise; +390 MiB min free (T3.31 idea 8) |

---

## 3. Profile presets

The pattern: a profile is a named `.env` file layered under the environment (`registry default < profile <
environment`), so "the qualified shape" is a file, an experiment is an env override on top of it, and the
resolved line records both. Below are the preset *shapes* for the use cases the owner cares about, expressed
on this project's qualified defaults (the worked example); a new recipe rebuilds each preset from its own
measurements.

### 3.1 `qualified` — the served default (262K × 4)

What every headline number was measured on: 219/165 experts via the el8 map, D=4, `KT_CPUINFER=16`, chunk
8192, fraction 0.96, 262K × 4 slots, capture bs 1–4, Engram `ahead` + chunk-ahead, expandable-after-capture,
NaN rejection, clamp, deterministic-greedy levers, `HOST_THREAD_CPUS=auto`, reasoning effort high, spec off.
C1 60.8/58.8, C4 124.0/118.9 wall, prefill 4,344/4,495 (16K/32K cold), needles 6/6, stress 278 s.

### 3.2 `coding-agent` (C1 / TTFT-sensitive — one user, one agent, long sessions)

The use case where the first token of every fresh file matters and batch throughput does not:

| Knob | Value | Why |
|---|---|---|
| `MAX_RUNNING_REQUESTS` | 1–2 | a single agent streams one request at a time; fewer slots = less scheduler interference and the full graph list [1,2] |
| `CHUNKED_PREFILL_SIZE` | 8192 (keep) | first-turn TTFT is what chunk buys for long coding sessions; 16384 costs −4.4% C1 for +22% prefill — measure before taking |
| `ENGRAM_ADMISSION_PREFETCH` | 1 | −0.2…−0.4 s off every cold first-request TTFT (the chunk-0 case the agent hits every session) |
| `KERNEL_CACHE_DIR` | persistent host dir | no +14–19 s Triton-compile tax on the first request after a serve restart |
| `REASONING_EFFORT` | high (owner call, T3.54) | the coding-agent harness at budgets 50/75/100 decides the default; benchmarks argue for 75–100, wall-time argues for 50 |
| Everything else | qualified | placement, D=4, determinism levers are already C1-optimal |

### 3.3 `multi-user` (C4 — several concurrent sessions)

| Knob | Value | Why |
|---|---|---|
| `MAX_RUNNING_REQUESTS` | 4 (the qualified max at 262K) | aggregate tok/s is the metric; 4 is the qualified ceiling at 262K context |
| `HOST_THREAD_CPUS` | auto (non-negotiable) | the C4 106/123 bimodality is a *shape* defect; `auto` is the only qualified setting |
| `CONTEXT_LENGTH` | 64K × 4 | if sessions are shorter, the 64K rung trades context for headroom (C4 121.0/116.2) |
| `CUDA_GRAPH_BS` | "1 2 3 4" | bs=3 capture is +28% at C3 (E26) |
| Speculation | off | every spec path measured loses at C4 (−63% for n-gram, −69…−77% for DSpark K5) |

### 3.4 `long-context-ladder` (single giant context)

| Rung | Overrides | Measured |
|---|---|---|
| 262K × 1 | `CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1` | needles 6/6 at 255,388 tok, TTFT 59 s |
| 512K × 1 | + `MEM_FRACTION_STATIC=0.962` | needles 6/6 at 511,145 tok, TTFT 142 s; C1 60.2/58.2 |
| 1M × 1 | `CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962` | needles 6/6 at 1,022,456 tok, TTFT 448 s; C1 59.1/57.2 |

Guards encode all three: >262K with any other fraction is refused; >1,048,576 is refused outright (never run).

### 3.5 `power-capped`

The power findings: decode is cap-insensitive on this model (the step is memory-bound), prefill pins to the
cap; GLM-5.3 was the opposite (+9% C16 at 325 W). So the preset is: keep 300 W (or 325 W) for decode-heavy
workloads; the only reason to raise the cap is prefill-heavy traffic; the only reason to lower it is heat/cost
— measure the prefill curve first (power-sweep bench pattern in `docs/BENCHMARKS.md`). On this box 3×300 W +
CPU fits the 1,500 W PSU only with the CPU PPT capped (see `docs/HARDWARE.md` power table).

### 3.6 `quality-first`

| Knob | Value | Why |
|---|---|---|
| `REASONING_EFFORT` | max (100) | benchmark-grade reasoning budget for thinking traffic |
| `KT_MAX_DEFERRED` | 4 (never 5) | D=5 fails NLL; the gate *is* the quality floor |
| `KT_NUM_GPU_EXPERTS` | never 230 | 230 fails the NLL gate (E29) |
| Determinism levers | on | bit-reproducible greedy is itself a quality property (reproducible evaluations) |
| HumanEval spot-check | per checkpoint | 161/164 baseline, 158/164 on 1017 — the public quality number, re-run per model update |

---

## 4. How a new recipe builds its own table

1. Copy the registry (see [TEMPLATE/serve_config.py](TEMPLATE/serve_config.py)); declare the new engine's knobs
   in the same shape (type/default/sink/evidence), with owned prefixes so typos refuse.
2. Fill the trade-off table **only from stamped evidence** — each row cites an evidence directory, and the
   headline numbers come from the generated table, never typed.
3. One profile per use case, each restating the registry defaults; `check-profiles` in CI fails if a profile
   drifts from the registry (the registry is the single source of truth).
4. Encode every condemned combination as a guard with the measured message and its override. The guards are
   the ledger's teeth: a rejected setting cannot silently come back.

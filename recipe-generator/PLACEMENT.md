# PLACEMENT.md — the planner spec

This file is the spec for the recipe generator's placement engine: what goes in, the tiering rules, every
formula, the constant register, and the **calibration rule** that decides whether a projection may be trusted.
The working implementation is [`placement_calc.py`](placement_calc.py); this document is the contract it
implements, and the tests in [`tests/test-placement-calc.py`](tests/test-placement-calc.py) pin the contract to
the repo's measured numbers.

The engine generalizes the pattern from `scripts/feasibility-calc.py` + `evidence/budget-*.json` (the parent
repo's P0 calculator). What it adds: hot-expert maps, deferral, chunk↔expert workspace trade, Engram row-gather
budgets, per-use-case KV shapes, and **projections** (C1/C4, prefill, TTFT) with an enforced calibration
discipline. It is the "chef's pass": inventory in, tier plan + lever settings + projections out.

---

## 1. The pipeline (mise en place for models)

1. **Inventory.** Read the checkpoint's structure without downloading weights: `scripts/checkpoint-budget.py`
   reads safetensors headers only and classifies every tensor into components. Add the `arch` section (activation
   structure) to get a budget JSON. Hardware is a JSON of capacities plus *measured* bandwidths — the P1 host
   microbenchmark suite of `recipe-factory/METHODOLOGY.md` §4, never vendor specs.
2. **The pass.** `placement_calc.py` assigns each component a tier (VRAM / DRAM+CPU / NVMe) and emits the lever
   settings that realize it (expert split + hot map, KV, chunk, deferral D, Engram tiering + prefetch modes).
3. **The play.** Tuning mode: every lever in [LEVERS.md](LEVERS.md) is exposed with its measured trade-off, and
   the profiles in [PROFILES.md](PROFILES.md) are generated from the same tuning model.
4. **The tasting.** The calibration rule below. A projection without a measured anchor is labeled *provisional*
   and may rank experiments, never promise numbers.

## 2. Input schemas

### 2.1 Budget JSON (`checkpoint-budget.py` output + `arch`)

```json
{
  "source": "deepseek-ai/DeepSeek-V4.1-Flash", "revision": "dba1be0a…",
  "baseline_dtype": "fp4-packed",
  "header_total_gib": 475.241,
  "components": {
    "routed_experts":  { "total_gib": 268.945 },
    "engram_table":    { "total_gib": 188.833 },
    "attention":       { "total_gib": 4.798 },  "shared_expert": { "total_gib": 1.32 },
    "embedding_head":  { "total_gib": 2.466 },  "router":        { "total_gib": 0.147 },
    "norms":           { "total_gib": 0.001 },  "hyper_connections": { "total_gib": 0.146 },
    "engram_proj":     { "total_gib": 0.293 },  "dspark":        { "total_gib": 7.388 },
    "vision":          { "total_gib": 0.904 },  "other":         { "total_gib": 0.0 }
  },
  "arch": {
    "moe_layers": 40, "routed_experts_per_layer": 384, "experts_per_token": 6,
    "attention_heads": 64, "kv_bytes_per_token": 890,
    "engram_rows_per_token": 48, "engram_row_bytes": 264
  }
}
```

- `baseline_dtype` declares the dtype the byte counts refer to (`fp4-packed` for FP4-shipped checkpoints,
  `bf16` for most HuggingFace models). Quantization size factors in [`quantization.json`](quantization.json)
  each declare their own baseline; the calculator warns on a mismatch. This is the "1% error = 5 GiB wrong
  verdict" rule from `recipe-factory/METHODOLOGY.md` §3.1 made mechanical.
- An example budget may use `"extends": "../../../evidence/budget-upstream-bf16fp8.json"` to inherit components
  and add only `arch` — no copied bytes, no drift.
- Dense models: `moe_layers: 0` (or no routed experts) → the engine emits a plain full-VRAM plan.

### 2.2 Hardware JSON

```json
{
  "name": "2x RTX PRO 6000 (96 GiB) + 9955WX 16C + 246.8 GiB DDR5 + 2x NVMe",
  "gpus": 2, "vram_gib_per_gpu": 95.59, "util": 0.965, "gpu_reserve_gib": 4.0,
  "host_ram_gib": 246.8, "host_reserve_gib": 24.0,
  "dram_read_gbps": 131, "pcie_gbps_per_gpu": 57,
  "nvme_sequential_gbps": 12, "nvme_random_rows_per_s": 403000,
  "cpu_cores": 16, "ccds": 2,
  "gpu_ms_per_token": null,
  "chunk_workspace_model": "sglang-fused-moe",
  "engram_row_cache_gib": 4.0,
  "calibration": {
    "gpus": 2, "evidence": "evidence/t352/default-flip",
    "c1_tok_s": 60.8, "c4_tok_s": 124.0,
    "prefill_16k_cold_tok_s": 4344, "prefill_16k_warm_tok_s": 4504
  }
}
```

Rules:

- Every bandwidth is **measured on the host** (P1 microbenchmarks). Vendor specs never enter.
- `gpu_ms_per_token: null` + a calibration C1 → the engine *inverts* it (see §5). Otherwise provide a census
  value (P2/P3 kernel census, `recipe-factory/METHODOLOGY.md` §5.1) or accept the sibling-model estimate (5.0 ms)
  with the estimate label.
- `chunk_workspace_model: "sglang-fused-moe"` (default) enables the chunk↔expert workspace trade (§4.4, a
  this-host fitted line); `"none"` for engines without a chunk-coupled workspace (llama.cpp and friends).
- `calibration` is the anchor: the deployment whose measured numbers the projections must reproduce (§5).
  `k_overrides` (optional) carries k-factors from another example on the same host/engine — weaker, labeled.

## 3. Tiering rules (where each ingredient goes)

| Component | Tier decision | Why |
|---|---|---|
| attention, shared_expert, embedding_head, router, norms, hyper_connections, engram_proj | always VRAM | touched every token, every layer — hot, small |
| dspark, vision | VRAM by default, disable-able | optional resident; speculation is off by default (ledger: DSpark/n-gram REJ) |
| routed_experts | VRAM first, then DRAM+CPU | only k of E fire per token; non-resident experts cost bandwidth when used, not compute when idle (MoE makes placement winnable) |
| KV cache | VRAM | latency-critical per step |
| engram_table | DRAM if it fits, else NVMe + page cache, else NVMe + denser row cache | lookup tables: row gathers are page-cache friendly; NVMe sustains decode-rate gathers (measured E3); re-quantizing shrinks bytes but NOT the page-cache footprint (E40) — the fix is cache hygiene + a denser row cache (T3.48) |

Decisions encoded in `tier_plan()`:

```
usable_vram  = gpus × (vram_gib × util − gpu_reserve_gib)         # reserve covers CUDA/NCCL/workspaces/graphs
fixed        = Σ dense components (× quant dense_factor)
kv_gib       = kv_bytes_per_token × context × slots / 2^30
expert_capacity = usable_vram − fixed − kv_gib
gpu_fraction = clamp(expert_capacity / experts_gib, 0, 1)          # 0 for dense models
engram tier  = DRAM  if cpu_experts_gib + engram_gib + host_reserve ≤ host_ram
               NVMe  otherwise (row budget checked in §4.5)
```

## 4. The formulas

### 4.1 Expert split and the hot-expert map

```
cap_per_layer = round(gpu_fraction × routed_experts_per_layer)
gpu_per_layer = cap_per_layer − workspace_penalty(chunk)           # §4.4, zero when VRAM-rich (cap ≥ 240)
cpu_share     = (1 − gpu_experts/total) × (1 − map_quality)
cpu_bytes_per_token = cpu_share × experts_per_token × moe_layers × per_expert_bytes
```

`map_quality` — the fraction of would-be CPU routes a placement catches beyond its count share:

| map | quality | measured CPU share (219/384 resident) | source |
|---|---|---|---|
| `checkpoint_order` | 0.0 | 42–46 % | T3.8: a random split |
| `fitted_static` (el8-style) | 0.65 | 12–15 % (15 % modeled) | T3.8b: C1 +37 %, C4 +120 % |
| `runtime_oracle` | 0.93 | 1–3 % | T3.8c per-set oracle; the only way past the static floor (PARKED, GitHub #6) |

llama.cpp's expert-cache routing study independently confirms the same shape: static hot-expert pinning covers
only ~10 % of unseen traffic, temporal LRU hits 67–81 % ([PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861)).
A *static* map's floor is 10–15 % CPU share; runtime replacement is the only documented path below it.

### 4.2 Decode model

```
gpu_ms      = census value, or 1000 / calibration.C1 × (calib_gpus / gpus)   (inverted; §5)
cpu_bw_ms   = cpu_bytes_per_token / (dram_gbps × 1e9) × 1e3
dispatch_ms = moe_layers × 0.09                       # per-layer CPU round-trip, this-host constant
serial_cost = cpu_bw_ms + dispatch_ms
penalty(D)  = serial_cost × 0.42                      if D == 0     # D0_PENALTY_FRACTION (fitted)
              max(0, serial_cost − D × gpu_ms / 2)    otherwise
wall_ms(n)  = max(gpu_ms, cpu_bw_ms) + penalty(D)                  n == 1
              max(gpu_ms × √n, n × cpu_bw_ms + dispatch_ms / n)    n ≥ 2
C_n         = 1000 × n / wall_ms(n)
```

Why this shape: with deferral, the CPU tier hides behind the GPU step (wall = max(gpu, cpu_bw)); at D=0 the
CPU round-trips serialize into the critical path, but only ~42 % of the naive serial cost shows up in
measurements (per-layer pipelining). D is capped at 4: D=5 fails the NLL gate (measured). The batch scaling
`gpu_ms × √n` reproduces the measured C1→C4 ratio (60.8 → 124.0 = 2.04×).

### 4.3 Prefill / TTFT model

```
gpu_prefill_ms = gpu_ms × √chunk                                  # batched-GEMM scaling law
cpu_stream_ms  = cpu_expert_gib × 2^30 × touch / dram_gbps × 1e3
                 touch = min(1, chunk × experts_per_token × cpu_share × moe_layers / cpu_experts)
engram_ms      = chunk × rows_per_token × row_bytes / nvme_gbps × 1e3     (NVMe tier only)
prefill_tok_s  = chunk × 1000 / max(gpu_prefill_ms, cpu_stream_ms, engram_ms)
TTFT(tokens)   = tokens / prefill_tok_s + 0.1 s
```

The cold-Engram chunk-0 penalty (+1.7–3.1 s, measured T3.49) is reported separately; admission-time prefetch
removes 0.2–0.4 s of it. Cold/warm prefill ratio 1.037 (measured).

### 4.4 Prefill chunk ↔ expert workspace trade (this-host fitted line)

The fused-MoE prefill workspace scales with the chunk and eats expert slots (measured points: 226 experts @
chunk 4096, 219 @ 8192, 230 @ 2048, "16384 costs 14 experts"):

```
CHUNK_REF = {2048: 233, 4096: 226, 8192: 219, 16384: 205}
gpu_per_layer = cap − (CHUNK_REF[4096] − CHUNK_REF[chunk])      # 7 experts per 4096→8192 step
chunk chosen  = first of [8192, 4096, 16384, 2048] satisfying gpu_per_layer ≤ CHUNK_REF[chunk]
```

16384 is last in preference because measurement rejected it: +22 % prefill for −4.4 % C1 (E34d-iv). VRAM-rich
plans (cap ≥ 240/layer) skip the trade. Engines without the fused-MoE workspace set
`chunk_workspace_model: "none"`. **The line is a this-host fit — recalibrate on a new host** (it exists because
it encodes three measured datapoints, not because it transfers).

### 4.5 Engram row-gather budget (NVMe tier)

```
rows_per_s = max(C4 × rows_per_token, chunk × rows_per_token / prefill_wall_s)
fits       = rows_per_s ≤ 0.8 × nvme_random_rows_per_s            # ENGRAM_MARGIN
```

If it fails: the honest fallback is a user-space row cache (T3.48: 264 B rows, ~22× denser than the page
cache), not re-quantization — re-quantized tables cannot shrink the page-cache footprint (E40). DRAM tier is
never row-bound (measured 4.15M rows/s, E3).

## 5. The calibration rule (prime directive)

> **Every projection must be validated against at least one measured datapoint before it is trusted.**

Mechanics in the engine:

1. **Anchor.** `hardware.calibration` names a measured deployment (`evidence`, `gpus`, C1/C4/prefill). The
   engine must reproduce it: with `gpu_ms_per_token: null` it inverts `gpu_ms = 1000/C1` (valid because at D=4
   with the CPU hidden, wall = gpu_ms), and computes k-factors `k = measured / modeled` per metric.
2. **Trust labels.** Each plan carries exactly one:
   - `calibrated` — this model on this GPU count, anchored to a local measured datapoint;
   - `calibrated-model (cross-hardware)` — anchored datapoint exists but GPU count differs (scaling is modeled);
   - `calibrated-model (k carried from another example)` — k-factors borrowed from another model on the same
     host/engine (`k_overrides`), the gpu_ms itself unvalidated;
   - `provisional` — no anchor: output says UNVALIDATED, for ranking experiments only.
3. **Self-application.** The calculator's own tests pin it to the repo's measured numbers (example 1 must
   reproduce 219/165 experts, C1 60.8, C4 124.0, prefill 4,344 — the qualified 1017 deployment). A change that
   breaks that reproduction fails CI. This is the poteto "verification contract" applied to ourselves: the
   calculator is a lever that was itself executed and validated against reality before being trusted.
4. **Model error is documented, not hidden.** Known gaps, each visible in a k-factor or a constant:
   - k_prefill ≈ 0.79 on the SGLang hybrid (the √chunk law overestimates prefill batching);
   - k_prefill ≈ 0.35/0.21 on llama.cpp (same law, different engine);
   - the D=0 serial penalty is a fitted fraction (0.42), not first-principles;
   - the chunk↔expert line and `DISPATCH_MS_PER_LAYER` are this-host constants.
   k-factors are host/engine properties — they carry across models on the same box (example 2 borrows them) —
   but the anchor for a *new* model is always its own first serve.

## 6. Constant register (every constant and where it came from)

| Constant | Value | Source |
|---|---|---|
| `DEFAULT_GPU_UTIL` | 0.95 (feasibility-calc default; the measured end on the reference host is 0.965 — fraction 0.96 + E28's 779 MiB-free fit) | feasibility-calc.py; E28 |
| `DEFAULT_GPU_RESERVE_GIB` | 4.0 | feasibility-calc.py (vLLM #56700 / 0xSero land 4–6 GiB) |
| `DEFAULT_HOST_RESERVE_GIB` | 24.0 | feasibility-calc.py |
| `DEFAULT_KV_BYTES_PER_TOKEN` | 890 (CSA2 KV, DSV4.1) | vLLM recipe + LibertAI card |
| `DISPATCH_MS_PER_LAYER` | 0.09 | derived from measured D=0→D=4 wall delta at 40 layers |
| `D0_PENALTY_FRACTION` | 0.42 | fitted to measured D=0 C1 49.8 vs D=4 60.8 |
| `D_MAX_QUALIFIED` | 4 | T3.15: D=5 fails NLL (+0.00287) |
| `COLD_ENGRAM_TAX_S` | 2.4 (range 1.7–3.1) | T3.49 |
| `GPU_BATCH_EXPONENT` | 0.5 | reproduces measured C4/C1 = 2.04 |
| `WARM_COLD_PREFILL_RATIO` | 1.037 (4,504/4,344) | T3.19d discipline |
| `MAP_QUALITY` | 0.0 / 0.65 / 0.93 | T3.8 / T3.8b / T3.8c |
| `CHUNK_REF`, `CHUNK_PREFERENCE` | §4.4 | E28/E29/T3.28/E34d-iv |
| `ENGRAM_MARGIN` | 0.8 | E3 measured 403K rows/s; keep below 80 % |

All are calibration values, never universal claims — the methodology's prime directive is that every number is
re-measured per host; this register exists so a new host knows *which* numbers to re-measure and *which
decision each feeds*.

## 7. Relationship to `scripts/feasibility-calc.py` and what is NOT in scope

The parent calculator decides **fits / does not fit** per layout × GPU count (CI pins the README verdict to
it). This engine is the next layer: given *that* a hybrid layout fits, it decides **the plan** (exact split,
map, levers) and **projects the numbers**. Exit codes stay compatible: 0 plan, 2 input error, 3 does not fit.

Deliberately out of scope (owned by `recipe-factory/METHODOLOGY.md` phases): engine choice and the kernel
census (P2/P3), the gate suite (NLL identity, needles, stress, batch consistency, health, power), the evidence
manifest discipline, and the benchmark harness contract. The generator *references* those; it does not replace
them. A plan from this engine is input to P2, never a qualification.

## 8. Engine selection (the fold-in, phase A of ROADMAP.md)

The placement math is engine-agnostic (bytes, bandwidths, rooflines); engines differ in **which levers
exist** and **how a recipe expresses them**. The classifier maps plan-shape → engine; the adapters map
lever-settings → engine flags. Profiles live in [`engines.json`](engines.json) (verified against
[research/engines.md](research/engines.md), fresh 2026-10-23); `placement_calc.py --engine <name>` feeds the
profile into the plan (chunk-workspace model, deferral/Engram availability, verdict warning), and
`emit_recipe.py --engine <name>` selects the adapter.

### 8.1 The fold-in map (plan shape → engine)

| Placement shape | Primary engine | Secondary | Notes |
|---|---|---|---|
| CPU-hybrid MoE (VRAM + DRAM + NVMe) | sglang-hybrid | vLLM (weight-offload style) | the reference recipe's shape; SGLang Paged Experts auto-K is our calibration point |
| GPU-resident (single/paired cards) | vLLM | llama.cpp | vLLM's recipes schema is the publication format |
| Quant-shrunk to fit | llama.cpp | TensorFold (gated) | GGUF today; EXL3 only after the reopen trigger |
| Multi-GPU EP (datacenter V4.x scale) | DeepInfer (watch) | SGLang/vLLM EPLB | DeepInfer is offline-only + SM100/Ascend today; borrow its gates now |
| Fleet recipe governance | none of the above | plain orchestration | FLARE is REJ/CLOSED: federated-learning machinery is dead weight for serving |

### 8.2 The rules

1. **The ledger wins.** An engine the tried-options ledger REJ-ed (TensorFold as engine, T3.32) may only be
   recommended when its reopen trigger is cited in the same breath — the calc prints the verdict + note
   whenever such a profile is selected. Research disagreements are recorded, not silently resolved
   (TensorFold: research says ADOPT for the quant-shrunk tier; the ledger says REJ on this hardware until
   gate #827 or a published SM120 EXL3 MoE kernel with a KL number — the profile carries both).
2. **Lever availability is a plan input, not an afterthought.** Engines without deferral force D=0 (with a
   guard note); engines without Engram support get the note; the chunk-workspace model is per-engine.
   A plan that leans on a lever its engine does not have is a refused plan, not a hopeful one.
3. **Hardware gates are first-class.** DeepInfer is not runnable on this SM120 box (SM100/Ascend 950 only,
   offline-only, no server layer yet) — ADOPT-as-reference, not ADOPT-as-recipe-target, until its reopen
   trigger.
4. **One adapter per phase, one calibration anchor per adapter.** Never an abstraction layer over five
   engines nobody has served; the phased rollout with entry/exit criteria is [ROADMAP.md](ROADMAP.md).
   Each engine's calibration anchor is named in its profile (`engines.json`); an engine with no anchor
   emits `trust: provisional` until its first measured serve.

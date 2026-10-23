# Recipe Factory — METHODOLOGY (the reusable playbook)

This is the playbook extracted from taking DeepSeek-V4.1-Flash (475 GiB, 384 routed experts/layer, 196B-parameter
Engram lookup tables) to a qualified three-tier serve on 2× RTX PRO 6000. Every rule here earned its place in a
measured experiment (cited by id, e.g. "E13", "T3.8b"); the ledger in [TRIED-OPTIONS-LEDGER.md](TRIED-OPTIONS-LEDGER.md)
holds the numbers. Read this file once per new recipe, then follow the phase exit criteria.

**Prime directive:** every number is re-measured on the host it will serve on. Numbers from other hosts, other
power conditions, or vendor spec sheets are cited as references, never mixed into tables of local results.

---

## 1. The one-paragraph map (write this first for any new recipe)

A serving recipe is memory-hierarchy engineering plus kernel engineering, nothing else:

- We want a **C GiB model on V GiB of VRAM**. If C ≤ V, the problem is kernel/flag selection on a known engine.
  If C > V, the problem is *placement*: which tensors are hot (touched per token) and which are cold (streamed,
  paged, or looked up rarely).
- MoE models make placement winnable: only k of E experts fire per token, so non-resident experts cost
  *bandwidth when used*, not compute when idle. Dense models do not get this — offload there means streaming
  everything.
- Three tiers in descending bandwidth: VRAM (≈1.8 TB/s) → host DRAM (≈125–400 GB/s, CCD-limited on desktop
  CPUs) → NVMe (≈14 GB/s sequential, but 300–500K random rows/s). The recipe's whole shape is a tiering decision.
- The GPU's 4-bit story decides the kernel thread: consumer Blackwell (SM120) runs MXFP4 natively through
  CUTLASS grouped GEMM; data-center Blackwell (SM100) has tcgen05/TMEM; engines' *fallback* paths (Marlin W4A16
  dequant) are 40–50% slower and change speculative acceptance. Know which kernel actually dispatches (census,
  not docs).

This project's version: [../docs/LEARNING.md](../docs/LEARNING.md) §0.

## 2. Phase exit criteria at a glance

| Phase | Work | Exit criterion (verifiable from the repo) |
|---|---|---|
| P0 | research + feasibility | placement table: fits/does-not-fit per layout × GPU count, with margins; ranked path list |
| P1 | host microbenchmarks | hardware doc has zero "estimate" rows; calculator recomputed on measured numbers |
| P2 | baseline | baseline tables produced by *this* harness (same fixture, same power condition) |
| P3 | build | decode/prefill/needle table with a pinned engine build, or a documented reason no route ran |
| P4 | scale | one table with all hardware rows, same fixture and power condition |

---

## 3. Feasibility math (P0): checkpoint-budget → placement calculator

### 3.1 The budget comes from headers, not downloads

`scripts/checkpoint-budget.py` (reference implementation in the parent repo) reads **safetensors headers only**
(8-byte length + JSON header, over HTTP range requests or local files) and classifies every tensor into
components: routed experts, Engram/embedding tables, draft/spec heads, attention, embeddings, shared expert,
vision encoder. Output: a `budget.json` with exact per-component byte totals.

Why headers first: the whole P0 verdict for V4.1-Flash (2 GPUs cannot hold it; shortfall 112 GiB = 42% of
experts) came from headers before any weight was downloaded. You cannot reason about offload without knowing
which tensors are hot. Average expert size × activations/token × layers = the DRAM-bytes-per-token figure that
drives every decode ceiling.

Rules:

- Count **exact bytes** (GiB = 2^30) per component; a 1% error in a 475 GiB model is a 5 GiB wrong verdict.
- Note format subtleties: "INT8-packed FP4 pairs" (two E2M1 nibbles per byte) vs NVFP4 (block 16, UE4M3, FP32
  per-tensor scale) vs MXFP4 (block 32, UE8M0). A re-quantization to NVFP4 can be *larger* (measured: 15.8 GiB
  larger for V4.1-Flash) — size and kernel compatibility are separate questions.
- Checkpoint-size claims in source documents are frequently wrong (the source Google Doc's "~151 GB backbone"
  was DeepSeek-V4-Flash, not V4.1-Flash). Audit every external claim against the budget JSON, in a table
  (pattern: [../docs/FEASIBILITY.md](../docs/FEASIBILITY.md) §6).

### 3.2 The placement calculator pattern

`scripts/feasibility-calc.py` is the reference implementation. The generalized shape:

```
usable_vram   = physical_vram × utilisation − reserve_per_gpu      (reserve: 4 GiB/GPU optimistic, covers
                                                                    CUDA ctx, NCCL, workspaces, activations)
expert_ceiling= usable_vram − non_expert_resident_weights
cpu_expert_bytes_per_token = k × (1 − resident_fraction) × avg_expert_bytes × layers
decode_ceiling = 1000 / (gpu_ms_per_token + cpu_expert_bytes_per_token / dram_gbps_read)
prefill_ceiling= (cpu_expert_bytes_per_chunk / pcie_gbps)  [streaming path]
```

Inputs that must be **measured, not estimated**: `dram_gbps_read` (P1), `pcie_gbps` (P1), `gpu_ms_per_token`
(P2/P3 census). The calculator supports layouts (all-GPU, Engram-host, hybrid-cpu, …) × GPU counts (2/3/4…)
and Engram tiers (nvme+row-cache, dram, quantized), and **exits 3 when a layout does not fit** so CI can pin
the verdict (`scripts/ci.sh` in the parent repo does exactly this).

Three rules that made the calculator trustworthy:

1. **Calibrate on a known deployment.** The calculator must reproduce a proven external result before its new
   rows are believed (0xSero's 4× RTX PRO 6000 deployment: FITS with 60 GiB VRAM margin / 40 GiB host margin;
   Engram-in-host-DRAM on 128 GB correctly returns DOES NOT FIT).
2. **Every ceiling column is a roofline with named assumptions.** The "split" column (CPU compute + GPU DMA in
   parallel) assumed the two DRAM readers are additive *in time*; they were additive in *bandwidth* only
   (E4b/E30: +2% C1, not +56%) — the assumption is stated next to the column so a falsified assumption is a
   struck-through row, not a silent wrong number.
3. **Parallelism validity is a first-class check.** 64 attention heads and 384 experts do not divide by 3:
   TP3 was invalid on this model, so the 3-GPU row became EP3/PP. Head-count and expert-count divisibility are
   calculator inputs, not afterthoughts.

### 3.3 The paths ranking

P0 ends with paths ranked by expected payoff *per unit of remaining software risk*, each with a status
(proven / build / blocked-on-hardware / research-only). This project's table (paths A/B/C/D/E, hybrid primary)
is in [../plan/README.md](../plan/README.md). The ranking rule: prefer the path that reuses a qualified engine
first (port), alternate engine second (llama.cpp branch), own prototype last — and only as the documented
fallback.

---

## 4. Host microbenchmark suite (P1): each number decides something

No model weights, no GPU contention (except the two GPU probes, which need a window). Every measurement
replaces a vendor spec in the hardware doc; the exit criterion is literally "no estimate rows left".

| Probe | What it measures | What the number decides |
|---|---|---|
| **DRAM read bandwidth** (E1/E1b; torch read at 8/16/32 threads, + a C/STREAM revalidation) | The CPU's *sustained read* ceiling, not the vendor's 409.6 GB/s. Measured 124.7 GB/s on a 16-core 2-CCD part = 30% of theoretical; per-CCD ≈62–63 GB/s (E25) | The CPU-offload decode ceiling: `tok/s ≈ 1000/(gpu_ms + cpu_bytes÷gbps)`. The CCD fabric is the wall — more DIMM speed won't help; more CCDs will |
| **CPU GEMV throughput** (E6 s0; bf16, 16 cores) | Sustained bandwidth *while computing* (94 GB/s vs 125 read-only) | Whether the CPU tier is bandwidth-bound (it was: 76 ms/token at 94 GB/s) — and the honest denominator for ceilings |
| **GDDR7 copy bandwidth** (E2) | ≈1,470 GB/s/card (82% of 1,792 spec) | The GPU roofline every kernel claim is judged against; "kernel at roofline" = no speed left there |
| **NVMe random row gather** (E3; 264 B rows, 48/token, DRAM vs both drives) | 4.15M rows/s DRAM, 403K Samsung 9100 PRO / 347K T705 | Whether lookup tables (Engram) can live on NVMe behind a page cache at decode rate — the tiering decision (they can) |
| **PCIe H2D + CPU/DMA additivity** (E4/E4b) | 53 GiB/s per card; CPU reads + 2-GPU DMA = 234 GB/s, additivity 0.98 | Whether splitting DRAM-resident experts between CPU compute and GPU streaming doubles the ceiling (bandwidth yes, time no — E30 killed B2 at +2%) |
| **GPU kernel census** (E12, needs one GPU window) | Which kernel each engine *actually* dispatches on this arch, % of step per kernel, ms/token (3.8–5.1 on SM120) | The `gpu_ms_per_token` constant for the calculator; whether the fallback (Marlin) is silently eating 40–50% |

The single most valuable artifact: one log-scale bar chart of VRAM / DRAM / NVMe bandwidths. That picture *is*
the tiering design (LEARNING.md §3 exercise).

---

## 5. Engine and kernel research checklist (per GPU architecture)

### 5.1 Which kernel actually runs? (census, never docs)

For each candidate engine, run the model on this arch and record the kernel names from the profiler, the share
of the decode step per kernel group, and ms/token. On SM120 the census found: experts 24.5%, dense FP8 27%,
attention 11%, draft 10% — the "expert GEMM" is never the whole story. Doc claims like "native FP4 support"
must be verified; engines fall back silently (vLLM's MXFP4→Marlin W4A16 fallback on SM120 was still an open
bug, vllm-project/vllm#30135).

### 5.2 Architecture cheat sheet (what changes what you can do)

| Arch | FP4 tensor-core story | Consequences |
|---|---|---|
| **SM90 (H100/H200)** | FP8 native; no FP4 MMA | FP4 models run via Marlin/dequant or FP8; FP8 is the native precision |
| **SM100 (B200, GB200/300)** | NVFP4 native via tcgen05 + TMEM | Data-center kernels (DeepGEMM, cuBLASLt block-scaled) work; MXFP4 needs re-quantization to NVFP4 or CUTLASS MX paths |
| **SM120 (RTX Blackwell / RTX PRO 6000)** | MXFP4 native via CUTLASS grouped GEMM (`kind::mxf4nvf4` MMA, block-32 UE8M0); **no TMEM, no tcgen05, 99 KiB smem** | Data-center kernels rejected at startup or silently fall back; the working path is a CUTLASS example-79d *type swap* (E13 measured 1,397 GB/s at decode M=1 = 93% of copy ceiling). Marlin W4A16 dequant costs 40–50% and shifts speculative acceptance |
| **Apple Silicon (M-series)** | Metal packed MXFP4 gather-QMM primitives; unified memory | Different tiering math entirely (no PCIe; NVMe streaming + wired DRAM); see the external-recipe evaluation doc for the M5-Max comparison |

### 5.3 Format trade-offs (from this project's measurements)

| Format | Bytes | Kernel reality (SM120) | Verdict from this project |
|---|---|---|---|
| **MXFP4** (E2M1 + UE8M0 block-32) | the upstream V4.1 checkpoint itself (268.9 GiB experts) | native CUTLASS SM120 grouped GEMM, roofline at decode | **adopted** — it is what the model was trained on; no re-quant step can beat "the checkpoint as shipped" |
| **NVFP4** (block-16 UE4M3 + FP32 scale) | 15.8 GiB *larger* for this model | cuBLAS/CUTLASS W4A4 paths, better engine support | useful only for kernel compatibility; never for size |
| **Marlin W4A16 dequant** | — | the silent fallback on SM120 | a 40–50% tax to detect and eliminate (E12) |
| **EXL3 2.9–3.0 bpw** (trellis) | smaller than FP4 | bespoke forks only (GLM-5.3 recipe needed a custom tpurtell fork); no SM120 MoE kernel in vLLM/SGLang | rejected for this model: sub-4-bit on already-FP4-trained experts, unknown quality, months of kernel work (FEASIBILITY §Option C) |
| **GGUF Q2_K/Q3_K_M/Q4_K_M** | 246–323 GiB for V4.1 | llama.cpp CPU+GPU offload, `-ot` regex expert offload; the `deepseek41` branch blocked on upstream CSA2 (ggml-org/llama.cpp#28696) | kept as the independent second engine; one build the day CSA2 lands (T3.7) |

### 5.4 The kernel research loop (this project's K-track)

1. Census the deployed path (E12) — what runs, what it costs, what it *should* cost.
2. Microbenchmark the native format at the model's shapes (E13: build CUTLASS 79d as a type swap, verify vs a
   bf16 reference, measure decode/prefill shapes at the power condition).
3. Tune what is *not* at roofline (E20: the dense FP8 GEMMs were 57% of the step at 3% occupancy; seven
   per-shape configs → 2.26× C1). Skip what is (T3.21 closed on the stop clause: dense GEMM = 4% of TTFT, so
   no prefill kernel can reach the +20% bar).
4. Park decode-shaped kernel ideas when the served path is already at the copy ceiling (K10: the only
   remaining idea, parked at roofline).

---

## 6. The gate discipline (P3: what "qualified" means, mechanically)

A serve shape is **qualified** only when every gate below passes and the stamped evidence is committed. The
gates are ordered cheap→expensive; a config that fails an early gate never runs the later ones.

| Gate | Tool (reference) | Criterion |
|---|---|---|
| **NLL identity** | `scripts/nll-score.py` (teacher-forced corpus, 65,653 tokens) | any approximation (deferral, kernel change, tiering) must be bit-identical, or ΔNLL ≤ +0.002 with \|z\| ≤ 2 vs a same-model baseline. Baseline is per-model; a config A/B compares within-model only |
| **Functionality + needles** | `scripts/verify.py` | alias + context ceiling reported; arithmetic; thinking toggle; tool call; **N-needle exact retrieval at ≥95% of the served context** with `finish_reason=stop` (6/6 at 255K/511K/1,022K on the ladder) |
| **Stress / fit** | `scripts/stress-kv-full.py` | fill *every* running slot with a long prompt and decode them together; report peak VRAM. A single 32K prefill at C1 is not the worst case a config admits |
| **Batch consistency** | `scripts/verify-batch-consistency.py --same-prompt` | N identical greedy prompts in one batch, every row must agree (min pairwise prefix ≥ gate; per-slot vs a bs=1 reference; gibberish ratio). This gate caught the bug-C bs=4 graph defect |
| **Determinism** | T3.15c levers + `greedy-logprobs.py` | temperature-0 output bit-identical run to run (the fused-MoE finalize order and the indexer top-k order were the two drift sources; both fixed at zero cost) |
| **Health** | serve log scan | 0 NaN, 0 Xid, 0 tracebacks, 0 OOM across the whole battery; per-request NaN rejection must contain a fault to one 500, not the server (KT_REJECT_NAN_ROWS) |
| **Power** | `nvidia-smi` sampling + `summarize-power.py` | busy-median power and temps recorded with every number; power condition named in every table (e.g. Max-Q 300 W / 325 W cap) |
| **Speculation consistency** | `scripts/spec-consistency.py` | for *any* speculative path: decode-vs-teacher-forced agreement (argmax flips > 0.5 nats, max \|Δlogp\|, accept length). NLL cannot see draft-rejection state corruption — this probe found one emitted token at −20.5 nats (E43) |
| **Context ladder** | rungs 64K×4 → 262K×4 → 512K×1 → 1M×1 | stop at first failure; each rung runs the full set. (The ladder also exposed the indexer-logits OOM that was *not* KV — E33/E34) |
| **Quality spot-check** | `bench-humaneval.py`, `bench-reasoning.py` | HumanEval pass@1 greedy (161/164 baseline; 158/164 on the 1017 checkpoint) as the public quality number; note ±2 run noise from greedy near-ties |

Two gate lessons worth generalizing:

- **Approximation gates must be teacher-forced.** A config that changed the model path (D=5 deferral) passed
  everything except NLL identity — it failed at ΔNLL +0.00287 with greedy divergence at 9–39 chars. The NLL
  gate is the cheapest gate that catches silent quality loss.
- **Concurrency claims need distinct prompts.** Same-prompt concurrency overstated MoE expert batching 2–3×
  (TensorFold issue #479 confirmed it independently). Every C2+ figure uses `--distinct-prompts`;
  `--same-prompt` exists only for the consistency gate.

---

## 7. Evidence discipline

1. **Per-serve manifest** (`scripts/evidence_manifest.py`): recipe commit SHA, engine overlay SHA, image
   digest + container id, the **resolved knob line** (profile name + every env override), BIOS slot +
   HSMP-measured clocks, GPU power caps, pre-serve memory and Engram/page-cache residency. The launcher writes
   one per serve; nothing is "measured" without one.
2. **Stamped JSON**: every evidence script stamps its output with `evidence.manifest_hash` and copies the
   manifest beside it as `manifest.json`. `ci.sh` refuses a stamped file whose sibling manifest does not
   match — a number can never outlive the serve that produced it.
3. **Evidence directories** (`evidence/<id>/`): a README that states the question, the profile, the result
   table, and the verdict (PASS/REJ/OPEN), with the stamped JSONs beside it. Skim the reference dirs
   (`evidence/t1017/`, `evidence/t38/`, `evidence/t320-engram-prefetch/`) before writing the first one.
4. **Generated headline tables**: the headline table at the top of the benchmarks doc is generated from
   stamped JSON only (`scripts/benchmarks_headline.py markdown|write|check`); CI fails when the committed block
   is stale. No performance claim is typed by hand.
5. **Provenance for artifacts**: every third-party artifact (model snapshot, image, overlay) is pinned by
   immutable revision/digest in a PROVENANCE doc, and the licenses are recorded before the first serve.

---

## 8. Benchmark harness spec (the C1/C4, prefill, TTFT, warmth contract)

The reference implementations are `scripts/bench-decode.py` and `scripts/bench-prefill.py`. A new recipe's
harness must honor this contract so numbers compare across recipes:

**Decode (C1/C4):**
- Fixed code-review fixtures; temperature 0.2, fixed seed, thinking off; forced output length
  (`min_tokens = max_tokens`, `ignore_eos`).
- Headline: 512 tokens, 1 warmup, 6 runs at C1, 3 at C4. Runs/warmups/tokens are read from the stamped JSON,
  never typed into the caption.
- **C1** = single stream; reported as median per-sequence and **wall** tok/s (total completion tokens ÷ batch
  wall time including TTFT) — wall is the honest number when the server queues.
- **C2+** measured with `--distinct-prompts` (see §6). `--same-prompt` only for the consistency gate, and
  labelled wherever its tok/s appears.
- Speculative acceptance = Prometheus counter delta per run, reported beside the number.
- Power sampled alongside (`nvidia-smi -lms 500`), summarized as busy-median.

**Prefill:**
- Unique seeded prompts per run (prefix/radix cache must not hit), one output token, 3 runs.
- Effective prefill = server `prompt_tokens` ÷ client TTFT; TTFT includes tokenization + one-token handoff.
- **Cold / warm pair**: the warm pass repeats every size with fresh seeds (seed+500) and the Engram/page-cache
  residency is recorded (`fincore`) before each pass. A comparator without recorded cache state drifted
  1,471–1,504 vs 1,820 tok/s at 16K on unrecorded state (T3.19d).

**TTFT:** reported per prefill size; the ladder adds needle-TTFT at 255K/511K/1,022K tokens.

**Warmth:** every serve measures cold first (page cache dropped / fresh checkpoint), then warm. The cold-vs-warm
gap is itself a finding (cold Engram costs +1.7–3.1 s per request here).

---

## 9. What this project's numbers say about the process (calibration expectations)

A realistic expectation for a frontier-MoE-on-undersized-hardware build, from this project's own trajectory
(2026-09-22 → 2026-10-11, same hardware): C1 14 → 58–60.8 tok/s, C4 — → 118.9 wall, 16K prefill 216 → 4,344
warm, qualified context 32K×4 → 262K×4 + 512K/1M×1. The levers, in the order they paid: D=4 expert deferral
(+22% C1), BIOS FCLK 2100 (+3.1% C1), 226-expert placement (+6.3%), indexer tiling + chunk 8192 (≈2× prefill),
Engram ahead-prefetch (+16–26% prefill), the per-layer hot-expert map (+37% C1 / +120% C4), streamed prefill
arenas (+34–41% prefill), host-thread pinning (kills a C4 bimodality). Every one of those was a small,
gated, revertible change — the ledger in [TRIED-OPTIONS-LEDGER.md](TRIED-OPTIONS-LEDGER.md) records the ones
that did not pay, because in this discipline a rejected experiment costs as much rigor as an adopted one.

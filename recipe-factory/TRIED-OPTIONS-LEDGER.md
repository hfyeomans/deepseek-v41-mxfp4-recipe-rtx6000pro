# Recipe Factory — TRIED-OPTIONS LEDGER

Every option this project tried, with its measured outcome and verdict, formatted so a **new recipe consults
this instead of re-measuring**. A rejected option here is a recorded dead end, not an open question — the
registry's guards even refuse to relaunch several of them (D≥5, 230 experts, chunk 8192 beside 226 experts).

Verdicts: **ADOPT** = became a default · **REJ** = measured and rejected (do not retry on this hardware without
a listed reopen trigger) · **CLOSED** = closed on gates/stop clause · **PARKED** = open, superseded or gated.

Evidence ids (E1–E46, T3.x) cite the parent repo's `evidence/` tree and `plan/ROADMAP.md` readout, which carry
the stamped JSONs.

---

## 1. Engine and toolchain options

| Option | What it was | Outcome (measured) | Verdict |
|---|---|---|---|
| **SGLang `dev-dsv41` + kt-kernel hybrid** (sglang-kt) | upstream SGLang serving `deepseek_v41` + kt-kernel computing the CPU-resident experts | the serving stack: C1 14 → 58 tok/s over the project, 262K×4 default | **ADOPT** (E6 route 1; dev-dsv41 is deleted upstream — pin `da64c5cb` or rebase) |
| vLLM (NVFP4 baseline + generic `--cpu-offload-gb`) | the V4-Flash baseline engine (E8: 171–239 tok/s C1) and the candidate V4.1 engine | parameter-level streaming with no routing awareness — offloads *everything*, so it cannot run the 219/165 split; SM120 MXFP4 Marlin fallback still an open bug upstream (#30135) | **REJ** as the V4.1 hybrid engine; kept as the baseline harness target and flag reference |
| llama.cpp `deepseek41` branch + `-ot` expert offload | the alternate 2-GPU route (Q4_K_M GGUF, regex expert offload) | loader/Engram/hyper-connections verified in the branch; CSA2 sparse attention pending (ggml-org/llama.cpp#28696 unmerged) | **PARKED** (T3.7/GitHub #1): one build the day CSA2 lands; lower GPU efficiency than SGLang expected |
| **TensorFold as engine** | EXL3 "recipe engine" with native CUDA registry | native registry is Nemotron-only; no CPU/DRAM expert tier; DeepSeek forks are 2.9–3.0 bpw EXL3; same-model TP4 builds stay 12–24% behind SGLang (issue #479) | **REJ** as engine; **methods borrowed** (bit-exact shape-autotune → T3.27; CED bounded replay → E35; ahead-of-use Engram I/O → T3.20b) |
| Own PCIe expert-streaming prototype | stream DRAM-resident experts over PCIe each step (Option B2) | built bit-equal and graph-replayed; **+2% C1 at 40 streamed experts/layer, −3% at 60** (0.21 ms/expert vs kt's 0.146; D=4 already hides most of the CPU path) | **REJ** (E30); code removed (T3.16j); the knob now refuses with the removal message |
| 4-GPU SGLang profile (E9) | TP4 + Engram-in-DRAM profile written per 0xSero's proven pattern | calculator: 100% experts resident, ~200 tok/s C1 / ~700 C8 / ~7K prefill; needs 2 more cards + a 2,000 W PSU (or 225 W/card) | **PARKED** on hardware + power decision |
| Third GPU (TP2 + asymmetric EP) | third card holding ~90–120 complete experts/layer (326/58 or 308/76) | the only large decode lever left: projected C1 42–60 tok/s; TP3 invalid (64 heads) → EP3/PP | **PARKED** on card purchase (GitHub #5; design kept in `docs/research/T322-DESIGN.md` §4) |
| Mac sidecar (Option M) | 128 GB Mac as remote Metal expert worker over TB5 | needs p99 RTT ≤0.15–0.20 ms for 40 dependent round trips/token; 176 experts/layer don't fit 128 GB | **REJ** for C1; ranked async services > draft model > prefill experts if a TB5 Linux controller ever exists |

## 2. Speculative decoding

| Option | Outcome (measured) | Verdict |
|---|---|---|
| DSpark/MTP K=1..5 | K=2 parity at C1, loses at C3/C4; K=5 **−16% C1, −69…−77% C3/C4** (every verify step routes bs×(K+1) rows of CPU experts on a bandwidth-bound tier; GPUs idled at 101–148 W). Three graph bugs fixed en route (A/B/C; bug C = kt temp-buffer lifetime) | **REJ — `DSPARK=0`** (E23). Reopen only after a CPU-tier change (third GPU / more CCDs) |
| n-gram (stock) | first request crashes both ranks — DSV4 pool has no `move_kv_cache` | **REJ** (T3.9h), never retried |
| n-gram done properly (chain-only verify, C128 clear) | +60/+76% C1 on copy-code/copy-table, **−26% review, −63% C4**; and a decode-state defect under rejected drafts (one emitted token at −20.5 nats, 8 flips >0.5 nats — invisible to NLL/needles) | **REJ — `NGRAM=0`, not recommended** (T3.9h-b/E43). Fix sketch kept; reopen only for a single-user copy-heavy deployment, and any speculative path must pass `spec-consistency.py` |
| Drafter model | worse than MTP on every axis on this tier | **REJ** — explicitly not scheduled |

## 3. Expert placement and map options

| Option | Outcome (measured) | Verdict |
|---|---|---|
| Positional 0–218 (checkpoint order) | a random split: **42–46% CPU route share** on every measured set (per layer 21–59%) | **REJ** as default (kept as `KT_EXPERT_MAP=""`) |
| 226 / 158 experts | C1 +6.3% wall, C4 +9.2% over 208; NLL neutral; 779 MiB free worst case | **ADOPT** until T3.28 |
| 230 / 154 experts | +2.5% C1 wall **for −40% prefill** (chunk 2048 + fraction 0.975); NLL +0.0088 **fails the gate**; logprob OOM killed the server | **REJ** (E29). Side effect adopted: `SGLANG_LOGPROB_CHUNK_SIZE=512` |
| **219 / 165 + chunk 8192** | prefill +38/+47% at 16K/32K; C1 wall −1.9%; pooled NLL +0.00075 ± 0.00098 | **ADOPT** (T3.28/E34d-ii) — the current default |
| **el8 mix hot-expert map** (`KT_EXPERT_MAP`) | per-layer permutation at load (routing-exact, zero VRAM): CPU share 42 → 15%; **C1 42.4 → 58.0 (+37%), C4 +120%**; NLL −0.00033 ± 0.00101; prefill unchanged; soak 9.6–10.7% CPU share on reasoning traffic | **ADOPT** (T3.8b, owner flip 2026-10-10). Transfers unchanged to the 1017 weights (14.4% re-measured) |
| Own-counts refit (T3.8c) | own maps overfit (held-out 37.5% on 6K tokens); leave-one-family-out loses to el8 at every blend (13.3 vs 16.4% decode CPU share); hot sets are content-specific → **static-map floor 10–15%** | **CLOSED — no refit**. Retry bar: ≥3 new prompt families × ≥20K decode tokens |
| Runtime expert placement (T3.8d) | the only way past the floor (per-set oracle 1–3% CPU share); blocked: SGLang kt path positional; kt dynamic update a no-op for MXFP4; no resident-weight pointer | **PARKED** (GitHub #6). Go bar: held-out CPU share <8%, gates unchanged. Vend ktransformers PR #2093 mask tracker; watch SGLang Paged Experts (PR #42557) and vLLM #48505 for MXFP4 formats |
| `frequency` placement from activation stats | kt's own strategy needs an explicit id list (compact view is a prefix split) | **REJ** for this model (E17 open items); the el8 map replaced it |

## 4. Deferral and CPU-tier scheduling

| Option | Outcome (measured) | Verdict |
|---|---|---|
| **D=4 deferral** | +22% C1 over D=0; NLL +0.00066 ± 0.00103 (inside the gate) | **ADOPT** (`KT_MAX_DEFERRED=4`) |
| D=5 | **NLL +0.00287 fails the gate**; greedy diverges at 9–39 chars | **REJ** (T3.15); guard refuses D≥5 without `KT_ALLOW_UNSAFE_DEFERRAL=1` |
| D=6 | loses 0.23 nats/token | **REJ** (E21) |
| `KT_CPUINFER` 24 / 30 | 24 → −12%; 30 → collapses to 5 tok/s (SMT starvation of the TP launch threads) | **REJ**; 16 stays; warned at resolve time above physical cores |
| CPU-tier route lookahead (E31) | not run (opt-in probe): ceiling ≈8% C1 only if top-6 hit ≥90% and ≥1.5 ms idle/step | **PARKED** — cheap probe, run it before believing the ceiling |

## 5. Prefill, chunk, and memory shape

| Option | Outcome (measured) | Verdict |
|---|---|---|
| Chunk 4096 | the pre-T3.28 default; 262K×4 at chunk 4096 stress 1,082 s | superseded; kept as the 226/4096 fallback rung |
| **Chunk 8192** | first-turn TTFT down; prefill +38/+47% beside 219 experts | **ADOPT** (with 219) |
| Chunk 16384 | +22% prefill **at −4.4% C1** and 14 fewer experts | **REJ** (E34d-iv) |
| Chunk 2048 | frees ~1.5 GB/rank **but −40% prefill** | **REJ** (E29, collateral of 230) |
| Expandable segments after capture | 262K×4 stress **678 vs 1,082 s**; NLL 0 moved; C1/C4/prefill within noise | **ADOPT** (`SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1`) |
| Decoder SWA bounded replay (E35) | +39% 16K prefill **but +1.13 nats/token on fresh prompts** | **REJ**; exact variant (E35-ii, tail = full 2,560-row receptive field, ~+20% first-turn prefill) **PARKED** with the archived patch |
| Large-M prefill GEMM tuning (T3.21) | dense GEMM = 4.0% of a 16K TTFT → no prefill kernel reaches the +20% bar; cuTile 2.3–2.7× slower than the E20 Triton kernel | **CLOSED** on the stop clause |
| Autotune of the launch table (T3.27) | E20 already swept 480 configs × 14M buckets × 7 shapes; residual was a GEMV kernel | **CLOSED** (superseded by T3.41) |

## 6. Engram tiering and I/O

| Option | Outcome (measured) | Verdict |
|---|---|---|
| Engram in DRAM (FP8, 188.8 GiB) | does not fit this host with the CPU experts (−78 GiB) | **REJ** on this host (4-GPU column keeps it) |
| Engram on NVMe + page cache (`file` layout) | E3 showed both drives sustain decode-rate lookups (403K rows/s); cold T=1 1.02 ms with WILLNEED-16 | **ADOPT** (T3.12/E18) |
| Re-quantized Engram (LibertAI FP4 / q3_K) | **cannot shrink the page-cache footprint**: real text touches ~1 row per 4 KiB page (~190 KiB/first-touched token); LRU replay only 38% hits at 16 GiB | **REJ** (E11/E40, GitHub #2 closed). The real fix was cache hygiene, below |
| `wait-ready.py` page-cache drop | ~73 GiB of stale non-Engram shard pages pinned the cache to ~10 GiB; dropping them: prefill 16K 3,311 → 4,248, stress 320 → 257 s | **ADOPT** (E40/T3.10) |
| `ENGRAM_FILE_PREFETCH=ahead` (+round 2, chunk-ahead) | prefill +16/+26% cold, Engram break −24%, C1 +1.7%; chunk-ahead +7.6/+11% | **ADOPT** (T3.20) |
| Admission-time chunk-0 prefetch (T3.49) | −0.2…−0.4 s (3–9%) on a cold 16K TTFT admitted during decode; null on idle cold TTFT | **BUILT, default off** — reopen for C4-heavy admission |
| User-space Engram row cache (T3.48) | 264 B rows, ~22× denser than the page cache (75.6 GiB of touches → ~3.5 GiB) | **PARKED** (registered, low; pairs with T3.49) |
| Per-shard `fadvise` during load (T3.50) | keeps Engram residency at ready ≥ pre-launch figure | **PARKED** (small, folds into serve-state work) |
| Persistent kernel cache | first request paid +14–19 s of serving-time Triton compile | **ADOPT** (`KERNEL_CACHE_DIR`, T3.47/E40) |

## 7. Quantization and kernel formats

| Option | Outcome (measured) | Verdict |
|---|---|---|
| **MXFP4 as shipped** (E2M1 + UE8M0 block-32) | the upstream checkpoint itself; native CUTLASS SM120 grouped GEMM at 1,397 GB/s decode M=1 (93% of copy ceiling), 1,060 TFLOPS dense at 300 W | **ADOPT** — the checkpoint needs no quantization step |
| NVFP4 re-quantization | 15.8 GiB **larger** (g16 FP8 scales); value is kernel compatibility, not size | **REJ** for size; kept as the baseline engine's format |
| Marlin W4A16 dequant fallback | 40–50% slower; shifts DSpark acceptance (draft heads trained on native FP4) | **REJ** — detected by the E12 census; native path shipped |
| Sub-4-bit experts (3.0 bpw / 2.6 bpw trellis) | no SM120 kernels in vLLM/SGLang for `deepseek_v41`; GLM-5.3's EXL3 K3 needed a bespoke fork; quality unknown on FP4-trained experts | **REJ** (Option C/O). Reopen only with a published SM120 EXL3 MoE kernel with a KL number |
| Expert pruning (Option D, ~39%) | research-grade; gated on E7 skew (skew was real: busiest expert 17–30× mean) but no evaluated artifact exists | **PARKED** — not shippable |
| W4A8 (MXFP8 activations) | 2× slower at decode than W4A4 (128-row M tile issues M=128 of MMA) | **REJ** for decode (E13) |
| Decode-shaped FP8 GEMV (E44) | +12…+51% per call at M=1 but **+1–2% C1 served, C4 flat** | **SHIPPED OPT-IN** (`SGLANG_DSV41_FP8_GEMV=0` until a pinned A/B ≥ +2%) |
| Dense W8A8 block-[32,32] FP8 configs (E20) | 5.6–12.1× per kernel at M=1; **C1 14.0 → 31.6 (2.26×)** | **ADOPT** — configs + `is_sm120` gate shipped (overlay `32a25f37`) |
| Decode-shaped MXFP4 grouped GEMM M-tile (K10) | `TILE_M=64` fails the SM120 TMA-WS static assert; served path already at roofline | **PARKED** at roofline |

## 8. CUDA graphs, threads, OS knobs

| Option | Outcome (measured) | Verdict |
|---|---|---|
| Eager (no graphs) | 4.4–5.6× slower (15.9 tok/s no-DSpark) | **REJ** — capture is on the critical path, not an optimization |
| Capture list "1 2 3 4" | derived [1,2,4] pads bs=3 → C3 −22%; explicit bs=3 capture costs ~nothing | **ADOPT** (`CUDA_GRAPH_BS="1 2 3 4"`) |
| `HOST_THREAD_CPUS=auto` | ends the C4 106/123 bimodality (host threads floating onto worker cores); C4 126.7/123.0; prefill −0.7…−1.9% | **ADOPT** (T3.52/E46). A list overlapping worker CPUs is refused (forced collision C4 33.8) |
| One-CCD pinning / physical-cores-only | 2× slower / 10× worse | **REJ** (E37) |
| THP / NUMA tuning | THP ≤1%; NUMA null | **REJ — nothing adopted; keep the server unpinned** (E37/T3.40) |
| `-march=native` rebuild | bit-identical | **REJ** (E37) |

## 9. BIOS / firmware / power (host-specific, but the *protocol* transfers)

| Option | Outcome (measured) | Verdict |
|---|---|---|
| FCLK 2000 (Auto) | the baseline | superseded |
| FCLK 2100 at Auto voltage | MCE storm (GMI PCS bank) — failed | **REJ** at Auto voltage |
| **FCLK 2100 + VSOC 1.05 V / VDDG 950 mV** (slot 3) | +6% per-CCD DRAM read, **+3.1% C1** | **ADOPT** — standing BIOS profile |
| FCLK 2200 | won't boot | **REJ** (ceiling found) |
| NPS2 | measured-and-not-run; "expect ≤5%, possibly 0" | **PARKED** (documented in the BIOS guide) |
| GPU power cap 325 W | decode cap-insensitive, prefill pins to the cap (this model); GLM-5.3 was +9% C16 at 325 W — model-dependent | **ADOPT** the *sweep protocol* (T2.9/E8b), not a universal number |

## 10. Quality / determinism machinery

| Option | Outcome (measured) | Verdict |
|---|---|---|
| Fused-MoE finalize + unsorted indexer top-k | the two sources of run-dependent greedy (drift 320–390 chars at bs=1) | **REJ** — replaced by the levers below |
| `FUSED_FINALIZE=0` + `SORT_DECODE_TOPK=1` | bit-reproducible greedy at zero cost; corpus bit-stable | **ADOPT** (T3.15c) |
| Exact deterministic decode top-k (patch 0055) | 38/38 tie cases exact at zero cost; 14,775-token residual survived → one fp32 tie at layer 36, logprobs-only, tokens identical | **ADOPT + open finding** (E41/E42; upstream-facing sglang #35257 post ready) |
| NaN per-request rejection (`KT_REJECT_NAN_ROWS`) | injected NaN → 1×500, others complete, server up | **ADOPT** (C10) |
| Clamp max_new_tokens (`SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS`) | a 128K agent ceiling no longer 400s a 262K server | **ADOPT** (T3.35) |

## 11. How a new recipe uses this ledger

1. **Consult before designing an experiment**: if an option family appears here with a REJ on the same tier
   shape (CPU-hybrid MoE on consumer Blackwell), start from the rejection reason, not from zero — but re-run
   the *gate*, not the measurement, if your hardware differs.
2. **Every new rejection lands here with numbers** (the measured outcome column is mandatory — "didn't help"
   is not an entry), plus the reopen trigger if one exists.
3. **The registry guards mirror the ledger**: a condemned combination refuses at launch with the ledger's
   message. That is how a rejection stays enforced instead of becoming folklore.
4. **Reopen triggers are the only legitimate way back**: each REJ lists what would have to change (e.g.
   "speculation reopens only after a CPU-tier change"). Without the trigger, a retry is re-measuring a
   recorded negative.

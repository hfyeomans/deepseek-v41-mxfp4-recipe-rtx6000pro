# E6 stage 2 — design: `deepseek_v41` on a 2-GPU three-tier hybrid

Status: **design, 2026-09-21** (T3.5). GPU-free. Items marked **[verify]** are
checked in T3.5b against the source shipped inside the two images T3.2 pulls
(`lmsysorg/sglang:dev-dsv41`, `approachingai/ktransformers:DSV4-specific`);
the design does not depend on their outcome, the work breakdown does.

Sources this doc is built on: [FEASIBILITY.md](../docs/FEASIBILITY.md) §3–4,
[RESEARCH.md](RESEARCH.md) (model, hybrid engines, SM120 kernels),
[EXPERIMENTS.md](EXPERIMENTS.md) E1b/E3/E4/E4b/E6 s0/E12/E13/E15/E16,
upstream SGLang `main` (`python/sglang/srt/models/deepseek_v4.py`,
`layers/engram.py`, read 2026-09-21), KTransformers
`doc/en/DeepSeek-V4-Flash.md` and `kt-kernel/README.md` (read 2026-09-21).

## 1. What we are building and what "done" means

One engine process that serves `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…`
(the upstream MXFP4 checkpoint, no re-quantisation) on this host with:

| Tier | Holds | Size |
|---|---|---|
| 2 × RTX PRO 6000 (VRAM, TP2) | attention/CSA2, embeddings, head, shared expert, router, hyper-connections, DSpark, KV, **≈58 % of routed experts** | ≈157 GiB of experts + 17.5 GiB rest + KV |
| Host DRAM (246.8 GiB usable) | **≈42 % of routed experts, computed on the CPU** (KT-Kernel MXFP4), Engram row cache, staging buffers | 112.8 GiB experts + cache + reserve |
| NVMe (nvme0, Samsung 9100 PRO) | Engram tables (188.8 GiB FP8) read row-wise, hot rows cached in DRAM | 189 GiB |

Pass (from E6): a decode/prefill/needle table for V4.1-Flash on two GPUs in
BENCHMARKS.md with engine commit, expert split, Engram mode and power cap.
Calculator ceiling to beat or explain: **50 tok/s** decode (CPU path alone at
125 GB/s), **77 tok/s** with the B2 split (experts also streamed over PCIe);
prefill 2–3K tok/s (`scripts/feasibility-calc.py … --layout hybrid-cpu`).
First serve is allowed to be far below that; the first milestone is *correct
tokens* (`verify.py`: `17*19=323`, needle at 32K), the second is the number.

## 2. Constraints the experiments already fixed

| Constraint | Source | Consequence for the design |
|---|---|---|
| The GPU step must run under CUDA graphs (or torch.compile). Eager costs 4.4–5.6× at C1 (63 vs 11.2 ms/step). | E15 | The CPU-expert call and the Engram lookup must be **graph-capturable**: enqueued on the stream as a host function / stream callback, never a Python-side sync inside the step. |
| DRAM read ceiling 125 GB/s (CPU), 94 GB/s while a bf16 GEMV computes; 32 threads slower than 16. | E1b, E6 s0 | `--kt-cpuinfer 16`, one thread pool (1 NUMA node). CPU expert bytes/token are the decode bound; everything else hides behind them. |
| CPU reads and GPU DMA reads of DRAM are additive (146 + 114 GB/s). | E4b | B2 split: let the GPUs also stream a share of the CPU-tier experts over PCIe; 77 tok/s ceiling vs 50. Stage 2b, not first serve. |
| PCIe H2D 53 GiB/s per card, 106 aggregate. | E4 | Layerwise GPU prefill moves 112.8 GiB per 4096-token chunk → 1.06 s floor → ≤ 3.9K tok/s prefill. |
| Engram gather: 48 rows/token in 11 µs from DRAM, 1.9 ms from NVMe (Python, 16 threads); 403K rows/s NVMe. | E3 | NVMe-tier Engram costs ≤ 10 % of a ~20 ms decode step if issued asynchronously; prefill 4096 × 48 rows ≈ 0.5 s/chunk — same order as the expert transfer, must overlap. |
| GPU step for V4-Flash: 15.3 ms per 6-token verify step; experts 24.5 %, dense FP8 27 %, attention 11 %. | E12 | The GPU side is not the bottleneck of the hybrid; `--gpu-ms-per-token 5` stays. Don't spend kernel effort on GPU experts before the CPU path works. |
| MXFP4 (E2M1 + UE8M0 g32) is native on SM120; our grouped GEMM runs at 97–99 % of the copy ceiling. | E13, TK.2b | GPU experts consume the checkpoint's format directly; the E13 kernel is the TK.4 replacement if the engine's SM120 MXFP4 MoE path turns out to be a dequant fallback. |
| DSpark block size is a training constant (K = 5); acceptance 54–60 % on V4-Flash. | E8, E15 | Under a CPU-bound MoE a 6-token verify step touches up to 6× the distinct CPU experts. E10 decides on/off; the design must let both run. |
| Power caps do not move NVFP4 decode (T2.9). | T2.9 | 325 W is irrelevant to the hybrid's decode; keep whatever the daily driver wants. |
| GLM's 16-slot cap is a KV/Mamba pool limit, not a scheduler one. | E16 | For V4.1 the KV is 890 B/token; slots are bounded by CPU expert traffic, not memory. Plan for C1–C4, not C16. |

Host prerequisites found while writing this (record in HARDWARE.md when T3.5b confirms):

- `transparent_hugepage/enabled = madvise` (fine for KT's and SGLang's `MADV_HUGEPAGE` anonymous maps) but **`shmem_enabled = never`** — SGLang's `shared` Engram host-table layout is memfd-backed and would get no huge pages ("expect ~10x slower lookups", `engram.py:_HostTable.finish_load`). Either `echo advise | sudo tee /sys/kernel/mm/transparent_hugepage/shmem_enabled` or use `per_rank` layout. Moot if Engram goes to NVMe (§4.3), relevant for the 4-GPU profile.
- 1 socket, 1 NUMA node, 16 cores / 32 threads → `--kt-threadpool-count 1 --kt-cpuinfer 16`.
- `free -g`: 246 GiB total, ~207 GiB currently page cache (the download). The Engram NVMe tier relies on the page cache; nothing else on the host may claim it while serving.

## 3. Engine route: bolt KT onto upstream SGLang, do not port V4.1 into the fork

The plan text said "port `deepseek_v41` into sglang-kt". Reading the two code
bases inverts that:

- **Upstream SGLang `main` already serves `deepseek_v41`.** It is not a new
  model file: `deepseek_v4.py` carries `is_dsv41 = model_type == "deepseek_v41"`
  and branches on it. The V4.1-only pieces are separate modules:
  `layers/engram.py` (`EngramLayout`, `EngramHasher`, `EngramEmbedding`,
  `_HostTable`, `Engram`), `layers/attention/dsv4/dsv41_sparse.py`
  (`DeepseekV41Compressor`, `DeepseekV41Indexer`),
  `layers/attention/deepseek_v4_backend.py`, `mem_cache/deepseek_v4_memory_pool.py`,
  `kernels/ops/attention/flash_mla_sm120` (an **SM120 decode kernel** with a
  `flashinfer_dsv4_decode_supports_num_heads` gate), `layers/hyperconnection.py`
  + `kernels/ops/layernorm/mhc` (TileLang; DeepGEMM prenorm path gated on
  `SGLANG_OPT_DEEPGEMM_HC_PRENORM`), `models/deepseek_v4_dspark.py` / `dspark.py`,
  `models/deepseek_v41_vit.py`, tokenizer/reasoning/tool parsers `deepseek-v41`.
  0xSero's 4 × RTX PRO 6000 run proves this path on SM120.
- **KT-Kernel's CPU side is a pip package, not a fork.** `kt_kernel.KTMoEWrapper`
  (`submit_forward(hidden, topk_ids, topk_weights, cuda_stream)` /
  `sync_forward(hidden, cuda_stream)`, `set_capture_batch_sizes([...])`,
  `method="MXFP4"`, `num_gpu_experts`, `max_deferred_experts_per_token`) is
  what does the CPU work. The fork's V4 support is **DeepSeek-V4-Flash-0731**
  (`configs/deepseek_v4.py` declares `model_type = "deepseek_v4"` only; no
  V4.1 branches; submodule `541ddc37cbc9…`, pins `kt-kernel==0.7.0.post4`).
  Its docker image validates SM120 (`triton_kernels` MXFP4 MoE, Triton NSA
  fallback) and captures CUDA graphs with CPU experts inside
  (`--cuda-graph-bs 1 --cuda-graph-max-bs 1` in the reference command), which
  is the existence proof for §4.1.
- **Upstream `main` already ships the KT glue** (verified 2026-09-21 against
  `sgl-project/sglang` source): `layers/moe/kt_ep_wrapper.py` defines
  `KTEPWrapperMethod(FusedMoEMethodBase)` (`_quant_wrapper_id = "kt_ep"`),
  `deepseek_v4.py` imports it by side effect
  (`_try_side_effect("sglang.srt.layers.moe.kt_ep_wrapper")`),
  `fused_moe_triton/layer.py::FusedMoE` installs it whenever
  `--kt-weight-path` is set, `server_args.py` has `--kt-weight-path
  --kt-method --kt-cpuinfer --kt-threadpool-count --kt-num-gpu-experts
  --kt-max-deferred-experts-per-token`, and
  `model_executor/runner/decode_cuda_graph_runner.py` calls
  `KTMoEWrapper.set_capture_batch_sizes([bs × num_tokens_per_bs …])` before
  capture. The fork's extra pieces (placement strategies, dynamic update,
  layerwise MXFP4 prefill) also exist in the fork's `kt_ep_wrapper.py`; how
  much of that is in upstream's copy is the remaining T3.5b diff.

Porting direction therefore: **Route 1 — take upstream SGLang at the
`dev-dsv41` image's commit (or `main`), add `kt-kernel`, and make the
existing `KTEPWrapperMethod` work for V4.1.** The V4.1 model code moves zero
lines and the glue already exists; the work is compatibility: (a) KT's
SiLU-only assertion vs V4.1's activation, (b) wrapping `Fp8MoEMethod` with
`is_fp4_expert=True` (the class upstream actually instantiates for V4.1
routed experts — see §4.1 rule 6) instead of a `Mxfp4MoEMethod`, (c) 384
experts / top-6 with `--kt-num-gpu-experts` as a **per-layer** count, (d) TP2
behaviour (§4.1 rule 4), (e) placement beyond the wrapper's contiguous
`expert_id >= num_gpu_experts` policy if we want `frequency`.

Route 2 (port V4.1 into the fork) stays as the fallback and is chosen only if
T3.5b finds that upstream's `kt_ep_wrapper.py` is a stub relative to the
fork's (e.g. lacks the MXFP4 layerwise prefill or the placement mask) and the
fork's version cannot be dropped in (**[verify]** in T3.5b: `diff` the two
files inside the two images).

Why not vLLM: `--cpu-offload-gb` is parameter-level streaming with no routing
awareness (E5 records its failure mode); no CPU expert kernel.

## 4. Component design

### 4.1 Hybrid MoE layer (upstream's `KTEPWrapperMethod`, adapted)

V4.1 has no `DeepseekV4MoE`: `deepseek_v4.py` reuses `deepseek_v2.py`'s MoE,
whose routed call is `self.experts(hidden_states, topk_output)` on a plain
`FusedMoE`. The seam is `FusedMoE`'s quant method: with `--kt-weight-path`
set, upstream wraps the selected `FusedMoEMethodBase` in `KTEPWrapperMethod`
(`layers/moe/kt_ep_wrapper.py`), whose `apply` does exactly the split below.
Nothing new is written unless the wrapper rejects V4.1's method/activation:

```diagram
                    router (GPU, unchanged): topk_ids[T,6], topk_w[T,6]
                                      │
                       placement map  ▼  phys expert → {GPU slot | CPU}
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
   GPU experts (≈58 %)                        CPU experts (≈42 %)
   existing FusedMoE MXFP4 path,              KTMoEWrapper.submit_forward(h, ids_cpu, w_cpu, stream)
   ids for CPU experts masked to -1           → cudaLaunchHostFunc-style callback on `stream`
                    │                            cpuinfer 16 threads, MXFP4 GEMV from DRAM
                    │                            D2H h (T×5120 bf16) … H2D partial sums
                    ▼                                    ▼
                    └────────────── sum ─────────────────┘  (sync_forward on the same stream)
                                      │
                             shared expert (GPU) + hc merge
```

Rules:

1. **Submit CPU first, then run GPU experts, then sync.** The CPU work
   overlaps the GPU experts, the shared expert, and — with
   `max_deferred_experts_per_token > 0` — the next layer's attention. Deferral
   trades accuracy for latency (KT's docs: 1–4 "recommended", 5–7 "noticeable
   accuracy loss"); first serve uses 0, E10/E6 s2b sweep 0–2 with `verify.py`.
2. **Graph capture.** Verified: `kt-kernel/cpu_backend/cpuinfer.h` implements
   `submit_with_cuda_stream` / `sync_with_cuda_stream` with
   **`cudaLaunchHostFunc`** (graph-capturable; no `cudaStreamAddCallback`
   anywhere), and upstream's `decode_cuda_graph_runner.py` calls
   `KTMoEWrapper.set_capture_batch_sizes([bs × num_tokens_per_bs for bs in
   capture_bs])` before capture, so verify-shaped M (slots × (K+1) = 4 × 6 =
   24 with DSpark on) is registered automatically. KT allocates pinned
   input/id/weight/output buffers per registered size (`experts_base.py`);
   the practical constraint is that those buffer addresses stay fixed across
   replays, which KT handles by allocating once per shape.
3. **Placement.** `--kt-num-gpu-experts` is a **per-MoE-layer** count
   (upstream multiplies by the number of MoE layers to form a global budget,
   then a `[num_layers, num_experts]` mask; ≈ 0.58 × 384 = 223 per layer).
   Strategies in the fork (and to confirm in upstream's copy): `uniform`,
   `front-loading`, `random`, `frequency` (needs a `logical_count` histogram
   from `init_expert_location`). Rank 0 builds the mask and broadcasts it.
   Verified in the `kt-kernel` wheel (T3.6a): `generate_gpu_experts_masks`
   takes the `[layers, experts]` frequency tensor and picks the global
   top-(layers × N), so the per-layer count is exact only for `uniform`;
   under `frequency` hot layers get more GPU experts than cold ones and the
   per-GPU byte budget must be checked against the *actual* mask sum.
   **`--kt-enable-dynamic-expert-update` is skipped for MXFP4** (the copier
   assumes INT4/FP8/BF16 layouts) — so V4.1 placement is static; E7-style
   histograms have to come from a separate routing capture, not from the
   flag. Start `uniform`; `frequency` is a T3.6f optimisation.
4. **TP2.** Verified in `kt_ep_wrapper.py`: **only TP rank 0 constructs the
   `KTMoEWrapper` and runs cpuinfer**; other ranks contribute zero CPU output
   and the model's existing TP all-reduce sums rank 0's CPU result with the
   TP-sharded GPU expert output. So the CPU experts are computed once, DRAM
   traffic is not duplicated, and no `expert_id % tp_rank` split is needed.
   GPU experts keep the engine's TP sharding (E13's `--tp 2` shapes N = 2,304
   / K = 1,152). Consequence for pinning: `--kt-cpuinfer 16` threads all
   belong to rank 0's process; rank 1's process should not be pinned onto the
   same cores.
5. **Weights.** Verified: `kt-kernel/python/utils/loader.py::MXFP4SafeTensorLoader`
   reads `layers.N.ffn.experts.E.w1|w2|w3.weight` + `.scale` directly (probes
   both `model.layers.N` and `layers.N` prefixes). Weights are viewed as
   `uint8` with the two-E2M1-per-byte packing unchanged; UE8M0 scales are
   **losslessly** shifted into BF16 storage (`(s.int32 << 7).int16.view(bf16)`)
   — no requantisation. The CPU kernel is `operators/amx/fp4-moe.hpp`
   (needs AVX-512F/BW/BF16 — Zen 5 qualifies; AMX not required) with an
   `operators/avx2/mxfp4-moe.hpp` fallback. CPU copy is 112.8 GiB at 58/42
   *in checkpoint bytes* — see the T3.6b correction below.

   **T3.6b correction (measured, E17).** Two things the reading above got
   wrong. (a) kt-kernel's C++ `load_weights` repacks **every** expert of the
   layer into anonymous host memory regardless of `gpu_experts_mask` —
   8.0 GiB per layer flat, independent of how many experts are on the GPU
   (KTransformers' memory model is "the CPU holds all experts; the GPU holds
   copies of the hot ones"). 40 layers × 8.3 GiB ≈ 330 GiB does not fit. The
   overlay therefore installs a **compact CPU view**: the loader returns only
   the CPU-resident experts renumbered `0..n_cpu−1`, the wrapper is built
   with `num_experts = n_cpu`, and `remap_topk_ids_for_cpu()` shifts router
   ids by `num_gpu_experts` (GPU slots become `−1`, kt-kernel's skip
   marker). This requires the identity physical→logical map, which is what
   `init_trivial` gives without EPLB; `frequency` placement would need the
   loader to take an explicit id list instead of a prefix split.
   (b) The packed layout costs **21.8 MiB per CPU expert**, not 17.9 (the
   UE8M0→BF16 scale widening plus the AVX-512 tile layout is +22 %). Budget:
   160 CPU experts/layer × 40 = 6,400 × 21.8 MiB = **136 GiB**, load
   0.9 s/layer (≈ 36 s for the model, from page cache).
   Correctness vs an independent FP32 dequant reference on layer 5: rel-L2
   3.8e-3, max abs 0.021 at |y|max 4.4 — bf16-level. `swiglu_limit` **must
   be passed** to `KTMoEWrapper` (verified by driving inputs ×8: kt matches
   the clamped reference at 2.8e-3 and is 77 % off the unclamped one; with
   `swiglu_limit=0` it flips). GPU-slot ids set to `−1` are skipped exactly
   (the same tokens' outputs match the GPU-only reference).
   Threads: **16 (one per physical core) is the optimum** — 8 threads is 2×
   slower at T=1, 32 (SMT) is 7× slower (AVX-512 unit contention plus
   spin-waiting), `threadpool_count=2` is a wash. Timing must use **cold
   ids** (`--bench-cold`): with the same six experts every iteration the
   64 MiB L3 holds half of the 131 MiB T=1 working set and reports a fake
   160 GB/s. Cold, 16 threads: T=1 **1.08 ms** (= 127 GB/s, i.e. E1b's
   125 GB/s CCD-fabric ceiling — the tier is DRAM-bound at T=1 and 16 cores
   are exactly enough to saturate it, ~8 GB/s per core), T=4 4.0, T=8 7.6,
   T=16 13.8, T=64 39.8 ms; warm T=256 65 ms. Per-token cost stays flat to
   T≈16 (bandwidth-bound on distinct experts) and only amortises past T≈64.
   Floor: 2.5 CPU experts/token/layer × 21.8 MiB = 54 MiB/layer → 0.45 ms →
   × 40 layers ≈ 18 ms/token → **≈ 55 tok/s single-stream CPU floor** if
   fully overlapped (calculator said 50; the B2 split ceiling of 77 needs the
   PCIe path too). More GPU experts, deferral, or the B2 split move it;
   faster DIMMs do not.
   Note the class is named `AMX_FP4_MOE_TP` on this Zen 5 host too; the name
   is the code path family, not an AMX requirement.
6. **GPU expert kernel on SM120.** Verified: upstream's V4.1 routed experts
   are `Fp8Config`/`Fp8MoEMethod` with `is_fp4_expert=True` (detected from
   `routed_experts_quant_method == "mxfp4"` in `model_config.py`), weights
   allocated as `int8 [E, 2I, H/2]`. The auto override for DeepSeek-V4 on
   SM90/100/**120** selects `flashinfer_mxfp4`, which on SM120 is
   **FlashInfer CUTLASS `cutlass_fused_moe` MXFP8 × MXFP4 (`cutlass_sm120`)**
   — activations are quantised to MXFP8, weights stay E2M1 with interleaved
   UE8M0 scales padded to 128-element alignment. TRT-LLM FP4 MoE is SM100
   only. `marlin` and `triton_kernels` are selectable alternatives. E12's
   census must be redone on `cutlass_sm120`; TK.4 (E13 kernel) is only
   relevant if that path underperforms the E13 copy ceiling.

### 4.2 Expert split and the memory budget

`scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 2
--layout hybrid-cpu --engram nvme --context-tokens 262144 --sequences 4`
(2026-09-21):

| GPU experts | CPU experts | CPU GiB | host need | host margin | GiB/token | CPU ceiling | split ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8,920 (58 %) | 6,440 | 112.8 | 200.8 | **46.0** | 1.76 | 50 tok/s | 77 tok/s |

Per card the calculator gives 95.6 × 0.95 − 4 = 86.8 GiB; the hybrid adds two
GPU buffers the calculator does not model yet and that T3.4 must size:

- **layerwise-prefill staging**: one layer's CPU experts, 161 × 17.9 MiB =
  2.8 GiB (1.4 GiB per rank if TP-split), double-buffered → ≤ 5.6 GiB;
- **KT activation buffers** for the captured batch sizes: T × 5120 × 2 B in
  and out, pinned host + device mirrors — MBs, not GBs.

So the real first-serve split is closer to **55 % GPU / 45 % CPU** (≈ 8,450
GPU experts, 121 GiB CPU experts, 1.9 GiB/token, ≈ 46 tok/s ceiling). Stage 1
(T3.4) calibrates by sweeping `--kt-num-gpu-experts` on V4-Flash until the
first OOM, then backing off one step.

### 4.3 Engram: NVMe tier with a DRAM row cache (the one piece nobody ships)

The conflict: upstream's host table (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`)
puts the **full 188.8 GiB** in DRAM (memfd + `MADV_HUGEPAGE` +
`cudaHostRegister`; the GPU reads rows by zero-copy in `engram_gather`).
With 112.8 GiB of CPU experts that is 325.6 GiB against 246.8 (calculator:
`--engram dram` → −78.8 GiB, does not fit). Three ways out, in order of code:

| Option | Host GiB | Quality | Code | Lookup cost (E3) |
|---|---:|---|---|---|
| **A-min** — file-backed table: `torch.frombuffer(mmap(shard, MAP_SHARED|PROT_READ))` for weight and scale, `EngramEmbedding._owned_rows` takes the existing non-CUDA branch (`self.weight[local]` on a CPU tensor → fp8×e8m0 dequant → H2D). Page cache is the row cache; its size is whatever DRAM is left (≈ 40–60 GiB). | 0 pinned; cache elastic | bit-exact | ~40 lines: a third layout `"file"` in `_HostTable`, skip `cudaHostRegister`, run the lookup on CPU | miss 1.9 ms/48 rows (Python; kernel-side gather will be faster), hit 11 µs; hit rate unknown → E11 sim |
| **A-full** — A-min plus: hash ids for *both* Engram layers are known at step start (`EngramHasher` emits `[T, 2, 24]` in one call), so issue the layer-14 rows at layer 0 and hide the NVMe latency behind 13 layers; gather with `io_uring`/`preadv2` in a pinned staging buffer; graph-capturable via the same host-function pattern as 4.1. | as A-min | bit-exact | ~300 lines + a small C++/ctypes gather | miss latency off the critical path; prefill 196K rows/chunk ≈ 0.2–0.5 s overlapped with the 1.06 s expert transfer |
| **B** — FP4 Engram in DRAM (`LibertAIDAI/…-NVFP4`, 97.3 GiB) through the unchanged host-table code. | 97.3 pinned + 112.8 experts → margin 12.8 GiB | lossy, cosine 0.9934, unevaluated | gather kernel must learn an FP4 row format; different checkpoint for two tensors | 11 µs |

Decision: **A-min for first serve** (bit-exact, smallest diff, and it is what
0xSero's `OFFLOAD_MODE=nvme` does in spirit), **A-full as the first
optimisation after the number exists**, **B only as an E11 A/B** if A-full
still costs > 10 % of the step. The lookup on the CPU is not
graph-capturable in A-min. Verified upstream structure: the **hasher** is
already graph-broken (`bcg_deepseek_v4_engram_hash_ids = eager_on_graph(True)(…)`,
"reads per-request rows"), while `engram_gather` and the Engram gate sit
inside captured segments. A-min would move the gather to the eager side too,
i.e. a second break per Engram layer; whether the breakable-graph runner
tolerates that at acceptable cost is a T3.5b/T3.6c measurement, and if not,
A-full's host-function form (a `cudaLaunchHostFunc` node that fills a pinned
staging buffer, the same pattern KT uses in §4.1) is mandatory from the
start. Upstream has only `shared` (memfd) and `per_rank` (anonymous) host
layouts, both `cudaHostRegister`ed and zero-copy read by `engram_gather`;
there is no file-backed layout and no DRAM cache — this piece is genuinely
new.

Sizing check for A: 48 rows × 264 B = 12.4 KiB/token of table reads;
at 50 tok/s that is 620 KiB/s — the NVMe is idle. The cost is latency, not
bandwidth, hence prefetch.

### 4.4 CSA2 attention, indexer, KV pool — reuse as-is

Upstream classes take `config.compress_ratios`, `kv_source_layer_ids`,
`index_source_layer_ids`, `index_topk`; the memory pool is
`DeepSeekV4TokenToKVPool` at 890 B/token. Two SM120 items to confirm
(**[verify]**, then T3.6 smoke test): the `flash_mla_sm120` decode kernel
accepts `n_local_heads = 32` (TP2; 0xSero ran TP4 = 16 heads) via
`flashinfer_dsv4_decode_supports_num_heads`, and the prefill path does not
route to a `DeepGEMM` kernel (no SM120 support; `DEEPGEMM_SCALE_UE8M0` import
suggests a gated path). Fallback is FlashInfer/Triton, slower but correct.

### 4.5 Hyper-connections (mHC) — reuse; watch two gates

`hc_mult = 4` → 20,480-wide residual; kernels in `kernels/ops/layernorm/mhc`
(TileLang) with fused paths gated on `get_platform().is_blackwell` and
`0 < T ≤ 8` or `128 ≤ T ≤ 384` (verify-shaped). SM120 is Blackwell for that
predicate, so the fused paths engage; `SGLANG_OPT_DEEPGEMM_HC_PRENORM` must be
off (DeepGEMM). E12 census on V4-Flash put hyper-connections at 4 % of GPU
time; nothing to design.

### 4.6 DSpark — reuse; the design question is CPU traffic, not code

`models/dspark.py` + `deepseek_v4_dspark.py`, `--speculative-algorithm DSPARK
--speculative-dspark-block-size 5`. The three MTP stages (7.4 GiB, own
128-expert MoEs) stay on the GPU. Traffic model for E10:

| Mode | distinct CPU experts / layer / step | CPU GiB / step | DRAM ms @125 GB/s | tokens / step (acc 55–60 %) | tok/s (GPU 5 ms/tok) |
|---|---:|---:|---:|---:|---:|
| DSpark off | 6 × 0.42 = 2.5 | 1.76 | 14 | 1 | **≈ 50** |
| DSpark K=5 (6-token verify, all distinct) | ≤ 36 × 0.42 = 15 | 10.5 | 84 | 3.3 | **≈ 35** |
| DSpark K=2 (3-token verify) | ≤ 18 × 0.42 = 7.6 | 5.3 | 42 | ≈ 2.2 | ≈ 40 |

Distinct-expert overlap across the 6 draft tokens (E7's histogram) pulls the
K=5 row up; the B2 split (GPU DMA reads additive, E4b) pulls every row up.
The design keeps DSpark switchable and the block size a flag; E10 measures
the table. First serve: **off**.

### 4.7 Prefill — KT layerwise GPU prefill, reused

For chunks above `--kt-gpu-prefill-token-threshold` (4096, = chunk size) the
fork streams each layer's CPU experts to the GPU and runs the MoE there;
below it, the CPU computes the routed experts of the chunk. Ceiling from E4:
112.8 GiB / 106 GiB/s = 1.06 s per 4096-token chunk → 3.9K tok/s; expected
2–3K tok/s with the Engram gather (4.3) and attention overlapped. 262K in
≈ 1.5–2 min; 1M in ≈ 6–8 min (GLM: 4.4 min, V4-Flash: 8.4 min). Verified
in the fork's `_Mxfp4LayerwisePrefillManager`: rank 0 asks KT to write each
CPU expert into double-buffered pinned shared memory (two raw MXFP4 slots +
two Marlin-prepared slots, allocated lazily), and **every TP rank DMA-copies
its own TP slice** — so the transfer does split across both x16 links. The
path is gated to CUDA CC `(8,9)` and `(12,0)`, so SM120 is supported. Note
the GPU-side kernel for streamed experts here is **Marlin**, not
`cutlass_sm120`; layerwise prefill therefore has its own kernel to census.
Open: whether upstream's `kt_ep_wrapper.py` carries this manager (T3.5b
diff).

### 4.8 Serving surface

`--tp 2 --context-length 262144 --chunked-prefill-size 4096
--max-running-requests 4 --cuda-graph-bs 1,2,4 (× 6 with DSpark)
--kt-method MXFP4 --kt-weight-path <model> --kt-cpuinfer 16
--kt-threadpool-count 1 --kt-num-gpu-experts <per-layer N from T3.4>
--kt-expert-placement-strategy uniform|frequency --kt-enable-dynamic-expert-update
--kt-gpu-prefill-token-threshold 4096 --reasoning-parser deepseek-v41
--tool-call-parser deepseekv41 --trust-remote-code`, plus
`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file`
(the new layout from 4.3). Profile `2gpu-hybrid` in `config/example.env` /
`scripts/serve.sh` (T3.11), unqualified until T3.6 lands numbers.

## 5. Work breakdown (replaces T3.6's one line)

| ID | Work | GPU? | Done when |
|---|---|---|---|
| T3.5b | Inspect both images (pulled 2026-09-21, digests in PROVENANCE.md): `dev-dsv41`'s SGLang commit from image metadata / `sglang.__version__` / `pip show`; does its `kt_ep_wrapper.py` exist and how does it differ from the fork's (placement mask, `_Mxfp4LayerwisePrefillManager`, SiLU assertion)? Is `kt_kernel` importable in `dev-dsv41`? FlashInfer version and `_DECODE_DSV4_DISPATCH` head counts (need 32 for TP2). Record in RESEARCH.md; flip the remaining **[verify]** items in §8. | no | §8 closed; route 1/2 confirmed |
| T3.4 | E6 stage 1 as planned (V4-Flash in the KT image, `--kt-num-gpu-experts` sweep, TP1 and TP2). Calibrates `--gpu-ms-per-token`, `--dram-gbps` and the OOM edge for §4.2 (the TP2 duplication question is already answered: rank 0 only). | yes | `results/e6-stage1.json`, calculator flags updated |
| T3.6a | ~~Fork upstream SGLang~~ **Done 2026-09-21:** `config/Dockerfile.dsv41-kt` → `dsv41-kt:recipe` (`dev-dsv41` digest + `kt-kernel==0.7.0.post4 --no-deps`); build-time `config/kt_smoke.py` passed (avx512_bf16 ext, mask helper). T3.6b's SGLang edits go in as a bind-mounted overlay of `/sgl-workspace/sglang/python/sglang` first, then get baked into the image once they work. | no | `import kt_kernel; KTMoEWrapper(...)` constructs |
| T3.6b | Make upstream's `KTEPWrapperMethod` accept V4.1: wrap `Fp8MoEMethod(is_fp4_expert=True)`, clear the SiLU-only assertion if V4.1's activation differs, confirm `MXFP4SafeTensorLoader` resolves V4.1's tensor names from `--kt-weight-path`, confirm 384/top-6 per-layer mask. Unit test on one layer with random MXFP4 weights against a dequant reference (reuse `kernels/e13` reference code). | no (CPU-only test) | one-layer hybrid output = reference within bf16 tolerance |
| T3.6c | Engram layout `file` (A-min, §4.3). Test: same rows from file-backed vs device table on a 10M-row synthetic. | no | bit-exact rows; hit/miss latency logged |
| T3.6d | **First serve**: V4.1-Flash, TP2, DSpark off, uniform placement, ~55/45. `verify.py`, needle 32K, `bench-decode.py` C1. Scrub logs to TROUBLESHOOTING.md whatever happens. | yes | first V4.1 decode number in BENCHMARKS.md (or a recorded blocker) |
| T3.6e | Stabilise: CUDA-graph capture with CPU experts, layerwise prefill, 262K needle, C1–C4, 32K prefill, power. E12 census on the hybrid. | yes | E6 pass table |
| T3.6f | A-full Engram prefetch (§4.3); E10 DSpark table (§4.6); deferred-experts sweep with `verify.py`; frequency placement from E7. | yes | each a BENCHMARKS.md row |
| T3.6g | B2 split: a share of the CPU-tier experts streamed to the GPU per step (PCIe reads add to CPU reads, E4b). Only if T3.6e sits at the CPU ceiling. | yes | ≥ +20 % decode or a recorded negative |
| TK.4 | E13 kernel into the GPU MoE path if T3.5b/E12 find a dequant fallback. | yes | E12 re-census |

Order: T3.5b → T3.4 → T3.6a–c (GPU-free, can run while V4-Flash/GLM serve) →
T3.6d in the next GPU window.

## 6. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| `kt-kernel` wheel ABI vs the image's torch/CUDA 13 | import error | build from source in the image; the AVX-512 variant is what we want anyway |
| CPU-expert callback breaks graph capture on this SGLang version | capture crash / eager-only | piecewise capture around the MoE (upstream `register_split_op`), accept partial eager, measure vs E15 |
| FlashInfer SM120 dispatch gaps for CSA2/indexer at TP2 head counts | NaN or unsupported-heads error | TP2 with `--attention-backend triton`, or attention DP; slower but correct |
| Page cache thrash: Engram misses evict expert pages? No — CPU experts are anonymous memory, not file-backed; but the download's 207 GiB of cache must be dropped first | `free -g` buff/cache | `echo 1 | sudo tee /proc/sys/vm/drop_caches` before serve; `vm.swappiness=0` |
| Host margin 46 GiB shrinks under the fork's pinned buffers | OOM-killer during load | lower `--kt-num-gpu-experts` (more GPU) is the wrong way; drop DSpark (7.4 GiB GPU → more GPU experts) or trim the Engram cache instead |
| Route 1 glue cannot be lifted | T3.5b diff is large and touches scheduler internals | Route 2 (port forward into the fork); llama.cpp `deepseek41` (T3.7) once CSA2 lands |
| Correctness drift from deferral / dynamic update | `verify.py` or needle fails | `--kt-max-deferred-experts-per-token 0`, static placement |

## 7. Measurement plan (what proves it)

1. `verify.py` (`17*19=323`, deterministic prompt set) after every stage.
2. Needle at 32K, then 262K (KV 233 MB; the cost is prefill time, ~1.5–2 min).
3. `bench-decode.py --concurrency 1,2,4 --max-tokens 512`, DSpark off and on,
   the same fixture as E8/E14 so the scoreboard row is comparable; report wall
   tok/s, TTFT, acceptance.
4. `bench-prefill.py` 32K with a fresh `--seed` (prefix cache).
5. Host counters during decode: DRAM GB/s (`perf`/`pcm` if available, else the
   E1b harness in parallel), CPU utilisation per core (16 vs 32 threads),
   NVMe IOPS for Engram, GPU power (expected 150–250 W, starved — that is the
   design, not a fault).
6. Kernel census (E12 procedure) on the hybrid to split the step into GPU /
   CPU-expert / Engram / sync time; the first optimisation target is whatever
   is not overlapped.

## 8. Open questions T3.5b must close (**[verify]** list)

Closed 2026-09-21 from source (Librarian, `sgl-project/sglang` main,
`kvcache-ai/sglang` main, `kvcache-ai/ktransformers` main) — see §3/§4:

- ~~1~~ Fork base: submodule `541ddc37cbc92c60dc748db5ff1a2aad0b069a80`
  (ktransformers commit `f7607c0f`), `sglang-kt`, `kt-kernel==0.7.0.post4`.
  **TP=2 does not duplicate CPU work**: rank 0 only.
- ~~2~~ `cudaLaunchHostFunc`; `set_capture_batch_sizes` receives
  `bs × num_tokens_per_bs` from upstream's decode graph runner, so verify
  shapes are covered.
- ~~3~~ `MXFP4SafeTensorLoader` reads `layers.N.ffn.experts.E.w{1,2,3}.{weight,scale}`
  directly; weights unchanged, scales losslessly UE8M0→BF16.
- ~~4 (backend)~~ SM120 → `flashinfer_mxfp4` → CUTLASS `cutlass_sm120`
  MXFP8×MXFP4; TRT-LLM FP4 MoE is SM100-only.
- ~~5 (structure)~~ Hasher is `eager_on_graph`; gather/gate are in-graph;
  no file-backed layout exists upstream.

Still open, need the images (GPU-free):

1. `dev-dsv41`'s SGLang commit (no source mapping exists for the tag; read
   image labels / `pip show sglang` / `sglang.version`) and whether its
   `kt_ep_wrapper.py` matches the fork's (placement strategies, layerwise
   MXFP4 prefill manager) or is a reduced copy.
2. Whether `kt_kernel` is installed in `dev-dsv41`; if not, whether the
   `kt-kernel==0.7.0.post4` wheel matches its torch/CUDA ABI.
3. `_DECODE_DSV4_DISPATCH` in the image's FlashInfer: is 32 heads (TP2) in
   the table? (`SM120_DECODE_MAX_TOKENS = 64`, so DSpark verify at 4 × 6 = 24
   rows is under the decode ceiling.)
4. Cost of a second graph break per Engram layer for A-min (measurement, T3.6c
   on synthetic data or T3.6d in the first serve).
5. V4.1's routed-expert activation vs KT's SiLU-only assertion.

## 9. Learning pointers for this piece

- KT-Kernel API and the CPU/GPU split: `kt-kernel/README.md` §"Direct Python
  API Usage" (`submit_forward` / `sync_forward`, deferred experts,
  `set_capture_batch_sizes`); `doc/en/kt-kernel/experts-sched-Tutorial.md`
  for placement strategies.
- Why a host callback can live inside a CUDA graph: CUDA Programming Guide
  §"Stream Ordered Host Functions" (`cudaLaunchHostFunc`) vs the deprecated
  `cudaStreamAddCallback`; then re-read LEARNING.md §6.1 with that in hand.
- Zero-copy host memory from a kernel: `cudaHostRegister` + UVA, and why
  hugepages matter (one TLB miss per 264-byte row otherwise) —
  `layers/engram.py:_HostTable` is a compact worked example.
- Linux page cache as a free cache tier: `mmap` + `MAP_SHARED` on a file,
  `madvise(MADV_RANDOM)`, `preadv2(RWF_NOWAIT)` to detect misses; io_uring for
  batched random reads (the A-full gather).
- The model pieces themselves: LEARNING.md §1 (CSA2, Engram, mHC, DSpark) and
  the SGLang blog `lmsys.org/blog/2026-09-10-deepseek-v41`.

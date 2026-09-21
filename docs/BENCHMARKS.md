# Benchmarks

Local observations for immutable profiles on this host. Nothing here is a universal performance claim. Power condition for every GPU measurement on this machine: RTX PRO 6000 Blackwell **Max-Q at 300 W** (325 W maximum). Numbers from 600 W cards or other hosts are cited as references, never mixed into these tables.

## Method (shared by every table)

- Decode: [`scripts/bench-decode.py`](../scripts/bench-decode.py). Fixed code-review fixture, temperature 0.2, fixed seed, thinking off, 256 forced output tokens per sequence, 2 warmups then 5 measured runs per concurrency level. Each sequence is timed from its own first streamed token to its last; aggregate = sum of per-sequence rates; TTFT excluded and reported separately. `wall tok/s` = total completion tokens ÷ batch wall time (TTFT included) — the honest aggregate when the server queues requests. Speculative acceptance = Prometheus counter delta per run.
- Power: `nvidia-smi --query-gpu=power.draw,clocks.current.sm,… -lms 500` sampled alongside a benchmark, summarised by [`scripts/summarize-power.py`](../scripts/summarize-power.py) as the busy-median (utilization > 50 %). [`scripts/bench-power-sweep.sh`](../scripts/bench-power-sweep.sh) repeats C1 decode and 32K prefill per power cap (root required).
- Prefill: [`scripts/bench-prefill.py`](../scripts/bench-prefill.py). Unique seeded prompts per run (prefix cache cannot hit), one output token, 3 runs. Effective prefill = server `prompt_tokens` ÷ client TTFT; TTFT includes tokenization and the one-token handoff.
- Functional: [`scripts/verify.py`](../scripts/verify.py). Alias and context ceiling, `17*19=323` with thinking off, thinking toggle, tool call, N-needle exact retrieval at the configured context with `finish_reason=stop` required.
- Raw JSON for every table lives in `evidence/` after the identifier scan in `scripts/ci.sh`.

## Checkpoint budget (measured from shard headers, P0)

Source: [`evidence/budget-*.json`](../evidence/) via [`scripts/checkpoint-budget.py`](../scripts/checkpoint-budget.py).

| Checkpoint | Total | Engram tables | Resident excl. Engram | Text-only excl. DSpark/vision |
|---|---:|---:|---:|---:|
| `deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a…` | 475.24 GiB | 188.83 GiB | 286.41 GiB | 278.12 GiB |
| `nvidia/DeepSeek-V4.1-Flash-NVFP4@3431dde3…` | 491.06 GiB | 188.83 GiB | 302.23 GiB | 293.94 GiB |
| `LibertAIDAI/DeepSeek-V4.1-Flash-NVFP4@dfce15b9…` | 399.90 GiB | 97.28 GiB | 302.62 GiB | 293.94 GiB |

## Placement verdict (derived, P0)

Source: [`scripts/feasibility-calc.py`](../scripts/feasibility-calc.py); table reproduced in [FEASIBILITY.md §3](FEASIBILITY.md). 2 × RTX PRO 6000 with Engram in host DRAM: **−112.1 GiB** VRAM margin. Does not fit.

## E1–E4 host microbenchmarks

E1 and E3 were run on this host while the GLM container was serving (GPU-free; CPU mostly idle). E2 and E4 need the GPUs. Raw JSON: [`evidence/e1-dram-bandwidth.json`](../evidence/e1-dram-bandwidth.json), [`evidence/e3-engram-gather-dram.json`](../evidence/e3-engram-gather-dram.json), [`evidence/e3-engram-gather-nvme0.json`](../evidence/e3-engram-gather-nvme0.json), [`evidence/e3-engram-gather-nvme1.json`](../evidence/e3-engram-gather-nvme1.json).

| Experiment | Metric | Hypothesis | Measured |
|---|---|---:|---:|
| E1 DRAM copy bandwidth, 1 process | GB/s (read+write) | 20–30 | **60.0** |
| E1 DRAM copy bandwidth, 16 processes | GB/s (read+write) | 250–320 | **139.9** (max 148.4) |
| E1 DRAM copy bandwidth, 32 processes | GB/s (read+write) | — | **146.2** (max 153.1) |
| E1b DRAM read-only, torch f32 sum, 16 threads (quiet host) | GB/s (read) | ≥ 200 | **124.7** (8 thr 67; 32 thr 118) |
| E2 GDDR7 D2D per card | GB/s (read+write counted) | 1,450–1,650 | **≈1,470** (1,370 GiB/s) |
| E3 Engram gather DRAM, 16 threads, 20 M-row table (4.9 GiB) | rows/s; 48-row p50 / p99 | ≥ 5 M; < 0.1 ms | **4.15 M**; 0.011 / 0.015 ms |
| E3 Engram gather Samsung 9100 PRO (O_DIRECT 4 KiB, 16 threads) | rows/s; 48-row p50 / p99 | ≥ 600K; < 1 ms | **403K**; 1.90 / 2.11 ms |
| E3 Engram gather Crucial T705 (O_DIRECT 4 KiB, 16 threads) | rows/s; 48-row p50 / p99 | lower than 9100 | **347K**; 2.20 / 2.38 ms |
| E4 PCIe H2D pinned, per card | GiB/s | 48–55 | **53.1 / 52.9** (D2H 53.3 / 52.4) |
| E4 PCIe H2D pinned, both cards concurrently | GiB/s aggregate | ≥ 90 | **106.1** (114 GB/s) |
| E4b CPU read + both-GPU H2D at once | GB/s combined; additivity | additive (≥ 0.9) | **233.9** (120.1 + 113.8); **0.984** |
| E6 stage 0 CPU expert GEMV, 101 experts/token, bf16, 16 threads | ms/token; GB/s | ≤ 13 ms (≤ DRAM time) | **76.1 ms** (7.15 GiB bf16) = **93.9 GB/s** → ≈20 ms at MXFP4 bytes |
| Hybrid decode ceiling, 2 GPUs, CPU compute (`feasibility-calc.py --layout hybrid-cpu`; 1000 / (5 ms + 1.75 GiB ÷ DRAM GB/s)) | tok/s | 56 @146; 89 @300 | **50 @125** (E1b); 40 @94 (E6 s0 sustained) |
| PCIe-streaming fallback ceiling, 2 GPUs (1000 / (5 ms + 1.75 GiB ÷ E4 aggregate)) | tok/s | ~42 @100 GiB/s | **46 @114 GB/s** |
| **Split ceiling** (Option B2: CPU compute + PCIe streaming, E4b additive) | tok/s | — | **78 @125+114; 71 @94+114** |
| E13 s1 MXFP4 grouped GEMM decode, gate/up 6×[1×4608×5120] | µs; weight GB/s | ≥ 1.3× Marlin | **53.8 µs; 1,397 GB/s** (93 % of E2) — Marlin pending (K5) |
| E13 s1 decode, down 6×[1×5120×2304] | µs; weight GB/s | — | **39.6 µs; 949 GB/s** |
| E13 s1 prefill, gate/up 384×[64×4608×5120] | ms; TFLOPS | — | **3.46 ms; 335 TFLOPS** |
| E13 s1 dense proxy, gate/up 6×[4096×4608×5120] | TFLOPS | ~2,000 dense spec (600 W card) | **1,060** at 300 W |

How to read E1 and E3:

- E1 is a Python `memoryview` memcpy in N processes (threads measured one core: the first version of the script held the GIL and reported a flat 61 GB/s at every thread count; that number is not in the table). 146 GB/s is 36 % of the 409.6 GB/s theoretical for 8 × DDR5-6400 and below the 250–320 GB/s hypothesis. It is a lower bound (no non-temporal stores, no hugepages, 4 KiB-page-faulted buffers); a STREAM run inside a container is the follow-up if the number ever matters. It now matters: the 2-GPU three-tier layout is DRAM-bandwidth-bound (FEASIBILITY.md §3a), so E1b replaces this lower bound with a C/torch read benchmark before the hybrid ceiling is quoted as final.
- E3 DRAM: 4.15 M rows/s from Python is 86K decode-token-equivalents per second and puts a 64K-token prefill's 3.1 M Engram lookups at ≈ 0.75 s. Even this unoptimised path is two orders of magnitude faster than decode needs. **Engram-in-DRAM is not a bottleneck on this host.**
- E3 NVMe: the harness does the 48 reads of one token *serially* on one thread, so the 1.90 ms p50 is 48 × ≈ 40 µs, i.e. the drive's unloaded 4 KiB O_DIRECT latency at queue depth 16. The `< 1 ms` pass criterion is therefore about the harness, not the drive: a native path issuing the 48 reads concurrently would see ≈ 40–60 µs per token. Throughput of 403K IOPS at QD 16 is consistent with the 2.2 M IOPS spec at QD ≥ 256. **Engram-from-NVMe is viable for decode on either drive** (8.4K token-equivalents/s on the 9100 PRO), and costs ≈ 7.7 s of lookups per 64K-token prefill versus 0.75 s from DRAM. That gap is why the 4-GPU profile puts Engram in DRAM, which this host can do and the 128 GB reference could not.

## E8 baseline: DeepSeek-V4-Flash-0731 NVFP4, TP2, 2 × RTX PRO 6000 Max-Q

Measured 2026-09-21 with this harness against the sibling deployment (`dsv4-nvfp4`: vLLM, `nvidia/DeepSeek-V4-Flash-0731-NVFP4`, TP2, `max_model_len` 1,000,000, `gpu_memory_utilization` 0.96, `max_num_seqs` **2**, DSpark K5, CUDA graphs on). Host otherwise idle (GLM container stopped). JSON in `evidence/baseline-v4-flash/`.

| Metric | Sibling recipe | This harness |
|---|---:|---:|
| Single-stream decode, no DSpark | 93.66 tok/s | **89.2 / 89.4 tok/s** (512 / 256 tok, 5 runs each, spread 89.1–89.5); C2 161 tok/s summed. Relaunched with `DSPARK=0`; 216 / 225 W per card |
| Single-stream decode, DSpark K5, 256 tok | 200.64 tok/s | **171–188 tok/s median** across two 5-run sets (min 160, max 238); acceptance 40–45 % |
| Single-stream decode, DSpark K5, 512 tok | — | **207–239 tok/s** (two 3-run sets; acceptance 51–63 %); 228 tok/s at 57 % after the DSpark relaunch (5 runs) |
| Aggregate decode C2 (summed per-seq rates) | 226 tok/s | **354 tok/s** (wall-clock 300 tok/s) |
| Aggregate decode C4 / C8 | — | wall-clock **301 / 277 tok/s**; the summed figure (659 / 1,239) is invalid because `max_num_seqs=2` queues the rest (TTFT 0.77 s / 2.5 s) |
| Prefill 8K / 32K / 128K | — | **6,123 / 7,810 / 6,157 tok/s** (TTFT 1.3 / 4.1 / 20.7 s, 3 runs each) |
| Prefill 512K (1 run) | — | **3,237 tok/s** (158 s) |
| ~1M-token prompt prefill | 513.8 s (≈ 1,944 tok/s, 998,869 tok) | **512.2 s (1,898 tok/s, 972,187 tok)**, 1 run; both cards at 300 W for 489 s, SM sagged to 1,867 / 1,815 MHz, peak 88 / 87 °C |
| 6-needle retrieval at ~222K tokens | 1M needle OK | **6/6**, TTFT 42.4 s (5.2K tok/s), `finish_reason=stop` |
| Context ceiling | 1,048,576 | 1,000,000 (`max_model_len` in the sibling `.env`) |
| GPU power, C1 decode | — | **242 / 249 W** busy-median per card, SM 2,272 / 2,310 MHz — below the 300 W cap |
| GPU power, 32K prefill | — | **300 / 300 W** (cap), SM 1,950 / 1,942 MHz — power-throttled |

Readout:

- **DSpark is worth 2.4–2.7× on this baseline, and its absence removes the noise.** Without speculation the single stream is 89 tok/s regardless of fixture or output length (11.2 ms per forward step, ±0.2 %), 4.8 % under the sibling's 93.66. Everything above 89 in the DSpark rows is acceptance, which is why those rows scatter 171–239. Energy follows: 0.20 tok/s/W without DSpark vs 0.46 with — speculation more than doubles decode efficiency because the verify step costs 15.3 ms (E12) for 2.5–3 accepted tokens instead of 11.2 ms for one.
- **The 5 % hypothesis failed on decode.** C1 with the 256-token fixture is 6–15 % under the sibling's 200.6; with 512 tokens it brackets it (207–239). The spread tracks DSpark acceptance (40 → 63 %), which depends on the content the fixture makes the model write, not on the GPUs. Comparisons between engines must therefore hold the fixture *and* output length fixed, and report acceptance next to tok/s. The 512-token row is the one to carry forward.
- **Prefill is length-dependent, not "~2K tok/s".** The sibling's 1,944 tok/s is a 1M-token number, reproduced here at 1,898 (−2 %); short prompts run 3–4× faster. The 1M run is also the only *sustained* load in the set: 8 minutes at the 300 W cap took the cards to 88 °C and SM clocks 6 % below the 32K-prefill figure (1.87 vs 1.98 GHz), so short prefill benches overstate long-context throughput by roughly that much on top of the length curve. The curve 7.8K (32K) → 6.2K (128K) → 5.2K (222K) → 3.2K (512K) → 1.9K (1M) is the attention/indexer cost growing with context; it is the baseline any V4.1 engine's CSA2 path is compared against.
- **`bench-decode.py` now reports `wall tok/s`** (total completion tokens ÷ batch wall time, TTFT included) beside the summed per-sequence rate. Use the wall figure whenever concurrency exceeds the server's `max_num_seqs`.
- **Power:** decode already sits at 242–249 W per card with the cap at 300 W, so a 250 W cap (the driver's `power.min_limit` on these cards — 225 W is not settable) should cost decode almost nothing; prefill runs into the cap and drops SM clocks ~15 %. Measured in E8b below.

### E8b power sweep: tok/s per watt at 325 / 300 / 275 / 250 W

`sudo bash scripts/bench-power-sweep.sh --url http://127.0.0.1:8000 --model dsv4-nvfp4 --caps 325,300,275,250` (owner ran it; `nvidia-smi -pl` needs root). Same C1 512-token decode fixture and 32K prefill as E8, 3 runs per cap, nvidia-smi sampled at 500 ms; "busy" = median over samples with utilization > 50 %. Caps run in the order listed, so the 325 W rows are on cold cards (44 °C) and the 250 W rows on warm ones (74 °C). Evidence: [`evidence/e8b-power-sweep/`](../evidence/e8b-power-sweep/) (`summary.json`, per-cap `decode-c1-*w.json`, `prefill-32k-*w.json`, `power-*.csv`).

| Cap | Decode tok/s (acceptance) | Decode busy W, both cards | SM MHz | tok/s/W | Prefill 32K tok/s | Prefill busy W | SM MHz | tok/s/W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 325 W | 229.0 (55.7 %) | 458 (228 / 231) | 2,265 / 2,310 | 0.500 | **8,770** | 646 (323 / 323) | 2,085 / 2,100 | 13.58 |
| 300 W | 216.1 (51.2 %) | 472 (233 / 239) | 2,265 / 2,302 | 0.457 | 8,428 | 600 (300 / 300) | 1,987 / 1,976 | 14.04 |
| 275 W | 226.3 (54.5 %) | 489 (241 / 248) | 2,265 / 2,302 | 0.463 | 7,991 | 550 (275 / 275) | 1,867 / 1,845 | 14.52 |
| 250 W | 226.3 (54.1 %) | 496 (246 / 251) | 2,257 / 2,265 | 0.456 | 7,482 | 500 (250 / 250) | 1,740 / 1,702 | **14.96** |

Readout:

- **Decode does not see the cap.** 216–229 tok/s at every setting, SM clocks 2.26–2.31 GHz throughout, and the cards never reach even 250 W (the 300 W row's 216 is its lower DSpark acceptance, 51 % vs 54–56 %). The ±20 W spread in decode draw follows card temperature over the sweep (228 W at 44 °C first, 251 W at 70 °C last), not the cap — leakage current rises with temperature. Capping four cards at 250 W costs decode nothing.
- **Prefill is exactly cap-bound.** Each row sits at its cap to within 0.5 W, and throughput follows SM clock almost linearly: 7,482 → 7,991 → 8,428 → 8,770 tok/s for 1.72 → 1.86 → 1.98 → 2.09 GHz. 250 W costs **11 % of prefill for 17 % less power**; 325 W buys **+4 % for +8 %**. Efficiency improves monotonically as the cap drops (13.6 → 15.0 tok/s/W), so on a power budget the right move is more cards at a lower cap, not fewer cards at 325 W.
- **Decision for T4.0 (four cards on the 1,500 W PSU):** the 250 W cap is confirmed as a cheap option — 0 % decode, −11 % prefill — matching the E8 prediction (0 / ~15 %). 325 W is a legitimate 2-GPU setting when prefill latency matters (2 × 325 W + CPU ≈ 1,100 W fits the PSU).
- **Method lesson:** the first run's 275/250 W prefill rows were prefix-cache hits (32K tokens in 0.14 s, "223K tok/s") because every cap used the same seeded prompt; the script now seeds `777 + cap`, and `summarize-power.py` flags prefill runs with TTFT < 1 s as suspect. Also, a sampler leaked from a run that died before the server was up kept writing into the next run's CSV for 1.7 hours; the sampler is now killed from the EXIT trap and the summarizer drops non-sample lines.

## E12 kernel census: what the V4-Flash NVFP4 TP2 baseline actually runs

Source: the sibling recipe's torch-profiler traces (`results/component-profiles/*.json`, 20 DSpark verify steps per case, rank 0), re-categorised by [`scripts/kernel-census.py`](../scripts/kernel-census.py) → `evidence/e12-kernel-census-v4-flash-tp2.json`. Sums include stream overlap; the union column is the wall-clock-relevant figure.

| Case | Summed ms/step | Union ms/step | MoE FP4 (CUTLASS SM120) | Dense FP8 | bf16 GEMM | Attention + indexer | Draft (DSpark) | Collective | Hyper-conn | Marlin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Short code C1 | 19.76 | 15.34 | 4.85 (24.5 %) | 5.34 (27.0 %) | 2.32 (11.8 %) | 2.17 (11.0 %) | 1.94 (9.8 %) | 0.62 | 0.80 | 0.24 |
| 48K-context code C1 | 21.27 | 16.84 | 5.01 (23.6 %) | 5.62 (26.4 %) | 2.32 | 3.22 (15.2 %) | 1.96 | 0.62 | 0.81 | 0.25 |
| Prose+code C2 | 28.34 | 24.02 | 8.88 (31.4 %) | 5.50 (19.4 %) | 1.87 | 2.23 | 2.60 | 4.50 (15.9 %) | 0.96 | 0.25 |

Top kernels (short C1, per step): `_w8a8_triton_block_scaled_mm` 3.16 ms (236 launches — the dense/attention FP8 projections), CUTLASS `GemmUniversal<GroupProblemShape…MainloopSm120ArrayTmaWarpSpecializedBlockScaled…KernelPtrArrayTmaWarpSpecializedCooperativeBlockScaledSm120, tile 128×64, float_e2m1_t × float_ue4m3_t>` 3.15 ms gate/up (43 launches × 73 µs) + 1.70 ms down (43 × 40 µs), `_deepseek_v4_sm12x_fp8_einsum_kernel` 1.21 ms, `cutlass_80_wmma … bf16` 0.77 ms, `flashinfer::sparse_mla_sm120::sparse_mla_decode_dsv4_kernel` 0.69 ms (1.52 ms at 48K), `per_token_group_quant_8bit` 0.66 ms, `cross_device_reduce_1stage` 0.58 ms (4.4 ms at C2).

Readout:

- **The E12 hypothesis (routed experts on Marlin W4A16) is false for this deployment.** vLLM dispatches the NVFP4 experts to a CUTLASS SM120 block-scaled **PtrArray grouped GEMM** — the same kernel family as E13, with `nv_float4_t`/`ue4m3` instead of `mx_float4_t`/`ue8m0`, cooperative schedule, 128×64 tile. Marlin appears only for one small projection (0.24 ms/step).
- **Experts are ≈ 25 % of GPU time, not ≥ 50 %.** Dense FP8 layers (attention projections, shared expert, MTP head) are the single largest category at 27 %, then bf16 GEMMs 12 % and attention 11 %. Speeding up the expert GEMM alone therefore caps at a ~25 % gain on this engine; the hybrid's GPU term is dominated by everything *except* the experts once they leave the GPU.
- **ms/token:** a verify step is 15.3 ms of union kernel time for 6 tokens (5 drafts + 1); at the measured 40–63 % acceptance it yields ≈ 3–4 accepted tokens, i.e. **3.8–5.1 ms per accepted token**. The calculator's `--gpu-ms-per-token 5` stands as the conservative end.
- **E13 versus the deployed kernel, same shapes.** At TP2 per-rank shapes (gate/up N=2,304 K=5,120; down N=5,120 K=1,152) and 36 groups (452 MB of weights, well past the 128 MB L2), E13 W4A4 streams **1,314–1,343 GB/s** on gate/up and **1,301–1,435 GB/s** on down; at TP1 with 451 MB it reaches **1,421–1,451 GB/s** (97–99 % of the E2 copy ceiling). vLLM's 73 µs gate/up launch at that bandwidth moves ~98 MB ≈ 15 distinct experts per 6-token step (6 × top-6 with overlap), and E13 at 15 groups runs the same launch in 72.5 µs. **Both kernels are at the memory roofline; there is no speed left to win in the expert GEMM on SM120 — E13's value is the MXFP4/UE8M0 format V4.1 ships in, not speed.** (Caveat on the g=15 row: 94 MB fits L2, so its M=4 figure of 1,611 GB/s is L2-assisted; use the g=36 rows for HBM numbers.)
- **Not covered:** a prefill census (the sibling traced decode only) and the V4.1 engines, which need the E6 build. `nsys` cannot attach to the running container; a future census uses vLLM's `--profiler-config` / `/start_profile` per the sibling's `docs/component-profiling.md`.

## E9: DeepSeek-V4.1-Flash on 4 × RTX PRO 6000 (blocked on hardware)

Reference: 0xSero, 4 × RTX PRO 6000 at 275 W, 128 GB DDR5, Engram on NVMe + 64 GiB cache, SGLang `dev-dsv41`, TP4/EP4, DSpark 5: C1 ~200–230 tok/s, C8 ~700–750 tok/s, prefill ~7,000 tok/s, acceptance 79–88%.

| Concurrency | Aggregate decode (Engram in DRAM) | Aggregate decode (Engram on NVMe) | Median acceptance |
|---:|---:|---:|---:|
| 1 | — | — | — |
| 2 | — | — | — |
| 4 | — | — | — |
| 8 | — | — | — |

| Prompt tokens | Prefill tok/s (DRAM) | Prefill tok/s (NVMe) | TTFT (DRAM) | TTFT (NVMe) |
|---:|---:|---:|---:|---:|
| 8,192 | — | — | — | — |
| 32,768 | — | — | — | — |
| 131,072 | — | — | — | — |
| 262,144 | — | — | — | — |

# Benchmarks

Local observations for immutable profiles on this host. Nothing here is a universal performance claim. Power condition for every GPU measurement on this machine: RTX PRO 6000 Blackwell **Max-Q at 300 W** (325 W maximum). Numbers from 600 W cards or other hosts are cited as references, never mixed into these tables.

## Method (shared by every table)

- Decode: [`scripts/bench-decode.py`](../scripts/bench-decode.py). Fixed code-review fixture, temperature 0.2, fixed seed, thinking off, 256 forced output tokens per sequence, 2 warmups then 5 measured runs per concurrency level. Each sequence is timed from its own first streamed token to its last; aggregate = sum of per-sequence rates; TTFT excluded and reported separately. `wall tok/s` = total completion tokens ÷ batch wall time (TTFT included) — the honest aggregate when the server queues requests. Speculative acceptance = Prometheus counter delta per run.
- Power: `nvidia-smi --query-gpu=power.draw,clocks.current.sm,… -lms 500` sampled alongside a benchmark, summarised by [`scripts/summarize-power.py`](../scripts/summarize-power.py) as the busy-median (utilization > 50 %). [`scripts/bench-power-sweep.sh`](../scripts/bench-power-sweep.sh) repeats C1 decode and 32K prefill per power cap (root required).
- Prefill: [`scripts/bench-prefill.py`](../scripts/bench-prefill.py). Unique seeded prompts per run (prefix cache cannot hit), one output token, 3 runs. Effective prefill = server `prompt_tokens` ÷ client TTFT; TTFT includes tokenization and the one-token handoff.
- Functional: [`scripts/verify.py`](../scripts/verify.py). Alias and context ceiling, `17*19=323` with thinking off, thinking toggle, tool call, N-needle exact retrieval at the configured context with `finish_reason=stop` required.
- Raw JSON for every table lives in `evidence/` after the identifier scan in `scripts/ci.sh`.

## Scoreboard — GLM-5.3-Flash vs DSV4-Flash on this box (as of 2026-09-21)

Best measured runs with one harness (`scripts/bench-decode.py`, `bench-prefill.py`, `verify.py`), 2 × RTX PRO 6000 Max-Q, 300 W unless stated. Medians are what to plan on; "best run" is the ceiling seen. Neither column measures answer quality beyond synthetic retrieval.

| Metric | GLM-5.3-Flash (EXL3 4-bit, DFlash K5) | DSV4-Flash (NVFP4, DSpark K5) | Winner |
|---|---|---|---|
| Single-stream decode, C1 | 132–135 tok/s (40 % accept) | **207–239 tok/s** (51–63 % accept) | DSV4 1.6–1.8× |
| Decode without speculation, C1 | not measurable (recipe always loads the draft) | 89 tok/s | — |
| C16 batched decode, wall, median | 589 (300 W) / **642 (325 W)** | **1,213** (1M profile) / 1,258 (262K profile) | DSV4 ~2× |
| C16 batched decode, best single run | 642 wall | **1,338 wall** | DSV4 |
| C16 sustained, 15 runs at 325 W | 607 | not run | — |
| Prefill 32K | 4,462 (E14) / 4,429 (325 W) | **8,582 best run**; 8,256–8,376 medians | DSV4 1.9× |
| Prefill 1M | **3.8K tok/s, 261 s** (flat curve) | 1.9K tok/s, 504 s | GLM 2× |
| Needles at 262K / 1M | 6/6 | pass | tie |
| Usable concurrent sequences | ≈ 20 (Mamba state ≈ 4.9 % of KV pool each) | 16 slots sharing 1.18M KV tokens | shape-dependent |
| Weights per card / free for KV | **64.7 GiB / 16.2 GiB** | 83.2 GiB / 5.7 GiB | GLM |
| Effect of 325 W cap | **+9 % C16, +5 % prefill** | 0 % / +1 % | — |
| tok/s per W, C16 decode | 0.98 | **2.02** | DSV4 2× |
| Modalities | text + images (16 per prompt) | text | GLM |

**Verdict:** DSV4-Flash wins throughput about 2× across decode and short prefill and 2× on efficiency; GLM wins long-context prefill (cold 1M prompt in 261 s vs 504 s), VRAM headroom and vision. DSV4-Flash's decode lead is partly DSpark's higher acceptance, which is content-dependent. Both recipes now carry their measured best profiles (GLM: 16 slots, capture derived; DSV4: 16 slots, util 0.9651, capture 96) and the 325 W cap decision is in HARDWARE.md.

**Where V4.1-Flash stands (2026-09-22):** the 2-GPU sglang-kt hybrid decodes at 38.5 tok/s C1 / 147 tok/s C4 and prefills at 216 tok/s — 3.4× below GLM's C1 and 18× below its prefill. See the E19–E21 section below.

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
| E25 DRAM read, C/OpenMP, 16 threads split across both CCDs (quiet host) | GB/s (read) | ≈ 125 (E1b) or higher if E1b was torch-limited | **125.0** (min 123.7 / max 125.4); 4 threads split already 125.2 |
| E25 DRAM read, one CCD (8 physical cores) / (2 cores) | GB/s (read) | ≈ 62 if fabric-bound | **63.1** / **62.1** — per-CCD link limit, direct |
| E25 write / write-nt / copy / copy-nt / triad, 16 threads split | GB/s (bytes moved) | — | 77.8 / 96.8 / 115.1 / 136.4 / 121.7 (each exactly 2× the one-CCD figure) |
| E25 kt-shaped expert reads (random 21.8 MiB chunks, barrier per 1 / 3 / 100 chunks), 16 threads | GB/s | ≈ E17's 127 | 121.0 / 120.2 / 124.8 (2 MiB pages); 4 KiB pages within 1 %; 32 SMT threads **91.8** |
| E25 CPU read + GPU0 / GPU1 / both H2D at once | GB/s | additive (E4b) | 122.0 + 57.0; 124.8 + 56.9; **121.2 + 57.0 + 57.0 = 235** (CPU −3 %) |
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

## E14 — same box, three models: GLM-5.3-Flash (reference) vs V4-Flash vs V4.1-Flash

Measured 2026-09-21 with this harness, same fixtures, same 512-token output, same seeds policy (fresh per model), both GPUs at the 300 W default cap, host otherwise idle. GLM column: the owner's recipe as configured (`~/dev/src/glm-5.3-flash-2x-rtx-pro-6000-blackwell`, EXL3 K3 ≈3.5 bpw target + DFlash2 K5 bf16 draft, tpurtell v0.6 runtime, TP2 + EP2 + DCP2, B12X sparse MLA, fp8 KV, `max_num_seqs` 16, 1,048,576 ctx). V4-Flash column: the sibling vLLM NVFP4 TP2 deployment (E8). V4.1 column: empty until E6 s2 serves. JSON: [`evidence/e14-glm53/`](../evidence/e14-glm53/), [`evidence/baseline-v4-flash/`](../evidence/baseline-v4-flash/).

| Metric | GLM-5.3-Flash EXL3 + DFlash2 | DeepSeek-V4-Flash NVFP4 + DSpark | V4.1-Flash |
|---|---:|---:|---:|
| Weights on disk | 128 GiB target + 2.2 GiB draft | ≈164 GB | 475 GiB (bf16/FP8) |
| VRAM after load, per card | 92.4 GiB (`gpu_memory_utilization` 0.95) | ≈94 GiB by config (0.96; not captured) | — |
| KV pool / context ceiling | 2,963,974 tokens (2.8 × 1M requests) / 1,048,576 | `max_num_seqs` 2 / 1,000,000 | — |
| Tokenizer, same fixture text | +16 % tokens vs V4 (38,169 for the 32K target) | 32,004 for the 32K target | — |
| C1 decode, speculation on, 512 tok | **132.5 tok/s** (DFlash acceptance 40 %; 135 at 256 tok, 42 %) | **207–239 tok/s** (DSpark 51–63 %); 228 on relaunch | 2-GPU hybrid, E23: DSpark K=2 **36.1 wall** / 38.9 summed (accept 0.67), K=5 30.1 / 32.1 (accept 0.38) — **speculation loses on the CPU-bound hybrid**, default off |
| C1 decode, speculation off | not run (recipe requires the draft; see below) | **89 tok/s** | 2-GPU hybrid, E23 R0: **36.0 wall** / 38.7 summed (E21 default D=4, 325 W) |
| Decode C2 / C4 / C8 / C16, wall-clock | 193 / 288 / 424 / **569 tok/s** (summed 203 / 306 / 462 / 629) | 300 / 301 / 277 / — (`max_num_seqs` 2 queues the rest) | — |
| GPU draw during C1 decode (sweep, 300 W cap) | 256 / 259 W per card, SM 2,197 / 2,242 MHz | 233–250 W per card, SM 2,265–2,310 MHz | — |
| GPU draw during C2→C16 decode | **300 / 300 W — pinned at the cap**, SM 1.89–1.91 GHz (580-sample median) | not pinned (C2 summed 300 tok/s at ~250 W) | — |
| Power sweep, C1 decode 300 → 250 W | 119.0 → 126.9 tok/s (acceptance 34 → 40 %; clocks −5 %; pinned at 250 W) | 216 → 226 tok/s (cap-insensitive, 228–251 W uncapped) | — |
| Power sweep, 32K prefill 300 → 250 W | 4,462 → 3,943 tok/s (**−11.6 %**), 1.98 → 1.76 GHz; tok/s/W 7.5 → 7.9 | 8,428 → 7,482 (−11.2 %); tok/s/W 14.0 → 15.0 | — |
| tok/s per W, C1 decode | 0.23 at 300 W, 0.25 at 250 W | 0.46 (0.20 without DSpark) | — |
| Prefill 32K / 128K / 512K target (actual GLM tokens 38K / 153K / 610K) | 4,213 / 4,184 / 4,005 tok/s — **flat** | 7,810 / 6,157 / 3,237 tok/s — length curve | — |
| ~1M prefill | **1,001,032 tok in 261.4 s = 3,829 tok/s** | 972,187 tok in 512.2 s = 1,898 tok/s | — |
| Prefill draw / clocks / peak temp | 300 W, 1.85–1.90 GHz, 89 °C | 300 W, 1.82–1.98 GHz, 88 °C | — |
| 6-needle retrieval, ~256K | **6/6** at 251,490 tok, TTFT 59.7 s | 6/6 at 222K, TTFT 42.4 s | — |
| Tool call / arithmetic / ceiling probes | pass / pass / pass | pass / pass / pass | — |
| Licence | DFlash2 draft **CC BY-NC-ND 4.0** (research only) | MIT model, Apache-2 engine | MIT |

Readout:

- **Single-stream: V4-Flash by 1.6–1.8×.** 207–239 vs 119–132 tok/s. Two causes, both measurable: DSpark accepts 51–63 % of its 5 drafted tokens against DFlash's 34–40 % *on our fixture* (the GLM README's 179.5 tok/s at 65 % used its own code-agent fixture — same lesson as E8: speculative tok/s is a property of the prompt as much as the engine), and GLM's decode step does more SM work per byte: EXL3 trellis weights are decoded on the SMs before each GEMM, so C1 draws 256–259 W (V4-Flash: 233–250) and **any concurrency pins both cards at 300 W** at 1.9 GHz, where V4-Flash never reaches the cap. The 250 W sweep confirms the split: GLM C1 decode is pinned at 250 W with clocks −5 % yet tok/s is unchanged within acceptance noise (34 % → 40 % swung it +7 %), and prefill loses 11.6 % — the same −11 % as V4-Flash. **T4.0 (250 W cap) holds for GLM too**: 0 % C1 decode, −12 % prefill, +8 % tok/s/W.
- *Correction (2026-09-21, after the sweep):* the first draft of this section said "297 W at C1". That figure was the sampler median across the whole C1→C16 run; the sweep isolates C1 and gives 256–259 W. GLM is cap-bound under concurrency, ~10 % hotter than V4-Flash at C1, not cap-bound at C1.
- **Throughput and long context: GLM by 2×.** 569 tok/s wall-clock at C16 against V4-Flash's 300 ceiling, because GLM runs 16 scheduler slots to V4-Flash's 2 (a config choice on the V4 side, T2.4). Prefill is flat at ~4.1K tok/s from 38K to 610K tokens and 3.8K at 1M, where V4-Flash's dense-attention curve falls from 7.8K to 1.9K; the crossover is ≈400K tokens. GLM's B12X sparse MLA is what a V4.1 CSA2 engine has to beat: **the V4.1 prefill target on 2 GPUs is ≥ 3.8K tok/s at 1M.**
- **Same thermal ceiling.** Both models push the cards to 87–89 °C during sustained prefill at 300 W with clocks 1.85–1.90 GHz; the chassis, not the model, sets that.
- **What is missing from the GLM column:** speculation-off C1 (the recipe validates `DFLASH_TOKENS` 1..5 and always passes `--speculative-config`; a hand-launched container without it would give the bandwidth floor) and a like-for-like `max_num_seqs` on the V4 side. The 250 W sweep is done ([`evidence/e14-glm53/power-sweep/summary.json`](../evidence/e14-glm53/power-sweep/summary.json)). The harness's thinking-toggle probe fails on GLM only because it sends V4's template kwarg; GLM's template uses `enable_thinking`.
- **Both recipes run decode under CUDA graphs** (GLM `--max-cudagraph-capture-size 96` = 16 seqs × 6 tokens per speculative step; V4-Flash `MAX_CUDAGRAPH_CAPTURE_SIZE=16`), so neither column pays kernel-launch overhead; the E12 census counted several hundred launches per V4-Flash decode step (236 for the FP8 block-scaled GEMM alone) in a 11–15 ms step. E15 measures what eager mode costs on this box, because the V4.1 engines (SGLang `dev-dsv41` capture crash on SM120 per vLLM #56700; sglang-kt's CPU expert call) may force it.
- **For the recipe's goal:** the box's best daily driver today is GLM for multi-user/long-context work and V4-Flash for single-stream latency. V4.1-Flash on 2 GPUs has to reach ≥ 130 tok/s C1 and ≥ 3.8K tok/s at 1M to displace either — and the hybrid ceiling from FEASIBILITY (50–78 tok/s decode) says it will *not* win single-stream on 2 cards; its case is quality and context, which is why the 3-/4-GPU columns matter.

## E15 + T2.4 slice — V4-Flash at 16 slots, and what eager mode costs (2026-09-21)

Same sibling deployment relaunched three times via `RECIPE_ENV_FILE`, host otherwise idle, 300 W cap, 512-token outputs, `bench-decode.py` 5 runs / 2 warmups (eager+no-DSpark: 3 / 1). JSON in [`evidence/e15-t24-v4-flash/`](../evidence/e15-t24-v4-flash/).

**Launch finding first.** `MAX_NUM_SEQS=16` at `MAX_MODEL_LEN=1000000` refused to start: weights take **83.22 GiB per card**, leaving **5.7 GiB of KV per card** at 0.96 utilisation, and one 1M request needs 5.73 GiB (the 16-slot profile reserves more activation memory, which tipped it under). The sibling's `max_num_seqs 2` is therefore a KV-budget consequence of the 1M profile, not a scheduler preference — the pool is ≈1M tokens total. Relaunched at `MAX_MODEL_LEN=262144`: KV pool 690,615 tokens, 2.63 concurrent 262K requests, capture size 96 (= 16 seqs × 6 tokens, the same ratio as the GLM recipe). GLM's 2.96M-token pool exists because its weights are 64 GiB per card.

| V4-Flash NVFP4 TP2, 262K, 16 slots | C1 | C2 | C4 | C8 | C16 | Power (C1→C16 run) |
|---|---:|---:|---:|---:|---:|---|
| CUDA graphs (capture 96) + DSpark, wall tok/s | **217** | 348 | 558 | 659 | **1,258** | 292–296 W median, pinned at the cap from ~C8; 2.24–2.27 GHz; 83 °C |
| — summed per-sequence tok/s | 217 | 385 | 629 | 891 | 1,506 | |
| — DSpark acceptance | 55 % | 59 % | 61 % | 58 % | 60 % | |
| Eager (`EAGER=1`: no torch.compile, no graphs) + DSpark, wall | **49.5** | 95.8 | 182 | 324 | 618 | **155 / 205 W** — GPU starved by the CPU |
| Eager, no DSpark, wall | **15.9** | — | — | 125 | 249 | 142 / 152 W |
| Reference: graphs, 2 slots, 1M (E8) | 207–239 | 300 | 301 | 277 | — | 233–250 W at C1 |
| Reference: graphs, no DSpark, 2 slots (E8) | 89 | 161 | — | — | — | 216–225 W |

DSpark per-position acceptance from vLLM's `SpecDecoding metrics` (C16 run): **0.85, 0.71, 0.58, 0.48, 0.40**; mean acceptance length ≈ 4.0 tokens per step; accepted 990 tok/s of 1,610 drafted.

Readout:

- **T2.4: the E14 "GLM wins throughput" was the 1M KV budget, not the engine.** At 262K with 16 slots V4-Flash reaches 1,258 tok/s wall at C16, **2.2× GLM's 569** on the same fixture. V4-Flash now leads every decode row; GLM keeps only long-context prefill (3.8K vs 1.9K tok/s at 1M) and the 1M-with-concurrency use case. The cost is the context ceiling: 262K per request instead of 1M. A `V4-Flash-262K-16slot` profile is the box's best multi-user configuration today.
- **E15: eager mode costs 4.4× at C1 and 2× at C16; without speculation 5.6×.** 15.9 tok/s without DSpark is 63 ms per step against 11.2 ms graphed — ≈52 ms of per-step CPU overhead (Python dispatch over thousands of ops plus lost torch.compile fusion; kernel-launch gaps alone would be ~5 ms). The power trace shows it: 142–155 W median with the SMs at 2.3 GHz and nothing to do. **Any V4.1 engine that cannot capture or compile its GPU step is not viable on this box**; the Google-doc recommendation of `--enforce-eager` would cost 4×. For the E6 hybrid design this settles the CPU-expert call pattern: it must live inside a captured graph (host-function node or KTransformers' cpuinfer submit-and-sync), or the GPU portion must at least stay compiled.
- **Speculation amortises a CPU-bound step 3×** (49.5 vs 15.9 tok/s eager). That is a direct preview of E10: the hybrid step will be CPU/DRAM-bound at 15–20 ms, and DSpark's ~4 accepted tokens per verify are how it gets back above 50 tok/s. Acceptance held at 51–60 % in eager mode — the draft's quality does not depend on the step time.
- **K=5 does not hinder V4-Flash.** Position 5 still lands 40 % of the time and verification is nearly free on a bandwidth-bound step, so the fifth token adds ≈0.4 tokens per step for a few hundred microseconds of draft work. A longer draft would likely still pay; a shorter one would lose. K is fixed by the DSpark checkpoint anyway (sibling `.env` note).
- **At concurrency every model is power-bound.** V4-Flash pinned both cards at 300 W from ~C8 (292–296 W median over the C1→C16 run), like GLM from C2. The 250 W cap decision (T4.0) trades a few percent at C8+ for the PSU margin; the C1 case stays free.

### T2.4 result: the 1M / 16-slot profile is qualified, and the DSpark draft graph is not worth it

Follow-up to the 16-slot launch failure above, same day. Three fixes were possible (utilisation 0.9651 as vLLM itself suggested; `max_model_len` 983,040; 8 slots). The first was tried first and worked.

| V4-Flash profile (all TP2, DSpark K5, fp8 KV, graphs) | KV pool | 1M fits? | C1 wall | C4 wall | C16 wall | 1M prefill |
|---|---:|---|---:|---:|---:|---:|
| Shipped: 1M, 2 slots, util 0.96, capture 16 | ≈1M tokens (5.7 GiB) | 1 request | 207–239 | 301 | — | 972,187 tok / 512 s = 1,898 tok/s |
| 262K, 16 slots, util 0.96, capture 96 | 690,615 | no (262K ceiling) | 217 | 558 | **1,258** | n/a |
| **1M, 16 slots, util 0.9651, capture 96** | **1,184,027** | **yes, 1.18×** | 209 | 535 | **1,086** (summed 1,444; run spread 947–1,452) | **972,187 tok / 504 s = 1,927 tok/s**, 300 W, 1.84–1.88 GHz, 88 °C |
| 1M, 16 slots + DSpark draft forward CUDA graph | 1,184,027 | yes | 224 (60 % acc.) | 492 | 1,216 | — |

- **Recommended multi-user profile for the sibling deployment:** `GPU_MEMORY_UTILIZATION=0.9651 MAX_NUM_SEQS=16 MAX_CUDAGRAPH_CAPTURE_SIZE=96` with `MAX_MODEL_LEN=1000000` unchanged. Nothing is lost against the shipped profile — C1 and the 1M prefill are identical — and C16 gains 3.6× over the 2-slot ceiling. The extra 0.51 % of VRAM is exactly the CUDA-graph pool vLLM reserves (0.38 GiB actual). What "16 slots" means for agents: up to 16 requests decode in one batch as long as their contexts sum to ≈1.18M tokens; beyond that vLLM preempts the newest (recompute later) or queues. Sixteen 60K-context sub-agents fit; sixteen 200K ones do not. Prefix caching shares KV for a common system prompt.
- **DSpark draft forward CUDA graph (opt-in prototype, `dspark_forward_cudagraph` + `_allow_tp` in `--speculative-config`; the sibling README lists it as untested):** captured for input shape `(5,)` only, so it engaged at C1 and fell back to the eager draft at C4/C16. At C1 the acceptance-normalised step time is 18.0 ms vs 17.8 ms without it — **neutral**; the C1 tok/s difference (224 vs 209) is the 60 % vs 54 % acceptance swing. The draft forward is a small slice of the step; target verification dominates. Not recommended; recorded so nobody re-tests it.
- **What was measured and left as-is:** `max_num_batched_tokens` 2048 / long-prefill threshold 1792 (protects decode latency under concurrent prefill — untested at 8192); DSpark K (fixed by checkpoint, and the per-position curve says 5 is right); FP8 KV (already on).

### Host CPU: does thread count matter, and are our scripts single-threaded?

Two different questions, both already measured in P1 (`evidence/e1-dram-bandwidth.json`, `e1b-dram-read-bandwidth.json`, `e6-stage0-cpu-expert-gemv.json`, `e4b-dram-contention-cpu-vs-dma.json`):

| Threads on the 9955WX (16 cores / 32 SMT, 2 CCDs) | DRAM read (torch f32 sum) | memcpy-style copy (N processes) | bf16 expert GEMV effective |
|---:|---:|---:|---:|
| 1 | — | 60 GB/s | — |
| 8 | 67 GB/s | 117 GB/s | — |
| 16 | **125 GB/s** | 140 GB/s | **94 GB/s** (76 ms/token) |
| 32 (SMT) | 118 GB/s | 146 GB/s | 78 GB/s (91 ms/token, p90 146) |

- **Multi-core matters and we already use it: 16 threads, one per physical core, is the sweet spot.** 8 threads gets half the bandwidth (one CCD's worth), 32 threads is slightly *worse* than 16 for reads and clearly worse for the GEMV (SMT siblings fight for the same load ports and caches). Every hybrid design number (50–78 tok/s) assumes 16 threads.
- **The DIMMs are not the limit; the CPU's fabric is.** Eight channels of DDR5-6400 are 409.6 GB/s theoretical, and the cores reach 125–146. E4b showed the GPUs can DMA another 114 GB/s from the same DRAM *at the same time* (234 GB/s combined, additivity 0.98), so the memory itself has headroom the cores cannot use — consistent with the 2-CCD Infinity Fabric read limit (~64 GB/s per CCD). That is why option **B2 (split)** in FEASIBILITY exists: serve some DRAM experts from CPU compute and stream others over PCIe, using both read paths. **E25 (2026-09-22) made this direct** — see the next table.

#### E25: the four bandwidth tiers, measured with a C/OpenMP suite (quiet host, no power/thermal limiting)

`scripts/bench-dram-stream.c` + `scripts/bench-dram-suite.py`, 8 GiB buffers, threads pinned per CCD; raw `evidence/e25-dram-revalidation/suite-cpu.json`, `suite-gpu-dma.json`. The E1b torch row above is kept as the historical measurement; E25 reproduces it within 0.3 %. Full method and result in [plan/EXPERIMENTS.md E25](../plan/EXPERIMENTS.md).

| Tier | Value | How known | What could move it |
|---|---:|---|---|
| Theoretical DIMM / channel | **409.6 GB/s** | 8 ch × 8 B × 6,400 MT/s | Memory Frequency (irrelevant while the CPU cannot reach it) |
| Socket-sustainable, all requesters | **≥ 235 GB/s** | CPU 121.2 + GPU0 57.0 + GPU1 57.0 measured concurrently; true ceiling unknown — only the GPUs' DMA engines reach past the CPU | more DMA requesters; DIMM counters (root) would give the ceiling |
| Reachable from this CPU (2 CCDs) | **125 GB/s read**; 78 write; 97 write-nt; 115 copy; 136 copy-nt; 122 triad; **62–63 GB/s per CCD** in every pattern | scaling: ccd0 ×1 39.9, ×2 62.1, ×4 62.8, ×8 63.1; split ×2 105.9, ×4 125.2, ×8 124.5, ×16 125.0; SMT ×32 118.6 | FCLK (per-CCD link; overclocking, BIOS guide Tier 3): 2100 not stable at Auto voltage (Boots 4–5); MEMCLK 6600 at FCLK 2000 → 0 % (Boot 6); **FCLK 2100 + VDDG 1050 mV / VSOC 1.20 V (Boot 7, T3-F1v, candidate): per-CCD 66.4, socket read 132.0, write-nt 102.5, copy-nt 140.4 — +5.5–6 %, C1 +3.1 % (37.19 wall), 0 MCE; at the voltage floor VSOC 1.05 V / VDDG 950 mV (Boots 8–9) the same bandwidth costs ≈ +7 W idle over Auto; **standing profile since 2026-09-23 (slot 3)**; FCLK 2200 reaches GRUB but will not boot the kernel at both allowed rungs — ceiling reached**. Beyond that: more CCDs (H1) or fewer CPU experts (T3.6g) |
| Effective for kt-kernel CPU experts | **≈ 120–127 GB/s** | E17 T=1 cold 1.08 ms/layer = 127; E25 kt-shaped random 21.8 MiB chunks with a barrier per layer 120–125; 32 SMT threads 92 | nothing kt-side — it is at the ceiling; per-layer sync costs ≈ 3 % |

Read scaling in one line: **one CCD saturates at 62 GB/s with two threads; four threads split across the two CCDs already give the socket's 125 GB/s; 8, 12 and 16 threads add nothing; SMT siblings subtract.** Page size (4 KiB vs 2 MiB) is within 1 % for streaming reads. Conditions: cores 2.8–5.2 GHz, socket 87–144 W, Tctl ≤ 57 °C. Consequence for every projection in this repo: none changed (`--dram-gbps 125`, CPU floor 50–55 tok/s, B2 78). The firmware knobs that can and cannot touch these tiers are mapped in [docs/BIOS-OPTIMIZATION-GUIDE.md](BIOS-OPTIMIZATION-GUIDE.md).
- **Our harness scripts are single-process Python with threads for concurrent HTTP streams** (`ThreadPoolExecutor` in `bench-decode.py`); they wait on the network, so the GIL is irrelevant and one core is plenty. The engines are different: vLLM's scheduler/step loop is one Python thread per engine (which is why eager mode collapsed in E15 — the fix is graphs, not cores), and the CPU-expert path in KTransformers/llama.cpp is a native thread pool that will use all 16 cores. For the E6 hybrid the placement rule is: 16 compute threads on 16 cores, the engine's Python loop and the NVMe reader on the SMT siblings, hugepages for the expert arena (T1.6).

### GLM-5.3-Flash: optimisations worth trying on the reference recipe (not run — it is the owner's daily driver)

From the E14 measurements, in order of expected value:

1. ~~**`MAX_NUM_SEQS` 32 with `--max-cudagraph-capture-size 192`**~~ **Tested (E16, below): no gain.** The 2.96M-token pool is attention-only; each request also pins ≈ 4.9 % of it as Mamba state, so ≈ 20 sequences run at once. C32 = 579 wall vs C16 572, TTFT +44 %. Keep 16.
2. ~~**250 W cap.**~~ **Retired (owner, 2026-09-21):** no 4-GPU future-proofing, so run at 300 W and try **325 W** where GLM is cap-pinned (C2+ decode, prefill) — see "Power policy" below. Thermal check first: GLM sat at 86–87 °C pinned at 300 W during E16 C32.
3. **`DFLASH_TOKENS` 4 instead of 5.** Only if the tpurtell runtime logs a per-position acceptance curve: at 34–40 % average acceptance the 5th position is probably < 20 %, and GLM's draft is a bf16 model whose extra token is not free the way DSpark's parallel draft is. Test at C1 on the code fixture; a 5-minute relaunch.
4. **Prefill chunk / `max_num_batched_tokens`.** GLM's prefill is flat at 4.1K tok/s from 38K to 1M and sits at 300 W / 1.9 GHz — compute-bound in the dequant, so chunk tuning will move little. V4-Flash reaches 7.8K at 32K because NVFP4 GEMMs are cheaper; this is a format property, not a tune.
5. **Structural, out of scope for a tune:** the decode ceiling is the EXL3 trellis dequant on the SMs (297 W under any load, 0.23 tok/s/W vs V4-Flash 0.46). A GLM-5.3-Flash checkpoint in a tensor-core-native 4-bit format (NVFP4/MXFP4) with an engine that has a Blackwell grouped GEMM would be the change that beats V4-Flash single-stream; that is a different recipe.

## E16 — GLM-5.3-Flash at 32 slots (2026-09-21)

`ENV_FILE=/tmp/glm-seqs32.env bash scripts/serve.sh` in the GLM repo (only `MAX_NUM_SEQS=32`; capture size now derived = 192), then `bench-decode.py --concurrency 1,16,32 --runs 5 --warmup 2 --max-tokens 512`. Evidence [`evidence/e16-glm-32slots/`](../evidence/e16-glm-32slots/).

| Concurrency | median per-stream tok/s | **wall tok/s** | TTFT (s) | DFlash accept | Engine log |
|---|---|---|---|---|---|
| 1 | 133.3 | 130.1 | 0.100 | 41.6 % | 1 running, KV 4.6–5.3 % |
| 16 | 622.2 | **571.9** | 0.610 | 37.0 % | 16 running, KV 74–83 % |
| 32 | 1,161.5 (summed rates, not throughput) | **578.9** | 0.875 | 37.3 % | **19–21 running, 9–13 waiting, KV 95–100 %** |

- **Slots above ≈ 20 do not run.** vLLM reports 16.19 GiB = 2,833,485 KV tokens ("2.70 × a 1M request"), but that counts attention blocks only. GLM-5.3-Flash is a hybrid; under `--mamba-cache-mode align` each sequence pins a fixed Mamba state ≈ 4.9 % of the pool (~0.8 GiB) whatever its length. 20 × 4.9 % = 98 %. The "median tok/s" column at C32 is the sum of per-stream rates while streams are running and hides the queue; wall tok/s is the honest number and it is flat.
- **Even if they did, the SMs are full.** Both cards pinned at 300–304 W, 1.82–1.85 GHz, **86–87 °C** for the whole C16/C32 phase — the same EXL3-dequant compute ceiling E14 found.
- **Decision:** GLM stays at 16 slots. The GLM `serve.sh` change (capture size derived from `MAX_NUM_SEQS × (DFLASH_TOKENS + 1)`, validator cap 32) is kept because it is identical at 16 and lets the next person test without editing scripts; `config/example.env` documents the ceiling. A larger `GPU_MEMORY_UTILIZATION` (validator caps it at 0.950) would buy ~1 slot per 0.8 GiB — not worth the graph-pool risk E15 showed.
- **Contrast with V4-Flash:** its KV is pure attention (FP8 MLA), so 16 slots at 262K are 16 *running* slots and C16 reached 1,258 wall (2.2 × GLM). Slot count helps the model whose per-slot state is small.

### Power policy (revised 2026-09-21)

Owner decision: no 4-GPU future-proofing. Run both recipes at the **300 W default**; use **325 W** (`power.max_limit`) only where a phase is cap-pinned *and* the card stays under ~85 °C: prefill on both models (+4 % measured on V4-Flash, 8,770 vs 8,428 tok/s at 32K), GLM decode from C2, V4-Flash decode from ~C8. Single-stream V4-Flash decode never sees the cap (216–229 tok/s at every setting) so 325 W is free but useless there. **Thermal is the gate:** under C16 both models already sit at 86–88 °C pinned at 300 W (E16, E14/E15); 325 W adds heat with maybe +4 % clocks. `bench-power-sweep.sh --concurrency 16 --caps 325,300` now measures exactly this and prints `temp_max_c` per row; if it crosses ~88 °C or clocks drop below the 300 W row, stay at 300. The 250 W option is retired — it only ever made sense as a 4 × GPU budget on the 1,500 W PSU.

### T2.9 — 325 W vs 300 W under load, GLM (2026-09-21, owner ran with sudo)

`sudo bash scripts/bench-power-sweep.sh --url http://127.0.0.1:8000 --model glm-5.3-flash-local --caps 325,300 --concurrency 16 --output evidence/t29-glm-325w`. Evidence [`evidence/t29-glm-325w/`](../evidence/t29-glm-325w/).

| Row | 325 W | 300 W | Δ | Busy W (both cards) | SM clock | Max temp | Fan max | Cap-flag samples | Thermal-flag samples |
|---|---|---|---|---|---|---|---|---|---|
| C16 decode, wall tok/s | **642.4** | 588.7 | **+9.1 %** | 647 / 600 | 2,062–2,070 / 1,860–1,875 MHz | 76 / 86 °C † | 44 / 58 % | ~100 % / ~100 % | 0 / 0 |
| 32K prefill, tok/s | **4,428.5** | 4,214.9 | **+5.1 %** | 644 / 600 | 2,002–2,010 / 1,867–1,905 MHz | 83 / 88 °C † | 53 / 67 % | ~100 % / ~100 % | 0 / 0 |
| tok/s per W, decode | 0.993 | 0.981 | +1 % | | | | | | |
| tok/s per W, prefill | 6.88 | 7.02 | −2 % | | | | | | |

† The 325 W rows ran first on cool cards and the 300 W rows second, so the temperatures reflect ordering, not the cap. The sustained check below is the thermal test.

- **325 W is used, not wasted.** The software power-cap flag was set in essentially every busy sample at both caps: GLM is still power-bound at 325 W, and clocks rose 10 % (1.87 → 2.06 GHz) for 8 % more power — the 300 W point sits on a steep part of the V/f curve. Efficiency is unchanged (decode +1 %, prefill −2 % tok/s/W).
- **No thermal event.** Zero samples with `sw_thermal_slowdown` / `hw_thermal_slowdown` at either cap; the firmware fan curve stayed at 44–67 %. The T.Limit margin readout puts the throttle point near 92 °C; the earlier 8-minute 300 W run reached 88–89 °C, so a sustained 325 W run is the number that decides — see below.
- **Decision:** adopt 325 W for GLM (owner policy: max watts where it pays and stays cool). Persist it with [`systemd/gpu-power-limit.service`](../systemd/gpu-power-limit.service); `nvidia-smi -pl` resets on reboot. Pending: sustained-load temperature at 325 W, and the same sweep on V4-Flash once servers are swapped.

### T2.9 — sustained 325 W (GLM) and first V4-Flash sweep (2026-09-21, owner ran)

**GLM sustained:** `--caps 325 --concurrency 16 --runs 15 --phases decode` (~3.5 min pinned). **607 wall tok/s, max 89 °C, fans 73 / 64 %, 0 thermal-throttle samples**, SM 1,957–1,972 MHz (−5 % from the cold 2.06 GHz burst — leakage at 89 °C, the same heat-soak E8 measured at 300 W). Sustained 325 W still beats the cold 300 W burst (589). Margin to the ~92 °C T.Limit point: 3 °C, with 27 % fan headroom. **Adopted;** persist with `systemd/gpu-power-limit.service`. Evidence [`evidence/t29-glm-325w-sustained/`](../evidence/t29-glm-325w-sustained/).

**V4-Flash, 16-slot profile, 325 → 300 W, 3 runs each** ([`evidence/t29-dsv4-325w/`](../evidence/t29-dsv4-325w/)) — medians unreliable, one stalled run per phase in the first (325 W) window of an 8-minute-old server:

| Cap | Prefill 32K per run (tok/s) | median | C16 decode wall per run | median | Busy W | SM MHz | Thermal flags |
|---|---|---|---|---|---|---|---|
| 325 W | 7,544 / **3,623** / 8,582 | 7,544 | 1,242 / 1,279 / **647** | 1,242 | 621–650 | 2,242–2,280 (decode), 2,051–2,055 (prefill) | 0 |
| 300 W | 7,817 / 7,905 / 8,068 | 7,905 | 1,188 / 1,188 / 1,165 | 1,188 | 600 | 2,062–2,096 / 1,905–1,920 | 0 |

Healthy runs say **+6 % prefill (8,582 vs 8,068) and +8 % decode (1,279 vs 1,188)** at 325 W, consistent with GLM and with E8b's single-stream +4 % prefill. Decode at C16 draws 621 W of the 650 available — V4-Flash is only partly cap-bound there (bandwidth-bound in the expert reads), so its decode gain is smaller than GLM's. **Method fix:** `summarize-power.py` now prints `run spread … stalled` when any run is below 70 % of the best and records `tps_per_run`; rerun ordered `--caps 300,325 --runs 5` pending.

### T2.9 — V4-Flash rerun, 300 → 325 W, 5 runs (2026-09-21, owner ran): 325 W is neutral for V4-Flash

[`evidence/t29-dsv4-325w-rerun/`](../evidence/t29-dsv4-325w-rerun/). Prefill rows exclude prefix-cache hits (TTFT < 1 s; the sweep reused the first run's seeds — fixed, seeds are now per invocation).

| | 300 W | 325 W | Δ |
|---|---|---|---|
| C16 decode wall tok/s, 5 runs | 1,213 (1,088–1,338) | 1,206 (1,092–1,302) | **0 %** |
| 32K prefill tok/s, clean runs | 8,256 (n=3: 8,244–8,281) | 8,360 (n=2: 8,343–8,376) | +1.3 % (noise) |
| SM clock, decode | 2,182–2,190 MHz | 2,175–2,205 MHz | same |
| Busy W / max °C / max fan | 600 / 71 / 38 % | 650 / 86 / 57 % | +50 W, +15 °C |
| Thermal-throttle samples | 0 | 0 | |

- **Same clocks, more watts, same speed.** At 325 W the cards drew all 650 W but ran at the same 2.19 GHz as at 300 W: the extra power is leakage on hotter silicon. V4-Flash C16 decode is bandwidth-bound in the NVFP4 expert reads (memory clock is fixed), so the cap is not the limiter even though the cap flag is set. Contrast GLM (+9 %): EXL3 trellis dequant is SM-bound and converts every watt into clock.
- **Efficiency cost:** decode 2.02 → 1.85 tok/s/W (−8 %) for nothing.
- **Decision:** one system-wide cap, **325 W**, because GLM is the daily driver and V4-Flash loses no throughput and shows no thermal event at 86 °C. If V4-Flash becomes primary, `nvidia-smi -pl 300` (or disable `gpu-power-limit.service`) recovers the efficiency. Corrects E8b's "+4 % prefill at 325 W" — that was a 3-run median; with 5 runs and cache hits removed the gain is ≈ 1 %.
- **Lesson for V4.1:** whatever engine serves it will be NVFP4/MXFP4 tensor-core GEMMs like V4-Flash, so expect power caps to be irrelevant to its decode and marginal to its prefill; the levers there are bytes per token, not watts.

## E26 Part B — concurrency on the patched image, and what the DRAM counter says about the CPU tier (2026-09-23)

Protocol in EXPERIMENTS E26. Same server as Part A plus `--cuda-graph-bs-decode 1 2 3 4` on every launch (the derived list from `--cuda-graph-max-bs-decode 4` is `[1, 2, 4]`); 325 W; every C > 1 cell preceded by `verify-batch-consistency.py --same-prompt` at C4 (PASS ×5, gibberish 0.02); 256 tokens, distinct prompts, seed 4242, 3 runs after 1 warmup unless stated. `hsmp-clocks.py --watch 1` sampled the memory controller's own bandwidth counter (all DRAM traffic, 1 s resolution) during the C1 and C4 cells of B2/B3; "busy" = samples above 30 GB/s, i.e. the ones inside the decode run. Evidence `evidence/e26-concurrency/`.

| Cell | `KT_CPUINFER` | D | C | summed tok/s | **wall tok/s** | per stream | TTFT s | DRAM GB/s busy median (p10–p90) | vs default |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 16 | 4 | 1 | 39.07 | **33.94** | 39.1 | 1.03 | — | = R3b 39.2 |
| B1 | 16 | 4 | 2 | 45.60 | **40.82** | 22.8 | 1.26 | — | = R3b 44.9 |
| B1 | 16 | 4 | **3** | **49.00** | **44.97** | 16.3 | 1.47 | — | **+28 % vs 38.3** (bs=3 captured, no padding) |
| B1 | 16 | 4 | 4 | 51.34 | **47.53** | 12.8 | 1.68 | — | = R3b 50.8 |
| B5 | 16 | 4 | 4 (1,024 tok, 2 runs) | 51.43 | **50.31** | 12.9 | 1.80 | — | per-row cost flat with output length |
| B2 | 14 | 4 | 1 | 38.64 | **33.20** | 38.6 | 1.12 | 98 (77–108) | −2 % (plateau) |
| B2 | 14 | 4 | 4 | 49.43 | **45.31** | 12.4 | 1.89 | 106 (95–110) | −5 % |
| B2 | 12 | 4 | 1 | 38.31 | **32.14** | 38.3 | 1.30 | 93.5 (47–102) | −5 % |
| B2 | 12 | 4 | 4 | 45.91 | **41.92** | 11.5 | 2.17 | 99 (78–107) | −12 % |
| B3 | 16 | **0** | 1 | 31.52 | **27.97** | 31.5 | 1.07 | **64.5** (58–88) | −18 % (deferral is worth +21 % wall at C1) |
| B3 | 16 | **0** | 4 | 45.67 | **42.55** | 11.4 | 1.74 | 96 (59–120) | −10 % (deferral is worth +12 % wall at C4) |

B4 (`kt-cpu-per-step.sh`, 200 tokens): step wall 30.0 / 50.5 / 71.0 / 85.5 ms at 1–4 rows; kt CPU-s per step 0.46 / 0.79 / 1.12 / 1.36; **kt threads busy 15.6–15.8 at every C.** The last column is the instrument failing, not a result: kt-kernel workers busy-spin 50 ms before sleeping and no per-layer wait is that long, so all 16 workers burn CPU for the whole step whether they compute or wait — CPU-seconds ≈ 16 × step time and cannot separate compute from serialisation. The DRAM-counter column above is the replacement instrument (B4′).

**What the DRAM counter shows (measured; the interpretation is inferred).** At the default (D=4, 14–16 threads) the memory controller moves ≈ 98 GB/s averaged over the whole C1 step and ≈ 106 GB/s at C4, against the 125 GB/s that this CPU's two CCDs can pull (E25/E27). Bytes per step ≈ 2.5–2.9 GB at C1 (FEASIBILITY's 101 CPU experts × 17.9 MiB ≈ 1.9 GB of expert weights plus Engram, activations, pinned copies) and ≈ 9 GB at C4 — bytes scale ≈ 3.1× for 4 rows, which is what near-zero expert reuse across rows predicts (≈ 2.75 CPU experts per token per layer out of 176). The tier is streaming weights for at least 78 % of the step wall (98 / 125) — it is **bandwidth-bound at bs=1 already**, not core-bound: 12 threads lose 5–12 % *and* pull 5–7 % less bandwidth; 14 → 16 threads is flat in both. D=0 is the control that makes this legible: same bytes, but the counter drops to 64.5 GB/s at C1 because the CPU reads no longer overlap the GPU phase — deferral raises *achieved* bandwidth (64.5 → 98), it does not shrink the byte count. Upper bound for this placement on this CPU: if the tier could read at the 125 GB/s ceiling for 100 % of the step, 2.5 GB/step → 20 ms → ≈ 50 tok/s summed at C1; the honest expectation for further overlap (D=5, scheduling) is well under that. The levers that move bytes or bandwidth — fewer CPU experts (placement T3.6g), more CCDs/GMI links (hardware), faster FCLK (E27: failed at 2100) — are the levers; more cores on the same two CCDs are not. Caveat: the counter is socket-wide (includes GPU DMA of pinned buffers and Engram page-cache reads) and 1 s samples average ≈ 12–33 steps, so the per-step byte figures are ± 15 % and the ≥ 78 % duty figure is a lower bound.

**Quality gates for the new capture list:** same-prompt gate PASS at C2/C3/C4 (min pairwise row prefix 603 / 310 / 155 chars — the C4 figure is below R3's 225 and is flagged as watch-worthy in E26; no bug-C signature); teacher-forced NLL at bs=1 on the D=0 server 1.48644 vs E21's D=0 1.48741, ΔNLL −0.00097 (z +0.27), mean |Δlogp| 0.123 = the known E22 streamed-prefill offset — the bs=1 graph is unchanged. **E24 20K soak on this list: pending** (the default in `serve-hybrid.sh` is provisional until it runs).

Verdict: `CUDA_GRAPH_BS="1 2 3 4"` becomes the launcher default (~~provisional, soak pending~~ qualified by the E27 served soak 2026-09-23, 141K tokens, 0 NaN); `KT_CPUINFER=16`, `KT_MAX_DEFERRED=4` stay. Four agents now get 12.8 tok/s each, three get 16.3, two get 22.8, one gets 39.1 summed / 33.9 wall. The cost model becomes: step ≈ 8 ms + 17.7 ms × rows because each row streams its own ≈ 1.9 GB of expert weights through a ≈ 100–125 GB/s pipe.

## E28 — expert placement: spending the KV pool on GPU experts (2026-09-23)

Protocol in EXPERIMENTS E28 (task T3.6g). Same server as E26 Part B (325 W, patched image, spec off, `KT_CPUINFER=16`, D=4, capture list `1 2 3 4`, 256 tokens, distinct prompts, seed 4242, 3 runs after 1 warmup) with three changes: `--max-total-tokens 147456` (SGLang had sized the KV pool at 1,933,056 tokens for a server that admits 4 × 32K), `MEM_FRACTION_STATIC=0.96`, and `KT_NUM_GPU_EXPERTS` stepped 208 → 222 → 226. GPU experts are the contiguous IDs 0..N−1 of every layer (`kt_ep_wrapper.py`), so each step moves the next IDs off the CPU tier. Every step passed `verify-batch-consistency.py --same-prompt` at C4 (min pairwise row prefix 310 chars, same as E26). Evidence [`evidence/t36g-placement/`](../evidence/t36g-placement/) (`launch-g*.txt`, `decode-g*-c*.json`, `hsmp-g*-c*.json`, `prefill-g*-32k.json`, `nll-g*-d4.json`, `stress-g*-c4-30k.json`, `vram-g*-*.txt`).

| GPU / CPU experts per layer | weights per rank | free before capture | C1 summed / **wall** | C3 | C4 | per stream C1 / C3 / C4 | TTFT C1 | DRAM GB/s busy median C1 / C4 | 32K prefill tok/s / TTFT | free after one 32K prefill (GPU0 / GPU1) | worst case: 4 × 30K decoding, min free GPU0 / GPU1 | ΔNLL vs E22 (z; >0.5-nat worse/better) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 208 / 176 (E26 B1) | 82.97 GB | 6.2 GB (pool 1.93 M tokens) | 39.07 / **33.94** | 49.00 / 44.97 | 51.34 / 47.53 | 39.1 / 16.3 / 12.8 | 1.03 s | 98 / 106 | 870 / 36.7 s | 3.3 GB / 3.2 GB | not measured | baseline 1.48613 |
| 222 / 162 | 87.73 GB (+4.76) | 5.28 GB | 40.53 / **35.39** (+4.3 %) | — | 54.08 / 50.12 (+5.4 %) | 40.5 / — / 13.5 | 0.95 s | 93.5 / 108 | 873 / 36.7 s | 2.8 GB / 2.5 GB | 2,937 MiB / 2,878 MiB (all four finished, 170 s wall) | +0.0055 (z +1.5; 160 / 140) |
| **226 / 158 (new default)** | 89.01 GB (+1.28) | 4.00 GB | **41.23 / 36.08 (+6.3 %)** | **53.04 / 48.72 (+8.3 %)** | **56.18 / 51.91 (+9.2 %)** | 41.2 / 17.7 / 14.0 | 0.91 s | 90 / 108 | 845 / 38.1 s | 1.6 GB / 1.5 GB | **779 MiB / 1,226 MiB** (all four finished, 146 s wall, zero OOM) | **+0.0018 (z −0.48; 143 / 148)** |

What the numbers say:

- **The prediction held.** E26 A4 projected ≈ +4 % C1 wall for 14 experts moved (−8 % of the CPU tier's bytes); 222 measured +4.3 % wall / +3.7 % summed. At 226 (−10.2 % bytes) C1 wall is +6.3 %. At C4 the gain equals the byte reduction (+9.2 % for −10.2 %), which is what a purely bandwidth-bound tier predicts; at C1 it is ≈ 60 % of the byte reduction because ≈ 15 ms of the 29 ms step is GPU work that placement does not touch (E26 A3).
- **The DRAM counter moved the right way.** C1 busy-median fell 98 → 93.5 → 90 GB/s as bytes per step fell faster than step time; C4 stayed at 106–108 GB/s (pipe saturated, so fewer bytes shorten the step directly). Same socket-wide caveats as E26 B4′.
- **Quality is neutral, and the 222 reading shows the noise floor.** 222 read ΔNLL +0.0055 (z +1.5) and 226 read +0.0018 (z −0.48) against the same E22 baseline; moving a tier's worth of experts from kt-kernel's CPU numerics to FlashInfer's MXFP8-activation path shifts individual token log-probs by a mean |Δlogp| ≈ 0.128 (the E22 offset) in both directions, and the mean lands wherever the 4,795-token sample puts it. The strict |Δ| < 0.002 line (T3.9e) is passed by 226 and missed by 222; neither is a one-sided loss (per-token shape symmetric). Against E21's pre-streaming D=4 file the 222 reading is +0.0103 (z −2.59) — that comparison also contains the E22 offset and is not the right one.
- **Where the VRAM went, and where the rest is.** Capping the pool returned ≈ 2.7 GB/rank (pool 3.33 → 0.60 GB); 18 experts cost 6.04 GB/rank (0.34 GB each) — the difference came from raising the fraction. Between "after capture" (93.7 GiB) and the worst-case peak (97.1 GiB on GPU0) sit ≈ 3.4 GB of dynamic allocations: prefill-chunk activations at 30K context, bs=4 decode workspaces, late-loaded Triton kernels (TP1 logged `free device mem: 0.97 GiB` warnings at 226 — none at 222). That 3.4 GB is the next placement budget and it is priced in prefill chunk size (T3.6g2), not in KV.
- **Prefill.** 4K unchanged (869 vs 862 tok/s); 32K 845 vs 870 (−3 %, one 2-run cell — within the run-to-run spread seen in E26 A). TTFT at 30K under four concurrent prefills is 39 / 70 / 100 / 128 s: chunked prefill serialises the four prompts, as expected.

Verdict: **226 GPU / 158 CPU experts is the launcher default** (`KT_NUM_GPU_EXPERTS=226`, `MEM_FRACTION_STATIC=0.96`, `MAX_TOTAL_TOKENS` derived as slots × context + 16K), ~~provisional in the same sense as the capture list — the E24 20K soak has not run on it~~ qualified 2026-09-23 by the E27 served soak on exactly these defaults (EXPERIMENTS E27 "Boot 12 served soak": C1 42.0 / C2 50–52 / C3 55.6–56.0 / C4 58.1–59.0 summed, NLL identical, 0 NaN). Fallback `KT_NUM_GPU_EXPERTS=222` for anyone who wants ≈ 2.9 GB of worst-case headroom instead of 0.8 (the four extra experts cost 1.28 GB of weights but 2.16 GB of worst-case peak on GPU0 — allocator fragmentation and late Triton loads eat the rest, so headroom is not linear in expert count). 228 is projected not to fit (≈ 0.1 GB worst-case free). `CHUNKED_PREFILL_SIZE=8192` (E22's 1.4K tok/s prefill) is incompatible with 226 — it needed ≈ 2.2 GB more than 4096 at 208. Current qualified single-user figure: **C1 41.2 summed / 36.1 wall tok/s, TTFT 0.9 s (short prompt), 845–870 tok/s streamed prefill.** The E26 Part A bar (C1 wall ≥ 37) is now 0.9 tok/s away.

## E29 — placement round 2: prefill chunk 2048 → 230 GPU experts (2026-09-24) — rejected

Protocol in EXPERIMENTS E29 (task T3.6g2). Launcher defaults except the columns shown (slot 3 BIOS, patched image, spec off, D=4, capture list `1 2 3 4`, 300 W — never binds, 512 tokens, distinct prompts, seed 4242, 12 runs). Controls are the E27 soak / E28 D=4 readings at the defaults. The owner's 340 GB `hf download` to the separate `/mnt/data` NVMe ran throughout (≈ 0.6 % C1 per E28's measurement).

| Config (chunk / GPU experts / fraction) | after capture (GPU0 / GPU1) | worst-case min free (4 × 30K) | C1 summed / **wall** / TTFT | C3 | C4 | 16K prefill tok/s / TTFT | 32K prefill | ΔNLL vs E22 (z; better/worse) |
|---|---|---|---|---|---|---|---|---|
| **4096 / 226 / 0.96 (default; control)** | 93,646 / 93,703 MiB | **1,349 / 1,290 MiB** (E28 read 779 / 1,226 for the same config — ≈ 0.5 GB run-to-run) | 41.76 / **39.83** / 0.617 s | 56.00 / 54.38 | 58.09 / 55.79 | 860 / 18.6 s | 845 / 38.1 s | +0.00182 (z −0.48; 143/148) — bit-identical to E28 |
| 2048 / 226 / 0.96 (A) | 93,646 / 93,703 | 2,923 / 2,462 | 42.10 / **40.16** / 0.609 | — | — | **513.9 / 31.2 s (−40 %)** | 518.3 / 61.8 s (−39 %) | +0.00033 (z −0.09; 121/149); repeat on same server bit-identical |
| 2048 / 230 / **0.975** (B; 0.96 refused, "minimum viable 0.9633") | 95,106 / 95,165 | 1,253 / 1,068 | **42.84 / 40.81 / 0.599 (+2.5 % wall vs control, +1.6 % vs A)** | 57.56 / 55.94 (+2.9 %) | 60.00 / 57.64 (+3.3 %) | — | — | **server killed: CUDA OOM** on the logprob request (1.06 GB fp32 logits transient), fresh or after stress |
| 2048 / 230 / 0.975 + `SGLANG_LOGPROB_CHUNK_SIZE=512` (D) | 95,106 / 95,165 | 1,025 / 838 | (as B) | | | 530.9 / 30.2 s | — | **+0.00877 (z −2.23; 141/125) — FAILS the 0.002 gate** |
| 4096 / 226 / 0.96 + `SGLANG_LOGPROB_CHUNK_SIZE=512` (G) | 93,646 / 93,703 | — | — | | | — | — | +0.00182 — **Δ 0.0 vs the default on every token** |

What the numbers say:

- **Chunk 2048 buys the headroom and costs 39–40 % of prefill.** The fit gate gains ≈ 1.6 / 1.2 GB per rank at 226, enough to admit 230 (with the static fraction raised to 0.975). But 16K prefill drops 860 → 514 tok/s and 32K 845 → 518: the per-chunk fixed cost (rank-0 CPU expert sync, Engram gather, Python launch) dominates, not GEMM efficiency. Chunk size is neutral on decode (A vs control within noise).
- **The 4 experts pay what the byte model says, again**: −2.5 % CPU-tier bytes → +1.6 % C1 wall, +2.9 % C3, +3.3 % C4.
- **Break-even (derived, not measured):** the decode gain is ≈ 0.0004 s per output token; a 16K prompt pays +12.6 s of TTFT (≈ 31K output tokens to break even), a 2K prompt +1.5 s (≈ 3.8K tokens). Only prompts under ≈ 1K tokens come out ahead.
- **NLL is deterministic per config**, so tier-swap deltas are real: the same config read the same value to the last digit across days (226/4096: 1.48795 in E28 G2, E29 C, E29 G) and across repeats on one server. The 230 tier moves it +0.0088 — the largest of the four swaps measured (222 +0.0055, 226 +0.0018, 230 +0.0088, all vs the E22 208-expert reference), and the first to fail the gate. Chunk size contributes +0.0003 and the logprob chunk exactly 0.
- **The largest single VRAM transient is a `return_logprob` request**, not the 4 × 30K decode: `logprob_chunk × vocab` fp32 logits = 2048 × 129,280 × 4 B = 1.06 GB plus a bf16 copy. At 230 it kills the server (the scheduler exception calls `kill_process_tree`; E28's note that an OOM fails the request was wrong). `SGLANG_LOGPROB_CHUNK_SIZE=512` cuts it 4× at zero numerical cost and no material latency change (fresh NLL 25.3 vs 28.9 s).
- GPU: 167 / 179 W and 74 / 71 °C peak at chunk 2048; 141–146 W / 69 °C at the default chunk. 0 NaN / Xid / Traceback except the two OOMs; EDAC 0/0.

Verdict: **rejected** — defaults stay `CHUNKED_PREFILL_SIZE=4096`, `KT_NUM_GPU_EXPERTS=226`, `MEM_FRACTION_STATIC=0.96`. Adopted from this round: `SGLANG_LOGPROB_CHUNK_SIZE=512` as a launcher default (bit-identical output, ≈ 0.8 GB of worst-case margin for any logprob caller). Evidence [`evidence/t36g2-placement-r2/`](../evidence/t36g2-placement-r2/).

## Erratum (2026-09-23): every V4.1-hybrid C3 and C4 decode figure from E19 to E23 R0 is an artifact of kt-kernel bug C

**What was wrong.** The unpatched kt-kernel bs=4 decode CUDA graph was baked to a pinned host buffer that the first prefill freed ([TROUBLESHOOTING, bug C and the entry after it](TROUBLESHOOTING.md#c4-decode-drops-from--148-to--50-toks-after-the-bug-c-patch--and-four-identical-greedy-prompts-in-one-batch-return-four-different-answers-on-the-old-build)). It never crashed; it computed CPU experts for effectively one row of the batch and copied stale bytes for rows 1–3. Two consequences: (1) the bs=4 step took as long as a bs=1 step, so C4 looked like 4 × C1 ("near-linear batching") and C3 (padded into the bs=4 graph) like 3 × C1; (2) rows 1–3 of every C3/C4 request since E19 were **wrong** — four identical greedy prompts in one batch returned four different answers on the old build, one of them having lost the prompt. bs=1 and bs=2 were always correct, so every C1 and C2 figure stands, as do prefill, TTFT, NLL (bs=1) and needle results.

**Corrected figures (measured 2026-09-23, patched image `dsv41-kt:recipe`, D=4, speculation off, 325 W, 256 tokens, distinct prompts, 3 runs after 1 warmup; `evidence/e23-spec-decode/r3/decode-patched-c1234.json`, `decode-patched-c4-sameprompt.json`):**

| C | summed tok/s (median) | wall tok/s | per stream | TTFT s | old (artifact) summed | note |
|---|---|---|---|---|---|---|
| 1 | **39.2** | 34.0 | 39.2 | 1.03 | 38.5–38.7 | unchanged |
| 2 | **44.9** | 40.6 | 22.5 | 1.27 | 45.4–47.8 | unchanged — the "bs=2 dip" was the honest row |
| 3 | **38.3** | 35.8 | 12.8 | 1.49 | 98–112 | pads into the bs=4 graph, so it pays four rows for three tokens |
| 4 | **50.8** | 47.0 | 12.7 | 1.69 | 145–148 | same-prompt (full expert reuse) 56.2 / 49.6 — only +11 % |

Step time fits ≈ 8 ms + 17.7 ms × rows. The CPU tier's decode cost is per (token, expert) evaluation — rows that route to the *same* experts save only 11 % — so at M ≤ 4 kt's MXFP4 CPU kernel does not amortise weight reads across rows. Batching buys ≈ +30 % aggregate at C4 for a 3× slower stream. This confirms the E23 verdict (a verify step multiplies rows) and invalidates every projection that multiplied C1 by ≈ 4 for C4: the four-agent router advice, the third-GPU C4 estimate and the "CPU tier is not read-bound" reading of E19 item 3 are marked pending recomputation where they appear. Historical rows below are kept as written with this erratum as the reader's guide; the same-prompt gate `scripts/verify-batch-consistency.py` is now part of the release gate (TASKS standing rules).

## E19–E21 — DeepSeek-V4.1-Flash on 2 GPUs: the sglang-kt hybrid (32K context, 2026-09-22)

> **Erratum 2026-09-23:** the C3 and C4 rows in this table and the "near-linear" readings below are artifacts of kt-kernel bug C — see the Erratum section above. C1, C2, prefill, NLL and needle rows stand.

The first V4.1-Flash that runs on this box. Layout from E6-DESIGN: 208 experts per layer on the GPUs (MXFP4), 176 per layer on the CPU via kt-kernel (16 cores, TP rank 0 only), Engram tables on NVMe. Launch: `bash scripts/serve-hybrid.sh` (container `dsv41-kt`, model name `dsv41-flash-hybrid`), 300 W cap. Same harness as the scoreboard above; decode is 256 tokens, 3 runs, distinct prompts. Full logs in `plan/EXPERIMENTS.md` E19–E21 and `evidence/t36d-first-serve/`, `evidence/e20-*/`, `evidence/e21-kt-fixed-floor/`.

| Metric | E19 first serve (attempt 6) | E20 tuned FP8 GEMM config | E21 deferral D=4 (**current default**) | E22 streamed prefill (`KT_GPU_PREFILL_MIN_TOKENS=1024`, **default since 2026-09-22**; chunk 4096 / 8192) |
|---|---|---|---|---|
| C1 decode, median | 13.98 tok/s | 31.60 tok/s | **38.53 tok/s** | not measured — server asserted before the decode check |
| C2 decode, aggregate | 28.0 | ≈ 38 (19/stream; the bs=2 dip) | 45.44 (22.7/stream; dip still there) | — |
| C3 decode, aggregate | 64.0 (64 tok) | — | 111.71 (soak, 37.2/stream) | — |
| C4 decode, aggregate | 86.7 | ≈ 120 | **146.96** (soak 146.86) | — |
| Per-step at bs=1 | ≈ 71 ms | ≈ 32 ms | ≈ 26 ms | unchanged path (decode stays on the CPU tier) |
| Prefill, 16K prompt | 215.6 tok/s (TTFT 75 s) | not re-run (same CPU path) | not re-run | **837 tok/s (TTFT 19.1 s)**; 32K: 811 tok/s (TTFT 38.4 s); 4.7–5.1 s streaming per 4K chunk |
| Needles, 3/3 at 13,463 tokens | pass, TTFT 57 s | pass, TTFT 74 s | pass, TTFT 57.9 s | pass, TTFT 35.5 s |
| `verify.py` 16K | all pass | all pass | all pass | all pass |
| Quality, teacher-forced NLL (`nll-score.py`) | — | 1.48741 (D=0, the baseline) | ΔNLL −0.0061, paired z +1.44 (neutral) | ΔNLL −0.0013 vs D=0, z +0.35 (neutral); mean \|Δlogp\| 0.125 (GPU-tier MXFP8 activations now compute the streamed experts) |
| GPU power during C1 / C4 decode | 144/159 W / 172/189 W | — | never near the 300 W cap | — |
| GPU temps C1 / C4 | 69/67 °C / 73/72 °C | — | — | — |

- **What moved it.** E19→E20 (2.3×): the SGLang fork's `fp8_gemm` Triton config was an H100 default, 57 % of the step; a SM120-tuned config fixed it. E20→E21 (+22 %): `--kt-max-deferred-experts-per-token 4` lets kt-kernel skip up to 4 of the 176 CPU experts per token per layer when the GPU tier has them; NLL says the quality cost is zero within noise. D=5/D=6 crash the sampler with `probability tensor contains inf/nan` at bs≥4 (D=6 also costs +0.23 nats), so 4 is the ceiling until the NaN is root-caused.
- **What has not moved.** Prefill runs the 176 CPU experts per layer on 16 cores: 216 tok/s, so a 16K prompt is 75 s TTFT and a 32K prompt would be ≈ 2.5 min. That is T3.6b2 (GPU-side layerwise prefill for CPU-resident experts). The bs=2 dip (per-stream throughput lower than bs=1 or bs=3) is open.
- **Against the targets.** E14 asked for ≥ 130 tok/s C1 and ≥ 3.8K tok/s prefill. The hybrid is at 38.5 and 216: 3.4× and 18× short. FEASIBILITY's 50–78 tok/s decode ceiling assumed bandwidth-bound expert reads; the measured step is overhead-bound (40 per-layer CPU-tier waits, Engram graph breaks of 2.4–3 ms/step, rank 0 doing all CPU work at ≈ 1,880 % CPU while rank 1 idles at 120 %). Watts are irrelevant here: the GPUs sit at 150–190 W under a 300 W cap, so the 325 W discussion above does not apply to this recipe.
- **Quality gate.** `scripts/nll-score.py` (teacher-forced NLL on a frozen 4,795-token corpus) is bit-identical across reruns and relaunches; greedy-token agreement is not (near-tie flips), so use NLL, not greedy diffs, to judge any change to the hybrid.

### E22 — streamed prefill (2026-09-22, closed)

`evidence/e22-layerwise-prefill/`. Each ≥ 1,024-token prefill chunk exports the 176 CPU experts of a layer from kt-kernel into a shared pinned buffer (6.5 GiB, POSIX shm, both TP ranks), DMAs each rank's slice (2.27 GiB GPU staging per rank) and runs the FlashInfer SM120 fused MoE on them, layer by layer, double-buffered. Export 51.8 GB/s at 16 threads (core-bound). Per-chunk host time is nearly flat in tokens: 4.7–5.1 s for a 4,096-token chunk, 5.4–5.8 s for 8,192.

| Prefill (tok/s, warm) | E21 CPU path | chunk 4096 (default) | chunk 8192 (`CHUNKED_PREFILL_SIZE=8192`) |
|---|---|---|---|
| 16K prompt | 216 | 837 (TTFT 19.1 s) | **1,375** (TTFT 11.6 s; first post-launch run 547) |
| 32K prompt | ≈ 210 | 811 (TTFT 38.4 s) | **1,404** (TTFT 22.8 s) |
| verify 16K (needles 3/3) | pass | pass, TTFT 35.5 s | pass, TTFT 9.8 s |
| ΔNLL vs `nll-d0.json` | — | −0.0013 (pass) | same kernel; not re-scored |
| free VRAM after a 32K prefill | — | ≈ 3.7 GiB | ≈ 1.2 GiB |

Decode is unchanged with the streamer resident (`decode-streamer-on.json`, 256 tokens × 3 runs): C1 38.41, C2 47.81, C4 144.97 tok/s aggregate vs E21 38.5 / 45.4 / 147. A 16,384 chunk does not fit: the KV pool is only ≈ 3.4 GB (MLA + sliding-window KV), so `MEM_FRACTION_STATIC` cannot free the ≈ 6 GB the activations would need; fewer GPU experts would. The E22c NaN (4-seq batch) was **not** the streamed path — the batch had 664 real tokens (< 1,024; the scheduler's `#new-token` is padded per sequence to 256) and ran the regular D=4 CPU path; it did not reproduce in 20 same-shape batches with a per-layer finite probe (`KT_NAN_PROBE_MIN_TOKENS`); tracked under E24.


### E23 — speculative decoding rounds (closed 2026-09-23: R0, R1 K=2/K=5 measured, R2 as a discriminator; speculation stays off)

> **Erratum 2026-09-23:** the R0 C3/C4 cells (98.2 / 131.9 wall) were measured on the unpatched kt-kernel and are artifacts (Erratum section above); the qualified spec-off C3/C4 are 35.8 / 47.0 wall. The K=5 rows were re-measured with the bug C patch bind-mounted and are real. The K=2 row predates the patch: its C1 and C3 cells used graphs with permanent buffers (3- and 9-token verify shapes) and stand; its **C4 cell (12-token verify shape, the same pre-capture temp mechanism) is unverified** — the wide run spread (59–70 summed) and the acceptance drop to 0.36 at C4 fit corrupted verify rows, so treat 48.3 as an upper bound until re-run on the patched image. The verdict does not change — speculation still loses at C1 (parity or −16 %); against the corrected C3/C4 baseline (35.8 / 47.0 wall) K=2 is roughly at parity at C3/C4 rather than a 60 % loss, and K=5 still loses — but parity is not the ≥ 10 % go bar, and a verify step multiplies rows on a tier whose cost is per row.

Protocol in EXPERIMENTS E23 / TASKS T3.9a–f. Reference row R0 is the E21 default (D=4, speculation off) re-measured with 512-token outputs. Every cell reports wall tok/s (per stream and aggregate), TTFT, ITL p50/p95, accepted tokens per verification, CPU-tier wait per accepted token, host DRAM GB/s, Engram ms/step, GPU W/util — from one JSON per cell under `evidence/e23-spec-decode/<round>/`.

| Round | D | Spec | C1 wall | C3 wall (per stream) | C4 wall (per stream) | acc. tok/verify | CPU wait ms / accepted tok | DRAM GB/s | Engram ms/step | GPU W (per card) | NLL Δ | soak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 4 | off | **36.0** (stream 38.7) | 98.2 (37.2) | 131.9 (37.0) | 1.00 | E21: ≈ 40 waits/step, hidden by D=4 | — | — | — | 0 (E21 D=4: −0.0013) | E22 verify pass |
| R1 | 4 | K=2 | 36.1 (stream 38.9) — 1.00× | 41.9 (15.7) — 0.43× | 48.3 (17.4) — 0.37× | K=2: accept rate 0.67 / 0.58 / 0.36 (len 1.8–2.15 of 3) | not collected (verdict clear from wall) | — | — | max 129 / 148 (GPUs idle behind the CPU tier) | not gated (fails go bar) | 3 runs × 3 cells, zero faults |
| R1 | 4 | K=5 | 30.1 (stream 32.1) — 0.84× | 30.9 (12.9) — 0.31× | 30.0 (9.3) — 0.23× | accept rate 0.38 / 0.38 / 0.35 (len 2.5–2.9 of 6) | not collected | — | — | med 101 / 123, max 112 / 131; GPU0 util 12 %, 67 °C | not gated | first pass crashed at C4 (bug C); rerun after the kt patch: 3 × 3 cells, zero faults, zero Xid |
| R2 | 0 | K=5 (discriminator) | probes only: C1 26.8, C2 26.9, C3 32.3 summed (96–128 tok, debug-sync server) | | crashed identically (CE4 write fault) → deferral not the cause | | | | | | — | off / K=2 at D=0 not run (family already rejected) |

R0 (2026-09-22, `r0-baseline-d4-specoff.json`, server in the E22 chunk-8192 config, TTFT 1.02 / 1.45 / 1.63 s, 512 tokens every stream, `finish_reason=length`): reproduces E21 within +0.5 % (C1 38.5 → 38.7, C4 147 → 148.3 summed). Wall tok/s is 7 % (C1) to 11 % (C4) below the summed stream rate because it includes TTFT and the ramp; that is the number speculation must beat by ≥ 10 % (C1 wall ≥ 39.6).

Fit check first (T3.9a): DSpark's three MTP stages are 7.4 GiB (≈ 3.7 GiB per card at TP2); the E22c server finished capture with 7.1–7.4 GB free and dipped to 0.98 GiB free on TP1 during a streamed prefill, so speculation + streamed prefill + 208 GPU experts may not coexist at 32K. Which lever gives (streamed prefill off, 200 GPU experts, lower `MEM_FRACTION_STATIC`) is recorded with the round.

Reading (2026-09-23): the fit check passed without moving an expert — the draft stages are 3.85 GB/rank and stay GPU-resident, `MEM_FRACTION_STATIC` 0.935 leaves a 258–280K-token KV pool and ≈ 5.6 GB free after capture — so memory was never the constraint; the CPU tier was. Both K rows are below R0 at every concurrency, and the GPU sampler says why: 101–148 W per card and GPU0 at 12 % utilisation while the CPU tier verifies `bs × (K+1)` tokens' worth of distinct experts per layer. Acceptance (0.67 at K=2 C1, 0.38 at K=5) is in the range E8 measured on the 4-GPU V4-Flash (51–63 %), so the drafts are fine; the *target* verify step is what the hybrid cannot afford. Three graph-mode bugs were fixed on the way (TROUBLESHOOTING: A multi-stream overlap, B page-split scratch freed under a graph, C kt temp pinned buffer freed under a graph); the kt fix is `config/kt-kernel-patches/0001-…patch` and is the reason K=5 C4 has a number at all. Files: `r0-baseline-d4-specoff.json`, `r1/k2-d4.json`, `r1/k5-d4.txt` (first pass, C4 crash), `r3/k5-d4-fix.json` (clean K=5), `crash/`, `r3/k5-d0-nofix-c4-segfault.txt`.

## E26 Part A — single-user levers on the patched image (2026-09-23)

Protocol in EXPERIMENTS E26. Server `serve-hybrid.sh` (208 GPU / 176 CPU experts, 32K context, spec off, 325 W, patched `dsv41-kt:recipe`), one environment knob changed per launch; C1 = 3 runs after 1 warmup, 256 tokens, distinct prompts, seed 4242; prefill = 4 runs of a 4,015-token prompt, seed 777, after the server had already served the decode cells (warm). Evidence `evidence/e26-single-user/`.

| Cell | `KT_CPUINFER` | `KT_MAX_DEFERRED` | C1 summed tok/s | **C1 wall tok/s** | C1 TTFT s | 4K prefill tok/s | 4K TTFT s | vs default |
|---|---|---|---|---|---|---|---|---|
| A1 | 14 | 4 | 38.99 | **33.34** | 1.14 | 797 | 5.04 | −1 % (plateau) |
| A1 (default) | 16 | 4 | 38.97 | **33.79** | 1.03 | 862 | 4.65 | — |
| A1 | 24 | 4 | 33.62 | **29.85** | 0.99 | 785 | 5.12 | −12 % |
| A1 | 30 | 4 | 5.04 | **4.86** | 2.07 | 471 | 8.55 | **−86 %** (collapse) |
| A2 | 16 | 0 | 31.51 | **27.96** | 1.07 | 822 | 4.89 | −17 % (D=4 is worth +21 % wall) |

Longer prompts at the default (16 / D=4): 16K **18.9 s TTFT / 845 tok/s**, 32K **36.7 s / 870 tok/s** — flat prefill rate from 4K to 32K (streamed layerwise GPU prefill, E22). Cold-start caveat: the first two 4K prefills after a fresh launch took 9.0 s and 18.2 s before settling at 4.65 s.

Step anatomy at the default (GPU-only profiler trace, 8 steps, categories only — the profiler inflates the span ≈ 1.8×, and a CPU+GPU trace inflates it 6×): **GPU busy 15.4 ms/step** — MoE GPU experts 3.95, dense W8A8 FP8 3.49, NCCL 2.01, BF16 GEMM 1.50, elementwise 1.48, attention indexer 1.29, Engram 0.014. The ten largest idle gaps (8.9–10.1 ms each) all sit between the GPU-expert kernel and the pinned HtoD copy of the CPU tier's result. Against the unprofiled 29.6 ms step this leaves ≈ 14–15 ms/token waiting on the kt tier: the C1 ceiling with 176 CPU experts is the CPU tier, and neither thread count nor the Engram breaks move it.

VRAM at the default: 91,354 MiB/GPU after capture, 94,468 / 94,596 MiB after a 32K prefill (of 97,887) — ≈ 3.3 GiB free per rank worst case, with a KV pool sized for 1.93 M tokens where the served config (4 × 32K) needs 131K.

Verdict: `KT_CPUINFER=16`, `KT_MAX_DEFERRED=4` stay; the +10 % Part A bar (C1 wall ≥ 37) is not reachable from environment knobs alone — placement (T3.6g), D=5 (T3.15) and FCLK (E27) each project ≈ +2–4 %.

### T3.15 — deferral D=0 / 4 / 5 on the qualified defaults (2026-09-23 evening, E24 lead result)

Same launcher defaults as the E27 Boot 12 soak (226 GPU / 158 CPU experts, capture `1 2 3 4`, spec off, FCLK 2100 slot 3, **300 W**, patched image), `KT_NAN_PROBE_MIN_TOKENS=1`, one launch per D; C1 = 12 runs × 512 tokens, distinct prompts, seed 4242, fixture offset 6; C2–C4 at D=5 with the soak's offsets. The owner's 340 GB download to the other NVMe ran throughout (it cost D=4 C1 ≈ 0.6 % vs the download-free soak), so the D column is apples-to-apples and the C2–C4 comparison to the soak carries that bias. Evidence [`evidence/e24-nan/t315-d5/`](../evidence/e24-nan/t315-d5/).

| `KT_MAX_DEFERRED` | C1 summed / **wall** tok/s | C1 TTFT s | C2 | C3 | C4 | ΔNLL vs E22 (gate < 0.002) | greedy bs=1 same-prompt, chars before divergence (gate ≥ 40) | NaN / Xid |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 33.58 / **32.04** | 0.74 | — | — | — | — (E21: neutral) | 388; repeats 363 / 323 / 323 | 0 / 0 |
| **4 (default)** | 41.76 / **39.83** | 0.62 | soak: 52.12 / 49.80 | 56.00 / 54.38 | 58.09 / 55.79 | +0.0018 (z −0.48) | 310–364; repeats 362 / 225 / 225 | 0 / 0 |
| 5 | 43.76 / **41.69** (+4.7 % wall) | 0.60 | 52.94 / 50.54 (+1.5 %) | 56.23 / 54.61 (+0.4 %) | 57.77 / 55.52 (−0.5 %) | **+0.00287** (z −0.74; 150 worse / 156 better) | **9–39**; repeats 9 / 33 / 9; C4 same-prompt rows agree for 9 chars | **0 / 0** over 63K tokens incl. the E21 trigger (first bs=4 step) |

Reading: D=5 no longer crashes (the E21 D≥5 assert was kt-kernel bug C) and clears the +3–5 % C1 bar, but it fails both quality gates. The consistency column is the one to understand: bs=1 greedy output is not run-to-run exact at *any* D (E21 saw this and moved the quality gate to teacher-forced NLL), because the serving path flips near-tie tokens; deferral pulls the first flip forward — 320–390 → 225–365 → 9–39 chars — and at D=5 the first sentence itself changes, so the +0.00287 is a real shift in the distribution, not noise. `--flush-cache` (new in `verify-batch-consistency.py`) rules the radix cache out; a librarian read of kt-kernel's `task_queue.cpp`/`moe_base.hpp` rules the deferred queue out (single FIFO dispatcher, ordered reductions). Decision: **`KT_MAX_DEFERRED=4` stays**; D=5 is a rejected lever, and the remaining single-user lever with the same coefficient is placement (E28 / T3.6g2). GPU peaks 167 / 178 W, 71 / 69 °C at 300 W — the cap never binds on the hybrid.

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

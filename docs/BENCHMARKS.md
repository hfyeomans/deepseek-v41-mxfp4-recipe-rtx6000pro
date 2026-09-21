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

## E14 — same box, three models: GLM-5.3-Flash (reference) vs V4-Flash vs V4.1-Flash

Measured 2026-09-21 with this harness, same fixtures, same 512-token output, same seeds policy (fresh per model), both GPUs at the 300 W default cap, host otherwise idle. GLM column: the owner's recipe as configured (`~/dev/src/glm-5.3-flash-2x-rtx-pro-6000-blackwell`, EXL3 K3 ≈3.5 bpw target + DFlash2 K5 bf16 draft, tpurtell v0.6 runtime, TP2 + EP2 + DCP2, B12X sparse MLA, fp8 KV, `max_num_seqs` 16, 1,048,576 ctx). V4-Flash column: the sibling vLLM NVFP4 TP2 deployment (E8). V4.1 column: empty until E6 s2 serves. JSON: [`evidence/e14-glm53/`](../evidence/e14-glm53/), [`evidence/baseline-v4-flash/`](../evidence/baseline-v4-flash/).

| Metric | GLM-5.3-Flash EXL3 + DFlash2 | DeepSeek-V4-Flash NVFP4 + DSpark | V4.1-Flash |
|---|---:|---:|---:|
| Weights on disk | 128 GiB target + 2.2 GiB draft | ≈164 GB | 475 GiB (bf16/FP8) |
| VRAM after load, per card | 92.4 GiB (`gpu_memory_utilization` 0.95) | ≈94 GiB by config (0.96; not captured) | — |
| KV pool / context ceiling | 2,963,974 tokens (2.8 × 1M requests) / 1,048,576 | `max_num_seqs` 2 / 1,000,000 | — |
| Tokenizer, same fixture text | +16 % tokens vs V4 (38,169 for the 32K target) | 32,004 for the 32K target | — |
| C1 decode, speculation on, 512 tok | **132.5 tok/s** (DFlash acceptance 40 %; 135 at 256 tok, 42 %) | **207–239 tok/s** (DSpark 51–63 %); 228 on relaunch | — |
| C1 decode, speculation off | not run (recipe requires the draft; see below) | **89 tok/s** | — |
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
- **The DIMMs are not the limit; the CPU's fabric is.** Eight channels of DDR5-6400 are 409.6 GB/s theoretical, and the cores reach 125–146. E4b showed the GPUs can DMA another 114 GB/s from the same DRAM *at the same time* (234 GB/s combined, additivity 0.98), so the memory itself has headroom the cores cannot use — consistent with the 2-CCD Infinity Fabric read limit (~64 GB/s per CCD). That is why option **B2 (split)** in FEASIBILITY exists: serve some DRAM experts from CPU compute and stream others over PCIe, using both read paths. A direct confirmation (STREAM in C or `pcm-memory` DIMM counters) is an open item in ROADMAP §measurements.
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

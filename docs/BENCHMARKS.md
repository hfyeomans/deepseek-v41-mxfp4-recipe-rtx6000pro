# Benchmarks

Local observations for immutable profiles on this host. Nothing here is a universal performance claim. Power condition for every GPU measurement on this machine: RTX PRO 6000 Blackwell **Max-Q at 300 W** (325 W maximum). Numbers from 600 W cards or other hosts are cited as references, never mixed into these tables.

## Method (shared by every table)

- Decode: [`scripts/bench-decode.py`](../scripts/bench-decode.py). Fixed code-review fixture, temperature 0.2, fixed seed, thinking off, 256 forced output tokens per sequence, 2 warmups then 5 measured runs per concurrency level. Each sequence is timed from its own first streamed token to its last; aggregate = sum of per-sequence rates; TTFT excluded and reported separately. Speculative acceptance = Prometheus counter delta per run.
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
| E2 GDDR7 D2D per card | GB/s | 1,450–1,650 | — |
| E3 Engram gather DRAM, 16 threads, 20 M-row table (4.9 GiB) | rows/s; 48-row p50 / p99 | ≥ 5 M; < 0.1 ms | **4.15 M**; 0.011 / 0.015 ms |
| E3 Engram gather Samsung 9100 PRO (O_DIRECT 4 KiB, 16 threads) | rows/s; 48-row p50 / p99 | ≥ 600K; < 1 ms | **403K**; 1.90 / 2.11 ms |
| E3 Engram gather Crucial T705 (O_DIRECT 4 KiB, 16 threads) | rows/s; 48-row p50 / p99 | lower than 9100 | **347K**; 2.20 / 2.38 ms |
| E4 PCIe H2D pinned, per card | GiB/s | 48–55 | — |
| E4 PCIe H2D pinned, both cards concurrently | GiB/s aggregate | ≥ 90 | — |
| Option B decode ceiling (E4 aggregate ÷ 1.64 GiB/token) | tok/s | ~60 | — |

How to read E1 and E3:

- E1 is a Python `memoryview` memcpy in N processes (threads measured one core: the first version of the script held the GIL and reported a flat 61 GB/s at every thread count; that number is not in the table). 146 GB/s is 36 % of the 409.6 GB/s theoretical for 8 × DDR5-6400 and below the 250–320 GB/s hypothesis. It is a lower bound (no non-temporal stores, no hugepages, 4 KiB-page-faulted buffers); a STREAM run inside a container is the follow-up if the number ever matters. It does not matter for the verdict: no option in FEASIBILITY.md is DRAM-bandwidth-bound.
- E3 DRAM: 4.15 M rows/s from Python is 86K decode-token-equivalents per second and puts a 64K-token prefill's 3.1 M Engram lookups at ≈ 0.75 s. Even this unoptimised path is two orders of magnitude faster than decode needs. **Engram-in-DRAM is not a bottleneck on this host.**
- E3 NVMe: the harness does the 48 reads of one token *serially* on one thread, so the 1.90 ms p50 is 48 × ≈ 40 µs, i.e. the drive's unloaded 4 KiB O_DIRECT latency at queue depth 16. The `< 1 ms` pass criterion is therefore about the harness, not the drive: a native path issuing the 48 reads concurrently would see ≈ 40–60 µs per token. Throughput of 403K IOPS at QD 16 is consistent with the 2.2 M IOPS spec at QD ≥ 256. **Engram-from-NVMe is viable for decode on either drive** (8.4K token-equivalents/s on the 9100 PRO), and costs ≈ 7.7 s of lookups per 64K-token prefill versus 0.75 s from DRAM. That gap is why the 4-GPU profile puts Engram in DRAM, which this host can do and the 128 GB reference could not.

## E8 baseline: DeepSeek-V4-Flash-0731 NVFP4, TP2, 2 × RTX PRO 6000 Max-Q

Reference values from the sibling recipe (same host, different harness), to be re-measured with this harness:

| Metric | Sibling recipe | This harness |
|---|---:|---:|
| Single-stream decode, no DSpark | 93.66 tok/s | — |
| Single-stream decode, DSpark K5 | 200.64 tok/s | — |
| Aggregate decode C2 | 226 tok/s | — |
| 998,869-token prompt prefill | 513.8 s (≈ 1,944 tok/s) | — |
| Context ceiling | 1,048,576 | — |

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

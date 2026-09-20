# Experiments

Each experiment states the question, the hypothesis with a number attached, the exact procedure, what counts as pass, and where the result lands. Negative results are results. All output JSON goes to `results/` (git-ignored) and the summarised numbers move into `docs/BENCHMARKS.md` by hand with the JSON committed under `evidence/` once reviewed for identifiers.

Legend: **GPU-free** experiments can run while the GLM container is serving. **GPU** experiments need `docker stop glm53-flash-2x-rtxpro6000` first and `docker start` afterwards.

## E1 — Host DRAM bandwidth (GPU-free)

- Question: how much of the 409.6 GB/s theoretical does the 16-core 9955WX reach?
- Hypothesis: 250–320 GB/s copy bandwidth with 16–32 workers; a single worker ≈ 20–30 GB/s.
- Procedure: `scripts/bench-dram.py --workers 1,2,4,8,16,32 --size-gib 8 --output results/e1.json` (stdlib `mmap` memcpy via `memoryview` slicing in N *processes*; threads hold the GIL and measure one core). STREAM inside a container is the optional follow-up.
- Pass: a measured number replaces the "estimate" row in HARDWARE.md. There is no fail.
- Feeds: Option B DRAM-side check; Engram-in-DRAM prefill estimate.
- **Result (run):** 60.0 GB/s single process, 139.9 GB/s at 16, 146.2 GB/s at 32 (read+write). Below hypothesis; a Python lower bound. Nothing in the verdict depends on it. [`evidence/e1-dram-bandwidth.json`](../evidence/e1-dram-bandwidth.json).

## E2 — GPU memory bandwidth (GPU)

- Question: achieved GDDR7 bandwidth per RTX PRO 6000 Max-Q at 300 W.
- Hypothesis: 1,450–1,650 GB/s device-to-device copy; well below 1,792 GB/s theoretical.
- Procedure: `scripts/bench-pcie-h2d.py --d2d` inside any torch-bearing image on this host (e.g. the `dsv4-sm120` images), 1–8 GiB buffers, 20 iterations, median.
- Pass: number recorded. Used for the decode roofline on the 4-GPU profile (weights per GPU ÷ bandwidth ≈ per-token floor).

## E3 — Engram-style random row gather: DRAM vs NVMe (GPU-free)

- Question: can either drive or DRAM serve Engram lookups at decode and prefill rates?
- Hypothesis: DRAM ≥ 5 M rows/s single-process; Samsung 9100 PRO ≥ 600K random 4 KiB reads/s with O_DIRECT at QD 32 from Python threads (well below the 2.2 M IOPS spec because of Python overhead; document that as a lower bound); T705 lower.
- Procedure: `scripts/bench-engram-gather.py --backend dram --rows 20000000 --output results/e3-dram.json`; `--backend nvme --path results/engram-nvme0.bin` (nvme0, root fs) and `--path /mnt/data/dsv41-engram-bench.bin` (nvme1). Synthetic 20 M × 264 B table (4.9 GiB); no model download. Report rows/s, p50/p99 latency per 48-row token batch.
- Pass: decode batch (48 rows) p99 < 1 ms on both backends; prefill 3.1 M rows in < 10 s on DRAM. Failure on NVMe means Option A must use Engram-in-DRAM mode, which this host can.
- Feeds: HARDWARE.md storage section; choice of `OFFLOAD_MODE` in the 4-GPU profile.
- **Result (run):** DRAM 4.15 M rows/s, batch p99 0.015 ms, 64K-prefill equivalent 0.75 s — pass. Samsung 9100 PRO 403K rows/s, batch p50 1.90 ms; Crucial T705 347K rows/s, p50 2.20 ms — the `< 1 ms` criterion **fails as measured**, but the harness issues a token's 48 reads serially, so the batch latency is 48 × the drive's ≈ 40–46 µs unloaded read latency, not a queueing limit. Corrected reading: both drives sustain decode-rate lookups (≥ 7K token-equivalents/s); NVMe costs ≈ 7.7–8.9 s of lookups per 64K-token prefill versus 0.75 s in DRAM. Decision: the 4-GPU profile keeps `OFFLOAD_MODE=engram-host` (DRAM); NVMe is the fallback if DRAM is needed for something else. Evidence: [`e3-engram-gather-dram.json`](../evidence/e3-engram-gather-dram.json), [`e3-engram-gather-nvme0.json`](../evidence/e3-engram-gather-nvme0.json), [`e3-engram-gather-nvme1.json`](../evidence/e3-engram-gather-nvme1.json).

## E4 — PCIe host-to-device bandwidth per GPU (GPU)

- Question: sustained pinned H2D and D2H bandwidth per card, and both cards concurrently.
- Hypothesis: 48–55 GiB/s per card pinned; both concurrently ≥ 90 GiB/s aggregate (separate x16 links); pageable ≈ half.
- Procedure: `scripts/bench-pcie-h2d.py --sizes 256M,1G,4G --pinned --both` inside a torch image.
- Pass: numbers recorded. **This is the input that recomputes the Option B ceiling**: decode tok/s ≤ measured aggregate GiB/s ÷ 1.64 GiB/token.

## E5 — V4.1-Flash on 2 GPUs with generic CPU offload (GPU, needs 475 GiB download)

- Question: what exactly happens when the two available engines are asked to do it?
- Hypothesis: SGLang `dev-dsv41` with `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 --tp 2` OOMs during weight load at ~140 GiB/GPU demand. vLLM `--cpu-offload-gb 110` either fails to construct the deepseek_v41 loader on SM120 or produces < 5 tok/s because it streams all offloaded parameters every step.
- Procedure: `scripts/download-model.sh` (pinned), then `ALLOW_UNQUALIFIED=1 scripts/serve.sh --profile 2gpu-attempt` with each engine; capture the log tail (scrubbed) into `results/e5-*.log`.
- Pass criterion is documentation: the failure mode is recorded verbatim in TROUBLESHOOTING.md. If it unexpectedly serves, run `verify.py` and `bench-decode.py` and treat the number as the new 2-GPU result.

## E6 — Expert-streaming prototype (GPU, synthetic first)

- Question: given E4's bandwidth, what decode rate can a 39%-offloaded expert set sustain?
- Hypothesis: synthetic loop (240 expert activations/token, 17.9 MiB each, 39% from pinned DRAM, per-step prefetch overlapped with a dummy GEMM) reaches 45–60 tok/s at batch 1; 2–4K tok/s prefill-equivalent for 4096-token chunks.
- Procedure: stage 1 `scripts/bench-expert-stream.py` (torch, synthetic); stage 2, only if stage 1 > 40 tok/s, an SGLang patch that keeps per-layer expert subsets host-resident and gathers by routing indices. Stage 2 is a separate design document before code.
- Pass: stage 1 number recorded either way. Stage 2 proceeds only if stage 1 ≥ 40 tok/s and the owner judges 40–60 tok/s worth having over the 200 tok/s V4-Flash baseline.

## E7 — Routing skew study for expert pruning (GPU, uses the V4-Flash baseline deployment)

- Question: is routing uniform enough that pruning 39% of experts would destroy quality, or skewed enough that a pruned checkpoint is plausible?
- Hypothesis: `noaux_tc` load balancing makes per-expert activation frequency within ±30% of uniform on general text; pruning is not viable without retraining.
- Procedure: enable expert-load logging on the V4-Flash baseline (same MoE family, 256 experts) over a 5 M-token mixed corpus; compute per-layer activation histograms and the cumulative share of the top-61% experts.
- Pass: if the top 61% of experts capture ≥ 90% of activations on ≥ 80% of layers, Option D graduates to a build task; otherwise it is closed with the histogram as evidence.

## E8 — Baseline re-measurement (GPU, existing V4-Flash deployment)

- Question: what does the qualified V4-Flash-0731 NVFP4 TP2 deployment do under *this* harness?
- Hypothesis: within ±5% of the numbers in the sibling repo (200 tok/s single-stream with DSpark K5, ~226 tok/s at C2, ~2K tok/s prefill).
- Procedure: `scripts/bench-decode.py --concurrency 1,2,4,8 --runs 5 --warmup 2`, `scripts/bench-prefill.py --tokens 8192,32768,131072,524288,1000000`, `scripts/verify.py --needles 6 --context 1000000`.
- Pass: tables in BENCHMARKS.md; JSON in `evidence/baseline-v4-flash/`.
- **E8b — power sweep (same window).** The host's PSU is 1,500 W; four Max-Q cards at 300 W do not fit it (HARDWARE.md power table). Repeat the C1 decode and 32K prefill rows at `nvidia-smi -pl 300`, `275`, `250`, `225` on both cards, logging `power.draw` and, if a wall meter is available, whole-system draw. Hypothesis: decode (bandwidth-bound) loses < 10 % from 300 → 225 W; prefill (compute-bound) loses 15–25 %. Output: the tok/s-per-watt curve that decides between "cap four cards at 225 W on the 1,500 W PSU" and "buy a 2,000 W PSU" for Option A. Restore `-pl 300` afterwards.

## E9 — 4-GPU V4.1-Flash qualification (blocked on hardware)

- Question: does the SGLang `dev-dsv41` + Engram-in-DRAM profile serve V4.1-Flash on 4 × RTX PRO 6000 Max-Q, and how does it compare with 0xSero's NVMe-Engram numbers and with E8?
- Hypothesis: 180–230 tok/s at C1 with DSpark 5, 600–750 tok/s at C8, 5–7K tok/s prefill, DSpark acceptance 75–88%; Engram-in-DRAM improves prefill TTFT on long prompts versus NVMe mode by a measurable margin.
- Procedure: `preflight.sh` (must pass on 4 GPUs), `serve.sh --profile 4gpu-sglang`, `wait-ready.py`, `verify.py`, E8's benchmark set, then the same with `OFFLOAD_MODE=nvme` for the A/B.
- Pass: verify 6/6 needles at 262,144 and at the configured maximum; `17*19=323` sanity; tables recorded with power condition (300 W Max-Q).

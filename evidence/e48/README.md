# E48 — one C1 decode step, rank by rank: where the rank-0/rank-1 dependency costs (2026-10-23, 1017 default)

Serve: PROFILE=developer, KT_TIMELINE=1 (per-layer device-event recorder), torch-profiler traces for
both ranks, HSMP DDR during C1. C1 during the window: 58.6-58.8 tok/s (16.46 ms/step, unperturbed
device events; the torch-profiler span reads 46.4 ms - the known profiler slowdown - so the ratios
come from KT_TIMELINE and the bench).

## The step, by rank (16.46 ms/step, KT_TIMELINE buckets)

| bucket | rank 0 | rank 1 |
|---|---|---|
| export (kt D2H + submit host node) | **2.274 ms (13.8 %)** | 0.024 ms |
| gpu_experts | 2.947 ms (17.9 %) | 2.836 ms (17.3 %) |
| **cpu_wait (exposed kt wait)** | **1.626 ms (9.9 %)** | 0.018 ms |
| h2d / combine | 0.244 / 0.231 ms | 0.017 / 0.024 ms |
| engram_break | 0.774 ms (4.7 %) | 0.719 ms (4.4 %) |
| other_gpu (attention, dense, norms, all-reduce, lm head) | 7.990 ms (48.5 %) | **12.542 ms (76.3 %)** |
| between_replays | 0.237 ms | 0.218 ms |

Torch-profiler category view (perturbed, ratios only): rank 0 GPU busy 14.2 ms with 32.2 ms idle
in ~24 gaps >= 100 us (the per-layer kt waits); rank 1 GPU busy 29.2 ms of which **collective
17.05 ms** vs rank 0's 0.66 ms - rank 1 spends its step in the all-reduce/spin while rank 0 alone
dispatches the CPU tier. HSMP DDR during C1: **87 / 410 GB/s peak (21 %)** - the CPU tier has
headroom to split (pre-map it sat at ~98 of 125 GB/s).

## Verdict: the T3.22 reopen trigger is MET

The measured kt submit+wait share on rank 0 is **export 2.274 + cpu_wait 1.626 = 3.90 ms/step
(23.7 %)** against the T3.22 decision's 1.5 ms/step reopen trigger. A rank split that overlaps the
two ranks' kt work and halves the export has a ceiling of ~3.9 ms of 16.5 (up to ~24 % C1 at
perfect overlap; the CPU-tier compute is per-layer and has the 87/410 GB/s headroom to run both
halves concurrently). The blocker is unchanged: kt-kernel dispatches on rank 0 only (single
threadpool; the wrapper constructs on rank 0 and other ranks return zeros - RESEARCH.md "TP2 does
not duplicate CPU work"), so a per-rank dispatch is a kt-kernel/SGLang feature, not an overlay
knob. This is the T3.57 design input; the other_gpu buckets (rank 1 12.54 ms vs rank 0 7.99 ms)
are where the rest of the step goes and are the T4.1b/hardware territory.

## Files

- traces: `/tmp/e48-prof/*.trace.json.gz` (both ranks; not committed - profiler scratch).
- timeline: `/tmp/e48-timeline/timeline-rank*.jsonl` (not committed; the report above is the
  distilled record; `scripts/kt-timeline-report.py` reproduces it).

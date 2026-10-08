# E34d-iv — prefill-chunk sweep above 8192 (2026-10-08)

Overlay `d6811728` (unchanged), image `dsv41-kt:recipe`, BIOS slot 3 (FCLK 2100), 300 W,
`CONTEXT_LENGTH=65536 MAX_RUNNING_REQUESTS=4 KT_NAN_PROBE_MIN_TOKENS=1 ALLOW_UNTESTED_FIT=1`, D=4,
spec off. Reference = the 2026-10-08 default, 219 / 165 + chunk 8192 (`evidence/t328/default-219-8192/`):
prefill 1,504 tok/s at 16K (TTFT 10.7 s), 1,526 at 32K (E34d-ii), C1 41.34 median / 39.39 wall,
5.56 / 5.50 GB free after capture.

Owner framing: 8192 was never meant as a ceiling; find the optimum, higher or lower. Winner =
best 16K-prefill gain per 1 % of C1 wall given up, with the 4 × 62K stress margin ≥ 1 GB on both ranks.

| dir | config | result |
|---|---|---|
| `c16384/` | 219 / 165, chunk 16384 | ready at 5.25 / 5.19 GB free after capture; **OOM on the first 16K prefill**: `cutlass_fused_moe` workspace request **1.86 GiB** (0.98 GiB at 8192 — scales with chunk tokens) against 631 / 572 MiB free, 1.07 GiB reserved-but-fragmented. `launch-container-219-OOM.txt`. |
| `c16384-212/` | 212 / 172, chunk 16384 | 7.52 / 7.46 GB free after capture; **OOM again** at the same request (1.87 GiB vs 1.50 / 1.44 GiB free) — the other chunk-proportional transients take ~6 GB before the workspace, so the 16K-prefill peak is ~8 GB. `launch-container-OOM.txt`. |
| `c16384-205/` | 205 / 179, chunk 16384 (measurement only) | 9.95 / 9.89 GB free after capture; **prefill 1,838 tok/s at 16K (TTFT 8.7 s; +22 %), 1,701 at 32K (TTFT 18.8 s; +11 %)**; **C1 39.56 median / 37.66 wall (−4.3 % / −4.4 % vs the default; −6 % wall vs the pre-10-08 226 / 4096 line)**; health 0, EDAC 0/0. NLL and stress not run (rejected on the ratio). |

## Decision

**Chunk 8192 stays the default.** Gain-per-cost: 4096→8192 bought +45 % prefill for −1 % C1 wall
(ratio 45); 8192→16384 buys +22 % for −4.4 % (ratio 5) because the extra ~4.5 GB of prefill
transient must come out of resident experts (14 of them, 0.34 GB each per rank). 12288 was not
run: projected peak ~6.7 GB does not fit at 219 (5.56 GB free) and leaves < 1 GB of stress margin
at 212, for a projected +10–12 % at −2 % — the same ratio as 16384, not competitive with 8192.
Below 8192 is closed by earlier evidence (2048 rejected, 4096 is the measured floor).

What the sweep did establish: the chunk-16384 gain is real (+22 % at 16K, +11 % at 32K), so it
becomes worth having only if the per-chunk transient stops scaling with the chunk — i.e. a
memory fix, not a placement trade. Registered as E34d-v (conditional; see EXPERIMENTS).

## Files

- `*/prefill-*.json` — `bench-prefill.py` output; `c16384-205/c1.json` — `bench-decode.py` C1.
- `*/launch-container*.txt` — container logs (OOM tracebacks for the two failed serves; health grep 0 for the third).

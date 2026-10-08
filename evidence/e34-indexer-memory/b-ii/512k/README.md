# E33/E34 ladder — 512K single-slot rung (2026-10-08)

Overlay `d6811728` (unchanged since E34b-ii), 226 / 158, D=4, spec off, chunk 4096,
`CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1` (launcher derives `MAX_TOTAL_TOKENS` = 540,672).
Two serves. **Result: 512K single-slot passes every gate at `MEM_FRACTION_STATIC=0.962`; at the
default 0.96 the KV-pool sizer silently caps the pool at 505,856 tokens and the server rejects
inputs over 505,850 with HTTP 400.** No OOM in either serve.

## Serve 1 — fraction 0.96 (this directory)

Ready in ~4 min; free after capture 3.26 / 3.19 GB (64K x 4: 3.48 / 3.42 — the larger pool).
Launch log: `DSV4 memory calculation: bytes_per_full_token=1630.00, available_bytes=1.00 GB,
swa_fixed=0.23 GB, full_token=505856` then `max_total_tokens=540672 is larger than the profiled
value 505856. Use the profiled value instead.` The pool budget is what is left after weights
(4.94 GB) minus the dynamic slack `pre-load free x (1 - fraction)` = 93.9 x 0.04 = 3.76 GB, i.e.
~ 1.0 GB; at 1,630 B per full token (plus a fixed 0.23 GB SWA pool) that is 505,856 tokens.

- `stress-kv-522k-rejected.*`: the first 522,240-target prompt (508,754 tokens) → **HTTP 400**
  "Input length (508754 tokens) exceeds the maximum allowed length (505850 tokens)". Not a crash.
- `stress-kv.*`: 516,000-target prompt = **502,649 tokens**, 256 out, TTFT **560.8 s** (896 tok/s
  average), min free **783 / 708 MiB**, finished. Decode 41.1 tok/s with 503K resident
  (`full token usage: 0.99`).
- `verify-503k.*`: `--context 503000` → **ALL PASS**, needles 6/6 at 425,604 tokens (TTFT 418 s).
- `nll.*`: **0 / 4,795 tokens moved** vs `nll-t316c.json`.
- `c1-off6.json`: C1 **41.93 median / 39.99 wall** tok/s (262K serve: 42.27 / 40.34).
- `prefill-16k/32k`: **983 / 985 tok/s** idle (64K x 4: 1,040 / 1,040).
- Health 0 lines, 0 Xid, EDAC 0/0. `launch-container.txt` is the full log.

## Serve 2 — fraction 0.962 (`f0962/`)

The pool is the one place where a higher `MEM_FRACTION_STATIC` helps: it moves bytes from the
dynamic slice into the pool budget. 0.962 → `available_bytes=1.19 GB`, profiled 629,504 tokens, so
the launcher's 540,672 cap takes over (no warning) and the pool grows only by the ~ 57 MB it
needed; free after capture **3.20 / 3.12 GB**.

- `stress-kv.*`: 536,000-target prompt = **522,237 tokens**, 256 out, TTFT **596.3 s** (876 tok/s
  average), min free **713 / 638 MiB**, finished.
- `verify-512k.*`: `--context 524288` → **ALL PASS**, needles 6/6 at **443,617** tokens (TTFT 444 s).
- `nll.*`: **0 / 4,795 moved**. `c1-off6.json`: C1 **42.05 / 40.10 wall**. `prefill-16k/32k`:
  943 / 943 idle (4 % under serve 1's 983; both idle; treated as run-to-run spread, not a gate).
- Health 0, 0 Xid, EDAC 0/0. `launch-container.txt` is the full log.

## Two measurements the rung adds

**Prefill now slopes with prefix.** Per-chunk `input throughput` from the two long prompts
(stress + needles, `TP0` lines, 4096-token chunks, 50K buckets):

| prefix | tok/s | prefix | tok/s |
|---|---|---|---|
| 0–50K | 868 (includes first-chunk compile) | 300–350K | 960 |
| 50–100K | 935 | 350–400K | 948 |
| 100–150K | 985 | 400–450K | 935 |
| 150–200K | 988 | 450–500K | 908 |
| 200–250K | 973 | 500–550K | 896 |
| 250–300K | 968 | | |

E34b-ii saw 953–962 flat to 253K; past ~ 300K the dense indexer's O(prefix) scoring per chunk
(memory is tiled, work is not) costs about −9 % by 512K. Mild, and the first evidence of it.

**Something ~ 3.5–4 KB/token still grows with prefix outside the preallocated pool.** Min free
during the long prompt: 1,807 MiB at 253K (E34b-ii), 783 MiB at 503K, 713 MiB at 522K — the pool
is static, so ≈ 1.0 GB per 250K of prefix is a prefix-proportional allocation the sizer does not
account for. Not identified in this rung (candidates, unverified: the DSA indexer key cache if it
lives outside the `full_token` pool at 1,630 B/token, radix-cache / block-table metadata). It is
the projected 1M binder (see below) and is registered as E34e.

## 1M projection (not run)

Pool for 1,048,576 + 16K tokens ≈ 1.74 GB + 0.23 GB SWA → needs `available_bytes` ≈ 2.0 GB →
fraction ≈ 0.971, which takes ~ 1.0 GB from the dynamic slice (≈ 2.2 GB after capture). The
long-prompt transient at 522K consumed ≈ 2.5 GB of the 3.2 GB slice; at 1M with the ~ 4 KB/token
growth it projects to ≈ 4.5 GB. 1M at 226 / 158 is therefore ~ 2.3 GB short — about 7 GPU experts
(219 / 165), or finding and bounding the E34e allocation. The ladder stops at the measured 512K
rung; 1M is a decision, not a serve.

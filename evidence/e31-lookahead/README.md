# E31 + idea-11 lite — CPU-tier route lookahead and the deferred queue (2026-10-23, 1017)

## E31 lookahead (offline, no GPU spent)

Question: do layer L's top-6 routed experts predict layer L+1's, well enough that a layer-ahead
prefetch/staging slot (BINDER idea 12) or any lookahead predictor could overlap the CPU tier?

Method: the T3.8 recorder dumps from `evidence/t1017/recorder/decode-distinct/` (gitignored .pt,
on quorra) hold per-pass, per-layer routed-expert counts. `lookahead.py` filters single-token
decode passes (route total <= 12 = 6 routes x the 2x all-reduce) and measures
|routes(L) ∩ routes(L+1)| / 6 per consecutive layer pair over 767 passes (4,602 single-token
layer pairs).

Result: **overlap 4,024 / 179,478 = 2.2 %** (chance is 6/384 = 1.6 %; the best pair, L26->L27,
is 8.2 %; every other pair is 1-5 %). The routed experts are essentially uncorrelated across
adjacent layers.

Verdict: **E31 closed with no build** — the go bar was a top-6 hit >= 90 %; measured 2.2 %.
This also kills BINDER idea 12 (one staging slot for high-confidence layer-ahead route hits:
a "high-confidence" set does not exist) and re-confirms the rejection of any
router-before-attention predictor on this stack (mlx-stream's lookahead wins on a different
cost structure: per-expert SSD streaming, where a miss is ~ms, not ~0.1 ms of CPU-tier time).

## Idea-11 lite (D-sweep + HSMP DRAM readout; one D=0 serve)

Question: is the D=4 deferred queue already continuously busy (no scheduling gap left)?

Method: `KT_MAX_DEFERRED=0` serve (profile qualified, override only) vs the default D=4 serve,
same fixtures; HSMP DDR counter (`scripts/hsmp-clocks.py`) sampled during the C1 benches.

| config | C1 median / wall tok/s | C4 aggregate / wall tok/s | DDR peak during C1 |
|---|---|---|---|
| D=4 (default) | 61.08 / 54.46 | 123.79 / 118.72 (headline serve) | 84 / 410 GB/s (20 %) |
| D=0 | 59.70 / 53.66 | 106.49 / 98.83 | 90-94 / 410 GB/s (22-23 %) |

Verdict: **deferral is still worth +2 % C1 and +14-17 % C4 on 1017** (the C1 share shrank from
E21's +22 % because the C1 step is now GPU-side dominated; at C4 the CPU tier is the step and
deferral still hides four experts per token). The D=0 DRAM peak is *higher*, not lower — D=0
serialises all six experts in one burst; D=4 spreads them across two layers — so the counter
alone cannot read the queue; the +14-17 % C4 is the overlap evidence. No timestamp work
scheduled: the overlap is measured, and D=5 was already rejected on NLL (T3.15).

## Files

- `lookahead.py` — the consecutive-layer overlap analysis (runs in the dsv41-kt container
  against the T3.8 recorder dumps; torch).

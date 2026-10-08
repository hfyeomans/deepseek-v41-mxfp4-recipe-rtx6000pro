# E34a — indexer score budget 1 GiB → 256 MiB (2026-10-07)

Protocol: [`plan/EXPERIMENTS.md` E34](../../../plan/EXPERIMENTS.md), part a. Task T3.26.
Overlay commit `1953a92e` (one line: `_TORCH_INDEXER_SCORE_BUDGET_BYTES = 256 << 20` in
`deepseek_v4_backend.py`), on top of `9fc14817`. Two serves on the qualified profile (226 / 158,
D=4, spec off, chunk 4096, `MEM_FRACTION_STATIC=0.96`), E33 gates (a)–(f) in order.

## Result (measured)

| Rung | Slots | Pool | Free after capture (GPU0 / GPU1) | Stress-kv | Needles | NLL vs T3.16c | C1 summed / wall / TTFT | Min free during stress | Health | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 64K | 4 | 278,528 | 3.48 / 3.42 GB (unchanged: the budget is a transient) | 4 × 62K all finished; TTFT 90.7 / 153.2 / 215.5 / 276.0 s (serialised chunked prefill), wall 293 s, peak 97,246 / 97,305 MiB | 6/6 at 55,206 tokens, TTFT 54.47 s | 0 of 4,795 moved | 42.20 / 40.26 / 0.592 s | **641 / 582 MiB** | 0 lines, EDAC 0/0 | **pass** (E33: OOM at ≈60K prefix) |
| 128K | 1 | 147,456 | 4.05 / 3.97 GB | **FAIL: CUDA OOM at ≈110K prefix** (19,277 tokens pending of 129K): 448 MiB requested, 196 MiB free on GPU1 | not reached | — | — | — | 0 NaN / Xid | **fail, gate (b)** (E33: OOM at ≈65K; E34a reaches ≈110K) |

## What moved, and what did not

- **The pad copy is gone as the binder.** E33's OOMs were the `F.pad` copy in `select_candidate_blocks`
  (1,024 / 960 MiB, exactly `step × lc × 4 B` = the budget). With a 256 MiB budget the 64K / 4-slot
  rung passes every gate, and the 128K rung gets from ≈65K to ≈110K of prefix before dying.
- **The remaining binder is the `[4096, prefix]` fp32 logits and their broadcast mask.** The 128K
  failure is at `deepseek_v4_backend.py:3313`, `scores.masked_fill_(j[None, :lc] >= lens, -inf)`:
  the comparison materialises a `[4096, lc]` bool (448 MiB at lc ≈ 110K) on top of the 1.75 GiB
  logits. The allocator-retry sizes in the log step 1,073 → 1,744 MiB in 64 MiB increments, which is
  4096 × prefix × 4 B for the prefix at each chunk — the logits themselves.
- **Prediction miss, recorded:** I projected ≈ 1.6 GB min free at 64K / 4 slots and a pass at
  128K / 1 slot with ≈ 0.5–0.8 GB. Measured: 641 / 582 MiB at 64K / 4 slots, and a fail at 128K.
  Unaccounted: the broadcast bool mask (0.25 × logits), and at 4 slots the other three streams'
  decode / sparse-attention working set while one prefills. The direction of the model is right,
  its margin arithmetic was ≈ 1 GB optimistic.
- **No cost measured.** C1 42.20 / 40.26 wall against 42.25 / 40.35 at E33 64K / 1 slot; NLL
  bit-identical; needles TTFT 54.5 s vs 57.1 s (run-to-run).

## Decision

- Keep the 256 MiB budget as the overlay default (`1953a92e`): it qualifies 64K / 4 slots at no
  measured cost. It is not the fix: E34b (tile the scoring over key blocks with a running top-k
  so neither the logits nor the mask are ever `[chunk, prefix]`) is the only part that reaches
  128K and beyond.
- 64K / 4-slot min free of ≈ 0.6 GB is thinner than the 32K / 4-slot deployment profile's
  ≈ 0.8–1.3 GB (E28, E29). ~~It is a qualified rung, not a new deployment default; the default stays
  32K / 4 slots until E34b lands and the margin is re-measured.~~ **Superseded the same day:** the
  owner's rule was "if 64K × 4 is qualified it becomes the default", qualification requiring the
  E27-shape served soak on top of the gates. The soak passed (below); `serve-hybrid.sh`
  `CONTEXT_LENGTH` default is 65536 as of 2026-10-07.

## Served soak, 64K × 4 slots (`soak-64k-4slot/`, 2026-10-07, one serve, 21:17–22:09)

Shape = E27 Boot 12 (`soak-64k-4slot/soak-64k.sh`): same-prompt C4 consistency → 8 bench blocks
(c4-off0 c3-off2 c2-off4 c4-off8 c1-off6 c4-off12 c3-off9 c2-off0; 12 runs × 512 tokens, distinct
prompts seed 4242, health count after each) → NLL vs T3.16c → 16K prefill → 4 × 62K stress →
64K needles → final health / EDAC. `KT_NAN_PROBE_MIN_TOKENS=1` throughout. Free after capture
3.48 / 3.42 GB (identical to the gate serve).

| Gate | Result |
|---|---|
| Same-prompt C4 consistency | PASS, min pairwise prefix 310 / 519 chars (gate ≥ 40; slot 2 exact) |
| C1 (c1-off6) | 41.89 median / **39.94 wall** tok/s, TTFT 0.61 s |
| C2 (off4, off0) | 52.10 / 49.72 and 50.47 / 47.92 |
| C3 (off2, off9) | 56.06 / 54.48 and 55.80 / 54.30 |
| C4 (off0, off8, off12) | 58.02 / 55.78, 57.84 / 55.53, 58.96 / 57.40 |
| Health after every block | 0 (NaN / probability tensor / Xid / Traceback / KTContract / Sealed) |
| NLL vs `nll-t316c.json` | 0 of 4,795 moved (Δ mean 0.0, max |Δ logp| 0.0) |
| 16K prefill | 841.8 tok/s, TTFT 18.98 s (15,978 tokens) |
| Stress 4 × 62K | all finished, wall 295 s; TTFT 76.3 / 144.3 / 212.4 / 277.6 s; peak 97,150 / 97,237 MiB; **min free 737 / 650 MiB** |
| Needles at 64K | 6/6 at 55,206 tokens, TTFT 57.5 s, `finish_reason=stop` |
| Final | health 0, EDAC 0 / 0, 0 Xid in container log |

Every decode number is within 0.5 % of the 32K × 4 profile (E24 / T3.19b); the context change
costs nothing on decode and nothing on quality. What it costs is margin: 0.65 GB worst case.

## Side observation

GPU0 read 100 % utilisation / 96 W with 2 MiB used after the 128K teardown (documented phantom,
`docs/TROUBLESHOOTING.md`); 0 Xid, both failures Python-level `torch.OutOfMemoryError`.

## Files

- `64k-4slot/`: `launch.txt`, `launch-container.txt`, `stress-kv.json/.txt`, `verify-64k.json/.txt`,
  `nll-64k.json/.txt`, `c1-off6.json`, `c1-bench.txt`.
- `128k/`: `launch.txt`, `launch-container.txt` (traceback, allocator-retry lines), `stress-kv.txt`.

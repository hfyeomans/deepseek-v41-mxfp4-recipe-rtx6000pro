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
  ≈ 0.8–1.3 GB (E28, E29). It is a qualified rung, not a new deployment default; the default stays
  32K / 4 slots until E34b lands and the margin is re-measured.

## Side observation

GPU0 read 100 % utilisation / 96 W with 2 MiB used after the 128K teardown (documented phantom,
`docs/TROUBLESHOOTING.md`); 0 Xid, both failures Python-level `torch.OutOfMemoryError`.

## Files

- `64k-4slot/`: `launch.txt`, `launch-container.txt`, `stress-kv.json/.txt`, `verify-64k.json/.txt`,
  `nll-64k.json/.txt`, `c1-off6.json`, `c1-bench.txt`.
- `128k/`: `launch.txt`, `launch-container.txt` (traceback, allocator-retry lines), `stress-kv.txt`.

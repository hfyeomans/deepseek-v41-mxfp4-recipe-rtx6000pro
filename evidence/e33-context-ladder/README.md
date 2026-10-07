# E33 — context ladder (2026-10-07)

Protocol: [`plan/EXPERIMENTS.md` E33](../../plan/EXPERIMENTS.md). Task T3.6e–g. One serve per
rung on the qualified hybrid profile (226 GPU / 158 CPU experts, D=4, spec off, chunk 4096,
`MEM_FRACTION_STATIC=0.96`, pool derived as slots × context + 16K), gates (a)–(f) in order on
the same server, stop at the first failing gate. No overlay change (HEAD `9fc14817`).

## Ladder (measured)

| Rung | Slots | Pool (tokens) | Free after capture (GPU0 / GPU1) | (a) launch | (b) stress-kv | (c) needles | (d) NLL | (e) health | (f) prefill tok/s / TTFT | C1 summed / wall / TTFT | Min free during stress (GPU0 / GPU1) | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32K | 4 | 147,456 | 4.24 / 4.18 GB (E29, 93,646 / 93,703 MiB used) | pass | pass (4 × 30K) | 6/6 | 0 moved | 0 / EDAC 0/0 | 845 / 38.1 s at 32K (E28, E29) | 42.04 / 40.15 / 0.594 s (T3.19b) | 779–1,349 MiB / 1,226–1,290 MiB (E28, E29) | **pass — deployment profile** |
| 64K | 1 | 81,920 | 4.19 / 4.11 GB | pass (capture bs=[1]) | pass: 61,927-token prompt, TTFT 84.33 s, peak 97,080 / 97,155 MiB | 6/6 at 55,206 tokens, TTFT 57.05 s | 0 of 4,795 moved (bit-identical to T3.16c) | 0 / EDAC 0/0 | ≈734 (61.9K) and ≈968 (55.2K) derived from TTFT | 42.25 / 40.35 / 0.592 s | 807 / 732 MiB | **pass — qualified limit** |
| 128K | 1 | 147,456 | 4.05 / 3.97 GB | pass | **FAIL: CUDA OOM** at ≈65K prefix (chunk 16): 1,024 MiB requested, 1,003.75 MiB free on GPU1 | not reached (server dead) | — | — | — | — | — | **fail, gate (b)** |
| 64K | 4 | 278,528 | 3.48 / 3.42 GB | pass (capture bs=[1 2 3 4], 19.5 s) | **FAIL: CUDA OOM** at ≈60K prefix on both ranks: 960 MiB requested, 903 / 842 MiB free | not reached | — | — | — | — | — | **fail, gate (b)** |

The pool itself is cheap: "Memory pool end" reads 4.58 → 4.48 → 4.15 GB free for pools of
81,920 → 147,456 → 278,528 tokens (≈ 0.1–0.2 GB per 65K tokens of DSV4 compressed KV, plus the bs=4 capture). The 262K,
512K and 1M rungs were not launched: they fail the same gate for the same reason (below)
before KV bytes matter.

Prefill rates at 64K are derived from TTFT (prompt tokens ÷ TTFT), not from a separate
prefill benchmark; they are lower than the 845 tok/s measured at 32K because the indexer's
dense scoring and the sparse-attention cost both grow with the prefix.

## Mechanism (verified in overlay source, not projected)

Both OOMs have the same stack: `dsv4/indexer.py select_candidate_blocks` → `F.pad(logits…)`
called from `deepseek_v4_backend.py _publish_or_consume_candidates` ←
`_low_ratio_index_topk_dense`. The DSV4 sparse-attention indexer scores every prefill chunk
against the whole prefix with a dense fp32 tensor `[chunk_tokens, lc]`, where
`lc = seq_len // compress_ratio` and the ratio is 1 on this model, so `lc` = prefix length:

- 4096 × prefix × 4 B = **64 MiB per 4K of prefix per chunk**: 1 GiB at 64K, 2 GiB at 128K,
  4 GiB at 262K, 16 GiB at 1M (per rank, per chunk, transient).
- `select_candidate_blocks` pads a copy of its rows, chunked by the upstream constant
  `_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30` (`deepseek_v4_backend.py:336`, comment:
  "transients run ~3x this"), so a second ≤ 1 GiB transient sits on top of the logits.
- The allocation sizes in the allocator-retry lines (604, 738, 805, 872, 939, 1,007, 1,074 MiB)
  are exactly 4096 × prefix × 4 B for the prefix at each chunk.

At 226 / 158 the ranks have ≈ 4.0 GB free after capture; a 64K single-slot prefill leaves
≈ 0.75 GB at its worst chunk, and the next chunk at 128K (or a second concurrent prefill at
64K) asks for more than is left. The 32K / 4-slot deployment profile sits ≈ 1 GB from the same
edge (E28 worst case 779 MiB), which is the transient E29 saw shrink by ≈ 1.5 GB at chunk 2048
(min free 2,923 vs 1,349 MiB) at a −40 % prefill cost.

## What follows (registered as E34, not run)

Each is a separate experiment per the E33 stop rules ("a candidate fix for a failed rung is a
separate experiment"):

- **E34a (overlay one-liner, projected):** lower `_TORCH_INDEXER_SCORE_BUDGET_BYTES` to 256 MiB.
  Bounds only the padded copy, not the `[chunk, lc]` logits, so it likely recovers 64K / 4-slot
  and 128K / 1-slot and cannot reach 262K.
- **E34b (real fix, upstream-PR candidate):** tile the dense indexer scoring over key blocks
  with a running top-k so the transient is O(chunk × block) instead of O(chunk × prefix).
  Removes the context dependence; cost is a second pass over the scores.
- **E34c (placement trade, derived):** 0.34 GB per GPU expert per rank (E28); 262K needs
  ≈ +4 GB (≈ 12 fewer GPU experts, ≈ −7 % C4 / −4 % C1 by the E28 byte model); 1M ≈ +16 GB
  (≈ 47 fewer, ≈ −30 % C4). Chunk 2048
  halves the logits at −40 % prefill (E29). Neither is attractive next to E34b.

## Side observation

GPU0 read 100 % utilisation / P0 with no model loaded between the 128K teardown and the
64K / 4-slot launch (documented phantom, `docs/TROUBLESHOOTING.md`). It cleared on its own:
after the final teardown GPU0 read 0 % / 15.8 W, `dmesg | rg -c Xid` = 0. Both failures were
Python-level `torch.OutOfMemoryError`, not device asserts.

## Files

- `64k/`: `launch.txt` (resolved line), `launch-container.txt` (redacted), `stress-kv.json/.txt`,
  `verify-64k.json/.txt`, `nll-64k.json/.txt`, `c1-off6.json`, `c1-bench.txt`.
- `128k/`: `launch.txt`, `launch-container.txt` (traceback and allocator-retry lines),
  `stress-kv.txt` (HTTP 500 traceback), `verify-128k.txt` (connection failure, server dead).
- `64k-4slot/`: `launch.txt`, `launch-container.txt` (both ranks' tracebacks), `stress-kv.txt`.

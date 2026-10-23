# T3.49 — admission-time Engram prefetch for chunk 0 (first cut, 2026-10-23)

Overlay branch `t349-admission-prefetch` (worktree `/tmp/sglang-t349`, commit `e496e9d3e` on `deec7a6f`):
a third `_FileTable` slot (`_admission`) adopted by the forward's submit path (same pattern as
T3.20 (c) chunk-ahead), filled by the scheduler at request admission
(`_engram_admission_prefetch`: hash the first prefill chunk's rows, submit to the shared worker).
Launcher knob `ENGRAM_ADMISSION_PREFETCH` (default 0; env `SGLANG_DSV41_ENGRAM_ADMISSION_PREFETCH`).
Purely advisory: a mismatch or a dropped fetch costs one extra worker fetch, never correctness.

## Regression gates (test serve, release-smoke.sh serve --full, 262K x 4)

**All gates passed** — same-prompt C4, NLL corpus identity (0 tokens moved vs the 1017 baseline),
needles 6/6, stress 4x260K, decode C1/C4 + prefill 16K/32K benches, power summary, health
(`evidence/t349/smoke/`).

## A/B (control = default overlay; test = t349 branch + ENGRAM_ADMISSION_PREFETCH=1)

Cold state: `scripts/drop-engram-cache.py <snapshot>` (posix_fadvise DONTNEED on the two Engram
shards) between the measurements; a run counts as cold only when the warmth bench reports the
cold pass loading ~700K-940K pages (resident < 13 GiB) and warm when it loads 0.

| instrument | control (no prefetch) | test (admission prefetch) | read |
|---|---|---|---|
| warmth cold TTFT, 16K real text (2,915-3,045 tok/s) | 5.561 / 6.735 s | 5.323 / 6.559 s | null (within +-0.2 s) |
| warmth cold TTFT, 16K synthetic | 5.569 s | 5.346 s (5.545 s first run) | null |
| queue-wait probe (16K prefill admitted 6 s into a decode), cold, seed 9242 | 5.411 / 5.190 / 5.129 s | 4.921 / 4.769 / 4.733 s (run 1); 5.892 / 5.022 / 4.910 s (run 2) | -0.2 to -0.4 s median (3-9 %) |
| queue-wait probe, warm cache | 4.68 / 4.02 / 3.80 s | 3.53 / 3.54 / 3.55 s | -0.3 to -0.5 s |

`evidence/t349/control-cold/` (tool-stamped) holds the control cold run; `evidence/t349/test/` and
`test2/` hold the test runs (warm and mixed states; the valid cold test numbers above are the
tool-refused runs' stdout, conditions recorded).

## Verdict

**Real but small: ~0.2-0.5 s (3-9 %) off a cold request admitted while a decode runs; null on
idle admission** (the ~3 GiB page-granular fetch cannot be hidden when the request starts
immediately — the ahead worker already hides what it can). The E40 +1.7-3.1 s cold figure is the
whole admission path, not the row fetch alone. The knob stays **off by default**; a C4-heavy
coding-agent deployment that admits requests during decode is the reopen condition (one flag,
already qualified for no regression).

## Scripts

- `scripts/bench-engram-admission.py` — the queue-wait probe (decode on a worker thread, prefill
  timed on the main thread).
- `scripts/drop-engram-cache.py` — evict the Engram shards' page cache for cold measurements.

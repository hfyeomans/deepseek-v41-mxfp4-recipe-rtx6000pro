# T3.20 / E36 — Engram rows fetched ahead of use (2026-10-09)

Three serves on the launcher default (219 / 165, D=4, chunk 8192, 64K × 4, `MEM_FRACTION_STATIC=0.96`,
expandable-after-capture on), one per value of the new knob `ENGRAM_FILE_PREFETCH`
(`SGLANG_DSV41_ENGRAM_FILE_PREFETCH`), KT timeline on (`KT_TIMELINE=1`), in the order pool → sync → ahead,
back to back (17:45–18:25). Overlay content = `ff516964` (served as `30e8c412`-dirty; the commit was made
after the serves with no further change, see the manifests' `overlay.dirty`). Recipe at `365040d`-dirty
(the knob and `scripts/bench-engram-advise.py`, committed with this README).

| mode | what the break does |
|---|---|
| `pool` | the served path until today (E18): `MADV_WILLNEED` per distinct page from a 16-thread pool, then `index_select`, in the layer's own eager break |
| `sync` | the same, one thread |
| `ahead` (new default) | layer 1's break does one D2H of *both* layers' ids and hands layer 14's to its table; one worker thread advises (libc `madvise`, GIL released) and gathers those rows into pinned memory while layers 2–13 run; layer 14's break only joins (ids compared with `torch.equal`; a mismatch falls back to a synchronous fetch) |

All three modes copy from pinned staging with `non_blocking=True` (new); the pool mode therefore differs from the
pre-T3.20 served path by that alone.

## Result

| | pool | sync | **ahead** | ahead vs pool |
|---|---|---|---|---|
| decode bs 1: Engram break per step (GPU gap, rank 0) | 2.200 ms (layer 1 1.278 + layer 14 0.935) | 1.800 (1.131 + 0.656) | **1.663 (1.235 + 0.448)** | **−24 %** (layer 14's break −52 %) |
| decode bs 4: Engram break per step | 3.191 (1.613 + 1.550) | 2.911 | **2.149 (1.551 + 0.579)** | **−33 %** |
| decode bs 1 step (timeline tick median) | 24.160 ms | 23.808 | **23.742** | −1.7 % |
| decode bs 4 step | 71.206 | 71.078 | 71.171 | 0 (CPU-tier bound) |
| C1 median / **wall** tok/s (6 runs, 512 tokens, offset 6) | 40.72 / 38.87 | 41.15 / 39.27 | **41.47 / 39.55** | **+1.7 %** wall |
| C4 median / wall (3 runs) | 55.66 / 53.55 | 55.87 / 53.74 | 55.90 / 53.77 | +0.4 % |
| prefill 16K cold (seed 777) / warm (seed 1277), tok/s | 1,592 / 1,485 (TTFT 10.1 s) | 1,323 / 1,453 (12.1 s) | **1,844 / 1,875 (8.7 s)** | **+16 % / +26 %** |
| prefill 32K cold (seed 1277) / warm (seed 1777) | 1,596 / 1,566 (20.0 s) | 1,514 / 1,501 (21.1 s) | **1,915 / 1,942 (16.7 s)** | **+20 % / +24 %** |
| Engram shard residency before the 16K cold pass (layer 1 / 14) | 1.45 % / 1.45 % | 5.4 % / 5.4 % | 1.48 % / 1.48 % | same cold state |
| per-chunk steady state on the 32K prompts (container log, 8192-token chunks) | (log not captured) | 1,470–1,678 | **1,947–2,026** | = the T3.31 best (1,939–2,020) |
| same-prompt C4 consistency (`--same-prompt`) | not run (script bug; see below) | PASS, min pairwise row prefix 290 | **PASS, 290** | gate ≥ 40 |
| corpus NLL vs `evidence/t328/default-219-8192/corpus/` | — | — | **0 of 65,653 moved** | bit-identical |
| needles 64K (55,206 tokens) | — | — | **6/6, TTFT 27.4 s** | T3.31: 6/6 |
| health (NaN / Xid / Traceback / OOM) | 0 | 0 | 0 | |

Gates. **E36 go met**: long-prefill wall falls ≥ 5 % (16K TTFT 10.1 → 8.7 s, −14 %; 32K 20.0 → 16.7 s, −17 %)
with NLL bit-identical and decode in band. **T3.20 (a)'s decode go not met**: break time −24 % (bar −50 %)
and C1 wall +1.7 % (bar +3 %); the ledger's stop clause (< +1 % after one tuning round) is not met either, so a
second round is admissible — see "What is left" below. `ahead` is the default on every count that moved and on
none that regressed, so the knob defaults to it and the recipe's qualified profile is this serve.

Default-profile stress on the committed overlay `ff516964` (`ahead-stress/`, launcher defaults, timeline off):
4 × ~62K prompts (seed 31, 256 tokens each) **4/4 finished by length**, wall 167.3 s, per-chunk prefill ≈ 1,970 tok/s,
**min free 2,203 / 2,144 MiB** (T3.31 soak on `30e8c412`: 2,179 / 2,122 — same band); VRAM after capture
92,776 / 92,835 MiB; needles **6/6 at 55,206 tokens** (TTFT 27.45 s); same-prompt C4 **PASS, 225**; health 0.

## Reading

1. **The layer-14 break is now a join**: 0.94 → 0.45 ms at bs 1, 1.55 → 0.58 at bs 4. What remains is the
   break floor — a D2H sync of the ids, `torch.equal`, the pinned → device copies, the dequant, and the
   relaunch of the next graph segment. The layer-1 break did not move (1.28 → 1.24): it still pays the fetch
   itself, and at decode most rows are cold (shards 1.5–14 % resident → one NVMe round trip, ≈ 0.4–0.8 ms),
   so its remaining ≈ 0.8 ms over the floor is NVMe latency that only an *earlier* issue can hide.
2. **Prefill gains more than decode because the hidden part is the NVMe-bound part**: one 8192-token chunk
   touches ≈ 786K distinct pages per layer (E36 (1), `io/README.md`: 0.5–0.6 s cold, 0.2 s warm per layer with
   serial issue). `ahead` moves layer 14's whole advise + gather off the launch thread and under layers
   2–13 of the same chunk; layer 1's stays. The +16–26 % at the same ≈ 1.5 % residency is that one layer's
   I/O hidden; the per-chunk steady state matches the best the box has read (T3.31, 26 % residency).
3. **`sync` prefill read below `pool`** (1,323 vs 1,592 at 16K) although the idle-box microbench had one
   thread ≤ the pool at every shape, and `ahead` — whose layer-1 issue is the same serial loop — beat both.
   One serve each, no explanation found; recorded, not built on. The box state differed too (the sync serve
   started with 2 GB swap used; residency during its passes 5 %, pool's and ahead's 1.5 %).
4. The pool mode's distinct-prompt consistency run (`pool/consistency.txt`, the default protocol without
   `--same-prompt`) printed FAIL on prefixes 62 / 8 chars; the same prompts fail the same way in `sync` and
   `ahead` (`consistency-distinct.*`, slot 2 flips `Mercury -` → `Mercury –` at the second token) and did in
   T3.15's D=5 run (`evidence/e24-nan/t315-d5/verify-c4-distinct.json`, prefixes 9 / 32). Distinct-prompt
   bs=1-vs-bs=4 is a near-tie flip under a different batch composition, never the gate; the gate is
   `--same-prompt` (rows of one batch byte-identical: 290 ≥ 40, E26 read 310), run on sync and ahead.
   pool's same-prompt gate was skipped by the script bug; its code path differs from ahead only by what
   ahead's NLL / same-prompt / needles gates cover.
5. `ahead_hit` / `ahead_miss` counters exist in `_FileTable.stats` but are not logged; the layer-14 gap
   (0.45 vs 0.66 ms for a sync fetch of warm rows) is the indirect evidence that decode steps hit.

## What is left (T3.20 round 2, not started)

The layer-1 break's ≈ 0.8 ms over the floor is the next step's fetch latency. Both layers' ids are
`EngramHasher(input_ids)`; in decode the next step's input id is the token just sampled, which the host already
holds for detokenization. Computing both layers' hash ids on the host at the end of step N and submitting both
fetches before step N+1 starts would make layer 1's break a join too: projected break 1.66 → ≈ 0.9 ms
(−59 % vs pool, meets the −50 % bar), step −3 %, C1 wall ≈ +3 % cumulative with round 1 ≈ +5 % (meets +3 %).
Cost: a host-side hasher that reproduces the GPU ids exactly, a hook after sampling in the model runner, a
per-request history window, and the miss fallback already in `gather()`. The cross-chunk prefill analogue
(submit layer 1's rows for chunk k+1 from chunk k) needs the next chunk's ids from the scheduler and is a
separate item.

## Files

Per mode: `launch.txt`, `ready.txt`, `engram-describe.txt` (the table's own description of its mode),
`consistency*.{txt,json}`, `c1.*`, `c4.*`, `prefill-16k.*`, `prefill-32k.*` (`--warm-pass`, residency recorded
per pass), `prefill-log.txt` (container `Prefill batch` lines; sync and ahead only — pool's `chunk-log.txt` is a
grep miss), `timeline-rank{0,1}.jsonl` + `report.{json,md,txt}` (`kt-timeline-report.py`), `manifest.json`,
`pre-serve-state.txt` (`free -g` + `fincore` of the two shards before the launch). `ahead/` also has
`corpus/` + `corpus.txt` (NLL) and `verify.{txt,json}` (needles). `ahead-stress/` is the default-profile stress on
the committed overlay (`stress-kv.{txt,json}`, `verify.*`, `consistency.*`, `vram-after-capture.txt`, `manifest.json`;
driver `.amp/in/t320/stress.sh`). `io/` is the E36 (1) microbench
(`scripts/bench-engram-advise.py`, idle box, no manifest). Timeline tensor dumps were not kept.
Driver: `.amp/in/t320/ab2.sh` (not committed; the pool run used its predecessor without `--same-prompt`).

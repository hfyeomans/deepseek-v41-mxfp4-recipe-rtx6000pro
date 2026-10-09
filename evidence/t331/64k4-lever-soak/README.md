# T3.31 idea 8 — 64K × 4 soak with `expandable_segments` after capture: lever cost 0, every gate passed (2026-10-09)

Deployment profile (`bash scripts/serve-hybrid.sh`, 219 / 165, D=4, chunk 8192, 64K × 4,
`MEM_FRACTION_STATIC=0.96`, spec off) with one change: `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1`.
Overlay `30e8c412`, image `dsv41-kt:recipe`, one serve, 20:03–21:08 UTC. Comparator: the same profile
with the lever off on `1eeb48cd`, `../../e34-indexer-memory/d-v-ii/64k4/` (E34d-v (ii), earlier the
same day), whose NLL corpus and seed-777 prefill prompts this run repeats.

Question (owner-stipulated before any default change): does the lever cost anything at the
deployment shape — decode, prefill, NLL, stability over the E27-shape soak — and does it change the
memory picture?

## Gates (E33 (a)–(f)) against the lever-off comparator

| gate | lever on (this run) | lever off (`d-v-ii/64k4/`) | verdict |
|---|---|---|---|
| after capture | **4.61 GB** available, 92,776 / 92,835 MiB used; hook log reserved 91,402 → 91,402 MiB (`empty_cache()` released nothing here either) | 4.61 GB | identical |
| (b) stress 4 × 62K (`--seed 31`) | all finished, wall **188.5 s**, TTFT 137 / 63 / 171 / 102 s, min free **2,179 / 2,122 MiB** | 183.8 s, min free 1,791 / 1,746 MiB | **pass**; +390 / +376 MiB at the trough |
| (c) needles (`verify.py --context 65536`) | **6/6 at 55,206**, TTFT 31.8 s; alias / ceiling / arithmetic / thinking toggle / tool call PASS | 6/6, 29.4 s | **pass** |
| (d) NLL corpus vs `../../t328/default-219-8192/corpus` | **0 of 65,653 tokens moved**, max\|Δlogp\| 0.000 on all 8 texts | 0 moved | **pass** |
| (f) decode C1 / C4 (`--distinct-prompts --seed 4242`) | C1 **41.57 / 39.66 wall**, TTFT 0.617 s; C4 **56.43 / 54.27 wall**, TTFT 1.516 s | C1 41.61 / 39.69; C4 56.20 / 54.03 | **pass** |
| (f) prefill 16K / 32K, seed 777 (cold pass) | **1,559 / 1,900 tok/s** median; per-chunk steady state on the 32K prompts **1,947–1,987 tok/s** | 1,887 / 1,901; per-chunk 1,939–2,020 on the same prompts | **lever cost 0** (see below) |
| (f) prefill 16K / 32K, seed 1277 (`--warm-pass`) | 1,615 / 1,607; rerun after `/flush_cache` with the rows page-resident: 1,592 / 1,603 | — | prompt-dependent, not lever-dependent (below) |
| (e) health, EDAC | `health-hits.txt` empty; **0** `expandable_segments: memory mapping failed` warnings (the 262K × 4 serve had 114); EDAC 0/0 | | **pass** |

## E27-shape soak, 8 blocks × 12 runs × 512 tokens (after the gates)

| block | C | median tok/s | min / max | TTFT s |
|---|---|---|---|---|
| c4-off0 | 4 | 56.18 | 55.93 / 56.41 | 1.55 |
| c3-off2 | 3 | 54.25 | 53.97 / 54.68 | 0.87 |
| c2-off4 | 2 | 50.96 | 50.20 / 51.21 | 0.97 |
| c4-off8 | 4 | 56.21 | 55.64 / 56.38 | 1.55 |
| c1-off6 | 1 | 41.61 | 41.24 / 41.80 | 0.62 |
| c4-off12 | 4 | 57.56 | 56.94 / 57.68 | 1.02 |
| c3-off9 | 3 | 53.88 | 53.57 / 54.14 | 0.84 |
| c2-off0 | 2 | 49.01 | 48.79 / 49.23 | 1.15 |

Health 0 after every block; no drift block to block (C4 56.18 → 56.21 → 57.56 is the fixture
offset, as in E34a's soak on 226 / 158, where the same three blocks read 58.02 / 57.84 / 58.96).

## The prefill reading, resolved by the per-chunk log

The idea 8 serve at 262K × 4 had read 1,532 / 1,541 at 16K / 32K and the question was whether
the lever costs prefill. The container log's per-chunk `input throughput` on the **same seed-777
32K prompts** settles it (first chunk excluded — it carries the per-request first-touch at
≈ 780–960 tok/s on every serve):

| serve | pool (tokens) | lever | 32K seed-777 chunks 2–6, tok/s |
|---|---|---|---|
| 64K × 4, `d-v-ii/64k4/` | 278,528 | off | 1,954 / 1,939 / 1,962 / 1,952 / 2,020 |
| **64K × 4, this run** | 278,528 | **on** | **1,956 / 1,983 / 1,947 / 1,987 / 1,966 / 1,976** |
| 262K × 4, `d-v-a/262k-4slot-idea8/` | 1,064,960 | on | 1,617 / 1,591 / 1,519 / 1,638 |

So: lever on vs off at the same pool is identical chunk for chunk; the −18 % at 262K × 4 (and the
1M × 1 serve's 1,659, the 512K × 1 serve's 1,627, lever off) goes with the **1,064,960-token pool**,
not the lever. Something on the chunk path scales with `max_total_tokens` — open, registered as a
strengthening item (T3.37) since every rung above 64K pays it.

Second reading, new: **chunk throughput depends on the prompt.** On this one serve, seed-777 32K
prompts ran their steady chunks at 1,947–1,987 and seed-1277 prompts at 1,620–1,713, three prompts
each, and the seed-1277 rerun after `/flush_cache` with their Engram rows page-resident read the same
1,634–1,698 — so it is the prompt, not page-cache state (the T3.19d residency explanation covers the
first chunk, not the steady ones). `make_prompt` draws random words with 20 % numeric suffixes from
the same list for both seeds, so the candidate is the per-chunk routing distribution over the 165
streamed CPU experts (tokens per expert, distinct experts hit), which the streamed-prefill path pays
in weight bytes; not measured here. Consequence for the recipe: a single `bench-prefill.py` seed is a
±10 % instrument across prompts, and only same-seed per-chunk comparisons are A/Bs.

## What this settles

- The lever has no measurable cost at the deployment shape: decode, NLL and per-chunk prefill are
  identical to lever-off; stress trough improves by ≈ 380 MiB; 8 soak blocks clean; zero
  allocator mapping retries at 64K × 4 (the 114 at 262K × 4 were the near-OOM fallback path).
- The owner-stipulated precondition for making it the default is met. Flipping the default is a
  separate owner decision (goldens and manifest change): the gain at 64K × 4 is ≈ 380 MiB of
  stress headroom and one fewer flag for the 262K × 4 rung; the risk is the near-OOM retry path
  behaving differently under a workload this soak did not exercise.

## Files

`manifest.json`, `launch.txt`, `launch-container.txt` (hook at ~472 / 476), `vram-after-capture.txt`,
`stress-kv.{json,txt}`, `verify.{json,txt}`, `corpus/` + `corpus.txt`, `c1.*`, `c4.*`,
`prefill-16k.*`, `prefill-32k.*` (cold + warm passes), `prefill-16k-resident-rows.*`,
`prefill-32k-resident-rows.*` (seed 1277 after `/flush_cache`), `c4-off0.*` … `c2-off0.*` (soak
blocks), `health-hits.txt` (empty), `edac.txt`.

# E34d-v (ii) — three prefill transients named and removed; 1M × 1 qualified (2026-10-09)

Overlay `1eeb48cd` (three bit-exact commits on top of d-v (i)'s `2323052b`), image `dsv41-kt:recipe`,
launcher default 219 / 165 + chunk 8192, BIOS slot 3, 300 W, D=4, spec off. Four serves, ~1 h 40 min:
three 1M × 1 attempts (`MEM_FRACTION_STATIC=0.962`, pool 1,064,960 tokens, capture-end avail
3.25 / 3.18 GB) — one per overlay commit, each stopped by the next transient — and one 64K × 4 at the
launcher default (`64k4/`) as the same-shape insurance for the deployment profile. Comparators:
`evidence/e34-indexer-memory/d-v-i/` (the 1M OOM at ≈ 680K and the 64K × 4 gate on `2323052b`),
`evidence/t329/512k-1slot/` (prefill 1,650 / 1,669, the previous ladder top),
`evidence/t328/default-219-8192/corpus/` (NLL baseline, same shape).

Question (EXPERIMENTS Part d-v (ii)): the d-v (i) 1M retry OOMed at ≈ 680K prefix on a 332 MiB
int64 index tensor the indexer's key gather builds (512 B of index per prefix token to read 68 B of
keys). Does indexing a `[pages, page_size, 64]` view with two `[n]` tensors (16 B per token) read the
same bytes, and does 1M × 1 then prefill to the end?

## What the three serves found

The gather was the binder at 680K, and behind it sat two more chunk-proportional transients that the
262K × 4 run of d-v (i) had already named by size (320 / 400 / 640 MiB retries, 234 MiB failure) but
not by owner. Each 1M serve removed one and ran into the next; the third finished the prompt.

| serve | overlay | change | 1M × 1 stress (1,046,528 target → 1,019,251 prompt tokens) |
|---|---|---|---|
| `1m-gather-only/` | `e5f60f54` | `DeepSeekV4IndexerPool._gather_fp4_index_k_rows`: `[pages, p, 64]` / `[pages, p, 4]` views indexed with `[n]` page and offset tensors — **16 B of index per slot instead of 544 B**; used by `get_index_k_fp4` and `get_index_k_dequant` | **OOM at ≈ 298K prefix** (12:31, full token usage 0.28) on a **640 MiB** `pre_hc_head.float()` at `deepseek_v4.py:4304`, 504 MiB free, 2.09 GiB reserved-but-unallocated. The gather site did not fail (it had at 680K). Container log lost; `crash-note.txt` transcribes it |
| `1m-gather-hc/` | `f18dc961` | `deepseek_v4.py:~4304`: pass the bf16 `pre_hc_head` to `hc_combine` (the kernel casts on load) — removes the **640 MiB fp32 copy** of the `[8192, 20480]` hyper-connection stream per chunk | **stress PASSED**: 1,019,251 tokens, TTFT **961 s**, 256 output tokens, decode 37.96 tok/s at 1.02M prefix, min free 547 / 552 MiB. Then the needle run **OOMed at ≈ 860K** on a **234 MiB int32** temporary in `kt_prefill.py:329` (`(raw.view(int16).to(int32) >> 7).to(uint8)`, the CPU-tier scale conversion), 192 MiB free, 919 MiB reserved-but-unallocated |
| `1m/` | `1eeb48cd` | `kt_prefill.py`: `e8m0_from_bf16_bits_(raw, dst)` — in-place `int16 >>= 7` then a narrowing `copy_` into the uint8 destination; removes **2 × 234 MiB int32** per streamed layer | **every gate passed** (table below); TTFT **803 s** |

The three transients, per 8192-row chunk at 1M: gather index 332 MiB → ≈ 10 MiB (per low-ratio layer),
`hc_combine` input 640 MiB → 0, kt scale conversion 2 × 234 MiB → 0. None changed a byte of output:
every one is a dtype or indexing change on a value that was already exact, and the GPU tests below
check bit-equality against the old expression.

The 961 → 803 s TTFT between the second and third serve (same prompt, same seed, −16 %) is the only
speed change; the kt scale fix removes two 234 MiB allocations per streamed layer per chunk (≈ 80 per
chunk, ≈ 10,000 over the prompt) that were being served under memory pressure, where the caching
allocator frees and retries. Interpretation, not profiled. On the 1M pool the 16K / 32K prefill figures (1,659 / 1,657) match
the 512K rung; on the 64K × 4 default they read 1,887 / 1,901 against d-v (i)'s 1,512 / 1,580
(table below) — the same mechanism with more headroom to give back.

## Gates on the final overlay (`1m/`)

| gate | comparator | measured | verdict |
|---|---|---|---|
| (b) stress 1M − 2K (`stress-kv-full.py --concurrency 1 --prompt-tokens 1046528 --seed 31`) | d-v (i): OOM at ≈ 680K | **1,019,251-token prompt completed**, TTFT 802.8 s (≈ 1,269 tok/s averaged over the prompt), 256 tokens out, min free **547 / 574 MiB**, peak used 97,340 / 97,313 MiB | **pass** |
| (c) needles (`verify.py --context 1048576`) | 512K rung: 6/6 at 443.6K | **6/6 at 887,387 tokens**, TTFT 550.3 s, `finish_reason` stop; models alias / ceiling 1,048,576 / arithmetic / thinking toggle / tool call all PASS | **pass** |
| (d) NLL corpus (`nll-score.py --corpus` vs t328 default) | 0 moved expected (same shape) | **0 of 65,653 moved**, every text ΔNLL −0.00000, max |Δlogp| 0.000 | **pass** |
| (f) prefill 16K / 32K (`bench-prefill.py --runs 3 --seed 777`) | 512K rung 1,650 / 1,669; d-v (i) 64K × 4 1,512 / 1,580 | **1,658.9 / 1,656.9 tok/s**, TTFT 9.64 / 19.33 s | within 1 % of the 512K rung: **pass** |
| (f) decode C1 (`bench-decode.py --concurrency 1 --runs 6 --warmup 1 --distinct-prompts --seed 4242`) | d-v (i) 41.50 / 39.59 | **41.31 summed / 39.41 wall tok/s**, TTFT 0.617 s | within the T3.19 band: **pass** |
| (e) health (`health-hits.txt`), EDAC (`edac.txt`) | 0, 0/0 | **2 hits, both the benign `[sm120-split] sealed page-split buffers` lines**; 0 NaN / Xid / Traceback / OOM; EDAC 0/0 | **pass** |

Pool 1,064,960 tokens; `launch-container.txt` is the full container log (redacted: `~/`,
`<docker-bridge-ip>`); `manifest.json` is the T3.19f manifest (hashed — do not edit).

## 64K × 4 on the launcher default (`64k4/`) — same-shape insurance

The 1M serves exercise bs-1 decode and single-request prefill only; the deployment profile captures
bs 1..4 and runs four-stream prefill. One serve, every gate on the T3.29 list. Capture-end avail
4.55 / 4.61 GB (d-v (i): 4.61 / 4.55 — unchanged, as expected: the three fixes are transients, not
residents).

| gate | comparator (d-v (i) `64k4/`, overlay `2323052b`) | measured (`1eeb48cd`) | verdict |
|---|---|---|---|
| (b) stress 4 × 62K (`stress-kv-full.py --concurrency 4 --prompt-tokens 63488 --seed 31`) | all finished, wall 175–184 s, min free **1,119 / 1,062 MiB** | all four finished, wall 183.8 s (TTFT 62.8 / 101.1 / 133.8 / 166.1 s), peak used 96,096 / 96,141 MiB, min free **1,791 / 1,746 MiB (+672 / +684 MiB)** | **pass** — the removed transients show up as margin on the deployment profile |
| (c) needles (`verify.py --context 65536`) | 6/6 at 55,206 | **6/6 at 55,206 tokens**, TTFT 29.4 s; alias / ceiling 65,536 / arithmetic / thinking toggle / tool call PASS | **pass** |
| (d) NLL corpus (`nll-score.py --corpus` vs t328 default) | 0 moved | **0 of 65,653 moved**, every text ΔNLL −0.00000 | **pass** |
| (f) prefill 16K / 32K (`bench-prefill.py --runs 3 --seed 777`, idle server) | 1,512 / 1,580 tok/s (t328: 1,504 at 16K) | **1,886.8 / 1,901.1 tok/s** (runs 1,878 / 1,888 / 1,887 at 16K), TTFT **8.51 / 16.85 s** | **+25 % / +20 %** — unexpected; see below |
| (f) decode C1 / C4 (`bench-decode.py --distinct-prompts --seed 4242`, 6 / 3 runs) | 41.50 / 39.59 wall; C4 (d-ii) 56.32 / 54.16 wall | C1 **41.61 / 39.69 wall**, TTFT 0.616 s; C4 **56.20 / 54.03 wall**, TTFT 1.53 s | unchanged: **pass** |
| (e) health, EDAC | 0, 0/0 | 2 hits, both the benign sealed-buffer lines; 0 NaN / Xid / Traceback / OOM; EDAC 0/0 | **pass** |

The prefill reading is the one number this part did not predict. The three fixes move bytes, not
FLOPs, and the 1M-pool serve on the same overlay read 1,659 / 1,657 — flat against the 512K rung. The
difference between the two serves is free VRAM at rest (4.55 GB at 64K × 4 vs 3.25 GB on the 1M pool):
with ≈ 0.87 GiB less chunk-proportional transient, the chunk-8192 set now fits the 64K × 4 headroom
without the caching allocator freeing and retrying, which is the same unprofiled mechanism E34b met
(+24 % when the 1 GiB + 256 MiB retries stopped) and the 961 → 803 s TTFT above. Interpretation, not
profiled: a `torch.cuda.memory_stats()` `num_alloc_retries` reading on both profiles would settle it
(one serve, noted under Part d-v (a)). The number itself is three tight runs on an idle server after
the C4 block, the same protocol as every prefill figure since E34b-ii; it equals E34d-iv's chunk-16384
reading (1,838 at 205 / 179) without that run's −4.4 % C1 cost.

## Tests added (recipe, run in-image on GPU 0 by `scripts/tests/run-prefill-transient-tests.sh`)

- `scripts/tests/test_indexer_k_gather.py` 3/3: packed gather and dequant bit-equal to the inline old
  reference (page-boundary, last partial page, unsorted and repeated slots, n up to 100,000, int32
  slots); transient 16 B/slot vs the reference's 1,068 B/slot; mutant (off-by-one page offset) caught.
- `scripts/tests/test_hc_combine_input.py` 1/1: `hc_combine` with bf16 vs fp32 `pre_hc_head` at rows
  1 / 37 / 8192 bit-equal.
- `scripts/tests/test_kt_prefill.py::test_e8m0_from_bf16_bits` (CPU, in `run-kt-prefill-tests.sh`,
  10/10): exhaustive over all 65,536 bf16 bit patterns against the old expression; mutant (shift 6)
  caught.

## Verdict — **1M × 1 is qualified on the 219 / 165 + chunk 8192 default at `MEM_FRACTION_STATIC=0.962`; idea 8 not needed**

- Part d-v (ii) Go met on the letter (stress completes, needles 6/6, NLL 0 moved, prefill within 2 %).
  The Stop clause ("the OOM moves to another prefix-linear site") fired twice on the way and was
  answered in place: both further sites were chunk-proportional, not prefix-linear, and each was one
  function.
- This is also the first instalment of Part d-v (a) — the owner's choice for 262K × 4: of the four
  transients that run named by size (320 / 400 / 640 / 234 MiB), the 640 and the 234 are gone, which
  is ≈ 0.87 GiB of the ≈ 3 GB that run needed against 2.9 GB after bs-4 capture. The 320 and 400 MiB
  requests are still unnamed. A 262K × 4 retry on `1eeb48cd` is the cheapest next datapoint.
- Margin at 1M is thin (547 / 574 MiB at the peak) and the 1M pool is a per-rung override
  (`CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962`), not a default; the
  launcher guard still refuses more than one slot above 64K without `ALLOW_UNTESTED_FIT=1`.
- Not re-run: 128K / 262K / 512K × 1 (the 1M × 1 pass covers them on the memory axis — same transients,
  larger pool — and the other gates are shape-independent; no default change is proposed, so T3.29's
  "re-run 512K before a default change" rule is not triggered).

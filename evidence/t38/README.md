# T3.8 — per-(layer, expert) routing counts on the served V4.1 hybrid

Question: how much of the routed-expert traffic does the positional 219/165 split (expert ids
0–218 on the GPU, 219–383 on the CPU tier) send to the CPU tier on real traffic, decode and
prefill separately, and how much would a matched per-layer map save? Reference: the el8 /
0xBakeer published counts recomputed at our budget (`plan/ROADMAP.md` 2026-10-10 research entry).

Profile for both rounds: launcher defaults (overlay `99128a7e` + the recorder fix below, 219/165,
D=4, chunk 8192, 64K × 4, `KT_CPUINFER=16`) plus `--expert-distribution-recorder-mode stat`.

## Round 1 (2026-10-10, one serve) — every dump all zeros

`round1/`: three sets, 53 dumps, `logical_count` zero in every pass. Raw `.pt` files removed from
the tree (they carried no information; still in git history); `round1/analysis/routes.py` is the
first version of the analysis. Three diagnostic serves (`round2/analysis/diag-recorder.sh`, env-gated
logging in the overlay, not kept) showed `_on_forward_pass_end` firing with zero hooks and
`TopK.forward_cuda` / `select_experts` never entered: **DeepSeek-V4.1 routes through
`sglang/srt/multimodal/dsv41/vl_routing.py::vision_topk`** (the gate carries
`e_score_correction_bias_vl`, so `DeepseekV2MoE.forward_normal` takes the vision branch for
every token), and `vision_topk` bypasses `select_experts`, hence never called
`on_select_experts`. Not a hybrid / kt problem — stock upstream behaviour for this model.

Fix: overlay `755fe3f0` (`dsv41 vl_routing: report routed ids to the expert distribution
recorder`): both branches of `vision_topk` report the routed columns (not the fused shared
slots) to the recorder. No-op when the recorder mode is unset (the Noop recorder), a graph-safe
`scatter_add_` when it is set. No qualification serve needed for the default profile (the call is
`pass` without the flag); every later serve carries it.

## Round 2 (2026-10-10, one serve, 10 min) — counts

`round2/histogram.sh`: warm, then four recorded sets with a 10 s dump loop (the 1000-pass buffer
never wraps; `dump_record` resets the buffer, so dumps do not double count):

| set | traffic | passes (decode / prefill) | tokens |
|---|---|---|---|
| `humaneval-c4` | HumanEval generate, 40 problems, C4, ≤ 2048 tokens (real code decode + 40 short prompt prefills) | 1,519 / 38 | 5.9K decode, 5.6K prefill |
| `decode-distinct` | distinct prose / code fixtures, C1 and C4 × 3 runs × 256 tokens | 1,646 / 8 | 4.3K decode, 1.7K prefill |
| `prefill-random` | one 16K `bench-prefill.py` prompt (pseudo-random words; el8 set C analogue) | 1 / 2 | 16K prefill |
| `prefill-text` | the NLL corpus, 8 in-house texts, 65.6K tokens, prefill-only (el8 sets A/B analogue) | 10 / 16 | 85K prefill |

Each dump is the all-reduce sum of both TP ranks, which see the same routes, so raw counts are
2× the real route count (the analysis halves them; shares are unaffected). A pass with more than
4 × 6 × 2 routes per layer is a prefill chunk (decode batches are ≤ 4 tokens), which is how the
short prompt prefills inside the decode sets are separated.

`round2/analysis/routes.py` (in-image, torch) → `routes.txt`; `counts-and-maps.json` holds the
halved [40 × 384] counts per set/phase and every candidate map (40 lists of 219 GPU expert ids);
`../el8-counts/counts-{A..E}.json` are the published el8 counts (A/B prefill text, C random ids,
D prose decode, E code decode) used for the el8 mix map (0.3 D + 0.3 E + 0.2 A + 0.2 B, each set
normalised per layer, top-219 per layer).

### Results

**The positional split is no better than a random one.** CPU share of routes under ids ≥ 219:
42.4 % (humaneval decode), 42.8 % (decode-distinct), 43.8 / 43.4 % (their prompt prefills),
43.6 % (prefill-text), 46.4 % (random words) — the uniform expectation is 165/384 = 43.0 %.
Per layer it ranges 21–59 %.

**Routing is heavily skewed, per layer and per phase.** 50 % of a layer's routes land in 20–32
experts (median; range 10–68), 90 % in 112–149; the busiest expert carries 17–30× the mean.
A per-layer top-219 map fitted on the same traffic ("own oracle") leaves 1.2–3.9 % on the CPU
tier; on random words 0 % (16K tokens hit only 162 experts per layer).

**CPU share of each candidate map on each set** (rows = maps, columns = held-out and fitted
sets; the fitted cell is in parentheses):

| map | humaneval decode | humaneval prefill | decode-distinct decode | decode-distinct prefill | random prefill | prefill-text |
|---|---|---|---|---|---|---|
| positional 0–218 | 42.4 | 43.8 | 42.8 | 43.4 | 46.4 | 43.6 |
| decode-fit (humaneval decode) | (1.6) | 7.8 | 37.5 | 23.7 | 30.6 | 41.1 |
| prefill-fit (prefill-text) | 35.4 | 33.1 | 29.3 | 29.9 | 38.9 | (3.9) |
| mix 0.6 decode-fit + 0.4 prefill-fit | (4.2) | 9.2 | 29.1 | 20.3 | 28.2 | (8.0) |
| **el8 mix** (published counts, none of ours) | **15.3** | **12.2** | **14.5** | **14.3** | 23.4 | **15.1** |

- Our own fitted maps overfit: 6K decode tokens are far too few (36K routes per layer over 384
  experts, and the long tail decides the 219th slot), so decode-fit on held-out decode is 37.5 % — barely
  better than positional — and cross-phase maps (decode map on prefill text 41 %, prefill map on
  decode 29–35 %) confirm the research note that a foreign-phase map is nearly positional.
- **The published el8 mix map, which uses none of our traffic, lands at 12–15 % on every real-text
  set of ours, decode and prefill alike** (23 % on random words), i.e. **2.8–3.5× fewer CPU-tier
  routes than positional**: at C1 decode 6 × 42.4 % = 2.5 CPU-expert reads per token today vs
  6 × 15 % = 0.9. Symmetric check: our maps on el8's counts give 22–45 % (overfit again), the
  el8 mix on its own sets 9–18 %. The el8 and our-mix maps share 77 % of their GPU slots; both
  share 57 % with positional (= chance).
- Decode and prefill maps overlap 65 % with each other, so one map can serve both phases if it is
  fitted on enough of both (the el8 mix is 60 % decode / 40 % prefill by weight).

### What this means for placement (T3.8b, owner decision)

A per-layer remap at load time (hot experts get ids < 219 so the positional kt contract stays
untouched; the gate's logit columns, the correction biases and the expert weight order permuted
per layer) would cut the CPU tier's decode reads ≈ 2.8× on real text with no VRAM cost and no
change to prefill bytes (the 165 streamed experts ship whole regardless of routing). Gate it on
C1 wall + kt CPU-tier bytes per step on held-out traffic, not on histogram shape (the benchmarks'
random-word prompts understate it: 23 % vs 15 %). Start from the el8 mix map; refit on our own
counts only once a soak has recorded ≥ 100K decode tokens per phase (the 10 s dump loop on a
served soak gives that for free now that the recorder works).

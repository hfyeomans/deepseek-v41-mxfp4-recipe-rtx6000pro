# External recipe evaluation: mlx-stream / EXL3 3.0 bpw on a 128 GB Mac vs this hybrid (T3.25, 2026-10-07)

An evaluation of where this recipe stands, where it is headed and whether it is on the right
track, against the one other published DeepSeek-V4.1-Flash recipe that solves the same problem
("the routed experts do not fit"): David Tai's **mlx-stream** plugin for mlx-serve, serving the
**OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw** repack on an Apple M5 Max with
128 GiB. Written at a pause between T3.19a and T3.19b, from the published model card, the plugin
README, the author's article, and this repository's evidence. **Nothing from the other recipe was
run here.**

Labels: **measured** = this repository's evidence; **published** = the other recipe's own numbers
on its own hardware, as printed by its author (not reproduced); **derived** = arithmetic from
either; **inferred** = a conclusion from reading, not from a run.

Sources (external):

- Model card: <https://huggingface.co/OpensourceWTF/DeepSeek-V4.1-Flash-streaming-repack-exl3-3.0bpw>
- Expert quantization it repacks: <https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw> (ExLlamaV3 v1.4.2, 3.02 bpw decoder average, `--hq`, 250 x 2048 calibration rows)
- Plugin: <https://github.com/davidtai/mlx-stream> (README, `docs/lookahead-predictor.md`); host <https://github.com/ddalcu/mlx-serve> (PR #749)
- Author's article (X, 2026-10-07): "DeepSeek-V4.1-Flash streams its experts from SSD" — the lookahead predictor write-up; the text was supplied by the owner because the article body is login-gated
- Prior art the article cites: Eliseev & Mazur arXiv:2312.17238 (next-layer gate on the current hidden state), ProMoE arXiv:2410.22134, HOBBIT arXiv:2411.01433

## 1. Verdict in one paragraph

The two recipes make opposite quantization choices and land at the **same single-stream decode
rate**: theirs 36–40 tok/s **published** with lossy 3.0 bpw experts *and* DSpark speculation on a
~134 ms step (**derived**); ours 42–42.6 summed / 37–40 wall tok/s **measured** with lossless
MXFP4 experts, speculation off, on a 23.6 ms step. Ours is ahead on everything around decode:
TTFT (0.6 s vs 12.7 s at ~1K tokens), prefill (845–1,375 vs 613–669 tok/s at 16K), concurrency
(C4 55–57 wall vs no published multi-stream number), and quality (exact experts vs a lossy
requantization of FP4-trained weights). Theirs is ahead on supported context (1M, slowly, vs our
32K qualified) and on a public task-level quality number (HumanEval 158/164). The comparison
confirms the project's direction — keep the weights exact, shrink or hide the CPU tier, refactor
the seams before changing ownership — and adds three concrete items: a route-lookahead probe for
the CPU tier (E31), a HumanEval comparability run (E32), and a decode-vs-context ladder folded
into the 262K qualification. It does **not** justify reopening sub-4-bit experts on this box
(§5), and it does not change the G1 → G6 gate sequence.

## 2. The two designs side by side

```diagram
 mlx-stream (Mac, 128 GiB unified)             this recipe (2x RTX PRO 6000 + TR 9955WX, 246 GiB)
 ┌─────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
 │ trunk resident (mxfp8/mxfp4 exact)  │        │ trunk + 226/384 experts per layer resident   │
 │ 165 of 384 expert slots per layer   │        │ in VRAM, MXFP4 exact (TP2)                   │
 │ EXL3 3.0 bpw, 13.3 MB per record    │        │ 158/384 experts per layer in DRAM, compact   │
 │ decode step ≈ 134 ms, DSpark K=5    │        │ MXFP4, computed by kt-kernel on 16 cores      │
 │   ≈ 5.3 tokens per verify step      │        │ decode step 23.6 ms, speculation off          │
 └───────────────┬─────────────────────┘        └───────────────┬──────────────────────────────┘
                 │ misses + 2 records lookahead                  │ 40 per-layer export/wait, D=4 deferral
 ┌───────────────┴─────────────────────┐        ┌───────────────┴──────────────────────────────┐
 │ SSD: experts.bin 204.5 GB           │        │ DRAM: 135 GiB CPU experts (E25: 125 GB/s     │
 │ up to 13.8 GB/s, uncached records   │        │ ceiling = 2 CCD links, not DIMMs)             │
 │ Engram tables 2 x 101.4 GB by offset│        │ NVMe: Engram 188.8 GiB file-backed, mmap      │
 └─────────────────────────────────────┘        └──────────────────────────────────────────────┘
```

| Aspect | mlx-stream / EXL3 repack (**published** unless noted) | this recipe (**measured** unless noted) |
|---|---|---|
| Hardware | Apple M5 Max, 128 GiB unified; SSD up to 13.8 GB/s | 2x RTX PRO 6000 Max-Q (96 GB each), TR PRO 9955WX 16c/2 CCD, 8x DDR5-6400 (246 GiB), NVMe |
| Routed experts | EXL3 3.0 bpw trellis (K=3, mul1, Hadamard) — **lossy** requantization of the release's FP4 experts; 13,316,096 B per record | release MXFP4 **exact**; ≈ 18 MiB per expert GPU-side (**derived**: 3 x 5120 x 2304 x 4.25 bpw), 21.8 MiB compact CPU view (E17) |
| Where experts live | 165–166 of 384 per layer in memory (decode, 16K profile), 135 during the prompt pass; the rest streamed from SSD per step | 226 of 384 per layer in VRAM (59 %), 158 in DRAM computed by the CPU; nothing streamed at decode |
| Trunk, MTP, Engram `wkv` | exact repacks (mxfp8 / mxfp4) | release FP8 / FP4 as shipped; dense W8A8 FP8 GEMMs tuned for SM120 (E20) |
| Engram tables | lossless, on SSD, 264-byte rows read by offset | lossless, file-backed on NVMe via mmap/page cache (T3.12 decision); 2 graph breaks per step, 1.82 ms exposed (T3.19a) |
| Speculation | DSpark in-checkpoint, on; 4.89 accepted per draft attempt (HumanEval), ≈ 5.3 tokens per verify step | DSpark measured and **off** (E23: K=2 parity at C1, −57 to −63 % at C3/C4; K=5 −16 % C1) |
| Decode, single stream | 39.6 tok/s at 1K ctx, 38.0 at 4K, 36.4 at 16K and 32K, 31.4 at 128K, 26.7 at 256K, 11.0 at 1M | C1 42.0–42.6 summed / 37.2–40.1 wall at short prompts (E28 / soak fixture, slot 3 BIOS); no decode-vs-context ladder yet |
| Decode, concurrent | not published (HumanEval aggregate 31.8 tok/s is sequential requests) | C2 50–52 / 48–50 wall, C3 55–56 / 54, C4 58–59 / 56–57 wall (13.9 per stream) |
| Prefill / TTFT | 1K: 80.9 tok/s, 12.7 s TTFT; 4K: 265 tok/s, 15.5 s; 16K: 613–669 tok/s, 24.5–26.7 s; 32K: 583 tok/s, 56 s; ~200-token prompts median TTFT 7.65 s | short prompts TTFT 0.61–0.62 s; 16K 845–1,375 tok/s, TTFT 11.6–18.9 s; 32K 811–870 tok/s, 36.7–38.4 s (E22/E26) |
| Context | up to 1,048,576 supported; rows traded for KV | 32K qualified (147,456-token KV pool); 262K qualification pending (T3.6e–g) |
| Quality evidence | HumanEval pass@1 0.9634 (158/164), greedy, stable across two builds; no KL / NLL / perplexity vs reference published | teacher-forced NLL on a frozen 4,795-token corpus, bit-identical across relaunches, gate < 0.002 vs the D=0 baseline; 16K needles 3/3; 141K-token soak 0 NaN (E27); no task-level benchmark |
| Peak memory | 106–108 GB of 128 GiB | GPUs 0.96 static fraction; host ≈ 155 GiB process + page cache |
| Engineering shape | Zig plugin behind an `sdk_ext` seam; itemized memory bill before any allocation; per-record sha256 receipts; Metal kernels pinned behind a manifest hash; evaluation levers off by default | Python SGLang overlay (rank-0 kt path, breakable CUDA graphs); fit gate; env-var launch (T3.24 replaces with typed profiles); T3.19 introduces the `CpuExpertProvider` / `BufferLease` seams; evidence manifests in T3.19f |

## 3. What the comparison says about our decode step

**Per step, we are ~5.7x faster; they recover the difference with speculation (derived).**
39.6 tok/s at ≈ 5.3 tokens per verify step is ≈ 7.5 verify steps/s ≈ 134 ms per step. Ours is
23.6 ms (42.4 steps/s, T3.19a). Their step is dominated by SSD reads that happen once per verify
step whatever the number of drafted tokens, so five tokens per step are nearly free. Ours is not:
T3.19a shows `cpu_wait` is the only bucket that scales with tokens per step (7.9 ms at bs 1 →
48.6 ms at bs 4) while the GPU-side buckets stay flat. That is exactly why E23's K=5 lost 16 % at
C1 and 69–77 % at C3/C4 here and wins there. The E23 verdict stands; it is a property of a
compute-bound CPU tier, not of DSpark.

**The fixed part of our step is as large as the CPU tier (measured).** At bs 1, other_gpu 8.1 ms
+ Engram gaps 1.8 ms + export 2.5 ms + resident GEMM 2.5 ms ≈ 14.9 ms of the 23.6 ms step do not
grow with tokens. Speculation would amortize those; it only loses because the CPU-tier part grows
faster than acceptance pays. The arithmetic for "speculation becomes a win" is therefore the same
as the roadmap's existing rule: re-open E23 only after a CPU-tier change (rank split T3.22, third
GPU, more CCDs) — unchanged, now with a per-bucket reason.

**Their predictor attacks a problem we do not have at prefill, and only a small one at decode
(inferred, projected).** Their prompt pass refills 135 slots per layer from SSD and used to wait
11.5 s of 42.6 s on reads; running the router before attention lets 81 % of those reads start
early (+10.3 % prefill in the release that added it, confounded with a third read window). Our
streamed prefill (E22) exports *all* 158 CPU experts of a layer to the GPUs for every ≥ 1,024-token
chunk because a chunk that size routes to nearly all of them — there is no guess to make. At
decode the analogous lever would be to score layer L+1's router on layer L's MoE input and hand
kt-kernel the predicted CPU experts early. Our CPU tier is throughput-bound (DRAM read at the
per-CCD ceiling, E25; D=4 already queues four layers of work), so earlier starts only help where
the pool is idle: the three segment starts after Engram breaks (T3.19a: layers 0/1/14 are cheap
because the pool drained its backlog while the GPU sat in the gap) and the D=0 last layer
(0.39 ms). Upper bound ≈ 2 ms of the 23.6 ms step (≈ 8 %) at perfect prediction; a wrong guess
costs CPU compute, our scarce resource, where theirs costs SSD bandwidth they have spare. Worth a
cheap accuracy probe (E31), not a build.

## 4. What the comparison says about our prefill and context

Prefill and TTFT are where the GPU box pays for itself: 0.6 s vs 7.6–12.7 s TTFT on short prompts,
and 1.4–2.2x their 16K prefill rate. Nothing to borrow; the streamed layerwise prefill (E22) and
the SM120 FP8 tuning (E20) are the reasons.

Context is where they are ahead in *coverage*: a published ladder from 1K to 1M with decode and
TTFT at every rung. We have 32K qualified and a 262K qualification task without a decode ladder.
Their table is a useful shape to copy: decode tok/s and TTFT at 1K, 4K, 16K, 32K (now) and 64K,
128K, 262K (when the KV pool allows). Folded into T3.6e–g below.

## 5. Quantization: should we revisit sub-4-bit experts? No — with the numbers

FEASIBILITY Option C rejected sub-4-bit experts for three reasons: no SM120 kernels in SGLang for
a `deepseek_v41` MoE in a trellis format, unknown quality on experts that DeepSeek trained
FP4-aware, and "3.0 bpw ≈ 190 GiB still does not fit". The published repack changes none of them
on this box; it does sharpen the numbers:

| Question | Answer | Label |
|---|---|---|
| Does an EXL3 3.0 bpw bank fit two 96 GB cards? | No. 204.5 GB (190.5 GiB) of experts alone exceeds 2 x 96 GB; the model card confirms 13.3 MB x 15,360 records | published, derived |
| How many experts per layer would fit in the VRAM our 226 MXFP4 experts use? | ≈ 158 GiB / (12.7 MiB x 40) ≈ **318 of 384** → a CPU tier of ≈ 66 experts per layer instead of 158 | derived from a derived 18 MiB per MXFP4 expert; ± 10 % |
| What would that buy? | cpu_wait scales with CPU experts per token: 7.9 → ≈ 3.3 ms at bs 1, 48.6 → ≈ 20 ms at bs 4 if linear; step 23.6 → ≈ 19 ms (≈ 50 tok/s C1 wall), bs 4 ≈ 40 ms (≈ 100 tok/s aggregate) | projected, optimistic: ignores trellis dequant cost |
| What would it cost on the GPU? | EXL3 weights are decoded on the SMs before every GEMM. The GLM-5.3-Flash EXL3 recipe on these cards pins both at 300 W from C2 and runs 0.23 tok/s/W vs 0.46 for a tensor-core-native format (E14); the hybrid today peaks at 183 W. Expect the resident-expert bucket (2.5 ms at bs 1) to grow several-fold and power to bind | measured (GLM), inferred (transfer) |
| What would it cost in quality? | Unknown. The only public number is HumanEval 158/164 greedy for the repack; no KL, NLL or perplexity against the release exists. The Mia-AiLab card says its experts are "mostly K=3" with attention/shared/Engram `wkv` at 4–6 bits — the lossy part is exactly the part we keep exact | published |
| What would it take to build? | An SM120 EXL3 grouped-MoE path inside SGLang for `deepseek_v41` (the GLM recipe needed a bespoke ExLlamaV3 fork), a loader for the Mia-AiLab shards plus the release's Engram shards 47/48, and a new placement/fit gate. Months, per the GLM precedent | inferred |
| Could we at least measure the quality? | Not cheaply: ExLlamaV3 cannot hold 205 GB on two cards and has no CPU offload; the Mac stack is Metal-only. A quality number would come only from the full build above | inferred |

**Decision:** sub-4-bit experts stay rejected on this box. The project's quality principle
(experts exact, approximations gated by the NLL corpus) is the one thing the other recipe cannot
claim, and the throughput case is at best +25 % C1 for a build of months with a power-bound GPU.
**Re-open only if a third card lands** (even then the exact MXFP4 bank alone is ≈ 270 GiB against
268 GiB of VRAM, so a third card still keeps a CPU tier — HETEROGENEOUS-ROUTING §2 — and a 3.0 bpw
bank would be the first format that makes the whole model GPU-resident on three cards) or if someone publishes an SM120
`deepseek_v41` EXL3 MoE kernel with a KL-vs-release number. Recorded as option O in the ROADMAP
options register.

## 6. What to borrow

| Idea (theirs) | Our analogue | Action |
|---|---|---|
| Next-layer router on the current hidden state to issue expert reads early (Eliseev & Mazur) | hand kt-kernel predicted CPU experts for layer L+1 during layer L; fills CPU idle at segment starts and layer 39 | **E31** — accuracy probe first, opt-in log like `KT_TIMELINE`; build only if top-6 hit rate ≥ 90 % *and* measured CPU-pool idle ≥ 1.5 ms/step |
| Prompt-pass expert frequency counting | we already export every CPU expert per prefill chunk | none |
| HumanEval pass@1 greedy as the published quality number | NLL corpus (private, relative); no absolute task number | **E32** — run HumanEval greedy through the served endpoint; also gives a public point of comparison for the exact-vs-3.0 bpw question |
| Decode / TTFT ladder by context length | none | fold into the 262K qualification (T3.6e–g) |
| Itemized memory bill before any allocation; refuse a request the bill does not cover | fit gate + `ALLOW_UNTESTED_FIT` override | already planned: T3.24 removes the override; T3.19e typed config carries the bill |
| Per-record sha256 receipts; kernel texts pinned behind a manifest hash | PROVENANCE.md, planned T3.19f evidence manifests | confirms T3.19f's scope (hash every claimed artefact, including kernel configs) |
| One typed seam (`sdk_ext`) between host and arch; nothing per layer crosses it | T3.19b `CpuExpertProvider` / `BufferLease` | confirms the G1 refactor direction; keep the per-layer path inside the provider |
| Evaluation levers compiled out of the served build | env flags read at construction, off by default | T3.24's deprecation list stands; consider a "served profile refuses unknown knobs" assertion in T3.19e |

## 7. Are we on the right track?

Yes, with two qualifications.

- **Right:** exact weights; attack the CPU tier by placement (E28), hiding (D=4), seams (T3.19) and
  then ownership (T3.22); speculation off until the CPU tier changes; prefill on the GPUs. The one
  independent recipe for this model on consumer-class hardware reaches the same decode rate only by
  giving up expert exactness and spending 5x the per-step time on speculation; it is 10–20x behind
  on TTFT.
- **Qualification 1 — we lack a public quality number.** The NLL gate is strong for A/B but says
  nothing to an outsider. E32 fixes that for ≈ 20 min of GPU time.
- **Qualification 2 — we lack a context ladder.** 32K qualified is honest but thin against a
  recipe that publishes 1K–1M. The 262K work should produce the ladder, not just a pass/fail.

Nothing here moves T3.19b. E31 and E32 are low-coupling, opt-in and cheap; they ride on G1 gate
serves or on T3.20's window and do not change the G1 → G6 sequence.

## 8. Registered follow-ups

| ID | What | Where |
|---|---|---|
| T3.25 | this evaluation | TASKS.md (done) |
| E31 | CPU-tier route-lookahead probe: top-6 hit rate of layer L+1's router scored on layer L's MoE input (and on the pre-attention residual), per layer, bs 1 and 4; CPU-pool idle per step from the T3.19a timeline; go/no-go for a dispatch build | EXPERIMENTS.md |
| E32 | HumanEval pass@1 greedy via the served endpoint, scored offline in a sandbox; report decode tok/s and TTFT distribution alongside | EXPERIMENTS.md |
| T3.6e–g addendum | decode tok/s + TTFT ladder at 1K / 4K / 16K / 32K now, 64K / 128K / 262K at qualification | TASKS.md |
| Option O | EXL3 3.0 bpw GPU tier (Option C revisited): rejected on two cards, re-open conditions | ROADMAP options register |

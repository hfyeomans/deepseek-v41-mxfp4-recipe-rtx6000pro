# T3.22 design record — multi-rank CPU-expert ownership (Stage 3)

Status: **owner-approved 2026-10-10 — T3.22 closed by decision at this topology (ledger G4 record in the migration roadmap); not built.** Was: written 2026-10-10 for owner approval. Recommendation: close T3.22 by decision at this topology; do not build it. The design below is complete enough to build if the owner overrules, and it is the design a third card (T4.1b) would reopen.

Scope of the record (as the gate ledger requires): partition, reduction order, failure semantics, NUMA/CCD ownership, rollback — plus the premise check that the T3.27 timeline forces. Evidence cited is on the qualified profile (overlay `124ca1de`, image `dsv41-kt:recipe`, el8 map 219 GPU / 165 CPU experts per layer, `KT_CPUINFER=16`).

## 1. What T3.22 was for, and what has changed since it was written

T3.22 was registered (2026-09-24) when:

- rank 0's process ran at ≈ 1,880 % CPU and rank 1's at ≈ 120 % (CURRENT-ARCHITECTURE §1.8 "rank asymmetry");
- prefill export (`sync_write_weight_scale_to_buffer`) was a 4.6 s per-chunk, core-bound host job on rank 0;
- the Engram host table and its gathers were rank-0 work with two graph breaks of 2.4–3 ms per step;
- the stated benefit was "use both TP ranks' host-side capacity, reduce rank-1 spin, improve overlap, establish EP/third-card semantics".

Since then:

| Item | Then | Now (measured) | Where |
|---|---|---|---|
| Prefill export | 4.6 s per chunk of host CPU on rank 0 | **gone**: kt loads into memfd arenas, both ranks `cudaHostRegister`, GPU DMA on a side stream, zero host CPU per layer | T3.39 window 2, `evidence/t339/window2/` |
| Engram ownership | rank 0 gathers all rows | **already rank-sharded**: file layout maps the shard on every rank, each rank owns its row range, the existing sharded all-reduce reassembles the lookup (`layers/engram.py` "private"/"file" layouts) | T3.12 / T3.20, `evidence/t320/` |
| Engram breaks | 2.4–3 ms/step | 0.77 ms/step at bs 1 (4.5 %) | T3.27 w1 timeline |
| Exposed CPU wait, bs 1 | 7.88 ms | 2.19 ms (12.7 %) | T3.27 w1 |
| Exposed CPU wait, bs 4 | 48.6 ms | 13.1 ms (39 %) | T3.27 w1 |
| Decode "export" bucket (per-layer D2H of hidden/topk + kt submit host node) | — | 2.51 ms/step (14.6 %) at bs 1, 2.86 at bs 4 | T3.27 w1 |
| Rank 1 | spins in NCCL | still waits 13.4 of 17.2 ms (bs 1) in the all-reduce for rank 0 | T3.27 w1 rank-1 timeline |

Two of the three host-side jobs the split was meant to share no longer exist on the host. What is left on rank 0 and not on rank 1 is the decode CPU-expert path: D2H → kt submit → kt compute (16 threads) → D2H of the result → add.

## 2. Premise check: what a rank split can and cannot move

**The CPU tier is DRAM-fabric-bound, not host-core-bound.** E25 measured the two CCD→IOD links at 62–63 GB/s each, 125 GB/s with threads on both CCDs; E21 measured kt at 0.19 ms per 21.8 MiB expert ≈ 120 GB/s, i.e. already at that roof with 16 unpinned threads (the scheduler spreads 16 busy threads over the 16 cores, so both CCDs are already drawing). The host is one NUMA node (HARDWARE.md). Splitting the CPU experts between two processes puts the same bytes through the same two links; the aggregate rate cannot rise. **The bs-4 `cpu_wait` of 13.1 ms does not move.**

**The D2H is duplicated, not halved.** Both ranks hold the same hidden states and topk ids (TP-replicated after each layer's all-reduce); in a split each rank copies them to its own host buffer and submits its own subset. Per rank, the export bucket shrinks only by the per-expert part of the kt submit call, not by the copy. The step is max(rank 0, rank 1), so the saving is bounded by the submit share of the 2.51 ms, which the current marks do not separate (one extra log-only mark would).

**What a split can move at this topology**

| Lever | Ceiling at bs 1 (of 17.26 ms) | Realistic | Note |
|---|---|---|---|
| kt submit host node halved per rank | ≤ 1.25 ms (≤ 7 %) if submit were the whole export bucket | 2–4 % | D2H part is duplicated |
| Engram | 0 | 0 | already sharded |
| CPU compute / wait | 0 | 0 | fabric-bound, both CCDs already drawing |
| Rank 1 "utilisation" | cosmetic | — | nvtop % is NCCL spin; the all-reduce still waits for the slower rank |

Upper bound ≈ 7 % C1, ≈ 0 % C4. For comparison, T3.41 (an M ≤ 4 block-scaled FP8 GEMV kernel) has a ≤ 9 % C1 ceiling with no distributed failure modes, and the measured C4 knee (39 % exposed CPU wait) answers only to fewer CPU bytes (map, third card) or more fabric (CCD count).

**Cost** is unchanged from the roadmap: two kt instances, two arena owners, per-rank deferral ledgers and leases, a new rank-coupled failure surface at every MoE layer, duplicated staging, stricter graph lifetimes, and a harder diagnosis story — the largest architecture change in the project.

## 3. Recommendation

Close T3.22 **by decision** (owner approval, same template the ledger gives T3.23), with this record as the decision record:

```
Decision record — Stage 3 multi-rank CPU-expert ownership not justified at this topology
Date / recipe SHA / overlay SHA: 2026-10-10 / <this commit> / 124ca1de
Evidence: evidence/t327/window1/timeline/report.md (bs 1, bs 4), evidence/t339/window2/, E21, E25
Host-side work a split could share: kt submit host node, ≤ 2.51 ms/step (14.6 %) including the duplicated D2H
Work a split cannot share: CPU compute (fabric-bound at ≈ 120 of 125 GB/s), Engram (already sharded), prefill export (gone)
Ceiling: ≤ 7 % C1, ≈ 0 % C4; realistic 2–4 % C1
Verdict: below the cost of the change; keep rank-0 ownership
What would reopen it: a third card (T4.1b) that adds an expert owner with its own fabric; a CPU with more CCDs
   (9975WX/9985WX) where per-rank CCD pinning would add bandwidth; a measured submit share > 1.5 ms/step
Reviewer: owner
```

Consequences if approved:

- T3.23 (native consolidation) closes by decision too: the T3.27 timeline already names every Python coordination cost on the decode path — 0.08 ms between replays, and the 2.51 ms export bucket, which is CUDA memcpy + a kt C++ call, not a Python loop. No loop crosses the ledger's 5 % / 1 ms bar.
- T3.13 / C10 (fail the request, not the server, on a NaN row) stays required: it is the Stage 4/6 release gate, not only T3.22's entry.
- The sequence becomes T3.13 → T3.24 (packaging, typed profiles, unsafe-flag retirement) → T3.30 (deployment snapshot), with T3.41 / T3.35 / T3.8c / T3.40 low.
- One optional cheap check before closing, if the owner wants a number instead of a bound: add a log-only `d2h` timeline mark between the copy and the kt submit (same mechanism as the T3.19a marks, no behaviour change), one instrumented serve, and read the submit share. If it is under 1.5 ms/step the bound above stands.

## 4. The design, if built (kept for T4.1b)

### 4.1 Partition

- Static, deterministic, generation-stamped. The el8 map already gives the compact CPU view `0..n_cpu-1` (`kt_remap.py`); a split assigns compact ids `[0, n0)` to rank 0 and `[n0, n_cpu)` to rank 1 with `n0 = ceil(n_cpu/2)` per layer, or, better, a per-layer balanced split by the same route-count histogram the map was built from (`evidence/t38b/` counts) so the two ranks carry equal expected bytes, not equal expert counts (target: no rank above 55 % of CPU-expert bytes under the recorded traffic).
- Each rank runs its own `KTMoEWrapper` over its subset only; `remap_topk_ids_for_cpu` gains a rank window (ids outside the rank's range → −1, the value kt already skips). The GPU mask is unchanged.
- The map file carries both the GPU/CPU split and the rank split; `KT_EXPERT_MAP` generation is hashed into the `BufferLease` so a stale map fails at the first submit, not silently.
- Weights: each rank creates the memfd arenas for its own subset (today `create_arena` is rank-0-only and rank 1 opens rank 0's through `/proc/<pid>/fd`; it becomes symmetric). Prefill streaming is unchanged: every rank still opens every arena and streams its TP slice of all CPU experts. Host RAM is unchanged (same bytes, different owners).

### 4.2 Reduction order

- No new collective. Today `output = gpu_partial + cpu_all` on rank 0 and `gpu_partial` on rank 1, then the layer's TP all-reduce sums the two. In the split each rank adds its own CPU partial before the same all-reduce: `(gpu0 + cpuA) + (gpu1 + cpuB)`.
- This is not bit-identical to rank-0 mode (different association), which the ledger allows; it must be run-to-run bit-identical, which the two-rank NCCL sum is (a + b with a fixed rank order). Accumulation dtype stays bf16 at the add as today; the kt output is bf16.
- Deferral D=4 stays per rank: each rank's `ForwardLedger` tracks its own deferred slots; the forward-end check runs on both ranks.

### 4.3 Failure semantics

- Per-rank kt timeout → that rank writes NaN into its partial for the affected rows → the all-reduce propagates NaN to both ranks → C10's per-request rejection fails that request only (this is why C10 is an entry gate).
- Rank loss → NCCL abort → process exit with a named cause (no partial fallback mid-step; a request may fall back only before execution starts).
- Stale map / wrong generation → `KTContractError` at submit, server refuses the batch.
- Buffer overrun → the existing `BufferLease` address assert, extended per rank.
- Capture: no allocation inside capture on either rank (T3.16a/c asserts run on both).

### 4.4 NUMA / CCD ownership

- The host is one NUMA node; per-CCD ownership needs either explicit affinity in kt's pool (`threadpool_count` / `numa_nodes` exist in `kt_kernel/experts.py` but are keyed by NUMA node) or the BIOS "L3 cache as NUMA domain" option, which presents each CCD as a node. Either way this is a bandwidth-neutral change here (both links are already drawn by 16 unpinned threads); it would matter on a CPU with more CCDs than busy threads.
- Rank 0 → CCD 0 (cores 0–7), rank 1 → CCD 1 (cores 8–15), 8 kt threads each; the HSMP DDR counter (`scripts/hsmp-clocks.py --watch`) is the balance gate.

### 4.5 Rollback

- `KT_RANK_SPLIT=0` (launch-time only) = today's rank-0 ownership from the same overlay commit; the launcher refuses the knob unless the map file carries a rank split. Switching modes is never a runtime action.

### 4.6 Gates (unchanged from the ledger)

Correctness on recorded tensors at bs 1..4; 100 repeated decodes bit-identical; injected timeout/NaN fail one request; CPU wait per rank within 20 %; capture/replay 1 2 3 4 per rank; rank-0 mode launchable; C1/C4 wall ≥ baseline; 20K soak 0 NaN/Xid; exit = T3.16i pass 3 with a per-layer timeline.

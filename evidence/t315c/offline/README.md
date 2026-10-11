# T3.15c offline probe — when does the DSV4 decode top-k pick a run-dependent SET?

Run inside `dsv41-kt:recipe` with the overlay mounted, one GPU, no server (same `docker run` line
as `evidence/t315b/offline/`). 2026-10-11, overlay `743f8fa9`, image `314749b5a268`.
Output: `topk-v2-ties-probe.txt`.

T3.15b showed `topk_v2` (`sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2`, top-k 512)
returns a stable SET in a run-dependent ORDER on tie-free random scores. The three T3.15c serves
still jitter at 14,775 prompt tokens with the order sorted, always from decode token 109 and always
into one of a few discrete alternatives, so this probe asks whether the SET moves when scores tie.
The indexer score is relu(q.k) summed over heads with bf16 rounding points, so exact ties (many of
them exactly 0.0) are the normal case, not a corner.

| Case | Scores | Result (30 repeats) |
|---|---|---|
| A, boundary inside the zero tie | n_pos < 512 distinct positives, every other position exactly 0.0; L = 1,000 … 262,144 | **set differs 28–29/29** at every L, even L = 1,000 (588 tied zeros, far under the 2,048 tie-buffer cap); the selected score multiset is still exact (which zeros, not whether zeros) |
| A, boundary above the tie | n_pos >= 512 | set stable 0/29 and exact — except **L = 262,144 with 262,144 distinct positives: set differs 29/29 and is not the exact top-512** (the persistent-cluster path used past the plan threshold) |
| B, > 2,048 distinct values inside one fp16 bin | `1.0 + U(0,1)·5e-4` for all L | L = 2,048 exact and stable; **L = 2,049 differs 26/29 and is inexact**; L = 4,096 / 14,884 stable but **inexact**; L = 65,536 differs 29/29 and inexact |
| C, control | randn, L = 14,884 / 65,536 | stable 0/29, exact (as in T3.15b) |

Where it comes from (`python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh`, same
code upstream at `d62b9c24`, reported as sgl-project/sglang **#35257**):

1. `radix_tie_select` resolves the threshold bin by a 4-round exact radix pass, but in the last round
   every candidate whose fp32 bits equal the threshold value gets
   `write_pos = topk - topk_remain + atomicAdd(&counter_final, 1)` — arrival order. Only the
   <= 128-candidate paths use the `(value desc, idx asc)` comparator. So any exact tie straddling
   slot 512 is broken by scheduling once more than 128 candidates share the bin (case A, L = 1,000).
2. The collect pass keeps at most `kMaxNumTie = 2048` threshold-bin candidates
   (`if (pos < kMaxNumTie) tie_values[pos] = …`, arrival order) and `handle_tie` only ever sees
   those, so a bin with more members than that yields an inexact top-k whose membership can move
   (case B).
3. The persistent-cluster path (long rows) has the same cap per CTA (`local_start_eq + t < kMaxNumTie`)
   and returned an inexact, moving set for 262,144 distinct scores (case A, last row).

What this means for the serves: with `SGLANG_DSV41_SORT_DECODE_TOPK=1` the attention order is fixed
but the top-k SET at a decode step can still differ between runs whenever the 512th-best indexer
score is tied — rare per step, hence 14 of 20 runs identical and all deviations at the same step
(token 109 of the 14,775-token prompt). Tokens never changed in any run. Fixing it means a
deterministic final-round rank (`(value, idx)` instead of `counter_final`) and a further radix pass
instead of the 2,048 cap; the kernel is JIT-compiled from the overlay header, so no sgl-kernel
rebuild is needed. Registered as T3.15d.

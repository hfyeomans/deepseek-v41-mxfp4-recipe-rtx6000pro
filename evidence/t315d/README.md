# T3.15d — exact, deterministic decode top-k (E41)

2026-10-11. Kernel: `python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh` in the
overlay (JIT header, no sgl-kernel rebuild), overlay commit `a8804318` = patch 0055, merged into
`dsv41-kt-hybrid`. Question (from E39): the stock `topk_v2` breaks exact score ties by `atomicAdd` arrival
order and keeps at most 2,048 candidates of the threshold bin, so the selected 512-slot SET is run-dependent
whenever the 512th-best indexer score is tied. Does an exact kernel remove the 14,775-token greedy residual,
and what does it cost?

## Files

| path | what |
|---|---|
| `offline/topk-v2-exact-probe.py` | stock vs patched kernel on one GPU inside the image: 30 repeats per case, exact multiset check against a reference sort, µs per call; cases cover zero-tie, tie × 2049, dense-bin, randn and short rows at 300 … 262,144 |
| `offline/topk-v2-exact-probe-stock.txt` | stock kernel: wrong SET in every tied case |
| `offline/topk-v2-exact-probe-t315d.txt`, `-v2.txt`, `-v3.txt`, `-v4.txt` | the four patched variants — all exact and stable; **v3 is the merged one** (64-bit key, per-round byte refinement, warp-aggregated histogram adds via shfl + `__all_sync`); v4 (`__match_any_sync`) was ~2× slower on dense-bin rows and is rejected |
| `offline/topk-v2-ties-probe.py`, `-t315d-v3.txt` | E39's 38 tie cases on v3: `differs 0/29` and an exact multiset in every case |
| `offline/indexer-logits-probe.py`, `.txt` | `deep_gemm.fp8_fp4_paged_mqa_logits` (the decode indexer score kernel) replayed 30× on fixed inputs at L = 700 / 4,099 / 14,884 / 65,537 / 262,147: bit-identical every time, 0 exact ties at the k-th score |
| `serve-a/` | one plain-launcher serve (`OVERLAY_DIR` = the `t315d-topk-ties` worktree; the manifest records `743f8fa9` dirty because the kernel edit was committed as `a8804318` after launch — the served tree is that commit, nothing was edited while the serve ran): release smoke 262K × 4 `--full`, `drift-fresh` (5 repeats right after ready), `drift-after-traffic` (5 repeats after the smoke) |

## Results

Kernel (µs per call, stock | v3): zero-tie 262K 64 | 199, 65K 25 | 60, 15K 14 | 22; tie × 2049 262K
43 | 118; dense bin 262K 64 | 200; randn 262K 41 | 40. Tie-free rows cost nothing; the worst tied row is
≈ 0.14 ms against a 17 ms decode step.

Serve A, smoke **12/12 PASS**: C1 60.13 / 58.2 wall, C4 124.79 / 119.6, prefill 16K 4,361 tok/s (TTFT
3.68 s), 32K 4,513 (7.10 s), stress 4 × 260,096 in 273.8 s, needles 6/6 at 255,388 tokens (TTFT 59.5 s),
corpus bit-identical to `t315c/default-flip` (0 of 65,653 tokens moved), Xid 0, health 0. Every number is
at or above the previous headline row (`t310/default-wait-ready`: 57.4 / 124.3 / 4,248 / 4,232 / 257 s), so
the exact kernel is the new headline row at zero measured cost.

**The residual is not the top-k.** `drift-fresh`: short prompts and 3,715 tokens bit-identical × 5;
14,775 tokens 3/5 bit-identical, 2 runs differ in logprobs from decode token 109 (max |Δlogp| 0.0507,
tokens identical). `drift-after-traffic`: 5/5 bit-identical at every length. That is the same picture
E39 measured on the stock kernel (14/20), so E39's attribution of the residual to the tie-break was wrong:
the tie-break was a real defect (the served SET is now exact), but not this one's cause. The indexer
logits probe excludes the score kernel as well. Excluded so far: the kt CPU tier (E38), `topk_v2`, sparse
MLA on a fixed list (E38), the indexer logits, the MoE finalize and the KV allocator (E39). Left: the c4
compressor's `atomicAdd` slot allocation, the index-k cache store order, the SWA-layer attention — T3.15e
(per-step capture of the selected indices on the serving path, via the overlay's indexer capturer hook).

Decision: patch 0055 merged, headline row → `serve-a`; T3.15e registered low. Posting the reproduction and
fix to sgl-project/sglang #35257 is an owner call.

# T3.15b offline probes (no server; one GPU or CPU only)

Run inside `dsv41-kt:recipe` with the overlay mounted (same `docker run` line as
`scripts/test-kt-one-layer.py`). Results 2026-10-11, overlay `05f9e88b`, image `314749b5a268`.

| Probe | What | Result |
|---|---|---|
| `../kt-repeat/layer5.{txt,json}` (`scripts/test-kt-determinism.py`) | kt-kernel CPU tier, layer 5, 165 CPU experts, same (x, routing) 20x at T=1,2,4,8,64 with 16 and 8 threads | **bit-identical** on every repeat and shape; 8-thread output identical to 16-thread (no partition dependence) |
| `topk-v2-probe.py` | overlay JIT `topk_v2` (`sglang.kernels.ops.attention.dsv4.topk.topk_transform_paged_v2`, the DSV4 indexer top-k), one row, top-k 512 (`index_topk` of V4.1-Flash), 30 repeats | seq_len <= 512: output sorted and identical every time. **seq_len > 512: the selected SET is identical but its ORDER differs on 29/29 repeats** (513, 600, 1000, 2048, 4096, 20000, 65536, 262144) |
| `topk-probe.py` | sgl_kernel `fast_topk_v2` (not the kernel on our path; control), top-k 2048 | same shape of result: order non-deterministic above top-k, set stable |
| `mla-probe.py` | SM120 sparse decode MLA (`flash_mla_with_kvcache_sm120` -> FlashInfer `_sparse_mla_sm120_paged_attention`, split-K over 64-token blocks), 64 heads, top-k 512, synthetic FP8 pages, 20 repeats | **bit-identical for a fixed index list** at every length; the same index SET in a different ORDER changes ~2,300 of 32,768 output elements (max rel 0.4-0.8 % of max |o|) |

Reading: the decode attention output is a function of the top-k index order; the top-k kernel
emits that order non-deterministically once a request's indexer length exceeds 512 tokens. So
any decode past 512 tokens of context is not bit-reproducible run to run, regardless of the
MoE tiers. Below 512 tokens the order is sorted and stable, so the short-prompt drift measured
in T3.15 (divergence after 320-390 characters, i.e. < 512 tokens of context) has another
source; the serve A/B under `../a-baseline` and `../b-*` isolates it.

NCCL is excluded by construction at TP2: a two-rank all-reduce computes a+b per element, which
is commutative in IEEE arithmetic, so the reduce order cannot change bits.

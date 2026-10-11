"""Run-to-run reproducibility of the SM120 decode indexer score kernel (T3.15d / E41).

The exact top-k (T3.15d) left the 14,775-token greedy residual unchanged, so the
selection flip at decode token 109 must come from its INPUT: the indexer logits,
whose low bits are invisible everywhere except through the top-k boundary.
This replays exactly the decode call `deepseek_v4_backend._fp4_paged_mqa_logits`
makes (`deep_gemm.fp8_fp4_paged_mqa_logits`, fp4 query tuple, 64 heads x 128,
fp4 index-k pool pages of 64 slots x 68 bytes, DeepGEMM metadata) 30x on fixed
inputs and reports whether every repeat is bit-identical, per context length.
It also reports how many positions sit within 1 ulp of the 512th score, i.e.
how exposed the boundary is to a one-ulp perturbation.

  docker run --rm --gpus '"device=0"' --ipc=host \
    -v <overlay>/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v "$PWD/evidence/t315d/offline:/recipe:ro" --entrypoint python dsv41-kt:recipe \
    /recipe/indexer-logits-probe.py
"""
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
import deep_gemm  # noqa: E402
from sglang.kernels.ops.attention.dsv4.fp4_indexer import (  # noqa: E402
    quantize_fp4_indexer_tensor,
    store_fp4_index_k_cache,
)

torch.manual_seed(0)
dev = torch.device("cuda:0")
HEADS, DIM, PAGE, TOPK, REPEATS = 64, 128, 64, 512, 30
NUM_SMS = deep_gemm.get_num_sms()


def build(L: int):
    num_pages = (L + PAGE - 1) // PAGE
    q = (torch.randn(1, HEADS, DIM, device=dev) * 0.7).to(torch.bfloat16)
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(1, 1, HEADS, 64)
    q_sf = q_sf.view(1, 1, HEADS)
    weights = (torch.rand(1, HEADS, device=dev) + 0.5).float()
    k = (torch.randn(L, DIM, device=dev) * 0.7).to(torch.bfloat16)
    cache = torch.zeros(num_pages, PAGE * 68, dtype=torch.uint8, device=dev)
    loc = torch.arange(L, device=dev, dtype=torch.int64)
    store_fp4_index_k_cache(k, cache, loc, page_size=PAGE, rne=True)
    k_cache = cache.view(num_pages, PAGE, 1, 68)
    page_table = torch.arange(num_pages, device=dev, dtype=torch.int32).view(1, -1)
    seq_lens = torch.tensor([[L]], device=dev, dtype=torch.int32)
    metadata = deep_gemm.get_paged_mqa_logits_metadata(seq_lens, PAGE, NUM_SMS)
    max_len = num_pages * PAGE
    return (q_fp4, q_sf), k_cache, weights, seq_lens, page_table, metadata, max_len


def run(args):
    q, k_cache, weights, seq_lens, page_table, metadata, max_len = args
    out = deep_gemm.fp8_fp4_paged_mqa_logits(
        q, k_cache, weights, seq_lens, page_table, metadata, max_len, False
    )
    torch.cuda.synchronize()
    return out.clone()


all_identical = True
for L in [700, 4099, 14884, 65537, 262147]:
    args = build(L)
    outs = [run(args) for _ in range(REPEATS)]
    first = outs[0]
    differing = sum(1 for o in outs[1:] if not torch.equal(o, first))
    valid = first[0, :L].float()
    finite = bool(torch.isfinite(valid).all())
    top = torch.topk(valid, TOPK, sorted=True).values
    kth = top[-1]
    ulp = torch.abs(torch.nextafter(kth, torch.tensor(float("inf"), device=dev)) - kth)
    near = int((torch.abs(valid - kth) <= ulp).sum()) - 1  # excluding the k-th itself
    exact_ties = int((valid == kth).sum()) - 1
    zeros = int((valid == 0).sum())
    all_identical &= differing == 0
    print(
        f"L={L:7d}: repeats differ {differing}/{REPEATS - 1}  finite={finite}  "
        f"k-th score {float(kth):.6g}  exact ties at k-th {exact_ties}  "
        f"within 1 ulp of k-th {near}  zeros {zeros}"
    )
print(f"ALL REPEATS BIT-IDENTICAL: {all_identical}")

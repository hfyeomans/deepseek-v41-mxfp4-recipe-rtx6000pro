"""T3.15e: does the DSV4 decode top-k ever select a position >= seq_len?

The step-fingerprint bisection (evidence/t315e/) put the 14,775-token greedy
residual at rank 1, layer 36, decode step 108: indexer input and logits bit-
identical, the selected set differs by one element, logical position 4,971
replaced by 10,595 -- but the row's compressed sequence length is 7,442, so
10,595 lies past the end of the row.  This probe runs the same kernel entry
(`topk_transform_paged_v2`, top-k 512) on a buffer that is wider than the row,
as the production logits buffer is, with stale values of several kinds past
seq_len, and counts selections >= seq_len.  Reference: torch stable sort of the
in-range columns.

Run inside dsv41-kt:recipe with the overlay mounted (docker run line in
scripts/test-kt-one-layer.py), no server.
"""
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

torch.manual_seed(0)
dev = torch.device("cuda:0")
TOPK = 512
REPEATS = 10


def reference(score, L):
    s = score[0, :L]
    order = torch.argsort(-s, stable=True)
    return torch.sort(order[: min(TOPK, L)].to(torch.int32))[0]


def run(score, L, page_size=64):
    lengths = torch.tensor([L], device=dev, dtype=torch.int32)
    meta = plan_topk_v2(lengths)
    worst_oob, moves, exact = 0, 0, True
    first = None
    for _ in range(REPEATS):
        out = torch.full((1, TOPK), -7, device=dev, dtype=torch.int32)
        topk_transform_paged_v2(score, lengths, None, out, page_size, meta)
        torch.cuda.synchronize()
        got = torch.sort(out.clone())[0]
        worst_oob = max(worst_oob, int((got >= L).sum()))
        if first is None:
            first = got
            ref = reference(score, L)
            g = got[got >= 0]
            exact = g.numel() == ref.numel() and torch.equal(g, ref)
        elif not torch.equal(got, first):
            moves += 1
    return exact, worst_oob, moves


def report(name, score, L):
    exact, oob, moves = run(score, L)
    print(f"  {name:58s} L={L:6d} W={score.shape[1]:6d}: exact={exact!s:5s} "
          f"selected>=L (worst of {REPEATS})={oob:3d} moves={moves}/{REPEATS-1}")
    return exact and oob == 0 and moves == 0


def relu_row(W, L, n_pos):
    s = torch.zeros(1, W, device=dev, dtype=torch.float32)
    n_pos = min(n_pos, L)
    idx = torch.randperm(L, device=dev)[:n_pos]
    s[0, idx] = torch.rand(n_pos, device=dev) * 5 + 0.1
    return s


ok = True
print("A) in-range relu rows (k-th score inside the zero tie), columns >= L hold stale values")
for L in [7442, 14884, 4603, 1001, 300]:
    for W in [(L + 3) // 4 * 4 + 8, 16384, 131072]:
        if W < L:
            continue
        for tail_name, tail in [("zeros", 0.0), ("large +100", 100.0), ("+inf", float("inf")),
                                ("nan", float("nan")), ("-inf", float("-inf"))]:
            for n_pos in [100, 511]:
                s = relu_row(W, L, n_pos)
                s[0, L:] = tail
                ok &= report(f"tail={tail_name} n_pos={n_pos}", s, L)
print("B) tie-free random rows, columns >= L hold larger random values")
for L in [7442, 14884, 65537]:
    for W in [16384, 131072]:
        if W < L:
            continue
        s = torch.randn(1, W, device=dev, dtype=torch.float32)
        s[0, L:] = s[0, L:].abs() * 10 + 50
        ok &= report("randn in-range, +50..+inf tail", s, L)
print("C) tail is a copy of the in-range row's own top scores (stale previous-layer logits)")
for L in [7442, 14884]:
    W = 131072
    s = relu_row(W, L, 511)
    s[0, L:2 * L] = s[0, :L]
    ok &= report("stale copy of the row past L", s, L)
print("D) rows whose L is not a multiple of 4 / 8 / 64 with a +100 tail (vector-load edge)")
for L in [7441, 7443, 7447, 7449, 14883, 14885, 4605, 4607]:
    s = relu_row(131072, L, 511)
    s[0, L:] = 100.0
    ok &= report("ragged L, +100 tail", s, L)
print("ALL OK" if ok else "FAIL: out-of-range or unstable selections seen")

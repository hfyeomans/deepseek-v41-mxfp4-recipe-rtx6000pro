"""T3.15d acceptance: is the DSV4 decode top-k SET exactly the (value desc, index
asc) top-512 -- not just a stable set with the right score multiset -- and what
does the exact tie handling cost per call?

Runs the same kernel entry as evidence/t315c/offline/topk-v2-ties-probe.py
(`topk_transform_paged_v2`, top-k 512) on score rows that put the k-th score
inside an exact tie of every size the kernel distinguishes (<= 128 candidates:
comparator path; <= 2,048: radix_tie_select; > 2,048: the T3.15d refinement),
on every kernel path (register rows, streaming rows, the persistent cluster past
the plan threshold), with ragged tails (L % 4 != 0) and rows shorter than k.
Reference: torch stable sort by (-value, index).  Also times each row shape
(median of 50 calls after 5 warmups) so the stock and T3.15d overlays can be
compared on the same cases.
"""
import statistics
import sys
import time

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

torch.manual_seed(0)
dev = torch.device("cuda:0")
TOPK = 512
REPEATS = 10


def reference(score, L):
    s = score[0, :L]
    # stable sort on index after sorting by value desc => (value desc, index asc)
    order = torch.argsort(-s, stable=True)
    return torch.sort(order[: min(TOPK, L)].to(torch.int32))[0]


def run(score, L):
    lengths = torch.tensor([L], device=dev, dtype=torch.int32)
    meta = plan_topk_v2(lengths)
    sets = []
    for _ in range(REPEATS):
        out = torch.full((1, TOPK), -7, device=dev, dtype=torch.int32)
        topk_transform_paged_v2(score, lengths, None, out, 64, meta)
        torch.cuda.synchronize()
        sets.append(torch.sort(out.clone())[0])
    ref = reference(score, L)
    got = sets[0][sets[0] >= 0]
    exact = got.numel() == ref.numel() and torch.equal(got, ref)
    moves = sum(1 for s in sets[1:] if not torch.equal(s, sets[0]))
    # timing
    out = torch.full((1, TOPK), -7, device=dev, dtype=torch.int32)
    for _ in range(5):
        topk_transform_paged_v2(score, lengths, None, out, 64, meta)
    torch.cuda.synchronize()
    ts = []
    for _ in range(50):
        t0 = time.perf_counter()
        topk_transform_paged_v2(score, lengths, None, out, 64, meta)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return exact, moves, statistics.median(ts)


def row(L):
    width = (max(L, TOPK) + 3) // 4 * 4
    return torch.zeros(1, width, device=dev, dtype=torch.float32)


def report(name, score, L):
    exact, moves, us = run(score, L)
    print(f"  {name:48s} L={L:7d}: exact={exact!s:5s} moves={moves}/{REPEATS-1} {us:8.1f} us/call")
    return exact and moves == 0


ok = True
print("A) relu rows: n_pos distinct positives, the rest exactly 0.0 (k-th score inside the zero tie)")
for L in [700, 1001, 4099, 14884, 65537, 262144, 262147]:
    for n_pos in [0, 100, 411, 511]:
        if n_pos > L:
            continue
        score = row(L)
        pos = torch.randperm(L, device=dev)[:n_pos]
        score[0, pos] = torch.rand(n_pos, device=dev) + 0.01
        ok &= report(f"zero tie, n_pos={n_pos}", score, L)

print("B) one positive value shared by t positions (tie of exact size t straddling slot 512)")
for L in [4099, 14884, 65537, 262147]:
    for t in [100, 1000, 2048, 2049, 5000, 60000]:
        if t > L:
            continue
        score = row(L)
        score[0, :L] = torch.rand(L, device=dev) * 0.5  # distinct-ish background below the tie
        pos = torch.randperm(L, device=dev)[:t]
        score[0, pos] = 1.5
        ok &= report(f"tie value 1.5 x{t}", score, L)

print("C) > 2,048 distinct values inside one coarse bin (1.0 + U(0,1)*5e-4)")
for L in [2049, 4099, 14884, 65537, 262147]:
    score = row(L)
    score[0, :L] = 1.0 + torch.rand(L, device=dev) * 5e-4
    ok &= report("dense bin", score, L)

print("D) controls: randn, and rows shorter than k")
for L in [14884, 65537, 262147]:
    score = row(L)
    score[0, :L] = torch.randn(L, device=dev)
    ok &= report("randn", score, L)
for L in [1, 3, 300, 511]:
    score = row(L)
    score[0, :L] = torch.relu(torch.randn(L, device=dev))
    ok &= report("short row, relu", score, L)

print(f"ALL EXACT AND STABLE: {ok}")

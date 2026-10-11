"""Does the DSV4 decode top-k kernel pick a run-dependent SET when scores tie at
the k boundary?  (recipe T3.15c; follows evidence/t315b/offline/topk-v2-probe.py)

The indexer score is relu(q.k) summed over heads with per-head weights, so a
large share of positions score exactly 0.0.  When fewer than top-k positions
score > 0 the boundary falls inside that tie.  Random (tie-free) scores were
already shown to give a stable set (T3.15b).  This probe uses relu'd scores
with a controlled number of positives: n_pos < TOPK puts the boundary in the
zero tie; n_pos > TOPK keeps it out.
"""
import sys, torch
sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2, plan_topk_v2

torch.manual_seed(0)
dev = torch.device("cuda:0")
TOPK = 512
REPEATS = 30


def repeat_sets(score, L):
    lengths = torch.tensor([L], device=dev, dtype=torch.int32)
    meta = plan_topk_v2(lengths)
    sets = []
    for _ in range(REPEATS):
        out = torch.full((1, TOPK), -7, device=dev, dtype=torch.int32)
        topk_transform_paged_v2(score, lengths, None, out, 64, meta)
        torch.cuda.synchronize()
        sets.append(torch.sort(out.clone())[0])
    first = sets[0]
    set_diff = sum(1 for s in sets[1:] if not torch.equal(s, first))
    exact = torch.sort(torch.topk(score[0, :L], TOPK).indices.to(torch.int32))[0]
    # exact top-k is ambiguous under ties; compare the selected SCORE multiset instead
    exact_scores = torch.sort(score[0, exact.long()])[0]
    got_scores = torch.sort(score[0, first[first >= 0].long()])[0]
    score_ok = got_scores.numel() == exact_scores.numel() and torch.equal(got_scores, exact_scores)
    return first, set_diff, score_ok


print("A) relu-style scores: n_pos distinct positives, the rest exactly 0.0")
for L in [1000, 4096, 14884, 65536, 262144]:
    for n_pos in [TOPK - 100, TOPK - 1, TOPK, TOPK + 1, TOPK + 100, L]:
        if n_pos > L:
            continue
        width = (max(L, TOPK) + 3) // 4 * 4
        score = torch.zeros(1, width, device=dev, dtype=torch.float32)
        pos = torch.randperm(L, device=dev)[:n_pos]
        score[0, pos] = torch.rand(n_pos, device=dev) + 0.01  # distinct positives
        first, set_diff, score_ok = repeat_sets(score, L)
        sel = first[first >= 0].long()
        zeros = int((score[0, sel] == 0).sum())
        print(
            f"  L={L:6d} n_pos={n_pos:6d}: set differs {set_diff}/{REPEATS-1}, "
            f"n_valid={int((first >= 0).sum())}, zero-score members={zeros}, "
            f"selected-score multiset exact={score_ok}"
        )

print("B) distinct dense scores: all L values inside one fp16 bin (1.0 + U(0,1)*5e-4)")
for L in [1000, 2048, 2049, 4096, 14884, 65536]:
    width = (max(L, TOPK) + 3) // 4 * 4
    score = torch.full((1, width), -1.0, device=dev, dtype=torch.float32)
    score[0, :L] = 1.0 + torch.rand(L, device=dev, dtype=torch.float64).float() * 5e-4
    first, set_diff, score_ok = repeat_sets(score, L)
    print(
        f"  L={L:6d}: set differs {set_diff}/{REPEATS-1}, "
        f"selected-score multiset exact={score_ok}"
    )

print("C) control: distinct scores spread over many bins (randn)")
for L in [14884, 65536]:
    width = (max(L, TOPK) + 3) // 4 * 4
    score = torch.randn(1, width, device=dev, dtype=torch.float32)
    first, set_diff, score_ok = repeat_sets(score, L)
    print(f"  L={L:6d}: set differs {set_diff}/{REPEATS-1}, selected-score multiset exact={score_ok}")

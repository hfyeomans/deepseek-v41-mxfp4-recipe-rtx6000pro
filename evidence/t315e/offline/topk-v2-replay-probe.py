"""T3.15e: replay the captured layer-36 top-k input row of the 14,775-token residual.

The trigger ring (evidence/t315e/trigger/, SGLANG_DSV41_STEP_FINGERPRINT_TRIGGER_SEQ_LEN=14884)
shows the residual exactly: at decode step 109 the layer-36 row is bit-identical across 15
repeats and across both ranks, positions 4,971 and 10,595 hold the same score and sit on the
512 / 513 boundary, and one call (rank 0, repeat 0) selected 10,595 while every other call
selected 4,971 (the patch-0055 rule: value desc, index asc).  This probe feeds that exact row
to `topk_transform_paged_v2` many times under the conditions that differed between the calls:
the uninitialised tail past seq_len (zeros on repeat 0, stale scores later), the uninitialised
`selected` output buffer (production uses torch.empty_like), the buffer width, and the batch
padding -- and counts which of the two tied positions each call returns.

  docker run --rm --gpus '"device=0"' --ipc=host \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v ~/.cache/dsv41-recipe/kernel-cache:/root/.cache \
    -v "$PWD/evidence/t315e/offline:/probe:ro" -v "$PWD/.amp/in/t315e/trig2:/data:ro" \
    --entrypoint python dsv41-kt:recipe /probe/topk-v2-replay-probe.py /data/rank0.pt
"""
import sys

import torch

sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_paged_v2

dev = torch.device("cuda:0")
TOPK = 512
LAYER = 36
A, B = 4971, 10595  # tied positions; the deterministic rule selects A


def reference(score, L):
    s = score[0, :L]
    order = torch.argsort(-s, stable=True)
    return torch.sort(order[: min(TOPK, L)].to(torch.int32))[0]


def load_rows(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    t = d["trigger"]
    hits = min(t["hits"], t["logits"].shape[0])
    meta = t["meta"][:hits]
    rows = []
    for i in range(hits):
        if int(meta[i, 1]) == LAYER and int(meta[i, 3]) == 1:
            rows.append((int(meta[i, 0]), int(meta[i, 2]), t["logits"][i].clone(), torch.sort(t["selected"][i])[0]))
    return rows


def run(score, L, out_init, repeats, bs_pad=1):
    """Return (#A selected, #B selected, #other outcomes, exact-vs-reference of first call)."""
    W = score.shape[1]
    s = score.to(dev)
    if bs_pad > 1:
        s = s.repeat(bs_pad, 1)
    lengths = torch.full((bs_pad,), L, device=dev, dtype=torch.int32)
    meta = plan_topk_v2(lengths)
    ref = reference(score.to(dev), L)
    nA = nB = nOther = 0
    exact_first = None
    for r in range(repeats):
        if out_init == "empty":
            out = torch.empty((bs_pad, TOPK), device=dev, dtype=torch.int32)
        elif out_init == "withB":
            out = torch.full((bs_pad, TOPK), B, device=dev, dtype=torch.int32)
        elif out_init == "withA":
            out = torch.full((bs_pad, TOPK), A, device=dev, dtype=torch.int32)
        else:
            out = torch.full((bs_pad, TOPK), int(out_init), device=dev, dtype=torch.int32)
        topk_transform_paged_v2(s, lengths, None, out, 64, meta)
        torch.cuda.synchronize()
        got = torch.sort(out[0].clone())[0]
        g = got[got >= 0]
        if exact_first is None:
            exact_first = g.numel() == ref.numel() and torch.equal(g, ref)
        hasA, hasB = bool((got == A).any()), bool((got == B).any())
        if hasA and not hasB:
            nA += 1
        elif hasB and not hasA:
            nB += 1
        else:
            nOther += 1
    return nA, nB, nOther, exact_first


def main(path):
    rows = load_rows(path)
    print(f"{path}: {len(rows)} layer-{LAYER} hits")
    steps = [r[0] for r in rows]
    L = rows[0][1]
    base = rows[0][2]
    for step, Li, row, sel in rows:
        assert Li == L and torch.equal(row[:L], base[:L])
    print(f"L={L}; all in-range rows identical; score[{A}]={float(base[A]):.8g} score[{B}]={float(base[B]):.8g}")
    vals, order = torch.sort(base[:L], descending=True, stable=True)
    print(f"stable rank: #512 pos {int(order[511])} = {float(vals[511]):.8g}, #513 pos {int(order[512])} = {float(vals[512]):.8g}, "
          f"#511 = {float(vals[510]):.8g}, #514 = {float(vals[513]):.8g}")
    ties = int((base[:L] == base[A]).sum())
    print(f"positions tied at that score: {ties}")
    for step, Li, row, sel in rows:
        print(f"  captured step {step}: selected has A={bool((sel == A).any())} B={bool((sel == B).any())}; "
              f"tail max {float(row[L:].max()):.6g} tail nonzero {int((row[L:] != 0).sum())}")
    # distinct tails among the captures
    tails = {}
    for step, Li, row, sel in rows:
        key = row[L:].numpy().tobytes()
        tails.setdefault(key, (step, row))
    print(f"distinct tails: {len(tails)}")
    REPEATS = 200
    print(f"\nreplay: {REPEATS} calls per condition; counts = (A selected, B selected, neither/both)")
    for tstep, trow in tails.values():
        for W in [131072, 16384, 14912, 14884]:
            score = trow[:W].clone().unsqueeze(0)
            for out_init in ["empty", "withB", "withA", "-1"]:
                for bs_pad in [1, 4]:
                    nA, nB, nO, exact = run(score, L, out_init, REPEATS, bs_pad)
                    flag = "" if (nB == 0 and nO == 0) else "   <-- NOT the deterministic rule"
                    print(f"  tail-of-step {tstep:4d} W={W:6d} out={out_init:5s} bs={bs_pad}: A={nA:3d} B={nB:3d} other={nO:3d} exact_first={exact}{flag}")
    # synthetic: same row but tail filled with +100 / -inf / nan
    for name, fill in [("+100", 100.0), ("-inf", float("-inf")), ("nan", float("nan")), ("tie value", float(base[A]))]:
        score = base.clone().unsqueeze(0)
        score[0, L:] = fill
        nA, nB, nO, exact = run(score, L, "empty", REPEATS)
        print(f"  tail={name:9s} W=131072 out=empty bs=1: A={nA:3d} B={nB:3d} other={nO:3d} exact_first={exact}")


if __name__ == "__main__":
    main(sys.argv[1])

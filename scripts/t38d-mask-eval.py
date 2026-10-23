#!/usr/bin/env python3
"""T3.8d step 1: can co-activation / EMA placement beat the static el8 map?

Offline, on the 1017 recorder dumps (evidence/t1017/recorder, gitignored .pt):
single-token decode passes give per-token routed-expert traces per layer (the
recorder's logical ids are post-el8-remap, so a mask over logical ids compares
directly with the el8 map's realized share). Trains the PR #2093 co-activation
planner (scripts/expert_placement.py, vendored) and its EMA tracker on the first
60 % of the decode-distinct passes; scores every mask on the held-out decode
passes and on the prefill-text set. The go bar (T3.8d): held-out CPU share < 8 %
(per-set oracle is 2-3 %, the el8 map reads 14.4-15 %).

Usage (inside dsv41-kt, torch): python3 t38d-mask-eval.py <recorder-root>
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_placement import EmaHotnessTracker, plan_gpu_expert_placement  # noqa: E402

G = 219
TOP = 6


def load_passes(root):
    files = sorted(glob.glob(os.path.join(root, "expert_distribution_recorder_*.pt")))
    keep, seen = [], set()
    for f in files:
        key = os.path.basename(f).split("_")[-1][:14]
        if key in seen:
            continue
        seen.add(key)
        keep.append(f)
    passes = []  # (per_layer_ids, is_single)
    for f in keep:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lc = (d["logical_count"] if isinstance(d, dict) else d).to(torch.int64)  # [P, L, E]
        L = lc.shape[1]
        per_pass = lc.sum(dim=(1, 2)) / L
        single = (per_pass > 0) & (per_pass <= 12)
        for p in single.nonzero()[:, 0].tolist():
            rows = []
            ok = True
            for l in range(L):
                ids = (lc[p, l] > 0).nonzero()[:, 0].tolist()
                if len(ids) != TOP:
                    ok = False
                    break
                rows.append(ids)
            if ok:
                passes.append(rows)
    return passes


def cpu_share(counts, mask):
    c = counts.to(torch.float64)
    tot = c.sum()
    gpu = c[mask].sum()
    return float((tot - gpu) / tot) if tot > 0 else 0.0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t38d-rec/decode-distinct"
    passes = load_passes(root)
    print(f"single-token decode passes: {len(passes)}")
    L = len(passes[0])
    split = int(len(passes) * 0.6)
    train, held = passes[:split], passes[split:]

    # per-layer token traces for the planner
    traces = {l: torch.tensor([[p[l] for p in train]], dtype=torch.int64).reshape(-1, TOP) for l in range(L)}
    cov = plan_gpu_expert_placement(L, 384, G, token_traces=traces, strategy="coverage")
    clu = plan_gpu_expert_placement(L, 384, G, token_traces=traces, strategy="cluster")

    # EMA tracker warmed on the train traces, emitted after the last token
    trk = EmaHotnessTracker(L, 384, G)
    for rows in train:
        for l in range(L):
            trk.observe(l, torch.tensor(rows[l], dtype=torch.int64))
    ema = trk.mask()

    # el8 map (logical ids 0..218 = GPU): mask over logical ids
    el8 = torch.zeros(L, 384, dtype=torch.bool)
    el8[:, :G] = True

    # held-out decode counts + prefill-text counts
    held_counts = torch.zeros(L, 384, dtype=torch.int64)
    for rows in held:
        for l in range(L):
            for e in rows[l]:
                held_counts[l, e] += 1

    def share(mask, counts):
        return sum(cpu_share(counts[l], mask[l]) for l in range(L)) / L

    print(f"held-out decode mean CPU share: el8 {share(el8, held_counts):.3f}  "
          f"co-activation {share(cov, held_counts):.3f}  cluster {share(clu, held_counts):.3f}  ema {share(ema, held_counts):.3f}")

    # prefill-text counts (route total > 12 = prefill passes)
    pt_root = os.path.join(os.path.dirname(root.rstrip("/")), "prefill-text")
    files = sorted(glob.glob(os.path.join(pt_root, "expert_distribution_recorder_*.pt")))
    keep, seen = [], set()
    for f in files:
        key = os.path.basename(f).split("_")[-1][:14]
        if key in seen:
            continue
        seen.add(key)
        keep.append(f)
    pre_counts = torch.zeros(L, 384, dtype=torch.int64)
    for f in keep:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lc = (d["logical_count"] if isinstance(d, dict) else d).to(torch.int64)
        pre_counts += lc.sum(0)
    print(f"held-out prefill mean CPU share: el8 {share(el8, pre_counts):.3f}  "
          f"co-activation {share(cov, pre_counts):.3f}  cluster {share(clu, pre_counts):.3f}  ema {share(ema, pre_counts):.3f}")

    out = {
        "train_passes": split,
        "held_passes": len(held),
        "held_decode_share": {"el8": share(el8, held_counts), "coactivation": share(cov, held_counts),
                              "cluster": share(clu, held_counts), "ema": share(ema, held_counts)},
        "prefill_share": {"el8": share(el8, pre_counts), "coactivation": share(cov, pre_counts),
                          "cluster": share(clu, pre_counts), "ema": share(ema, pre_counts)},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

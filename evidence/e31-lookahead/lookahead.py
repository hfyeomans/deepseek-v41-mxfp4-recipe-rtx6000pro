import glob
import os
import sys

import torch

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/t1017-rec/decode-distinct"
files = sorted(glob.glob(os.path.join(root, "expert_distribution_recorder_*.pt")))
keep, seen = [], set()
for f in files:
    key = os.path.basename(f).split("_")[-1][:14]
    if key in seen:
        continue
    seen.add(key)
    keep.append(f)

per_layer = {l: [0, 0] for l in range(39)}  # [hits, routes]
passes = 0
skipped = 0
for f in keep:
    d = torch.load(f, map_location="cpu", weights_only=False)
    lc = (d["logical_count"] if isinstance(d, dict) else d).to(torch.int64)  # [P, L, E]
    L = lc.shape[1]
    per_pass = lc.sum(dim=(1, 2)) / L
    single = (per_pass > 0) & (per_pass <= 12)  # bs=1 decode: 6 routes x 2 (all-reduce)
    for p in single.nonzero()[:, 0].tolist():
        prev = None
        for l in range(L):
            s = set((lc[p, l] > 0).nonzero()[:, 0].tolist())
            if len(s) != 6:
                skipped += 1
                prev = None
                continue
            if prev is not None:
                hits = len(prev & s)
                per_layer[l - 1][0] += hits
                per_layer[l - 1][1] += 6
            prev = s
        passes += 1

tot = sum(v[0] for v in per_layer.values())
n = sum(v[1] for v in per_layer.values())
print(f"single-token decode passes used: {passes}; layers skipped: {skipped}")
print(f"layer L routes -> layer L+1 routes overlap: {tot}/{n} = {tot / max(n, 1):.3f}")
for l in range(39):
    h, nn = per_layer[l]
    print(f"  L{l:02d}->L{l+1:02d}: {h / max(nn, 1):.3f} ({h}/{nn})")

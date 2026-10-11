#!/usr/bin/env python3
"""Offline identity check between two nll-score.py corpus directories (same serve day, same prompts):
per-token logprob deltas pooled over every text, the count of tokens whose logprob moved at all, and
the count moved by > 0.5 nats.  Usage: identity.py <candidate-dir> <baseline-dir>"""
import json, pathlib, sys

cand, base = (pathlib.Path(p) for p in sys.argv[1:3])
tot = moved = big = 0
maxd = 0.0
for cf in sorted(cand.glob("*.json")):
    if cf.name in ("pooled.json", "manifest.json"):
        continue
    bf = base / cf.name
    if not bf.exists():
        continue
    c, b = json.load(open(cf)), json.load(open(bf))
    n = min(len(c["logprobs"]), len(b["logprobs"]))
    assert c["token_ids"][: n + 1] == b["token_ids"][: n + 1], cf.name
    d = [abs((c["logprobs"][i] or 0.0) - (b["logprobs"][i] or 0.0)) for i in range(n)]
    tot += n
    moved += sum(1 for x in d if x != 0.0)
    big += sum(1 for x in d if x > 0.5)
    maxd = max(maxd, max(d))
print(f"IDENTITY {cand} vs {base}: {moved} of {tot} logprobs moved (>0.5 nats: {big}), max|dlogp| {maxd:.4f}")

"""Scratch: big gaps with timestamps + per-period GPU-idle distribution + in-loop gap anatomy."""
import gzip, json, sys, statistics, collections
p = sys.argv[1]
ev = json.load(gzip.open(p))["traceEvents"]
gpu = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e], key=lambda e: e["ts"])
t0 = gpu[0]["ts"]
w13 = sorted(e["ts"] for e in gpu if e.get("cat") == "gpu_memcpy" and e["args"].get("bytes") == 973209600)
print("w13 launches", len(w13), "first at %.0f ms" % ((w13[0]-t0)/1e3))
gaps = []
for a, b in zip(gpu, gpu[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    if g >= 1000:
        gaps.append((a["ts"] + a["dur"], b["ts"], a["name"][:50], b["name"][:50], a["args"].get("bytes"), b["args"].get("bytes")))
print("gaps >= 15 ms:")
for s, e, an, bn, ab, bb in gaps:
    if e - s >= 15000:
        print(f"  at {(s-t0)/1e3:7.0f} ms  {(e-s)/1e3:7.1f} ms  after {an} ({ab})  before {bn} ({bb})")
# per-period idle
per = []
for i, (a, b) in enumerate(zip(w13, w13[1:])):
    if b - a > 300_000:
        continue
    idle = sum(min(e, b) - max(s, a) for s, e, *_ in gaps if s < b and e > a)
    per.append((idle / 1e3, (b - a) / 1e3, i))
idle_only = [x[0] for x in per]
print(f"per-period idle (ms): median {statistics.median(idle_only):.1f} p10 {sorted(idle_only)[len(idle_only)//10]:.1f} p90 {sorted(idle_only)[9*len(idle_only)//10]:.1f} sum {sum(idle_only):.0f} n={len(per)}")
# in-loop gaps: which boundaries
agg = collections.Counter(); cnt = collections.Counter()
for s, e, an, bn, ab, bb in gaps:
    if e - s >= 15000:
        continue
    key = f"{an[:30]} -> {bn[:30]}"
    agg[key] += (e - s) / 1e3; cnt[key] += 1
print("in-loop gap (< 15 ms) by boundary, total ms / count:")
for k, v in agg.most_common(12):
    print(f"  {v:7.0f} ms  n={cnt[k]:3d}  {k}")
# chunk-start accounting: gaps between last w13 of chunk and first w13 of next
total_in_loop = sum((e - s) for s, e, *_ in gaps if e - s < 15000) / 1e3
total_big = sum((e - s) for s, e, *_ in gaps if e - s >= 15000) / 1e3
print(f"in-loop gap total {total_in_loop:.0f} ms, big-gap total {total_big:.0f} ms")

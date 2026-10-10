"""Scratch: host-side anatomy of one streamed layer period (rank 0 trace).

For each w13 HtoD (973,209,600 B) launch we take the period to the next one and
split it into: host busy (union of top-level cpu_op intervals on the launch
thread), host idle inside the period, and the GPU busy time.
"""
import gzip, json, sys, statistics, collections

p = sys.argv[1]
ev = json.load(gzip.open(p))["traceEvents"]
gpu = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e], key=lambda e: e["ts"])
cpu = sorted([e for e in ev if e.get("cat") == "cpu_op" and "dur" in e], key=lambda e: e["ts"])
# launch thread = the tid issuing the big copies' host-side op
w13 = [e for e in gpu if e.get("cat") == "gpu_memcpy" and e["args"].get("bytes") == 973209600]
# map gpu memcpy -> host op via flow? simpler: use host 'aten::copy_' ops nearest before each w13 start
tid_count = collections.Counter(c["tid"] for c in cpu)
launch_tid = tid_count.most_common(1)[0][0]
host = [c for c in cpu if c["tid"] == launch_tid]
# top-level ops only (not nested): sweep
top = []
end = -1
for c in host:
    if c["ts"] >= end:
        top.append(c)
        end = c["ts"] + c["dur"]
print(f"launch tid {launch_tid}: {len(host)} ops, {len(top)} top-level")

def union(iv):
    s = 0
    cur_s, cur_e = None, None
    for a, b in sorted(iv):
        if cur_e is None or a > cur_e:
            if cur_e is not None:
                s += cur_e - cur_s
            cur_s, cur_e = a, b
        else:
            cur_e = max(cur_e, b)
    if cur_e is not None:
        s += cur_e - cur_s
    return s

starts = sorted(e["ts"] for e in w13)
rows = []
for a, b in zip(starts, starts[1:]):
    if b - a > 300_000:
        continue
    hb = union([(max(c["ts"], a), min(c["ts"] + c["dur"], b)) for c in top if c["ts"] < b and c["ts"] + c["dur"] > a])
    gb = union([(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for e in gpu if e["ts"] < b and e["ts"] + e["dur"] > a])
    # longest host-idle stretch in the period
    pts = sorted([(max(c["ts"], a), min(c["ts"] + c["dur"], b)) for c in top if c["ts"] < b and c["ts"] + c["dur"] > a])
    idle = []
    cur = a
    for s, e in pts:
        if s > cur:
            idle.append((s - cur, cur - a))
        cur = max(cur, e)
    if b > cur:
        idle.append((b - cur, cur - a))
    idle.sort(reverse=True)
    rows.append(((b - a) / 1e3, hb / 1e3, gb / 1e3, [(d / 1e3, o / 1e3) for d, o in idle[:3]]))

med = lambda xs: statistics.median(xs)
print(f"n={len(rows)} period {med([r[0] for r in rows]):.1f} ms, host busy {med([r[1] for r in rows]):.1f} ms, gpu busy {med([r[2] for r in rows]):.1f} ms")
print("host idle stretches (ms, offset from w13 launch) for 5 mid periods:")
for r in rows[len(rows)//2 : len(rows)//2 + 5]:
    print(f"  period {r[0]:.1f} host {r[1]:.1f} gpu {r[2]:.1f} idle {[(round(d,1), round(o,1)) for d,o in r[3]]}")
# host op time by name within periods
agg = collections.Counter()
for a, b in zip(starts, starts[1:]):
    if b - a > 300_000:
        continue
    for c in top:
        if c["ts"] < b and c["ts"] + c["dur"] > a:
            agg[c["name"][:60]] += min(c["ts"] + c["dur"], b) - max(c["ts"], a)
n = len(rows)
print("top-level host op time per period (ms):")
for k, v in agg.most_common(15):
    print(f"  {v/1e3/n:6.2f}  {k}")

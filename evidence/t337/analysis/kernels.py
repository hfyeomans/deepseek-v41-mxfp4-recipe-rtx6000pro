"""T3.37: which GPU kernels scale with the KV pool? Sum GPU time per kernel name in two traces of the
same 32K prefill (same seeds) and print the names whose total differs the most.

usage: python3 kernels.py A.trace.json.gz B.trace.json.gz [top]
"""
import collections
import gzip
import json
import sys


def load(p):
    ev = json.load(gzip.open(p))["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    gpu.sort(key=lambda e: e["ts"])
    t0 = gpu[0]["ts"]
    span = (gpu[-1]["ts"] + gpu[-1]["dur"] - t0) / 1e3
    busy = sum(e["dur"] for e in gpu) / 1e3
    by = collections.defaultdict(lambda: [0.0, 0])
    for e in gpu:
        k = e["name"][:110]
        by[k][0] += e["dur"] / 1e3
        by[k][1] += 1
    gaps = 0.0
    for a, b in zip(gpu, gpu[1:]):
        g = b["ts"] - (a["ts"] + a["dur"])
        if g > 0:
            gaps += g / 1e3
    return span, busy, gaps, by


a, b = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
sa, ba, ga, A = load(a)
sb, bb, gb, B = load(b)
print(f"A {a}\n  span {sa:.0f} ms  gpu busy {ba:.0f} ms  idle {ga:.0f} ms")
print(f"B {b}\n  span {sb:.0f} ms  gpu busy {bb:.0f} ms  idle {gb:.0f} ms")
print(f"delta: span {sb - sa:+.0f} ms, busy {bb - ba:+.0f} ms, idle {gb - ga:+.0f} ms\n")
names = set(A) | set(B)
rows = []
for n in names:
    ta, ca = A.get(n, [0.0, 0])
    tb, cb = B.get(n, [0.0, 0])
    rows.append((tb - ta, ta, tb, ca, cb, n))
rows.sort(key=lambda r: -abs(r[0]))
print(f"{'delta ms':>9} {'A ms':>8} {'B ms':>8} {'A n':>6} {'B n':>6}  kernel")
for d, ta, tb, ca, cb, n in rows[:top]:
    print(f"{d:>+9.0f} {ta:>8.0f} {tb:>8.0f} {ca:>6} {cb:>6}  {n}")
print("\nlargest by B total:")
for d, ta, tb, ca, cb, n in sorted(rows, key=lambda r: -r[2])[:12]:
    print(f"{d:>+9.0f} {ta:>8.0f} {tb:>8.0f} {ca:>6} {cb:>6}  {n}")

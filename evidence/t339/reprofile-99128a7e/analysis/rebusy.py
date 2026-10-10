"""Scratch: what fills the GPU-busy part of a streamed layer period; memcpy vs kernel; stream ids; rank asymmetry."""
import gzip, json, sys, statistics, collections
p = sys.argv[1]
ev = json.load(gzip.open(p))["traceEvents"]
gpu = sorted([e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e], key=lambda e: e["ts"])
w13 = sorted(e["ts"] for e in gpu if e.get("cat") == "gpu_memcpy" and e["args"].get("bytes") == 973209600)
streams = collections.Counter((e.get("cat"), e["args"].get("stream")) for e in gpu)
print("events by (cat, stream):", dict(streams))

def union(iv):
    s = 0; cs = ce = None
    for a, b in sorted(iv):
        if ce is None or a > ce:
            if ce is not None: s += ce - cs
            cs, ce = a, b
        else: ce = max(ce, b)
    if ce is not None: s += ce - cs
    return s

def cls(e):
    n = e["name"]
    if e["cat"] == "gpu_memcpy":
        b = e["args"].get("bytes")
        if b == 973209600: return "memcpy w13 HtoD"
        if b == 486604800: return "memcpy w2 HtoD"
        return f"memcpy other ({n[:20]})"
    if e["cat"] == "gpu_memset": return "memset"
    if "cutlass_kernels" in n or "moe" in n.lower(): return "MoE (cutlass/tllm)"
    if "flashinfer" in n.lower() or "attention" in n.lower() or "mla" in n.lower() or "fmha" in n.lower(): return "attention"
    if "gemm" in n.lower() or "cublas" in n.lower() or "nvjet" in n.lower() or "sm90" in n.lower() or "sm100" in n.lower() or "sm120" in n.lower(): return "dense GEMM"
    if "all_reduce" in n.lower() or "reduce" in n.lower() or "nccl" in n.lower(): return "allreduce"
    if "quant" in n.lower(): return "quant"
    return "other"

periods = [(a, b) for a, b in zip(w13, w13[1:]) if b - a < 300_000]
agg = collections.Counter(); sumdur = collections.Counter()
names = collections.Counter()
for a, b in periods:
    seg = [e for e in gpu if e["ts"] < b and e["ts"] + e["dur"] > a]
    for e in seg:
        d = min(e["ts"] + e["dur"], b) - max(e["ts"], a)
        agg[cls(e)] += d
        names[e["name"][:90]] += d
n = len(periods)
tot = sum((b - a) for a, b in periods)
print(f"n={n} periods, mean period {tot/n/1e3:.1f} ms; GPU time per period by class (ms, summed, may overlap across streams):")
for k, v in agg.most_common():
    print(f"  {v/1e3/n:6.1f}  {k}")
print("top kernels per period (ms):")
for k, v in names.most_common(14):
    print(f"  {v/1e3/n:6.2f}  {k}")
# memcpy vs kernel union per period
mc = statistics.median([union([(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for e in gpu if e["cat"] == "gpu_memcpy" and e["ts"] < b and e["ts"] + e["dur"] > a]) for a, b in periods]) / 1e3
kn = statistics.median([union([(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for e in gpu if e["cat"] == "kernel" and e["ts"] < b and e["ts"] + e["dur"] > a]) for a, b in periods]) / 1e3
both = statistics.median([union([(max(e["ts"], a), min(e["ts"] + e["dur"], b)) for e in gpu if e["ts"] < b and e["ts"] + e["dur"] > a]) for a, b in periods]) / 1e3
print(f"median per period: memcpy union {mc:.1f} ms, kernel union {kn:.1f} ms, all union {both:.1f} ms -> overlap {mc+kn-both:.1f} ms")
# whole-trace class totals (for rank asymmetry)
tot_cls = collections.Counter()
for e in gpu: tot_cls[cls(e)] += e["dur"]
print("whole-trace GPU time by class (ms):", {k: round(v/1e3) for k, v in tot_cls.most_common()})

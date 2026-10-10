"""T3.8 round 2: per-(layer, expert) routing counts from SGLang's stat recorder dumps.

Usage (inside dsv41-kt:recipe, torch needed):
    python3 routes.py <round dir> [--gpu 219] [--el8 <dir with counts-A..E.json>] [--out maps.json]

Each dump is a dict with ``logical_count`` [passes, layers, experts] int32. Both TP ranks
write one file per dump; the recorder's collect() all-reduces (sum) the per-rank buffer and
both ranks see the same routes, so (1) only the first file of each timestamp pair is read
and (2) every count is 2x the real route count (ratios are unaffected; the tokens/pass
column is divided by 2). A pass is one forward; prefill passes are told apart by their
route total (a decode pass has <= 4 tokens; anything larger is a prefill chunk, including
the short prompt prefills inside the decode sets).

Reports, per set and phase: positional CPU share (ids >= N_GPU on CPU), oracle (top-N_GPU
of the same counts), per-layer skew; then CPU shares of candidate maps (decode-fit,
prefill-fit, 60/40 mix, el8 mix) applied to every held-out set; and the symmetric
el8 comparison (our maps on el8 counts).
"""
import argparse, glob, json, os
import torch

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--gpu", type=int, default=219)
ap.add_argument("--sets", nargs="*", default=["humaneval-c4", "decode-distinct", "prefill-random", "prefill-text"])
ap.add_argument("--prefill-min-routes", type=int, default=4 * 6 * 2, help="routes/pass/layer above this = prefill pass: a decode pass has <= max_running_requests (4) tokens x top-k 6 x 2 (counts are 2x)")
ap.add_argument("--el8", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
G = a.gpu


def load(set_dir):
    files = sorted(glob.glob(os.path.join(set_dir, "expert_distribution_recorder_*.pt")))
    keep, seen = [], set()
    for f in files:
        key = os.path.basename(f).split("_")[-1][:14]  # timestamp prefix: pairs the two ranks
        if key in seen:
            continue
        seen.add(key)
        keep.append(f)
    dec = pre = None
    n_dec = n_pre = 0
    for f in keep:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lc = (d["logical_count"] if isinstance(d, dict) else d).to(torch.int64)
        per_pass = lc.sum(dim=(1, 2)) / lc.shape[1]
        live = per_pass > 0
        is_pre = live & (per_pass > a.prefill_min_routes)
        is_dec = live & ~is_pre
        if is_pre.any():
            s = lc[is_pre].sum(0)
            pre = s if pre is None else pre + s
            n_pre += int(is_pre.sum())
        if is_dec.any():
            s = lc[is_dec].sum(0)
            dec = s if dec is None else dec + s
            n_dec += int(is_dec.sum())
    return dec, n_dec, pre, n_pre, len(keep)


def cpu_share(c, gpu_mask):
    return c[~gpu_mask].sum().item() / c.sum().item()


def positional_mask(L, E):
    m = torch.zeros(L, E, dtype=torch.bool)
    m[:, :G] = True
    return m


def topk_mask(c):
    m = torch.zeros_like(c, dtype=torch.bool)
    m.scatter_(1, torch.topk(c.double(), G, dim=1).indices, True)
    return m


def normalise(c):
    return c.double() / c.double().sum(1, keepdim=True).clamp(min=1)


def describe(name, c, n_pass):
    L, E = c.shape
    tot = c.sum().item()
    pos = c[:, G:].sum().item() / tot
    top = torch.topk(c.double(), G, dim=1).values.sum().item() / tot
    per_layer_pos = c[:, G:].sum(1).double() / c.sum(1).double()
    nz = (c > 0).sum(1).double()
    srt = torch.sort(c.double(), dim=1, descending=True).values
    cum = srt.cumsum(1) / srt.sum(1, keepdim=True)
    n50 = (cum < 0.5).sum(1) + 1
    n90 = (cum < 0.9).sum(1) + 1
    peak = (srt[:, 0] / srt.mean(1))
    print(
        f"  {name}: {n_pass} passes, {tot/2/1e6:.2f} M routes ({tot/2/n_pass/L/6:.0f} tokens/pass); "
        f"positional CPU share {pos*100:.1f} % [layer {per_layer_pos.min()*100:.1f}-{per_layer_pos.max()*100:.1f}]; "
        f"oracle top-{G} on GPU {top*100:.1f} %; experts hit/layer median {nz.median():.0f}; "
        f"50 % of routes in {n50.median():.0f} experts (min {n50.min()}, max {n50.max()}), 90 % in {n90.median():.0f}; "
        f"busiest expert {peak.median():.1f}x mean (min {peak.min():.1f}, max {peak.max():.1f})"
    )


counts = {}
for s in a.sets:
    p = os.path.join(a.root, s)
    if not os.path.isdir(p):
        print(f"== {s}: missing")
        continue
    dec, n_dec, pre, n_pre, nfiles = load(p)
    print(f"== {s}: {nfiles} rank-0 dumps")
    if n_dec:
        describe(f"{s}/decode", dec, n_dec)
        counts[s + "/decode"] = dec
    if n_pre:
        describe(f"{s}/prefill", pre, n_pre)
        counts[s + "/prefill"] = pre

L, E = next(iter(counts.values())).shape

# --- candidate maps ---------------------------------------------------------
maps = {"positional": positional_mask(L, E)}
dec_sets = {k: v for k, v in counts.items() if k.endswith("/decode")}
pre_sets = {k: v for k, v in counts.items() if k.endswith("/prefill")}
if "humaneval-c4/decode" in counts:
    maps["decode-fit (humaneval-c4)"] = topk_mask(counts["humaneval-c4/decode"])
if "prefill-text/prefill" in counts:
    maps["prefill-fit (prefill-text)"] = topk_mask(counts["prefill-text/prefill"])
if "humaneval-c4/decode" in counts and "prefill-text/prefill" in counts:
    mix = 0.6 * normalise(counts["humaneval-c4/decode"]) + 0.4 * normalise(counts["prefill-text/prefill"])
    maps["mix 0.6 decode + 0.4 prefill-text"] = topk_mask(mix)
el8 = {}
if a.el8:
    for k in "ABCDE":
        with open(os.path.join(a.el8, f"counts-{k}.json")) as f:
            d = json.load(f)["layers"]
        el8[k] = torch.tensor([d[str(i)] for i in range(L)], dtype=torch.int64)
    el8_mix = sum(w * normalise(el8[k]) for k, w in [("D", .3), ("E", .3), ("A", .2), ("B", .2)])
    maps["el8 mix (0.3D+0.3E+0.2A+0.2B)"] = topk_mask(el8_mix)

print(f"\nCPU share (fraction of routes sent to the CPU tier) of each map on each set, N_GPU={G}/layer:")
hdr = "map".ljust(36) + "".join(k.rjust(24) for k in counts)
print(hdr)
for name, m in maps.items():
    print(name.ljust(36) + "".join(f"{cpu_share(c, m)*100:23.1f}%" for c in counts.values()))
print("own oracle".ljust(36) + "".join(f"{(1-torch.topk(c.double(), G, dim=1).values.sum().item()/c.sum().item())*100:23.1f}%" for c in counts.values()))

print("\nmap overlap (GPU slots shared, % of L*G):")
names = list(maps)
for i, n1 in enumerate(names):
    for n2 in names[i + 1:]:
        ov = (maps[n1] & maps[n2]).sum().item() / (L * G)
        print(f"  {n1} vs {n2}: {ov*100:.1f} %")

if el8:
    print(f"\nsymmetric check: our maps on el8's counts (CPU share), el8 sets A/B prefill text, C random ids, D prose decode, E code decode:")
    for name, m in maps.items():
        print(name.ljust(36) + "".join(f"{cpu_share(el8[k], m)*100:9.1f}%" for k in "ABCDE"))

if a.out:
    out = {name: m.nonzero()[:, 1].view(L, G).tolist() if m.sum(1).eq(G).all() else None for name, m in maps.items()}
    out["counts"] = {k: (v // 2).tolist() for k, v in counts.items()}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"\nwrote {a.out}")

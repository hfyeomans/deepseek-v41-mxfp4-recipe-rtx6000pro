"""T3.8 round 1: per-(layer, expert) routing counts from SGLang's stat recorder dumps.

Usage (inside dsv41-kt:recipe, torch needed):
    python3 routes.py <round dir> [--gpu N] [--fit SET] [--holdout SET ...]

Each dump is a dict with ``logical_count`` [passes, layers, experts] int32 (one file per
TP rank; the ranks see the same routes, so only rank-0 files -- the first of each
timestamp pair -- are summed). A pass is one forward; prefill passes are told apart by
their route total (6 x tokens, >> a decode batch).

Reports, for every set: routes per layer, CPU share under the positional split
(ids >= N_GPU on CPU), concentration (share of routes in the top-N_GPU experts of the
same set = oracle), and the held-out CPU share when the top-N_GPU map fitted on the
fit set is applied to the other sets.
"""
import argparse, glob, os, sys
import torch

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--gpu", type=int, default=219)
ap.add_argument("--fit", default="humaneval-c4")
ap.add_argument("--sets", nargs="*", default=["humaneval-c4", "decode-distinct", "prefill"])
ap.add_argument("--prefill-min-routes", type=int, default=6 * 512, help="a pass with more routes than this is a prefill pass")
a = ap.parse_args()


def load(set_dir):
    files = sorted(glob.glob(os.path.join(set_dir, "expert_distribution_recorder_*.pt")))
    # pair by timestamp prefix (two ranks); keep the first of each pair
    keep, seen = [], set()
    for f in files:
        key = os.path.basename(f).split("_")[-1][:14]
        if key in seen:
            continue
        seen.add(key)
        keep.append(f)
    dec = None
    pre = None
    n_dec = n_pre = 0
    for f in keep:
        d = torch.load(f, map_location="cpu", weights_only=False)
        lc = d["logical_count"] if isinstance(d, dict) else d
        lc = lc.to(torch.int64)
        per_pass = lc.sum(dim=(1, 2)) / lc.shape[1]  # routes per pass per layer
        is_pre = per_pass > a.prefill_min_routes
        if is_pre.any():
            s = lc[is_pre].sum(0)
            pre = s if pre is None else pre + s
            n_pre += int(is_pre.sum())
        if (~is_pre).any():
            s = lc[~is_pre].sum(0)
            dec = s if dec is None else dec + s
            n_dec += int((~is_pre).sum())
    return dec, n_dec, pre, n_pre, len(keep)


def describe(name, c, n_pass):
    if c is None:
        print(f"{name}: no passes")
        return None
    L, E = c.shape
    tot = c.sum().item()
    cpu_pos = c[:, a.gpu :].sum().item() / tot
    top = torch.topk(c, a.gpu, dim=1).values.sum().item() / tot
    per_layer_cpu = (c[:, a.gpu :].sum(1).double() / c.sum(1).double())
    per_layer_top = (torch.topk(c, a.gpu, dim=1).values.sum(1).double() / c.sum(1).double())
    nz = (c > 0).sum(1).double()
    print(
        f"{name}: {n_pass} passes, {tot/1e6:.2f} M routes ({tot/n_pass/L/6:.0f} tokens/pass); "
        f"positional CPU share (ids>={a.gpu}) {cpu_pos*100:.1f} % [layer min {per_layer_cpu.min()*100:.1f}, max {per_layer_cpu.max()*100:.1f}]; "
        f"oracle top-{a.gpu} covers {top*100:.1f} % [layer min {per_layer_top.min()*100:.1f}, max {per_layer_top.max()*100:.1f}]; "
        f"experts hit per layer median {nz.median():.0f}"
    )
    return c


sets = {}
for s in a.sets:
    dec, n_dec, pre, n_pre, nfiles = load(os.path.join(a.root, s))
    print(f"== {s}: {nfiles} rank-0 dumps")
    sets[s + "/decode"] = describe(f"  {s} decode passes", dec, n_dec) if n_dec else None
    sets[s + "/prefill"] = describe(f"  {s} prefill passes", pre, n_pre) if n_pre else None

fit = sets.get(a.fit + "/decode")
if fit is None:
    sys.exit("no fit set")
# a per-layer GPU map: the top-N_GPU experts of the fit set's decode passes
gpu_mask = torch.zeros_like(fit, dtype=torch.bool)
idx = torch.topk(fit, a.gpu, dim=1).indices
gpu_mask.scatter_(1, idx, True)
print(f"\nmap fitted on {a.fit}/decode (per-layer top-{a.gpu}); CPU share when applied to:")
for name, c in sets.items():
    if c is None:
        continue
    tot = c.sum().item()
    cpu = c[~gpu_mask].sum().item() / tot
    pos = c[:, a.gpu :].sum().item() / tot
    print(f"  {name:28s} fitted map {cpu*100:5.1f} %   (positional {pos*100:5.1f} %, own oracle {torch.topk(c, a.gpu, dim=1).values.sum().item()/tot*100:5.1f} %)")
# overlap between the fitted map and the positional map
overlap = (gpu_mask[:, : a.gpu]).sum().item() / (gpu_mask.shape[0] * a.gpu)
print(f"fitted map shares {overlap*100:.1f} % of its GPU slots with the positional map (ids < {a.gpu})")
# per-layer skew of the fit set: how many experts carry 50 % / 90 % of routes
srt = torch.sort(fit, dim=1, descending=True).values.double()
cum = srt.cumsum(1) / srt.sum(1, keepdim=True)
n50 = (cum < 0.5).sum(1) + 1
n90 = (cum < 0.9).sum(1) + 1
print(f"fit set per layer: experts carrying 50 % of routes median {n50.median():.0f} (min {n50.min()}, max {n50.max()}); 90 %: median {n90.median():.0f} (min {n90.min()}, max {n90.max()})")

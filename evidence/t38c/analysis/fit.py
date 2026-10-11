"""T3.8c: fit a 219-per-layer expert map on our own recorded routing counts and score it held-out.

Usage (inside dsv41-kt:recipe, torch needed, no GPU):
    python3 fit.py --map config/expert-map-el8-mix-219.json \
        --set histogram-el8-mix=evidence/t38b/histogram-el8-mix \
        --set soak-default=evidence/t38b/soak-default [--el8 evidence/t38/el8-counts] \
        [--out evidence/t38c/analysis/maps.json] [--write-map config/expert-map-own-219.json]

Every dump under the given roots was recorded on a serve that ran with ``--map`` (the el8 mix
remap), so the recorder's ``logical_count`` is indexed by *physical* slot: slot p of layer l
holds checkpoint expert perm[l][p] (the map's GPU list in ascending id order, then the remaining
ids ascending; ``kt_remap.py``). Counts are un-permuted to checkpoint ids here, so every map is
compared in one id space and the el8 map's CPU share reproduces the serve's live share (sanity
row). Dumps are read as in ``t38/round2/analysis/routes.py``: one file per rank pair, counts are
2x the real routes, a pass with <= 4 x 6 x 2 routes per layer is decode, anything larger prefill.

Fits: ``topk`` of a per-layer-normalised, weighted sum of count tables. Phase weights follow
el8's recipe (decode-heavy, 0.6 decode / 0.4 prefill); traffic sets inside a phase are weighted
by their token count. Scored:
  * leave-one-set-out: fit on every other set, score on the held-out set (the honest number);
  * fit on all own counts (the candidate map);
  * own + el8 blend (0.5 / 0.5), to see whether el8's larger corpus regularises the fit;
all against the el8 mix map (today's default) and the oracle (top-219 of the held-out set's own
counts) on the same set. The map written by ``--write-map`` is the fit on all own counts.
"""
import argparse, glob, json, os, re
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--map", required=True, help="map the recording serves ran with (physical -> checkpoint)")
ap.add_argument("--set", action="append", default=[], help="label=dir; every subdir with dumps is a traffic set")
ap.add_argument("--gpu", type=int, default=219)
ap.add_argument("--prefill-min-routes", type=int, default=4 * 6 * 2)
ap.add_argument("--decode-weight", type=float, default=0.6)
ap.add_argument("--el8", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--write-map", default=None)
a = ap.parse_args()
G = a.gpu

with open(a.map) as f:
    served = json.load(f)
E = served["num_experts"]
assert served["num_gpu_experts"] == G, "served map and --gpu disagree"
perm = {}
for layer, hot in served["gpu"].items():
    hot = sorted(int(e) for e in hot)
    rest = sorted(set(range(E)) - set(hot))
    perm[int(layer)] = torch.tensor(hot + rest, dtype=torch.int64)
L = len(perm)
perm_t = torch.stack([perm[l] for l in range(L)])  # [L, E]: physical slot -> checkpoint id


def unpermute(c):
    """c [L, E] indexed by physical slot -> [L, E] indexed by checkpoint id."""
    out = torch.zeros_like(c)
    out.scatter_(1, perm_t, c)
    return out


def load(set_dir):
    files = sorted(glob.glob(os.path.join(set_dir, "expert_distribution_recorder_*.pt")))
    keep, seen = [], set()
    for f in files:
        key = os.path.basename(f).split("_")[-1][:14]
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


def topk_mask(score):
    m = torch.zeros_like(score, dtype=torch.bool)
    m.scatter_(1, torch.topk(score.double(), G, dim=1).indices, True)
    return m


def normalise(c):
    return c.double() / c.double().sum(1, keepdim=True).clamp(min=1)


def mask_from_gpu_lists(gpu):
    m = torch.zeros(L, E, dtype=torch.bool)
    for layer, ids in gpu.items():
        m[int(layer), torch.tensor([int(e) for e in ids])] = True
    return m


def tokens(c):
    return c.sum().item() / 2 / L / 6


# --- load every set, un-permuted to checkpoint ids ------------------------------------------
counts = {}  # "label/set/phase" -> [L, E] checkpoint-id counts (2x routes)
for spec in a.set:
    label, root = spec.split("=", 1)
    for sub in sorted(os.listdir(root)):
        p = os.path.join(root, sub)
        if not os.path.isdir(p) or not glob.glob(os.path.join(p, "expert_distribution_recorder_*.pt")):
            continue
        dec, n_dec, pre, n_pre, nfiles = load(p)
        if dec is not None and dec.shape[0] != L:
            raise SystemExit(f"{p}: {dec.shape[0]} layers in dumps, {L} in the map")
        if n_dec:
            counts[f"{label}/{sub}/decode"] = unpermute(dec)
        if n_pre:
            counts[f"{label}/{sub}/prefill"] = unpermute(pre)
        print(f"== {label}/{sub}: {nfiles} rank-0 dumps, decode {n_dec} passes ({tokens(dec):,.0f} tokens), "
              f"prefill {n_pre} passes ({tokens(pre):,.0f} tokens)" if dec is not None and pre is not None else
              f"== {label}/{sub}: {nfiles} rank-0 dumps, decode {n_dec} passes, prefill {n_pre} passes")

el8_mask = mask_from_gpu_lists(served["gpu"])
dec_sets = {k: v for k, v in counts.items() if k.endswith("/decode")}
pre_sets = {k: v for k, v in counts.items() if k.endswith("/prefill")}
print(f"\ndecode tokens recorded: {sum(tokens(v) for v in dec_sets.values()):,.0f}; "
      f"prefill tokens: {sum(tokens(v) for v in pre_sets.values()):,.0f}")

# sanity: el8 map in checkpoint space must reproduce the serve's physical 'positional' share
print("\nsanity (el8 map on un-permuted counts = the serve's live CPU share, must match routes-soak.txt):")
for k, c in counts.items():
    print(f"  {k:48s} el8 {cpu_share(c, el8_mask)*100:5.1f} %   oracle {(1 - torch.topk(c.double(), G, 1).values.sum().item() / c.sum().item())*100:5.1f} %")


def family(k):
    """Traffic family of a set key: the same prompts recorded on two serves (or at C1 and C4)
    must leave the fit together, or the held-out number is leaked."""
    name = k.split("/")[1]
    for pat, fam in (("humaneval", "humaneval"), ("reasoning", "reasoning"), ("decode-distinct", "distinct"),
                     ("prefill-text", "text"), ("prefill-random", "random")):
        if name.startswith(pat):
            return fam
    return name


def fit(sets, exclude=None):
    """decode-weight * token-weighted decode mix + (1-w) * token-weighted prefill mix,
    leaving out every set of ``exclude``'s traffic family."""
    score = torch.zeros(L, E, dtype=torch.double)
    for phase, w in (("/decode", a.decode_weight), ("/prefill", 1 - a.decode_weight)):
        sel = {k: v for k, v in sets.items() if k.endswith(phase) and (exclude is None or family(k) != family(exclude))}
        if not sel:
            continue
        tot = sum(tokens(v) for v in sel.values())
        for k, v in sel.items():
            score += w * (tokens(v) / tot) * normalise(v)
    return score


el8_counts = None
if a.el8:
    el8 = {}
    for k in "ABCDE":
        with open(os.path.join(a.el8, f"counts-{k}.json")) as f:
            d = json.load(f)["layers"]
        el8[k] = torch.tensor([d[str(i)] for i in range(L)], dtype=torch.int64)
    el8_counts = sum(w * normalise(el8[k]) for k, w in [("D", .3), ("E", .3), ("A", .2), ("B", .2)])
    el8_refit = topk_mask(el8_counts)
    print(f"\nel8 mix map rebuilt from counts-A..E agrees with {a.map} on {(el8_refit & el8_mask).sum().item() / (L*G)*100:.1f} % of slots")

# --- leave-one-set-out ------------------------------------------------------------------------
print(f"\nleave-one-traffic-family-out (every set sharing the held-out set's prompts, both phases, excluded from the fit), CPU share of routes, N_GPU={G}:")
print(f"  {'held-out set':48s} {'el8 map':>8s} {'own fit':>8s} {'own+el8':>8s} {'oracle':>8s}   tokens")
rows = {}
for k, c in counts.items():
    base = k.rsplit("/", 1)[0]
    own = topk_mask(fit(counts, exclude=k))
    blend = topk_mask(0.5 * fit(counts, exclude=k) + 0.5 * el8_counts) if el8_counts is not None else None
    r = dict(el8=cpu_share(c, el8_mask), own=cpu_share(c, own),
             blend=cpu_share(c, blend) if blend is not None else float("nan"),
             oracle=1 - torch.topk(c.double(), G, 1).values.sum().item() / c.sum().item(),
             tokens=tokens(c))
    rows[k] = r
    print(f"  {k:48s} {r['el8']*100:7.1f}% {r['own']*100:7.1f}% {r['blend']*100:7.1f}% {r['oracle']*100:7.1f}%   {r['tokens']:,.0f}   [{family(k)}]")


def wavg(key, phase):
    sel = [r for k, r in rows.items() if k.endswith(phase)]
    t = sum(r["tokens"] for r in sel)
    return sum(r[key] * r["tokens"] for r in sel) / t


for phase in ("/decode", "/prefill"):
    print(f"  token-weighted held-out mean, {phase[1:]:8s}             "
          f"{wavg('el8', phase)*100:7.1f}% {wavg('own', phase)*100:7.1f}% {wavg('blend', phase)*100:7.1f}% {wavg('oracle', phase)*100:7.1f}%")

if el8_counts is not None:
    print("\nblend weight sweep (alpha * own leave-family-out fit + (1 - alpha) * el8 mix), token-weighted held-out CPU share:")
    for alpha in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        acc = {"/decode": [0.0, 0.0], "/prefill": [0.0, 0.0]}
        for k, c in counts.items():
            m = topk_mask(alpha * fit(counts, exclude=k) + (1 - alpha) * el8_counts)
            ph = "/decode" if k.endswith("/decode") else "/prefill"
            acc[ph][0] += cpu_share(c, m) * tokens(c)
            acc[ph][1] += tokens(c)
        print(f"  alpha {alpha:.1f}: decode {acc['/decode'][0] / acc['/decode'][1] * 100:5.1f} %  prefill {acc['/prefill'][0] / acc['/prefill'][1] * 100:5.1f} %")

# --- the candidate: fit on everything --------------------------------------------------------
own_all = topk_mask(fit(counts))
print(f"\nfit on all own counts: {(own_all & el8_mask).sum().item() / (L*G)*100:.1f} % of GPU slots shared with the el8 map "
      f"({(own_all & ~el8_mask).sum().item()} experts move GPU<-CPU in total, {(own_all & ~el8_mask).sum(1).double().mean():.1f} per layer)")
print("  in-sample CPU share (not a held-out number):")
for k, c in counts.items():
    print(f"    {k:48s} el8 {cpu_share(c, el8_mask)*100:5.1f} %  own {cpu_share(c, own_all)*100:5.1f} %")
if el8_counts is not None:
    print("  the own-fit map on el8's own counts (A/B prefill text, C random ids, D prose decode, E code decode):")
    for name, m in (("el8 map", el8_mask), ("own fit", own_all)):
        print(f"    {name:12s}" + "".join(f"  {k} {cpu_share(el8[k], m)*100:5.1f} %" for k in "ABCDE"))

if a.out:
    with open(a.out, "w") as f:
        json.dump({"held_out": rows, "own_all_gpu": {str(l): own_all[l].nonzero()[:, 0].tolist() for l in range(L)}}, f)
    print(f"\nwrote {a.out}")
if a.write_map:
    out = {"num_experts": E, "num_gpu_experts": G,
           "source": f"T3.8c own-count fit, {a.decode_weight:.1f} decode + {1-a.decode_weight:.1f} prefill, token-weighted over "
                     f"{len(dec_sets)} decode / {len(pre_sets)} prefill sets ({sum(tokens(v) for v in dec_sets.values()):,.0f} / "
                     f"{sum(tokens(v) for v in pre_sets.values()):,.0f} tokens); evidence/t38c/analysis/fit.py",
           "gpu": {str(l): own_all[l].nonzero()[:, 0].tolist() for l in range(L)}}
    with open(a.write_map, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.write_map}")

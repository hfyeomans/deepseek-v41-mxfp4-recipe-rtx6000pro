"""T3.15e: narrow the trigger-ring analysis to the residual.

Groups the captured top-k inputs by (rank, layer, in-range row content) so that
rows from different prompts (same compressed seq_len) are never compared.  Inside
each group it reports only pairs whose [0, L) scores are bit-identical but whose
selected set differs, with the moved positions, their in-range/tail scores and
whether any selected position is >= L.

  ~/.venv/pytorch/bin/python evidence/t315e/offline/trigger-narrow.py .amp/in/t315e/trig
"""
import hashlib
import pathlib
import sys
from collections import defaultdict

import torch

TOPK = 512


def reference(row, L):
    order = torch.argsort(-row[:L], stable=True)
    return torch.sort(order[: min(TOPK, L)].to(torch.int32))[0]


def main(d: pathlib.Path) -> None:
    for path in sorted(d.glob("rank*.pt")):
        dump = torch.load(path, map_location="cpu", weights_only=False)
        t = dump.get("trigger")
        if t is None:
            print(f"{path.name}: no trigger dump")
            continue
        hits = min(t["hits"], t["logits"].shape[0])
        meta = t["meta"][:hits]
        print(f"{path.name}: trigger seq_len {t['trigger']}, {t['hits']} hits ({hits} kept)")
        groups = defaultdict(list)
        for i in range(hits):
            if int(meta[i, 3]) != 1:
                continue
            layer, L = int(meta[i, 1]), int(meta[i, 2])
            key = (layer, L, hashlib.sha1(t["logits"][i][:L].numpy().tobytes()).hexdigest()[:10])
            groups[key].append(i)
        for (layer, L, h), idx in sorted(groups.items()):
            sels = [torch.sort(t["selected"][i])[0] for i in idx]
            sets = {tuple(s.tolist()) for s in sels}
            ref = reference(t["logits"][idx[0]], L)
            base_ok = torch.equal(sels[0][sels[0] >= 0], ref)
            mx = max(int(s.max()) for s in sels)
            tails = {hashlib.sha1(t["logits"][i][L:].numpy().tobytes()).hexdigest()[:10] for i in idx}
            print(f"  layer {layer:2d} row {h} L={L}: {len(idx)} hits steps {[int(meta[i, 0]) for i in idx]}; "
                  f"distinct sets {len(sets)}; distinct tails {len(tails)}; set0==stable topk {base_ok}; max sel {mx}")
            if len(sets) < 2:
                continue
            base_i, base_sel = idx[0], sels[0]
            for j, i in enumerate(idx[1:], 1):
                sel = sels[j]
                if torch.equal(sel, base_sel):
                    continue
                row, base_row = t["logits"][i], t["logits"][base_i]
                removed = sorted(set(base_sel.tolist()) - set(sel.tolist()))
                added = sorted(set(sel.tolist()) - set(base_sel.tolist()))
                print(f"    RESIDUAL step {int(meta[base_i, 0])} vs {int(meta[i, 0])}: removed {removed} added {added}")
                for p in removed + added:
                    if 0 <= p < row.numel():
                        tag = "IN" if p < L else "TAIL"
                        print(f"      pos {p} ({tag}): base={float(base_row[p]):.8g} this={float(row[p]):.8g}")
                # k-th / (k+1)-th boundary of the in-range row
                vals, order = torch.sort(base_row[:L], descending=True, stable=True)
                print(f"      in-range boundary: #512={float(vals[511]):.8g} (pos {int(order[511])}) "
                      f"#513={float(vals[512]):.8g} (pos {int(order[512])})")
                print(f"      tail max: base={float(base_row[L:].max()):.8g} this={float(row[L:].max()):.8g}; "
                      f"this set == stable topk of own row: {torch.equal(sel[sel >= 0], reference(row, L))}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))

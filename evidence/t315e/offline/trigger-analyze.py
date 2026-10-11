"""T3.15e: compare the full top-k input rows and selected sets captured by the
SGLANG_DSV41_STEP_FINGERPRINT_TRIGGER_SEQ_LEN dump across repeats of one prompt.

For each (rank, layer) with >= 2 hits: are the logits rows identical inside
[0, seq_len)?  Outside?  Do the selected sets differ, and if so which positions
moved and what score did each have?  Is every selected position < seq_len?  Does
the set equal torch's stable (value desc, index asc) top-512 of the in-range row?

  ~/.venv/pytorch/bin/python evidence/t315e/offline/trigger-analyze.py evidence/t315e/fp-trigger
"""
import pathlib
import sys

import torch

TOPK = 512


def reference(row, L):
    s = row[:L]
    order = torch.argsort(-s, stable=True)
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
        layers = sorted(set(meta[:, 1].tolist()))
        for layer in layers:
            idx = [i for i in range(hits) if int(meta[i, 1]) == layer and int(meta[i, 3]) == 1]
            if len(idx) < 2:
                continue
            L = int(meta[idx[0], 2])
            base_row, base_sel = t["logits"][idx[0]], torch.sort(t["selected"][idx[0]])[0]
            ref = reference(base_row, L)
            base_ok = torch.equal(base_sel[base_sel >= 0], ref)
            print(f"  layer {layer:2d}: {len(idx)} hits at steps {[int(meta[i, 0]) for i in idx]}; "
                  f"hit0 set == stable top-k of in-range row: {base_ok}; max selected {int(base_sel.max())} (L={L})")
            for i in idx[1:]:
                row, sel = t["logits"][i], torch.sort(t["selected"][i])[0]
                in_same = torch.equal(row[:L], base_row[:L])
                out_same = torch.equal(row[L:], base_row[L:])
                same_set = torch.equal(sel, base_sel)
                line = (f"    vs hit at step {int(meta[i, 0])}: logits[0:L] identical={in_same}, "
                        f"logits[L:] identical={out_same}, selected set identical={same_set}")
                if not in_same:
                    diff = (row[:L] != base_row[:L]).nonzero().flatten()
                    line += f"; {diff.numel()} in-range scores differ, first at {int(diff[0])}"
                if not same_set:
                    a = set(base_sel.tolist()) - set(sel.tolist())
                    b = set(sel.tolist()) - set(base_sel.tolist())
                    line += f"; removed {sorted(a)} added {sorted(b)}"
                    for p_ in sorted(a | b):
                        if 0 <= p_ < row.numel():
                            line += f"; score[{p_}] base={float(base_row[p_]):.6g} this={float(row[p_]):.6g}"
                    oob = [p_ for p_ in b if p_ >= L]
                    if oob:
                        line += f"; OUT OF RANGE added: {oob}"
                    ref_i = reference(row, L)
                    line += f"; this set == stable top-k of its own row: {torch.equal(sel[sel >= 0], ref_i)}"
                print(line)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))

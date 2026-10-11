#!/usr/bin/env python3
"""Align repeats of one prompt in a step-fingerprint dump and name the first stage that differs.

T3.15e. The overlay's SGLANG_DSV41_STEP_FINGERPRINT_DIR recorder writes, per rank,
<dir>/rank<r>.pt: an int64 buffer [steps, layers, columns] of order-independent
fingerprints (bit-pattern sum and sum of squares) of six decode-path stages for
every decode step and attention layer, plus row 0's sequence length and the batch
size. This script:

  * splits the step axis into requests wherever seq_len stops increasing by one;
  * keeps the requests whose first seq_len matches --prompt-tokens (default: the
    most common first seq_len across requests, i.e. the repeated drift prompt);
  * walks the repeats step by step against the first repeat and reports, per pair,
    the first (step, layer, stage) whose fingerprint differs. Stage order along
    one layer's decode path is x -> logits -> sel -> slots -> q -> o, so the first
    differing stage names the producer: x = upstream of this layer's attention
    (an earlier layer's MoE / residual), logits = the indexer GEMM, sel = the
    top-k, slots = the KV allocator, q = the query projection, o = the sparse
    attention kernel itself.

  python3 scripts/fingerprint-diff.py evidence/t315e/fp --output evidence/t315e/diff.json

Exit 0 always; this is a measurement, not a gate.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import torch

STAGES = ("x", "logits", "sel", "slots", "q", "o")


def load(path: pathlib.Path) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=False)
    cols = list(d["columns"])
    buf = d["buf"][: d["steps"]]
    return {"columns": cols, "buf": buf, "rank": d["rank"], "steps": int(d["steps"])}


def marker_layer(buf: torch.Tensor, cols: list[str]) -> int:
    """seq_len is recorded by the indexer top-k path, which only the candidate-source layers
    run (every other layer filters that layer's candidates); use the one with the most records."""
    counts = (buf[:, :, cols.index("seq_len")] != 0).sum(dim=0)
    return int(counts.argmax())


def split_requests(buf: torch.Tensor, cols: list[str], layer: int) -> list[tuple[int, int]]:
    """[start, end) step ranges where the marker layer's seq_len never decreases and grows by at most
    one per step. The recorded value is the indexer's compressed sequence length (c4_seq_lens, the
    token count over the layer's compression ratio), so consecutive steps repeat or add one; a drop
    is a new request. Steps without a seq_len (dense-attention decode of prompts below the indexer
    threshold) are skipped."""
    seq = buf[:, layer, cols.index("seq_len")].tolist()
    ranges: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(seq):
        if s == 0:  # step 0 or an unrecorded row
            if start is not None:
                ranges.append((start, i))
                start = None
            continue
        if start is None or s < seq[i - 1] or s > seq[i - 1] + 1:
            if start is not None:
                ranges.append((start, i))
            start = i
    if start is not None:
        ranges.append((start, len(seq)))
    return ranges


def first_diff(a: torch.Tensor, b: torch.Tensor, cols: list[str], layers: int, marker: int) -> dict | None:
    """a, b: [steps, layers, columns] for two repeats (truncated to the shorter)."""
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    for step in range(n):
        for layer in range(layers):
            for stage in STAGES:
                c = cols.index(f"{stage}_sum")
                if a[step, layer, c] != b[step, layer, c] or a[step, layer, c + 1] != b[step, layer, c + 1]:
                    return {"step": step, "layer": layer, "stage": stage,
                            "seq_len": int(a[step, marker, cols.index("seq_len")])}
    return None


def per_stage_summary(a: torch.Tensor, b: torch.Tensor, cols: list[str], layers: int) -> dict:
    n = min(a.shape[0], b.shape[0])
    out = {}
    for stage in STAGES:
        c = cols.index(f"{stage}_sum")
        diff = (a[:n, :layers, c] != b[:n, :layers, c]) | (a[:n, :layers, c + 1] != b[:n, :layers, c + 1])
        steps = diff.any(dim=1).nonzero().flatten().tolist()
        out[stage] = {"first_step": steps[0] if steps else None, "steps_differing": len(steps),
                      "first_layer": int(diff[steps[0]].nonzero().flatten()[0]) if steps else None}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=pathlib.Path, help="directory holding rank<r>.pt dumps")
    ap.add_argument("--prompt-tokens", type=int, default=None,
                    help="first (compressed) seq_len of the requests to align (default: the most common one)")
    ap.add_argument("--layers", type=int, default=None, help="attention layers recorded (default: infer)")
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report = {"dir": str(args.dir), "ranks": {}}
    for path in sorted(args.dir.glob("rank*.pt")):
        d = load(path)
        cols, buf = d["columns"], d["buf"]
        layers = args.layers
        if layers is None:
            used = (buf != 0).any(dim=2).any(dim=0)
            layers = int(used.nonzero().flatten()[-1]) + 1 if used.any() else 0
        marker = marker_layer(buf, cols)
        ranges = split_requests(buf, cols, marker)
        firsts = [int(buf[s, marker, cols.index("seq_len")]) for s, _ in ranges]
        target = args.prompt_tokens or (collections.Counter(firsts).most_common(1)[0][0] if firsts else None)
        reps = [(s, e) for (s, e), f in zip(ranges, firsts) if f == target]
        rank_rep = {"steps": d["steps"], "layers": layers, "marker_layer": marker, "requests": len(ranges),
                    "prompt_tokens": target, "repeats": len(reps),
                    "repeat_lengths": [e - s for s, e in reps], "pairs": []}
        if reps:
            s0, e0 = reps[0]
            base = buf[s0:e0, :layers]
            for i, (s, e) in enumerate(reps[1:], start=1):
                other = buf[s:e, :layers]
                rank_rep["pairs"].append({"pair": f"0 vs {i}", "first_diff": first_diff(base, other, cols, layers, marker),
                                          "stages": per_stage_summary(base, other, cols, layers)})
        report["ranks"][f"rank{d['rank']}"] = rank_rep
        print(f"rank{d['rank']}: {d['steps']} steps, {layers} layers, {len(ranges)} requests, "
              f"{len(reps)} repeats of prompt_tokens={target}")
        for p in rank_rep["pairs"]:
            fd = p["first_diff"]
            where = "identical" if fd is None else f"step {fd['step']} layer {fd['layer']} stage {fd['stage']} (seq_len {fd['seq_len']})"
            print(f"  {p['pair']}: {where}")
            for stage, v in p["stages"].items():
                if v["first_step"] is not None:
                    print(f"      {stage:7s} first step {v['first_step']} layer {v['first_layer']}, {v['steps_differing']} steps differ")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarize a KT_TIMELINE run (overlay kt_timeline.py) into per-step buckets
and a per-layer table, and check the sum identity (recipe T3.19a).

Input: one or more timeline-rank*.jsonl files (records "tick" for every replay,
"step" for sampled replays with per-segment / per-break / per-layer timings).

Per sampled step the buckets are (all ms, GPU timeline unless noted):
    export        D2H + submit host node, summed over MoE layers
    gpu_experts   resident-expert kernels, summed
    cpu_wait      exposed kt wait (sync host node), summed   <- the ~40 waits
    h2d           pinned output -> device, summed
    combine       output + cpu_output (+ B2 join), summed
    engram_break  GPU idle between segments = the eager Engram lookups (gaps)
    other_gpu     segments minus MoE layer spans: attention, dense, norms,
                  all-reduce, lm head
    between_replays  replay period minus gpu_span: GPU work outside the graph
                  (logits post-processing, sampler) plus any GPU idle
The step time these are compared against is the replay period: the host wall
time between consecutive replays of the same key, taken from the "tick"
records of *unsampled* replays (no sync, no perturbation) - 1000 / period is
the decode tok/s the scheduler can deliver for that batch size.
Checks:
  * step identity: sum(7 GPU buckets) vs replay period; exit 2 if the median
    relative error exceeds --tolerance (default 5 %).  Meaningful because the
    buckets come from device events and the period from the host clock.
  * event consistency: gpu_span == sum(segments) + sum(gaps) and every MoE
    layer span == sum of its five buckets (same tolerance; catches mis-ordered
    marks).
Also reported, not part of the identity:
  * gpu_backlog_at_start = (replay_wall + final_sync) - gpu_span for the
    sampled replays: how far the host runs ahead of the GPU when a replay is
    launched.  Large means the GPU is the critical path and the host is not.
  * --bench-ms-per-token (1000 / bench-decode wall tok/s; per stream for C4)
    adds outside_replay = bench - period: scheduler, detokenize, HTTP.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LAYER_BUCKETS = ("export", "gpu_experts", "cpu_wait", "h2d", "combine")
GPU_BUCKETS = LAYER_BUCKETS + ("engram_break", "other_gpu")


def load(paths):
    steps, ticks = [], []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                (steps if rec.get("t") == "step" else ticks).append(rec)
    return steps, ticks


def step_buckets(rec: dict) -> dict:
    layers = rec["layers"]
    b = {k: sum(L[k] for L in layers) for k in LAYER_BUCKETS}
    seg_total = sum(s["gpu_ms"] for s in rec["segments"])
    layer_spans = sum(L["span_ms"] for L in layers)
    b["engram_break"] = sum(g["gpu_gap_ms"] for g in rec["gaps"])
    b["other_gpu"] = seg_total - layer_spans
    host_step = rec["host"]["replay_wall_ms"] + rec["host"]["final_sync_ms"]
    b["_gpu_backlog_ms"] = host_step - rec["gpu_span_ms"]
    b["_host_step_ms"] = host_step
    b["_gpu_span_ms"] = rec["gpu_span_ms"]
    b["_gpu_sum_ms"] = sum(b[k] for k in GPU_BUCKETS)
    seg_gap = sum(s["gpu_ms"] for s in rec["segments"]) + b["engram_break"]
    b["_span_err"] = (seg_gap - rec["gpu_span_ms"]) / rec["gpu_span_ms"] if rec["gpu_span_ms"] else float("nan")
    layer_err = [
        (sum(L[k] for k in LAYER_BUCKETS) - L["span_ms"]) / L["span_ms"] for L in layers if L["span_ms"]
    ]
    b["_layer_err"] = max((abs(e) for e in layer_err), default=0.0)
    b["_cpu_total_ms"] = sum(L["cpu_total_ms"] for L in layers)
    b["_n_layers"] = len(layers)
    return b


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def fmt(x, w=8, p=3):
    return f"{x:{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="timeline-rank*.jsonl")
    ap.add_argument("--tolerance", type=float, default=0.05, help="max median relative event-consistency error")
    ap.add_argument("--bench-ms-per-token", type=float, help="1000 / bench-decode wall tok/s (per stream) for the ladder")
    ap.add_argument("--min-steps", type=int, default=3, help="ignore keys with fewer sampled steps")
    ap.add_argument("--skip", type=int, default=1, help="drop the first N sampled steps per key (warm-up)")
    ap.add_argument("--json", type=Path, help="write the summary as JSON here")
    ap.add_argument("--md", type=Path, help="write the summary as Markdown here")
    args = ap.parse_args()

    steps, ticks = load(args.files)
    by_key = defaultdict(list)
    for rec in steps:
        by_key[(rec["rank"], rec["key"])].append(rec)
    periods = defaultdict(list)
    last = {}
    for t in sorted(ticks, key=lambda r: (r["key"], r["n"])):
        k = t["key"]
        if k in last and t["n"] == last[k]["n"] + 1:
            periods[k].append((t["t0_ns"] - last[k]["t0_ns"]) / 1e6)
        last[k] = t

    summary = {"keys": {}, "tolerance": args.tolerance}
    md = ["# KT timeline report", ""]
    worst = 0.0
    for (rank, key) in sorted(by_key):
        recs = sorted(by_key[(rank, key)], key=lambda r: r["n"])[args.skip:]
        if len(recs) < args.min_steps:
            continue
        bs = [step_buckets(r) for r in recs]
        med = {k: median([b[k] for b in bs]) for k in bs[0]}
        event_err = max(median([abs(b["_span_err"]) for b in bs]), median([b["_layer_err"] for b in bs]))
        period = median(periods.get(key, []))
        if period != period:  # no unsampled ticks (e.g. SAMPLE=1): fall back to the synced host step
            period = med["_host_step_ms"]
            period_source = "host_step (no unsampled ticks)"
        else:
            period_source = "unsampled tick period"
        med["between_replays"] = period - med["_gpu_span_ms"]
        # between_replays is a residual, so the informative number is the GPU sum vs the period:
        gpu_share = med["_gpu_sum_ms"] / period if period else float("nan")
        identity_err = abs(1.0 - gpu_share)
        worst = max(worst, event_err, identity_err)
        per_layer = defaultdict(lambda: defaultdict(list))
        for r in recs:
            for L in r["layers"]:
                for k in LAYER_BUCKETS + ("span_ms", "cpu_total_ms"):
                    per_layer[L["layer"]][k].append(L[k])
        layer_rows = {
            layer: {k: median(v) for k, v in cols.items()} for layer, cols in sorted(per_layer.items())
        }
        breaks = defaultdict(list)
        for r in recs:
            for b in r["breaks"]:
                breaks[(b["i"], b["name"])].append((b["host_ms"], b["gpu_gap_ms"]))
        break_rows = [
            {"i": i, "name": name, "host_ms": median([h for h, _ in v]), "gpu_gap_ms": median([g for _, g in v])}
            for (i, name), v in sorted(breaks.items())
        ]
        seg_rows = [
            {"i": i, "gpu_ms": median([r["segments"][i]["gpu_ms"] for r in recs if i < len(r["segments"])])}
            for i in range(max(len(r["segments"]) for r in recs))
        ]
        ladder = {"gpu_bucket_sum_ms": med["_gpu_sum_ms"], "gpu_span_ms": med["_gpu_span_ms"],
                  "replay_period_ms": period, "between_replays_ms": med["between_replays"],
                  "sampled_host_step_ms": med["_host_step_ms"], "gpu_backlog_at_start_ms": med["_gpu_backlog_ms"]}
        if args.bench_ms_per_token:
            ladder["bench_ms_per_token"] = args.bench_ms_per_token
            ladder["outside_replay_ms"] = args.bench_ms_per_token - period
        entry = {
            "rank": rank,
            "key": key,
            "sampled_steps": len(recs),
            "all_replays": len([t for t in ticks if t["key"] == key]),
            "median_buckets_ms": {k: med[k] for k in GPU_BUCKETS + ("between_replays",)},
            "step_ms": period,
            "step_source": period_source,
            "gpu_bucket_sum_ms": med["_gpu_sum_ms"],
            "gpu_share_of_step": gpu_share,
            "identity_err": identity_err,
            "median_event_consistency_err": event_err,
            "median_gpu_span_ms": med["_gpu_span_ms"],
            "median_sampled_host_step_ms": med["_host_step_ms"],
            "median_gpu_backlog_at_start_ms": med["_gpu_backlog_ms"],
            "ladder_ms": ladder,
            "median_cpu_total_ms": med["_cpu_total_ms"],
            "n_moe_layers": int(med["_n_layers"]),
            "segments": seg_rows,
            "breaks": break_rows,
            "layers": layer_rows,
        }
        summary["keys"][f"rank{rank}/{key}"] = entry

        md.append(f"## rank {rank} / {key}  ({len(recs)} sampled of {entry['all_replays']} replays)")
        md.append("")
        md.append(
            f"step = {period_source} median {period:.3f} ms ({1000/period:.2f} steps/s); "
            f"GPU bucket sum {med['_gpu_sum_ms']:.3f} ms = {gpu_share*100:.1f} % of step; "
            f"GPU span {med['_gpu_span_ms']:.3f} ms; event-consistency err {event_err*100:.2f} %; "
            f"GPU backlog at replay start (host runs ahead) {med['_gpu_backlog_ms']:.3f} ms"
        )
        md.append("")
        md.append("| ladder (ms) | value |")
        md.append("|---|---:|")
        for k, v in ladder.items():
            md.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | n/a |")
        md.append("")
        md.append("| bucket | median ms | share of step |")
        md.append("|---|---:|---:|")
        for k in GPU_BUCKETS + ("between_replays",):
            md.append(f"| {k} | {med[k]:.3f} | {100*med[k]/period:.1f} % |")
        md.append(f"| (derived) cpu_total = sync - export, summed | {med['_cpu_total_ms']:.3f} | {100*med['_cpu_total_ms']/period:.1f} % |")
        md.append("")
        md.append("| segment | GPU ms |")
        md.append("|---|---:|")
        for s in seg_rows:
            md.append(f"| {s['i']} | {s['gpu_ms']:.3f} |")
        md.append("")
        md.append("| break | name | host ms | GPU gap ms |")
        md.append("|---|---|---:|---:|")
        for b in break_rows:
            md.append(f"| {b['i']} | {b['name']} | {b['host_ms']:.3f} | {b['gpu_gap_ms']:.3f} |")
        md.append("")
        md.append("| layer | span | export | gpu_experts | cpu_wait | h2d | combine | cpu_total |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for layer, row in layer_rows.items():
            md.append(
                f"| {layer} | {row['span_ms']:.3f} | {row['export']:.3f} | {row['gpu_experts']:.3f} | "
                f"{row['cpu_wait']:.3f} | {row['h2d']:.3f} | {row['combine']:.3f} | {row['cpu_total_ms']:.3f} |"
            )
        md.append("")

        print(f"rank {rank} {key}: {len(recs)} sampled; step {period:.3f} ms ({period_source}), "
              f"GPU buckets {med['_gpu_sum_ms']:.3f} ms = {gpu_share*100:.1f} %, gpu span {med['_gpu_span_ms']:.3f} ms, "
              f"event err {event_err*100:.2f} %, GPU backlog at start {med['_gpu_backlog_ms']:.3f} ms")
        if args.bench_ms_per_token:
            print(f"    bench {args.bench_ms_per_token:.3f} ms/token -> outside replay "
                  f"{ladder['outside_replay_ms']:.3f} ms ({100*ladder['outside_replay_ms']/args.bench_ms_per_token:.1f} %)")
        for k in GPU_BUCKETS + ("between_replays",):
            print(f"    {k:15s} {fmt(med[k])} ms  {100*med[k]/period:5.1f} %")

    summary["worst_err"] = worst
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    if args.md:
        args.md.write_text("\n".join(md) + "\n")
    if not summary["keys"]:
        print("no key with enough sampled steps", file=sys.stderr)
        return 1
    if worst > args.tolerance:
        print(f"FAIL: identity / event-consistency error {worst*100:.2f} % > {args.tolerance*100:.1f} %", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

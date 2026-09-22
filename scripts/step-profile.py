#!/usr/bin/env python3
"""Per-step GPU time budget from an SGLang `/start_profile` trace (torch profiler, Chrome JSON).

Answers "where does one decode step go?" for a hybrid (GPU + CPU expert) server:
GPU-busy time by kernel category, idle gaps between kernels (CPU-tier waits,
graph breaks, host syncs), and the largest single kernels — all divided by the
number of profiled steps so the numbers are ms/step and comparable across
traces (bs=1 vs bs=2, before vs after a kernel change).

Input: one or more `*.trace.json.gz` (one per TP rank) as SGLang writes them to
`output_dir`. Steps: `--steps N` (the `num_steps` you passed to `/start_profile`).

Usage:
  python3 scripts/step-profile.py evidence/e20-dense-fp8-gemm/prof-e20-bs1/*TP-0*.gz --steps 8
  python3 scripts/step-profile.py A/*TP-0*.gz --steps 8 --compare B/*TP-0*.gz --output out.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("dense_fp8_gemm", re.compile(r"w8a8_block_fp8|_w8a8")),
    ("fp8_quant", re.compile(r"per_token_group_quant|quant_fp8|fp8_quant", re.I)),
    ("moe_expert_gpu", re.compile(r"GroupProblemShape|grouped_gemm|fused_moe|moe_align|moe_sum|silu_and_mul|marlin|fp4_gemm|cutlass", re.I)),
    ("attention_indexer", re.compile(r"sparse_mla|mqa_logits|indexer|flash_mla|flashmla|paged_attention|page_split|page_mark|kv_compress|topk|top_k", re.I)),
    ("engram", re.compile(r"engram|hash|embedding|index_select", re.I)),
    ("collective", re.compile(r"ncclDevKernel|nccl|allreduce|all_reduce|cross_device_reduce|custom_all_reduce", re.I)),
    ("rope_norm_act", re.compile(r"rope|rotary|norm|silu|gelu|act_and_mul", re.I)),
    ("memcpy_memset", re.compile(r"Memcpy|Memset|memcpy|memset", re.I)),
    ("elementwise_index", re.compile(r"elementwise|vectorized|fill|copy|cast|convert|index|scatter|gather|reduce_kernel|cat|arange|where|masked|cumsum|sort|unique", re.I)),
    ("sampling", re.compile(r"sampl|softmax|argmax|multinomial|topp|top_p|logits", re.I)),
    ("bf16_gemm", re.compile(r"gemm|gemv|cublas|wmma|nvjet|sm90_xmma|sm120", re.I)),
]


def categorize(name: str) -> str:
    for label, rx in CATEGORIES:
        if rx.search(name):
            return label
    return "other"


def short(name: str, n: int = 90) -> str:
    name = re.sub(r"\(.*$", "", name)  # drop template/arg tails
    name = name.replace("void ", "")
    return name if len(name) <= n else name[: n - 1] + "…"


def union_ms(intervals: list[tuple[float, float]]) -> tuple[float, list[tuple[float, float]]]:
    """Total covered time (ms) and the list of idle gaps (start, end) in µs."""
    if not intervals:
        return 0.0, []
    intervals.sort()
    covered = 0.0
    gaps: list[tuple[float, float]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            gaps.append((cur_e, s))
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    covered += cur_e - cur_s
    return covered / 1000.0, gaps


def analyze(path: Path, steps: int, top: int, gap_floor_us: float) -> dict:
    with gzip.open(path, "rt") as fh:
        events = json.load(fh)["traceEvents"]
    gpu = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not gpu:
        raise SystemExit(f"{path}: no GPU events (was the profile captured with activities GPU?)")

    by_cat: dict[str, float] = defaultdict(float)
    by_name: dict[str, list[float]] = defaultdict(list)
    intervals: list[tuple[float, float]] = []
    for e in gpu:
        name = e["name"] if e["cat"] == "kernel" else f"[{e['cat']}] {e['name']}"
        dur = float(e["dur"])
        by_cat[categorize(name)] += dur
        by_name[name].append(dur)
        intervals.append((float(e["ts"]), float(e["ts"]) + dur))

    busy_ms, gaps = union_ms(intervals)
    span_ms = (max(e for _, e in intervals) - min(s for s, _ in intervals)) / 1000.0
    big_gaps = [(s, e) for s, e in gaps if (e - s) >= gap_floor_us]
    gap_ms = sum(e - s for s, e in gaps) / 1000.0
    big_gap_ms = sum(e - s for s, e in big_gaps) / 1000.0

    kernels = sorted(by_name.items(), key=lambda kv: -sum(kv[1]))[:top]
    return {
        "trace": str(path),
        "steps": steps,
        "gpu_events": len(gpu),
        "per_step_ms": {
            "span": round(span_ms / steps, 3),
            "gpu_busy": round(busy_ms / steps, 3),
            "idle_total": round(gap_ms / steps, 3),
            f"idle_gaps_ge_{int(gap_floor_us)}us": round(big_gap_ms / steps, 3),
            f"n_gaps_ge_{int(gap_floor_us)}us": round(len(big_gaps) / steps, 1),
        },
        "by_category_ms_per_step": {k: round(v / 1000.0 / steps, 3) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        "top_kernels": [
            {
                "name": short(n),
                "calls_per_step": round(len(d) / steps, 1),
                "ms_per_step": round(sum(d) / 1000.0 / steps, 3),
                "avg_us": round(sum(d) / len(d), 1),
                "category": categorize(n),
            }
            for n, d in kernels
        ],
        "largest_gaps_us": [round(e - s, 1) for s, e in sorted(big_gaps, key=lambda g: -(g[1] - g[0]))[:10]],
    }


def print_report(r: dict, other: dict | None = None) -> None:
    def cell(x: object) -> str:
        return f"{x:>9}" if not isinstance(x, float) else f"{x:9.3f}"

    print(f"# {r['trace']}  ({r['gpu_events']} GPU events / {r['steps']} steps)")
    hdr = "  ms/step" + ("   compare     delta" if other else "")
    print(f"{'segment':<34}{hdr}")
    for k, v in r["per_step_ms"].items():
        line = f"{k:<34}{cell(v)}"
        if other:
            ov = other["per_step_ms"].get(k, 0.0)
            line += f"{cell(ov)}{cell(round(ov - v, 3))}"
        print(line)
    print()
    print(f"{'category':<34}{hdr}")
    cats = list(r["by_category_ms_per_step"])
    if other:
        cats += [c for c in other["by_category_ms_per_step"] if c not in cats]
    for c in cats:
        v = r["by_category_ms_per_step"].get(c, 0.0)
        line = f"{c:<34}{cell(v)}"
        if other:
            ov = other["by_category_ms_per_step"].get(c, 0.0)
            line += f"{cell(ov)}{cell(round(ov - v, 3))}"
        print(line)
    print()
    print(f"{'top kernels (this trace)':<92}{'calls/st':>9}{'ms/step':>9}{'avg us':>9}")
    for k in r["top_kernels"]:
        print(f"{k['name']:<92}{k['calls_per_step']:>9}{k['ms_per_step']:>9.3f}{k['avg_us']:>9.1f}")
    print(f"\nlargest idle gaps (us): {r['largest_gaps_us']}")
    if other:
        print(f"\n{'top kernels (compare trace)':<92}{'calls/st':>9}{'ms/step':>9}{'avg us':>9}")
        for k in other["top_kernels"]:
            print(f"{k['name']:<92}{k['calls_per_step']:>9}{k['ms_per_step']:>9.3f}{k['avg_us']:>9.1f}")
        print(f"\nlargest idle gaps, compare (us): {other['largest_gaps_us']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", type=Path, help="*.trace.json.gz for one rank")
    ap.add_argument("--steps", type=int, required=True, help="num_steps passed to /start_profile")
    ap.add_argument("--compare", type=Path, help="second trace (same rank) to diff against")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--gap-floor-us", type=float, default=100.0, help="count idle gaps at least this long")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    r = analyze(args.trace, args.steps, args.top, args.gap_floor_us)
    o = analyze(args.compare, args.steps, args.top, args.gap_floor_us) if args.compare else None
    print_report(r, o)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"trace": r, "compare": o}, indent=2) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

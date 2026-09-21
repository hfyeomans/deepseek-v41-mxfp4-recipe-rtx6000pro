#!/usr/bin/env python3
"""E12 kernel census from a parsed torch-profiler component report.

Input: one or more `*-rankN.json` files produced by the sibling V4-Flash recipe's
`experiments/profile_trace_analysis.py` (each has a `kernel_names` list with
`name`, `count`, `sum_ms`, `component`, plus `steps`). Output: per-step ms by
category, the top-N kernels, and which MoE backend actually ran (CUTLASS SM120
block-scaled grouped GEMM vs Marlin), as JSON for `evidence/`.

Categories are assigned by kernel-name regex, first match wins. Sums include
stream overlap (the parser's `union_ms` is the wall-clock-relevant figure and is
reported alongside).

Usage:
  python3 scripts/kernel-census.py path/to/code-c1-rank0.json [more.json] --output evidence/e12.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("moe_expert_fp4_cutlass_sm120", re.compile(r"GroupProblemShape.*Sm120.*BlockScaled")),
    ("moe_expert_marlin", re.compile(r"marlin", re.I)),
    ("dense_fp8", re.compile(r"w8a8|per_token_group_quant|fp8_einsum|fp8_gemm|_fp8_", re.I)),
    ("attention_indexer", re.compile(r"sparse_mla|mqa_logits|kv_compress|sparse_attn|indexer|rope|topk|top_k", re.I)),
    ("collective", re.compile(r"cross_device_reduce|allreduce|all_reduce|nccl|ncclDevKernel", re.I)),
    ("hyper_connections", re.compile(r"^mhc_")),
    ("bf16_gemm", re.compile(r"wmma_tensorop|gemm_bf16|cublas|gemv", re.I)),
    ("norm_act_elementwise", re.compile(r"norm|silu|gelu|act_and_mul|elementwise|vectorized|fill|copy|cast|convert|index|scatter|gather|reduce_kernel", re.I)),
    ("sampling", re.compile(r"sampl|softmax|argmax|multinomial|topp|top_p", re.I)),
]


def categorize(name: str) -> str:
    for label, rx in CATEGORIES:
        if rx.search(name):
            return label
    return "other"


def short(name: str, n: int = 110) -> str:
    if name.startswith("_ZN"):
        name = re.sub(r"ILi(\d+)E", r"\1,", name)
    return name[:n]


def census(path: Path, top: int) -> dict:
    d = json.loads(path.read_text())
    kernels = d["kernel_names"]
    steps = d["steps"]
    n_steps = len(steps) if isinstance(steps, list) else int(steps)
    total = sum(k["sum_ms"] for k in kernels)
    by_cat: dict[str, dict] = {}
    for k in kernels:
        cat = k["component"] if k["component"] == "draft_pipeline" else categorize(k["name"])
        c = by_cat.setdefault(cat, {"sum_ms": 0.0, "count": 0, "kernels": 0})
        c["sum_ms"] += k["sum_ms"]
        c["count"] += k["count"]
        c["kernels"] += 1
    for c in by_cat.values():
        c["ms_per_step"] = round(c["sum_ms"] / n_steps, 4)
        c["share_pct"] = round(100 * c["sum_ms"] / total, 2)
        c["sum_ms"] = round(c["sum_ms"], 3)
    kt = d.get("kernel_total", {}).get("by_device", {}).get("0", {})
    top_k = [
        {
            "name": short(k["name"]),
            "category": categorize(k["name"]),
            "component": k["component"],
            "count": k["count"],
            "launches_per_step": round(k["count"] / n_steps, 2),
            "sum_ms": round(k["sum_ms"], 3),
            "us_per_launch": round(1000 * k["sum_ms"] / k["count"], 1),
            "share_pct": round(100 * k["sum_ms"] / total, 2),
        }
        for k in sorted(kernels, key=lambda k: -k["sum_ms"])[:top]
    ]
    moe_backend = "cutlass_sm120_block_scaled_grouped" if any(
        categorize(k["name"]) == "moe_expert_fp4_cutlass_sm120" for k in kernels
    ) else ("marlin" if any("marlin" in k["name"].lower() for k in kernels) else "unknown")
    return {
        "source": str(path.name),
        "case": d.get("case"),
        "rank": d.get("rank"),
        "steps": n_steps,
        "kernel_sum_ms": round(total, 3),
        "kernel_union_ms": kt.get("union_ms"),
        "kernel_span_ms": kt.get("span_ms"),
        "sum_ms_per_step": round(total / n_steps, 3),
        "union_ms_per_step": round(kt["union_ms"] / n_steps, 3) if kt.get("union_ms") else None,
        "moe_backend_observed": moe_backend,
        "marlin_ms_per_step": round(sum(k["sum_ms"] for k in kernels if "marlin" in k["name"].lower()) / n_steps, 4),
        "by_category": dict(sorted(by_cat.items(), key=lambda kv: -kv[1]["sum_ms"])),
        "top_kernels": top_k,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    out = {"experiment": "E12", "method": "torch profiler component report; sums include stream overlap", "reports": [census(p, a.top) for p in a.reports]}
    for r in out["reports"]:
        print(f"== {r['case']} rank{r['rank']}: {r['steps']} steps, {r['sum_ms_per_step']} ms/step summed, {r['union_ms_per_step']} ms/step union; MoE backend: {r['moe_backend_observed']}")
        for cat, c in r["by_category"].items():
            print(f"  {c['ms_per_step']:7.3f} ms/step {c['share_pct']:5.1f}%  {cat}")
        print("  top kernels:")
        for k in r["top_kernels"][:8]:
            print(f"    {k['sum_ms']:7.2f} ms x{k['count']:5d} {k['us_per_launch']:6.1f} us  {k['category']:28s} {k['name'][:70]}")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.output}")


if __name__ == "__main__":
    main()

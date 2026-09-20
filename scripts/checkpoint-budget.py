#!/usr/bin/env python3
"""Compute an exact per-component byte budget for a safetensors checkpoint.

Reads only each shard's safetensors header (8-byte length prefix + JSON) via
HTTP range requests against the Hugging Face resolve endpoint, or from a local
snapshot directory. No weight bytes are downloaded. Standard library only.

Usage:
  scripts/checkpoint-budget.py deepseek-ai/DeepSeek-V4.1-Flash --revision <sha> --output results/budget.json
  scripts/checkpoint-budget.py /path/to/local/snapshot --output results/budget.json

The component classifier is specific to the DeepSeek-V4.1-Flash tensor
namespace (layers.N.engram.*, layers.N.ffn.experts.*, mtp.*, vision.*,
aligner.*) but degrades to "other" for unknown names, so it is safe to run
against sibling checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

GIB = 2**30

# Bits per element by safetensors dtype string.
DTYPE_BITS = {
    "F64": 64, "I64": 64, "U64": 64,
    "F32": 32, "I32": 32, "U32": 32,
    "F16": 16, "BF16": 16, "I16": 16, "U16": 16,
    "F8_E4M3": 8, "F8_E5M2": 8, "F8_E8M0": 8, "I8": 8, "U8": 8, "BOOL": 8,
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
}

CLASSIFIERS = [
    ("engram_table", re.compile(r"^layers\.\d+\.engram\.embed\.")),
    ("engram_proj", re.compile(r"^layers\.\d+\.engram\.")),
    ("dspark", re.compile(r"^mtp\.")),
    ("routed_experts", re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")),
    ("shared_expert", re.compile(r"^layers\.\d+\.ffn\.shared_experts\.")),
    ("router", re.compile(r"^layers\.\d+\.ffn\.gate\.")),
    ("dense_ffn", re.compile(r"^layers\.\d+\.ffn\.")),
    ("attention", re.compile(r"^layers\.\d+\.attn\.")),
    ("compressor_indexer", re.compile(r"^layers\.\d+\.(compress|indexer|kv_compress|index)")),
    ("hyper_connections", re.compile(r"^layers\.\d+\.hc_")),
    ("norms", re.compile(r"^layers\.\d+\.(attn_norm|ffn_norm|.*norm)")),
    ("vision", re.compile(r"^(vision|aligner|projector)\.")),
    ("embedding_head", re.compile(r"^(embed|head|norm|lm_head|model\.embed)")),
]


def classify(name: str) -> str:
    for label, rx in CLASSIFIERS:
        if rx.search(name):
            return label
    return "other"


def is_scale(name: str) -> bool:
    return name.endswith(("scale", "scale_2", "scale_inv", "weight_scale", "input_scale"))


def tensor_bytes(info: dict) -> int:
    off = info.get("data_offsets")
    if off:
        return int(off[1]) - int(off[0])
    n = 1
    for d in info["shape"]:
        n *= int(d)
    return (n * DTYPE_BITS[info["dtype"]] + 7) // 8


def read_header_local(path: Path) -> dict:
    with path.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        return json.loads(fh.read(n))


def http_get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    tok = os.environ.get("HF_TOKEN")
    if tok and "huggingface.co" in url:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def read_header_remote(repo: str, revision: str, filename: str) -> dict:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    first = http_get(url, {"Range": "bytes=0-7"})
    if len(first) != 8:
        raise RuntimeError(f"range request unsupported for {filename}: got {len(first)} bytes")
    (n,) = struct.unpack("<Q", first)
    body = http_get(url, {"Range": f"bytes=8-{8 + n - 1}"})
    if len(body) != n:
        raise RuntimeError(f"short header for {filename}: {len(body)} != {n}")
    return json.loads(body)


def load_index(source: str, revision: str) -> tuple[dict, bool]:
    p = Path(source)
    if p.is_dir():
        return json.loads((p / "model.safetensors.index.json").read_text()), True
    data = http_get(f"https://huggingface.co/{source}/resolve/{revision}/model.safetensors.index.json")
    return json.loads(data), False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="HF repo id (org/name) or local snapshot directory")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--output", help="write JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    index, local = load_index(args.source, args.revision)
    shards = sorted(set(index["weight_map"].values()))
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))
    log(f"{args.source}@{args.revision}: {len(index['weight_map'])} tensors in {len(shards)} shards")

    by_component: dict[str, dict] = defaultdict(lambda: {"bytes": 0, "scale_bytes": 0, "tensors": 0, "dtypes": defaultdict(int)})
    per_layer_expert_bytes: dict[int, int] = defaultdict(int)
    engram_tables: dict[str, dict] = {}
    total = 0
    for i, shard in enumerate(shards):
        try:
            header = read_header_local(Path(args.source) / shard) if local else read_header_remote(args.source, args.revision, shard)
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            print(f"error reading {shard}: {exc}", file=sys.stderr)
            return 2
        for name, info in header.items():
            if name == "__metadata__":
                continue
            nbytes = tensor_bytes(info)
            total += nbytes
            comp = classify(name)
            entry = by_component[comp]
            entry["tensors"] += 1
            entry["dtypes"][info["dtype"]] += nbytes
            if is_scale(name):
                entry["scale_bytes"] += nbytes
            else:
                entry["bytes"] += nbytes
            if comp == "routed_experts":
                layer = int(name.split(".")[1])
                per_layer_expert_bytes[layer] += nbytes
            if comp == "engram_table":
                engram_tables[name] = {"dtype": info["dtype"], "shape": info["shape"], "bytes": nbytes}
        log(f"  [{i + 1}/{len(shards)}] {shard}: running total {total / GIB:.2f} GiB")

    report = {
        "source": args.source,
        "revision": args.revision,
        "shards": len(shards),
        "tensors": len(index["weight_map"]),
        "index_total_size_bytes": index.get("metadata", {}).get("total_size"),
        "header_total_bytes": total,
        "header_total_gib": round(total / GIB, 3),
        "components": {},
        "engram_tables": engram_tables,
        "routed_expert_layers": len(per_layer_expert_bytes),
        "routed_expert_bytes_per_layer_mean": (sum(per_layer_expert_bytes.values()) / len(per_layer_expert_bytes)) if per_layer_expert_bytes else 0,
    }
    for comp, e in sorted(by_component.items(), key=lambda kv: -(kv[1]["bytes"] + kv[1]["scale_bytes"])):
        report["components"][comp] = {
            "weight_bytes": e["bytes"],
            "scale_bytes": e["scale_bytes"],
            "total_gib": round((e["bytes"] + e["scale_bytes"]) / GIB, 3),
            "tensors": e["tensors"],
            "dtypes_gib": {k: round(v / GIB, 3) for k, v in sorted(e["dtypes"].items(), key=lambda kv: -kv[1])},
        }
    engram_total = sum(t["bytes"] for t in engram_tables.values())
    dspark_vision = sum(by_component[c]["bytes"] + by_component[c]["scale_bytes"] for c in ("dspark", "vision") if c in by_component)
    report["summary_bytes"] = {
        "total": total,
        "engram_tables": engram_total,
        "resident_without_engram_tables": total - engram_total,
        "resident_without_engram_dspark_vision": total - engram_total - dspark_vision,
    }
    report["summary_gib"] = {k: round(v / GIB, 3) for k, v in report["summary_bytes"].items()}

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_gib"], indent=2))
    for comp, c in report["components"].items():
        print(f"{comp:24s} {c['total_gib']:9.3f} GiB  ({c['tensors']} tensors; scales {c['scale_bytes'] / GIB:.2f} GiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Tests for checkpoint-budget.py against a synthetic two-shard checkpoint."""
from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("budget", HERE / "checkpoint-budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def write_shard(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    header, off = {}, 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        nbytes = (n * budget.DTYPE_BITS[dtype] + 7) // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    hb = json.dumps(header).encode()
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        fh.write(b"\0" * off)


def test_classifier():
    cases = {
        "layers.1.engram.embed.weight": "engram_table",
        "layers.1.engram.embed.scale": "engram_table",
        "layers.14.engram.wkv.weight": "engram_proj",
        "layers.3.ffn.experts.17.w1.weight": "routed_experts",
        "layers.3.ffn.experts.17.w1.scale": "routed_experts",
        "layers.3.ffn.shared_experts.w2.weight": "shared_expert",
        "layers.3.ffn.gate.bias_vl": "router",
        "layers.3.attn.indexer.wq_b.weight": "attention",
        "layers.3.attn.compressor.wkv.weight": "attention",
        "layers.3.attn_norm.weight": "norms",
        "layers.3.hc_attn_scale": "hyper_connections",
        "mtp.0.ffn.experts.5.w3.weight": "dspark",
        "mtp.2.markov_head.embed.weight": "dspark",
        "vision.blocks.4.attn.wqkv.weight": "vision",
        "aligner.w1.weight": "vision",
        "embed.weight": "embedding_head",
        "head.weight": "embedding_head",
        "norm.weight": "embedding_head",
        "image_newline": "other",
    }
    for name, want in cases.items():
        got = budget.classify(name)
        assert got == want, f"{name}: {got} != {want}"
    assert budget.is_scale("layers.1.ffn.experts.0.w1.scale")
    assert not budget.is_scale("layers.1.ffn.experts.0.w1.weight")


def test_end_to_end_local():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # expert weight: 128 x 64 INT8-packed FP4 pairs => 8192 bytes; scale 128 x 2 E8M0 => 256 bytes
        s1 = {"layers.0.ffn.experts.0.w1.weight": ("I8", [128, 64]), "layers.0.ffn.experts.0.w1.scale": ("F8_E8M0", [128, 2]),
              "layers.0.attn.wq_a.weight": ("F8_E4M3", [64, 64])}
        # engram table 1000 rows x 256 FP8 = 256000; scale 1000 x 8 = 8000; dspark bf16 16x16 = 512
        s2 = {"layers.1.engram.embed.weight": ("F8_E4M3", [1000, 256]), "layers.1.engram.embed.scale": ("F8_E8M0", [1000, 8]),
              "mtp.0.norm.weight": ("BF16", [16, 16])}
        write_shard(root / "a.safetensors", s1)
        write_shard(root / "b.safetensors", s2)
        wm = {k: "a.safetensors" for k in s1} | {k: "b.safetensors" for k in s2}
        (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 8192 + 256 + 4096 + 256000 + 8000 + 512}, "weight_map": wm}))
        out = root / "budget.json"
        subprocess.run([sys.executable, str(HERE / "checkpoint-budget.py"), str(root), "--quiet", "--output", str(out)], check=True, capture_output=True)
        rep = json.loads(out.read_text())
        assert rep["header_total_bytes"] == 8192 + 256 + 4096 + 256000 + 8000 + 512, rep["header_total_bytes"]
        assert rep["components"]["routed_experts"]["weight_bytes"] == 8192
        assert rep["components"]["routed_experts"]["scale_bytes"] == 256
        assert rep["components"]["engram_table"]["weight_bytes"] == 256000
        assert rep["components"]["engram_table"]["scale_bytes"] == 8000
        assert rep["components"]["dspark"]["weight_bytes"] == 512
        assert rep["components"]["attention"]["weight_bytes"] == 4096
        # resident_without_engram_tables must exclude both table weight and table scale
        assert rep["summary_bytes"]["engram_tables"] == 256000 + 8000
        assert rep["summary_bytes"]["resident_without_engram_tables"] == 8192 + 256 + 4096 + 512
        assert rep["summary_bytes"]["resident_without_engram_dspark_vision"] == 8192 + 256 + 4096
        assert rep["engram_tables"]["layers.1.engram.embed.weight"]["shape"] == [1000, 256]


def test_committed_evidence_consistent():
    ev = HERE.parent / "evidence" / "budget-upstream-bf16fp8.json"
    rep = json.loads(ev.read_text())
    assert rep["header_total_bytes"] == rep["index_total_size_bytes"] == 510286023000
    comps = rep["components"]
    total = sum(c["weight_bytes"] + c["scale_bytes"] for c in comps.values())
    assert total == rep["header_total_bytes"]
    assert comps["engram_table"]["tensors"] == 4
    assert rep["routed_expert_layers"] == 40
    assert comps["routed_experts"]["tensors"] == 40 * 384 * 6


if __name__ == "__main__":
    for fn in (test_classifier, test_end_to_end_local, test_committed_evidence_consistent):
        fn()
        print(f"ok {fn.__name__}")

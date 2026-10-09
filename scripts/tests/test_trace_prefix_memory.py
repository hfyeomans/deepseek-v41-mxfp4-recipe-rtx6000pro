#!/usr/bin/env python3
"""Tests for the allocator-snapshot summary in `trace-prefix-memory.py` (E34e e-ii). Pure Python on a
hand-built `_dump_snapshot`-shaped dict; no torch, no server.

Wrong implementations these are written against:
  - counting inactive (cached, free) blocks as allocated, which would hide the retention the
    experiment is looking for -> test_reserved_allocated_inactive.
  - attributing a block to the torch-internal frame on top of the stack instead of the sglang /
    kt caller, or dropping blocks that predate recording -> test_site_attribution.
  - replaying the window trace without frees, so the high-water mark equals the sum of allocs
    -> test_window_peak.
  - a diff that only reports sites present in both snapshots -> test_diff_sites.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("trace_prefix_memory", HERE.parent / "trace-prefix-memory.py")
tpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tpm)

TORCH = {"filename": "/usr/lib/python3/site-packages/torch/_tensor.py", "line": 10, "name": "clone"}
SGL = {"filename": "/sgl-workspace/sglang/python/sglang/srt/layers/attention/nsa/indexer.py", "line": 1132,
       "name": "select_candidate_block_keep"}
KT = {"filename": "/sgl-workspace/sglang/python/sglang/srt/layers/moe/kt_prefill.py", "line": 40, "name": "export"}
CPP = {"filename": "c10/cuda/CUDACachingAllocator.cpp", "line": 1, "name": "malloc"}


def block(size, state, frames=None):
    return {"size": size, "requested_size": size, "address": 0, "state": state, "frames": frames}


def snapshot():
    return {
        "segments": [
            {"device": 0, "total_size": 1000, "stream": 0, "blocks": [
                block(300, "active_allocated", [CPP, TORCH, SGL]),
                block(200, "inactive"),
                block(100, "active_allocated", None),            # predates recording
                block(400, "inactive"),
            ]},
            {"device": 0, "total_size": 500, "stream": 0, "blocks": [
                block(500, "active_allocated", [TORCH, KT, SGL]),  # innermost own frame is KT
            ]},
        ],
        "device_traces": [[
            {"action": "alloc", "size": 100}, {"action": "alloc", "size": 300},
            {"action": "free_completed", "size": 100}, {"action": "alloc", "size": 50},
            {"action": "segment_alloc", "size": 500}, {"action": "free_completed", "size": 300},
        ]],
    }


def test_reserved_allocated_inactive():
    s = tpm.summarise_snapshot(snapshot())["0"]
    assert s["reserved_bytes"] == 1500 and s["segments"] == 2
    assert s["allocated_bytes"] == 900, s
    assert s["inactive_bytes"] == 600 and s["inactive_blocks"] == 2 and s["largest_inactive_bytes"] == 400


def test_site_attribution():
    assert tpm.site_of(None) == "pre-recording" and tpm.site_of([]) == "pre-recording"
    assert tpm.site_of([CPP, TORCH, SGL]).startswith("sglang/srt/layers/attention/nsa/indexer.py:1132 select_candidate_block_keep")
    assert tpm.site_of([TORCH, KT, SGL]).startswith("sglang/srt/layers/moe/kt_prefill.py:40")
    assert tpm.site_of([CPP, TORCH]).startswith("torch/_tensor.py:10")  # no own frame: innermost Python frame
    assert tpm.site_of([CPP]) == "c10/cuda/CUDACachingAllocator.cpp:1 malloc"
    sites = {x["site"]: x for x in tpm.summarise_snapshot(snapshot())["0"]["live_sites"]}
    assert sites["pre-recording"]["bytes"] == 100
    assert sum(x["bytes"] for x in sites.values()) == 900 and len(sites) == 3


def test_window_peak():
    w = tpm.summarise_snapshot(snapshot())["0"]["window"]
    assert w["peak_delta_bytes"] == 400, w   # 100 + 300, before the first free; not 450 (sum of allocs)
    assert w["net_delta_bytes"] == 50 and w["allocs"] == 3 and w["frees"] == 2 and w["segment_allocs"] == 1


def test_diff_sites():
    a = tpm.summarise_snapshot(snapshot())
    b_snap = snapshot()
    b_snap["segments"][0]["blocks"][0]["size"] = 700          # SGL site grows by 400
    b_snap["segments"][1]["blocks"] = [block(500, "inactive")]  # KT block freed (now cached)
    b_snap["segments"].append({"device": 0, "total_size": 64, "stream": 0, "blocks": [
        block(64, "active_allocated", [TORCH, {"filename": "/sgl-workspace/sglang/python/sglang/srt/new.py", "line": 1, "name": "f"}])]})
    b = tpm.summarise_snapshot(b_snap)
    d = tpm.diff_devices(a, b)["0"]
    assert d["reserved_bytes"] == 64 and d["allocated_bytes"] == 700 + 100 + 64 - 900
    assert d["inactive_bytes"] == 500
    by = {m["site"].split(" ")[0]: m["delta_bytes"] for m in d["sites"]}
    assert by["sglang/srt/layers/attention/nsa/indexer.py:1132"] == 400
    assert by["sglang/srt/layers/moe/kt_prefill.py:40"] == -500       # vanished site still reported
    assert by["sglang/srt/new.py:1"] == 64                             # new site reported
    assert d["sites"][0]["site"].startswith("sglang/srt/layers/moe/kt_prefill.py")  # sorted by |delta|
    assert tpm.diff_devices(None, b) == {}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"ok: {len(fns)} trace-prefix-memory tests")

#!/usr/bin/env python3
"""Tests for feasibility-calc.py: the boundary cases the verdict rests on."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent / "evidence" / "budget-upstream-bf16fp8.json"
spec = importlib.util.spec_from_file_location("calc", HERE / "feasibility-calc.py")
calc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calc)
GIB = 2**30


def run(*args: str) -> tuple[int, dict]:
    p = subprocess.run([sys.executable, str(HERE / "feasibility-calc.py"), str(EVIDENCE), "--json", *args], capture_output=True, text=True)
    return p.returncode, json.loads(p.stdout)


def test_two_gpu_does_not_fit_any_layout():
    rc, out = run("--gpus", "2")
    assert rc == 3
    for row in out["layouts"]:
        assert not row["fits"], row
        assert row["vram_margin_gib"] < -100, row  # the shortfall is >100 GiB, not marginal


def test_two_gpu_text_only_still_does_not_fit_even_at_full_utilisation():
    rc, out = run("--gpus", "2", "--no-dspark", "--util", "1.0", "--gpu-reserve-gib", "0", "--context-tokens", "1", "--layout", "engram-host")
    assert rc == 3
    assert out["layouts"][0]["vram_margin_gib"] < -80, out


def test_four_gpu_calibration_against_0xsero():
    # 4 x RTX PRO 6000, 128 GB host: Engram on NVMe fits, Engram in host DRAM does not.
    rc, out = run("--gpus", "4", "--host-gib", "128", "--context-tokens", "524288", "--sequences", "4")
    assert rc == 3  # 'all' includes gpu-only and engram-host which fail
    rows = {r["layout"]: r for r in out["layouts"]}
    assert rows["engram-nvme"]["fits"]
    assert not rows["engram-host"]["fits"] and rows["engram-host"]["host_margin_gib"] < 0
    assert not rows["gpu-only"]["fits"]


def test_four_gpu_this_host_engram_in_dram_fits():
    rc, out = run("--gpus", "4", "--host-gib", "246.8", "--layout", "engram-host", "--context-tokens", "262144", "--sequences", "8")
    assert rc == 0, out
    row = out["layouts"][0]
    assert row["fits"] and row["host_margin_gib"] > 30 and row["vram_margin_gib"] > 50


def test_kv_bytes_math():
    b = calc.load_budget(EVIDENCE)
    r = calc.evaluate(b, gpus=4, gpu_gib=95.6, host_gib=246.8, util=0.9, gpu_reserve_gib=4, host_reserve_gib=24,
                      context_tokens=1_000_000, sequences=1, include_dspark=True, include_vision=False, layout="engram-host", engram_cache_gib=64)
    assert abs(r["kv_cache_gib"] - 890 * 1_000_000 / GIB) < 1e-3
    r2 = calc.evaluate(b, gpus=4, gpu_gib=95.6, host_gib=246.8, util=0.9, gpu_reserve_gib=4, host_reserve_gib=24,
                       context_tokens=1_000_000, sequences=3, include_dspark=True, include_vision=False, layout="engram-host", engram_cache_gib=64)
    assert abs(r2["kv_cache_gib"] - 3 * r["kv_cache_gib"]) < 1e-3


def test_engram_tables_counted_only_in_gpu_only_layout():
    b = calc.load_budget(EVIDENCE)
    common = dict(gpus=8, gpu_gib=95.6, host_gib=512, util=0.9, gpu_reserve_gib=4, host_reserve_gib=24,
                  context_tokens=1, sequences=1, include_dspark=False, include_vision=False, engram_cache_gib=64)
    g = calc.evaluate(b, layout="gpu-only", **common)
    h = calc.evaluate(b, layout="engram-host", **common)
    assert abs((g["resident_weights_gib"] - h["resident_weights_gib"]) - b["engram_tables"] / GIB) < 0.01


if __name__ == "__main__":
    for fn in (test_two_gpu_does_not_fit_any_layout, test_two_gpu_text_only_still_does_not_fit_even_at_full_utilisation,
               test_four_gpu_calibration_against_0xsero, test_four_gpu_this_host_engram_in_dram_fits, test_kv_bytes_math,
               test_engram_tables_counted_only_in_gpu_only_layout):
        fn()
        print(f"ok {fn.__name__}")

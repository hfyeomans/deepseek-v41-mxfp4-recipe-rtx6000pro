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


def resident_rows(out: dict) -> list[dict]:
    return [r for r in out["layouts"] if not r["layout"].startswith("hybrid")]


def hybrid_rows(out: dict) -> list[dict]:
    return [r for r in out["layouts"] if r["layout"].startswith("hybrid")]


def test_two_gpu_does_not_fit_any_resident_layout():
    rc, out = run("--gpus", "2")
    assert rc == 3
    rows = resident_rows(out)
    assert len(rows) == 3
    for row in rows:
        assert not row["fits"], row
        assert row["vram_margin_gib"] < -100, row  # the shortfall is >100 GiB, not marginal


def test_two_gpu_hybrid_fits_with_nvme_engram_but_not_fp8_dram_engram():
    rc, out = run("--gpus", "2", "--layout", "hybrid-cpu", "--engram", "nvme")
    assert rc == 0, out
    (row,) = hybrid_rows(out)
    assert row["fits"] and row["tp_valid"]
    assert 0.5 < row["gpu_expert_fraction"] < 0.65, row  # ~58 % of routed experts fit next to the non-expert weights
    assert 100 < row["cpu_expert_gib"] < 125, row
    assert row["gpu_experts"] + row["cpu_experts"] == calc.MOE_LAYERS * calc.ROUTED_EXPERTS
    assert row["decode_ceiling_cpu_tok_s"] > row["decode_ceiling_pcie_stream_tok_s"], row  # DRAM beats PCIe streaming at defaults

    rc, out = run("--gpus", "2", "--layout", "hybrid-cpu", "--engram", "dram")
    assert rc == 3
    (row,) = hybrid_rows(out)
    assert not row["fits"] and row["host_margin_gib"] < -50, row
    for mode in ("dram-fp4", "dram-q3k"):
        rc, out = run("--gpus", "2", "--layout", "hybrid-cpu", "--engram", mode)
        assert rc == 0, (mode, out)


def test_hybrid_gpu_count_scaling():
    rc, out = run("--gpus", "2,3,4", "--layout", "hybrid-cpu")
    assert rc == 0, out
    rows = {r["gpus"]: r for r in hybrid_rows(out)}
    assert set(rows) == {2, 3, 4}
    assert rows[4]["gpu_expert_fraction"] == 1.0 and rows[4]["cpu_experts"] == 0 and rows[4]["dram_bytes_per_token_gib"] == 0
    assert rows[4]["decode_ceiling_cpu_tok_s"] == round(1e3 / calc.DEFAULT_GPU_MS_PER_TOKEN, 1)
    assert rows[2]["gpu_expert_fraction"] < rows[3]["gpu_expert_fraction"] < rows[4]["gpu_expert_fraction"]
    assert rows[2]["decode_ceiling_cpu_tok_s"] < rows[3]["decode_ceiling_cpu_tok_s"] < rows[4]["decode_ceiling_cpu_tok_s"]
    assert rows[2]["tp_valid"] and rows[4]["tp_valid"] and not rows[3]["tp_valid"]


def test_hybrid_ceiling_responds_to_bandwidth_and_skew():
    b = calc.load_budget(EVIDENCE)
    common = dict(gpus=2, gpu_gib=95.6, host_gib=246.8, util=0.95, gpu_reserve_gib=4, host_reserve_gib=24,
                  context_tokens=262144, sequences=1, include_dspark=True, include_vision=False, engram="nvme",
                  engram_cache_gib=64, pcie_gbps_per_gpu=50, gpu_ms_per_token=5.0)
    slow = calc.evaluate_hybrid(b, dram_gbps=146, activation_skew=0.0, **common)
    fast = calc.evaluate_hybrid(b, dram_gbps=292, activation_skew=0.0, **common)
    skewed = calc.evaluate_hybrid(b, dram_gbps=146, activation_skew=0.5, **common)
    assert slow["decode_ceiling_cpu_tok_s"] < fast["decode_ceiling_cpu_tok_s"]
    assert abs(fast["dram_ms_per_token"] * 2 - slow["dram_ms_per_token"]) < 0.02
    assert abs(skewed["dram_bytes_per_token_gib"] * 2 - slow["dram_bytes_per_token_gib"]) < 0.01
    # Per-token DRAM traffic = 240 activations x CPU share x bytes per expert.
    per_expert = b["routed_experts"] / (calc.MOE_LAYERS * calc.ROUTED_EXPERTS)
    expected = calc.MOE_LAYERS * calc.EXPERTS_PER_TOKEN * (1 - slow["gpu_expert_fraction"]) * per_expert / GIB
    assert abs(slow["dram_bytes_per_token_gib"] - expected) < 0.01


def test_split_ceiling_sums_bandwidths_not_times():
    """Option B2 (E4b): CPU reads and GPU DMA reads of DRAM are additive, so the split
    path's DRAM time is bytes / (dram + pcie x gpus). A wrong implementation that summed
    the two times, or averaged the two ceilings, would fail the asymmetric case below."""
    b = calc.load_budget(EVIDENCE)
    common = dict(gpus=2, gpu_gib=95.6, host_gib=246.8, util=0.95, gpu_reserve_gib=4, host_reserve_gib=24,
                  context_tokens=262144, sequences=1, include_dspark=True, include_vision=False, engram="nvme",
                  engram_cache_gib=64, gpu_ms_per_token=5.0, activation_skew=0.0)
    r = calc.evaluate_hybrid(b, dram_gbps=100, pcie_gbps_per_gpu=30, **common)  # 100 + 60 = 160 GB/s
    bytes_per_token = r["dram_bytes_per_token_gib"] * GIB
    assert abs(r["split_ms_per_token"] - bytes_per_token / 160e9 * 1e3) < 0.02
    assert r["decode_ceiling_split_tok_s"] > r["decode_ceiling_cpu_tok_s"] > r["decode_ceiling_pcie_stream_tok_s"]
    assert abs(r["decode_ceiling_split_tok_s"] - 1e3 / (5.0 + r["split_ms_per_token"])) < 0.1
    # The FCLK-2000 figures (E1b 125, E4 57 x 2) give the FEASIBILITY.md table's 50 / 78 tok/s.
    d = calc.evaluate_hybrid(b, dram_gbps=125.0, pcie_gbps_per_gpu=calc.DEFAULT_PCIE_GBPS_PER_GPU, **common)
    assert 77 <= d["decode_ceiling_split_tok_s"] <= 79, d["decode_ceiling_split_tok_s"]
    assert 49 <= d["decode_ceiling_cpu_tok_s"] <= 51, d["decode_ceiling_cpu_tok_s"]
    # The default is the BIOS slot-3 (FCLK 2100, E27) ceiling: 131 GB/s -> 52 / 79 tok/s.
    assert calc.DEFAULT_DRAM_GBPS == 131.0
    d = calc.evaluate_hybrid(b, dram_gbps=calc.DEFAULT_DRAM_GBPS, pcie_gbps_per_gpu=calc.DEFAULT_PCIE_GBPS_PER_GPU, **common)
    assert 78 <= d["decode_ceiling_split_tok_s"] <= 80, d["decode_ceiling_split_tok_s"]
    assert 51 <= d["decode_ceiling_cpu_tok_s"] <= 53, d["decode_ceiling_cpu_tok_s"]


def test_bad_gpu_list_is_an_input_error():
    p = subprocess.run([sys.executable, str(HERE / "feasibility-calc.py"), str(EVIDENCE), "--gpus", "2,x"], capture_output=True, text=True)
    assert p.returncode == 2 and "--gpus" in p.stderr


def test_two_gpu_text_only_still_does_not_fit_even_at_full_utilisation():
    rc, out = run("--gpus", "2", "--no-dspark", "--util", "1.0", "--gpu-reserve-gib", "0", "--context-tokens", "1", "--layout", "engram-host")
    assert rc == 3
    assert out["layouts"][0]["vram_margin_gib"] < -80, out


def test_four_gpu_calibration_against_0xsero():
    # 4 x RTX PRO 6000, 128 GB host: Engram on NVMe fits, Engram in host DRAM does not.
    rc, out = run("--gpus", "4", "--host-gib", "128", "--context-tokens", "524288", "--sequences", "4")
    assert rc == 3  # 'all' includes gpu-only and engram-host which fail
    rows = {r["layout"]: r for r in resident_rows(out)}
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
    for fn in (test_two_gpu_does_not_fit_any_resident_layout, test_two_gpu_hybrid_fits_with_nvme_engram_but_not_fp8_dram_engram,
               test_hybrid_gpu_count_scaling, test_hybrid_ceiling_responds_to_bandwidth_and_skew, test_split_ceiling_sums_bandwidths_not_times,
               test_bad_gpu_list_is_an_input_error,
               test_two_gpu_text_only_still_does_not_fit_even_at_full_utilisation,
               test_four_gpu_calibration_against_0xsero, test_four_gpu_this_host_engram_in_dram_fits, test_kv_bytes_math,
               test_engram_tables_counted_only_in_gpu_only_layout):
        fn()
        print(f"ok {fn.__name__}")

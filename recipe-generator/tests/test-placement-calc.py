#!/usr/bin/env python3
"""Self-tests for placement_calc.py.

The core assertion is the calibration rule applied to ourselves: the calculator
must reproduce the repo's measured qualified deployment (219/165 experts, C1
60.8, C4 124.0, prefill 4,344) from the budget + hardware JSONs alone, and it
must refuse to label unanchored projections as trusted. Run GPU-free:
    python3 recipe-generator/tests/test-placement-calc.py
Standard library only.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
CALC = HERE.parent / "placement_calc.py"


def run(*args):
    return subprocess.run([sys.executable, str(CALC), *args],
                          capture_output=True, text=True)


def die(msg):
    print(f"FAIL {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    # 1. The real DSV4.1 case must reproduce the qualified deployment exactly
    #    (this is the calculator's own calibration anchor - evidence/t352).
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/dsv41-flash/hardware.json", "--json")
    if r.returncode != 0:
        die(f"dsv41 case exit {r.returncode}: {r.stderr}")
    out = json.loads(r.stdout)
    plan = out["plan"]
    if plan["trust"] != "calibrated":
        die(f"trust {plan['trust']!r}, want calibrated")
    if plan["gpu_experts"] != 8760 or plan["cpu_experts"] != 6600:
        die(f"expert split {plan['gpu_experts']}/{plan['cpu_experts']}, want 8760/6600 (219/165 per layer)")
    if plan["cpu_share"] < 0.14 or plan["cpu_share"] > 0.16:
        die(f"cpu_share {plan['cpu_share']:.3f}, want ~0.15 (measured 12-15%)")
    if abs(plan["projected"]["c1_tok_s"] - 60.8) > 0.5:
        die(f"projected C1 {plan['projected']['c1_tok_s']}, want 60.8 (measured)")
    if abs(plan["projected"]["c4_tok_s"] - 124.0) > 2.0:
        die(f"projected C4 {plan['projected']['c4_tok_s']}, want 124.0 (measured)")
    if abs(plan["projected"]["prefill_cold_tok_s"] - 4344) > 50:
        die(f"projected prefill {plan['projected']['prefill_cold_tok_s']}, want 4344 (measured)")
    if plan["engram_tier"] != "nvme" or plan["deferral"] != 4 or plan["chunk"] != 8192:
        die(f"levers engram={plan['engram_tier']} D={plan['deferral']} chunk={plan['chunk']}, want nvme/4/8192")
    print("ok   dsv41 real case reproduces the measured qualified deployment")

    # 2. Without calibration the same inputs must be labeled provisional.
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/dsv41-flash/hardware.json", "--json", "--no-calibration")
    out = json.loads(r.stdout)
    if out["plan"]["trust"] != "provisional":
        die(f"no-calibration trust {out['plan']['trust']!r}, want provisional")
    print("ok   calibration rule enforced (provisional without an anchor)")

    # 3. The synthetic 211 GiB case: every expert must fit VRAM (the owner's case).
    r = run("examples/synthetic-211g/budget.json",
            "--hardware", "examples/synthetic-211g/hardware.json", "--json")
    if r.returncode != 0:
        die(f"synthetic case exit {r.returncode}: {r.stderr}")
    out = json.loads(r.stdout)
    plan = out["plan"]
    if plan["cpu_experts"] != 0 or plan["gpu_experts"] != 15360:
        die(f"synthetic split {plan['gpu_experts']}/{plan['cpu_experts']}, want 15360/0 (all resident)")
    if plan["engram_tier"] != "dram":
        die(f"synthetic engram tier {plan['engram_tier']!r}, want dram (60 GiB fits 246.8 GiB host)")
    print("ok   synthetic 211 GiB case: 100% experts in VRAM, Engram in DRAM")

    # 4. Small-model published-reference cases (llama.cpp, no workspace model).
    r = run("examples/qwen3-30b-a3b/budget.json",
            "--hardware", "examples/qwen3-30b-a3b/hardware-rtx5090.json",
            "--quant", "gguf-q4_k_m", "--context-tokens", "32768", "--sequences", "1", "--json")
    if r.returncode != 0:
        die(f"5090 case exit {r.returncode}: {r.stderr}")
    out = json.loads(r.stdout)
    if abs(out["plan"]["projected"]["c1_tok_s"] - 226.1) > 0.5:
        die(f"5090 C1 {out['plan']['projected']['c1_tok_s']}, want 226.1 (published)")
    if out["plan"]["projected"]["prefill_cold_tok_s"] < 7000:
        die(f"5090 prefill {out['plan']['projected']['prefill_cold_tok_s']}, want ~7093 (published)")
    print("ok   Qwen3-30B-A3B on RTX 5090 reproduces the published anchor")

    # 5. A dense model (no experts) must produce a plain full-VRAM plan.
    r = run("examples/qwen3-30b-a3b/budget.json",
            "--hardware", "examples/qwen3-30b-a3b/hardware-rtx3090.json",
            "--quant", "gguf-q4_k_m", "--context-tokens", "4096", "--sequences", "1", "--json")
    if r.returncode != 0:
        die(f"3090 case exit {r.returncode}: {r.stderr}")
    out = json.loads(r.stdout)
    if abs(out["plan"]["projected"]["c1_tok_s"] - 153.6) > 0.5:
        die(f"3090 C1 {out['plan']['projected']['c1_tok_s']}, want 153.6 (published)")
    print("ok   Qwen3-30B-A3B on RTX 3090 reproduces the published anchor")

    # 6. A layout that cannot fit must exit 3 (the CI-pinnable verdict):
    #    the 475 GiB model cannot fit a 24 GiB card + 64 GiB host anywhere
    #    (zero VRAM expert capacity and the host cannot hold experts+Engram).
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/qwen3-30b-a3b/hardware-rtx3090.json",
            "--context-tokens", "4096", "--sequences", "1", "--json")
    if r.returncode != 3:
        die(f"475 GiB model on a 24 GiB card exit {r.returncode}, want 3 (does not fit)")
    print("ok   impossible layout exits 3 (fits/does-not-fit is CI-pinnable)")

    # 7. The emitter must produce a complete, runnable, identifier-clean tree.
    import re
    import tempfile
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as tmp:
        out = P(tmp) / "recipe"
        r = subprocess.run(
            [sys.executable, str(HERE.parent / "emit_recipe.py"),
             "--budget", "examples/qwen3-30b-a3b/budget.json",
             "--hardware", "examples/qwen3-30b-a3b/hardware-rtx3090.json",
             "--quant", "gguf-q4_k_m", "--context-tokens", "4096", "--sequences", "1",
             "--engine", "llama-cpp", "--out", str(out),
             "--model", "Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M"],
            capture_output=True, text=True)
        if r.returncode != 0:
            die(f"emitter exit {r.returncode}: {r.stderr}")
        for name in ("README.md", "MANIFEST.json", "PROVENANCE.md", "launch.sh",
                     "preflight.sh", "verify.sh", "engine-notes.md",
                     "registry-surface.json", ".gitignore"):
            if not (out / name).is_file():
                die(f"emitted tree missing {name}")
        bad = re.compile(r'/(?:home|media)/[A-Za-z0-9._-]+/|\b(?:192\.168|10\.\d{1,3}'
                         r'|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b')
        for p in out.rglob("*"):
            if p.is_file() and bad.search(p.read_text(errors="ignore")):
                die(f"emitted file {p.name} contains a CI-forbidden identifier")
    print("ok   emitter produces a complete runnable identifier-clean tree")

    # 8. Engine selection (ROADMAP phase A): the engine profile must feed the plan.
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/dsv41-flash/hardware.json",
            "--engine", "llama-cpp", "--json")
    if r.returncode != 0:
        die(f"llama-cpp engine case exit {r.returncode}: {r.stderr}")
    out = json.loads(r.stdout)
    if out.get("engine", {}).get("name") != "llama-cpp":
        die("plan missing engine block")
    if out["plan"]["deferral"] != 0:
        die(f"deferral {out['plan']['deferral']}, want 0 (llama-cpp has no deferral lever)")
    if "forced to 0" not in (out["plan"].get("d_guard") or ""):
        die("missing engine deferral guard note")
    if "deferral unavailable" not in (out["engine"].get("notes") or ""):
        die("engine notes should say deferral is unavailable")
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/dsv41-flash/hardware.json",
            "--engine", "tensorfold")
    if "REJ" not in r.stdout:
        die("tensorfold plan should carry the ledger REJ verdict + reopen trigger")
    r = run("examples/dsv41-flash/budget.json",
            "--hardware", "examples/dsv41-flash/hardware.json",
            "--engine", "sglang-hybrid", "--json")
    out = json.loads(r.stdout)
    if out["plan"]["gpu_experts"] != 8760 or out["plan"]["deferral"] != 4:
        die("sglang-hybrid engine profile broke the calibrated plan (219/165, D=4)")
    print("ok   engine profiles feed the plan (deferral guard, REJ warning, anchor intact)")

    print("all placement-calc self-tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

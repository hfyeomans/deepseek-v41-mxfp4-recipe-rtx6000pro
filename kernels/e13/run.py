#!/usr/bin/env python3
"""E13 stage 1 driver: run the MXFP4 grouped GEMM binary at DeepSeek-V4.1-Flash expert shapes.

Stdlib only. Generates CUTLASS-style benchmark files (one "idx MxNxK" line per group), runs the binary
inside the CUDA devel image with --gpus all, parses the two schedules' output and writes JSON.

Per-expert weight shapes (plan/RESEARCH.md): gate/up fused [N=4608, K=5120], down [N=5120, K=2304].
Decode: top-6 experts, 1..8 tokens each (M = tokens routed to that expert).
Prefill: a 4096-token chunk spread over 384 experts × top-6 = 24,576 token-expert pairs ≈ 64 tokens/expert.

Usage:
  kernels/e13/run.py [--binary kernels/e13/mxf4_grouped_gemm_n128k128] [--output results/e13-stage1.json]
                     [--iterations 50] [--skip-prefill] [--verify-decode]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get("IMAGE", "dsv4-sm120:cuda-devel-https")

SHAPES = {
    "gate_up": {"n": 4608, "k": 5120},
    "down": {"n": 5120, "k": 2304},
}


def tp_shapes(tp: int):
    # Tensor parallel splits the expert intermediate dimension: gate/up loses rows (N),
    # down loses its reduction dimension (K). TP1 returns SHAPES unchanged.
    return {
        "gate_up": {"n": SHAPES["gate_up"]["n"] // tp, "k": SHAPES["gate_up"]["k"]},
        "down": {"n": SHAPES["down"]["n"], "k": SHAPES["down"]["k"] // tp},
    }


def cases(skip_prefill: bool, tp: int = 1, decode_groups: int = 6):
    out = []
    for proj, s in tp_shapes(tp).items():
        for m in (1, 4, 8):
            out.append({"name": f"decode_{proj}_m{m}_g{decode_groups}", "proj": proj, "phase": "decode",
                        "groups": [(m, s["n"], s["k"])] * decode_groups})
        if not skip_prefill:
            out.append({"name": f"prefill_{proj}_m64_g384", "proj": proj, "phase": "prefill",
                        "groups": [(64, s["n"], s["k"])] * 384})
            out.append({"name": f"prefill_{proj}_m4096_g6", "proj": proj, "phase": "prefill_dense_proxy",
                        "groups": [(4096, s["n"], s["k"])] * 6})
    return out


def weight_bytes(groups):
    # MXFP4: 0.5 byte/element + 1 byte UE8M0 scale per 32 elements
    return sum(n * k * 0.5 + n * k / 32 for _, n, k in groups)


def flops(groups):
    return sum(2 * m * n * k for m, n, k in groups)


RE_SCHED = re.compile(r"Running kernel with (\w+) kernel schedule")
RE_RT = re.compile(r"Avg runtime : ([0-9.eE+-]+) ms")
RE_TF = re.compile(r"TFLOPS\s*: ([0-9.eE+-]+)")
RE_DISP = re.compile(r"Disposition: (\w+)")


def parse(text):
    res, cur = {}, None
    for line in text.splitlines():
        m = RE_SCHED.search(line)
        if m:
            cur = m.group(1).lower()
            res[cur] = {}
            continue
        if cur is None:
            continue
        for key, rx in (("avg_ms", RE_RT), ("tflops", RE_TF), ("verified", RE_DISP)):
            m = rx.search(line)
            if m:
                v = m.group(1)
                res[cur][key] = (v == "Passed") if key == "verified" else float(v)
    return res


def run_case(binary, case, iterations, verify, no_reinit=False):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", dir=REPO / "results", delete=False) as f:
        for i, (m, n, k) in enumerate(case["groups"]):
            f.write(f"{i} {m}x{n}x{k}\n")
        bench = Path(f.name)
    try:
        cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{REPO}:/w", "-w", "/w",
               "--entrypoint", f"./{binary}", IMAGE,
               f"--benchmark=/w/{bench.relative_to(REPO)}", f"--iterations={iterations}"]
        if not verify:
            cmd.append("--no_verif")
        if no_reinit:
            cmd.append("--no_reinit")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        out = p.stdout + p.stderr
        if p.returncode != 0:
            print(out[-2000:], file=sys.stderr)
            return {"error": f"exit {p.returncode}", "raw_tail": out[-2000:]}
        parsed = parse(out)
        wb = weight_bytes(case["groups"])
        fl = flops(case["groups"])
        for sched in parsed.values():
            if "avg_ms" in sched:
                sched["weight_GBps"] = wb / (sched["avg_ms"] * 1e-3) / 1e9
                sched["flops_check_tflops"] = fl / (sched["avg_ms"] * 1e-3) / 1e12
        return {"schedules": parsed, "weight_bytes": wb, "flops": fl, "wall_s": round(wall, 1)}
    finally:
        bench.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="kernels/e13/mxf4_grouped_gemm_n128k128")
    ap.add_argument("--output", default="results/e13-stage1.json")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--no-reinit", action="store_true",
                    help="pass --no_reinit to the binary: time gemm.run() only (drop per-iteration initialize)")
    ap.add_argument("--verify-decode", action="store_true",
                    help="run the host reference check on decode cases (slow: ~1 GFLOP on CPU per case)")
    ap.add_argument("--tp", type=int, default=1, help="per-rank shapes under tensor parallel (gate/up N and down K divided by TP)")
    ap.add_argument("--decode-groups", type=int, default=6,
                    help="expert groups per decode launch (6 = one token x top-6; 36 = a 6-token DSpark verify step)")
    args = ap.parse_args()
    (REPO / "results").mkdir(exist_ok=True)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,power.limit",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    results = {"experiment": "E13 stage 1", "binary": args.binary, "image": IMAGE, "iterations": args.iterations, "reinit_in_loop": not args.no_reinit,
               "gpu0": gpu[0] if gpu else None, "shapes": tp_shapes(args.tp), "tp": args.tp, "decode_groups": args.decode_groups,
               "notes": ["MXFP4 (E2M1 + UE8M0 block-32) x MXFP4 -> BF16, CUTLASS 4.7.1 SM120 block-scaled "
                         "grouped GEMM (kind::mxf4nvf4), built for sm_120a",
                         "avg_ms includes gemm.initialize() per iteration as in the CUTLASS example unless reinit_in_loop=false",
                         "weight_GBps = MXFP4 weight+scale bytes / avg time; decode is weight-bandwidth bound",
                         "prefill_dense_proxy = 6 groups x M=4096 (all tokens to one expert), an upper bound"],
               "cases": {}}
    for case in cases(args.skip_prefill, args.tp, args.decode_groups):
        verify = args.verify_decode and case["phase"] == "decode"
        print(f"== {case['name']}  groups={len(case['groups'])} M,N,K={case['groups'][0]} verify={verify}",
              flush=True)
        r = run_case(args.binary, case, args.iterations, verify, args.no_reinit)
        r.update({"phase": case["phase"], "proj": case["proj"], "num_groups": len(case["groups"]),
                  "mnk": case["groups"][0]})
        results["cases"][case["name"]] = r
        for sname, s in r.get("schedules", {}).items():
            if "avg_ms" in s:
                print(f"   {sname:12s} {s['avg_ms']*1000:9.1f} us  {s['tflops']:8.1f} TFLOPS  "
                      f"{s['weight_GBps']:7.0f} GB/s weights  verified={s.get('verified')}", flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

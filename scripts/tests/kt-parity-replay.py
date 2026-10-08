#!/usr/bin/env python3
"""T3.16e(3) / T3.19b: replay recorded decode steps through kt-kernel and compare bit-for-bit.

The timeline dumps (KT_TIMELINE=1, KT_TIMELINE_DUMP_LAYERS=13,14,19,20) record, for a
sampled decode step, each dumped MoE layer's inputs as kt saw them (hidden_states bf16,
global topk_ids, topk_weights fp32) and kt's output for that layer. With D=4 a single
layer cannot be replayed on its own: kt_output[L] = immediate(L) + deferred(L-1), so the
harness replays the adjacent pair (L-1, L) from the same step and checks that kt's
standalone output for L equals the recorded kt_output[L] exactly (torch.equal on bf16).
What that proves: the serving path (graph-captured, through the provider seam, with
the compact CPU view and the overlay's deferral selector) asks kt for exactly the work
a standalone kt wrapper does on the same inputs; what it does not prove: that kt's
arithmetic is right (that is scripts/test-kt-one-layer.py's dequantized reference).

Modes:
  run    GPU + model, inside the recipe image; loads the two layers' compact CPU experts
         (158 each), replays every (key, n) pair found in --dumps, writes
         <out>/replay-<key>-layer<L>-<n>.pt (the standalone kt output) and <out>/parity.json.
  check  CPU-only; compares every replay-*.pt against the recorded dump's kt_output and
         the recorded parity.json; exit 0 = every pair bit-equal, 1 = mismatch,
         3 = no evidence present (ci.sh treats 3 as skipped).

run (inside the image; GPUs must be free of the server, kt needs the host cores):
  docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v "$PWD/scripts:/recipe:ro" -v /tmp/kt-timeline:/dumps:ro -v "$PWD/evidence/t319/b/parity:/out" \
    -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python dsv41-kt:recipe \
    /recipe/tests/kt-parity-replay.py run --weight-path /hfm/snapshots/<sha> --dumps /dumps --out /out
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

PAIRS = ((13, 14), (19, 20))
DUMP_RE = re.compile(r"dump-rank0-(?P<key>.+)-layer(?P<layer>\d+)-(?P<n>\d+)\.pt$")


def find_pairs(dumps: Path):
    """-> [(key, n, L, path_prev, path_cur)] for every dump of L whose L-1 twin exists."""
    out = []
    for p in sorted(dumps.glob("dump-rank0-*-layer*-*.pt")):
        m = DUMP_RE.search(p.name)
        if not m:
            continue
        L = int(m["layer"])
        if L not in {b for _, b in PAIRS}:
            continue
        prev = dumps / f"dump-rank0-{m['key']}-layer{L - 1}-{m['n']}.pt"
        if prev.exists():
            out.append((m["key"], int(m["n"]), L, prev, p))
    return out


def load(p: Path) -> dict:
    return torch.load(p, map_location="cpu", weights_only=False)


# ----------------------------------------------------------------------------- run

def make_wrapper(ktw, KTMoEWrapper, args, layer: int, d: int):
    import types

    w = KTMoEWrapper(
        layer_idx=layer,
        num_experts=args.num_experts - args.num_gpu_experts,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        gpu_experts_mask=None,
        cpuinfer_threads=args.threads,
        threadpool_count=1,
        weight_path=args.weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        method="MXFP4",
        max_deferred_experts_per_token=d,
        swiglu_limit=args.swiglu_limit,
    )
    if d > 0:
        w.select_deferred_experts = types.MethodType(ktw._select_deferred_experts_compact, w)
    return w


def run(args) -> int:
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang.srt.layers.moe.kt_ep_wrapper as ktw
    from kt_kernel import KTMoEWrapper
    from kt_kernel.experts_base import BaseMoEWrapper

    dumps, out = Path(args.dumps), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(dumps)
    if not pairs:
        print(f"no (L-1, L) dump pairs under {dumps}", file=sys.stderr)
        return 3
    dev = torch.device("cuda:0")
    n_cpu = args.num_experts - args.num_gpu_experts
    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    gpu_mask[: args.num_gpu_experts] = True
    from sglang.srt.layers.moe.kt_runtime import KtRuntime  # compact CPU-expert load, as the server does (T3.19c)

    runtime = KtRuntime(tp_rank=0)
    wrappers = {}
    for a, b in PAIRS:
        if not any(L == b for _, _, L, _, _ in pairs):
            continue
        for layer in (a, b):
            d = 0 if layer == args.num_layers - 1 else args.max_deferred
            t0 = time.time()
            w = make_wrapper(ktw, KTMoEWrapper, args, layer, d)
            runtime.load_cpu_experts(w, torch.arange(n_cpu), gpu_mask)
            wrappers[layer] = w
            print(f"loaded layer {layer} ({n_cpu} CPU experts, D={d}) in {time.time() - t0:.1f}s")

    stream = torch.cuda.current_stream(dev).cuda_stream
    results = []
    for key, n, L, p_prev, p_cur in pairs:
        prev, cur = load(p_prev), load(p_cur)
        BaseMoEWrapper._layer_has_pending_deferred.clear()
        outputs = {}
        steps = ((L, cur),) if args.control else ((L - 1, prev), (L, cur))
        for layer, d in steps:
            t = d["tensors"]
            x = t["hidden_states"].to(dev)
            ids = ktw.remap_topk_ids_for_cpu(t["topk_ids"].to(dev), args.num_gpu_experts, n_cpu)
            wts = t["topk_weights"].to(dev)
            w = wrappers[layer]
            w.submit_forward(x, ids, wts, stream)
            outputs[layer] = w.sync_forward(x, stream).clone()
        torch.cuda.synchronize()
        got = outputs[L].to("cpu")
        want = cur["tensors"]["kt_output"]
        equal = bool(torch.equal(got, want))
        diff = (got.float() - want.float()).abs()
        rec = {
            "key": key,
            "n": n,
            "layer": L,
            "tokens": int(got.shape[0]),
            "bit_equal": equal,
            "max_abs_diff": float(diff.max()),
            "mismatching_elements": int((diff > 0).sum()),
            "cpu_slots": int((ktw.remap_topk_ids_for_cpu(cur["tensors"]["topk_ids"], args.num_gpu_experts, n_cpu) >= 0).sum()),
            "prev_cpu_slots": int((ktw.remap_topk_ids_for_cpu(prev["tensors"]["topk_ids"], args.num_gpu_experts, n_cpu) >= 0).sum()),
            "replay_file": f"replay-{key}-layer{L}-{n}.pt",
            "dump_file": p_cur.name,
        }
        if not args.control:
            torch.save({"kt_output": got, "layer_idx": L, "key": key, "n": n}, out / rec["replay_file"])
        results.append(rec)
        print(f"{'ok  ' if equal else 'FAIL'} {key} n={n} layer {L}: tokens={rec['tokens']} "
              f"max|diff|={rec['max_abs_diff']:.3g} mismatching={rec['mismatching_elements']}")
    report = {
        "harness": "scripts/tests/kt-parity-replay.py",
        "mode": "control (layer L alone, L-1's deferred experts missing; expected NOT bit-equal)" if args.control else "run",
        "pairs": [list(p) for p in PAIRS],
        "config": {k: getattr(args, k) for k in ("num_experts", "num_gpu_experts", "topk", "hidden", "intermediate",
                                                  "threads", "chunked_prefill_size", "max_deferred", "swiglu_limit", "num_layers")},
        "results": results,
        "all_bit_equal": all(r["bit_equal"] for r in results),
    }
    name = "control.json" if args.control else "parity.json"
    (out / name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"{sum(r['bit_equal'] for r in results)}/{len(results)} pairs bit-equal -> {out / name}")
    if args.control:
        return 0
    return 0 if report["all_bit_equal"] else 1


# --------------------------------------------------------------------------- check

def check(args) -> int:
    out, dumps = Path(args.out), Path(args.dumps)
    pj = out / "parity.json"
    if not pj.exists():
        print(f"kt parity check: {pj} absent; skipped", file=sys.stderr)
        return 3
    report = json.loads(pj.read_text())
    failed = 0
    for rec in report["results"]:
        replay = out / rec["replay_file"]
        dump = dumps / rec["dump_file"]
        if not replay.exists() or not dump.exists():
            print(f"FAIL {rec['key']} n={rec['n']} layer {rec['layer']}: missing {replay.name if not replay.exists() else dump.name}")
            failed += 1
            continue
        got = load(replay)["kt_output"]
        want = load(dump)["tensors"]["kt_output"]
        equal = bool(torch.equal(got, want))
        if not equal or not rec["bit_equal"]:
            failed += 1
        print(f"{'ok  ' if equal else 'FAIL'} {rec['key']} n={rec['n']} layer {rec['layer']}: "
              f"tokens={got.shape[0]} bit_equal={equal} (recorded {rec['bit_equal']})")
    n = len(report["results"])
    print(f"{n - failed}/{n} pairs bit-equal")
    return 1 if failed or n == 0 else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "check"))
    ap.add_argument("--dumps", default="evidence/t319/b/dumps")
    ap.add_argument("--out", default="evidence/t319/b/parity")
    ap.add_argument("--weight-path")
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--num-gpu-experts", type=int, default=226)
    ap.add_argument("--num-layers", type=int, default=40)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--chunked-prefill-size", type=int, default=4096)
    ap.add_argument("--max-deferred", type=int, default=4)
    ap.add_argument("--control", action="store_true",
                    help="run only: replay layer L without L-1 (negative control; writes control.json, never replay-*.pt)")
    args = ap.parse_args()
    if args.mode == "run":
        if not args.weight_path:
            ap.error("run needs --weight-path")
        return run(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())

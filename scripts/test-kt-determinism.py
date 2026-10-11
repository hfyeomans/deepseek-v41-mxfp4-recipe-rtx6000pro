#!/usr/bin/env python3
"""T3.15b: is kt-kernel's CPU expert tier bit-reproducible run to run?

The hybrid's greedy bs=1 decode drifts between runs of the same server (E21
result 7, T3.15: 320-390 characters at D=0). The CPU tier is one candidate: a
thread-pool reduction whose order depends on scheduling would make the same
(x, routing) give different bits each call. This script loads one MoE layer
through the same wrapper the server uses (test-kt-one-layer.py), then pushes
one fixed input through it N times at decode-like and small-batch shapes and
compares every result bitwise with the first. It also runs the same input under
a different thread count, which tells partition dependence (deterministic, but
a function of the thread count) apart from non-determinism.

Run inside the recipe image exactly like test-kt-one-layer.py:

  docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v "$PWD/scripts:/recipe:ro" --entrypoint python dsv41-kt:recipe \
    /recipe/test-kt-determinism.py --weight-path /hfm/snapshots/<sha> --json /tmp/out.json

Exit 0 when every repeat is bit-identical to the first for every shape and
thread count; 1 otherwise. The report names the first differing element.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def build_wrapper(args, threads: int, max_deferred: int):
    from kt_kernel import KTMoEWrapper
    from sglang.srt.layers.moe.kt_runtime import KtRuntime

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    gpu_mask[: args.num_gpu_experts] = True
    n_cpu = int((~gpu_mask).sum())
    wrapper = KTMoEWrapper(
        layer_idx=args.layer,
        num_experts=n_cpu,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        gpu_experts_mask=None,
        cpuinfer_threads=threads,
        threadpool_count=1,
        weight_path=args.weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        method="MXFP4",
        max_deferred_experts_per_token=max_deferred,
        swiglu_limit=args.swiglu_limit,
    )
    KtRuntime(tp_rank=0).load_cpu_experts(wrapper, torch.arange(n_cpu), gpu_mask)
    return wrapper, gpu_mask, n_cpu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-path", required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--num-gpu-experts", type=int, default=219)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--threads", default="16,8", help="comma list; the first is the reference")
    ap.add_argument("--chunked-prefill-size", type=int, default=256)
    ap.add_argument("--tokens", default="1,2,4,8,64")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--cpu-slots-per-token", type=int, default=3,
                    help="slots routed to CPU experts per token (the served mix is ~3 of 6)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang.srt.layers.moe.kt_ep_wrapper as ktw

    dev = torch.device("cuda:0")
    stream = torch.cuda.current_stream(dev).cuda_stream
    thread_counts = [int(t) for t in args.threads.split(",") if t]
    shapes = [int(t) for t in args.tokens.split(",") if t]
    report: dict = {"args": vars(args), "torch": torch.__version__, "runs": {}}

    # Fixed inputs, generated once so every thread count sees the same bits.
    torch.manual_seed(args.seed)
    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    gpu_mask[: args.num_gpu_experts] = True
    cpu_ids = torch.nonzero(~gpu_mask).flatten()
    n_cpu = len(cpu_ids)
    inputs = {}
    for T in shapes:
        x = torch.randn(T, args.hidden).to(torch.bfloat16)
        ids = torch.stack([cpu_ids[torch.randperm(n_cpu)[: args.topk]] for _ in range(T)])
        if args.cpu_slots_per_token < args.topk:
            # Remaining slots go to GPU ids, which contribute nothing on the CPU tier.
            ids[:, args.cpu_slots_per_token:] = torch.randint(
                0, args.num_gpu_experts, (T, args.topk - args.cpu_slots_per_token)
            )
        w = torch.rand(T, args.topk) + 0.1
        w = w / w.sum(dim=1, keepdim=True)
        inputs[T] = (x.to(dev), ids.to(torch.int64).to(dev), w.to(torch.float32).to(dev))

    reference: dict[int, torch.Tensor] = {}
    all_identical = True
    for ti, threads in enumerate(thread_counts):
        wrapper, _, _ = build_wrapper(args, threads, 0)
        for T in shapes:
            x, ids, w = inputs[T]
            kt_ids = ktw.remap_topk_ids_for_cpu(ids, args.num_gpu_experts, n_cpu)
            outs = []
            for _ in range(args.repeats):
                out = wrapper.forward(x, kt_ids, w, stream)
                torch.cuda.synchronize()
                outs.append(out.detach().clone().cpu())
            first = outs[0]
            diffs = [int((o.view(torch.int16) != first.view(torch.int16)).sum()) for o in outs[1:]]
            max_abs = max(float((o.float() - first.float()).abs().max()) for o in outs[1:])
            entry = {
                "threads": threads,
                "tokens": T,
                "repeats": args.repeats,
                "repeats_differing": sum(1 for d in diffs if d),
                "elements_differing_max": max(diffs) if diffs else 0,
                "max_abs_diff_vs_first": max_abs,
                "out_finite": bool(torch.isfinite(first.float()).all()),
            }
            if ti == 0:
                reference[T] = first
            else:
                ref = reference[T]
                n_diff = int((first.view(torch.int16) != ref.view(torch.int16)).sum())
                entry["vs_reference_threads"] = {
                    "threads": thread_counts[0],
                    "elements_differing": n_diff,
                    "max_abs_diff": float((first.float() - ref.float()).abs().max()),
                }
            if entry["repeats_differing"]:
                all_identical = False
                o = next(o for o in outs[1:] if (o.view(torch.int16) != first.view(torch.int16)).any())
                idx = torch.nonzero(o.view(torch.int16) != first.view(torch.int16))[0].tolist()
                entry["first_diff"] = {
                    "index": idx,
                    "first": float(first[tuple(idx)]),
                    "other": float(o[tuple(idx)]),
                }
            report["runs"][f"t{threads}-T{T}"] = entry
            tail = ""
            if "vs_reference_threads" in entry:
                v = entry["vs_reference_threads"]
                tail = f"  vs {v['threads']} threads: {v['elements_differing']} elems differ, max|d| {v['max_abs_diff']:.3g}"
            print(
                f"threads={threads:2d} T={T:3d}: {entry['repeats_differing']}/{args.repeats - 1} repeats differ "
                f"(max {entry['elements_differing_max']} elems, max|d| {max_abs:.3g}){tail}"
            )
        del wrapper

    report["all_repeats_identical"] = all_identical
    print("RESULT:", "bit-identical across repeats" if all_identical else "NOT reproducible")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    return 0 if all_identical else 1


if __name__ == "__main__":
    sys.exit(main())

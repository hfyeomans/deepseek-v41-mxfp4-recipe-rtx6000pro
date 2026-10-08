#!/usr/bin/env python3
"""T3.6b: one MoE layer of DeepSeek-V4.1-Flash through kt-kernel's MXFP4 CPU path.

Checks the CPU tier of the hybrid recipe in isolation, before any GPU serving:

  1. kt-kernel 0.7.0.post4 accepts the layer with a gpu_experts_mask and the
     V4.1 swiglu_limit, loads the MXFP4 experts straight from the HF snapshot
     (no re-quantization) and runs them.
  2. Its output matches an independent dequantized reference (E2M1 grid x
     2^(ue8m0-127), SwiGLU with the V4.1 clamp, top-k weighted sum) for
     routings that hit CPU experts only, and ignores GPU-expert ids.
  3. CPU time per layer at decode-like and prefill-like token counts.
  4. Optional: RSS with and without the overlay's loader GPU-skip patch.

Run inside the recipe image with the overlay and HF cache mounted:

  docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v "$PWD/scripts:/recipe:ro" --entrypoint python dsv41-kt:recipe \
    /recipe/test-kt-one-layer.py --weight-path /hfm/snapshots/<sha> --json /tmp/out.json

Writes a JSON report (numbers only; no weights) for evidence/.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def rss_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


def anon_gib() -> float:
    """Anonymous (non file-backed) resident memory: what the scale copies cost."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("RssAnon:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


class Snapshot:
    """Read individual expert tensors from the HF safetensors shards."""

    def __init__(self, root: Path):
        from safetensors import safe_open

        self.root = root
        idx = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map = idx["weight_map"]
        self._safe_open = safe_open
        self._handles: dict[str, object] = {}

    def get(self, key: str) -> torch.Tensor:
        fn = self.weight_map[key]
        h = self._handles.get(fn)
        if h is None:
            h = self._safe_open(str(self.root / fn), framework="pt", device="cpu")
            self._handles[fn] = h
        return h.get_tensor(key)


def dequant_mxfp4(packed: torch.Tensor, scale_u8: torch.Tensor, dev) -> torch.Tensor:
    """[N, K/2] uint8 nibbles (low nibble first) + [N, K/32] ue8m0 -> float32 [N, K]."""
    packed = packed.view(torch.uint8).to(dev)  # checkpoint stores int8; avoid arithmetic >>
    n, half_k = packed.shape
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(n, half_k * 2)  # k order: lo, hi
    mag = E2M1_GRID.to(dev)[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    vals = mag * sign
    # safetensors yields F8_E8M0 as torch.float8_e8m0fnu; reinterpret the bits
    # (a value cast would turn 2^(e-127) into an integer and lose it).
    exp = scale_u8.view(torch.uint8).to(dev).to(torch.int32) - 127
    scales = torch.pow(2.0, exp.to(torch.float32))  # [N, K/32]
    scales = scales.repeat_interleave(32, dim=1)
    return vals * scales


def reference_layer(
    snap: Snapshot,
    layer: int,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    cpu_expert_mask: torch.Tensor,
    swiglu_limit: float,
    dev,
) -> torch.Tensor:
    """CPU-tier contribution only: sum_k w_k * expert_k(x) for k routed to CPU."""
    T = x.shape[0]
    out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=dev)
    xf = x.to(dev).float()
    topk_ids = topk_ids.to(dev)
    topk_w = topk_w.to(dev)
    cpu_expert_mask = cpu_expert_mask.to(dev)
    needed = sorted(set(topk_ids.flatten().tolist()) & set(
        torch.nonzero(cpu_expert_mask).flatten().tolist()
    ))
    for e in needed:
        base = f"layers.{layer}.ffn.experts.{e}"
        w1 = dequant_mxfp4(snap.get(f"{base}.w1.weight"), snap.get(f"{base}.w1.scale"), dev)
        w3 = dequant_mxfp4(snap.get(f"{base}.w3.weight"), snap.get(f"{base}.w3.scale"), dev)
        w2 = dequant_mxfp4(snap.get(f"{base}.w2.weight"), snap.get(f"{base}.w2.scale"), dev)
        rows, slots = torch.nonzero(topk_ids == e, as_tuple=True)
        xe = xf[rows]
        gate = xe @ w1.T
        up = xe @ w3.T
        if swiglu_limit > 0:
            # deep_gemm _apply_swiglu_limit / trtllm gemm1_clamp_limit semantics
            gate = gate.clamp(max=swiglu_limit)
            up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
        h = torch.nn.functional.silu(gate) * up
        y = h @ w2.T
        out.index_add_(0, rows, y * topk_w[rows, slots].to(dev).float().unsqueeze(1))
        del w1, w3, w2
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-path", required=True, help="HF snapshot dir")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--num-gpu-experts", type=int, default=224)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--threadpool-count", type=int, default=1)
    ap.add_argument("--chunked-prefill-size", type=int, default=256)
    ap.add_argument("--check-tokens", type=int, default=8)
    ap.add_argument("--bench-tokens", default="1,4,8,16,64,256")
    ap.add_argument("--bench-iters", type=int, default=20)
    ap.add_argument("--cpu-slots-per-token", type=int, default=None,
                    help="bench routing: how many of the top-k slots hit CPU experts "
                         "(rest get GPU ids kt must skip). Default: all top-k (worst case). "
                         "0 = every slot on GPU -> measures kt's fixed per-call floor.")
    ap.add_argument("--shared-routing", action="store_true",
                    help="bench routing: every token in the batch selects the same expert ids "
                         "(tests whether cost follows distinct experts or token*expert calls)")
    ap.add_argument("--bench-cold", action="store_true",
                    help="draw fresh expert ids every iteration so the weights are not L3-hot "
                         "(the 64 MiB L3 holds half of a T=1 working set otherwise)")
    ap.add_argument("--no-loader-patch", action="store_true", help="kt-kernel native model: load all experts, mask at run time")
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x-scale", type=float, default=1.0,
                    help="scale random inputs; >~8 drives gate/up past the clamp")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = torch.device("cuda:0")
    report: dict = {"args": vars(args), "kt_kernel": None, "torch": torch.__version__}

    import kt_kernel
    from kt_kernel import KTMoEWrapper

    report["kt_kernel"] = getattr(kt_kernel, "__version__", "unknown")

    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    gpu_mask[: args.num_gpu_experts] = True
    cpu_mask = ~gpu_mask

    n_cpu = int(cpu_mask.sum())
    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang.srt.layers.moe.kt_ep_wrapper as ktw

    from sglang.srt.layers.moe.kt_runtime import KtRuntime

    compact = not args.no_loader_patch
    report["compact_cpu_view"] = compact
    if compact:
        # Same path the serving wrapper takes (T3.19c): KtRuntime.load_cpu_experts
        # hands kt a CompactMXFP4Loader for this layer's mask, so kt-kernel sees an
        # n_cpu-expert MoE and ids are remapped; it raises if kt did not use it.
        kt_num_experts, kt_mask = n_cpu, None
    else:
        # kt-kernel's native memory model: all 384 experts loaded, mask skips
        # GPU experts at run time only.
        kt_num_experts, kt_mask = args.num_experts, gpu_mask

    rss0, anon0 = rss_gib(), anon_gib()
    t0 = time.time()
    wrapper = KTMoEWrapper(
        layer_idx=args.layer,
        num_experts=kt_num_experts,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        gpu_experts_mask=kt_mask,
        cpuinfer_threads=args.threads,
        threadpool_count=args.threadpool_count,
        weight_path=args.weight_path,
        chunked_prefill_size=args.chunked_prefill_size,
        method="MXFP4",
        max_deferred_experts_per_token=0,
        swiglu_limit=args.swiglu_limit,
    )
    if compact:
        KtRuntime(tp_rank=0).load_cpu_experts(wrapper, torch.arange(n_cpu), gpu_mask)
    else:
        wrapper.load_weights(torch.arange(kt_num_experts))
    t_load = time.time() - t0
    rss1, anon1 = rss_gib(), anon_gib()
    report["load"] = {
        "seconds": round(t_load, 2),
        "rss_gib_before": round(rss0, 3),
        "rss_gib_after": round(rss1, 3),
        "anon_gib_before": round(anon0, 3),
        "anon_gib_after": round(anon1, 3),
        "anon_gib_delta": round(anon1 - anon0, 3),
    }
    report["load"]["cpu_experts"] = n_cpu
    report["load"]["anon_mib_per_cpu_expert"] = round((anon1 - anon0) * 1024 / n_cpu, 2)
    print(
        f"load: {t_load:.1f}s  RSS {rss0:.2f}->{rss1:.2f} GiB  anon {anon0:.2f}->{anon1:.2f} GiB  "
        f"({(anon1 - anon0) * 1024 / n_cpu:.1f} MiB per CPU expert; packed FP4+scale "
        f"in checkpoint = {3 * 2304 * 5120 * (0.5 + 1 / 32) / 2**20:.1f} MiB)"
    )
    def to_kt_ids(ids: torch.Tensor) -> torch.Tensor:
        if compact:
            return ktw.remap_topk_ids_for_cpu(ids, args.num_gpu_experts, n_cpu)
        return ids

    if args.bench_tokens == "" and args.skip_reference:
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
        return 0

    stream = torch.cuda.current_stream(dev).cuda_stream
    cpu_ids = torch.nonzero(cpu_mask).flatten()

    def random_routing(T: int, gpu_frac_tokens: int = 0):
        """top-k distinct CPU expert ids per token; first gpu_frac_tokens tokens
        get one slot replaced by a GPU id (must contribute nothing on CPU)."""
        ids = torch.stack([cpu_ids[torch.randperm(len(cpu_ids))[: args.topk]] for _ in range(T)])
        if gpu_frac_tokens:
            ids[:gpu_frac_tokens, 0] = torch.randint(0, args.num_gpu_experts, (gpu_frac_tokens,))
        w = torch.rand(T, args.topk) + 0.1
        w = w / w.sum(dim=1, keepdim=True)
        return ids.to(torch.int64), w.to(torch.float32)

    # ---- correctness -----------------------------------------------------
    if not args.skip_reference:
        T = args.check_tokens
        x = (torch.randn(T, args.hidden) * args.x_scale).to(torch.bfloat16)
        ids, w = random_routing(T, gpu_frac_tokens=T // 2)
        x_g, ids_g, w_g = x.to(dev), ids.to(dev), w.to(dev)
        out = wrapper.forward(x_g, to_kt_ids(ids_g), w_g, stream)
        torch.cuda.synchronize()
        out = out.float()

        snap = Snapshot(Path(args.weight_path))
        ref = reference_layer(snap, args.layer, x, ids, w, cpu_mask, args.swiglu_limit, dev)
        # Same routing but clamp disabled in the reference -> shows the clamp
        # is actually applied by kt-kernel (must be WORSE than with clamp
        # whenever any activation exceeds the limit).
        ref_noclamp = reference_layer(snap, args.layer, x, ids, w, cpu_mask, 0.0, dev)

        err = (out - ref).abs()
        rel = err.norm() / ref.norm()
        err_nc = (out - ref_noclamp).abs()
        rel_nc = err_nc.norm() / ref_noclamp.norm()
        # tokens with a GPU id in slot 0: their CPU output must equal the
        # reference that already excludes that slot (weight on GPU id ignored)
        report["correctness"] = {
            "tokens": T,
            "tokens_with_gpu_id": T // 2,
            "max_abs_err": float(err.max()),
            "mean_abs_err": float(err.mean()),
            "rel_l2_err": float(rel),
            "ref_abs_mean": float(ref.abs().mean()),
            "ref_abs_max": float(ref.abs().max()),
            "rel_l2_err_vs_noclamp_ref": float(rel_nc),
            "out_finite": bool(torch.isfinite(out).all()),
        }
        print(
            f"correctness: max|err| {err.max():.4g}  mean|err| {err.mean():.4g}  "
            f"rel-L2 {rel:.4g}  (ref mean|y| {ref.abs().mean():.4g}, max {ref.abs().max():.4g}); "
            f"rel-L2 vs no-clamp ref {rel_nc:.4g}"
        )

    # ---- timing ----------------------------------------------------------
    bench = {}
    for T in [int(t) for t in args.bench_tokens.split(",") if t]:
        if T > args.chunked_prefill_size:
            continue
        x = torch.randn(T, args.hidden).to(torch.bfloat16).to(dev)
        n_sets = args.bench_iters if args.bench_cold else 1
        routing = []
        cpu_slots = args.topk if args.cpu_slots_per_token is None else args.cpu_slots_per_token
        for _ in range(n_sets):
            ids, w = random_routing(T)
            if args.shared_routing:
                ids = ids[:1].expand(T, -1).clone()
            if cpu_slots < args.topk:
                # slots [cpu_slots, topk) -> distinct GPU expert ids (kt skips them)
                ids[:, cpu_slots:] = torch.stack(
                    [torch.randperm(args.num_gpu_experts)[: args.topk - cpu_slots] for _ in range(T)]
                )
            routing.append((to_kt_ids(ids.to(dev)), w.to(dev)))
        for i in range(3):
            ids, w = routing[i % n_sets]
            wrapper.forward(x, ids, w, stream)
        torch.cuda.synchronize()
        times = []
        for i in range(args.bench_iters):
            ids, w = routing[i % n_sets]
            t = time.perf_counter()
            wrapper.forward(x, ids, w, stream)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1e3)
        times.sort()
        med = times[len(times) // 2]
        bench[str(T)] = {
            "ms_median": round(med, 3),
            "ms_min": round(times[0], 3),
            "ms_per_token": round(med / T, 4),
            "expert_calls_per_step": T * cpu_slots,
            "cpu_slots_per_token": cpu_slots,
            "cold_ids": bool(args.bench_cold),
        }
        print(f"bench T={T:4d}: median {med:8.3f} ms  min {times[0]:8.3f} ms  ({med / T:.3f} ms/token)")
    report["bench_ms_per_layer"] = bench
    report["note"] = (
        "Every token routes all top-k slots to CPU experts here (worst case); "
        "in the hybrid recipe only ~(1 - gpu_share) of slots hit CPU."
    )
    report["maxrss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

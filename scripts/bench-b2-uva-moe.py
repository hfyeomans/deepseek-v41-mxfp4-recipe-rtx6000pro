#!/usr/bin/env python3
"""T3.18 / B2 microbenchmark: can a GPU compute CPU-resident MXFP4 experts
straight out of pinned host memory over PCIe, fast enough to matter?

B2 (migration roadmap Stage 2 item 0) moves a fixed subset S of the CPU-tier
experts off the CCD path: their weights stay in host DRAM but are read by the
GPU over PCIe (E4b: PCIe DMA and CPU-core reads are ~98 % additive). The
fetch is data-dependent (only experts routed to in this step move), so it
must be issued by the GPU itself under CUDA graph capture. Two variants:

  V0  "direct UVA": hand FlashInfer's SM120 CUTLASS MXFP4 MoE weight tensors
      whose storage is pinned host memory (UVA device pointer). The grouped
      GEMM then reads only the routed experts' rows over PCIe. No new kernel.
  V1  "gather to staging": torch.index_select from the UVA tensors into a
      fixed device staging buffer (device-side indices, graph-capturable),
      then the same MoE kernel on the staging buffer.

This script measures, on one GPU and with no server running:
  1. raw PCIe read paths for expert-sized blobs: cudaMemcpyAsync (copy
     engine) vs a copy kernel reading UVA vs index_select gather from UVA
  2. the fused MoE at decode M (1..4 tokens, top-6) with weights on the
     device (reference), on UVA (V0), and gathered-then-computed (V1)
  3. bit-equality of V0/V1 outputs against the device reference
  4. CUDA-graph capture + replay of V0 and V1 (the production path replays
     a captured decode graph)

Weights are random bytes with unit (E8M0 = 127) block scales, in exactly the
layout the streamed-prefill path (E22, kt_ep_wrapper._StreamedExpertSlots)
leaves on the device: w13 [E, 2*inter_tp, hidden/2] int8, s13 [E, 2*inter_tp,
hidden/32] uint8 (interleaved), w2 [E, hidden, inter_tp/2], s2.

Run inside the recipe image (FlashInfer with the SM120 MXFP4 path):

  docker run --rm --gpus all --ipc=host -v "$PWD/scripts:/recipe:ro" \
    -v "$PWD/evidence:/ev" --entrypoint python dsv41-kt:recipe \
    /recipe/bench-b2-uva-moe.py --output /ev/t318-b2-prototype/uva-moe-gpu0.json

Add --device 1 (and run two containers at once) for the both-links case.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable, Dict, List

import torch

HIDDEN = 5120
INTER = 2304  # moe_intermediate_size; per TP rank INTER // tp_size


def uva_view(pinned: torch.Tensor, device: torch.device) -> torch.Tensor:
    """A CUDA tensor aliasing a pinned host tensor (zero-copy over PCIe).

    Uses the CUDA array interface so PyTorch wraps the device pointer without
    copying. On UVA systems (every 64-bit Linux host) cudaHostAlloc'd memory is
    mapped into the device address space, so the device pointer is the host
    pointer; we still ask the runtime rather than assume it.
    """
    assert pinned.is_pinned(), "host tensor must be pinned"
    nbytes = pinned.numel() * pinned.element_size()
    try:
        import cuda.bindings.runtime as rt

        err, dptr = rt.cudaHostGetDevicePointer(pinned.data_ptr(), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer: {err}")
        dptr = int(dptr)
    except ImportError:
        dptr = pinned.data_ptr()  # UVA: identical mapping

    class _CAI:
        __cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (dptr, False),
            "version": 3,
            "strides": None,
        }

    t = torch.as_tensor(_CAI(), device=device)
    # keep the pinned tensor alive as long as the view
    t._b2_host_owner = pinned  # type: ignore[attr-defined]
    return t.view(pinned.dtype).view(pinned.shape)


def time_cuda(fn: Callable[[], None], iters: int, warmup: int = 3) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: List[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return {
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))],
        "iters": iters,
    }


class Experts:
    """One layer's TP slice of S experts, on device + pinned host + UVA view."""

    def __init__(self, n: int, hidden: int, inter_tp: int, device: torch.device, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.n, self.hidden, self.inter_tp = n, hidden, inter_tp
        shapes = {
            "w13": ((n, 2 * inter_tp, hidden // 2), torch.int8),
            "s13": ((n, 2 * inter_tp, hidden // 32), torch.uint8),
            "w2": ((n, hidden, inter_tp // 2), torch.int8),
            "s2": ((n, hidden, inter_tp // 32), torch.uint8),
        }
        self.dev: Dict[str, torch.Tensor] = {}
        self.host: Dict[str, torch.Tensor] = {}
        self.uva: Dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in shapes.items():
            if dtype is torch.int8:
                # random FP4 nibbles; mask magnitude bits so activations stay finite
                h = torch.randint(0, 256, shape, generator=g, dtype=torch.int32)
                h = (h & 0x33).to(torch.uint8).view(torch.int8)  # |x| <= 1.5 per nibble
            else:
                h = torch.full(shape, 127, dtype=torch.uint8)  # E8M0 exponent 0 -> 1.0
            self.host[name] = h.pin_memory()
            self.dev[name] = self.host[name].to(device, non_blocking=True)
            self.uva[name] = uva_view(self.host[name], device)
        torch.cuda.synchronize()

    def expert_bytes(self) -> int:
        return sum(t[0].numel() * t.element_size() for t in self.dev.values())


def fused_moe(
    x_bf16: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_w: torch.Tensor,
    w: Dict[str, torch.Tensor],
    n_experts: int,
    tp_size: int,
    swiglu_step: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    """Same call the recipe's SM120 MXFP4 path makes (moe_runner/flashinfer_cutlass.py)."""
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

    dev = x_bf16.device
    gscale = torch.ones(n_experts, dtype=torch.float32, device=dev)
    limit = torch.full((n_experts,), 7.0, dtype=torch.float32, device=dev)
    xq, x_sf = mxfp8_quantize(x_bf16, is_sf_swizzled_layout=True, alignment=32)
    cutlass_fused_moe(
        input=xq,
        token_selected_experts=topk_ids.to(torch.int32),
        token_final_scales=topk_w,
        fc1_expert_weights=w["w13"].view(torch.int64),
        fc2_expert_weights=w["w2"].view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[w["s13"].view(torch.int32), gscale, w["s2"].view(torch.int32), gscale],
        input_sf=x_sf,
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=limit,
        tp_size=tp_size,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.SwigluStep if swiglu_step else ActivationType.Swiglu,
        tune_max_num_tokens=max(1, 1 << (x_bf16.shape[0] - 1).bit_length()),
        output=out,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--experts", type=int, default=40, help="|S|: experts resident in host memory")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--gpu-experts", type=int, default=226, help="device-resident GPU-tier experts for the overlap test")
    ap.add_argument("--tokens", default="1,4", help="decode batch sizes M to test")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=318)
    ap.add_argument("--no-swiglu-step", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    inter_tp = INTER // args.tp_size
    ex = Experts(args.experts, HIDDEN, inter_tp, device, args.seed)
    eb = ex.expert_bytes()
    report: Dict[str, object] = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "experts_S": args.experts,
        "topk": args.topk,
        "tp_size": args.tp_size,
        "hidden": HIDDEN,
        "inter_tp": inter_tp,
        "bytes_per_expert_slice": eb,
        "mib_per_expert_slice": eb / 2**20,
        "iters": args.iters,
        "note": "random FP4 nibbles, unit E8M0 scales; timing + bit-equality only, not model numerics",
    }
    print(f"[b2] {report['gpu']} S={args.experts} slice={eb/2**20:.2f} MiB/expert (per TP rank)")

    # ---- 1. raw PCIe read paths --------------------------------------------
    raw: Dict[str, object] = {}
    for k in (1, args.topk):
        stage = {n: torch.empty((k,) + tuple(t.shape[1:]), dtype=t.dtype, device=device) for n, t in ex.dev.items()}
        idx = torch.randperm(args.experts, device=device)[:k]
        nbytes = k * eb

        def memcpy() -> None:  # copy engine, host-known addresses (not data-dependent)
            for n in stage:
                stage[n].copy_(ex.host[n][:k], non_blocking=True)

        def copy_kernel() -> None:  # SM copy kernel reading UVA, contiguous
            for n in stage:
                stage[n].copy_(ex.uva[n][:k])

        def gather() -> None:  # data-dependent: device-side indices
            for n in stage:
                torch.index_select(ex.uva[n], 0, idx, out=stage[n])

        def gather_dev() -> None:  # reference: same gather from device memory
            for n in stage:
                torch.index_select(ex.dev[n], 0, idx, out=stage[n])

        for name, fn in (("memcpy_async", memcpy), ("copy_kernel_uva", copy_kernel), ("index_select_uva", gather), ("index_select_device", gather_dev)):
            t = time_cuda(fn, args.iters)
            t["gib_s"] = nbytes / 2**30 / (t["median_ms"] / 1e3)
            t["bytes"] = nbytes
            raw[f"{name}_k{k}"] = t
            print(f"[b2] raw {name:20s} k={k}: {t['median_ms']:.3f} ms median -> {t['gib_s']:.1f} GiB/s")
    report["raw_pcie"] = raw

    # ---- 2./3./4. fused MoE variants ---------------------------------------
    moe: Dict[str, object] = {}
    g = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    for m in [int(s) for s in args.tokens.split(",")]:
        x = (torch.randn((m, HIDDEN), generator=g) * 0.5).to(torch.bfloat16).to(device)
        # distinct experts per token; across tokens may repeat (as real routing does)
        ids = torch.stack([torch.randperm(args.experts, generator=g)[: args.topk] for _ in range(m)]).to(device)
        w = torch.softmax(torch.randn((m, args.topk), generator=g), dim=-1).to(device)
        n_hits = int(torch.unique(ids).numel())
        out_ref = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
        out_v0 = torch.empty_like(out_ref)
        out_v1 = torch.empty_like(out_ref)
        step = not args.no_swiglu_step

        def ref() -> None:
            fused_moe(x, ids, w, ex.dev, args.experts, args.tp_size, step, out_ref)

        def v0() -> None:
            fused_moe(x, ids, w, ex.uva, args.experts, args.tp_size, step, out_v0)

        # V1: gather hit experts into staging slots; slot ids are fixed per (token, k)
        # (duplicates gathered twice -- what a first prototype without dedupe does).
        slots = m * args.topk
        stage = {n: torch.empty((slots,) + tuple(t.shape[1:]), dtype=t.dtype, device=device) for n, t in ex.dev.items()}
        flat_ids = ids.reshape(-1)
        slot_ids = torch.arange(slots, device=device, dtype=ids.dtype).view(m, args.topk)

        def v1() -> None:
            for n in stage:
                torch.index_select(ex.uva[n], 0, flat_ids, out=stage[n])
            fused_moe(x, slot_ids, w, stage, slots, args.tp_size, step, out_v1)

        r = {"M": m, "routes": slots, "distinct_hits": n_hits}
        r["device_ref"] = time_cuda(ref, args.iters)
        r["v0_direct_uva"] = time_cuda(v0, args.iters)
        r["v1_gather_uva"] = time_cuda(v1, args.iters)
        torch.cuda.synchronize()
        r["v0_bit_equal"] = bool(torch.equal(out_ref, out_v0))
        r["v1_bit_equal"] = bool(torch.equal(out_ref, out_v1))
        r["v1_max_abs_diff"] = float((out_ref.float() - out_v1.float()).abs().max())
        r["ref_finite"] = bool(torch.isfinite(out_ref).all())
        hit_bytes = n_hits * eb
        r["v0_effective_gib_s"] = hit_bytes / 2**30 / (r["v0_direct_uva"]["median_ms"] / 1e3)
        r["v1_effective_gib_s"] = slots * eb / 2**30 / (r["v1_gather_uva"]["median_ms"] / 1e3)

        # CUDA graph capture + replay of V0 and V1
        for name, fn, out_t in (("v0", v0, out_v0), ("v1", v1, out_v1)):
            try:
                s = torch.cuda.Stream()
                with torch.cuda.stream(s):
                    fn()  # warm the kernels / autotuner outside capture
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    fn()
                torch.cuda.synchronize()
                out_t.zero_()
                t = time_cuda(graph.replay, args.iters)
                torch.cuda.synchronize()
                t["bit_equal_after_replay"] = bool(torch.equal(out_ref, out_t))
                r[f"{name}_graph_replay"] = t
            except Exception as e:  # noqa: BLE001
                r[f"{name}_graph_replay"] = {"error": repr(e)[:400]}
        moe[f"M{m}"] = r
        print(
            f"[b2] M={m} routes={slots} distinct={n_hits}: ref {r['device_ref']['median_ms']:.3f} ms | "
            f"V0 direct-UVA {r['v0_direct_uva']['median_ms']:.3f} ms ({r['v0_effective_gib_s']:.1f} GiB/s, eq={r['v0_bit_equal']}) | "
            f"V1 gather {r['v1_gather_uva']['median_ms']:.3f} ms ({r['v1_effective_gib_s']:.1f} GiB/s, eq={r['v1_bit_equal']}) | "
            f"graph V0 {r['v0_graph_replay'].get('median_ms', 'ERR')} V1 {r['v1_graph_replay'].get('median_ms', 'ERR')}"
        )
    report["moe"] = moe

    # ---- 5. sparse hits: what the model's B2 path actually sends ------------
    # In the model, non-B2 slots carry -1 (as the retired overlay path remap_topk_ids_for_b2 did; removed at T3.16j, this script is the E30 record), so at
    # bs=1 a layer typically has 0-2 of its 6 slots inside S. Time V0 (direct UVA) and the
    # device reference for h valid slots out of top-k, rest -1, M in --tokens. The fixed cost
    # at h=0 is the kernel's floor for |S| problems with no tiles.
    sparse: Dict[str, object] = {}
    step = not args.no_swiglu_step
    hits_list = [h for h in (0, 1, 2, args.topk) if h <= args.topk]
    for m in [int(s) for s in args.tokens.split(",")]:
        x = (torch.randn((m, HIDDEN), generator=g) * 0.5).to(torch.bfloat16).to(device)
        for h in hits_list:
            ids = torch.full((m, args.topk), -1, dtype=torch.int64)
            for row in range(m):
                ids[row, :h] = torch.randperm(args.experts, generator=g)[:h]
            ids = ids.to(device)
            w = torch.softmax(torch.randn((m, args.topk), generator=g), dim=-1).to(device)
            n_hits = int(torch.unique(ids[ids >= 0]).numel())
            out_ref = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
            out_v0 = torch.empty_like(out_ref)

            def ref_h() -> None:
                fused_moe(x, ids, w, ex.dev, args.experts, args.tp_size, step, out_ref)

            def v0_h() -> None:
                fused_moe(x, ids, w, ex.uva, args.experts, args.tp_size, step, out_v0)

            r = {"M": m, "valid_slots_per_token": h, "distinct_hits": n_hits}
            r["device_ref"] = time_cuda(ref_h, args.iters)
            r["v0_direct_uva"] = time_cuda(v0_h, args.iters)
            torch.cuda.synchronize()
            r["v0_bit_equal"] = bool(torch.equal(out_ref, out_v0))
            r["ref_finite"] = bool(torch.isfinite(out_ref).all())
            r["v0_effective_gib_s"] = (n_hits * eb / 2**30 / (r["v0_direct_uva"]["median_ms"] / 1e3)) if n_hits else 0.0
            sparse[f"M{m}_h{h}"] = r
            print(
                f"[b2] sparse M={m} valid={h} distinct={n_hits}: ref {r['device_ref']['median_ms']:.3f} ms | "
                f"V0 direct-UVA {r['v0_direct_uva']['median_ms']:.3f} ms ({r['v0_effective_gib_s']:.1f} GiB/s, eq={r['v0_bit_equal']})"
            )
    report["sparse_hits"] = sparse

    # ---- 6. overlap: B2 on a side stream beside the GPU tier's own MoE -------
    # The server's per-layer stream order is GPU-tier MoE -> B2 -> wait for kt. If B2 sits on
    # the main stream its PCIe-bound time adds to the critical path; on a forked stream it can
    # overlap the GPU tier (and the kt wait). Measure serial vs two-stream, eager and graphed,
    # with a device-resident --gpu-experts tier at the same M and h B2 hits per token.
    overlap: Dict[str, object] = {}
    gpu_tier = Experts(args.gpu_experts, HIDDEN, inter_tp, device, args.seed + 7)
    side = torch.cuda.Stream()
    for m in [int(s) for s in args.tokens.split(",")]:
        x = (torch.randn((m, HIDDEN), generator=g) * 0.5).to(torch.bfloat16).to(device)
        ids_g = torch.stack([torch.randperm(args.gpu_experts, generator=g)[: args.topk] for _ in range(m)]).to(device)
        w = torch.softmax(torch.randn((m, args.topk), generator=g), dim=-1).to(device)
        out_g = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
        for h in [hh for hh in (1, 2, args.topk) if hh <= args.topk]:
            ids_b = torch.full((m, args.topk), -1, dtype=torch.int64)
            for row in range(m):
                ids_b[row, :h] = torch.randperm(args.experts, generator=g)[:h]
            ids_b = ids_b.to(device)
            out_b = torch.empty_like(out_g)

            def tier() -> None:
                fused_moe(x, ids_g, w, gpu_tier.dev, args.gpu_experts, args.tp_size, step, out_g)

            def b2() -> None:
                fused_moe(x, ids_b, w, ex.uva, args.experts, args.tp_size, step, out_b)

            def serial() -> None:
                tier()
                b2()

            def forked() -> None:
                main = torch.cuda.current_stream()
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    b2()
                tier()
                main.wait_stream(side)

            r = {"M": m, "b2_valid_slots": h, "gpu_experts": args.gpu_experts}
            for name, fn in (("tier_only", tier), ("b2_only", b2), ("sequential", serial), ("forked", forked)):
                r[name] = time_cuda(fn, args.iters)
                try:
                    s = torch.cuda.Stream()
                    with torch.cuda.stream(s):
                        fn()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=s):
                        fn()
                    torch.cuda.synchronize()
                    r[name + "_graph"] = time_cuda(graph.replay, args.iters)
                except Exception as e:  # noqa: BLE001
                    r[name + "_graph"] = {"error": repr(e)[:400]}
            overlap[f"M{m}_h{h}"] = r
            gm = lambda k: r[k].get("median_ms", float("nan"))  # noqa: E731
            print(
                f"[b2] overlap M={m} h={h}: tier {gm('tier_only'):.3f} | b2 {gm('b2_only'):.3f} | serial {gm('sequential'):.3f} | forked {gm('forked'):.3f} ms"
                f"  (graph: tier {gm('tier_only_graph'):.3f} | b2 {gm('b2_only_graph'):.3f} | serial {gm('sequential_graph'):.3f} | forked {gm('forked_graph'):.3f})"
            )
    report["overlap"] = overlap
    report["wall_s"] = time.time()
    with open(args.output, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[b2] wrote {args.output}")


if __name__ == "__main__":
    main()

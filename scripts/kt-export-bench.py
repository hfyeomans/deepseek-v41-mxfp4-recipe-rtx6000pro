#!/usr/bin/env python3
"""E22a: how fast can kt-kernel hand its CPU-resident MXFP4 experts to the GPUs?

Layerwise GPU prefill (T3.6b2) re-exports every CPU expert of a layer once per
prefill chunk: kt-kernel's `write_weight_scale_to_buffer` copies the expert's
packed FP4 bytes and bf16 scales into pinned host buffers (one TP slice per
GPU rank), each rank DMAs its slice, and the GPU fixes the layout up for
FlashInfer's SM120 CUTLASS MoE (gate/up half swap, bf16 -> E8M0 shift,
`block_scale_interleave`). This script measures each stage on one real layer
of DeepSeek-V4.1-Flash, with no server running:

  1. byte-compare the export against the safetensors shards (both rank slices)
  2. export throughput: all experts, one sync; and grouped (sync per group)
  3. GPU prep: per-expert swap-copies to both cards concurrently + scale fix-up
  4. the pipelined per-layer cost: write(group g+1) while H2D(group g)

Run inside the recipe image (kt-kernel 0.7.0.post4, FlashInfer):

  docker run --rm --gpus all --ipc=host \
    -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \
    -v "$PWD/scripts:/recipe:ro" --entrypoint python dsv41-kt:recipe \
    /recipe/kt-export-bench.py --weight-path /hfm/snapshots/<sha> --json /tmp/out.json

Writes a JSON report (numbers only) for evidence/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def gib(nbytes: int) -> float:
    return nbytes / 2**30


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


class HostSlots:
    """Pinned host buffers for `slots` experts of one TP rank's slice.

    Layout is what kt-kernel writes: w13 = [gate rows; up rows] per expert,
    scales as bf16 (value 2^(e-127), bits = e8m0 << 7).
    """

    def __init__(self, slots: int, hidden: int, inter_tp: int):
        self.slots = slots
        self.w13 = torch.empty((slots, 2 * inter_tp, hidden // 2), dtype=torch.uint8, pin_memory=True)
        self.s13 = torch.empty((slots, 2 * inter_tp, hidden // 32), dtype=torch.bfloat16, pin_memory=True)
        self.w2 = torch.empty((slots, hidden, inter_tp // 2), dtype=torch.uint8, pin_memory=True)
        self.s2 = torch.empty((slots, hidden, inter_tp // 32), dtype=torch.bfloat16, pin_memory=True)
        self.tensors = (self.w13, self.s13, self.w2, self.s2)

    def ptrs(self, slot: int) -> tuple[int, int, int, int]:
        return tuple(int(t[slot].data_ptr()) for t in self.tensors)

    @property
    def bytes_per_expert(self) -> int:
        return sum(t[0].numel() * t[0].element_size() for t in self.tensors)


class DeviceLayer:
    """One layer's worth of streamed experts on one GPU, in SGLang's layout."""

    def __init__(self, num_experts: int, hidden: int, inter_tp: int, device: torch.device):
        self.device = device
        self.inter_tp = inter_tp
        self.w13 = torch.empty((num_experts, 2 * inter_tp, hidden // 2), dtype=torch.int8, device=device)
        self.s13 = torch.empty((num_experts, 2 * inter_tp, hidden // 32), dtype=torch.uint8, device=device)
        self.w2 = torch.empty((num_experts, hidden, inter_tp // 2), dtype=torch.int8, device=device)
        self.s2 = torch.empty((num_experts, hidden, inter_tp // 32), dtype=torch.uint8, device=device)
        # bf16 scale scratch; the E8M0 shift + interleave read from here.
        self.s13_raw = torch.empty((num_experts, 2 * inter_tp, hidden // 32), dtype=torch.bfloat16, device=device)
        self.s2_raw = torch.empty((num_experts, hidden, inter_tp // 32), dtype=torch.bfloat16, device=device)
        self.stream = torch.cuda.Stream(device=device)

    def copy_group(self, host: HostSlots, host_slots: range, dst_experts: range) -> None:
        """H2D for a group of experts, swapping the gate/up halves of w13."""
        it = self.inter_tp
        with torch.cuda.stream(self.stream):
            for hs, de in zip(host_slots, dst_experts):
                # host [gate; up]  ->  device [up; gate]
                self.w13[de, :it].copy_(host.w13[hs, it:].view(torch.int8), non_blocking=True)
                self.w13[de, it:].copy_(host.w13[hs, :it].view(torch.int8), non_blocking=True)
                self.s13_raw[de, :it].copy_(host.s13[hs, it:], non_blocking=True)
                self.s13_raw[de, it:].copy_(host.s13[hs, :it], non_blocking=True)
                self.w2[de].copy_(host.w2[hs].view(torch.int8), non_blocking=True)
                self.s2_raw[de].copy_(host.s2[hs], non_blocking=True)

    def finish_scales(self) -> None:
        """bf16 (e8m0 << 7) -> uint8 E8M0, then FlashInfer's SM120 interleave.

        Mirrors Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading,
        which interleaves the whole [E, N, K/32] tensor in place.
        """
        from flashinfer import block_scale_interleave

        with torch.cuda.stream(self.stream):
            for raw, dst in ((self.s13_raw, self.s13), (self.s2_raw, self.s2)):
                dst.copy_((raw.view(torch.int16).to(torch.int32) >> 7).to(torch.uint8))
                dst.copy_(block_scale_interleave(dst).reshape_as(dst))


def expected_slices(snap: Snapshot, layer: int, expert: int, rank: int, tp: int, inter: int):
    """What kt-kernel must have written for (expert, rank): checkpoint bytes, sliced."""
    base = f"layers.{layer}.ffn.experts.{expert}"
    it = inter // tp
    rows = slice(rank * it, (rank + 1) * it)
    gate_w = snap.get(f"{base}.w1.weight").view(torch.uint8)[rows]
    up_w = snap.get(f"{base}.w3.weight").view(torch.uint8)[rows]
    gate_s = snap.get(f"{base}.w1.scale").view(torch.uint8)[rows]
    up_s = snap.get(f"{base}.w3.scale").view(torch.uint8)[rows]
    w2_w = snap.get(f"{base}.w2.weight").view(torch.uint8)[:, rank * it // 2 : (rank + 1) * it // 2]
    w2_s = snap.get(f"{base}.w2.scale").view(torch.uint8)[:, rank * it // 32 : (rank + 1) * it // 32]

    def to_bf16_bits(e8m0: torch.Tensor) -> torch.Tensor:
        return (e8m0.to(torch.int32) << 7).to(torch.int16).view(torch.bfloat16)

    return (
        torch.cat([gate_w, up_w], dim=0).contiguous(),
        to_bf16_bits(torch.cat([gate_s, up_s], dim=0)).contiguous(),
        w2_w.contiguous(),
        to_bf16_bits(w2_s).contiguous(),
    )


E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(packed: torch.Tensor, scale_u8: torch.Tensor, dev) -> torch.Tensor:
    """[N, K/2] uint8 nibbles (low nibble first) + [N, K/32] ue8m0 -> float32 [N, K] (as test-kt-one-layer)."""
    packed = packed.view(torch.uint8).to(dev)
    n, half_k = packed.shape
    lo = packed & 0x0F
    hi = packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(n, half_k * 2)
    mag = E2M1_GRID.to(dev)[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    exp = scale_u8.view(torch.uint8).to(dev).to(torch.int32) - 127
    scales = torch.pow(2.0, exp.to(torch.float32)).repeat_interleave(32, dim=1)
    return mag * sign * scales


def reference_cpu_tier(snap: Snapshot, layer: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor,
                       first_cpu_id: int, swiglu_limit: float, dev) -> torch.Tensor:
    """sum_k w_k * expert_k(x) over slots routed to CPU experts (global id >= first_cpu_id), fp32."""
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=dev)
    xf = x.to(dev).float()
    for e in sorted(set(topk_ids.flatten().tolist())):
        if e < first_cpu_id:
            continue
        base = f"layers.{layer}.ffn.experts.{e}"
        w1 = dequant_mxfp4(snap.get(f"{base}.w1.weight"), snap.get(f"{base}.w1.scale"), dev)
        w3 = dequant_mxfp4(snap.get(f"{base}.w3.weight"), snap.get(f"{base}.w3.scale"), dev)
        w2 = dequant_mxfp4(snap.get(f"{base}.w2.weight"), snap.get(f"{base}.w2.scale"), dev)
        rows, slots = torch.nonzero(topk_ids.to(dev) == e, as_tuple=True)
        xe = xf[rows]
        gate = (xe @ w1.T).clamp(max=swiglu_limit)
        up = (xe @ w3.T).clamp(min=-swiglu_limit, max=swiglu_limit)
        y = (torch.nn.functional.silu(gate) * up) @ w2.T
        out.index_add_(0, rows, y * topk_w.to(dev)[rows, slots].float().unsqueeze(1))
    return out


def resident_layout(snap: Snapshot, layer: int, global_ids: list, rank: int, tp: int, inter: int, dev):
    """Build [n, ...] expert tensors the way SGLang's resident loader does (w13 = [up; gate], interleaved E8M0)."""
    from flashinfer import block_scale_interleave

    it = inter // tp
    rows = slice(rank * it, (rank + 1) * it)
    w13, s13, w2, s2 = [], [], [], []
    for e in global_ids:
        base = f"layers.{layer}.ffn.experts.{e}"
        w13.append(torch.cat([snap.get(f"{base}.w3.weight").view(torch.int8)[rows], snap.get(f"{base}.w1.weight").view(torch.int8)[rows]]))
        s13.append(torch.cat([snap.get(f"{base}.w3.scale").view(torch.uint8)[rows], snap.get(f"{base}.w1.scale").view(torch.uint8)[rows]]))
        w2.append(snap.get(f"{base}.w2.weight").view(torch.int8)[:, rank * it // 2 : (rank + 1) * it // 2])
        s2.append(snap.get(f"{base}.w2.scale").view(torch.uint8)[:, rank * it // 32 : (rank + 1) * it // 32])
    out = [torch.stack(t).contiguous().to(dev) for t in (w13, s13, w2, s2)]
    for i in (1, 3):
        out[i].copy_(block_scale_interleave(out[i]).reshape_as(out[i]))
    return out


def run_sm120_mxfp4_moe(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, w13, s13, w2, s2,
                        swiglu_limit: float, tp: int, rank: int) -> torch.Tensor:
    """FlashInfer CUTLASS MXFP8xMXFP4 fused MoE exactly as SGLang's flashinfer_mxfp4 runner calls it."""
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import ActivationType, cutlass_fused_moe

    n = w13.shape[0]
    xq, x_sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
    ones = torch.ones(n, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape[0], x.shape[1], dtype=torch.bfloat16, device=x.device)
    cutlass_fused_moe(
        input=xq,
        token_selected_experts=ids.to(torch.int32),
        token_final_scales=weights.to(torch.float32),
        fc1_expert_weights=w13.view(torch.int64),
        fc2_expert_weights=w2.view(torch.int64),
        output_dtype=torch.bfloat16,
        quant_scales=[s13.view(torch.int32), ones, s2.view(torch.int32), ones],
        input_sf=x_sf,
        swiglu_limit=torch.full((n,), swiglu_limit, dtype=torch.float32, device=x.device),
        tp_size=tp,
        tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=True,
        activation_type=ActivationType.Swiglu,
        tune_max_num_tokens=max(8, 1 << (x.shape[0] - 1).bit_length()),
        output=out,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-path", required=True, help="HF snapshot dir")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--num-gpu-experts", type=int, default=208)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--intermediate", type=int, default=2304)
    ap.add_argument("--swiglu-limit", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--threadpool-count", type=int, default=1)
    ap.add_argument("--gpu-tp", type=int, default=2)
    ap.add_argument("--groups", default="8,16,32,44", help="experts per group for the grouped/pipelined runs")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--check-experts", default="0,1,175", help="compact CPU expert ids to byte-compare")
    ap.add_argument("--skip-gpu", action="store_true", help="host export only (no CUDA)")
    ap.add_argument("--check-kernel", dest="check_kernel", action="store_true", default=True,
                    help="stage the layer through the overlay classes and run the fused MoE (default)")
    ap.add_argument("--no-check-kernel", dest="check_kernel", action="store_false")
    ap.add_argument("--check-tokens", type=int, default=16)
    ap.add_argument("--x-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-timing", action="store_true", help="stop after the correctness checks")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    report: dict = {"args": vars(args), "torch": torch.__version__}

    import kt_kernel
    from kt_kernel import KTMoEWrapper

    report["kt_kernel"] = getattr(kt_kernel, "__version__", "unknown")

    sys.path.insert(0, "/sgl-workspace/sglang/python")
    import sglang.srt.layers.moe.kt_ep_wrapper as ktw
    import sglang.srt.layers.moe.kt_prefill as ktp  # _HostSlice/_StreamedExpertSlots live here since T3.19d

    n_cpu = args.num_experts - args.num_gpu_experts
    gpu_mask = torch.zeros(args.num_experts, dtype=torch.bool)
    gpu_mask[: args.num_gpu_experts] = True

    from sglang.srt.layers.moe.kt_runtime import KtRuntime  # compact CPU-expert load, as the server does (T3.19c)

    t0 = time.time()
    wrapper = KTMoEWrapper(
        layer_idx=args.layer,
        num_experts=n_cpu,
        num_experts_per_tok=args.topk,
        hidden_size=args.hidden,
        moe_intermediate_size=args.intermediate,
        gpu_experts_mask=None,
        cpuinfer_threads=args.threads,
        threadpool_count=args.threadpool_count,
        weight_path=args.weight_path,
        chunked_prefill_size=256,
        method="MXFP4",
        max_deferred_experts_per_token=0,
        swiglu_limit=args.swiglu_limit,
    )
    KtRuntime(tp_rank=0).load_cpu_experts(wrapper, torch.arange(n_cpu), gpu_mask)
    report["load_seconds"] = round(time.time() - t0, 2)
    print(f"loaded layer {args.layer}: {n_cpu} CPU experts in {report['load_seconds']} s")

    tp = args.gpu_tp
    inter_tp = args.intermediate // tp
    hosts = [HostSlots(n_cpu, args.hidden, inter_tp) for _ in range(tp)]
    per_expert = hosts[0].bytes_per_expert  # one rank's slice
    report["bytes_per_expert_per_rank"] = per_expert
    report["bytes_per_layer_all_ranks"] = per_expert * tp * n_cpu
    print(
        f"slice per expert per rank {per_expert / 2**20:.2f} MiB; "
        f"layer export (all ranks) {gib(per_expert * tp * n_cpu):.3f} GiB; pinned host {gib(per_expert * tp * n_cpu):.2f} GiB"
    )

    def submit(expert: int, slot: int) -> None:
        p = [h.ptrs(slot) for h in hosts]
        wrapper.submit_write_weight_scale_to_buffer(
            tp, expert, [q[0] for q in p], [q[1] for q in p], [q[2] for q in p], [q[3] for q in p]
        )

    # ---- 1. byte-compare -------------------------------------------------
    snap = Snapshot(Path(args.weight_path))
    cpu_global_ids = torch.nonzero(~gpu_mask).flatten().tolist()
    checks = []
    for c in (int(x) for x in args.check_experts.split(",") if x != ""):
        submit(c, c)
        wrapper.sync_write_weight_scale_to_buffer()
        g = cpu_global_ids[c]
        for r in range(tp):
            exp = expected_slices(snap, args.layer, g, r, tp, args.intermediate)
            got = tuple(t[c] for t in hosts[r].tensors)
            names = ("w13_weight", "w13_scale_bf16", "w2_weight", "w2_scale_bf16")
            row = {"compact_id": c, "global_id": g, "rank": r}
            for name, e, gt in zip(names, exp, got):
                same = e.shape == gt.shape and torch.equal(e.view(torch.uint8).flatten(), gt.view(torch.uint8).flatten())
                row[name] = "identical" if same else f"MISMATCH shape {tuple(gt.shape)} vs {tuple(e.shape)}"
            checks.append(row)
            print(f"export check expert {c} (global {g}) rank {r}: " + ", ".join(f"{k}={row[k]}" for k in names))
    report["byte_compare"] = checks
    all_ok = all(v == "identical" for row in checks for k, v in row.items() if k.endswith(("weight", "bf16")))
    report["byte_compare_all_identical"] = all_ok
    if not all_ok:
        print("BYTE COMPARE FAILED — layout assumption wrong; stopping before timing")
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
        return 1

    # ---- 1b. streamed path end to end (overlay classes + FlashInfer kernel) -------
    # Export every expert through the overlay's _HostSlice layout, stage on each
    # GPU with _StreamedExpertSlots.load(), run the SM120 fused MoE per TP rank,
    # sum the rank slices and compare with (a) the same kernel on tensors built
    # the resident-loader way (expect identical bytes -> identical output) and
    # (b) the dequantized fp32 reference.
    if not args.skip_gpu and args.check_kernel:
        devs = [torch.device(f"cuda:{r}") for r in range(tp)]
        if torch.cuda.device_count() < tp:
            print(f"kernel check needs {tp} GPUs, have {torch.cuda.device_count()}; skipping")
        else:
            t = time.perf_counter()
            raws, hslices, slots = [], [], []
            for r in range(tp):
                nb = ktp._HostSlice.nbytes(n_cpu, args.hidden, inter_tp)
                raw = torch.empty(nb, dtype=torch.uint8, pin_memory=True)
                raws.append(raw)
                hslices.append(ktp._HostSlice(raw, n_cpu, args.hidden, inter_tp))
                slots.append(ktp._StreamedExpertSlots(n_cpu, args.hidden, inter_tp, devs[r]))
            for e in range(n_cpu):
                p = [h.ptrs(e) for h in hslices]
                wrapper.submit_write_weight_scale_to_buffer(
                    tp, e, [q[0] for q in p], [q[1] for q in p], [q[2] for q in p], [q[3] for q in p]
                )
            wrapper.sync_write_weight_scale_to_buffer()
            for r in range(tp):
                with torch.cuda.device(devs[r]):
                    slots[r].load(hslices[r])
            torch.cuda.synchronize()

            g = torch.Generator().manual_seed(args.seed)
            T = args.check_tokens
            x = (torch.randn(T, args.hidden, generator=g) * args.x_scale).to(torch.bfloat16)
            # top-k distinct experts per token: 1 GPU-resident id (must be skipped) + CPU ids
            ids = torch.empty(T, args.topk, dtype=torch.int64)
            for i in range(T):
                perm = torch.randperm(n_cpu, generator=g)[: args.topk - 1] + args.num_gpu_experts
                ids[i] = torch.cat([torch.randint(0, args.num_gpu_experts, (1,), generator=g), perm])
            w = torch.softmax(torch.rand(T, args.topk, generator=g), dim=-1)
            compact = torch.where(ids >= args.num_gpu_experts, ids - args.num_gpu_experts, torch.full_like(ids, -1))

            layout_same, out_streamed, out_resident = [], None, None
            for r in range(tp):
                d = devs[r]
                with torch.cuda.device(d):
                    res = resident_layout(snap, args.layer, cpu_global_ids, r, tp, args.intermediate, d)
                    st = (slots[r].w13, slots[r].s13, slots[r].w2, slots[r].s2)
                    layout_same.append([bool(torch.equal(a.view(torch.uint8), b.view(torch.uint8))) for a, b in zip(st, res)])
                    ys = run_sm120_mxfp4_moe(x.to(d), compact.to(d), w.to(d), *st, args.swiglu_limit, tp, r)
                    yr = run_sm120_mxfp4_moe(x.to(d), compact.to(d), w.to(d), *res, args.swiglu_limit, tp, r)
                    torch.cuda.synchronize(d)
                out_streamed = ys.to(devs[0]).float() if out_streamed is None else out_streamed + ys.to(devs[0]).float()
                out_resident = yr.to(devs[0]).float() if out_resident is None else out_resident + yr.to(devs[0]).float()
                del res
            ref = reference_cpu_tier(snap, args.layer, x, ids, w, args.num_gpu_experts, args.swiglu_limit, devs[0])
            err_s = (out_streamed - ref).norm() / ref.norm()
            err_r = (out_resident - ref).norm() / ref.norm()
            diff_sr = (out_streamed - out_resident).abs().max()
            report["kernel_check"] = {
                "tokens": T,
                "layout_bytes_identical_per_rank": layout_same,  # [w13, s13, w2, s2] per rank
                "streamed_vs_resident_max_abs_diff": float(diff_sr),
                "streamed_rel_l2_err_vs_ref": float(err_s),
                "resident_rel_l2_err_vs_ref": float(err_r),
                "ref_abs_mean": float(ref.abs().mean()),
                "out_finite": bool(torch.isfinite(out_streamed).all()),
                "seconds": round(time.perf_counter() - t, 1),
            }
            print(
                f"kernel check: staged layout == resident layout per rank {layout_same}; "
                f"streamed vs resident max|diff| {float(diff_sr):.4g}; rel-L2 vs fp32 ref: streamed {float(err_s):.4g}, "
                f"resident {float(err_r):.4g} (ref mean|y| {float(ref.abs().mean()):.3g}); finite={bool(torch.isfinite(out_streamed).all())}"
            )
            del slots, raws, hslices
            torch.cuda.empty_cache()

    if args.skip_timing:
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
        return 0

    # ---- 2. export throughput ------------------------------------------------
    layer_bytes = per_expert * tp * n_cpu

    def time_export_all() -> float:
        t = time.perf_counter()
        for e in range(n_cpu):
            submit(e, e)
        wrapper.sync_write_weight_scale_to_buffer()
        return time.perf_counter() - t

    times = [time_export_all() for _ in range(args.iters)]
    best = min(times)
    report["export_all"] = {
        "seconds": [round(x, 4) for x in times],
        "best_seconds": round(best, 4),
        "gbps_best": round(layer_bytes / best / 1e9, 2),
        "ms_per_expert_best": round(best / n_cpu * 1e3, 3),
    }
    print(
        f"export all {n_cpu} experts (one sync): best {best * 1e3:.1f} ms = {layer_bytes / best / 1e9:.1f} GB/s, "
        f"{best / n_cpu * 1e3:.3f} ms/expert; runs {[round(x * 1e3, 1) for x in times]} ms"
    )

    groups = [int(g) for g in args.groups.split(",") if g != ""]
    grouped = {}
    for G in groups:
        t = time.perf_counter()
        for start in range(0, n_cpu, G):
            for e in range(start, min(start + G, n_cpu)):
                submit(e, e)
            wrapper.sync_write_weight_scale_to_buffer()
        dt = time.perf_counter() - t
        grouped[G] = {"seconds": round(dt, 4), "gbps": round(layer_bytes / dt / 1e9, 2)}
        print(f"export grouped G={G}: {dt * 1e3:.1f} ms = {layer_bytes / dt / 1e9:.1f} GB/s")
    report["export_grouped"] = grouped

    if args.skip_gpu:
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
        return 0

    # ---- 3. GPU prep: H2D swap-copies to every card concurrently + scale fix-up ----
    devices = [torch.device(f"cuda:{i}") for i in range(min(tp, torch.cuda.device_count()))]
    if len(devices) < tp:
        print(f"only {len(devices)} GPUs visible; timing H2D on those (rank slices {list(range(len(devices)))})")
    layers = [DeviceLayer(n_cpu, args.hidden, inter_tp, d) for d in devices]

    def sync_all() -> None:
        for L in layers:
            L.stream.synchronize()

    def time_h2d_full(with_scales: bool) -> float:
        sync_all()
        t = time.perf_counter()
        for r, L in enumerate(layers):
            L.copy_group(hosts[r], range(n_cpu), range(n_cpu))
            if with_scales:
                L.finish_scales()
        sync_all()
        return time.perf_counter() - t

    time_h2d_full(True)  # warm-up (allocator, interleave kernel)
    h2d = [time_h2d_full(False) for _ in range(args.iters)]
    h2d_s = [time_h2d_full(True) for _ in range(args.iters)]
    per_rank_bytes = per_expert * n_cpu
    report["h2d_full_layer"] = {
        "cards": len(devices),
        "copy_only_seconds": [round(x, 4) for x in h2d],
        "copy_plus_scales_seconds": [round(x, 4) for x in h2d_s],
        "gib_per_s_per_card_best": round(gib(per_rank_bytes) / min(h2d), 2),
    }
    print(
        f"H2D full layer to {len(devices)} card(s) concurrently: copy {min(h2d) * 1e3:.1f} ms "
        f"({gib(per_rank_bytes) / min(h2d):.1f} GiB/s per card), copy+scale fix-up {min(h2d_s) * 1e3:.1f} ms"
    )

    # ---- 4. pipelined: export(g+1) while H2D(g) ---------------------------------
    pipelined = {}
    for G in groups:
        n_groups = (n_cpu + G - 1) // G
        # two host slots of G experts each would be the serving layout; here the
        # host buffer holds the whole layer, so slot == expert and only the
        # ordering (write g+1 after launching H2D g) is what is measured.
        def run_once() -> float:
            sync_all()
            t = time.perf_counter()
            for e in range(0, min(G, n_cpu)):
                submit(e, e)
            wrapper.sync_write_weight_scale_to_buffer()
            for gi in range(n_groups):
                cur = range(gi * G, min((gi + 1) * G, n_cpu))
                for r, L in enumerate(layers):
                    L.copy_group(hosts[r], cur, cur)
                nxt = range((gi + 1) * G, min((gi + 2) * G, n_cpu))
                for e in nxt:
                    submit(e, e)
                wrapper.sync_write_weight_scale_to_buffer()
            for L in layers:
                L.finish_scales()
            sync_all()
            return time.perf_counter() - t

        ts = [run_once() for _ in range(args.iters)]
        pipelined[G] = {
            "seconds": [round(x, 4) for x in ts],
            "best_seconds": round(min(ts), 4),
            "layer_gbps_best": round(layer_bytes / min(ts) / 1e9, 2),
        }
        print(f"pipelined G={G}: best {min(ts) * 1e3:.1f} ms per layer ({layer_bytes / min(ts) / 1e9:.1f} GB/s exported)")
    report["pipelined"] = pipelined

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

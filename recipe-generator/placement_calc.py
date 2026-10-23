#!/usr/bin/env python3
"""Recipe Generator placement calculator: budget JSON + hardware JSON -> tier plan + projections.

This is the generalized, model-agnostic successor to scripts/feasibility-calc.py. It
decides WHERE each checkpoint component goes (GPU VRAM / host DRAM+CPU / NVMe), which
levers realize that placement (GPU-expert count + hot-expert map, KV settings, prefill
chunk size, deferral D, Engram tiering, row cache, prefetch modes), and projects the
serving numbers (C1/C4 decode, prefill, TTFT) that placement should deliver.

Inputs
------
budget JSON   The output of scripts/checkpoint-budget.py (per-component bytes from
              safetensors headers only), extended with an "arch" section describing the
              model's activation structure (or an "extends" key pointing at such a
              budget; see examples/). PLACEMENT.md documents the schema.
hardware JSON Capacities plus the bandwidths that matter, all MEASURED on the target
              host (never vendor specs): GPU DRAM bandwidth, host DRAM read bandwidth,
              PCIe per GPU, NVMe random row gather + sequential read, plus a
              calibration section - a known deployment whose numbers the projections
              must reproduce.

The calibration rule (the generator's prime directive, see PLACEMENT.md):
  EVERY projection must be validated against at least one measured datapoint before it
  is trusted. With `hardware.calibration` present the calculator anchors its model
  constants to it (inverting gpu_ms_per_token from the measured C1 when needed) and
  labels projections `trust: calibrated`. Without calibration everything is
  `trust: provisional` and the plan header says UNVALIDATED - provisional numbers are
  for ranking experiments, never for promises. A calibration from a different GPU
  count scales the model but stays `calibrated-model` (cross-hardware): the scaled
  row still needs its own datapoint before it is quoted.

Exit status: 0 plan produced (trust may still be provisional), 2 input error, 3 a
requested layout does not fit (mirrors scripts/feasibility-calc.py so CI can pin
fits/does-not-fit verdicts). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GIB = 2**30

# ---------------------------------------------------------------------------
# Constants. Every number is measured on the reference host
# (2x RTX PRO 6000 + Ryzen TR PRO 9955WX 16C, evidence/ tree) or a published
# reference value; all are calibration values, never universal claims.
# Citations: PLACEMENT.md "Constant register".
# ---------------------------------------------------------------------------
DEFAULT_GPU_UTIL = 0.95                    # fraction of VRAM the engine may claim (feasibility-calc)
DEFAULT_GPU_RESERVE_GIB = 4.0              # CUDA ctx, NCCL, workspaces, graphs, activations (feasibility-calc)
DEFAULT_HOST_RESERVE_GIB = 24.0            # OS, page cache, serving process, pinned staging (feasibility-calc)
DEFAULT_KV_BYTES_PER_TOKEN = 890           # CSA2 KV measured for DSV4.1 (vLLM recipe / LibertAI card)
DISPATCH_MS_PER_LAYER = 0.09               # derived: CPU-tier round-trip cost per MoE layer (evidence/e21, T3.15)
D0_PENALTY_FRACTION = 0.42                 # D=0 pays this fraction of the CPU serial cost (fitted to measured
                                           # D=0 49.8 tok/s vs D=4 60.8 tok/s: the CPU path still pipelines a bit)
D_MAX_QUALIFIED = 4                        # D=5 fails the NLL gate (measured; T3.15) - the guard
COLD_ENGRAM_TAX_S = 2.4                    # cold first-request chunk-0 Engram fetch, measured 1.7-3.1 s (T3.49)
TTFT_OVERHEAD_S = 0.1                      # tokenize + one-token handoff (bench-prefill contract)
GPU_BATCH_EXPONENT = 0.5                   # decode batch scaling gpu_ms*sqrt(n); reproduces measured C1->C4 (2.04x)
WARM_COLD_PREFILL_RATIO = 1.037            # measured warm/cold prefill ratio 4,504/4,344 (T3.19d discipline)
# Hot-expert map quality: the fraction of would-be CPU routes a placement catches
# beyond its count share. checkpoint_order = 0 (random split, measured 42-46% CPU
# share); fitted_static = 0.65 (el8 mix map: 43% -> 15%, T3.8b); runtime_oracle =
# 0.93 (per-set oracle 1-3% CPU share, T3.8c).
MAP_QUALITY = {"checkpoint_order": 0.0, "fitted_static": 0.65, "runtime_oracle": 0.93}
# This-host fitted line: how many GPU experts per layer coexist with a prefill chunk
# size (fused-MoE workspace). Anchored at three measured points (219@8192, 226@4096,
# 230@2048, "16384 costs 14 experts"); recalibrate on a new host.
CHUNK_REF = {2048: 233, 4096: 226, 8192: 219, 16384: 205}
CHUNK_PREFERENCE = [8192, 4096, 16384, 2048]   # 16384 rejected by measurement (-4.4% C1, E34d-iv)
ENGRAM_MARGIN = 0.8                        # run NVMe row gather at <=80% of its measured rate


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_budget(path: Path) -> dict:
    """Load a checkpoint-budget.py-style JSON. Accepts `total_gib` inside each
    component (checkpoint-budget.py v2) or computes it from weight+scale bytes.
    An "extends" key names another budget JSON to inherit (so example files add
    only the arch section without copying component bytes and drifting)."""
    data = json.loads(path.read_text())
    if data.get("extends"):
        base = load_budget((path.parent / data["extends"]).resolve())
        base_arch = dict(base["arch"])
        base_arch.update(data.get("arch") or {})
        data = dict(data)
        data["arch"] = base_arch
        data["components"] = dict(data.get("components") or {}) or base["components"]
        for key in ("header_total_gib", "header_total_bytes", "source", "revision"):
            if data.get(key) is None:
                data[key] = base.get(key)
    comps = {}
    for name, raw in (data.get("components") or {}).items():
        if "total_gib" in raw:
            total = raw["total_gib"]
        elif "gib" in raw:
            total = raw["gib"]
        else:
            total = (raw.get("weight_bytes", 0) + raw.get("scale_bytes", 0)) / GIB
        comps[name] = {"gib": float(total), "detail": raw}
    total = data.get("header_total_gib")
    if total is None:
        total = (data.get("header_total_bytes") or data.get("index_total_size_bytes") or 0) / GIB
    arch = data.get("arch") or {}
    return {
        "source": str(data.get("source", path.name)),
        "revision": data.get("revision"),
        "total_gib": float(total),
        "baseline_dtype": data.get("baseline_dtype"),
        "components": comps,
        "arch": {
            "moe_layers": int(arch.get("moe_layers", data.get("routed_expert_layers", 0))),
            "routed_experts_per_layer": int(arch.get("routed_experts_per_layer", 0)),
            "experts_per_token": int(arch.get("experts_per_token", 0)),
            "attention_heads": int(arch.get("attention_heads", 1)),
            "kv_bytes_per_token": float(arch.get("kv_bytes_per_token", DEFAULT_KV_BYTES_PER_TOKEN)),
            "engram_rows_per_token": float(arch.get("engram_rows_per_token", 0)),
            "engram_row_bytes": float(arch.get("engram_row_bytes", 0)),
        },
        "routed_expert_layers": int(data.get("routed_expert_layers", arch.get("moe_layers", 0))),
    }


def load_hardware(path: Path) -> dict:
    return json.loads(path.read_text())


def load_quant(path: Path) -> tuple:
    data = json.loads(path.read_text())
    return data.get("options", {}), data.get("default", "mxfp4")


# ---------------------------------------------------------------------------
# Tier planning
# ---------------------------------------------------------------------------
VRAM_ALWAYS = ["attention", "shared_expert", "embedding_head", "router", "norms",
               "hyper_connections", "engram_proj", "other"]
OPTIONAL_VRAM = {"dspark": "include_dspark", "vision": "include_vision"}


def component_gib(comps, names, quant, dense: bool) -> float:
    factor = quant["dense_factor" if dense else "expert_factor"]
    return sum(comps.get(n, {"gib": 0.0})["gib"] * factor for n in names)


def tier_plan(budget, hw, quant, opts):
    """Assign every component a tier and produce the lever settings that realize it."""
    comps = budget["components"]
    arch = budget["arch"]
    gpus = int(hw.get("gpus", len(hw.get("gpu_list", []))))
    vram_per_gpu = float(hw["vram_gib_per_gpu"])
    host_gib = float(hw["host_ram_gib"])
    util = float(hw.get("util", DEFAULT_GPU_UTIL))
    gpu_reserve = float(hw.get("gpu_reserve_gib", DEFAULT_GPU_RESERVE_GIB))
    host_reserve = float(hw.get("host_reserve_gib", DEFAULT_HOST_RESERVE_GIB))

    usable_vram = gpus * (vram_per_gpu * util - gpu_reserve)

    # Non-expert resident weights.
    fixed = component_gib(comps, VRAM_ALWAYS, quant, dense=True)
    for name in OPTIONAL_VRAM:
        if opts.get(OPTIONAL_VRAM[name], True) and name in comps:
            fixed += comps[name]["gib"] * quant["dense_factor"]

    # KV cache: context x slots.
    ctx = opts.get("context_tokens", 262144)
    slots = opts.get("sequences", 4)
    kv_gib = arch["kv_bytes_per_token"] * ctx * slots / GIB
    kv_fraction = kv_gib / usable_vram if usable_vram > 0 else 0.0

    # Routed experts: capacity in VRAM, then the chunk-workspace trade.
    experts_gib = comps.get("routed_experts", {"gib": 0.0})["gib"] * quant["expert_factor"]
    moe = arch["moe_layers"] > 0 and arch["routed_experts_per_layer"] > 0
    total_experts = arch["moe_layers"] * arch["routed_experts_per_layer"]
    per_expert_gib = experts_gib / total_experts if (moe and total_experts) else 0.0
    expert_capacity = usable_vram - fixed - kv_gib
    gpu_fraction = min(1.0, max(0.0, expert_capacity / experts_gib)) if (moe and experts_gib > 0) else 0.0
    cap_per_layer = int(round(gpu_fraction * total_experts / arch["moe_layers"])) if moe else 0

    chunk = opts.get("chunk_size")
    workspace_model = hw.get("chunk_workspace_model", "sglang-fused-moe")
    if moe and experts_gib > 0 and workspace_model == "none":
        # Engines without a chunk-coupled fused-MoE workspace (llama.cpp and
        # friends): chunk size is a pure TTFT/prefill lever.
        gpu_per_layer = cap_per_layer
        chunk = chunk or CHUNK_PREFERENCE[0]
    elif moe and experts_gib > 0:
        if cap_per_layer >= 240:
            # VRAM-rich: the fused-MoE workspace is not binding; keep the
            # measured-preferred chunk (16384 stays opt-in: -4.4% C1, E34d-iv).
            gpu_per_layer = cap_per_layer
            chunk = chunk or CHUNK_PREFERENCE[0]
        else:
            chosen = None
            for c in CHUNK_PREFERENCE:
                eff = cap_per_layer - (CHUNK_REF[4096] - CHUNK_REF[c])
                if eff >= 0 and eff <= CHUNK_REF[c]:
                    chosen = c
                    break
            chosen = chosen or CHUNK_PREFERENCE[-1]
            chunk = chunk or chosen
            gpu_per_layer = max(0, min(cap_per_layer,
                                       cap_per_layer - (CHUNK_REF[4096] - CHUNK_REF[chunk])))
    else:
        gpu_per_layer = 0
        chunk = chunk or CHUNK_PREFERENCE[0]

    gpu_experts = gpu_per_layer * arch["moe_layers"] if moe else 0
    cpu_experts = total_experts - gpu_experts if moe else 0
    gpu_fraction = gpu_experts / total_experts if (moe and total_experts) else 0.0
    cpu_experts_gib = experts_gib * (1 - gpu_fraction) if moe else 0.0

    # Engram tiering: DRAM (as shipped) vs NVMe + page cache (refined in projections).
    engram_gib = comps.get("engram_table", {"gib": 0.0})["gib"] * quant["engram_factor"]
    has_engram = engram_gib > 0
    dram_ok = (cpu_experts_gib + engram_gib + host_reserve) <= host_gib
    if not has_engram:
        engram_tier = "none"
        engram_host_gib = 0.0
    elif dram_ok:
        engram_tier = "dram"
        engram_host_gib = engram_gib
    else:
        engram_tier = "nvme"
        engram_host_gib = float(hw.get("engram_row_cache_gib", 4.0))

    # Deferral.
    d = opts.get("deferral")
    if d is None:
        d = D_MAX_QUALIFIED if (moe and cpu_experts > 0) else 0
    d_guard = ""
    if d > D_MAX_QUALIFIED:
        d_guard = f"D={d} exceeds the qualified max {D_MAX_QUALIFIED} (D=5 failed NLL, T3.15); treat as experiment"

    # Hot-expert map.
    map_name = opts.get("map")
    if map_name is None:
        map_name = "fitted_static" if (moe and cpu_experts > 0) else "checkpoint_order"
    map_quality = MAP_QUALITY.get(map_name, 0.0)
    cpu_share = max(0.0, (1 - gpu_fraction) * (1 - map_quality)) if (moe and cpu_experts > 0) else 0.0

    host_need = cpu_experts_gib + engram_host_gib + host_reserve
    host_margin = host_gib - host_need
    vram_margin = expert_capacity - experts_gib * gpu_fraction

    rows = []
    def row(name, gib, tier, reason):
        rows.append({"component": name, "gib": round(gib, 2), "tier": tier, "reason": reason})
    for name in VRAM_ALWAYS:
        if name in comps and comps[name]["gib"] > 0:
            row(name, comps[name]["gib"] * quant["dense_factor"], "vram", "hot: touched every token/layer")
    for name in OPTIONAL_VRAM:
        if opts.get(OPTIONAL_VRAM[name], True) and name in comps and comps[name]["gib"] > 0:
            row(name, comps[name]["gib"] * quant["dense_factor"], "vram", "optional resident (disable to reclaim)")
    if moe and experts_gib > 0:
        row("routed_experts", experts_gib * gpu_fraction, "vram",
            f"{gpu_per_layer} experts/layer resident")
        if cpu_experts_gib > 0:
            row("routed_experts(cpu)", cpu_experts_gib, "dram",
                f"{cpu_experts // arch['moe_layers']} experts/layer computed on CPU")
    if has_engram:
        row("engram_table", engram_gib, engram_tier,
            "lookup: touched via row gathers, page-cache friendly")
    row("kv_cache", kv_gib, "vram", f"{ctx:,} tokens x {slots} slots @ {arch['kv_bytes_per_token']:.0f} B/token")

    fits = (expert_capacity >= 0 and host_margin >= 0)
    return {
        "rows": rows,
        "usable_vram_gib": usable_vram,
        "fixed_resident_gib": fixed,
        "kv_gib": kv_gib,
        "kv_fraction": kv_fraction,
        "experts_gib": experts_gib,
        "gpu_fraction": gpu_fraction,
        "gpu_experts": gpu_experts,
        "gpu_experts_per_layer": gpu_per_layer,
        "cpu_experts": cpu_experts,
        "cpu_experts_per_layer": cpu_experts // arch["moe_layers"] if moe else 0,
        "cpu_expert_gib": cpu_experts_gib,
        "per_expert_gib": per_expert_gib,
        "engram_tier": engram_tier,
        "engram_host_gib": engram_host_gib,
        "host_need_gib": host_need,
        "host_margin_gib": host_margin,
        "vram_margin_gib": vram_margin,
        "cpu_share": cpu_share,
        "map": map_name,
        "chunk": chunk,
        "deferral": d,
        "d_guard": d_guard,
        "tp_valid": arch["attention_heads"] % gpus == 0,
        "fits": fits,
    }


# ---------------------------------------------------------------------------
# Projections
# ---------------------------------------------------------------------------
def project_engram(budget, hw, plan, c4_tok_s, prefill_ms_chunk, chunk):
    """Row-gather budget check. DRAM gather is never binding (measured 4.15M
    rows/s, E3); the NVMe tier must stay <= 80% of its measured row rate."""
    arch = budget["arch"]
    if plan["engram_tier"] in ("none", "dram") or arch["engram_rows_per_token"] <= 0:
        return plan["engram_tier"], 0.0, 0.0, ""
    rows_per_s = max(c4_tok_s * arch["engram_rows_per_token"],
                     chunk * arch["engram_rows_per_token"] / (prefill_ms_chunk / 1000))
    nvme_rows = float(hw.get("nvme_random_rows_per_s", 0))
    if nvme_rows <= 0 or rows_per_s <= nvme_rows * ENGRAM_MARGIN:
        return plan["engram_tier"], rows_per_s, 0.0, ""
    # Re-quantized tables shrink bytes but NOT the page-cache footprint (E40);
    # the honest fallback is a denser user-space row cache (T3.48: 264 B rows,
    # ~22x denser than the page cache).
    need_gib = rows_per_s * arch["engram_row_bytes"] / GIB / 22.0
    note = ("row budget exceeds 80% of measured NVMe row gather; "
            "add a user-space row cache (T3.48) or move tables to DRAM")
    return "nvme+row-cache", rows_per_s, round(need_gib, 2), note


def project(budget, hw, quant, opts, calib):
    plan = tier_plan(budget, hw, quant, opts)
    arch = budget["arch"]
    gpus = int(hw.get("gpus", len(hw.get("gpu_list", []))))
    calib_gpus = int(calib.get("gpus", hw.get("gpus", gpus))) if calib else gpus

    # ---- gpu_ms_per_token: census value, or inverted from the calibration C1 ----
    gpu_ms = hw.get("gpu_ms_per_token")
    inverted = False
    if gpu_ms is None and calib and calib.get("c1_tok_s"):
        # At D=4 with the CPU path hidden, wall_ms = gpu_ms: invert, then scale
        # to the current GPU count relative to the measured deployment.
        gpu_ms = 1000.0 / calib["c1_tok_s"] * (calib_gpus / gpus)
        inverted = True
    if gpu_ms is None:
        gpu_ms = 5.0  # sibling V4-Flash NVFP4 TP2 reference; an ESTIMATE until a census exists
    gpu_ms = float(gpu_ms)

    dram_gbps = float(hw.get("dram_read_gbps", 0))
    cpu_bytes_per_token = 0.0
    if plan["cpu_experts"] > 0 and dram_gbps > 0:
        cpu_bytes_per_token = (plan["cpu_share"] * arch["experts_per_token"] *
                               arch["moe_layers"] * plan["per_expert_gib"] * GIB)
    cpu_bw_ms = cpu_bytes_per_token / (dram_gbps * 1e9) * 1e3
    dispatch_ms = arch["moe_layers"] * DISPATCH_MS_PER_LAYER if plan["cpu_experts"] > 0 else 0.0

    # ---- decode model ----
    d = plan["deferral"]
    serial_cost = cpu_bw_ms + dispatch_ms
    if d == 0:
        penalty = serial_cost * D0_PENALTY_FRACTION
    else:
        penalty = max(0.0, serial_cost - d * gpu_ms / 2.0)
    wall_1 = max(gpu_ms, cpu_bw_ms) + penalty
    c1 = 1000.0 / wall_1 if wall_1 > 0 else 0.0
    cn = {}
    for n in (2, 3, 4):
        wall_n = max(gpu_ms * (n ** GPU_BATCH_EXPONENT),
                     n * cpu_bw_ms + dispatch_ms / max(1, n))
        cn[n] = 1000.0 * n / wall_n if wall_n > 0 else 0.0
    c4 = cn[4]

    # ---- prefill model ----
    chunk = plan["chunk"]
    gpu_prefill_ms = gpu_ms * (chunk ** 0.5)
    cpu_stream_ms = 0.0
    if plan["cpu_experts"] > 0 and dram_gbps > 0:
        # Batching shares expert weights: the chunk touches roughly
        # chunk * routes * cpu_share / cpu_experts of the CPU-resident experts.
        touch = min(1.0, chunk * arch["experts_per_token"] * plan["cpu_share"] *
                    arch["moe_layers"] / max(1, plan["cpu_experts"]))
        cpu_stream_ms = (plan["cpu_expert_gib"] * GIB * touch) / (dram_gbps * 1e9) * 1e3
    engram_ms = 0.0
    if plan["engram_tier"].startswith("nvme") and arch["engram_rows_per_token"] > 0:
        seq_gbps = float(hw.get("nvme_sequential_gbps", 0)) or dram_gbps
        engram_ms = (chunk * arch["engram_rows_per_token"] * arch["engram_row_bytes"]
                     / (seq_gbps * 1e9) * 1e3)
    prefill_wall_ms = max(gpu_prefill_ms, cpu_stream_ms, engram_ms)
    prefill_tok_s = chunk * 1000.0 / prefill_wall_ms if prefill_wall_ms > 0 else 0.0

    engram_tier, rows_per_s, row_cache_gib, engram_note = project_engram(
        budget, hw, plan, c4, prefill_wall_ms, chunk)
    plan["engram_tier"] = engram_tier
    plan["engram_note"] = engram_note
    cold_tax = COLD_ENGRAM_TAX_S if engram_tier.startswith("nvme") and not opts.get("admission_prefetch") else 0.0

    # ---- calibration factors ----
    k_decode = k_c4 = k_prefill = None
    trust = "provisional"
    if calib:
        trust = "calibrated" if calib_gpus == gpus else "calibrated-model (cross-hardware)"
        if calib.get("c1_tok_s"):
            k_decode = calib["c1_tok_s"] / c1 if c1 else None
        if calib.get("c4_tok_s"):
            k_c4 = calib["c4_tok_s"] / c4 if c4 else None
        if calib.get("prefill_16k_cold_tok_s") or calib.get("prefill_tok_s"):
            measured_prefill = calib.get("prefill_16k_cold_tok_s") or calib.get("prefill_tok_s")
            k_prefill = measured_prefill / prefill_tok_s if prefill_tok_s else None
    elif hw.get("k_overrides"):
        # Host/engine calibration factors carried over from another example's
        # measured anchor (same host, same engine family). Weaker than a real
        # datapoint for THIS model - labeled as such.
        ko = hw["k_overrides"]
        k_decode, k_c4, k_prefill = ko.get("decode"), ko.get("c4"), ko.get("prefill")
        trust = "calibrated-model (k carried from another example)"

    plan["calib_note"] = (calib.get("evidence") if calib else
                          (hw.get("k_overrides") or {}).get("source", "none"))
    proj = {
        "trust": trust,
        "gpu_ms_per_token": round(gpu_ms, 2),
        "gpu_ms_inverted_from_c1": inverted,
        "cpu_share": round(plan["cpu_share"], 3),
        "cpu_bytes_per_token_mib": round(cpu_bytes_per_token / 2**20, 1),
        "cpu_bw_ms_per_token": round(cpu_bw_ms, 2),
        "dispatch_ms_per_token": round(dispatch_ms, 2),
        "decode_modeled": {"c1": round(c1, 1), "c2": round(cn[2], 1),
                           "c3": round(cn[3], 1), "c4": round(c4, 1)},
        "prefill_modeled": {"chunk": chunk, "prefill_tok_s": round(prefill_tok_s, 0),
                            "cpu_stream_ms": round(cpu_stream_ms, 0),
                            "gpu_prefill_ms": round(gpu_prefill_ms, 0)},
        "engram": {"rows_per_s_needed": round(rows_per_s, 0), "row_cache_gib": row_cache_gib},
        "k": {"decode": k_decode, "c4": k_c4, "prefill": k_prefill},
    }
    # Headline numbers: model x calibration factor (k=1 model-only when no datapoint).
    def ttft(tokens, rate):
        return tokens / rate + TTFT_OVERHEAD_S
    proj["projected"] = {
        "c1_tok_s": round(c1 * (k_decode or 1.0), 1),
        "c2_tok_s": round(cn[2] * (k_decode or 1.0), 1),
        "c3_tok_s": round(cn[3] * (k_decode or 1.0), 1),
        "c4_tok_s": round(c4 * (k_c4 or 1.0), 1),
        "prefill_cold_tok_s": round(prefill_tok_s * (k_prefill or 1.0), 0),
        "prefill_warm_tok_s": round(prefill_tok_s * (k_prefill or 1.0) * WARM_COLD_PREFILL_RATIO, 0),
        "ttft_16k_cold_s": round(ttft(16384, prefill_tok_s * (k_prefill or 1.0)), 1),
        "ttft_32k_cold_s": round(ttft(32768, prefill_tok_s * (k_prefill or 1.0)), 1),
        "cold_engram_tax_s": cold_tax,
        "cold_engram_note": ("chunk-0 Engram fetch adds 1.7-3.1 s on the first cold request "
                             "(T3.49); admission prefetch removes 0.2-0.4 s of it" if cold_tax else ""),
    }
    plan.update(proj)
    return plan


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(plan, budget, hw, quant_name):
    out = []
    out.append(f"model: {budget['source']}@{budget.get('revision') or '?'}  "
               f"({budget['total_gib']:.1f} GiB checkpoint)")
    header = (f"hardware: {hw.get('name', '?')}  quantization: {quant_name}  "
              f"trust: {plan['trust']}")
    if plan["trust"] == "provisional":
        header += "  [UNVALIDATED - provisional projections, not promises]"
    else:
        header += f"  [anchor: {plan.get('calib_note', '?')}]"
    out.append(header)
    out.append("")
    out.append("TIER PLAN")
    out.append(f"  {'component':24s} {'GiB':>8s}  {'tier':16s} reason")
    for r in plan["rows"]:
        out.append(f"  {r['component']:24s} {r['gib']:8.2f}  {r['tier']:16s} {r['reason']}")
    out.append("")
    out.append("LEVER SETTINGS (the resolved line)")
    out.append(f"  experts gpu={plan['gpu_experts']} cpu={plan['cpu_experts']} map={plan['map']} "
               f"cpu_share={plan['cpu_share']*100:.0f}% deferral D={plan['deferral']} "
               f"chunk={plan['chunk']} kv={plan['kv_gib']:.2f} GiB ({plan['kv_fraction']*100:.1f}% VRAM) "
               f"engram={plan['engram_tier']}")
    if plan["d_guard"]:
        out.append(f"  GUARD: {plan['d_guard']}")
    if plan.get("engram_note"):
        out.append(f"  ENGRAM: {plan['engram_note']}")
    if not plan["tp_valid"]:
        out.append("  GUARD: attention heads do not divide by GPU count - use EP/PP, not TP")
    out.append("")
    out.append("MARGINS")
    out.append(f"  usable VRAM {plan['usable_vram_gib']:.1f} GiB | fixed resident {plan['fixed_resident_gib']:.1f} GiB "
               f"| KV {plan['kv_gib']:.2f} GiB | VRAM margin {plan['vram_margin_gib']:.1f} GiB | "
               f"host need {plan['host_need_gib']:.1f} GiB | host margin {plan['host_margin_gib']:.1f} GiB | "
               f"fits: {'YES' if plan['fits'] else 'NO'}")
    out.append("")
    out.append("PROJECTIONS")
    p = plan["projected"]
    out.append(f"  decode C1 {p['c1_tok_s']} tok/s  C2 {p['c2_tok_s']}  C3 {p['c3_tok_s']}  C4 {p['c4_tok_s']} tok/s")
    out.append(f"  prefill cold {p['prefill_cold_tok_s']:.0f} tok/s / warm {p['prefill_warm_tok_s']:.0f}  "
               f"TTFT 16K cold {p['ttft_16k_cold_s']} s  32K cold {p['ttft_32k_cold_s']} s")
    if p.get("cold_engram_note"):
        out.append(f"  note: {p['cold_engram_note']}")
    out.append(f"  model internals: gpu_ms={plan['gpu_ms_per_token']} ms"
               + (" (inverted from calibration C1)" if plan['gpu_ms_inverted_from_c1'] else "") +
               f", cpu {plan['cpu_bytes_per_token_mib']} MiB/token, "
               f"cpu_bw {plan['cpu_bw_ms_per_token']} ms, dispatch {plan['dispatch_ms_per_token']} ms, "
               f"prefill cpu-stream {plan['prefill_modeled']['cpu_stream_ms']:.0f} ms/chunk")
    if plan["k"]["decode"] is not None:
        kd = plan["k"]["decode"]
        kc4 = plan["k"]["c4"]
        kp = plan["k"]["prefill"]
        out.append(f"  calibration k: decode {kd:.3f}"
                   + (f"  c4 {kc4:.3f}" if kc4 is not None else "")
                   + (f"  prefill {kp:.3f}" if kp is not None else ""))
    out.append(f"  engram rows/s needed {plan['engram']['rows_per_s_needed']:.0f}"
               + (f" (row cache suggestion {plan['engram']['row_cache_gib']} GiB)"
                  if plan["engram"]["row_cache_gib"] else ""))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("budget", type=Path, help="budget JSON (checkpoint-budget.py output, with arch section)")
    ap.add_argument("--hardware", type=Path, required=True, help="hardware JSON (capacities + measured bandwidths + calibration)")
    ap.add_argument("--quant", default=None, help="quantization option key (see quantization.json)")
    ap.add_argument("--quant-json", type=Path, default=None)
    ap.add_argument("--context-tokens", type=int, default=262144)
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=None, help="override prefill chunk (2048/4096/8192/16384)")
    ap.add_argument("--deferral", type=int, default=None, help="override deferral D")
    ap.add_argument("--map", choices=sorted(MAP_QUALITY), default=None)
    ap.add_argument("--no-dspark", action="store_true")
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--admission-prefetch", action="store_true",
                    help="assume chunk-0 Engram admission prefetch (kills the cold-Engram tax)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-calibration", action="store_true",
                    help="ignore the hardware calibration section (shows provisional output)")
    args = ap.parse_args()

    try:
        budget = load_budget(args.budget)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: cannot read budget: {exc}", file=sys.stderr)
        return 2
    try:
        hw = load_hardware(args.hardware)
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: cannot read hardware: {exc}", file=sys.stderr)
        return 2

    quant_path = Path(args.quant_json) if args.quant_json else Path(__file__).parent / "quantization.json"
    try:
        quant_options, default_quant = load_quant(quant_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read quantization options: {exc}", file=sys.stderr)
        return 2
    quant_name = args.quant or default_quant
    if quant_name not in quant_options:
        print(f"error: unknown --quant {quant_name!r} (have: {', '.join(sorted(quant_options))})", file=sys.stderr)
        return 2
    quant = quant_options[quant_name]
    if (budget.get("baseline_dtype") and quant.get("baseline")
            and budget["baseline_dtype"] != quant["baseline"]):
        print(f"warning: budget baseline_dtype {budget['baseline_dtype']!r} but "
              f"quantization {quant_name!r} declares baseline {quant['baseline']!r}; "
              "size factors are relative to different dtypes", file=sys.stderr)

    opts = {
        "context_tokens": args.context_tokens,
        "sequences": args.sequences,
        "chunk_size": args.chunk_size,
        "deferral": args.deferral,
        "map": args.map,
        "include_dspark": not args.no_dspark,
        "include_vision": not args.no_vision,
        "admission_prefetch": args.admission_prefetch,
    }
    calib = None if args.no_calibration else hw.get("calibration")
    plan = project(budget, hw, quant, opts, calib)

    out = {
        "source": f"{budget['source']}@{budget.get('revision')}",
        "hardware": hw.get("name"),
        "quantization": quant_name,
        "trust": plan["trust"],
        "plan": {k: v for k, v in plan.items() if k != "rows"},
        "tier_plan": plan["rows"],
    }
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(render(plan, budget, hw, quant_name))
    if not plan["fits"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

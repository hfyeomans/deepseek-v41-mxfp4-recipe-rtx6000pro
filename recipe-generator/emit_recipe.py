#!/usr/bin/env python3
"""Recipe Generator emitter: turn a placement plan into a RUNNABLE recipe.

    python3 emit_recipe.py PLAN.json --engine llama-cpp --out recipes/qwen3-30b-a3b-rtx3090 \
        --model Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M --repo-url https://github.com/<owner>/<repo>

The plan comes from `placement_calc.py --json` (or pass --budget/--hardware and the
calc flags to run it inline). The emitter writes a complete recipe tree:

    README.md            generated from the plan: model/hardware/quant, tier plan table,
                         the resolved line, run commands, gate checklist, trust banner
    MANIFEST.json        the plan + pins + emitter version (what the UI renders)
    PROVENANCE.md        immutable pins: model snapshot, engine base/overlay/image, k-factors
    config/profiles/*.env  profile files from the plan's lever settings + the preset deltas
    launch.sh            engine-specific launcher (docker run for sglang-hybrid; llama-server
                         flags for llama-cpp)
    preflight.sh         memory/bandwidth checks against the plan's margins
    verify.sh            smoke: endpoint up + one completion (+ the parent recipe's gates)
    engine-notes.md      the deep-lever surface + the advanced patch inventory and which
                         patches port to a new engine vs stay recipe-local
    .gitignore

With --repo-url the README gains a publish section (`gh repo create`, commit, push)
and the repo URL is stamped into MANIFEST/PROVENANCE. With --init-git the tree is
`git init`-ed and committed (local only; pushing is the owner's action).

The emitter NEVER runs a serve, downloads weights, or touches the GPU box - it only
writes files. GPU-free by design; the owner executes the emitted recipe on the host.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = "0.1.0"

# The complete knob surface of the parent recipe's typed registry
# (scripts/serve_config.py), extracted 2026-10-23 - the deep-lever coverage a
# generated sglang-hybrid recipe inherits. Single source of truth is
# registry-surface.json beside this file; the UI's knob panel never invents a
# knob the registry refuses.
REGISTRY_SURFACE = [
    (k["name"], k["type"], k["default"])
    for k in json.loads((HERE / "registry-surface.json").read_text())["knobs"]
]

# The advanced patch inventory: what the reference recipe changed in its engine,
# grouped by function, with a portability classification. See patches/sglang/README.md.
PATCH_INVENTORY = [
    ("placement / expert map", "0001, 0044-0046, 0061", "compact CPU-expert view, per-expert runner, per-layer expert remap at load, cluster-path fix", "RFC + local map file (the el8 map is model-specific)"),
    ("Engram file layout + prefetch", "0002, 0040-0043, 0047", "tables stay in the checkpoint shard; ahead-of-use row fetch, one hasher break, pinned staging, chunked hashing", "recipe-local (model-specific lookup tables); the prefetch *pattern* ports"),
    ("breakable CUDA graphs", "0003, 0009-0017", "BCG dataclass outputs, per-field row counts, capture-width scaling, fault attribution, page-split never-free", "upstream candidates PR 3/4 + diagnostics"),
    ("W8A8 block-FP8 [32,32] configs", "0004", "seven tuned Triton configs for RTX PRO 6000 Max-Q; 2.26x C1 (E20)", "upstream candidate PR 1"),
    ("page-split seals", "0005, 0015, 0022", "zero-fill scratch (stale FP8 NaN), reserve-split seal", "upstream candidate PR 3/4"),
    ("kt deferral", "0006", "expert deferral with the compact CPU view (D=4, +22% C1)", "RFC"),
    ("streamed prefill", "0007, 0030, 0035, 0048-0049", "layerwise GPU prefill coordinator, PrefillPipeline, E8M0 scale staging, shared weight arenas (+34-41% prefill)", "RFC"),
    ("B2 direct-UVA experts", "0019, 0029", "built, measured +2%, retired", "local (retired - do not port)"),
    ("kt boundary contract", "0020, 0024, 0028", "executable SGLang-kt contract, CpuExpertProvider seam, KtRuntime ownership", "RFC"),
    ("DSpark overlap gate", "0021", "multi-stream overlap forced off for DSpark on SM120 (Bug A)", "upstream candidate"),
    ("timeline / fingerprints", "0023, 0056-0058", "per-layer decode timeline recorder + step fingerprints (opt-in diagnostics)", "diagnostics (opt-in)"),
    ("indexer tiling + score budget", "0025-0027, 0031, 0033", "256 MiB score budget, tiled dense fp4 scoring, block-id candidate masks, page-offset gathers (~2x prefill)", "upstream candidate PR 6"),
    ("expandable segments after capture", "0039", "allocator switch kills 262Kx4 stress fragmentation (678 vs 1,082 s)", "upstream candidate"),
    ("robustness", "0050-0052", "per-request NaN row rejection, max_new_tokens clamp, API-key file", "upstream candidates"),
    ("deterministic decode", "0053-0055", "sorted decode top-k + exact deterministic ties (bit-reproducible greedy)", "upstream candidate (sglang #35257 post ready)"),
    ("NGRAM chain-only verify", "0036-0038, 0059", "speculation for pools without move_kv_cache; inert unless NGRAM=1 (E43 found unsafe)", "local, off by default"),
    ("decode-shape FP8 GEMV", "0060", "opt-in +1-2% C1 served; stays 0 until a pinned A/B", "local, opt-in"),
]

DEEP_LEVERS = """\
The full deep-lever surface a generated sglang-hybrid recipe inherits (each with the
measured number behind its default; complete table in registry-surface.json):

- KV cache: CONTEXT_LENGTH x MAX_RUNNING_REQUESTS (262K x 4 default; ladder 512K/1M needs
  MEM_FRACTION_STATIC=0.962 - at 0.96 the pool silently caps at 505,856 tokens), MAX_TOTAL_TOKENS
  (engine default sized the pool 15x too large, E26), SGLANG_LOGPROB_CHUNK_SIZE=512 (a logprob
  request otherwise materializes 1.06 GB of fp32 logits and a CUDA OOM there kills the server).
- Expert placement: KT_NUM_GPU_EXPERTS + KT_EXPERT_MAP (219/165 + el8 map: CPU share 42->15%,
  C1 +37%, C4 +120%), KT_CPUINFER=16 (24 -> -12%; 30 -> 5 tok/s collapse), KT_MAX_DEFERRED=4
  (+22% C1; D=5 fails NLL - guarded), KT_GPU_PREFILL_MIN_TOKENS=1024 (prefill 216->1,375 tok/s).
- Prefill: CHUNKED_PREFILL_SIZE=8192 (16384: +22% prefill for -4.4% C1; guarded beside >219 experts).
- Engram/NVMe: ENGRAM_FILE_PREFETCH=ahead (+16-26% prefill), ENGRAM_CHUNK_AHEAD=1 (+7.6-11%),
  ENGRAM_ADMISSION_PREFETCH (off; -0.2-0.4 s cold TTFT when on), page-cache hygiene at ready.
- CUDA graphs: CUDA_GRAPH_BS="1 2 3 4" (eager is 4.4-5.6x slower; bs=3 capture +28% C3).
- Determinism: SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0 + SGLANG_DSV41_SORT_DECODE_TOPK=1
  (bit-reproducible greedy at zero cost), exact top-k patch 0055.
- Robustness: KT_REJECT_NAN_ROWS=1 (one 500, server up), SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS=1.
- Allocator: SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1 (stress 678 vs 1,082 s).
- Host: HOST_THREAD_CPUS=auto (kills the C4 106/123 bimodality), BIOS FCLK 2100 slot 3 (+3.1% C1),
  power sweep protocol (decode cap-insensitive, prefill pins to the cap - per-model).
- Kernel: KERNEL_CACHE_DIR (kills the +14-19 s first-request Triton compile), SGLANG_DSV41_FP8_GEMV
  (opt-in +1-2% C1), REASONING_EFFORT (prompt-prefix only).
- Speculation: DSPARK=0 / NGRAM=0 with reopen triggers (CPU-tier change; spec-consistency.py gate).
- Guards/escape hatches: ALLOW_UNTESTED_FIT, KT_ALLOW_UNSAFE_DEFERRAL, ALLOW_UNKNOWN_KNOBS,
  EXTRA_* - exist so an experiment is *recorded*, refused everywhere except PROFILE=developer.
"""


def run_calc(args) -> dict:
    """Run placement_calc.py and return its JSON output."""
    cmd = [sys.executable, str(HERE / "placement_calc.py"), args.budget,
           "--hardware", args.hardware, "--json"]
    if args.quant:
        cmd += ["--quant", args.quant]
    cmd += ["--context-tokens", str(args.context_tokens),
            "--sequences", str(args.sequences)]
    if args.map:
        cmd += ["--map", args.map]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"error: placement_calc.py failed (rc {r.returncode}): {r.stderr}", file=sys.stderr)
        sys.exit(r.returncode or 2)
    return json.loads(r.stdout)


def resolved_line(plan: dict) -> str:
    p = plan["plan"]
    return (f"experts gpu={p['gpu_experts']} cpu={p['cpu_experts']} map={p['map']} "
            f"cpu_share={p['cpu_share']*100:.0f}% deferral D={p['deferral']} chunk={p['chunk']} "
            f"kv={p['kv_gib']:.2f}GiB ({p['kv_fraction']*100:.1f}%) engram={p['engram_tier']} "
            f"trust={p['trust']}")


def tier_table(plan: dict) -> str:
    lines = ["| component | GiB | tier | reason |", "|---|---|---|---|"]
    for r in plan["tier_plan"]:
        lines.append(f"| {r['component']} | {r['gib']} | {r['tier']} | {r['reason']} |")
    return "\n".join(lines)


def readme(plan, engine, args, calc_cmd) -> str:
    p = plan["plan"]
    trust_banner = {
        "calibrated": "Calibrated: projections reproduce a measured deployment anchor.",
        "calibrated-model (cross-hardware)": "Calibrated model, cross-hardware scaling: verify with a first serve.",
        "calibrated-model (k carried from another example)": "Calibrated-model prior: k-factors carried from another model on this host/engine; the first serve is the anchor.",
        "provisional": "UNVALIDATED: provisional projections - do not promise these numbers until the first measured serve.",
    }.get(p["trust"], p["trust"])
    publish = ""
    if args.repo_url:
        publish = f"""
## Publish to GitHub

```bash
cd {args.out}
git init -b main && git add -A && git commit -m "recipe: {plan['source']} on {plan['hardware']}"
gh repo create {args.repo_url.split('/')[-2] + '/' + args.repo_url.rstrip('/').split('/')[-1]} --source=. --private --push
```

The recipe is a complete, self-contained tree: the pins in PROVENANCE.md, the profile
files, the launcher, preflight and verify scripts. Fork it, change the model pins, re-emit.
"""
    return f"""# Recipe: {plan['source']} on {plan['hardware']}

Generated by recipe-generator/emit_recipe.py v{VERSION} from placement plan
(`{calc_cmd}`). **Trust: {p['trust']}.** {trust_banner}

## Resolved line

```
resolved: {resolved_line(plan)}
```

## Tier plan

{tier_table(plan)}

## Projections

- decode C1 {p['projected']['c1_tok_s']} tok/s · C4 {p['projected']['c4_tok_s']} tok/s
- prefill cold {p['projected']['prefill_cold_tok_s']} tok/s · warm {p['projected']['prefill_warm_tok_s']}
- TTFT 16K cold {p['projected']['ttft_16k_cold_s']} s · 32K cold {p['projected']['ttft_32k_cold_s']} s
- margins: VRAM {p['vram_margin_gib']:.1f} GiB · host {p['host_margin_gib']:.1f} GiB · fits {'YES' if p['fits'] else 'NO'}
""" + (f"- {p['cold_engram_note']}\n" if p.get('cold_engram_note') else "") + f"""
## Run it

```bash
bash preflight.sh      # memory/bandwidth checks against the plan's margins
bash launch.sh         # starts the serve
bash verify.sh         # smoke: endpoint + one completion, then the gate checklist below
```

## Gate checklist (what 'qualified' means for this recipe)

- [ ] NLL identity vs a same-model baseline (delta <= +0.002, |z| <= 2)
- [ ] verify: alias, arithmetic, thinking toggle, tool call, needles at >= 95% of served context
- [ ] stress: every slot full of long prompts, decode together, peak VRAM recorded
- [ ] batch consistency (distinct prompts; same-prompt only for the consistency gate)
- [ ] determinism: temperature-0 output bit-identical run to run
- [ ] health: 0 NaN, 0 Xid, 0 OOM across the battery; power sampled and recorded
- [ ] speculation consistency (only if speculation is on - it is not, by default)

See engine-notes.md for the deep-lever surface and the advanced patch inventory.
The evidence discipline (per-serve manifest, stamped JSON) is inherited from the
recipe-factory methodology: nothing is 'measured' without a manifest.
{publish}
"""


def sglang_profiles(plan, out: Path):
    """Profile .env files from the plan's lever settings + the preset deltas."""
    d = out / "config" / "profiles"
    d.mkdir(parents=True, exist_ok=True)
    p = plan["plan"]
    qualified = f"""# Generated by recipe-generator/emit_recipe.py from the placement plan.
# registry default < this file < environment (the parent recipe's layering rule).
KT_NUM_GPU_EXPERTS={p['gpu_experts_per_layer']}
KT_CPUINFER=16
KT_MAX_DEFERRED={p['deferral']}
KT_GPU_PREFILL_MIN_TOKENS=1024
DSPARK=0
NGRAM=0
MEM_FRACTION_STATIC=0.96
CONTEXT_LENGTH=262144
MAX_RUNNING_REQUESTS=4
CHUNKED_PREFILL_SIZE={p['chunk']}
SGLANG_LOGPROB_CHUNK_SIZE=512
CUDA_GRAPH_BS=1 2 3 4
ENGRAM_FILE_PREFETCH=ahead
ENGRAM_CHUNK_AHEAD=1
ENGRAM_ADMISSION_PREFETCH=0
HOST_THREAD_CPUS=auto
SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1
KT_REJECT_NAN_ROWS=1
SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS=1
SGLANG_FLASHINFER_MOE_FUSED_FINALIZE=0
SGLANG_DSV41_SORT_DECODE_TOPK=1
SGLANG_DSV41_FP8_GEMV=0
REASONING_EFFORT=high
# KT_EXPERT_MAP must be set to a hot-expert map file (fitted_static quality) when
# experts live on the CPU tier; without it the split is checkpoint order (random).
"""
    (d / "qualified.env").write_text(qualified)
    (d / "developer.env").write_text(
        "# Escape-hatch profile: guards may be bypassed here for recorded experiments.\n"
        "PROFILE_OVERRIDES=qualified\n")


def sglang_launcher(out: Path):
    (out / "launch.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Generated launcher (sglang-hybrid engine shape). Set the pins first:\n"
        "#   HF_MODEL_DIR=<models dir> SNAPSHOT=<checkpoint revision>\n"
        "#   OVERLAY_DIR=<checkout with the patch series applied>  (see engine-notes.md)\n"
        "#   IMAGE=dsv41-kt:recipe (built by config/Dockerfile.dsv41-kt in the parent recipe)\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd -- \"$(dirname -- \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "PROFILE=\"${PROFILE:-qualified}\"\n"
        "# The parent recipe's typed registry refuses unknown/mistyped/removed/unsafe knobs\n"
        "# before launch. Point this at its scripts/serve_config.py to get that protection:\n"
        "if [[ -n \"${SERVE_CONFIG:-}\" ]]; then\n"
        "  eval \"$(python3 \"$SERVE_CONFIG\" shell)\"\n"
        "fi\n"
        "echo \"resolved profile=$PROFILE - see config/profiles/$PROFILE.env\"\n"
        "echo \"launch: docker run (image from PROVENANCE.md) with the overlay bind-mounted\"\n"
        "exit 1  # replace with the parent recipe's serve-hybrid.sh invocation for this host\n")


def llama_files(plan, out: Path, model: str):
    p = plan["plan"]
    (out / "launch.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"# Generated llama.cpp launcher. Model: {model}\n"
        "set -euo pipefail\n"
        "MODEL=\"${1:-model.gguf}\"\n"
        "exec llama-server -m \"$MODEL\" \\\n"
        "  -ngl 99 -fa 1 -c ${CONTEXT_LENGTH:-4096} --host 127.0.0.1 --port 8000\n"
        "# Offload variants if the full model does not fit (see engine-notes.md):\n"
        "#   -ot exps=CPU                  regex expert offload to CPU\n"
        "#   --n-cpu-moe N / --cpu-moe      put N MoE experts (or all) on CPU\n"
        "#   --moe-expert-cache N          LRU VRAM cache over host experts (draft PR)\n")
    (out / "config" / "profiles" / "qualified.env").parent.mkdir(parents=True, exist_ok=True)
    (out / "config" / "profiles" / "qualified.env").write_text(
        f"# llama.cpp flag profile generated from the placement plan\n"
        f"CONTEXT_LENGTH=4096\n"
        f"QUANT={model}\n"
        f"EXPERT_SPLIT=gpu={p['gpu_experts']} cpu={p['cpu_experts']} (0 cpu = -ngl 99 fits)\n")


def preflight(plan, out: Path):
    p = plan["plan"]
    (out / "preflight.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Generated preflight: refuse to serve if the host is below the plan's margins.\n"
        "set -euo pipefail\n"
        f"NEED_VRAM_GIB={max(0, p['vram_margin_gib']):.1f}   # plan's VRAM margin (free after load)\n"
        f"NEED_HOST_GIB={max(0, p['host_margin_gib']):.1f}   # plan's host margin\n"
        "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || exit 1\n"
        "free -g | awk 'NR==2{print \"host free GiB:\", $7}'\n"
        "echo \"compare against the margins above (generated from the placement plan)\"\n")


def verify_sh(out: Path):
    (out / "verify.sh").write_text(
        "#!/usr/bin/env bash\n"
        "# Generated smoke: endpoint up + one completion. The full gate suite\n"
        "# (NLL identity, needles, stress, batch consistency, health) is the\n"
        "# recipe-factory gate discipline - see README's checklist.\n"
        "set -euo pipefail\n"
        "curl -sf http://127.0.0.1:8000/v1/models\n"
        "curl -sf http://127.0.0.1:8000/v1/chat/completions \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        "  -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8}'\n")


def engine_notes(out: Path, engine: str):
    if engine == "sglang-hybrid":
        body = ("# Engine notes - sglang-hybrid (the reference recipe's engine)\n\n"
                "## The advanced patch inventory (what this project changed in SGLang)\n\n"
                "| group | patches | what | portability |\n|---|---|---|---|\n")
        for group, ids, what, port in PATCH_INVENTORY:
            body += f"| {group} | {ids} | {what} | {port} |\n"
        body += ("\nBase `da64c5cb` + 61-patch series (`patches/sglang/` in the parent recipe; "
                 "exported head `deec7a6f`, tree `adfc1b0f`); kt-kernel side is 3 patches in "
                 "`config/kt-kernel-patches/`; kernels `e13` (MXFP4 grouped GEMM), `e20` (W8A8 FP8 "
                 "configs), `t321`, `t341`. The image never contains the patches: they are "
                 "bind-mounted at run time.\n\n"
                 "## The deep-lever surface\n\n" + DEEP_LEVERS +
                 "\n\n## What ports vs stays\n\n"
                 "- Ports to a new engine/model: the W8A8 FP8 configs (PR 1), page-split seals (PR 3/4), "
                 "indexer tiling (PR 6), the overlap gate, the robustness clamps, the deterministic "
                 "top-k (upstream post ready), the expandable-segments switch.\n"
                 "- Stays recipe-local: the Engram file layout + prefetch (model-specific lookup "
                 "tables), the el8 hot-expert map file (fitted to this model's gates), the B2 UVA "
                 "residue (retired), the FP8 GEMV opt-in (until a pinned A/B clears +2%).\n"
                 "- Diagnostics (opt-in, inert by default): BCG debug, timelines, step fingerprints.\n")
    else:
        body = ("# Engine notes - llama.cpp\n\n"
                "- Placement: `-ngl/--gpu-layers`, mmap, `-ot/--override-tensor` regex per-tensor "
                "placement, `--n-cpu-moe/--cpu-moe` MoE offload, auto-fit `--fit on`.\n"
                "- Expert caching: draft `--moe-expert-cache N` LRU VRAM cache over host experts "
                "(+31% decode; LRU-64 hits 67% - temporal locality beats static hot lists).\n"
                "- KV cache: `-c` context is the whole lever; FlashAttention `-fa 1` shrinks the KV "
                "footprint; KV offload tiers do not exist here - KV is GPU or the request fails.\n"
                "- Deep-lever carry-over: CUDA-graph-like capture is internal; there is no deferral "
                "(no CPU-expert compute path on the GPU engine), no Engram, no DSpark; determinism "
                "is upstream's. The CSA2/Engram deepseek41 branch is blocked on the upstream CSA2 "
                "merge (ggml-org/llama.cpp#28696) - the parked route (T3.7).\n"
                "- Speculation: n-gram/DSpark-style drafts are not wired to the MoE offload path; "
                "treat any speculative flags as unvalidated on this engine.\n")
    (out / "engine-notes.md").write_text(body)


def provenance(plan, args, calc_cmd) -> str:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"""# Provenance (generated {stamp})

- Model: {plan['source']}  ({args.model or 'see MANIFEST.json'})
- Hardware: {plan['hardware']}
- Quantization: {plan['quantization']}
- Plan: emitted by recipe-generator/emit_recipe.py v{VERSION} from `{calc_cmd}`
- Trust: {plan['plan']['trust']}
- Calibration anchor: {plan['plan'].get('calib_note', 'none - provisional')}
- Engine pins (sglang-hybrid): base da64c5cb, overlay head deec7a6f (tree adfc1b0f),
  image dsv41-kt:recipe; see engine-notes.md for the patch inventory.
- k-factors: {plan['plan']['k']}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", nargs="?", help="placement_calc.py --json output")
    ap.add_argument("--budget", help="run placement_calc.py inline: budget JSON")
    ap.add_argument("--hardware", help="run placement_calc.py inline: hardware JSON")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--context-tokens", type=int, default=262144)
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--map", choices=["checkpoint_order", "fitted_static", "runtime_oracle"], default=None)
    ap.add_argument("--engine", required=True, choices=["sglang-hybrid", "llama-cpp"])
    ap.add_argument("--out", required=True, help="directory to write the recipe tree into")
    ap.add_argument("--model", help="model pin for PROVENANCE (e.g. org/model:quant)")
    ap.add_argument("--repo-url", help="optional GitHub repo to publish the recipe to (stamped + README section)")
    ap.add_argument("--init-git", action="store_true", help="git init + initial commit in --out (local only)")
    args = ap.parse_args()

    if args.plan:
        try:
            plan = json.loads(Path(args.plan).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: cannot read plan: {exc}", file=sys.stderr)
            return 2
        calc_cmd = f"placement_calc.py --json (from {args.plan})"
    elif args.budget and args.hardware:
        plan = run_calc(args)
        calc_cmd = f"placement_calc.py {args.budget} --hardware {args.hardware} --json"
    else:
        print("error: pass a plan JSON or --budget + --hardware", file=sys.stderr)
        return 2

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"error: --out {out} exists and is not empty; refusing to overwrite", file=sys.stderr)
        return 2
    (out / "config" / "profiles").mkdir(parents=True, exist_ok=True)

    (out / "README.md").write_text(readme(plan, args.engine, args, calc_cmd))
    (out / "MANIFEST.json").write_text(json.dumps({
        "kind": "recipe", "emitter_version": VERSION, "engine": args.engine,
        "model": args.model, "repo_url": args.repo_url,
        "plan": plan, "registry_surface": REGISTRY_SURFACE,
    }, indent=2) + "\n")
    (out / "PROVENANCE.md").write_text(provenance(plan, args, calc_cmd))
    engine_notes(out, args.engine)
    if args.engine == "sglang-hybrid":
        sglang_profiles(plan, out)
        sglang_launcher(out)
    else:
        llama_files(plan, out, args.model or plan["source"])
    preflight(plan, out)
    verify_sh(out)
    (out / "registry-surface.json").write_text(json.dumps(
        [{"name": n, "type": t, "default": d} for n, t, d in REGISTRY_SURFACE], indent=2) + "\n")
    (out / ".gitignore").write_text("*.gguf\n*.log\n__pycache__/\n")

    for sh in ("launch.sh", "preflight.sh", "verify.sh"):
        Path(out, sh).chmod(0o755)

    if args.init_git:
        r = subprocess.run(["git", "-C", str(out), "init", "-q", "-b", "main"], capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "-C", str(out), "add", "-A"], check=False)
            subprocess.run(["git", "-C", str(out), "commit", "-q", "-m",
                            f"recipe: {plan['source']} on {plan['hardware']} (emitted v{VERSION})"], check=False)

    n = len(list(out.rglob("*")))
    print(f"emitted {n} files to {out} (engine={args.engine}, trust={plan['plan']['trust']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

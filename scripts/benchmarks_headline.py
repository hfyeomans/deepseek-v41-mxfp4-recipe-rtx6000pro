#!/usr/bin/env python3
"""Generate the headline table in docs/BENCHMARKS.md from manifest-stamped evidence.

The rungs come from config/benchmarks-headline.json (one evidence directory each). Every
number in the table is read from the stamped JSON a benchmark tool wrote beside the serve's
manifest.json, and every file is refused unless its `evidence.manifest_hash` equals the
sibling manifest's content hash (the same rule as `evidence_manifest.py check-evidence`).
Nothing in the table is typed by hand.

    python3 scripts/benchmarks_headline.py markdown   # print the block
    python3 scripts/benchmarks_headline.py check      # 0 = docs/BENCHMARKS.md is current, 1 = stale
    python3 scripts/benchmarks_headline.py write      # rewrite the block in docs/BENCHMARKS.md

`scripts/ci.sh` runs `check`, so a re-qualified rung is a config edit plus `write`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evidence_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config" / "benchmarks-headline.json"
DOC = ROOT / "docs" / "BENCHMARKS.md"
BEGIN = "<!-- generated: python3 scripts/benchmarks_headline.py markdown -->"
END = "<!-- /generated: benchmarks_headline -->"

FILES = {
    "c1": "c1.json",
    "c4": "c4.json",
    "prefill16": "prefill-16k.json",
    "prefill32": "prefill-32k.json",
    "verify": "verify.json",
    "stress": "stress-kv.json",
    "pooled": "corpus/pooled.json",
}


class HeadlineError(Exception):
    pass


def _load_stamped(path: Path, manifest: dict) -> dict | None:
    """The stamped JSON at path, or None when the file is absent; refuses a bad stamp."""
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    ev = doc.get("evidence") if isinstance(doc, dict) else None
    if not isinstance(ev, dict) or ev.get("manifest_hash") is None:
        raise HeadlineError(f"{path}: not stamped against a manifest; the headline only reads stamped evidence")
    if ev["manifest_hash"] != manifest["content_hash"]:
        raise HeadlineError(f"{path}: stamp {ev['manifest_hash'][:12]} != sibling manifest {manifest['content_hash'][:12]}")
    return doc


def _decode(doc: dict | None, concurrency: int) -> str:
    if doc is None:
        return "—"
    levels = [lv for lv in doc.get("levels", []) if lv.get("concurrency") == concurrency]
    if not levels:
        return "—"
    lv = levels[0]
    tag = "" if doc.get("distinct_prompts") else " (same prompt)"
    return f"{lv['median_tps']:.1f} / {lv['median_wall_tps']:.1f}{tag}"


def _prefill(doc: dict | None) -> str:
    if doc is None:
        return "—"
    passes = doc.get("passes") or [{"pass": "cold", "sizes": doc.get("sizes", [])}]
    parts = []
    for p in passes:
        sizes = p.get("sizes") or []
        if not sizes:
            continue
        s = sizes[0]
        # R28: the cold-pass median hides the first prompt the serve ever prefilled (t339 16K:
        # 662 tok/s / 24.1 s against a 2,934 median), so the first run is shown beside it.
        runs = s.get("runs") or []
        first = ""
        if p["pass"] == "cold" and runs and runs[0].get("prefill_tps"):
            first = f"; first run {runs[0]['prefill_tps']:,.0f}, TTFT {runs[0]['ttft_s']:.1f} s"
        parts.append(f"{s['median_prefill_tps']:,.0f} {p['pass']} (TTFT {s['median_ttft_s']:.1f} s{first})")
    return " / ".join(parts) if parts else "—"


def _method(docs: dict[str, dict | None]) -> dict[str, Any]:
    """What the stamped files say the benches did (R28: the caption used to be typed by hand and
    said 5 runs / 2 warmups / 256 tokens while the smoke ran 6+3 / 1 / 512)."""
    out: dict[str, Any] = {}
    for key in ("c1", "c4"):
        d = docs.get(key)
        if d:
            out[key] = {"runs": d.get("runs"), "warmup": d.get("warmup"), "max_tokens": d.get("max_tokens"),
                        "distinct_prompts": bool(d.get("distinct_prompts"))}
    for key in ("prefill16", "prefill32"):
        d = docs.get(key)
        if d:
            out[key] = {"runs": d.get("runs")}
    pooled = docs.get("pooled")
    if pooled and pooled.get("compare"):
        out["corpus_tokens"] = pooled["compare"].get("n")
    return out


def _caption(rows: list[dict[str, Any]]) -> str:
    def distinct(values: list[Any]) -> str:
        vals = sorted({str(v) for v in values if v is not None})
        return " / ".join(vals) if vals else "?"

    methods = [r["method"] for r in rows]
    c1 = [m["c1"] for m in methods if "c1" in m]
    c4 = [m["c4"] for m in methods if "c4" in m]
    decode = c1 + c4
    runs_c1 = distinct([m["runs"] for m in c1])
    runs_c4 = distinct([m["runs"] for m in c4])
    warm = distinct([m["warmup"] for m in decode])
    tokens = distinct([m["max_tokens"] for m in decode])
    prompts = "distinct prompts" if all(m["distinct_prompts"] for m in decode) else "**same prompt in some rungs**"
    prefill_runs = distinct([m[k]["runs"] for m in methods for k in ("prefill16", "prefill32") if k in m])
    corpus = distinct([m.get("corpus_tokens") for m in methods])
    corpus = f"{int(corpus):,}" if corpus.isdigit() else corpus
    return (
        f"Decode = median per-sequence / wall tok/s over {runs_c1} runs (C1) and {runs_c4} runs (C4) of {tokens} forced "
        f"tokens after {warm} warmup, {prompts}; prefill = median tok/s of {prefill_runs} unique prompts per pass with the "
        f"client TTFT, the cold pass also naming its first run; needles = `verify.py` exact retrieval at the rung's context; "
        f"stress = `stress-kv-full.py` filling every slot; NLL = pooled {corpus}-token corpus against the T3.28 baseline "
        "(gate ΔNLL ≤ +0.002 and \\|z\\| ≤ 2 since T3.43b; \\|z\\| ≤ 2 alone before). Runs, warmups, tokens and corpus size "
        "are read from the stamped JSONs, not typed. The serve column is the manifest, not a claim."
    )


def _needles(doc: dict | None) -> str:
    if doc is None:
        return "—"
    checks = {c["name"]: c for c in doc.get("checks", [])}
    n = checks.get("multi_needle")
    if n is None:
        return "—"
    ceiling = checks.get("context_ceiling", {}).get("max_model_len")
    ctx = f", ceiling {ceiling:,}" if ceiling else ""
    return f"{n['found']}/{n['needles']} at {n['prompt_tokens']:,} tokens, TTFT {n['ttft_s']:.0f} s{ctx}"


def _stress(doc: dict | None) -> str:
    if doc is None:
        return "—"
    prompts = [r.get("prompt_tokens") for r in doc.get("results", []) if r.get("prompt_tokens")]
    ttfts = [r.get("ttft_s") for r in doc.get("results", []) if r.get("ttft_s") is not None]
    tokens = f"{median(prompts):,.0f}" if prompts else f"target {doc['prompt_tokens_target']:,}"
    done = "all finished" if doc.get("all_finished") else "**not all finished**"
    ttft = f", TTFT max {max(ttfts):.0f} s" if ttfts else ""
    return f"C{doc['concurrency']} × {tokens} tokens, wall {doc['wall_s']:.0f} s{ttft}, {done}"


def _nll(doc: dict | None) -> str:
    if doc is None:
        return "—"
    cmp = doc.get("compare")
    if not cmp:
        return f"mean {doc['mean_nll_token_weighted']:.4f} (no comparator)"
    base = cmp.get("baseline_dir", "?").replace("evidence/", "")
    return f"Δ {cmp['delta_mean_nll']:+.5f} ± {cmp.get('se_cluster', cmp['se']):.5f} (z {cmp.get('z_cluster', cmp['z_paired']):+.2f}, {cmp['tokens_changed']:,} of {cmp['n']:,} moved) vs `{base}`"


def _serve(manifest: dict) -> str:
    c = manifest.get("config", {})
    gpu_experts = c.get("KT_NUM_GPU_EXPERTS", "?")
    cpu_experts = str(384 - int(gpu_experts)) if str(gpu_experts).isdigit() else "?"
    expert_map = Path(c["KT_EXPERT_MAP"]).name if c.get("KT_EXPERT_MAP") else "checkpoint order"
    lever = "on" if c.get("SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE") == "1" else "off"
    caps = sorted({g.get("power_limit_w") for g in manifest.get("gpus", []) if g.get("power_limit_w") is not None})
    cap = ", ".join(f"{w:.0f} W" for w in caps) if caps else "?"
    return (
        f"{manifest['written_utc'][:10]}, overlay `{manifest['overlay']['short']}`, image `{manifest['image']['id'][7:19]}`, "
        f"{gpu_experts} / {cpu_experts} experts ({expert_map}), chunk {c.get('CHUNKED_PREFILL_SIZE', '?')}, "
        f"D={c.get('KT_MAX_DEFERRED', '?')}, fraction {c.get('MEM_FRACTION_STATIC', '?')}, lever {lever}, cap {cap}"
    )


def build_rows(config: dict, root: Path = ROOT) -> list[dict[str, Any]]:
    rows = []
    for entry in config["rows"]:
        d = root / entry["dir"]
        if not (d / "manifest.json").exists():
            raise HeadlineError(f"{entry['dir']}: no manifest.json")
        manifest = evidence_manifest.load(d / "manifest.json")
        docs = {k: _load_stamped(d / f, manifest) for k, f in FILES.items()}
        if all(v is None for v in docs.values()):
            raise HeadlineError(f"{entry['dir']}: none of {sorted(FILES.values())} present")
        rows.append({
            "rung": entry["rung"],
            "dir": entry["dir"],
            "note": entry.get("note", ""),
            "serve": _serve(manifest),
            "c1": _decode(docs["c1"], 1),
            "c4": _decode(docs["c4"], 4),
            "prefill16": _prefill(docs["prefill16"]),
            "prefill32": _prefill(docs["prefill32"]),
            "needles": _needles(docs["verify"]),
            "stress": _stress(docs["stress"]),
            "nll": _nll(docs["pooled"]),
            "method": _method(docs),
        })
    return rows


def markdown(config: dict | None = None, root: Path = ROOT) -> str:
    config = config or json.loads(CONFIG.read_text())
    rows = build_rows(config, root)
    out = [
        BEGIN,
        "",
        "Generated by `python3 scripts/benchmarks_headline.py write` from the stamped JSON under each evidence directory "
        "(rungs listed in `config/benchmarks-headline.json`); `scripts/ci.sh` fails when this block is stale. "
        + _caption(rows),
        "",
        "| Rung | C1 decode tok/s | C4 decode tok/s | Prefill 16K | Prefill 32K | Needles | Stress | NLL vs baseline | Evidence |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| **{r['rung']}** | {r['c1']} | {r['c4']} | {r['prefill16']} | {r['prefill32']} | {r['needles']} | "
            f"{r['stress']} | {r['nll']} | [`{r['dir']}`](../{r['dir']}/) |"
        )
    out.append("")
    for r in rows:
        note = f" — {r['note']}" if r["note"] else ""
        out.append(f"- **{r['rung']}**: {r['serve']}{note}.")
    out.append("")
    out.append(END)
    return "\n".join(out) + "\n"


def current_block(text: str) -> str | None:
    start = text.find(BEGIN)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        return None
    return text[start:end + len(END)] + "\n"


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "markdown"
    try:
        block = markdown()
    except (HeadlineError, evidence_manifest.ManifestError, KeyError, ValueError) as e:
        print(f"benchmarks_headline: {e}", file=sys.stderr)
        return 2
    if cmd == "markdown":
        sys.stdout.write(block)
        return 0
    text = DOC.read_text()
    existing = current_block(text)
    if cmd == "check":
        if existing is None:
            print(f"FAIL {DOC.relative_to(ROOT)} has no generated headline block (markers {BEGIN!r} … {END!r})")
            return 1
        if existing != block:
            print(f"FAIL {DOC.relative_to(ROOT)} headline table is stale: regenerate with python3 scripts/benchmarks_headline.py write")
            return 1
        print("ok benchmarks headline is current")
        return 0
    if cmd == "write":
        if existing is None:
            print(f"benchmarks_headline: {DOC.relative_to(ROOT)} has no generated block to replace", file=sys.stderr)
            return 2
        DOC.write_text(text.replace(existing, block))
        print(f"wrote {DOC.relative_to(ROOT)}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

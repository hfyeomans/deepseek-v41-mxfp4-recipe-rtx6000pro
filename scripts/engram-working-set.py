#!/usr/bin/env python3
"""Engram page-cache working set of a traffic mix, computed without a GPU (T3.10, issue #2).

The Engram tables live in two 94.6 GiB checkpoint shards that the E18 `file` layout mmaps
(overlay `engram.py` `_FileTable`): the page cache is the row cache, and the only question
tiering can answer is whether the pages real traffic touches fit in what the kt CPU experts
leave of host memory. This replays a traffic mix through the hasher's CPU fallback
(`compute_engram_hash_ids`, the same token map / multipliers / primes as the serve) and reports,
per Engram layer:

  lookups            rows gathered = tokens x (n-gram sizes - 1) x heads (24 per token here)
  distinct rows      unique table rows, split by n-gram column (2-gram, 3-gram, 4-gram)
  distinct pages     unique 4 KiB pages of weight (256 B rows, 16 per page) and scale (8 B rows,
                     512 per page) bytes, i.e. the page-cache footprint the traffic needs
  per-request cold   pages first touched by each request, in traffic order
  LRU hit rate       request-granularity LRU of weight+scale pages at a few cache sizes (a page
                     hits when it was touched in the most recent requests whose union fits)

Traffic sources (in order, the replay keeps that order):
  --replay-smoke CONTEXT SLOTS   the release-smoke --full traffic: stress SLOTS x (CONTEXT-2048)
                                 seed 31, same-prompt batch, NLL corpus, needles at CONTEXT, decode
                                 C1/C<SLOTS> fixtures, prefill 16K/32K x 3 seed 777 (same builders)
  --texts PATH...                real text: files or directories (.md .py .sh .txt .json .yaml),
                                 one request per file (a coding agent's context is the repo)
  --repeat N                     replay --texts N times (steady state of repeated agent traffic)

Runs inside the recipe image (needs sglang, tokenizers, sympy; CPU only):
  docker run --rm -v ~/dev/src/sglang-dsv41-overlay/python/sglang:/sgl-workspace/sglang/python/sglang:ro \\
    -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro -v "$PWD":/repo:ro \\
    --entrypoint python dsv41-kt:recipe /repo/scripts/engram-working-set.py --model-dir /hfm/snapshots/<SNAP> \\
    --replay-smoke 262144 4 --output /tmp/ws.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

PAGE = 4096
GIB = 1 << 30
TEXT_SUFFIXES = {".md", ".py", ".sh", ".txt", ".json", ".yaml", ".yml", ".toml", ".cu", ".h", ".c", ".cpp"}


def _load_script(name: str):
    """Import a sibling script with a dash in its name (its argparse main is not run)."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def smoke_requests(context: int, slots: int, corpus_dir: Path) -> list[tuple[str, str]]:
    """(label, prompt) in the order release-smoke.sh --full sends them."""
    prefill = _load_script("bench-prefill")
    stress = _load_script("stress-kv-full")
    verify = _load_script("verify")
    decode = _load_script("bench-decode")
    reqs: list[tuple[str, str]] = []
    n = context - 2048
    for i in range(slots):
        reqs.append((f"stress-{i}", stress.make_prompt(n, random.Random(31 + i))))
    reqs.append(("same-prompt", decode.FIXTURE))
    for f in sorted(corpus_dir.glob("*.txt")):
        reqs.append((f"corpus-{f.stem}", f.read_text()))
    prompt, _ = verify.build_needle_prompt(context, 6, random.Random(20260920))
    reqs.append((f"needles-{context}", prompt))
    fixtures = decode.DISTINCT_FIXTURES
    for c in (1, slots):
        for i in range(c):
            reqs.append((f"decode-c{c}-{i}", fixtures[(6 + i) % len(fixtures)]))
    for size in (16384, 32768):
        for i in range(3):
            reqs.append((f"prefill-{size}-{i}", prefill.make_prompt(size, random.Random(777 + 1000 * size + i))))
    return reqs


def text_requests(paths: list[str], repeat: int) -> list[tuple[str, str]]:
    files: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(f for f in pp.rglob("*") if f.is_file() and f.suffix in TEXT_SUFFIXES))
        elif pp.is_file():
            files.append(pp)
    out = []
    for r in range(repeat):
        for f in files:
            try:
                out.append((f"text-{r}-{f.name}", f.read_text(errors="replace")))
            except OSError:
                continue
    return out


class Hasher:
    """The serve's hash function on the CPU: token ids -> [T, layers, 24] table rows."""

    def __init__(self, model_dir: Path):
        from transformers import AutoConfig, AutoTokenizer

        from sglang.srt.layers.engram import (
            EngramLayout,
            build_compressed_token_map,
            compute_hash_multipliers,
        )

        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        layout = EngramLayout.build(
            layer_ids=tuple(cfg.engram_layer_ids),
            num_embeddings=tuple(cfg.engram_num_embeddings),
            max_ngram_size=cfg.engram_max_ngram_size,
            n_heads=cfg.engram_n_heads,
            head_dim=cfg.engram_head_dim,
            vocab_size=cfg.engram_vocab_size,
        )
        token_map, vocab = build_compressed_token_map(self.tok)
        assert vocab == cfg.engram_compressed_vocab_size, (vocab, cfg.engram_compressed_vocab_size)
        flat = [[p for per_ngram in layer for p in per_ngram] for layer in layout.primes]
        self.layout = layout
        self.n = layout.max_ngram_size
        self.heads = layout.n_heads
        self.dim = layout.head_dim
        self.pad_id = token_map[cfg.engram_pad_token_id]
        self.token_map = torch.tensor(token_map)
        self.multipliers = compute_hash_multipliers(layout.layer_ids, self.n, vocab)
        self.primes = torch.tensor(layout.primes)
        self.offsets = torch.tensor(np.array([np.cumsum([0, *s[:-1]]) for s in flat]))
        self.num_embeddings = list(cfg.engram_num_embeddings)
        self.layer_ids = list(cfg.engram_layer_ids)
        # Byte offsets of each layer's weight / scale blocks inside its shard, for page ids
        # that match what the serve's _FileTable._pages() computes.
        self.w_off, self.s_off, self.shards = [], [], []
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
        for lid in self.layer_ids:
            shard = index[f"layers.{lid}.engram.embed.weight"]
            with open(model_dir / shard, "rb") as fh:
                hn = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(hn))
            base = 8 + hn
            self.w_off.append(base + header[f"layers.{lid}.engram.embed.weight"]["data_offsets"][0])
            self.s_off.append(base + header[f"layers.{lid}.engram.embed.scale"]["data_offsets"][0])
            self.shards.append(shard)

    def ids(self, text: str) -> torch.Tensor:
        from sglang.srt.layers.engram import compute_engram_hash_ids

        toks = torch.tensor(self.tok(text, add_special_tokens=True)["input_ids"], dtype=torch.int64)
        t = toks.shape[0]
        shifts = torch.arange(self.n)
        pos = torch.arange(t)
        tokens = toks[(pos.unsqueeze(-1) - shifts).clamp_min(0)]
        blocked = pos.unsqueeze(-1) < shifts
        return compute_engram_hash_ids(
            tokens, blocked, self.pad_id, self.token_map, self.multipliers, self.primes, self.offsets
        )

    def pages(self, layer: int, rows: torch.Tensor) -> torch.Tensor:
        """Distinct page ids (weight pages, then scale pages, disjoint ranges) the rows touch."""
        w = (self.w_off[layer] + rows * self.dim) // PAGE
        s = (self.s_off[layer] + rows * (self.dim // 32)) // PAGE
        return torch.unique(torch.cat([w, s]))


def lru_request_hits(touched: list[torch.Tensor], capacities_pages: list[int]) -> dict[str, dict]:
    """Request-granularity LRU over page sets: for each request, a page hits if it is in the
    union of the most recent earlier requests whose union still fits the capacity."""
    out = {}
    for cap in capacities_pages:
        hits = misses = 0
        for i, pages in enumerate(touched):
            cache = []
            size = 0
            for prev in reversed(touched[:i]):
                size += prev.numel()
                if size > cap and cache:
                    break
                cache.append(prev)
                if size > cap:
                    break
            if cache:
                resident = torch.unique(torch.cat(cache))
                hit = torch.isin(pages, resident).sum().item()
            else:
                hit = 0
            hits += hit
            misses += pages.numel() - hit
        out[str(cap * PAGE // GIB)] = {
            "cache_gib": cap * PAGE // GIB,
            "page_hits": hits,
            "page_misses": misses,
            "hit_rate": hits / max(1, hits + misses),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True, help="checkpoint snapshot (config, tokenizer, index)")
    ap.add_argument("--replay-smoke", nargs=2, type=int, metavar=("CONTEXT", "SLOTS"))
    ap.add_argument("--corpus", type=Path, default=Path(__file__).resolve().parent / "fixtures" / "nll-corpus")
    ap.add_argument("--texts", nargs="*", default=[])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--cache-gib", default="16,32,64,86", help="LRU capacities to simulate")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    t0 = time.time()
    h = Hasher(args.model_dir)
    print(f"hasher ready in {time.time() - t0:.1f}s: layers {h.layer_ids}, rows/layer {h.num_embeddings}, "
          f"{(h.n - 1) * h.heads} rows per token per layer, shards {h.shards}")
    reqs: list[tuple[str, str]] = []
    if args.replay_smoke:
        reqs += smoke_requests(args.replay_smoke[0], args.replay_smoke[1], args.corpus)
    reqs += text_requests(args.texts, args.repeat)
    if not reqs:
        print("nothing to replay: pass --replay-smoke and/or --texts", file=sys.stderr)
        return 2

    n_layers = len(h.layer_ids)
    cols = (h.n - 1) * h.heads
    all_rows: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    touched: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    seen_pages: list[set] = [set() for _ in range(n_layers)]
    per_request = []
    total_tokens = 0
    for label, text in reqs:
        t1 = time.time()
        ids = h.ids(text)  # [T, layers, cols]
        t = ids.shape[0]
        total_tokens += t
        row = {"request": label, "tokens": t, "layers": []}
        for li in range(n_layers):
            rows = ids[:, li, :].reshape(-1)
            all_rows[li].append(rows)
            pages = h.pages(li, rows)
            touched[li].append(pages)
            before = len(seen_pages[li])
            seen_pages[li].update(pages.tolist())
            new = len(seen_pages[li]) - before
            row["layers"].append({"distinct_rows": int(torch.unique(rows).numel()), "distinct_pages": int(pages.numel()),
                                  "new_pages": new, "cumulative_pages": len(seen_pages[li])})
        per_request.append(row)
        l0 = row["layers"][0]
        print(f"{label:>28} {t:>8} tok  L{h.layer_ids[0]}: rows {l0['distinct_rows']:>9}  pages {l0['distinct_pages']:>9}  "
              f"new {l0['new_pages']:>9}  cum {l0['cumulative_pages']:>10} ({l0['cumulative_pages'] * PAGE / GIB:6.2f} GiB)  "
              f"{time.time() - t1:4.1f}s")

    caps = [int(float(c) * GIB) // PAGE for c in args.cache_gib.split(",")]
    layers_out = []
    for li in range(n_layers):
        rows = torch.cat(all_rows[li])
        lookups = rows.numel()
        uniq = torch.unique(rows)
        by_col = []
        flat_rows = torch.cat([r.view(-1, cols) for r in all_rows[li]])
        for g in range(h.n - 1):
            col = flat_rows[:, g * h.heads:(g + 1) * h.heads].reshape(-1)
            by_col.append({"ngram": g + 2, "lookups": int(col.numel()), "distinct_rows": int(torch.unique(col).numel())})
        pages = h.pages(li, uniq)
        w_pages = int((pages < (h.s_off[li] // PAGE)).sum().item())
        layers_out.append({
            "layer_id": h.layer_ids[li],
            "shard": h.shards[li],
            "table_rows": h.num_embeddings[li],
            "lookups": lookups,
            "distinct_rows": int(uniq.numel()),
            "distinct_rows_by_ngram": by_col,
            "distinct_pages": int(pages.numel()),
            "weight_pages": w_pages,
            "scale_pages": int(pages.numel()) - w_pages,
            "footprint_gib": pages.numel() * PAGE / GIB,
            "rows_per_page_touched": uniq.numel() / max(1, w_pages),
            "lru_request_level": lru_request_hits(touched[li], caps),
        })
    report = {
        "model_dir": str(args.model_dir),
        "requests": len(reqs),
        "tokens": total_tokens,
        "rows_per_token_per_layer": cols,
        "page_bytes": PAGE,
        "layers": layers_out,
        "total_footprint_gib": sum(l["footprint_gib"] for l in layers_out),
        "per_request": per_request,
        "elapsed_s": time.time() - t0,
    }
    print()
    for l in layers_out:
        print(f"layer {l['layer_id']} ({l['shard']}): {l['lookups']:,} lookups, {l['distinct_rows']:,} distinct rows "
              f"({100 * l['distinct_rows'] / l['lookups']:.1f} %), pages {l['distinct_pages']:,} "
              f"(weight {l['weight_pages']:,}, scale {l['scale_pages']:,}) = {l['footprint_gib']:.2f} GiB, "
              f"{l['rows_per_page_touched']:.2f} rows per touched weight page")
        for c in l["distinct_rows_by_ngram"]:
            print(f"    {c['ngram']}-gram columns: {c['distinct_rows']:,} distinct of {c['lookups']:,}")
        for k, v in l["lru_request_level"].items():
            print(f"    LRU {k:>3} GiB: hit rate {100 * v['hit_rate']:.1f} % ({v['page_hits']:,} / {v['page_hits'] + v['page_misses']:,} page touches)")
    print(f"total footprint both layers {report['total_footprint_gib']:.2f} GiB for {total_tokens:,} tokens in {len(reqs)} requests "
          f"({report['total_footprint_gib'] * GIB / max(1, total_tokens) / 1024:.1f} KiB per token); {report['elapsed_s']:.0f}s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

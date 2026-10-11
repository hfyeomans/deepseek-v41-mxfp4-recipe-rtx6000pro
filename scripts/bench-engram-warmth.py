#!/usr/bin/env python3
"""Cold vs warm Engram page cache: how much prefill time the NVMe row fetch costs (T3.10 / T3.47).

For each prompt the server prefills it three times with the radix cache flushed in between, so
every pass re-prefills the whole prompt and the only thing that changes between pass 1 and
passes 2-3 is whether the Engram rows the prompt hashes to are already in the page cache:

  cold   first contact: rows the traffic never touched are read from the NVMe shard
  warm   same prompt again: every page is resident (engram-working-set.py says how many pages)

`fincore` on the two Engram shards before and after each pass gives the pages actually loaded.
Two prompt kinds: real text (files concatenated to about --tokens; a coding agent's context)
and the synthetic 30-word prompt the release smoke uses (bench-prefill.py's builder), whose
n-grams repeat so most of its rows are resident after any earlier smoke traffic.

  python3 scripts/bench-engram-warmth.py --url http://127.0.0.1:8000 --tokens 16384 32768 \\
    --texts docs/*.md --output evidence/t310/warmth/warmth.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import engram_residency, load, locate, stamp_or_exit  # noqa: E402
from openai_client import chat_stream, get_json  # noqa: E402

CHARS_PER_TOKEN = 3.2  # markdown / code at this tokenizer (nll-corpus README: 3.1-3.4)


def _load_script(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RealText:
    """Disjoint slices of the concatenated files, about `tokens` tokens each, handed out in order
    so no two prompts share text (a shared slice would warm the second prompt's rows)."""

    def __init__(self, files: list[Path]):
        self.blob = "\n\n".join(f.read_text(errors="replace") for f in files)
        self.cursor = 0

    def take(self, tokens: int) -> str:
        need = int(tokens * CHARS_PER_TOKEN)
        if self.cursor + need > len(self.blob):
            raise SystemExit(f"--texts hold {len(self.blob)} chars; ran out at {self.cursor} needing {need} more")
        s = self.blob[self.cursor:self.cursor + need]
        self.cursor += need
        return s + "\n\nReply with the single word DONE."


def residency() -> dict[str, int]:
    path = locate()
    cfg = load(path)["config"]
    return {r["shard"]: r["resident_bytes"] for r in engram_residency(cfg["HF_MODEL_DIR"], cfg["SNAPSHOT"])}


def one(url: str, model: str, prompt: str, flush) -> dict:
    flush(url)
    before = residency()
    r = chat_stream(url, model, [{"role": "user", "content": prompt}], max_tokens=8, temperature=0.0, seed=0)
    after = residency()
    loaded = sum(after.values()) - sum(before.values())
    return {"prompt_tokens": r.prompt_tokens, "ttft_s": round(r.ttft_s, 3),
            "prefill_tps": round(r.prompt_tokens / r.ttft_s, 1) if r.prompt_tokens else None,
            "pages_loaded": loaded // 4096, "loaded_gib": round(loaded / 2**30, 3),
            "resident_gib_after": round(sum(after.values()) / 2**30, 2)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", nargs="+", type=int, default=[16384, 32768])
    ap.add_argument("--texts", nargs="+", type=Path, required=True, help="real-text files (concatenated)")
    ap.add_argument("--prompts", type=int, default=2, help="distinct real-text prompts per size")
    ap.add_argument("--passes", type=int, default=3, help="prefills per prompt (first is cold)")
    ap.add_argument("--seed", type=int, default=9001, help="synthetic prompt seed (fresh vs the smoke's 777)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    flush = _load_script("verify-batch-consistency").flush_cache
    prefill = _load_script("bench-prefill")
    model = args.model or get_json(args.url + "/v1/models")["data"][0]["id"]
    files = [f for f in args.texts if f.is_file()]
    text = RealText(files)
    rows = []
    for tokens in args.tokens:
        prompts = [(f"real-{tokens}-{i}", text.take(tokens)) for i in range(args.prompts)]
        prompts.append((f"synthetic-{tokens}", prefill.make_prompt(tokens, random.Random(args.seed + tokens))))
        for label, prompt in prompts:
            passes = []
            for p in range(args.passes):
                res = one(args.url, model, prompt, flush)
                res["pass"] = "cold" if p == 0 else f"warm{p}"
                passes.append(res)
                print(f"{label:>16} {res['pass']:>5}: {res['prompt_tokens']:>6} tok, TTFT {res['ttft_s']:7.3f} s, "
                      f"{res['prefill_tps']:8.1f} tok/s, loaded {res['pages_loaded']:>8} pages ({res['loaded_gib']:.3f} GiB), "
                      f"resident {res['resident_gib_after']:.2f} GiB")
            cold, warm = passes[0], passes[1:]
            best_warm = min(w["ttft_s"] for w in warm) if warm else None
            rows.append({"label": label, "tokens": tokens, "passes": passes,
                         "cold_minus_warm_s": round(cold["ttft_s"] - best_warm, 3) if best_warm else None})
            if best_warm:
                print(f"{'':>16}   cold - warm = {cold['ttft_s'] - best_warm:+.3f} s "
                      f"({100 * (cold['ttft_s'] - best_warm) / best_warm:+.1f} % of warm TTFT)")
    report = {"url": args.url, "model": model, "texts": [str(f) for f in files], "passes": args.passes, "rows": rows}
    if args.output:
        stamp_or_exit(report, args.output, "bench-engram-warmth")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

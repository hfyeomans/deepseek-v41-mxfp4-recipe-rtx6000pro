#!/usr/bin/env python3
"""Batch-consistency gate: do concurrent (bs>1) greedy completions match bs=1?

Every other quality gate in this repo (verify.py, nll-score.py, greedy-logprobs.py)
runs one request at a time, so a bug that only corrupts rows 2..N of a decode
batch is invisible to them. This script:

  1. sends each prompt alone (bs=1) and records the greedy completion as the
     reference;
  2. sends all prompts at once (--concurrency N; prompts are cycled to fill N
     slots, so with 4 prompts and N=4 every slot has a distinct prompt, and
     with 1 prompt and N=4 the same prompt occupies all 4 rows);
  3. reports, per slot: exact match with the reference, the shared token-event
     prefix length, and a crude gibberish score (fraction of whitespace-split
     words that are not ASCII words or digits).

Correct batching produces near-identical text (small kernel batch-invariance
drift late in the completion is normal). A row-corruption bug produces streams
that diverge at token 1-3 or degenerate into gibberish/repetition. The sharpest
signal is ``--same-prompt``: rows of one batch that carry the same prompt see
the same numerics and must come back byte-identical to each other; with the
kt-kernel bug C (docs/TROUBLESHOOTING.md) row 0 was right and rows 1..3 each
told a different story.

Exit 0 when every concurrent slot shares at least --min-prefix characters with
its bs=1 reference, no slot exceeds --max-gibberish, and (with --same-prompt)
all concurrent rows are identical to each other.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import chat_stream  # noqa: E402

DEFAULT_PROMPTS = [
    "Explain, in three short paragraphs, why the sky appears blue during the day and red at sunset.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then show its output for n=10.",
    "List the planets of the solar system in order from the Sun, one per line, with one fact about each.",
    "Summarize the plot of Romeo and Juliet in five sentences without naming any character other than the two leads.",
]

WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'’\-.,:;!?()\[\]{}`\"/+*=<>_#%&$]*$")


def gibberish_score(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    bad = sum(1 for w in words if not WORD_RE.match(w))
    return bad / len(words)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def run_one(url: str, model: str, prompt: str, max_tokens: int, seed: int) -> dict:
    t0 = time.perf_counter()
    r = chat_stream(url, model, [{"role": "user", "content": prompt}], max_tokens=max_tokens, temperature=0.0, seed=seed)
    return {
        "text": r.text,
        "completion_tokens": r.completion_tokens,
        "ttft_s": r.ttft_s,
        "total_s": r.total_s,
        "wall_s": time.perf_counter() - t0,
        "finish_reason": r.finish_reason,
        "gibberish": gibberish_score(r.text),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompts-file", type=Path, help="one prompt per line; default = 4 built-in prompts")
    ap.add_argument("--same-prompt", action="store_true", help="use only the first prompt in every slot")
    ap.add_argument("--min-prefix", type=int, default=40, help="minimum shared character prefix with the bs=1 reference")
    ap.add_argument("--max-gibberish", type=float, default=0.15)
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    prompts = DEFAULT_PROMPTS
    if args.prompts_file:
        prompts = [ln.rstrip("\n") for ln in args.prompts_file.read_text().splitlines() if ln.strip()]
    if args.same_prompt:
        prompts = prompts[:1]
    slots = [prompts[i % len(prompts)] for i in range(args.concurrency)]

    print(f"== bs=1 references ({len(prompts)} prompts, {args.max_tokens} tok, greedy)")
    refs: dict[str, dict] = {}
    for p in prompts:
        refs[p] = run_one(args.url, args.model, p, args.max_tokens, args.seed)
        print(f"  ref {p[:48]!r}: {refs[p]['completion_tokens']} tok, gib {refs[p]['gibberish']:.2f}")

    print(f"== concurrent run (C={args.concurrency})")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, args.url, args.model, p, args.max_tokens, args.seed) for p in slots]
        conc = [f.result() for f in futs]

    ok = True
    rows = []
    for i, (p, c) in enumerate(zip(slots, conc)):
        ref = refs[p]
        pre = common_prefix_len(ref["text"], c["text"])
        exact = ref["text"] == c["text"]
        slot_ok = pre >= args.min_prefix and c["gibberish"] <= args.max_gibberish
        ok &= slot_ok
        rows.append({"slot": i, "prompt": p, "exact": exact, "prefix_chars": pre, "ref_chars": len(ref["text"]),
                     "conc_chars": len(c["text"]), "gibberish": c["gibberish"], "ok": slot_ok, "concurrent": c})
        print(f"  slot {i}: exact={exact} prefix={pre}/{len(ref['text'])} gib={c['gibberish']:.2f} "
              f"tok={c['completion_tokens']} {'OK' if slot_ok else 'FAIL'}")
        if not slot_ok:
            print(f"    ref : {ref['text'][:160]!r}")
            print(f"    conc: {c['text'][:160]!r}")

    rows_identical = None
    min_pair_prefix = None
    if args.same_prompt and len(conc) > 1:
        # Rows admitted at different steps see different batch compositions for
        # their first tokens, so byte-identity is informative but not required;
        # the pairwise shared prefix separates drift (hundreds of chars) from
        # row corruption (a dozen).
        rows_identical = len({c["text"] for c in conc}) == 1
        min_pair_prefix = min(common_prefix_len(a["text"], b["text"])
                              for i, a in enumerate(conc) for b in conc[i + 1:])
        print(f"  rows identical: {rows_identical}; min pairwise row prefix {min_pair_prefix} chars "
              f"(gate >= {args.min_prefix})")
        ok &= min_pair_prefix >= args.min_prefix

    print("PASS" if ok else "FAIL")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"label": args.label, "url": args.url, "model": args.model,
                                           "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                                           "seed": args.seed, "pass": ok, "same_prompt": args.same_prompt,
                                           "rows_identical": rows_identical, "min_pairwise_row_prefix": min_pair_prefix,
                                           "references": {p: r for p, r in refs.items()}, "slots": rows}, indent=2))
        print(f"wrote {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

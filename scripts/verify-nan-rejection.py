#!/usr/bin/env python3
"""C10 gate (T3.13): a NaN in one request's forward fails that request, not the server.

Run against a serve launched with the injection hook armed, e.g.

    KT_NAN_INJECT=row=1,bs=4,step=3,layer=20 bash scripts/serve-hybrid.sh

The hook (overlay kt_nan_inject.py) adds NaN to row 1's hidden state after MoE
layer 20 on rank 0's third bs=4 decode replay. This script then:

  1. records a bs=1 greedy reference for each of the N prompts;
  2. sends the N prompts concurrently (non-streaming, so an abort is an HTTP
     status, not a chunk) and expects exactly --expect-aborted of them to come
     back HTTP 500 whose detail names non-finite logits, every other one to
     come back 200, run to max_tokens (or a stop), and read no worse than its
     bs=1 reference (gibberish score within --gibberish-slack of the
     reference's own; the shared prefix is reported, not gated: distinct
     prompts at bs=4 already drift from bs=1 within a few tokens on this
     stack, see verify-batch-consistency.py, and the batch shrinks to N-1
     rows mid-decode here);
  3. sends the same N prompts concurrently again and expects N x 200 under the
     same checks: the hook fires once, the server is still up, the slot of the
     aborted request was released.

Exit 0 when all three hold. --output writes the stamped report (manifest check
like every other gate).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers  # noqa: E402

DEFAULT_PROMPTS = [
    "Explain, in three short paragraphs, why the sky appears blue during the day and red at sunset.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, then show its output for n=10.",
    "List the planets of the solar system in order from the Sun, one per line, with one fact about each.",
    "Summarize the plot of Romeo and Juliet in five sentences without naming any character other than the two leads.",
]
WORD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9'’\-.,:;!?()\[\]{}`\"/+*=<>_#%&$]*$")
ABORT_MARK = "non-finite logits"


def gibberish_score(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    return sum(1 for w in words if not WORD_RE.match(w)) / len(words)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def chat_once(url: str, model: str, prompt: str, max_tokens: int, seed: int, timeout: float = 600.0) -> dict:
    """One non-streaming chat completion; HTTP errors are returned, not raised."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(f"{url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth_headers()}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            doc = json.loads(resp.read())
            choice = doc["choices"][0]
            text = choice["message"].get("content") or ""
            return {"status": resp.status, "text": text, "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": doc.get("usage", {}).get("completion_tokens"),
                    "gibberish": gibberish_score(text), "wall_s": time.perf_counter() - t0}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        return {"status": e.code, "text": "", "finish_reason": None, "completion_tokens": None,
                "gibberish": None, "error": detail[:600], "wall_s": time.perf_counter() - t0}


def health(url: str) -> int:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def run_group(url: str, model: str, prompts: list[str], max_tokens: int, seed: int) -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return [f.result() for f in [ex.submit(chat_once, url, model, p, max_tokens, seed) for p in prompts]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect-aborted", type=int, default=1, help="requests the hook must fail in the injected group")
    ap.add_argument("--gibberish-slack", type=float, default=0.05,
                    help="a surviving row may score at most this much more gibberish than its bs=1 reference")
    ap.add_argument("--label", default="")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(args.concurrency)]
    ok = True

    def survived(r: dict, ref: dict) -> bool:
        ran_out = r["completion_tokens"] == args.max_tokens or r["finish_reason"] == "stop"
        return (r["status"] == 200 and ran_out
                and (r["gibberish"] or 0.0) <= (ref["gibberish"] or 0.0) + args.gibberish_slack)

    print(f"== bs=1 references ({len(prompts)} prompts, {args.max_tokens} tok, greedy)")
    refs = []
    for p in prompts:
        r = chat_once(args.url, args.model, p, args.max_tokens, args.seed)
        refs.append(r)
        print(f"  ref {p[:48]!r}: HTTP {r['status']} {r['completion_tokens']} tok")
        ok &= r["status"] == 200

    print(f"== injected group (C={args.concurrency}; expect {args.expect_aborted} aborted)")
    injected = run_group(args.url, args.model, prompts, args.max_tokens, args.seed)
    aborted = [i for i, r in enumerate(injected) if r["status"] == 500 and ABORT_MARK in r.get("error", "")]
    for i, (p, r, ref) in enumerate(zip(prompts, injected, refs)):
        if i in aborted:
            print(f"  slot {i}: ABORTED HTTP 500 {r['error'][:120]!r}")
            continue
        pre = common_prefix_len(ref["text"], r["text"])
        slot_ok = survived(r, ref)
        ok &= slot_ok
        print(f"  slot {i}: HTTP {r['status']} prefix={pre}/{len(ref['text'])} gib={r['gibberish']:.2f} "
              f"(ref {ref['gibberish']:.2f}) tok={r['completion_tokens']} {'OK' if slot_ok else 'FAIL'}")
        if not slot_ok:
            print(f"    ref : {ref['text'][:160]!r}")
            print(f"    got : {(r.get('text') or r.get('error', ''))[:160]!r}")
    if len(aborted) != args.expect_aborted:
        ok = False
        print(f"  FAIL: {len(aborted)} aborted (expected {args.expect_aborted}); statuses "
              f"{[r['status'] for r in injected]}")

    print(f"== clean group (C={args.concurrency}; expect all 200)")
    clean = run_group(args.url, args.model, prompts, args.max_tokens, args.seed)
    for i, (r, ref) in enumerate(zip(clean, refs)):
        pre = common_prefix_len(ref["text"], r["text"])
        slot_ok = survived(r, ref)
        ok &= slot_ok
        print(f"  slot {i}: HTTP {r['status']} prefix={pre}/{len(ref['text'])} gib={(r['gibberish'] or 0.0):.2f} "
              f"(ref {ref['gibberish']:.2f}) tok={r['completion_tokens']} {'OK' if slot_ok else 'FAIL'}")
    h = health(args.url)
    print(f"== /health after: HTTP {h}")
    ok &= h == 200

    print("PASS" if ok else "FAIL")
    if args.output:
        report = {"label": args.label, "url": args.url, "model": args.model, "concurrency": args.concurrency,
                  "max_tokens": args.max_tokens, "seed": args.seed, "expect_aborted": args.expect_aborted,
                  "gibberish_slack": args.gibberish_slack, "pass": ok, "aborted_slots": aborted, "health_after": h,
                  "references": refs, "injected": injected, "clean": clean}
        stamp_or_exit(report, args.output, "verify-nan-rejection.py")
        args.output.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

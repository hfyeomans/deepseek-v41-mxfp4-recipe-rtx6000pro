#!/usr/bin/env python3
"""Functional acceptance for a DeepSeek-V4/V4.1 OpenAI-compatible endpoint.

Checks, in order:
  1. /v1/models lists the expected alias and reports max_model_len >= --context
  2. exact arithmetic sanity (17*19=323, the vLLM recipe's smoke test), thinking off
  3. thinking on produces reasoning_content or a longer answer (mode toggle works)
  4. structured tool call round-trip
  5. multi-needle retrieval at --context tokens: N records planted at spread
     depths in synthetic filler; all must be recalled exactly, and the server's
     prompt_tokens must be >= 95 % of --context (the prompt targets 97.5 %, leaving
     room for the answer under the ceiling)

Writes a JSON report with --output. Exit 0 only if every check passes.

Fill calibration (R11, T3.43b): the filler assumed 1.3 tokens per word, but the
20-word vocabulary tokenises at 1.10-1.105 on DeepSeek's tokenizer, so before
2026-10-11 every needle prompt reached only ~84.5 % of --context (55,206 /
65,536; 221,756 / 262,144; 887,387 / 1,048,576) while being reported as "at the
configured context". The server's prompt_tokens is now asserted, not assumed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers, chat_stream, get_json  # noqa: E402

FILLER_WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit", "timber", "quartz",
                "velvet", "canyon", "pixel", "saffron", "anchor", "glacier", "marble", "cobalt", "ember", "willow")
TOKENS_PER_WORD = 1.10     # server prompt_tokens / prompt words = 1.100-1.105 on every committed needle run (64K, 262K, 1M)
NEEDLE_DEPTHS = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
PROMPT_TARGET_FRACTION = 0.975   # of --context; the answer (64 tokens per needle) must fit under the ceiling
PROMPT_MIN_FRACTION = 0.95       # gate: server prompt_tokens must reach this


def filler(n_tokens: int, rng: random.Random) -> str:
    words = [rng.choice(FILLER_WORDS) for _ in range(int(n_tokens / TOKENS_PER_WORD))]
    return " ".join(words)


def build_needle_prompt(context_tokens: int, needles: int, rng: random.Random) -> tuple[str, dict[str, str]]:
    keys = {f"record-{rng.randrange(10**6):06d}": f"{rng.randrange(10**8):08d}" for _ in range(needles)}
    depths = NEEDLE_DEPTHS[:needles]
    body_tokens = int(context_tokens * PROMPT_TARGET_FRACTION) - 600
    segments = []
    prev = 0.0
    for (k, v), d in zip(keys.items(), depths):
        segments.append(filler(int(body_tokens * (d - prev)), rng))
        segments.append(f"\nThe access code for {k} is {v}.\n")
        prev = d
    segments.append(filler(int(body_tokens * (1.0 - prev)), rng))
    question = "\n\nList the access code for each of these records, one per line as 'record: code', and nothing else: " + ", ".join(keys)
    return "".join(segments) + question, keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default=None)
    ap.add_argument("--context", type=int, default=262144, help="needle prompt size in tokens (approximate; server usage is recorded)")
    ap.add_argument("--needles", type=int, default=6, help=f"1..{len(NEEDLE_DEPTHS)} records at fixed depths")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--skip-needle", action="store_true")
    args = ap.parse_args()
    if not 1 <= args.needles <= len(NEEDLE_DEPTHS):
        ap.error(f"--needles must be 1..{len(NEEDLE_DEPTHS)} (only {len(NEEDLE_DEPTHS)} depths are planted; "
                 f"a larger count could never be found and 0 would pass vacuously)")

    report: dict = {"url": args.url, "checks": []}
    ok_all = True

    def record(name: str, ok: bool, **detail):
        nonlocal ok_all
        ok_all &= ok
        report["checks"].append({"name": name, "pass": ok, **detail})
        print(f"{'PASS' if ok else 'FAIL'} {name} {json.dumps(detail, default=str)[:300]}")

    models = get_json(f"{args.url}/v1/models")["data"]
    model = args.model or models[0]["id"]
    m = next((x for x in models if x["id"] == model), None)
    max_len = (m or {}).get("max_model_len")
    record("models_alias", m is not None, model=model, max_model_len=max_len)
    # A server that does not report its ceiling cannot be certified for --context (R11: this used
    # to be skipped silently, so the headline's "ceiling" column read as unverified-but-passing).
    record("context_ceiling", max_len is not None and int(max_len) >= args.context, max_model_len=max_len, required=args.context)

    r = chat_stream(args.url, model, [{"role": "user", "content": "What is 17*19? Reply with only the number."}], max_tokens=16, thinking=False)
    record("arithmetic_thinking_off", "323" in r.text, reply=r.text.strip(), ttft_s=round(r.ttft_s, 3), usage=r.raw_usage)

    r_on = chat_stream(args.url, model, [{"role": "user", "content": "What is 17*19? Reply with only the number."}], max_tokens=512, thinking=True)
    toggled = (r_on.completion_tokens or r_on.stream_token_events) > (r.completion_tokens or r.stream_token_events)
    record("thinking_toggle", toggled and "323" in r_on.text, tokens_on=r_on.completion_tokens, tokens_off=r.completion_tokens)

    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    try:
        body = {"model": model, "messages": [{"role": "user", "content": "Use the tool to get the weather in Lisbon."}],
                "tools": tools, "tool_choice": "auto", "max_tokens": 128, "temperature": 0,
                "chat_template_kwargs": {"thinking": False}}
        import urllib.request
        req = urllib.request.Request(f"{args.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **auth_headers()})
        resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
        calls = resp["choices"][0]["message"].get("tool_calls") or []
        arg_ok = any(c["function"]["name"] == "get_weather" and "lisbon" in c["function"]["arguments"].lower() for c in calls)
        record("tool_call", arg_ok, tool_calls=calls[:1])
    except Exception as exc:  # noqa: BLE001
        record("tool_call", False, error=repr(exc))

    if not args.skip_needle:
        rng = random.Random(args.seed)
        prompt, keys = build_needle_prompt(args.context, args.needles, rng)
        t0 = time.perf_counter()
        r = chat_stream(args.url, model, [{"role": "user", "content": prompt}], max_tokens=64 * args.needles, thinking=False, timeout=7200)
        found = sum(1 for v in keys.values() if v in r.text)
        min_prompt = int(args.context * PROMPT_MIN_FRACTION)
        filled = r.prompt_tokens is not None and r.prompt_tokens >= min_prompt
        record("multi_needle", found == args.needles and r.finish_reason == "stop" and filled, found=found, needles=args.needles,
               prompt_tokens=r.prompt_tokens, prompt_tokens_min=min_prompt, ttft_s=round(r.ttft_s, 2),
               total_s=round(time.perf_counter() - t0, 2), finish_reason=r.finish_reason)

    report["pass"] = ok_all
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stamp_or_exit(report, args.output, "verify.py")
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

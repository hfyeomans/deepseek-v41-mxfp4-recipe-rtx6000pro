#!/usr/bin/env python3
"""Block until the server answers /v1/models and one tiny completion, or time out.

Weight load for the 475 GiB checkpoint takes many minutes; default timeout is
one hour. Exit 0 when ready, 1 on timeout.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import chat_stream, get_json, is_http_error  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    ap.add_argument("--model", default=None, help="served model name; default: first from /v1/models")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--interval", type=float, default=10)
    args = ap.parse_args()

    deadline = time.monotonic() + args.timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            models = get_json(f"{args.url}/v1/models")["data"]
            model = args.model or models[0]["id"]
            r = chat_stream(args.url, model, [{"role": "user", "content": "Reply with the single word READY."}], max_tokens=8, timeout=120)
            if r.text.strip():
                print(f"ready: model={model} ttft={r.ttft_s:.2f}s reply={r.text.strip()!r}")
                return 0
            last = "empty reply"
        except Exception as exc:  # noqa: BLE001 - we retry on anything until the deadline
            last = exc.__class__.__name__ if is_http_error(exc) else repr(exc)
        time.sleep(args.interval)
    print(f"timeout after {args.timeout:.0f}s; last error: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

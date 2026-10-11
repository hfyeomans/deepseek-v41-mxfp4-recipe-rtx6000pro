#!/usr/bin/env python3
"""Tests for openai_client.py against a fake SSE server on localhost."""
from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import StreamResult, auth_headers, chat_stream  # noqa: E402

RECEIVED: list[dict] = []
RECEIVED_AUTH: list[str | None] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RECEIVED.append(body)
        RECEIVED_AUTH.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # reasoning delta first (must not count as content but is the first token, so it
        # sets the TTFT), then 3 content events carrying 5 tokens total per usage, then
        # usage-only chunk
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "thinking..."}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "lo wor"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "ld"}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5}},
        ]
        for i, c in enumerate(chunks):
            if i == 0:
                time.sleep(0.05)  # TTFT
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def test_stream_parsing_and_request_shape():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        r = chat_stream(url, "m", [{"role": "user", "content": "hi"}], max_tokens=8, thinking=False, extra={"min_tokens": 8})
    finally:
        srv.shutdown()
    assert r.text == "Hello world", r.text
    assert r.reasoning == "thinking...", r.reasoning
    assert r.stream_token_events == 4
    assert r.completion_tokens == 5 and r.prompt_tokens == 11
    assert r.finish_reason == "stop"
    assert r.ttft_s >= 0.045, r.ttft_s
    assert r.total_s >= r.ttft_s
    body = RECEIVED[-1]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"thinking": False}
    assert body["min_tokens"] == 8 and body["max_tokens"] == 8


def test_api_key_file_sets_bearer_on_every_request():
    """T3.35: a serve launched with API_KEY_FILE answers 401 to bare calls, so the evidence
    scripts must send the file's token (whitespace-stripped); a literal API_KEY wins over the
    file; with neither set no Authorization header is sent."""
    import os
    import tempfile
    key_file = Path(tempfile.mkdtemp()) / "key"
    key_file.write_text("sk-local-test \n")
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        os.environ["API_KEY_FILE"] = str(key_file)
        assert auth_headers() == {"Authorization": "Bearer sk-local-test"}
        chat_stream(url, "m", [{"role": "user", "content": "hi"}], max_tokens=1)
        assert RECEIVED_AUTH[-1] == "Bearer sk-local-test", RECEIVED_AUTH[-1]
        os.environ["API_KEY"] = "literal-wins"
        assert auth_headers() == {"Authorization": "Bearer literal-wins"}
    finally:
        os.environ.pop("API_KEY_FILE", None)
        os.environ.pop("API_KEY", None)
        srv.shutdown()
    assert auth_headers() == {}


def test_decode_tps_uses_usage_and_excludes_ttft():
    r = StreamResult(text="", ttft_s=1.0, total_s=3.0, completion_tokens=201, prompt_tokens=1, finish_reason="length", stream_token_events=50)
    # 200 inter-token intervals over 2.0 s of decode
    assert abs(r.decode_tps() - 100.0) < 1e-9
    r2 = StreamResult(text="", ttft_s=1.0, total_s=3.0, completion_tokens=None, prompt_tokens=1, finish_reason=None, stream_token_events=51)
    assert abs(r2.decode_tps() - 25.0) < 1e-9
    r3 = StreamResult(text="", ttft_s=1.0, total_s=1.5, completion_tokens=1, prompt_tokens=1, finish_reason=None, stream_token_events=1)
    assert r3.decode_tps() is None
    # R27: usage says 512 tokens but nothing was streamed (ttft_s == total_s): not 5e11 tok/s, None
    r4 = StreamResult(text="", ttft_s=3.0, total_s=3.0, completion_tokens=512, prompt_tokens=1, finish_reason="stop", stream_token_events=0)
    assert r4.decode_tps() is None


if __name__ == "__main__":
    for fn in (test_stream_parsing_and_request_shape, test_api_key_file_sets_bearer_on_every_request,
               test_decode_tps_uses_usage_and_excludes_ttft):
        fn()
        print(f"ok {fn.__name__}")

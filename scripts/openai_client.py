"""Minimal OpenAI-compatible chat client on the standard library.

Shared by wait-ready.py, verify.py, bench-decode.py and bench-prefill.py so all
of them time the same way: request send -> first content token -> last token.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def auth_headers() -> dict[str, str]:
    """Bearer header for a serve launched with API_KEY_FILE (T3.35): the same env the launcher
    reads, so one `export API_KEY_FILE=...` covers the serve and every evidence script. API_KEY
    (literal) is accepted for one-off shells. Empty when neither is set; SGLang exempts only
    /health* and /metrics from the key, so every other request site must send this."""
    key = os.environ.get("API_KEY", "")
    path = os.environ.get("API_KEY_FILE", "")
    if not key and path:
        try:
            key = read_api_key(path)
        except (OSError, ValueError) as e:
            raise SystemExit(f"API_KEY_FILE={path}: {e}") from e
    return {"Authorization": f"Bearer {key}"} if key else {}


def read_api_key(path: str) -> str:
    """The one reader of API_KEY_FILE, shared by the launcher (serve_config.guards) and the clients
    (auth_headers), so both sides agree on what the key is: a regular file holding exactly one
    non-empty line (surrounding whitespace / trailing newline ignored). Anything else raises -
    before 2026-10-11 the launcher `tr -d`-joined a multi-line file into a key no client ever sent
    and the healthy server 401-ed every evidence script (T3.43a / R4)."""
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        raise ValueError("not a regular file")
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one non-empty line, found {len(lines)}")
    if any(ch.isspace() for ch in lines[0]):
        raise ValueError("the key line contains whitespace")
    return lines[0]


@dataclass
class StreamResult:
    text: str
    ttft_s: float
    total_s: float
    completion_tokens: int | None
    prompt_tokens: int | None
    finish_reason: str | None
    stream_token_events: int
    raw_usage: dict = field(default_factory=dict)
    reasoning: str = ""  # reasoning_content deltas (thinking on); counted in ttft_s and stream_token_events, not in text

    @property
    def decode_s(self) -> float:
        return max(self.total_s - self.ttft_s, 1e-9)

    def decode_tps(self) -> float | None:
        """Tokens per second from first to last token, excluding TTFT.

        Uses the server's completion_tokens when reported; otherwise the number
        of streamed content events, which undercounts when the server batches
        several tokens per SSE event (DSpark does), so callers should prefer
        servers that return usage.
        """
        n = self.completion_tokens if self.completion_tokens is not None else self.stream_token_events
        if n is None or n <= 1:
            return None
        if self.stream_token_events == 0:
            # No token ever arrived on the stream, so ttft_s is the whole call and decode_s is the
            # 1e-9 floor: usage alone (a keyed 401 body, a server that reports tokens but streams
            # no deltas) would otherwise read as ~1e10 tok/s (T3.43c / R27).
            return None
        return (n - 1) / self.decode_s


def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=auth_headers()), timeout=timeout) as r:
        return json.loads(r.read())


def chat_stream(base_url: str, model: str, messages: list[dict], *, max_tokens: int, temperature: float = 0.0,
                seed: int | None = 0, thinking: bool = False, extra: dict | None = None, timeout: float = 3600.0) -> StreamResult:
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        # DeepSeek-V4.1 reasoning is on by default at effort 50 in vLLM/SGLang.
        # Pin it explicitly so every measurement is the same mode.
        "chat_template_kwargs": {"thinking": thinking},
    }
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth_headers()}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    events = 0
    usage: dict = {}
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                # The first streamed token of either kind is the TTFT; with thinking on the
                # reasoning tokens are decode tokens too (usage counts them), so decode_tps
                # must span them.
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    reasoning_parts.append(reasoning)
                    events += 1
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(content)
                    events += 1
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    total = time.perf_counter() - t0
    return StreamResult(
        text="".join(text_parts),
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        completion_tokens=usage.get("completion_tokens"),
        prompt_tokens=usage.get("prompt_tokens"),
        finish_reason=finish,
        stream_token_events=events,
        raw_usage=usage,
        reasoning="".join(reasoning_parts),
    )


def is_http_error(exc: BaseException) -> bool:
    return isinstance(exc, (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError, OSError))

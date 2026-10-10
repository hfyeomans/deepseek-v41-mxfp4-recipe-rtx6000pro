"""Minimal OpenAI-compatible chat client on the standard library.

Shared by wait-ready.py, verify.py, bench-decode.py and bench-prefill.py so all
of them time the same way: request send -> first content token -> last token.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


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
        return (n - 1) / self.decode_s


def get_json(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
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
                                 headers={"Content-Type": "application/json"}, method="POST")
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

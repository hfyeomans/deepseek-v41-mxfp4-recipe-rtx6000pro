"""Does a client that closes its HTTP connection mid-decode free its slot? (T3.44)

Four concurrent streaming chat requests with ~32K-token prompts and max_tokens 2048 (long decode).
Client 0 closes its socket after --cut-tokens streamed tokens; the other three run to completion.
In parallel the scheduler log is tailed for `#running-req` and abort lines, so the result records
when (and whether) the server's running count dropped from 4 to 3 relative to the cut.
Exit 0 when `#running-req` reaches <= 3 within --abort-within seconds of the cut, else 1.

    python3 scripts/verify-disconnect-abort.py --output evidence/t344/disconnect-c4-32k.json
"""
from __future__ import annotations

import argparse, json, random, subprocess, sys, threading, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers  # noqa: E402

WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit", "timber", "quartz",
         "velvet", "canyon", "pixel", "saffron", "anchor", "glacier", "marble", "cobalt", "ember", "willow")


def make_prompt(tokens: int, rng: random.Random) -> str:
    body = " ".join(rng.choice(WORDS) + (str(rng.randrange(1000)) if rng.random() < 0.2 else "") for _ in range(int(tokens / 1.1)))
    return body + "\n\nWrite a very long, detailed essay about the words above, at least two thousand words."


def stream(i: int, url: str, model: str, prompt: str, max_tokens: int, cut_tokens: int | None, out: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "seed": i, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth_headers()}, method="POST")
    t0 = time.time(); n = 0; first = None; finish = None; usage = {}
    resp = urllib.request.urlopen(req, timeout=3600)
    try:
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
            for ch in chunk.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    n += 1
                    if first is None:
                        first = time.time()
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if cut_tokens is not None and n >= cut_tokens:
                resp.fp.raw._sock.close() if hasattr(resp.fp, "raw") else None
                resp.close()
                out[i] = {"cut_at_utc": time.time(), "tokens_before_cut": n, "ttft_s": (first or time.time()) - t0}
                return
    finally:
        resp.close()
    out[i] = {"tokens": n, "finish": finish, "usage": usage, "ttft_s": (first or time.time()) - t0, "total_s": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41-flash-hybrid")
    ap.add_argument("--container", default="dsv41-kt")
    ap.add_argument("--prompt-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--cut-tokens", type=int, default=40)
    ap.add_argument("--abort-within", type=float, default=10.0, help="seconds after the cut by which #running-req must be <= 3")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    rng = random.Random(344)
    prompts = [make_prompt(a.prompt_tokens, rng) for _ in range(4)]
    out: dict = {}
    log_t0 = time.time()
    tail = subprocess.Popen(["docker", "logs", "-f", "--since", "1s", a.container], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    log_lines: list[tuple[float, str]] = []

    def pump() -> None:
        for line in tail.stdout:  # type: ignore[union-attr]
            if "running-req" in line or "bort" in line.lower() or "disconnect" in line.lower():
                log_lines.append((time.time(), line.rstrip()))
    threading.Thread(target=pump, daemon=True).start()

    threads = [threading.Thread(target=stream, args=(i, a.url, a.model, prompts[i], a.max_tokens, a.cut_tokens if i == 0 else None, out))
               for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(5)
    tail.terminate()

    cut = out.get(0, {}).get("cut_at_utc")
    running: list[tuple[float, int]] = []
    for ts, line in log_lines:
        if "#running-req:" in line:
            try:
                running.append((ts - cut if cut else ts - log_t0, int(line.split("#running-req:")[1].split(",")[0])))
            except ValueError:
                pass
    first_drop = next((t for t, n in running if cut and t > 0 and n <= 3), None)
    abort_lines = [(round(ts - cut, 1) if cut else None, line) for ts, line in log_lines if "#running-req" not in line]
    res = {"args": vars(a), "clients": out, "cut_at_utc": cut,
           "running_req_vs_cut_s": [(round(t, 1), n) for t, n in running],
           "first_running_le_3_after_cut_s": None if first_drop is None else round(first_drop, 1),
           "abort_or_disconnect_lines": abort_lines}
    res["pass"] = first_drop is not None and first_drop <= a.abort_within
    stamp_or_exit(res, Path(a.output), "verify-disconnect-abort.py")
    print(json.dumps({k: v for k, v in res.items() if k != "running_req_vs_cut_s"}, indent=1))
    print("running-req vs cut (s, n):", res["running_req_vs_cut_s"][:80])
    print("PASS" if res["pass"] else "FAIL", f"slot freed {res['first_running_le_3_after_cut_s']} s after the client closed its socket (limit {a.abort_within})")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

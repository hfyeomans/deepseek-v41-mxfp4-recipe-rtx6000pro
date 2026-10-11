#!/usr/bin/env python3
"""Block until the server answers /v1/models and one tiny completion, or time out.

Weight load for the 475 GiB checkpoint takes many minutes; default timeout is
one hour. Exit 0 when ready, 1 on timeout.

``--url`` / ``--container`` default to the serve this environment names (``serve_config.py endpoint``:
port 8000 / dsv41-kt for the hybrid; before 2026-10-11 the defaults mixed the legacy port 8010
with the hybrid container and matched no deployment, T3.43a / R3). The legacy serve.sh chain
passes both explicitly (README).

After ready, scans the container log (``--container``; "" to skip)
for the KV-pool sizer's silent cap: when MAX_TOTAL_TOKENS exceeds what the static
budget holds, SGLang logs "max_total_tokens=N is larger than the profiled value M"
once at startup and then returns HTTP 400 for inputs over M. Found on the 512K
ladder rung (2026-10-08); printed as a WARNING, exit code unchanged.

Also after ready (T3.10, 2026-10-11): ``posix_fadvise(DONTNEED)`` on every checkpoint shard
except the two Engram shards. Weight load leaves ~73 GiB of shard pages in the page cache that
nothing reads again (the weights are on the GPUs and in the kt CPU-expert shm), and on a 246 GiB
host they pin the Engram tables' page cache - the row cache of the E18 ``file`` layout - to about
10 GiB (`evidence/t310/warmth/`). Dropping them is a no-root advisory call on clean pages; the
Engram shards then grow to the ~85 GiB the kt experts leave. ``--keep-shard-cache`` skips it.

Then primes the server (T3.47 / E40, 2026-10-11): one synthetic prompt of a full prefill chunk plus
an odd tail, so the first *user* request does not pay what the server's own warmup (a 256-token
prompt) never reaches - the first chunk-sized prefill compiles `_hc_mix_reduce_sinkhorn_kernel`
(14 s), builds the CPU-tier MoE runner and device-loads a dozen Triton kernels: 25 s on a 16K
prompt instead of 5-7 s (`evidence/t310/default-wait-ready/warmth.txt`). With the launcher's
KERNEL_CACHE_DIR mount the compile is cached across serves and the prime costs a few seconds;
without it the prime pays the 20 s so the user does not. ``--no-prime`` skips it.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_config  # noqa: E402
from evidence_manifest import ENGRAM_SHARDS  # noqa: E402
from openai_client import chat_stream, get_json, is_http_error  # noqa: E402


_CAP_RE = re.compile(r"max_total_tokens=(\d+) is larger than the profiled value (\d+)")


def drop_stale_shard_cache(model_dir: str, snapshot: str) -> tuple[int, int]:
    """Advise the kernel to drop the page cache of every shard but the Engram ones.

    Returns (shards advised, bytes resident before). Clean, unmapped pages go at once; the
    Engram shards stay mapped by the server and are left alone."""
    snap = Path(os.path.expanduser(model_dir)) / "snapshots" / snapshot
    shards = sorted(p for p in snap.glob("*.safetensors") if p.name not in ENGRAM_SHARDS)
    if not shards:
        return 0, 0
    resident = 0
    try:
        out = subprocess.run(["fincore", "-b", "--noheadings", "--output", "RES", *map(str, shards)],
                             capture_output=True, text=True, timeout=120)
        resident = sum(int(x) for x in out.stdout.split())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    for p in shards:
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    return len(shards), resident


PRIME_TAIL_TOKENS = 1_000   # past the chunk, and not a multiple of 16: Triton specializes on that
_PRIME_WORDS = ("ledger", "granite", "harbor", "signal", "meadow", "copper", "lantern", "orbit",
                "timber", "quartz", "compile", "buffer", "kernel", "socket", "thread", "cache")


def prime(url: str, model: str, chunk_tokens: int) -> tuple[int, float]:
    """One prefill of a full chunk plus an odd tail; returns (prompt tokens, TTFT s).

    Numbered words tokenize at about one token each, so the prompt lands near its target;
    the exact count does not matter as long as the first chunk is full and a tail remains."""
    rng = random.Random(0)
    n = chunk_tokens + PRIME_TAIL_TOKENS
    body = " ".join(f"{rng.choice(_PRIME_WORDS)}{rng.randrange(100)}" for _ in range(n // 2))
    r = chat_stream(url, model, [{"role": "user", "content": body + "\n\nReply with the single word DONE."}],
                    max_tokens=4, timeout=600)
    return r.prompt_tokens or 0, r.ttft_s


def warn_pool_cap(container: str) -> None:
    if not container:
        return
    try:
        log = subprocess.run(["docker", "logs", container], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return
    m = _CAP_RE.search(log.stdout + log.stderr)
    if m:
        want, got = int(m.group(1)), int(m.group(2))
        print(f"WARNING: KV pool capped: MAX_TOTAL_TOKENS={want:,} but the static budget holds {got:,} tokens; "
              f"inputs over {got - 6:,} tokens will get HTTP 400. Raise MEM_FRACTION_STATIC "
              f"(512K x 1 slot needed 0.962) or lower CONTEXT_LENGTH / MAX_RUNNING_REQUESTS.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="default: the port serve_config.py resolves from this environment")
    ap.add_argument("--model", default=None, help="served model name; default: first from /v1/models")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("--container", default=None, help="docker container whose log is scanned for the KV-pool cap "
                    "(default: serve_config.py CONTAINER_NAME); \"\" to skip")
    ap.add_argument("--keep-shard-cache", action="store_true",
                    help="do not fadvise(DONTNEED) the non-Engram shards after ready (default: drop them so the "
                    "Engram tables get the page cache; T3.10)")
    ap.add_argument("--no-prime", action="store_true",
                    help="skip the chunk-sized prime prefill after ready (default: run it so the first user "
                    "request does not pay the serving-time Triton compiles and runner builds; T3.47)")
    args = ap.parse_args()
    cfg: dict = {}
    if args.url is None or args.container is None or not args.keep_shard_cache:
        try:
            cfg = serve_config.resolve(dict(os.environ))
            ep = dict(line.split("=", 1) for line in serve_config.endpoint_lines(cfg))
        except serve_config.ConfigError as exc:
            if args.url is None or args.container is None:
                print(f"FAIL config: {exc} (pass --url and --container explicitly)", file=sys.stderr)
                return 2
            print(f"note: {exc}; the non-Engram shard cache is kept (no model location to drop)")
            args.keep_shard_cache = True
        else:
            args.url = ep["URL"] if args.url is None else args.url
            args.container = ep["CONTAINER"] if args.container is None else args.container

    deadline = time.monotonic() + args.timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            models = get_json(f"{args.url}/v1/models")["data"]
            model = args.model or models[0]["id"]
            r = chat_stream(args.url, model, [{"role": "user", "content": "Reply with the single word READY."}], max_tokens=8, timeout=120)
            if r.text.strip():
                print(f"ready: model={model} ttft={r.ttft_s:.2f}s reply={r.text.strip()!r}")
                warn_pool_cap(args.container)
                if not args.keep_shard_cache:
                    try:
                        n, resident = drop_stale_shard_cache(cfg["HF_MODEL_DIR"], cfg["SNAPSHOT"])
                        print(f"dropped the page cache of {n} non-Engram shards ({resident / 2**30:.1f} GiB was resident); "
                              "the Engram tables get the rest of host memory")
                    except OSError as exc:
                        print(f"WARNING: could not drop the non-Engram shard cache: {exc}")
                if not args.no_prime:
                    chunk = int(cfg.get("CHUNKED_PREFILL_SIZE", 8192))
                    try:
                        n_tok, ttft = prime(args.url, model, chunk)
                        print(f"primed: {n_tok:,}-token prefill (chunk {chunk:,} + tail) ttft={ttft:.1f}s; "
                              "the first user request will not pay the serving-time kernel compiles")
                    except Exception as exc:  # noqa: BLE001 - the server is ready; a failed prime is a warning
                        print(f"WARNING: prime prefill failed: {exc!r}")
                return 0
            last = "empty reply"
        except Exception as exc:  # noqa: BLE001 - we retry on anything until the deadline
            last = exc.__class__.__name__ if is_http_error(exc) else repr(exc)
        time.sleep(args.interval)
    print(f"timeout after {args.timeout:.0f}s; last error: {last}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

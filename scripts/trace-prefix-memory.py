#!/usr/bin/env python3
"""E34e e-ii: where does VRAM go as the prefix grows? Allocator snapshots at fixed prefix lengths.

Drives the served model through one near-fresh prompt per requested prefix length (ascending),
recording the CUDA caching allocator between `/start_profile {"activities": ["MEM"]}` and
`/stop_profile`, which makes every scheduler rank dump `torch.cuda.memory._dump_snapshot` into
the container. The pickles are copied out with `docker cp`, summarised per device (reserved vs
allocated vs inactive bytes, live bytes per allocation site, the in-window high-water mark), and
diffed against the previous prefix. `nvidia-smi` is sampled at 1 Hz throughout so the torch
reserved total can be set against the driver's view: growth nvidia-smi sees but torch does not is
not the caching allocator (kt-kernel, NCCL, cuBLAS workspaces).

    python3 scripts/trace-prefix-memory.py --prefixes 131072,262144,393216,507904 \
        --snapshot-dir .amp/in/e34e-ii/snapshots --output evidence/e34-indexer-memory/e-ii/trace.json

Reads: `make_prompt` / `sample_vram` from `stress-kv-full.py` (same prompt recipe, distinct seed per
prefix so the radix cache cannot serve any of it). The JSON is stamped with the evidence manifest.
The pickles stay under `--snapshot-dir` (hundreds of MB, not evidence); the summary is.

Reading the output. The Part e question is whether the above-262K staircase is allocator
high-water retention (reserved grows, allocated does not; superseded grow-on-demand buffers sit
`inactive`) or genuine per-token state (allocated grows in lockstep). `delta_vs_previous` names
the sites whose live bytes changed; `window_peak_delta_bytes` is how far allocation climbed above
its start inside the window, i.e. the transient the chunk needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers, chat_stream, get_json  # noqa: E402

_spec = importlib.util.spec_from_file_location("stress_kv_full", HERE / "stress-kv-full.py")
_skf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_skf)
make_prompt, sample_vram = _skf.make_prompt, _skf.sample_vram

CONTAINER_DIR = "/tmp/memtrace"
TOP_SITES = 25
HISTORY_MAX_ENTRIES = 100_000  # SGLang's SGLANG_MEM_PROFILE_MAX_ENTRIES default


# ---- snapshot summary (pure; tested in scripts/tests/test_trace_prefix_memory.py) ---------------

def site_of(frames: list[dict] | None) -> str:
    """The allocation site: the innermost frame in sglang / kt / flashinfer code, else the innermost
    Python frame, else `pre-recording` (the block predates `_record_memory_history`)."""
    if not frames:
        return "pre-recording"
    py = [f for f in frames if f.get("filename", "").endswith(".py")]
    own = [f for f in py if any(k in f["filename"] for k in ("/sglang/", "/kt_", "kt_kernel", "/flashinfer/"))
           and "/torch/" not in f["filename"]]
    f = (own or py or frames)[0]
    name = f["filename"]
    for marker in ("/site-packages/", "/python/"):
        if marker in name:
            name = name.split(marker, 1)[1]
    return f"{name}:{f.get('line', 0)} {f.get('name', '')}".rstrip()


def summarise_snapshot(snap: dict) -> dict:
    """Per device: reserved / allocated / inactive bytes, live bytes per site, inactive fragments,
    and the in-window high-water of allocation replayed from `device_traces`."""
    per_dev: dict[int, dict] = {}
    for seg in snap.get("segments", []):
        d = per_dev.setdefault(seg["device"], {"reserved_bytes": 0, "allocated_bytes": 0, "inactive_bytes": 0,
                                               "segments": 0, "live_sites": defaultdict(lambda: [0, 0]),
                                               "inactive_blocks": 0, "largest_inactive_bytes": 0})
        d["reserved_bytes"] += seg["total_size"]
        d["segments"] += 1
        for b in seg.get("blocks", []):
            if b["state"] == "active_allocated":
                d["allocated_bytes"] += b["size"]
                s = d["live_sites"][site_of(b.get("frames"))]
                s[0] += b["size"]
                s[1] += 1
            elif b["state"] == "inactive":
                d["inactive_bytes"] += b["size"]
                d["inactive_blocks"] += 1
                d["largest_inactive_bytes"] = max(d["largest_inactive_bytes"], b["size"])
    for dev, traces in enumerate(snap.get("device_traces", []) or []):
        d = per_dev.setdefault(dev, {"reserved_bytes": 0, "allocated_bytes": 0, "inactive_bytes": 0, "segments": 0,
                                     "live_sites": defaultdict(lambda: [0, 0]), "inactive_blocks": 0,
                                     "largest_inactive_bytes": 0})
        cur = peak = 0
        allocs = frees = segs_alloc = segs_free = 0
        for ev in traces:
            a = ev.get("action")
            if a == "alloc":
                cur += ev["size"]
                allocs += 1
                peak = max(peak, cur)
            elif a == "free_completed":
                cur -= ev["size"]
                frees += 1
            elif a == "segment_alloc":
                segs_alloc += 1
            elif a == "segment_free":
                segs_free += 1
        d["window"] = {"events": len(traces), "allocs": allocs, "frees": frees, "segment_allocs": segs_alloc,
                       "segment_frees": segs_free, "peak_delta_bytes": peak, "net_delta_bytes": cur,
                       # SGLANG_MEM_PROFILE_MAX_ENTRIES defaults to 100000: at the cap the ring has wrapped and
                       # the replayed peak / net are from a truncated window. The live-block state is unaffected.
                       "possibly_truncated": len(traces) >= HISTORY_MAX_ENTRIES}
    out = {}
    for dev, d in sorted(per_dev.items()):
        if d["reserved_bytes"] == 0 and not d.get("window", {}).get("events"):
            continue  # the other rank's device: every process snapshot lists all devices
        sites = sorted(((k, v[0], v[1]) for k, v in d["live_sites"].items()), key=lambda x: -x[1])
        out[str(dev)] = {k: v for k, v in d.items() if k != "live_sites"}
        out[str(dev)]["live_sites"] = [{"site": k, "bytes": b, "blocks": n} for k, b, n in sites[:TOP_SITES]]
        out[str(dev)]["live_sites_total"] = len(sites)
    return out


def diff_devices(prev: dict | None, cur: dict) -> dict:
    """Per device: change in reserved / allocated / inactive and the sites whose live bytes moved."""
    if prev is None:
        return {}
    out = {}
    for dev, c in cur.items():
        p = prev.get(dev)
        if p is None:
            continue
        ps = {s["site"]: s["bytes"] for s in p["live_sites"]}
        cs = {s["site"]: s["bytes"] for s in c["live_sites"]}
        moved = [{"site": k, "delta_bytes": cs.get(k, 0) - ps.get(k, 0)} for k in set(ps) | set(cs)]
        moved = [m for m in moved if m["delta_bytes"] != 0]
        moved.sort(key=lambda m: -abs(m["delta_bytes"]))
        out[dev] = {k: c[k] - p[k] for k in ("reserved_bytes", "allocated_bytes", "inactive_bytes")}
        out[dev]["sites"] = moved[:TOP_SITES]
    return out


def load_snapshots(local_dir: Path) -> dict[str, dict]:
    """`{tp_rank: summary}` for every `*-TP-<rank>-memory*.pickle` under local_dir."""
    out = {}
    for p in sorted(local_dir.glob("*-memory*.pickle")):
        rank = p.name.split("-TP-")[1].split("-")[0]
        with p.open("rb") as fh:
            out[rank] = summarise_snapshot(pickle.load(fh))
    return out


# ---- driver ------------------------------------------------------------------------------------

def post(url: str, body: dict | None = None, timeout: float = 600) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **auth_headers()}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read()
    try:
        return json.loads(txt)
    except ValueError:
        return {"raw": txt.decode(errors="replace")}


def snapshot(url: str, container: str, label: str, local_dir: Path, run) -> tuple[dict, dict]:
    """start_profile(MEM) → run() → stop_profile → docker cp → summaries. Returns (run result, {rank: summary})."""
    remote = f"{CONTAINER_DIR}/{label}"
    post(f"{url}/start_profile", {"activities": ["MEM"], "output_dir": remote, "with_stack": True})
    try:
        result = run()
    finally:
        post(f"{url}/stop_profile")
    dest = local_dir / label
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["docker", "cp", f"{container}:{remote}", str(dest)], check=True, capture_output=True)
    return result, load_snapshots(dest)


def nearest_sample(samples: list[dict], t: float) -> dict | None:
    rows = [s for s in samples if "error" not in s]
    return min(rows, key=lambda s: abs(s["t"] - t)) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--container", default="dsv41-kt")
    ap.add_argument("--prefixes", default="131072,262144,393216,507904",
                    help="ascending target prompt token counts; one prompt each")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--snapshot-dir", type=Path, required=True, help="where the pickles land (not evidence)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    prefixes = [int(x) for x in args.prefixes.split(",")]
    model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
    args.snapshot_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    stop = threading.Event()
    threading.Thread(target=sample_vram, args=(stop, samples), daemon=True).start()
    time.sleep(2)

    steps = []
    _, idle = snapshot(args.url, args.container, "00-idle", args.snapshot_dir, lambda: None)
    t_idle = time.time()
    steps.append({"label": "00-idle", "prompt_tokens_target": 0, "snapshot": idle, "nvidia_smi": nearest_sample(samples, t_idle)})
    print(f"idle: " + "; ".join(f"rank {r}: reserved {s[next(iter(s))]['reserved_bytes'] / 2**30:.2f} GiB, "
                                f"allocated {s[next(iter(s))]['allocated_bytes'] / 2**30:.2f} GiB" for r, s in idle.items()))
    prev = idle
    for i, n in enumerate(prefixes):
        label = f"{i + 1:02d}-{n}"
        rng = random.Random(args.seed + i)
        prompt = make_prompt(n, rng)

        def run():
            t0 = time.time()
            r = chat_stream(args.url, model, [{"role": "user", "content": prompt}], max_tokens=args.max_tokens,
                            thinking=False, timeout=7200)
            return {"prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "ttft_s": r.ttft_s,
                    "total_s": r.total_s, "finish_reason": r.finish_reason, "t_start": t0, "t_end": time.time()}

        res, summ = snapshot(args.url, args.container, label, args.snapshot_dir, run)
        step = {"label": label, "prompt_tokens_target": n, "result": res, "snapshot": summ,
                "nvidia_smi": nearest_sample(samples, res["t_end"]),
                "delta_vs_previous": {r: diff_devices(prev.get(r), summ[r]) for r in summ}}
        steps.append(step)
        prev = summ
        for r, s in sorted(summ.items()):
            for dev, d in s.items():
                w = d.get("window", {})
                print(f"{label}: rank {r} dev {dev}: reserved {d['reserved_bytes'] / 2**30:.2f} GiB, allocated "
                      f"{d['allocated_bytes'] / 2**30:.2f}, inactive {d['inactive_bytes'] / 2**30:.2f} "
                      f"({d['inactive_blocks']} blocks, largest {d['largest_inactive_bytes'] / 2**20:.0f} MiB); "
                      f"window peak +{w.get('peak_delta_bytes', 0) / 2**30:.2f} GiB, net {w.get('net_delta_bytes', 0) / 2**20:+.0f} MiB; "
                      f"prompt {res['prompt_tokens']} tok, TTFT {res['ttft_s']:.1f} s")
                dd = step["delta_vs_previous"].get(r, {}).get(dev, {})
                for m in dd.get("sites", [])[:6]:
                    print(f"    {m['delta_bytes'] / 2**20:+9.0f} MiB  {m['site']}")
    stop.set()
    report = {"url": args.url, "model": model, "container": args.container, "prefixes": prefixes,
              "max_tokens": args.max_tokens, "seed": args.seed, "steps": steps, "vram_samples": samples,
              "snapshot_dir": str(args.snapshot_dir), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    stamp_or_exit(report, args.output, "trace-prefix-memory.py")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

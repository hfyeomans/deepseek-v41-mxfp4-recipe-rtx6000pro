#!/usr/bin/env python3
"""T3.6c / E18: Engram lookups straight from the checkpoint shard on NVMe.

V4.1-Flash stores each Engram table as one contiguous fp8 tensor plus an e8m0
scale tensor inside a single safetensors shard (layer 1 -> model-00047,
layer 14 -> model-00048; 384,006,168 rows x 256 B + 8 B = 94.4 GiB each).
A file-backed layout is therefore an mmap of the shard and two views -- the
page cache is the row cache. This script answers, on the real file:

  1. bit-exactness: rows read through the mmap == rows read via safetensors;
  2. miss latency: random rows on a cold cache, T=1 (48 rows) and a 4,096-token
     prefill chunk (196,608 rows), for three gather strategies:
       torch   torch.index_select on the mmap'd tensor (faults serialised per
               torch thread);
       preadv  N Python threads doing os.preadv of 264 B rows (kernel readahead
               off, queue depth = threads);
       willneed madvise(MADV_WILLNEED) on every row's page, then index_select;
  3. hit latency: the same ids again (page-cache warm).

Run inside dsv41-kt:recipe (needs torch) with the HF dir mounted at /hfm:

  docker run --rm --ipc=host -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v $PWD/scripts:/recipe:ro -v $PWD/evidence/e18-engram-file:/out --entrypoint python dsv41-kt:recipe \
    /recipe/bench-engram-file.py --snapshot /hfm/snapshots/<hash> --json /out/cold.json

Drop the page cache (owner: `sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches'`)
before a cold run, or run right after a reboot. Nothing here needs a GPU.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

FP8_BLOCK = 32
DIM = 256
ROW_BYTES = DIM + DIM // FP8_BLOCK  # 264


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    return header, 8 + n


def locate(snapshot: Path, layer: int) -> dict:
    idx = json.load((snapshot / "model.safetensors.index.json").open())["weight_map"]
    w_name = f"layers.{layer}.engram.embed.weight"
    s_name = f"layers.{layer}.engram.embed.scale"
    shard = snapshot / idx[w_name]
    assert idx[s_name] == idx[w_name], "weight and scale must share a shard"
    header, base = read_header(shard)
    w, s = header[w_name], header[s_name]
    assert w["dtype"] == "F8_E4M3" and s["dtype"] == "F8_E8M0", (w["dtype"], s["dtype"])
    return {
        "shard": shard,
        "rows": w["shape"][0],
        "dim": w["shape"][1],
        "w_off": base + w["data_offsets"][0],
        "w_len": w["data_offsets"][1] - w["data_offsets"][0],
        "s_off": base + s["data_offsets"][0],
        "s_len": s["data_offsets"][1] - s["data_offsets"][0],
    }


class FileTable:
    """The `file` layout: read-only MAP_SHARED mmap of the shard, two views."""

    def __init__(self, loc: dict):
        self.loc = loc
        self.fd = os.open(loc["shard"], os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        # MADV_RANDOM: a 48-row lookup must not drag readahead windows in.
        self.mm.madvise(mmap.MADV_RANDOM)
        raw = torch.frombuffer(self.mm, dtype=torch.uint8)
        self.weight = raw[loc["w_off"] : loc["w_off"] + loc["w_len"]].view(
            torch.float8_e4m3fn
        ).view(loc["rows"], loc["dim"])
        self.scale = raw[loc["s_off"] : loc["s_off"] + loc["s_len"]].view(
            torch.float8_e8m0fnu
        ).view(loc["rows"], loc["dim"] // FP8_BLOCK)

    # --- gather strategies -------------------------------------------------
    def gather_torch(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight.index_select(0, ids), self.scale.index_select(0, ids)

    def gather_preadv(self, ids: torch.Tensor, threads: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = ids.numel()
        w_out = torch.empty(n, DIM, dtype=torch.uint8)
        s_out = torch.empty(n, DIM // FP8_BLOCK, dtype=torch.uint8)
        w_np, s_np = w_out.numpy(), s_out.numpy()
        idl = ids.tolist()
        w_off, s_off = self.loc["w_off"], self.loc["s_off"]

        def work(lo: int, hi: int) -> None:
            for j in range(lo, hi):
                r = idl[j]
                os.preadv(self.fd, [memoryview(w_np[j])], w_off + r * DIM)
                os.preadv(self.fd, [memoryview(s_np[j])], s_off + r * (DIM // FP8_BLOCK))

        step = (n + threads - 1) // threads
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda k: work(k, min(n, k + step)), range(0, n, step)))
        return w_out.view(torch.float8_e4m3fn), s_out.view(torch.float8_e8m0fnu)

    def gather_willneed(self, ids: torch.Tensor, threads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Advise every row's page first (kernel issues the reads asynchronously,
        queue depth limited only by the device), then gather from the mapping."""
        page = mmap.PAGESIZE
        idl = ids.tolist()
        w_off, s_off = self.loc["w_off"], self.loc["s_off"]

        def advise(lo: int, hi: int) -> None:
            for j in range(lo, hi):
                r = idl[j]
                # Advise the whole byte range of each row, not just its first page: a 256 B
                # weight row starts in one page and ends in the next ~6 % of the time.
                a0 = w_off + r * DIM
                a = a0 // page * page
                self.mm.madvise(mmap.MADV_WILLNEED, a, (a0 + DIM - 1) // page * page - a + page)
                b0 = s_off + r * (DIM // FP8_BLOCK)
                b = b0 // page * page
                self.mm.madvise(mmap.MADV_WILLNEED, b, (b0 + DIM // FP8_BLOCK - 1) // page * page - b + page)

        n = len(idl)
        step = (n + threads - 1) // threads
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda k: advise(k, min(n, k + step)), range(0, n, step)))
        return self.gather_torch(ids)


def dequant(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    rows = w.float().unflatten(-1, (-1, FP8_BLOCK))
    return (rows * s.float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, (time.perf_counter() - t) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", default="1,16,4096", help="tokens per lookup; 48 rows each")
    ap.add_argument("--rows-per-token", type=int, default=48)
    ap.add_argument("--threads", default="16,64", help="thread counts for preadv/willneed")
    ap.add_argument("--torch-threads", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3, help="cold reps use fresh ids each")
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    loc = locate(args.snapshot, args.layer)
    tbl = FileTable(loc)
    print(f"layer {args.layer}: {loc['shard'].name}  rows {loc['rows']:,}  "
          f"table {(loc['w_len'] + loc['s_len']) / 2**30:.1f} GiB  {ROW_BYTES} B/row")
    report = {"layer": args.layer, "rows": loc["rows"], "shard": loc["shard"].name,
              "row_bytes": ROW_BYTES, "torch_threads": args.torch_threads, "results": []}

    def fresh_ids(n: int) -> torch.Tensor:
        return torch.randint(0, loc["rows"], (n,), dtype=torch.int64)

    # ---- 1. bit-exactness against safetensors' own reader ------------------
    if not args.skip_exact:
        from safetensors import safe_open

        ids = fresh_ids(64)
        w, s = tbl.gather_torch(ids)
        with safe_open(str(loc["shard"]), framework="pt") as f:
            ws = f.get_slice(f"layers.{args.layer}.engram.embed.weight")
            ss = f.get_slice(f"layers.{args.layer}.engram.embed.scale")
            ok = all(
                torch.equal(w[j].view(torch.uint8), ws[int(r)].view(torch.uint8).view(-1))
                and torch.equal(s[j].view(torch.uint8), ss[int(r)].view(torch.uint8).view(-1))
                for j, r in enumerate(ids.tolist())
            )
        print(f"bit-exact vs safetensors get_slice on 64 random rows: {ok}")
        report["bit_exact_64_rows"] = ok
        if not ok:
            return 1
        # and the dequant path works on the mapped dtypes
        y = dequant(w, s)
        print(f"dequant ok: {tuple(y.shape)} bf16, mean|y| {y.float().abs().mean():.4f}")

    thread_list = [int(t) for t in args.threads.split(",") if t]
    strategies = [("torch", lambda ids: tbl.gather_torch(ids))]
    for t in thread_list:
        strategies.append((f"preadv-{t}", lambda ids, t=t: tbl.gather_preadv(ids, t)))
        strategies.append((f"willneed-{t}", lambda ids, t=t: tbl.gather_willneed(ids, t)))

    print(f"{'tokens':>7} {'rows':>8} {'strategy':<14} {'cold ms (per rep)':<34} {'warm ms':>9} {'µs/row cold':>12}")
    for T in [int(t) for t in args.tokens.split(",") if t]:
        n = T * args.rows_per_token
        for name, fn in strategies:
            cold = []
            last_ids = None
            for _ in range(args.reps):
                ids = fresh_ids(n)
                _, ms = timed(fn, ids)
                cold.append(ms)
                last_ids = ids
            _, warm = timed(fn, last_ids)
            med = sorted(cold)[len(cold) // 2]
            print(f"{T:>7} {n:>8} {name:<14} {' '.join(f'{c:8.1f}' for c in cold):<34} {warm:9.2f} {med * 1e3 / n:12.1f}")
            report["results"].append({"tokens": T, "rows": n, "strategy": name,
                                      "cold_ms": [round(c, 2) for c in cold],
                                      "cold_ms_median": round(med, 2), "warm_ms": round(warm, 3),
                                      "cold_us_per_row": round(med * 1e3 / n, 2)})
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

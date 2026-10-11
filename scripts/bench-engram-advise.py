#!/usr/bin/env python3
"""E36 step 1/2 (T3.20 part b): what an Engram row lookup costs on the `file`
layout, split into the *issue* phase (telling the kernel which pages to read)
and the *gather* phase (index_select from the mapping), per issue mechanism.

The served path (`_FileTable.gather`, overlay engram.py) advises every row's
page with mmap.madvise(MADV_WILLNEED) from a 16-thread pool, then
index_selects. E18 measured that pool at 0.7 ms for 48 rows *warm* (vs 3 us
for a bare index_select): CPython's mmap.madvise holds the GIL, so the pool
serialises 96 syscalls through 16 thread hand-offs. This script times, on
the real shard and fresh random rows (mostly cold: the page cache holds a
fraction of the 94 GiB table), the mechanisms a step-start prefetch could use:

  pool-16   the served path (ThreadPoolExecutor(16), mmap.madvise per page)
  one-thread  one thread, mmap.madvise per page (no hand-offs)
  ctypes    one thread, libc madvise(2) via ctypes (releases the GIL per call)
  bg        ctypes madvise on a helper thread while the caller gathers at once
  pmadv     process_madvise(2) on the calling process: one syscall per 1024
            page ranges (iovec batch), GIL released by ctypes
  none      no advice: index_select faults the pages in, serialised

Residency is measured with mincore(2) on every row's weight page *before*
the lookup, so each sample reports how cold it really was; a second pass on
the same ids gives the warm cost. Nothing here needs a GPU; run in the image:

  docker run --rm --ipc=host -v ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro \
    -v $PWD/scripts:/recipe:ro -v $PWD/evidence/t320-engram-prefetch/io:/out --entrypoint python dsv41-kt:recipe \
    /recipe/bench-engram-advise.py --snapshot /hfm/snapshots/<hash> --json /out/advise.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import statistics
import threading
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

FP8_BLOCK = 32
DIM = 256
SCALE_COLS = DIM // FP8_BLOCK
PAGE = mmap.PAGESIZE
NR_process_madvise = 440  # x86_64
NR_pidfd_open = 434
IOV_MAX = 1024

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.syscall.restype = ctypes.c_long


class iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]


def locate(snapshot: Path, layer: int) -> dict:
    idx = json.load((snapshot / "model.safetensors.index.json").open())["weight_map"]
    w_name = f"layers.{layer}.engram.embed.weight"
    s_name = f"layers.{layer}.engram.embed.scale"
    shard = snapshot / idx[w_name]
    assert idx[s_name] == idx[w_name]
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    base = 8 + n
    w, s = header[w_name], header[s_name]
    assert w["dtype"] == "F8_E4M3" and s["dtype"] == "F8_E8M0"
    return {
        "shard": shard,
        "rows": w["shape"][0],
        "w_off": base + w["data_offsets"][0],
        "w_len": w["data_offsets"][1] - w["data_offsets"][0],
        "s_off": base + s["data_offsets"][0],
        "s_len": s["data_offsets"][1] - s["data_offsets"][0],
    }


class FileTable:
    def __init__(self, loc: dict):
        self.loc = loc
        self.fd = os.open(loc["shard"], os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        self.mm.madvise(mmap.MADV_RANDOM)
        raw = torch.frombuffer(self.mm, dtype=torch.uint8)
        self.base_addr = raw.data_ptr()
        self.weight = raw[loc["w_off"] : loc["w_off"] + loc["w_len"]].view(torch.float8_e4m3fn).view(loc["rows"], DIM)
        self.scale = raw[loc["s_off"] : loc["s_off"] + loc["s_len"]].view(torch.float8_e8m0fnu).view(loc["rows"], SCALE_COLS)
        self.pidfd = _libc.syscall(NR_pidfd_open, os.getpid(), 0)
        if self.pidfd < 0:
            raise OSError(ctypes.get_errno(), "pidfd_open")

    # -- page sets ---------------------------------------------------------------
    def pages(self, ids: np.ndarray) -> np.ndarray:
        """Sorted unique page offsets (into the mapping) the rows' weight and scale bytes touch."""
        w0 = self.loc["w_off"] + ids * DIM
        s0 = self.loc["s_off"] + ids * SCALE_COLS
        # First and last page of each row's bytes (a 256 B row straddles a page boundary ~6 % of
        # the time; the scale row is 8 B and straddles ~0.2 %); np.unique dedupes the overlap.
        w = np.concatenate([w0 // PAGE * PAGE, (w0 + DIM - 1) // PAGE * PAGE])
        s = np.concatenate([s0 // PAGE * PAGE, (s0 + SCALE_COLS - 1) // PAGE * PAGE])
        return np.unique(np.concatenate([w, s]))

    def resident_fraction(self, pages: np.ndarray) -> float:
        """mincore on each page: fraction already in the page cache."""
        vec = (ctypes.c_ubyte * 1)()
        hit = 0
        for p in pages.tolist():
            if _libc.mincore(ctypes.c_void_p(self.base_addr + p), ctypes.c_size_t(PAGE), vec) == 0 and vec[0] & 1:
                hit += 1
        return hit / max(1, len(pages))

    # -- issue mechanisms ----------------------------------------------------------
    def issue_pool(self, pages: np.ndarray, threads: int) -> None:
        mm, pl = self.mm, pages.tolist()
        n = len(pl)

        def advise(lo: int, hi: int) -> None:
            for p in pl[lo:hi]:
                mm.madvise(mmap.MADV_WILLNEED, p, PAGE)

        step = max(1, (n + threads - 1) // threads)
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda k: advise(k, min(n, k + step)), range(0, n, step)))

    def issue_serial(self, pages: np.ndarray) -> None:
        mm = self.mm
        for p in pages.tolist():
            mm.madvise(mmap.MADV_WILLNEED, p, PAGE)

    def issue_ctypes(self, pages: np.ndarray) -> None:
        """One thread, libc madvise(2) through ctypes: the GIL is released around each call."""
        base, adv = self.base_addr, _libc.madvise
        for p in pages.tolist():
            adv(ctypes.c_void_p(base + p), ctypes.c_size_t(PAGE), mmap.MADV_WILLNEED)

    def issue_background(self, pages: np.ndarray) -> threading.Thread:
        """ctypes madvise on a helper thread; the caller gathers concurrently and joins afterwards."""
        t = threading.Thread(target=self.issue_ctypes, args=(pages,), daemon=True)
        t.start()
        return t

    def issue_pmadv(self, pages: np.ndarray) -> None:
        base = self.base_addr
        pl = pages.tolist()
        for lo in range(0, len(pl), IOV_MAX):
            chunk = pl[lo : lo + IOV_MAX]
            arr = (iovec * len(chunk))()
            for j, p in enumerate(chunk):
                arr[j].iov_base = base + p
                arr[j].iov_len = PAGE
            r = _libc.syscall(NR_process_madvise, self.pidfd, arr, len(chunk), mmap.MADV_WILLNEED, 0)
            if r < 0:
                raise OSError(ctypes.get_errno(), "process_madvise(MADV_WILLNEED)")

    # -- gather ------------------------------------------------------------------
    def gather(self, ids: torch.Tensor, pin: bool) -> tuple[torch.Tensor, torch.Tensor]:
        n = ids.numel()
        if pin:
            w = torch.empty(n, DIM, dtype=torch.float8_e4m3fn, pin_memory=torch.cuda.is_available())
            s = torch.empty(n, SCALE_COLS, dtype=torch.float8_e8m0fnu, pin_memory=torch.cuda.is_available())
            torch.index_select(self.weight, 0, ids, out=w)
            torch.index_select(self.scale, 0, ids, out=s)
            return w, s
        return self.weight.index_select(0, ids), self.scale.index_select(0, ids)


def timed(tbl: FileTable, issue, pages: np.ndarray, ids: torch.Tensor) -> tuple[float, float]:
    """(issue ms, gather ms); a mechanism returning a thread is joined inside the gather phase."""
    t0 = time.perf_counter()
    pending = issue(pages)
    t1 = time.perf_counter()
    tbl.gather(ids, pin=False)
    if pending is not None:
        pending.join()
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", default="1,4,16,8192")
    ap.add_argument("--rows-per-token", type=int, default=48)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--torch-threads", type=int, default=16)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    loc = locate(args.snapshot, args.layer)
    tbl = FileTable(loc)
    print(f"layer {args.layer}: {loc['shard'].name}  rows {loc['rows']:,}  page {PAGE}")
    try:
        tbl.issue_pmadv(tbl.pages(rng.integers(0, loc["rows"], 4)))
        pmadv_ok = True
    except OSError as e:
        pmadv_ok = False
        print(f"process_madvise unavailable: {e}")

    mechanisms = [("pool-16", lambda pg: tbl.issue_pool(pg, 16)), ("one-thread", tbl.issue_serial),
                  ("ctypes", tbl.issue_ctypes), ("bg", tbl.issue_background)]
    if pmadv_ok:
        mechanisms.append(("pmadv", tbl.issue_pmadv))
    mechanisms.append(("none", lambda pg: None))

    report = {"layer": args.layer, "rows_per_token": args.rows_per_token, "page": PAGE,
              "process_madvise": pmadv_ok, "reps": args.reps, "results": []}
    print(f"{'tokens':>6} {'mech':>8} {'resident':>8} {'issue ms':>9} {'gather ms':>9} {'total ms':>9} | warm: issue gather")
    for tok in [int(t) for t in args.tokens.split(",") if t]:
        n = tok * args.rows_per_token
        for name, issue in mechanisms:
            cold, warm, res = [], [], []
            for _ in range(args.reps):
                ids_np = rng.integers(0, loc["rows"], n, dtype=np.int64)
                ids = torch.from_numpy(ids_np)
                pages = tbl.pages(ids_np)
                res.append(tbl.resident_fraction(pages))
                cold.append(timed(tbl, issue, pages, ids))
                warm.append(timed(tbl, issue, pages, ids))
            ci = statistics.median(c[0] for c in cold)
            cg = statistics.median(c[1] for c in cold)
            wi = statistics.median(w[0] for w in warm)
            wg = statistics.median(w[1] for w in warm)
            row = {"tokens": tok, "rows": n, "mechanism": name, "resident_before": statistics.median(res),
                   "cold_issue_ms": ci, "cold_gather_ms": cg, "cold_total_ms": ci + cg,
                   "warm_issue_ms": wi, "warm_gather_ms": wg, "warm_total_ms": wi + wg,
                   "cold_samples": cold, "warm_samples": warm}
            report["results"].append(row)
            print(f"{tok:>6} {name:>8} {row['resident_before']:>8.1%} {ci:>9.3f} {cg:>9.3f} {ci + cg:>9.3f} | "
                  f"{wi:>7.3f} {wg:>7.3f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

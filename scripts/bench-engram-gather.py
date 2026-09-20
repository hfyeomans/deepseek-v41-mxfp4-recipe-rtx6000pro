#!/usr/bin/env python3
"""Experiment E3: Engram-style random row gather from DRAM or NVMe.

DeepSeek-V4.1-Flash's Engram layers do 48 random row reads per decoded token
(2 layers x 8 heads x 3 n-gram sizes), each row 256 B FP8 + 8 B UE8M0 scale.
This benchmark builds a synthetic table of --rows rows x --row-bytes bytes and
measures random gathers in batches of --batch rows (48 = one decode token).

Backends:
  dram  table in anonymous memory (mmap), gathered via memoryview slicing
  nvme  table in a file on the given path, gathered with os.pread using
        O_DIRECT (4 KiB aligned reads) from --threads threads

Python overhead makes both numbers lower bounds on what a C/CUDA gather path
achieves; that is stated in the output. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import random
import statistics
import sys
import threading
import time
from pathlib import Path

ALIGN = 4096


def build_table(rows: int, row_bytes: int, path: Path | None) -> tuple[object, int]:
    total = rows * row_bytes
    if path is None:
        m = mmap.mmap(-1, total)
        # touch every page so the measurement is not page-fault bound
        step = mmap.PAGESIZE
        for off in range(0, total, step):
            m[off] = 1
        return m, total
    if path.exists() and path.stat().st_size == total:
        return path, total
    with path.open("wb") as fh:
        chunk = os.urandom(1 << 20)
        written = 0
        while written < total:
            n = min(len(chunk), total - written)
            fh.write(chunk[:n])
            written += n
        fh.flush()
        os.fsync(fh.fileno())
    return path, total


def gather_dram(m: mmap.mmap, offsets: list[int], row_bytes: int) -> int:
    mv = memoryview(m)
    acc = 0
    for off in offsets:
        acc += mv[off]  # touch the row; a full copy would measure Python, not memory
        acc += mv[off + row_bytes - 1]
    return acc


def gather_nvme(fd: int, offsets: list[int]) -> int:
    acc = 0
    for off in offsets:
        aligned = off & ~(ALIGN - 1)
        buf = os.pread(fd, ALIGN, aligned)
        acc += len(buf)
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["dram", "nvme"], required=True)
    ap.add_argument("--path", type=Path, help="file path on the NVMe to test (nvme backend)")
    ap.add_argument("--rows", type=int, default=50_000_000, help="synthetic rows (50M x 264 B = 12.3 GiB)")
    ap.add_argument("--row-bytes", type=int, default=264)
    ap.add_argument("--batch", type=int, default=48, help="rows per token (48 = V4.1 decode)")
    ap.add_argument("--batches", type=int, default=20_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--keep-file", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    table_bytes = args.rows * args.row_bytes
    print(f"backend={args.backend} rows={args.rows:,} row_bytes={args.row_bytes} table={table_bytes / 2**30:.2f} GiB "
          f"batch={args.batch} batches={args.batches} threads={args.threads}", file=sys.stderr)

    all_offsets = [[rng.randrange(args.rows) * args.row_bytes for _ in range(args.batch)] for _ in range(args.batches)]
    latencies: list[float] = []
    lock = threading.Lock()

    if args.backend == "dram":
        t_build = time.perf_counter()
        table, _ = build_table(args.rows, args.row_bytes, None)
        build_s = time.perf_counter() - t_build

        def worker(chunk: list[list[int]]):
            local = []
            for offs in chunk:
                t0 = time.perf_counter()
                gather_dram(table, offs, args.row_bytes)
                local.append(time.perf_counter() - t0)
            with lock:
                latencies.extend(local)
    else:
        if not args.path:
            print("error: --path required for nvme backend", file=sys.stderr)
            return 2
        t_build = time.perf_counter()
        path, _ = build_table(args.rows, args.row_bytes, args.path)
        build_s = time.perf_counter() - t_build
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        fds = [os.open(path, flags) for _ in range(args.threads)]

        def worker(chunk: list[list[int]], fd_idx=[0]):
            with lock:
                fd = fds[fd_idx[0] % len(fds)]
                fd_idx[0] += 1
            local = []
            for offs in chunk:
                t0 = time.perf_counter()
                gather_nvme(fd, offs)
                local.append(time.perf_counter() - t0)
            with lock:
                latencies.extend(local)

    per = max(1, len(all_offsets) // args.threads)
    chunks = [all_offsets[i:i + per] for i in range(0, len(all_offsets), per)]
    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0

    rows_total = args.batch * len(all_offsets)
    lat_sorted = sorted(latencies)
    pct = lambda p: lat_sorted[min(len(lat_sorted) - 1, int(p * len(lat_sorted)))]
    result = {
        "backend": args.backend, "path": str(args.path) if args.path else None, "rows": args.rows, "row_bytes": args.row_bytes,
        "table_gib": round(table_bytes / 2**30, 3), "batch_rows": args.batch, "batches": len(all_offsets), "threads": args.threads,
        "build_s": round(build_s, 2), "wall_s": round(wall, 3),
        "rows_per_s": round(rows_total / wall), "tokens_per_s_equiv": round(len(all_offsets) / wall),
        "batch_latency_ms": {"p50": round(pct(0.5) * 1e3, 4), "p99": round(pct(0.99) * 1e3, 4), "max": round(lat_sorted[-1] * 1e3, 4),
                              "mean": round(statistics.fmean(latencies) * 1e3, 4)},
        "prefill_64k_estimate_s": round(3_100_000 / (rows_total / wall), 2),
        "note": "Python-side gather; lower bound on a native implementation",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.backend == "nvme" and not args.keep_file:
        args.path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

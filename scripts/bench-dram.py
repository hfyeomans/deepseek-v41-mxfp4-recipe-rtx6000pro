#!/usr/bin/env python3
"""Experiment E1: host DRAM copy bandwidth, standard library only.

Each worker *process* allocates its own src/dst buffers of --size-gib/N and
copies src to dst with memoryview slice assignment (a memcpy). Processes, not
threads, because CPython holds the GIL during the copy; a threaded version
measures one core. All workers start on a shared Barrier so the copies
overlap. Reports aggregate GB/s (read + write traffic) per worker count.

This is a lower bound versus a tuned STREAM (no non-temporal stores, no
huge-page control); it is good enough to replace the "estimate" row in
docs/HARDWARE.md.
"""
from __future__ import annotations

import argparse
import json
import mmap
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path


def touch(m: mmap.mmap) -> None:
    for off in range(0, len(m), mmap.PAGESIZE):
        m[off] = 1


def worker(per: int, reps: int, barrier, q) -> None:
    src = mmap.mmap(-1, per)
    dst = mmap.mmap(-1, per)
    touch(src)
    touch(dst)
    svs = memoryview(src)
    dvs = memoryview(dst)
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(reps):
        dvs[:] = svs[:]
    dt = time.perf_counter() - t0
    svs.release()
    dvs.release()
    src.close()
    dst.close()
    q.put(dt)


def run(workers: int, size: int, reps: int) -> float:
    per = size // workers
    per -= per % mmap.PAGESIZE
    barrier = mp.Barrier(workers)
    q: mp.Queue = mp.Queue()
    ps = [mp.Process(target=worker, args=(per, reps, barrier, q)) for _ in range(workers)]
    for p in ps:
        p.start()
    dts = [q.get() for _ in ps]
    for p in ps:
        p.join()
    # aggregate: every worker moved 2*per*reps bytes; the run lasts as long as
    # the slowest worker.
    return 2 * per * workers * reps / max(dts) / 1e9


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", default="1,2,4,8,16,32")
    ap.add_argument("--size-gib", type=float, default=8.0, help="total bytes copied per rep across all workers")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    size = int(args.size_gib * 2**30)
    out = {
        "size_gib": args.size_gib,
        "reps": args.reps,
        "trials": args.trials,
        "results": [],
        "note": "memcpy-style copy in N processes, read+write counted; lower bound vs STREAM",
    }
    print(f"{'workers':>7} {'median GB/s':>12} {'max GB/s':>9}")
    for n in (int(x) for x in args.workers.split(",")):
        vals = [run(n, size, args.reps) for _ in range(args.trials)]
        r = {"workers": n, "median_gbps": round(statistics.median(vals), 1), "max_gbps": round(max(vals), 1)}
        out["results"].append(r)
        print(f"{n:>7} {r['median_gbps']:>12} {r['max_gbps']:>9}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

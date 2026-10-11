#!/usr/bin/env python3
"""E25: controlled host-DRAM bandwidth suite (revalidates E1b / E4b / E17 on the 9955WX + 8x DDR5-6400).

Drives scripts/bench-dram-stream.c (C/OpenMP, built to --bin if missing) across
  * thread count x placement: physical cores on one CCD, split across both CCDs, SMT siblings
  * pattern: read / write / write-nt / copy / copy-nt / triad, and the kt-kernel-shaped
    expert-chunk / expert-rows reads (21.8 MiB chunks, 100 random chunks per iteration = one
    token's CPU-expert bytes at 2.5 experts x 40 layers)
  * page size: 4 KiB (host THP mode is madvise) vs 2 MiB (MADV_HUGEPAGE)
  * optional --gpu-dma 0,1: concurrent pinned host->device DMA from N GPUs (torch container),
    to re-measure E4b additivity with the C reader
and samples per-core clocks (cpufreq), k10temp Tctl and the amd_hsmp socket power during every run.

Host-only, unprivileged, stdlib only. Writes one JSON with every iteration and a Markdown table.

  python3 scripts/bench-dram-suite.py --output evidence/e25-dram-revalidation/suite.json
  python3 scripts/bench-dram-suite.py --only scaling,patterns --iters 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "scripts" / "bench-dram-stream.c"


def read_topology() -> dict:
    """Physical-core list per L3 (CCD) and SMT siblings, from sysfs."""
    per_l3: dict[int, list[int]] = {}
    smt: dict[int, int] = {}
    for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*"), key=lambda s: int(s.rsplit("cpu", 1)[1])):
        cpu = int(p.rsplit("cpu", 1)[1])
        sibs = [int(x) for x in Path(p, "topology/thread_siblings_list").read_text().replace("-", ",").split(",")]
        if cpu != min(sibs):
            continue  # SMT sibling, not the primary
        l3 = int(Path(p, "cache/index3/id").read_text())
        per_l3.setdefault(l3, []).append(cpu)
        smt[cpu] = max(sibs)
    return {"ccds": per_l3, "smt_sibling": smt,
            "numa_nodes": len(glob.glob("/sys/devices/system/node/node[0-9]*"))}


def placement(topo: dict, kind: str, threads: int) -> list[int]:
    ccds = [topo["ccds"][k] for k in sorted(topo["ccds"])]
    if kind == "ccd0":
        cpus = ccds[0][:threads]
    elif kind == "ccd1":
        cpus = ccds[1][:threads]
    elif kind == "split":  # round-robin across CCDs
        cpus = []
        i = 0
        while len(cpus) < threads:
            cpus.append(ccds[i % len(ccds)][i // len(ccds)])
            i += 1
    elif kind == "smt-all":  # both hyperthreads of every core, split across CCDs
        prim = placement(topo, "split", threads // 2)
        cpus = prim + [topo["smt_sibling"][c] for c in prim]
    elif kind == "smt-ccd0":
        prim = ccds[0][: threads // 2]
        cpus = prim + [topo["smt_sibling"][c] for c in prim]
    else:
        raise ValueError(kind)
    assert len(cpus) == threads, (kind, threads, cpus)
    return cpus


class Monitor(threading.Thread):
    """Samples cpufreq of the pinned cpus, k10temp Tctl and amd_hsmp socket power."""

    def __init__(self, cpus: list[int], period: float = 0.2):
        super().__init__(daemon=True)
        self.cpus, self.period, self.stop = cpus, period, threading.Event()
        self.mhz: list[float] = []
        self.temp: list[float] = []
        self.watts: list[float] = []
        self.temp_path = self._hwmon("k10temp", "temp1_input")
        self.power_path = self._hwmon("amd_hsmp_hwmon", "power1_input")

    @staticmethod
    def _hwmon(name: str, attr: str):
        for h in glob.glob("/sys/class/hwmon/hwmon*"):
            try:
                if Path(h, "name").read_text().strip() == name and Path(h, attr).exists():
                    return Path(h, attr)
            except OSError:
                pass
        return None

    def run(self):
        while not self.stop.is_set():
            try:
                f = [int(Path(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read_text()) / 1000 for c in self.cpus]
                self.mhz.append(statistics.mean(f))
                if self.temp_path:
                    self.temp.append(int(self.temp_path.read_text()) / 1000)
                if self.power_path:
                    self.watts.append(int(self.power_path.read_text()) / 1e6)
            except OSError:
                pass
            time.sleep(self.period)

    def summary(self) -> dict:
        def s(v):
            return {"mean": round(statistics.mean(v), 1), "max": round(max(v), 1)} if v else None
        return {"core_mhz": s(self.mhz), "tctl_c": s(self.temp), "socket_w": s(self.watts), "samples": len(self.mhz)}


def run_one(binary: Path, cpus: list[int], pattern: str, size_gib: float, iters: int, hugepage: bool,
            nt: bool = False, extra: list[str] | None = None) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS=str(len(cpus)), OMP_PROC_BIND="true",
               GOMP_CPU_AFFINITY=" ".join(str(c) for c in cpus))
    cmd = [str(binary), "--pattern", pattern, "--size-gib", str(size_gib), "--iters", str(iters), "--json"]
    if hugepage:
        cmd.append("--hugepage")
    if nt:
        cmd.append("--nt")
    cmd += extra or []
    mon = Monitor(cpus)
    mon.start()
    t0 = time.perf_counter()
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True).stdout
    wall = time.perf_counter() - t0
    mon.stop.set()
    mon.join()
    r = json.loads(out.strip().splitlines()[-1])
    steady = r["gbps"][1:]  # drop warm-up iteration
    r.update({"cpus": cpus, "pattern_label": pattern + ("-nt" if nt else ""),
              "gbps_min": min(steady), "gbps_median": statistics.median(steady), "gbps_max": max(steady),
              "wall_s": round(wall, 2), "monitor": mon.summary()})
    return r


def gpu_dma_process(gpus: list[int], seconds: int, image: str) -> subprocess.Popen:
    """Pinned H2D copy loop on the given GPUs, in the project's torch container (needs docker + --gpus)."""
    code = (
        "import torch,time,threading,sys\n"
        f"gpus={gpus}; secs={seconds}\n"
        "def go(d):\n"
        "  torch.cuda.set_device(d); h=torch.empty(2**30,dtype=torch.uint8).pin_memory(); x=torch.empty_like(h,device=f'cuda:{d}')\n"
        "  x.copy_(h); torch.cuda.synchronize(d); print(f'gpu{d} dma-ready',flush=True)\n"
        "  n=0; t0=time.perf_counter()\n"
        "  while time.perf_counter()-t0<secs:\n"
        "    x.copy_(h,non_blocking=True); n+=1\n"
        "    if n%4==0: torch.cuda.synchronize(d)\n"
        "  torch.cuda.synchronize(d); dt=time.perf_counter()-t0\n"
        "  print(f'gpu{d} h2d {n*2**30/dt/1e9:.1f} GB/s over {dt:.1f}s',flush=True)\n"
        "ts=[threading.Thread(target=go,args=(d,)) for d in gpus]\n"
        "[t.start() for t in ts]; [t.join() for t in ts]\n"
    )
    # Only the requested GPUs are exposed; inside the container they are renumbered 0..n-1, so the
    # loop addresses local indices and the caller keeps the host ids for the report.
    code = code.replace(f"gpus={gpus};", f"gpus={list(range(len(gpus)))};")
    devices = "device=" + ",".join(str(g) for g in gpus)
    cmd = ["docker", "run", "--rm", "--gpus", f'"{devices}"', "--ipc=host", "--entrypoint", "python3", image, "-c", code]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def wait_dma_ready(proc: subprocess.Popen, n_gpus: int, timeout_s: float = 120.0) -> list[str]:
    """Block until every GPU thread printed `dma-ready` (container start + pin_memory + first copy
    done), or the process exits / the timeout passes. Returns the lines consumed so far."""
    lines: list[str] = []
    deadline = time.perf_counter() + timeout_s
    ready = 0
    assert proc.stdout is not None
    while ready < n_gpus and time.perf_counter() < deadline and proc.poll() is None:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip("\n"))
        if "dma-ready" in line:
            ready += 1
    if ready < n_gpus:
        raise RuntimeError(f"gpu-dma helper: only {ready}/{n_gpus} GPUs reported ready; output so far: {lines}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="/tmp/bench-dram-stream")
    ap.add_argument("--size-gib", type=float, default=8.0)
    ap.add_argument("--iters", type=int, default=6, help="iterations per config; first is warm-up")
    ap.add_argument("--only", default="scaling,patterns,expert,gpu-dma",
                    help="comma list of sections: scaling,patterns,expert,gpu-dma")
    ap.add_argument("--gpu-dma", default="", help="run gpu-dma section for these GPU sets, e.g. '0;0,1'")
    ap.add_argument("--image", default="dsv4-nvfp4:recipe")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    binary = Path(args.bin)
    if not binary.exists() or binary.stat().st_mtime < SRC.stat().st_mtime:
        subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-o", str(binary), str(SRC)], check=True)
    topo = read_topology()
    ccds = [topo["ccds"][k] for k in sorted(topo["ccds"])]
    n_cores = sum(len(c) for c in ccds)
    sections = set(args.only.split(","))
    results: list[dict] = []

    def rec(section: str, label: str, r: dict):
        r.update({"section": section, "label": label})
        results.append(r)
        m = r["monitor"]
        mhz = m["core_mhz"]["mean"] if m["core_mhz"] else float("nan")
        w = m["socket_w"]["mean"] if m["socket_w"] else float("nan")
        t = m["tctl_c"]["max"] if m["tctl_c"] else float("nan")
        print(f"{section:8s} {label:34s} {r['pattern_label']:12s} thr={r['threads']:2d} "
              f"{r['gbps_min']:6.1f} / {r['gbps_median']:6.1f} / {r['gbps_max']:6.1f} GB/s  "
              f"{mhz:5.0f} MHz {w:4.0f} W {t:3.0f} C", flush=True)

    print(f"topology: {len(ccds)} CCDs {[len(c) for c in ccds]} cores, {topo['numa_nodes']} NUMA node(s); "
          f"size {args.size_gib} GiB; iters {args.iters} (first = warm-up); GB/s = min / median / max of steady iterations")

    if "scaling" in sections:
        for kind, counts in (("ccd0", (1, 2, 4, 8)), ("ccd1", (8,)), ("split", (2, 4, 8, 12, n_cores)),
                             ("smt-ccd0", (16,)), ("smt-all", (2 * n_cores,))):
            for th in counts:
                cpus = placement(topo, kind, th)
                rec("scaling", f"{kind} x{th}", run_one(binary, cpus, "read", args.size_gib, args.iters, hugepage=True))
        rec("scaling", "split x16 4KiB pages", run_one(binary, placement(topo, "split", n_cores), "read", args.size_gib, args.iters, hugepage=False))
        # drift check: repeat the headline config
        rec("scaling", "split x16 repeat", run_one(binary, placement(topo, "split", n_cores), "read", args.size_gib, args.iters, hugepage=True))

    if "patterns" in sections:
        cpus16 = placement(topo, "split", n_cores)
        cpus8 = placement(topo, "ccd0", 8)
        for pattern, nt in (("write", False), ("write", True), ("copy", False), ("copy", True), ("triad", False)):
            rec("patterns", "split x16", run_one(binary, cpus16, pattern, args.size_gib, args.iters, hugepage=True, nt=nt))
            rec("patterns", "ccd0 x8", run_one(binary, cpus8, pattern, args.size_gib, args.iters, hugepage=True, nt=nt))
        rec("patterns", "split x16 4KiB pages", run_one(binary, cpus16, "copy", args.size_gib, args.iters, hugepage=False))

    if "expert" in sections:
        cpus16 = placement(topo, "split", n_cores)
        pool = max(args.size_gib, 32.0)  # 32 GiB pool of 21.8 MiB experts, 100 random per iteration (2.2 GB)
        for pattern, extra in (("expert-chunk", []), ("expert-rows", ["--chunks-per-sync", "3"]),
                               ("expert-rows", ["--chunks-per-sync", "100"]), ("expert-rows", ["--chunks-per-sync", "1"])):
            for hp in (True, False):
                label = f"split x16 {'2MiB' if hp else '4KiB'} " + " ".join(extra)
                rec("expert", label, run_one(binary, cpus16, pattern, pool, args.iters, hugepage=hp, extra=extra))
        rec("expert", "smt-all x32 2MiB", run_one(binary, placement(topo, "smt-all", 2 * n_cores), "expert-rows", pool, args.iters, hugepage=True, extra=["--chunks-per-sync", "3"]))
        rec("expert", "ccd0 x8 2MiB", run_one(binary, placement(topo, "ccd0", 8), "expert-rows", pool, args.iters, hugepage=True, extra=["--chunks-per-sync", "3"]))

    if "gpu-dma" in sections and args.gpu_dma:
        cpus16 = placement(topo, "split", n_cores)
        for spec in args.gpu_dma.split(";"):
            gpus = [int(g) for g in spec.split(",")]
            secs = 40
            proc = gpu_dma_process(gpus, secs, args.image)
            head = wait_dma_ready(proc, len(gpus))  # DMA is running from here on
            r = run_one(binary, cpus16, "read", args.size_gib, args.iters, hugepage=True)
            out, _ = proc.communicate(timeout=180)
            dma = [line for line in head + out.splitlines() if "h2d" in line]
            r["gpu_dma"] = {"gpus": gpus, "container_lines": dma}
            rec("gpu-dma", f"cpu read + DMA gpus {spec}", r)
            for line in dma:
                print("   ", line)

    payload = {"experiment": "E25 host DRAM bandwidth revalidation", "date": time.strftime("%Y-%m-%d"),
               "topology": {"ccds": {str(k): v for k, v in topo["ccds"].items()}, "numa_nodes": topo["numa_nodes"]},
               "size_gib": args.size_gib, "iters": args.iters, "units": "GB/s decimal; read counts S, copy 2S, triad 3S",
               "results": results}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(payload, indent=1))
        print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

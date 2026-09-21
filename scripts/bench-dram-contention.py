#!/usr/bin/env python3
"""E4b: do CPU reads and GPU DMA reads of host DRAM contend, or add up?

Runs inside a torch container with --gpus all. Measures, each for --seconds:
  1. CPU read-only bandwidth alone (torch f32 sum over an 8 GiB tensor, --cpu-threads intra-op threads)
  2. Pinned host-to-device copy alone on every GPU concurrently (one stream per GPU, 4 GiB buffers)
  3. Both at once
and reports GB/s for each side and the combined total. If the CPU-side ceiling (~125 GB/s read on the
16-core 9955WX, E1b) is a core-complex-to-IO-die fabric limit rather than a DIMM limit, DMA traffic that
enters at the IO die should add on top of it; if it is a DIMM/controller limit, the two sides will share.

Feeds: FEASIBILITY.md Option B ("split DRAM-resident experts between CPU compute and PCIe streaming").

  docker run --rm --gpus all -v "$PWD/scripts:/s:ro" -v "$PWD/results:/r" --entrypoint python3 \
      dsv4-nvfp4:recipe /s/bench-dram-contention.py --output /r/e4b.json
"""
import argparse
import json
import threading
import time

import torch


def cpu_reader(x, stop, out):
    n = 0
    t0 = time.perf_counter()
    while not stop.is_set():
        x.sum()
        n += 1
    dt = time.perf_counter() - t0
    out["bytes"] = n * x.numel() * x.element_size()
    out["seconds"] = dt
    out["gbps"] = out["bytes"] / dt / 1e9


def gpu_copier(dev, host_buf, stop, out):
    torch.cuda.set_device(dev)
    dst = torch.empty_like(host_buf, device=f"cuda:{dev}")
    stream = torch.cuda.Stream(device=dev)
    n = 0
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        while not stop.is_set():
            dst.copy_(host_buf, non_blocking=True)
            n += 1
            if n % 4 == 0:
                stream.synchronize()
    stream.synchronize()
    dt = time.perf_counter() - t0
    out["bytes"] = n * host_buf.numel() * host_buf.element_size()
    out["seconds"] = dt
    out["gbps"] = out["bytes"] / dt / 1e9


def run_phase(name, seconds, cpu_x, gpu_bufs, do_cpu, do_gpu):
    stop = threading.Event()
    results = {"cpu": {}, "gpus": [{} for _ in gpu_bufs]}
    threads = []
    if do_cpu:
        threads.append(threading.Thread(target=cpu_reader, args=(cpu_x, stop, results["cpu"])))
    if do_gpu:
        for i, buf in enumerate(gpu_bufs):
            threads.append(threading.Thread(target=gpu_copier, args=(i, buf, stop, results["gpus"][i])))
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join()
    cpu = results["cpu"].get("gbps", 0.0)
    gpu = sum(g.get("gbps", 0.0) for g in results["gpus"])
    print(f"{name:28s} cpu_read={cpu:6.1f} GB/s  h2d_total={gpu:6.1f} GB/s  "
          f"(per GPU: {', '.join(f'{g.get('gbps', 0):.1f}' for g in results['gpus'])})  combined={cpu + gpu:6.1f} GB/s",
          flush=True)
    return {"cpu_read_gbps": round(cpu, 1), "h2d_total_gbps": round(gpu, 1),
            "h2d_per_gpu_gbps": [round(g.get("gbps", 0.0), 1) for g in results["gpus"]],
            "combined_gbps": round(cpu + gpu, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--cpu-threads", type=int, default=16)
    ap.add_argument("--cpu-gib", type=int, default=8)
    ap.add_argument("--gpu-gib", type=int, default=4)
    ap.add_argument("--output")
    args = ap.parse_args()

    torch.set_num_threads(args.cpu_threads)
    ngpu = torch.cuda.device_count()
    print(f"GPUs: {ngpu}; CPU threads: {args.cpu_threads}; CPU buffer {args.cpu_gib} GiB; GPU buffers {args.gpu_gib} GiB pinned each", flush=True)
    cpu_x = torch.ones(args.cpu_gib * 2**30 // 4, dtype=torch.float32)
    gpu_bufs = [torch.ones(args.gpu_gib * 2**30 // 4, dtype=torch.float32).pin_memory() for _ in range(ngpu)]

    res = {"experiment": "E4b DRAM contention: CPU read vs GPU DMA", "gpus": ngpu, "cpu_threads": args.cpu_threads,
           "seconds_per_phase": args.seconds, "phases": {}}
    res["phases"]["cpu_only"] = run_phase("CPU read only", args.seconds, cpu_x, gpu_bufs, True, False)
    res["phases"]["h2d_only"] = run_phase("H2D all GPUs only", args.seconds, cpu_x, gpu_bufs, False, True)
    res["phases"]["both"] = run_phase("CPU read + H2D all GPUs", args.seconds, cpu_x, gpu_bufs, True, True)
    # repeat the solo CPU phase so thermal/clock drift is visible
    res["phases"]["cpu_only_repeat"] = run_phase("CPU read only (repeat)", args.seconds, cpu_x, gpu_bufs, True, False)

    solo = res["phases"]["cpu_only"]["cpu_read_gbps"] + res["phases"]["h2d_only"]["h2d_total_gbps"]
    both = res["phases"]["both"]["combined_gbps"]
    res["additivity"] = round(both / solo, 3) if solo else None
    print(f"additivity (both / sum of solos) = {res['additivity']}  -> 1.0 means the two paths do not contend", flush=True)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(res, f, indent=1)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

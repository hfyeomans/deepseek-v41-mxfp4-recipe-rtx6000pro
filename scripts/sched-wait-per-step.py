#!/usr/bin/env python3
"""Per-thread run-queue wait and preemption during a decode bench, from /proc.

Answers "is the GPU waiting on the CPU tier because the CPU threads are
descheduled?" without perf/root: /proc/<pid>/task/<tid>/schedstat gives
(ns on CPU, ns waiting on a run queue, timeslices) per thread and
/proc/<pid>/task/<tid>/status gives voluntary/nonvoluntary context switches.

  python3 scripts/sched-wait-per-step.py --container dsv41-kt \
      --url http://127.0.0.1:8000 --model dsv41-flash-hybrid \
      --concurrency 2 --max-tokens 200 [--output evidence/.../sched-c2.txt]

Snapshots every thread of every process in the container before and after one
`bench-decode.py` run and prints the deltas per decode step (steps = max_tokens),
grouped by thread name.
"""
import argparse
import collections
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def container_pids(name: str) -> list[int]:
    root = int(subprocess.check_output(["docker", "inspect", "-f", "{{.State.Pid}}", name]).decode())
    pids, todo = [], [root]
    while todo:
        p = todo.pop()
        pids.append(p)
        try:
            kids = subprocess.check_output(["pgrep", "-P", str(p)]).decode().split()
        except subprocess.CalledProcessError:
            kids = []
        todo.extend(int(k) for k in kids)
    return pids


def snapshot(pids: list[int]) -> dict[tuple[int, int], tuple[str, int, int, int, int, int]]:
    out = {}
    for pid in pids:
        base = pathlib.Path(f"/proc/{pid}/task")
        if not base.exists():
            continue
        for t in base.iterdir():
            try:
                comm = (t / "comm").read_text().strip()
                run_ns, wait_ns, slices = (int(x) for x in (t / "schedstat").read_text().split())
                st = (t / "status").read_text()
                vol = int(re.search(r"voluntary_ctxt_switches:\s+(\d+)", st).group(1))
                nonvol = int(re.search(r"nonvoluntary_ctxt_switches:\s+(\d+)", st).group(1))
            except (FileNotFoundError, ProcessLookupError, AttributeError):
                continue
            out[(pid, int(t.name))] = (comm, run_ns, wait_ns, slices, vol, nonvol)
    return out


def group_name(comm: str) -> str:
    if comm.startswith("numa_"):
        return "kt workers (numa_*)"
    if comm.startswith("sglang::sched"):
        return "sglang::scheduler (TP main + helpers)"
    return comm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--container", default="dsv41-kt")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="dsv41-flash-hybrid")
    ap.add_argument("--concurrency", default="2")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--output")
    a = ap.parse_args()

    pids = container_pids(a.container)
    before = snapshot(pids)
    bench = subprocess.run(
        [sys.executable, str(HERE / "bench-decode.py"), "--url", a.url, "--model", a.model,
         "--concurrency", a.concurrency, "--max-tokens", str(a.max_tokens), "--runs", "1",
         "--warmup", "0", "--distinct-prompts"],
        capture_output=True, text=True,
    )
    after = snapshot(pids)
    steps = a.max_tokens

    # per-thread deltas, then aggregate by name group (main scheduler thread kept separate)
    rows = []
    for key, (comm, r1, w1, s1, v1, n1) in after.items():
        if key not in before:
            continue
        _, r0, w0, s0, v0, n0 = before[key]
        d = (r1 - r0, w1 - w0, s1 - s0, v1 - v0, n1 - n0)
        if d[0] + d[1] < 1_000_000:  # < 1 ms total: idle thread, skip
            continue
        pid, tid = key
        name = "TP main thread" if tid == pid and comm.startswith("sglang::sched") else group_name(comm)
        rows.append((name, pid, tid, comm) + d)

    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    for name, pid, tid, comm, run, wait, sl, vol, nonvol in rows:
        g = agg[name]
        g[0] += 1
        g[1] += run
        g[2] += wait
        g[3] += sl
        g[4] += vol
        g[5] += nonvol

    lines = [
        f"concurrency={a.concurrency} max_tokens={a.max_tokens} steps={steps}",
        bench.stdout.strip().splitlines()[-1] if bench.stdout.strip() else bench.stderr.strip()[-300:],
        "",
        f"{'thread group':44s} {'n':>3s} {'run ms/step':>12s} {'rq-wait ms/step':>16s} {'wait/run':>9s} {'slices/step':>12s} {'vol cs/step':>12s} {'nonvol cs/step':>15s}",
    ]
    for name, (n, run, wait, sl, vol, nonvol) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        lines.append(
            f"{name:44s} {n:3d} {run/1e6/steps:12.3f} {wait/1e6/steps:16.3f} "
            f"{(wait/run if run else 0):9.3f} {sl/steps:12.1f} {vol/steps:12.1f} {nonvol/steps:15.2f}"
        )
    lines.append("")
    lines.append("per thread (top 12 by rq-wait):")
    for name, pid, tid, comm, run, wait, sl, vol, nonvol in sorted(rows, key=lambda r: -r[5])[:12]:
        lines.append(
            f"  {comm:16s} pid {pid} tid {tid}: run {run/1e6/steps:7.3f} ms/step, rq-wait {wait/1e6/steps:7.3f} ms/step, "
            f"nonvol {nonvol/steps:6.2f}/step, vol {vol/steps:7.1f}/step"
        )
    text = "\n".join(lines)
    print(text)
    if a.output:
        pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.output).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Summarize nvidia-smi power samples next to benchmark JSON (E8b).

Reads `power-*.csv` (from `nvidia-smi --query-gpu=... --format=csv,nounits -lms`)
and the matching `decode-c1-*w.json` / `prefill-32k-*w.json` in a directory and
prints tok/s, per-GPU busy-median power, SM clock and tok/s per watt. Busy =
samples with utilization > 50 %, so idle tails between requests do not dilute
the median. Also works on a single CSV: `summarize-power.py results/x.csv`.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path

PK, CK, UK, TK = "power.draw [W]", "clocks.current.sm [MHz]", "utilization.gpu [%]", "temperature.gpu"


def read_samples(path: Path) -> list[dict[str, str]]:
    """nvidia-smi rows only: a leaked sampler from an earlier run writing into the same
    file leaves NUL holes and stale rows; drop anything that is not `YYYY/MM/DD ...`."""
    lines = path.read_text(errors="replace").splitlines()
    header, body = lines[0], [l for l in lines[1:] if re.match(r"\d{4}/\d{2}/\d{2} ", l)]
    return list(csv.DictReader([header, *body], skipinitialspace=True))


def summarize_csv(path: Path) -> dict[str, dict]:
    rows = read_samples(path)
    out: dict[str, dict] = {}
    for gpu in sorted({r["index"].strip() for r in rows}):
        g = [r for r in rows if r["index"].strip() == gpu]
        p = [float(r[PK]) for r in g]
        u = [float(r[UK]) for r in g]
        busy = [(float(r[PK]), float(r[CK]), float(r[TK])) for r in g if float(r[UK]) > 50]
        out[gpu] = {
            "samples": len(p),
            "busy_samples": len(busy),
            "power_busy_median_w": round(statistics.median(b[0] for b in busy), 1) if busy else None,
            "power_max_w": round(max(p), 1),
            "sm_clock_busy_median_mhz": round(statistics.median(b[1] for b in busy)) if busy else None,
            "temp_max_c": max(b[2] for b in busy) if busy else None,
            "util_median_pct": statistics.median(u),
        }
    return out


def main() -> None:
    target = Path(sys.argv[1])
    if target.is_file():
        print(json.dumps({target.name: summarize_csv(target)}, indent=2))
        return
    report = {}
    for csv_path in sorted(target.glob("power-*.csv")):
        m = re.match(r"power-(decode|prefill)-(\d+)w\.csv", csv_path.name)
        if not m:
            continue
        kind, cap = m.group(1), int(m.group(2))
        power = summarize_csv(csv_path)
        bench = target / (f"decode-c1-{cap}w.json" if kind == "decode" else f"prefill-32k-{cap}w.json")
        tps = None
        suspect = None
        if bench.exists():
            b = json.loads(bench.read_text())
            if kind == "decode":
                tps = b["levels"][0]["median_tps"]
            else:
                size = b["sizes"][0]
                tps = size["median_prefill_tps"]
                # A 32K prompt cannot prefill in well under a second on these cards; that is a
                # prefix-cache hit (same seeded prompt already in the server's KV).
                cached = [r for r in size["runs"] if r["ttft_s"] < 1.0]
                if cached:
                    suspect = f"{len(cached)}/{len(size['runs'])} runs look prefix-cached (TTFT < 1 s); rerun with a new --seed"
        total_w = sum(v["power_busy_median_w"] or 0 for v in power.values())
        report[f"{kind}-{cap}w"] = {"cap_w": cap, "kind": kind, "tps": tps, "gpus": power,
                                   "total_busy_w": round(total_w, 1),
                                   "tps_per_w": round(tps / total_w, 4) if tps and total_w else None,
                                   "suspect": suspect}
        print(f"{kind:8s} cap {cap:3d} W: {tps if tps is not None else 'n/a':>10} tok/s, busy {total_w:6.0f} W total, "
              f"clk {[v['sm_clock_busy_median_mhz'] for v in power.values()]} MHz, "
              f"tok/s/W {report[f'{kind}-{cap}w']['tps_per_w']}" + (f"  ** {suspect}" if suspect else ""))
    (target / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {target / 'summary.json'}")


if __name__ == "__main__":
    main()

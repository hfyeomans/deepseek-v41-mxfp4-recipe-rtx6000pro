#!/usr/bin/env python3
"""Summarize nvidia-smi power samples next to benchmark JSON (E8b).

Reads `power-*.csv` (from `nvidia-smi --query-gpu=... --format=csv,nounits -lms`)
and the matching `decode-c1-*w.json` / `prefill-32k-*w.json` in a directory and
prints tok/s, per-GPU busy-median power, SM clock and tok/s per watt.
`--smoke <dir>` reads the `power-<label>.csv` files release-smoke.sh --full writes. Busy =
samples with utilization > 50 %, so idle tails between requests do not dilute
the median. Also works on a single CSV: `summarize-power.py results/x.csv`.

A CSV with no sample rows (sampler died, nvidia-smi missing) is recorded as
`{"error": "no samples"}` under its label and makes the exit status 1; with
`--expect-labels a,b,...` a label without a CSV does the same. release-smoke.sh
gates on that (R10: before T3.43b a `--full` smoke could pass with no power trace).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

PK, CK, UK, TK = "power.draw [W]", "clocks.current.sm [MHz]", "utilization.gpu [%]", "temperature.gpu"
FK, RK = "fan.speed [%]", "clocks_event_reasons.active"  # optional; older CSVs predate them
THERMAL_MASK = 0x20 | 0x40  # sw_thermal_slowdown | hw_thermal_slowdown
POWER_CAP_MASK = 0x4        # sw_power_cap


def read_samples(path: Path) -> list[dict[str, str]]:
    """nvidia-smi rows only: a leaked sampler from an earlier run writing into the same
    file leaves NUL holes and stale rows; drop anything that is not `YYYY/MM/DD ...`."""
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        return []
    header, body = lines[0], [l for l in lines[1:] if re.match(r"\d{4}/\d{2}/\d{2} ", l)]
    return list(csv.DictReader([header, *body], skipinitialspace=True))


def summarize_csv(path: Path) -> dict[str, dict]:
    rows = read_samples(path)
    out: dict[str, dict] = {}
    for gpu in sorted({r["index"].strip() for r in rows}):
        g = [r for r in rows if r["index"].strip() == gpu]
        p = [float(r[PK]) for r in g]
        u = [float(r[UK]) for r in g]
        busy_rows = [r for r in g if float(r[UK]) > 50]
        busy = [(float(r[PK]), float(r[CK]), float(r[TK])) for r in busy_rows]
        out[gpu] = {
            "samples": len(p),
            "busy_samples": len(busy),
            "power_busy_median_w": round(statistics.median(b[0] for b in busy), 1) if busy else None,
            "power_max_w": round(max(p), 1),
            "sm_clock_busy_median_mhz": round(statistics.median(b[1] for b in busy)) if busy else None,
            "temp_max_c": max(b[2] for b in busy) if busy else None,
            "util_median_pct": statistics.median(u),
        }
        if busy_rows and FK in busy_rows[0] and RK in busy_rows[0]:
            reasons = [int(r[RK], 16) for r in busy_rows]
            out[gpu]["fan_max_pct"] = max(float(r[FK]) for r in busy_rows)
            out[gpu]["thermal_slowdown_samples"] = sum(1 for x in reasons if x & THERMAL_MASK)
            out[gpu]["power_cap_samples"] = sum(1 for x in reasons if x & POWER_CAP_MASK)
    return out


def summarize_smoke(out: Path, expect_labels: list[str]) -> int:
    """release-smoke.sh --full layout: power-<label>.csv next to c1.json / c4.json /
    prefill-16k.json; writes power-summary.json and prints one line per bench. Returns 1
    when any CSV has no samples or an expected label has no CSV (the summary still records
    every label, so the failure is readable)."""
    report: dict[str, dict] = {}
    bad: list[str] = []
    found = {p.stem[len("power-"):]: p for p in sorted(out.glob("power-*.csv"))}
    for label in sorted(set(found) | set(expect_labels)):
        csv_path = found.get(label)
        power = summarize_csv(csv_path) if csv_path else {}
        if not power:
            report[label] = {"error": "no samples" if csv_path else "no csv"}
            print(f"{label:14s}: {report[label]['error']} ({csv_path.name if csv_path else 'power-' + label + '.csv missing'})")
            bad.append(label)
            continue
        total_w = sum(v["power_busy_median_w"] or 0 for v in power.values())
        temp_max = max((v["temp_max_c"] or 0) for v in power.values())
        report[label] = {"gpus": power, "total_busy_w": round(total_w, 1), "temp_max_c": temp_max}
        print(f"{label:14s}: busy {total_w:6.0f} W total {[v['power_busy_median_w'] for v in power.values()]}, "
              f"clk {[v['sm_clock_busy_median_mhz'] for v in power.values()]} MHz, max {temp_max:.0f} C, "
              f"thermal-throttle samples {[v.get('thermal_slowdown_samples') for v in power.values()]}, "
              f"power-cap samples {[v.get('power_cap_samples') for v in power.values()]}")
    (out / "power-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out / 'power-summary.json'}" + (f"; FAIL no samples for: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=Path, help="a power-*.csv, a directory of bench-power-sweep.sh CSVs, or (with --smoke) a release-smoke directory")
    ap.add_argument("--smoke", action="store_true", help="TARGET is a release-smoke.sh --full output directory")
    ap.add_argument("--expect-labels", default="", help="--smoke: comma-separated labels that must each have a sampled CSV")
    args = ap.parse_args()
    target = args.target
    if args.smoke:
        return summarize_smoke(target, [x for x in args.expect_labels.split(",") if x])
    if target.is_file():
        power = summarize_csv(target)
        print(json.dumps({target.name: power or {"error": "no samples"}}, indent=2))
        return 0 if power else 1
    report = {}
    for csv_path in sorted(target.glob("power-*.csv")):
        m = re.match(r"power-(decode|prefill)(?:-c(\d+))?-(\d+)w\.csv", csv_path.name)
        if not m:
            continue
        kind, conc, cap = m.group(1), int(m.group(2) or 1), int(m.group(3))
        power = summarize_csv(csv_path)
        bench = target / (f"decode-c{conc}-{cap}w.json" if kind == "decode" else f"prefill-32k-{cap}w.json")
        tps = None
        suspect = None
        per_run: list[float] = []
        if bench.exists():
            b = json.loads(bench.read_text())
            if kind == "decode":
                # Per-stream tok/s at C1; aggregate wall tok/s once streams are batched.
                lvl = b["levels"][0]
                tps = lvl["median_tps"] if conc == 1 else lvl["median_wall_tps"]
                per_run = [r["wall_tps" if conc != 1 else "per_seq_tps"] for r in lvl["runs"]]
                per_run = [x if isinstance(x, (int, float)) else statistics.median(x) for x in per_run]
            else:
                size = b["sizes"][0]
                tps = size["median_prefill_tps"]
                per_run = [r["prefill_tps"] for r in size["runs"]]
                # A 32K prompt cannot prefill in well under a second on these cards; that is a
                # prefix-cache hit (same seeded prompt already in the server's KV).
                cached = [r for r in size["runs"] if r["ttft_s"] < 1.0]
                if cached:
                    suspect = f"{len(cached)}/{len(size['runs'])} runs look prefix-cached (TTFT < 1 s); rerun with a new --seed"
            # One stalled run (first shapes after launch, a hiccup) can drag a 3-run median by
            # 5-10 %: T2.9 dsv4 had 7,544 / 3,623 / 8,582. Flag any run below 70 % of the best.
            if per_run and min(per_run) < 0.7 * max(per_run):
                stall = f"run spread {min(per_run):.0f}-{max(per_run):.0f} tok/s: one or more runs stalled; rerun with --runs 5"
                suspect = f"{suspect}; {stall}" if suspect else stall
        total_w = sum(v["power_busy_median_w"] or 0 for v in power.values())
        temp_max = max((v["temp_max_c"] or 0) for v in power.values())
        label = f"{kind}-c{conc}-{cap}w" if kind == "decode" and conc != 1 else f"{kind}-{cap}w"
        report[label] = {"cap_w": cap, "kind": kind, "concurrency": conc if kind == "decode" else None,
                                   "tps": tps, "tps_per_run": [round(x, 1) for x in per_run] if bench.exists() else None,
                                   "gpus": power, "temp_max_c": temp_max,
                                   "total_busy_w": round(total_w, 1),
                                   "tps_per_w": round(tps / total_w, 4) if tps and total_w else None,
                                   "suspect": suspect}
        shown = f"{tps:.1f}" if tps is not None else "n/a"
        print(f"{label:18s}: {shown:>10} tok/s, busy {total_w:6.0f} W total, "
              f"clk {[v['sm_clock_busy_median_mhz'] for v in power.values()]} MHz, max {temp_max:.0f} C"
              + (f", fan {[v.get('fan_max_pct') for v in power.values()]} %, thermal-throttle samples "
                 f"{[v.get('thermal_slowdown_samples') for v in power.values()]}" if any('fan_max_pct' in v for v in power.values()) else "")
              + ", "
              f"tok/s/W {report[label]['tps_per_w']}" + (f"  ** {suspect}" if suspect else ""))
    (target / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {target / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

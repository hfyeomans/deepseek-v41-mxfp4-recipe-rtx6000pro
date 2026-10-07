#!/usr/bin/env python3
"""Deterministic, GPU-free tests for the T3.19a decode timeline recorder
(overlay python/sglang/srt/layers/moe/kt_timeline.py) and its report script.

Runs INSIDE the recipe image with the overlay mounted over the installed sglang
(scripts/tests/run-kt-timeline-tests.sh). CUDA events are replaced by fake events
driven by a scripted clock, so the arithmetic in `_write_sample` and the sampling /
dump gating are tested with known inputs; the GPU smoke test (evidence/t319/a/)
covers the real-event path.

What a wrong implementation would get wrong, and which test catches it:
  - a bucket taken between the wrong pair of marks (e.g. cpu_wait = sync - start)
    -> test_bucket_math_matches_scripted_clock (independent expected values).
  - gaps / gpu_span computed from host clocks instead of device events
    -> test_bucket_math_matches_scripted_clock (host and device clocks differ on purpose).
  - a layer assigned to the wrong segment -> same test (layer 1 sits in segment 1).
  - sampling every replay (cost) or never after the first (no data), or ignoring
    KT_TIMELINE_MAX_RECORDS -> test_sampling_schedule.
  - dumping prefill chunks or capture-warmup forwards, or more than DUMP_MAX times
    -> test_dump_gating.
  - the kt_provider `KtKernelProvider.wait(mark=...)` mirror (T3.19b; was
    kt_ep_wrapper `_sync_timed`) drifting from the installed kt-kernel
    `sync_forward` (wait, then H2D) -> test_kt_sync_forward_order_still_matches_mirror.
  - the recipe report script disagreeing with the recorder about bucket definitions,
    or its step identity being satisfiable by construction
    -> test_report_script_reproduces_buckets (scripted data must FAIL the identity).
"""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="kt-timeline-test-")
os.environ["KT_TIMELINE"] = "1"
os.environ["KT_TIMELINE_DIR"] = TMP
os.environ["KT_TIMELINE_SAMPLE"] = "2"
os.environ["KT_TIMELINE_MAX_RECORDS"] = "2"
os.environ["KT_TIMELINE_DUMP_LAYERS"] = "3"
os.environ["KT_TIMELINE_DUMP_MAX"] = "1"
os.environ["KT_TIMELINE_DUMP_MAX_TOKENS"] = "4"

import torch  # noqa: E402

torch.cuda.is_current_stream_capturing = lambda: False  # no CUDA context in the test image


class _FakeStream:
    def synchronize(self):
        pass


torch.cuda.current_stream = lambda *a, **k: _FakeStream()

from sglang.srt.layers.moe import kt_timeline as m  # noqa: E402


class Clock:
    """Scripted device clock (ms). Each record() takes the next value."""

    def __init__(self):
        self.values = []

    def push(self, *vals):
        self.values.extend(vals)

    def pop(self):
        return self.values.pop(0)


CLOCK = Clock()


class FakeEvent:
    def __init__(self):
        self.ts = None

    def record(self):
        self.ts = CLOCK.pop()

    def elapsed_time(self, other):
        return other.ts - self.ts


m._new_event = lambda external: FakeEvent()
m._in_capture_mode = lambda: False

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print("FAIL:", msg)
    else:
        print("ok:  ", msg)


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def replay(tl, key, sampled_expected):
    """One replay of a 2-segment / 1-break graph with a scripted device clock.

    Device timeline (ms):  seg0 [10, 12]  gap  seg1 [15, 18]  -> span 8, gap 3.
    Layer 0 in seg0: start 10.0 export 10.5 gpu 11.0 sync 11.6 h2d 11.7 end 11.9
    Layer 1 in seg1: start 15.0 export 15.2 gpu 16.0 sync 17.0 h2d 17.2 end 17.5
    Layer marks are recorded at capture; here they are pre-set on the events.
    """
    rec = tl.begin_step(key, n_segments=2, n_breaks=1)
    check(rec.sampled == sampled_expected, f"replay n={rec.n} sampled={rec.sampled} expected {sampled_expected}")
    if rec.sampled:
        CLOCK.push(10.0, 12.0, 15.0, 18.0)  # seg0 start/end, seg1 start/end
    rec.seg_start(0)
    rec.seg_end(0)
    rec.break_start(0, "engram")
    rec.break_end(0)
    rec.seg_start(1)
    rec.seg_end(1)
    rec.finish()
    return rec


def read_records():
    steps, ticks = [], []
    with open(os.path.join(TMP, "timeline-rank0.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            (steps if r["t"] == "step" else ticks).append(r)
    return steps, ticks


def test_bucket_math_matches_scripted_clock():
    tl = m.StepTimeline(rank=0)
    tl.begin_capture("decode:1")
    mark0 = tl.layer_marker(0)
    CLOCK.push(10.0, 10.5, 11.0, 11.6, 11.7, 11.9)
    for name in m.LAYER_MARKS:
        mark0(name)
    mark1 = tl.layer_marker(1)
    CLOCK.push(15.0, 15.2, 16.0, 17.0, 17.2, 17.5)
    for name in m.LAYER_MARKS:
        mark1(name)
    tl.end_capture()
    check(tl.layer_marker(0) is None, "layer_marker returns None outside capture")

    replay(tl, "decode:1", sampled_expected=True)
    tl._flush_ticks()
    steps, ticks = read_records()
    check(len(steps) == 1 and len(ticks) == 1, f"one step + one tick written ({len(steps)}, {len(ticks)})")
    s = steps[0]
    check(approx(s["gpu_span_ms"], 8.0), f"gpu_span from device events = 8.0 (got {s['gpu_span_ms']})")
    check([approx(x["gpu_ms"], y) for x, y in zip(s["segments"], (2.0, 3.0))] == [True, True], "segment GPU ms 2.0 / 3.0")
    check(approx(s["gaps"][0]["gpu_gap_ms"], 3.0), "gap after segment 0 = 3.0 (Engram break)")
    check(s["breaks"][0]["name"] == "engram" and approx(s["breaks"][0]["gpu_gap_ms"], 3.0), "break row carries name + gap")
    L0, L1 = s["layers"]
    exp0 = {"export": 0.5, "gpu_experts": 0.5, "cpu_wait": 0.6, "h2d": 0.1, "combine": 0.2, "span_ms": 1.9, "cpu_total_ms": 1.1}
    exp1 = {"export": 0.2, "gpu_experts": 0.8, "cpu_wait": 1.0, "h2d": 0.2, "combine": 0.3, "span_ms": 2.5, "cpu_total_ms": 1.8}
    for row, exp, name in ((L0, exp0, "layer 0"), (L1, exp1, "layer 1")):
        for k, v in exp.items():
            check(approx(row[k], v), f"{name} {k} = {v} (got {row[k]:.4f})")
        check(approx(sum(row[b] for b, _, _ in m.LAYER_BUCKETS), row["span_ms"]), f"{name} buckets sum to span")
    check(L0["segment"] == 0 and L1["segment"] == 1, "layers assigned to their segments by device timestamp")
    check(s["host"]["replay_wall_ms"] >= 0 and s["host"]["final_sync_ms"] >= 0, "host clocks present")
    return tl


def test_sampling_schedule(tl):
    # already did n=0 (sampled). SAMPLE=2, MAX_RECORDS=2 -> n=1 no, n=2 yes, n=3 no, n=4 no (cap), n=5 no
    for n, exp in ((1, False), (2, True), (3, False), (4, False), (5, False)):
        replay(tl, "decode:1", sampled_expected=exp)
    tl._flush_ticks()
    steps, ticks = read_records()
    check(len(steps) == 2, f"2 sampled steps after 6 replays with SAMPLE=2, MAX_RECORDS=2 (got {len(steps)})")
    check([t["n"] for t in ticks] == list(range(6)), "a tick for every replay, in order")
    # a second key has its own counter
    tl.begin_capture("decode:4")
    tl.end_capture()
    r = tl.begin_step("decode:4", 2, 1)
    check(r.n == 0 and r.sampled, "per-key replay counter starts at 0 and samples")


def test_dump_gating(tl):
    check(not tl.dump_wanted(2, 1), "layer not in KT_TIMELINE_DUMP_LAYERS -> no dump")
    check(not tl.dump_wanted(3, 8), "eager forward with > DUMP_MAX_TOKENS tokens (prefill chunk) -> no dump")
    check(tl.dump_wanted(3, 1), "eager decode-size forward at dump layer -> dump")
    m._in_capture_mode = lambda: True
    check(not tl.dump_wanted(3, 1), "capture-mode warmup forward -> no dump")
    m._in_capture_mode = lambda: False
    tl.begin_capture("decode:2")
    check(tl.dump_wanted(3, 64), "during capture -> dump regardless of token count")
    tl.end_capture()
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    h = tl.dump_begin(3, {"hidden_states": x, "topk_ids": None})
    x.mul_(0)  # the in-place mask that follows in apply()
    check(torch.equal(h["tensors"]["hidden_states"], torch.arange(8, dtype=torch.float32).view(2, 4)), "dump_begin snapshots before the in-place mask")
    check("topk_ids" not in h["tensors"], "None tensors are skipped")
    tl.dump_end(h, {"kt_output": torch.ones(2, 4)})
    files = sorted(f for f in os.listdir(TMP) if f.startswith("dump-rank0-eager-layer3"))
    check(files == ["dump-rank0-eager-layer3-0.pt"], f"eager dump saved once ({files})")
    check(not tl.dump_wanted(3, 1), "after DUMP_MAX eager dumps -> no more")
    payload = torch.load(os.path.join(TMP, files[0]))
    check(payload["num_tokens"] == 2 and set(payload["tensors"]) == {"hidden_states", "kt_output"}, "dump payload shape")


def test_kt_sync_forward_order_still_matches_mirror():
    try:
        from kt_kernel.experts_base import BaseMoEWrapper
    except Exception as e:  # pragma: no cover
        check(False, f"installed kt_kernel importable ({e})")
        return
    src = inspect.getsource(BaseMoEWrapper.sync_forward)
    i_wait = src.find("sync_with_cuda_stream(cuda_stream, allow_pending)")
    i_h2d = src.find("return self.copy_forward_output_to_device(hidden_states)")
    check(i_wait > 0 and i_h2d > i_wait, "kt sync_forward CUDA path = sync_with_cuda_stream then copy_forward_output_to_device")
    check("_layer_has_pending_deferred.get(self.layer_idx" in src, "kt sync_forward derives allow_pending from _layer_has_pending_deferred")
    from sglang.srt.layers.moe import kt_provider as kp
    wsrc = inspect.getsource(kp.KtKernelProvider.wait)
    j_wait = wsrc.find("sync_with_cuda_stream(stream, allow_pending)")
    j_mark = wsrc.find('mark("sync")', j_wait)
    j_h2d = wsrc.find("copy_forward_output_to_device(x)", j_mark)
    check(0 < j_wait < j_mark < j_h2d, "overlay KtKernelProvider.wait(mark=...) mirrors that order with the sync mark in between")
    from sglang.srt.layers.moe import kt_ep_wrapper as w
    esrc = inspect.getsource(w.KTEPWrapperMethod._sync_timed)
    check("provider.wait(" in esrc and "w.cpu_infer.sync_with_cuda_stream" not in esrc,
          "overlay _sync_timed delegates to the provider (no second copy of the mirror)")


def test_report_script_reproduces_buckets():
    report = os.environ.get("KT_TIMELINE_REPORT", "/t/kt-timeline-report.py")
    if not os.path.exists(report):
        check(False, f"report script mounted at {report}")
        return
    out_json = os.path.join(TMP, "report.json")
    r = subprocess.run(
        [sys.executable, report, os.path.join(TMP, "timeline-rank0.jsonl"), "--skip", "0", "--min-steps", "1",
         "--json", out_json, "--bench-ms-per-token", "20.0"],
        capture_output=True, text=True,
    )
    check(r.returncode in (0, 2), f"report ran (rc={r.returncode}) {r.stderr.strip()[:200]}")
    summ = json.load(open(out_json))
    e = summ["keys"]["rank0/decode:1"]
    b = e["median_buckets_ms"]
    exp = {"export": 0.7, "gpu_experts": 1.3, "cpu_wait": 1.6, "h2d": 0.3, "combine": 0.5, "engram_break": 3.0}
    for k, v in exp.items():
        check(approx(b[k], v, 1e-4), f"report {k} = {v} (got {b[k]:.4f})")
    # other_gpu = segments (5.0) - layer spans (1.9 + 2.5)
    check(approx(b["other_gpu"], 0.6, 1e-4), f"report other_gpu = 0.6 (got {b['other_gpu']:.4f})")
    check(approx(e["gpu_bucket_sum_ms"], 8.0, 1e-4), "report GPU bucket sum = 8.0 = gpu span")
    check(approx(e["median_gpu_span_ms"], 8.0, 1e-6), "report gpu span 8.0")
    check(e["median_event_consistency_err"] < 1e-6, "event consistency exact on scripted clock")
    # the scripted replays have no GPU work, so the host period is tiny and the GPU sum (8 ms) cannot
    # match it: the identity check must FAIL here (exit 2), proving it is not tautological.
    check(r.returncode == 2 and e["identity_err"] > 0.05, f"identity check fails on scripted data (rc={r.returncode}, err={e['identity_err']:.2f})")
    check(e["step_source"] == "unsampled tick period", "step time comes from unsampled ticks when present")
    check(e["ladder_ms"]["bench_ms_per_token"] == 20.0 and "outside_replay_ms" in e["ladder_ms"], "ladder filled from --bench-ms-per-token")
    check(len(e["layers"]) == 2 and e["breaks"][0]["name"] == "engram", "report keeps per-layer rows and break names")


def main():
    tl = test_bucket_math_matches_scripted_clock()
    test_sampling_schedule(tl)
    test_dump_gating(tl)
    test_kt_sync_forward_order_still_matches_mirror()
    test_report_script_reproduces_buckets()
    print(f"\n{len(FAILURES)} failed checks")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())

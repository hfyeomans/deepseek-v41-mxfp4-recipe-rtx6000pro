#!/usr/bin/env python3
"""Tests for the DSpark multi-stream-overlap gate (review C4, Bug A).

Pure Python, no torch, no docker: the gate module is loaded by file path from the
overlay (OVERLAY_DIR, default ~/dev/src/sglang-dsv41-overlay). The wrong implementations
these are written against:
  - gating on "any speculation" or on the platform alone would force the overlap off for
    the qualified plain-decode profile (consistency PASS at bs 1..4 *with* overlap on)
    -> test_plain_decode_untouched, test_dspark_off_sm120_untouched.
  - silently overriding an explicit `=1` would hide that the operator asked for the
    corrupting combination -> test_explicit_true_is_refused.
  - honouring only "unset" and not an explicit `=0` would leave a launch that already
    carries the old workaround without the log line -> test_explicit_false_forced_off.
  - case-sensitive algorithm match ("dspark" from a lower-cased CLI) -> test_case_insensitive.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

OVERLAY = Path(
    os.environ.get("OVERLAY_DIR", str(Path.home() / "dev/src/sglang-dsv41-overlay"))
)
MODULE = OVERLAY / "python/sglang/srt/arg_groups/dspark_overlap_gate.py"

if not MODULE.is_file():
    print(f"dspark overlap gate tests: {MODULE} not found (set OVERLAY_DIR); skipped")
    sys.exit(3)

spec = importlib.util.spec_from_file_location("dspark_overlap_gate", MODULE)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

decide = gate.decide_multi_stream_overlap
OverlapGateError = gate.OverlapGateError


def test_dspark_unset_forced_off():
    d = decide(speculative_algorithm="DSPARK", is_sm120=True, env_is_set=False, env_value=True)
    assert d.force_off, d
    assert "Bug A" in d.reason


def test_explicit_false_forced_off():
    # A launch still carrying the old workaround gets the same decision and the log line.
    d = decide(speculative_algorithm="DSPARK", is_sm120=True, env_is_set=True, env_value=False)
    assert d.force_off, d


def test_explicit_true_is_refused():
    try:
        decide(speculative_algorithm="DSPARK", is_sm120=True, env_is_set=True, env_value=True)
    except OverlapGateError as e:
        assert "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1" in str(e)
        assert "DSPARK" in str(e)
    else:
        raise AssertionError("explicit overlap=1 with DSpark on SM120 must be refused")


def test_plain_decode_untouched():
    for is_set, val in ((False, True), (True, True), (True, False)):
        d = decide(speculative_algorithm=None, is_sm120=True, env_is_set=is_set, env_value=val)
        assert not d.force_off, (is_set, val, d)


def test_dspark_off_sm120_untouched():
    # Not this overlay's problem on Hopper/Blackwell-datacenter parts; leave upstream behaviour.
    d = decide(speculative_algorithm="DSPARK", is_sm120=False, env_is_set=True, env_value=True)
    assert not d.force_off, d


def test_eagle_untouched():
    d = decide(speculative_algorithm="EAGLE", is_sm120=True, env_is_set=False, env_value=True)
    assert not d.force_off, d


def test_case_insensitive():
    d = decide(speculative_algorithm="dspark", is_sm120=True, env_is_set=False, env_value=True)
    assert d.force_off, d


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

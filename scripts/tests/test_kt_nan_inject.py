#!/usr/bin/env python3
"""T3.13 / C10 unit tests (CPU torch, in-image): the NaN injector's state machine, the
sampler's flag-then-sanitize step, and the scheduler's per-row abort.

The served-path proof is scripts/verify-nan-rejection.py (evidence/e24-nan/c10/); this
file pins the pieces that proof depends on so a refactor cannot silently turn the
per-request abort back into a sanitized sample or a whole-batch failure.
"""
from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.kt_nan_inject import NanInjector


def test_injector_fires_once_on_the_nth_replay_of_its_batch_size() -> None:
    inj = NanInjector.from_env("row=1,bs=4,step=3,layer=20")
    out = torch.zeros(4, 8)
    assert inj.add_to(19, out) is out, "other layers are untouched"
    assert torch.equal(inj.add_to(20, out), out), "unarmed: adds zeros"
    for bs in (1, 4, 4, 2, 4):  # the 3rd bs=4 replay arms it; bs=1/2 replays do not count
        inj.before_replay(bs)
    assert inj.fired and inj.armed and inj.replays_at_bs == 3
    poisoned = inj.add_to(20, out)
    assert math.isnan(poisoned[1, 0].item())
    assert torch.isfinite(poisoned[0]).all() and torch.isfinite(poisoned[2:]).all(), "one row only"
    inj.before_replay(4)
    assert not inj.armed and bool((inj.buf == 0).all()), "cleared before the next replay"
    for _ in range(5):
        inj.before_replay(4)
    assert bool((inj.buf == 0).all()), "fires once"
    big = inj.add_to(20, torch.zeros(100, 8))
    assert torch.isfinite(big).all(), "prefill-sized forwards are skipped"
    assert NanInjector.from_env("") is None


def test_sampler_flags_nan_and_posinf_rows_and_keeps_good_rows_bit_identical() -> None:
    # Mirrors Sampler.forward (recipe T3.13): flags, then nan_to_num_ that keeps -inf.
    logits = torch.tensor([[1.5, float("-inf"), 2.0],            # grammar-masked, well formed
                           [float("nan"), 1.0, float("inf")],    # broken
                           [0.1, -0.2, 0.3]])                    # plain
    before = logits.clone()
    rows = torch.isnan(logits).any(dim=-1) | torch.isposinf(logits).any(dim=-1)
    torch.nan_to_num_(logits, nan=-1e30, posinf=1e30, neginf=float("-inf"))
    assert rows.tolist() == [False, True, False]
    assert torch.equal(logits[0], before[0]) and torch.equal(logits[2], before[2])
    assert torch.isfinite(logits[1]).all()


def test_scheduler_aborts_only_flagged_rows() -> None:
    from http import HTTPStatus

    from sglang.srt.managers.scheduler_components import batch_result_processor as brp

    class Req:
        def __init__(self, rid):
            self.rid, self.to_finish, self._finished = rid, None, False

        def finished(self):
            return self._finished

    reqs = [Req("a"), Req("b"), Req("c"), Req("d")]
    batch = SimpleNamespace(reqs=reqs, spec_algorithm=SimpleNamespace(is_none=lambda: True))
    flags = brp._non_finite_rows(batch, SimpleNamespace(nan_rows=torch.tensor([False, True, False, False])))
    assert flags == [False, True, False, False]
    for i, (req, bad) in enumerate(zip(reqs, flags)):
        if bad:
            brp._reject_non_finite_row(req, i, "decode")
    assert [r.to_finish is None for r in reqs] == [True, False, True, True]
    ab = reqs[1].to_finish
    assert ab.status_code == HTTPStatus.INTERNAL_SERVER_ERROR and ab.err_type == "NonFiniteLogits"
    assert "non-finite logits" in ab.message
    # Guards: check off, spec layout, row-count mismatch -> nothing rejected.
    assert brp._non_finite_rows(batch, SimpleNamespace(nan_rows=None)) is None
    assert brp._non_finite_rows(batch, None) is None
    assert brp._non_finite_rows(batch, SimpleNamespace(nan_rows=torch.tensor([True, True]))) is None
    spec = SimpleNamespace(reqs=reqs, spec_algorithm=SimpleNamespace(is_none=lambda: False))
    assert brp._non_finite_rows(spec, SimpleNamespace(nan_rows=torch.tensor([True] * 4))) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok   {t.__name__}")
    print(f"{len(tests)} passed")
    sys.exit(0)

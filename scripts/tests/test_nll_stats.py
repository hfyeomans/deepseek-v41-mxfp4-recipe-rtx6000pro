#!/usr/bin/env python3
"""Unit tests for the paired/pooled NLL statistics behind the T3.28 shape-aware gate (no server)."""
import importlib.util
import math
import pathlib
import sys

spec = importlib.util.spec_from_file_location("nll_score", pathlib.Path(__file__).resolve().parents[1] / "nll-score.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_common_prefix():
    assert m.common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert m.common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert m.common_prefix_len([1, 2], [1, 2, 3]) == 2
    assert m.common_prefix_len([], [1]) == 0


def test_paired_sign_and_magnitude():
    # new is worse by exactly 0.1 nats on every token -> delta_mean_nll = +0.1, sd 0, z 0 (degenerate), 4 changed
    base = [-1.0, -2.0, -0.5, -3.0]
    new = [x - 0.1 for x in base]
    st = m.paired_stats(new, base)
    assert st["n"] == 4 and close(st["delta_mean_nll"], 0.1) and close(st["sd_delta_logp"], 0.0)
    assert st["tokens_changed"] == 4 and st["tokens_worse_gt_0p5"] == 0


def test_paired_hand_computed():
    base = [-1.0, -1.0, -1.0, -1.0]
    new = [-1.6, -0.4, -1.0, -1.0]  # deltas: -0.6, +0.6, 0, 0 -> mean 0, sd = sqrt((0.36+0.36)/3)
    st = m.paired_stats(new, base)
    assert close(st["delta_mean_nll"], 0.0) and close(st["z_paired"], 0.0)
    assert close(st["sd_delta_logp"], math.sqrt(0.72 / 3)) and close(st["se"], math.sqrt(0.72 / 3) / 2)
    assert st["tokens_worse_gt_0p5"] == 1 and st["tokens_better_gt_0p5"] == 1 and st["tokens_changed"] == 2
    assert close(st["max_abs_delta_logp"], 0.6) and close(st["mean_abs_delta_logp"], 0.3)


def test_paired_truncates_to_n():
    base = [-1.0, -1.0, -1.0]
    new = [-1.0, -1.0, -5.0]
    assert m.paired_stats(new, base, 2)["tokens_changed"] == 0
    assert m.paired_stats(new, base)["tokens_changed"] == 1
    assert m.paired_stats([], [])["n"] == 0


def test_pooled_is_token_weighted_not_text_weighted():
    # text A: 1 token, delta +1.0 ; text B: 3 tokens, delta 0 -> pooled mean = -0.25 logp -> delta_mean_nll +0.25
    a = ([-2.0], [-1.0], 1)
    b = ([-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], 3)
    st = m.pooled_stats([a, b])
    assert st["n"] == 4 and close(st["delta_mean_nll"], 0.25)
    # z is finite and positive-in-logp direction is negative here (new worse)
    assert st["z_paired"] < 0
    # cluster-robust se: mean_d = -0.25; cluster sums 1*(-1) - 1*(-0.25) = -0.75 and 0 - 3*(-0.25) = +0.75
    assert close(st["se_cluster"], math.sqrt(0.75 ** 2 + 0.75 ** 2) / 4) and st["texts"] == 2


def test_zero_se_does_not_hide_a_shift():
    # R7: one text, every token worse by the same 0.9 nats -> se_cluster == 0 for any delta.
    # The old code reported z 0.0 and |z| <= 2 passed; now z is +inf (and -inf for an improvement).
    base = [-1.0] * 5
    st = m.pooled_stats([([x - 0.9 for x in base], base, 5)])
    assert close(st["delta_mean_nll"], 0.9) and st["se_cluster"] == 0.0 and st["texts"] == 1
    assert st["z_cluster"] == -math.inf and st["z_paired"] < -1e6  # iid se is fp residue, not exactly 0
    better = m.pooled_stats([([x + 0.9 for x in base], base, 5)])
    assert better["z_cluster"] == math.inf
    # bit-identical stays z 0 (the qualified 64K x 4 case)
    same = m.pooled_stats([(base, base, 5)])
    assert same["z_cluster"] == 0.0 and same["tokens_changed"] == 0
    assert m.z_score(0.0, 0.0) == 0.0 and m.z_score(0.5, 0.25) == 2.0


def test_cluster_se_exceeds_iid_when_texts_disagree():
    # two texts with opposite constant shifts: iid se sees 6 tokens of +-0.1, cluster se sees two blocks
    a = ([-1.1] * 3, [-1.0] * 3, 3)
    b = ([-0.9] * 3, [-1.0] * 3, 3)
    st = m.pooled_stats([a, b])
    assert close(st["delta_mean_nll"], 0.0)
    assert st["se_cluster"] > st["se"]


def test_compare_to_uses_common_token_prefix():
    base = {"token_ids": [10, 11, 12, 13, 14], "logprobs": [-1.0, -1.0, -1.0, -1.0], "mean_nll": 1.0}
    res = {"token_ids": [10, 11, 12, 99, 14], "logprobs": [-1.0, -1.0, -9.0, -9.0], "mean_nll": 5.0}
    c = m.compare_to(res, {"mean_nll": 5.0}, base, "b.json")
    # ids agree for 3 positions -> 2 paired logprobs (positions 1, 2), neither changed
    assert c["n"] == 2 and c["tokens_changed"] == 0 and close(c["delta_full_text_nll"], 4.0)


def test_pooled_empty_keeps_cluster_keys():
    # R34: an empty pairing must not make the gate KeyError on z_cluster; it must see n == 0.
    st = m.pooled_stats([([], [], 0), ([], [], 0)])
    assert st["n"] == 0 and st["texts"] == 2 and st["z_cluster"] == 0.0 and st["se_cluster"] == 0.0


def test_summarize_refuses_no_tokens():
    try:
        m.summarize([], 1024)
    except SystemExit as e:
        assert "no tokens scored" in str(e)
    else:
        raise AssertionError("summarize([]) must exit, not divide by zero")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"ok: {len(fns)} nll-stats tests")
    sys.exit(0)

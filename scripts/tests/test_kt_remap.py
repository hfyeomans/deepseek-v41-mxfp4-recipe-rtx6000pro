#!/usr/bin/env python3
"""Tests for the per-layer expert remap (overlay kt_remap.py, recipe T3.8b).

Runs INSIDE the recipe image via scripts/tests/run-kt-remap-tests.sh against a stub
sglang package holding only kt_contract.py and kt_remap.py (torch, CPU only).

What a wrong implementation would get wrong, and which test catches it:
  - perm and its inverse swapped (expert tensors renamed by perm instead of
    inv, or the gate rows gathered by inv instead of perm): the GPU slots would
    hold experts the gate does not score there        -> test_routing_is_invariant
  - the hot set not landing in slots 0..G-1, or the cold slots not in
    ascending checkpoint order                        -> test_slots
  - the fused shared expert (renamed to experts.<n> later) or a layer the map
    does not cover being permuted                     -> test_untouched
  - a hash-routed layer (tid2eid) silently remapped   -> test_untouched
  - a map for another placement (num_gpu_experts) or
    with duplicate / out-of-range ids accepted        -> test_file_validation
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

from sglang.srt.layers.moe.kt_contract import KTContractError  # noqa: E402
from sglang.srt.layers.moe.kt_remap import ExpertRemap, load_cached  # noqa: E402

N, G = 12, 5  # experts per layer, GPU slots


def make() -> ExpertRemap:
    return ExpertRemap.from_gpu_sets({0: [11, 3, 7, 0, 9], 2: [1, 2, 3, 4, 5]}, N, G, source="t")


def test_slots() -> None:
    r = make()
    assert r.perm(0) == (0, 3, 7, 9, 11, 1, 2, 4, 5, 6, 8, 10)
    assert r.cpu_checkpoint_ids(0) == (1, 2, 4, 5, 6, 8, 10)
    assert r.perm(2) == (1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11)
    assert r.perm(1) == tuple(range(N)) and not r.is_remapped(1) and r.is_remapped(0)
    inv = r.layers[0][1]
    assert all(r.perm(0)[inv[e]] == e for e in range(N))


def test_untouched() -> None:
    r = make()
    w = torch.arange(N * 4, dtype=torch.float32).reshape(N, 4)
    # layer not in the map: identity, same object
    name, out = r.remap_weight("model.layers.1.mlp.experts.7.gate_proj.weight", w)
    assert name == "model.layers.1.mlp.experts.7.gate_proj.weight" and out is w
    name, out = r.remap_weight("model.layers.1.mlp.gate.weight", w)
    assert name == "model.layers.1.mlp.gate.weight" and out is w
    # shared experts and non-MoE tensors are not expert tensors
    for n in ("model.layers.0.mlp.shared_experts.gate_proj.weight",
              "model.layers.0.self_attn.wq_a.weight", "model.layers.0.mlp.gate.weight_scale_inv"):
        assert r.remap_weight(n, w) == (n, w)
    # the fused shared expert must not reach the remap once renamed to experts.<N>
    try:
        r.remap_weight(f"model.layers.0.mlp.experts.{N}.gate_proj.weight", w)
    except KTContractError as e:
        assert "outside" in str(e), e
    else:
        raise AssertionError("expert id N accepted")
    # hash-routed layers cannot be remapped
    try:
        r.remap_weight("model.layers.0.mlp.topk.tid2eid", w)
    except KTContractError as e:
        assert "hash" in str(e), e
    else:
        raise AssertionError("tid2eid of a remapped layer accepted")
    assert r.remap_weight("model.layers.1.mlp.topk.tid2eid", w) == ("model.layers.1.mlp.topk.tid2eid", w)
    # a gate of the wrong width is refused, not silently gathered
    try:
        r.remap_weight("model.layers.0.mlp.gate.weight", w[:3])
    except KTContractError:
        pass
    else:
        raise AssertionError("gate with 3 rows accepted")


def test_routing_is_invariant() -> None:
    """Loading through the remap must route every token to the same checkpoint experts.

    Simulate the load: checkpoint expert e's tensor is renamed to slot inv[e]; the
    gate rows / biases are permuted. Then top-k over the permuted gate, mapped back
    through perm, must equal top-k over the original gate; and the slot a token is
    sent to must hold that expert's tensor.
    """
    torch.manual_seed(0)
    r = make()
    hidden, tokens, k = 16, 64, 3
    gate_w = torch.randn(N, hidden)
    bias = torch.randn(N)
    bias_vl = torch.randn(N)
    expert_tensor = {e: torch.full((2, 2), float(e)) for e in range(N)}

    slots = {}
    for e, t in expert_tensor.items():
        name, out = r.remap_weight(f"model.layers.0.mlp.experts.{e}.down_proj.weight", t)
        slot = int(name.split(".experts.")[1].split(".")[0])
        assert name.endswith(".down_proj.weight") and out is t
        assert slot not in slots, "two experts renamed to one slot"
        slots[slot] = t
    assert sorted(slots) == list(range(N))
    _, gate_p = r.remap_weight("model.layers.0.mlp.gate.weight", gate_w)
    _, bias_p = r.remap_weight("model.layers.0.mlp.gate.e_score_correction_bias", bias)
    _, bias_vl_p = r.remap_weight("model.layers.0.mlp.gate.e_score_correction_bias_vl", bias_vl)
    assert gate_p.shape == gate_w.shape and gate_p.is_contiguous()

    x = torch.randn(tokens, hidden)
    for b, bp in ((bias, bias_p), (bias_vl, bias_vl_p)):
        want = (x @ gate_w.T + b).topk(k, dim=-1).indices  # checkpoint ids
        got_slots = (x @ gate_p.T + bp).topk(k, dim=-1).indices  # physical slots
        perm = torch.tensor(r.perm(0))
        assert torch.equal(perm[got_slots], want), "permuted gate routes to different experts"
        # and every chosen slot holds the tensor of the expert the original gate chose
        for tok in range(tokens):
            for j in range(k):
                assert float(slots[int(got_slots[tok, j])][0, 0]) == float(want[tok, j])
    # hot set on GPU: every id the map lists is in a slot < G
    assert {int(perm[s]) for s in range(G)} == {11, 3, 7, 0, 9}


def test_file_validation() -> None:
    good = {"num_experts": N, "num_gpu_experts": G, "source": "unit",
            "gpu": {"0": [11, 3, 7, 0, 9], "1": [0, 1, 2, 3, 4]}}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "map.json")
        with open(p, "w") as f:
            json.dump(good, f)
        r = ExpertRemap.from_file(p, N, G)
        assert r.source == "unit" and sorted(r.layers) == [0, 1]
        assert load_cached(p, N, G) is load_cached(p, N, G) and load_cached(None, N, G) is None
        # a map for another placement
        try:
            ExpertRemap.from_file(p, N, G + 1)
        except KTContractError as e:
            assert "num_gpu_experts" in str(e), e
        else:
            raise AssertionError("map for G accepted at G+1")
        for bad_gpu, needle in (({"0": [11, 3, 7, 0]}, "lists 4"),
                                ({"0": [11, 3, 7, 0, 0]}, "duplicate"),
                                ({"0": [11, 3, 7, 0, N]}, "out-of-range"),
                                ({}, "no 'gpu'")):
            with open(p, "w") as f:
                json.dump({**good, "gpu": bad_gpu}, f)
            try:
                ExpertRemap.from_file(p, N, G)
            except KTContractError as e:
                assert needle in str(e), (bad_gpu, e)
            else:
                raise AssertionError(f"{bad_gpu} accepted")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

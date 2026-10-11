#!/usr/bin/env python3
"""Tests for the generated BENCHMARKS headline table (T3.24e). Pure Python, temp dir only.

Wrong implementations these are written against:
  - a table that carries a number no stamped file holds (the hand-edited paragraph it replaces
    mixed two serves) -> test_numbers_come_from_the_stamped_files.
  - reading a file whose stamp does not match its sibling manifest (a re-launched serve's
    evidence copied over an old manifest), or an unstamped file at all -> test_refuses_bad_stamps.
  - `check` passing on a stale block, or on a document without the markers -> test_check_detects_stale.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
em_spec = importlib.util.spec_from_file_location("evidence_manifest", HERE.parent / "evidence_manifest.py")
em = importlib.util.module_from_spec(em_spec)
em_spec.loader.exec_module(em)
bh_spec = importlib.util.spec_from_file_location("benchmarks_headline", HERE.parent / "benchmarks_headline.py")
bh = importlib.util.module_from_spec(bh_spec)
bh_spec.loader.exec_module(bh)


def fake_manifest() -> dict:
    return em.finalize({
        "schema": 1, "written_utc": "2026-10-10T12:00:00Z",
        "recipe": {"sha": "a" * 40, "short": "aaaaaaaa"},
        "overlay": {"sha": "b" * 40, "short": "bbbbbbbb"},
        "image": {"tag": "dsv41-kt:recipe", "id": "sha256:" + "c" * 64},
        "config": {"KT_NUM_GPU_EXPERTS": "219", "KT_EXPERT_MAP": "~/x/config/map.json", "CHUNKED_PREFILL_SIZE": "8192",
                   "KT_MAX_DEFERRED": "4", "MEM_FRACTION_STATIC": "0.96", "SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE": "1"},
        "gpus": [{"power_limit_w": 325.0}, {"power_limit_w": 325.0}],
    })


def stamped(doc: dict, manifest: dict) -> dict:
    out = dict(doc)
    out["evidence"] = {"tool": "t", "recorded_utc": "2026-10-10T12:30:00Z", "manifest_hash": manifest["content_hash"]}
    return out


def write_rung(root: pathlib.Path, name: str, c1_median: float = 57.25, bad_stamp: bool = False, unstamped: bool = False) -> dict:
    d = root / "evidence" / name
    (d / "corpus").mkdir(parents=True)
    m = fake_manifest()
    (d / "manifest.json").write_text(json.dumps(m))
    c1 = {"distinct_prompts": True, "runs": 6, "warmup": 1, "max_tokens": 512,
          "levels": [{"concurrency": 1, "median_tps": c1_median, "median_wall_tps": 55.5}]}
    c1s = stamped(c1, m)
    if bad_stamp:
        c1s["evidence"]["manifest_hash"] = "0" * 64
    (d / "c1.json").write_text(json.dumps(c1 if unstamped else c1s))
    (d / "prefill-16k.json").write_text(json.dumps(stamped({"passes": [
        {"pass": "cold", "sizes": [{"median_prefill_tps": 2934.4, "median_ttft_s": 5.51,
                                    "runs": [{"prefill_tps": 662.0, "ttft_s": 24.1}, {"prefill_tps": 2934.4, "ttft_s": 5.51}]}]},
        {"pass": "warm", "sizes": [{"median_prefill_tps": 3264.0, "median_ttft_s": 4.9}]}]}, m)))
    (d / "verify.json").write_text(json.dumps(stamped({"checks": [
        {"name": "context_ceiling", "pass": True, "max_model_len": 65536},
        {"name": "multi_needle", "pass": True, "found": 6, "needles": 6, "prompt_tokens": 55206, "ttft_s": 12.6}]}, m)))
    (d / "stress-kv.json").write_text(json.dumps(stamped({"concurrency": 4, "prompt_tokens_target": 63488, "wall_s": 80.5,
        "all_finished": True, "results": [{"prompt_tokens": 61785, "ttft_s": 22.0}, {"prompt_tokens": 61743, "ttft_s": 73.3}]}, m)))
    (d / "corpus" / "pooled.json").write_text(json.dumps(stamped({"mean_nll_token_weighted": 1.5766, "compare": {
        "baseline_dir": "evidence/t328/x/corpus", "n": 65653, "delta_mean_nll": -0.00033, "se": 0.00101, "z_paired": 0.33,
        "se_cluster": 0.0013, "z_cluster": 0.26, "tokens_changed": 65226}}, m)))
    return {"rung": "64K × 4", "dir": f"evidence/{name}", "note": "n"}


def test_numbers_come_from_the_stamped_files():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        cfg = {"rows": [write_rung(root, "r1", c1_median=57.25)]}
        md = bh.markdown(cfg, root)
        assert "| 57.2 / 55.5 |" in md, md  # c1 median / wall, from c1.json
        # R28: the first (truly cold) run is named beside the cold median; the warm pass has no first run
        assert "2,934 cold (TTFT 5.5 s; first run 662, TTFT 24.1 s) / 3,264 warm (TTFT 4.9 s)" in md, md
        # R28: the caption's runs / warmups / tokens come from the stamped files, not a typed sentence
        assert "over 6 runs (C1) and ? runs (C4) of 512 forced tokens after 1 warmup, distinct prompts" in md, md
        assert "6/6 at 55,206 tokens, TTFT 13 s, ceiling 65,536" in md, md
        assert "C4 × 61,764 tokens, wall 80 s, TTFT max 73 s, all finished" in md, md
        assert "z +0.26, 65,226 of 65,653 moved" in md, md
        assert "| — |" in md  # c4.json absent -> dash, never a guessed number
        assert "overlay `bbbbbbbb`, image `cccccccccccc`, 219 / 165 experts (map.json), chunk 8192, D=4, fraction 0.96, lever on, cap 325 W" in md, md
        # a changed stamped number changes the table (no cached or typed value survives)
        cfg2 = {"rows": [write_rung(root, "r2", c1_median=41.3)]}
        assert "| 41.3 / 55.5 |" in bh.markdown(cfg2, root)


def test_refuses_bad_stamps():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        for kw in ({"bad_stamp": True}, {"unstamped": True}):
            cfg = {"rows": [write_rung(root, "bad-" + next(iter(kw)), **kw)]}
            try:
                bh.markdown(cfg, root)
            except bh.HeadlineError as e:
                assert "c1.json" in str(e), e
            else:
                raise AssertionError(f"{kw}: accepted")
        # a tampered sibling manifest (content hash no longer recomputes) is refused too
        cfg = {"rows": [write_rung(root, "tampered")]}
        mp = root / "evidence" / "tampered" / "manifest.json"
        m = json.loads(mp.read_text()); m["config"]["KT_NUM_GPU_EXPERTS"] = "230"; mp.write_text(json.dumps(m))
        try:
            bh.markdown(cfg, root)
        except bh.evidence_manifest.ManifestError:
            pass
        else:
            raise AssertionError("tampered manifest accepted")


def test_check_detects_stale():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        cfg = {"rows": [write_rung(root, "r1")]}
        block = bh.markdown(cfg, root)
        doc = "# B\n\nintro\n\n" + block + "\nrest\n"
        assert bh.current_block(doc) == block
        stale = doc.replace("| 57.2 / 55.5 |", "| 58.0 / 56.2 |")
        assert bh.current_block(stale) != block
        assert bh.current_block("# B\n\nno markers\n") is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"ok: {len(fns)} benchmarks-headline tests")
    sys.exit(0)

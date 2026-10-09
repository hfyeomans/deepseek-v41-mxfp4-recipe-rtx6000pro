#!/usr/bin/env python3
"""Tests for the immutable evidence manifests (T3.19f). Pure Python: collectors are injected, no
docker/git/HSMP/nvidia-smi is touched, everything lands in a temp dir.

Wrong implementations these are written against:
  - hashing a non-canonical dump (key order, whitespace) so a re-serialised manifest "fails"
    verify, or hashing only some fields so an edit to `config` keeps verifying
    -> test_hash_round_trip_and_tamper.
  - a collector failure aborting the launcher (the manifest must still be written on a box
    without docker/HSMP) or silently dropping the field -> test_collect_records_failures_as_null.
  - storing /home/<name> paths (ci.sh identifier scan rejects them) -> test_collect_tildes_paths.
  - stamping a report with the hash but not copying the manifest beside it, or letting a second
    serve's evidence overwrite a directory's manifest.json -> test_stamp_sibling_rules.
  - a stale manifest (launcher restarted by hand) stamped silently -> test_stamp_container_check.
  - `EVIDENCE_MANIFEST=none` written as if it had a manifest -> test_stamp_none_is_explicit.
  - check_evidence rejecting every pre-cut-over file, or accepting a tampered sibling
    -> test_check_evidence.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("evidence_manifest", HERE.parent / "evidence_manifest.py")
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)

FIXED_NOW = 1_790_000_000.0  # 2026-09-21T...Z, any fixed instant


def fake_collectors(**overrides):
    c = {
        "git_state": lambda repo: {"sha": "a" * 40, "short": "aaaaaaaa", "branch": "main", "dirty": False, "dir": str(repo)},
        "docker_image_id": lambda tag: "sha256:" + "1" * 64,
        "docker_container": lambda name: {"name": name, "id": "c" * 64, "created": "2026-10-08T00:00:00Z", "image_id": "sha256:" + "1" * 64},
        "gpu_state": lambda: [{"index": 0, "name": "gpu", "power_limit_w": 325.0, "driver": "x", "memory_total_mib": 1}],
        "hsmp_sample": lambda: {"fclk_mhz": 2100, "mclk_mhz": 3200},
        "meminfo": lambda: {"MemTotal": 1, "SwapUsed": 0},
        "pressure": lambda: {"io": "x"},
        "engram_residency": lambda d, s: [{"shard": "s", "resident_bytes": 1, "size_bytes": 2, "resident_pct": 50.0}],
    }
    c.update(overrides)
    return c


def base_env(home: str) -> dict:
    return {"HOME": home, "HF_MODEL_DIR": f"{home}/hf", "OVERLAY_DIR": f"{home}/overlay"}


def make_manifest(home: str, container: str | None = "dsv41-kt", **overrides) -> dict:
    return em.finalize(em.collect(base_env(home), container=container, collectors=fake_collectors(**overrides),
                                  now=lambda: FIXED_NOW, recipe_dir=pathlib.Path(home) / "recipe"))


def test_hash_round_trip_and_tamper():
    with tempfile.TemporaryDirectory() as home:
        m = make_manifest(home)
        assert em.verify(m)
        # key order and whitespace must not matter: a re-serialised manifest still verifies
        reordered = json.loads(json.dumps(dict(sorted(m.items(), reverse=True)), separators=(",", ":")))
        assert em.verify(reordered)
        # every other field is covered by the hash
        for key in ("config", "box", "bios", "written_utc", "errors"):
            t = json.loads(json.dumps(m))
            t[key] = {"edited": True} if key != "written_utc" else "1999-01-01T00:00:00Z"
            assert not em.verify(t), key
        # write/load: load refuses a tampered file
        d = pathlib.Path(home) / "mf"
        p = em.write(em.collect(base_env(home), container="dsv41-kt", collectors=fake_collectors(), now=lambda: FIXED_NOW,
                                recipe_dir=pathlib.Path(home)), d, "dsv41-kt")
        assert p.name.endswith("-dsv41-kt.json") and (d / "current.json").read_text() == p.read_text()
        assert em.load(p)["content_hash"] == m["content_hash"] or True  # dirs differ; just must load
        body = json.loads(p.read_text())
        body["config"]["EXPERTS_GPU"] = "999"
        p.write_text(json.dumps(body))
        try:
            em.load(p)
            raise AssertionError("tampered manifest loaded")
        except em.ManifestError:
            pass


def test_collect_records_failures_as_null():
    def boom(*a):
        raise RuntimeError("no docker here")

    with tempfile.TemporaryDirectory() as home:
        m = make_manifest(home, docker_image_id=boom, hsmp_sample=boom, engram_residency=boom, gpu_state=boom)
        assert m["image"]["id"] is None and m["bios"]["hsmp"] is None and m["gpus"] is None
        assert m["box"]["engram_residency"] is None
        assert sorted(e.split(":")[0] for e in m["errors"]) == ["engram", "gpus", "hsmp", "image"]
        assert all("no docker here" in e for e in m["errors"])
        assert em.verify(m)
        # a run without a container records that, not an error
        assert make_manifest(home, container=None)["container"] is None
        assert make_manifest(home, container=None)["errors"] == []


def test_collect_tildes_paths():
    with tempfile.TemporaryDirectory() as home:
        m = make_manifest(home)
        text = json.dumps(m)
        assert home not in text, "a $HOME path leaked into the manifest"
        assert m["config"]["HF_MODEL_DIR"] == "~/hf" and m["config"]["OVERLAY_DIR"] == "~/overlay"
        assert m["recipe"]["dir"] == "~/recipe" and m["overlay"]["dir"] == "~/overlay"
        assert m["config"]["BIOS_SLOT"] == "3" and m["bios"]["slot_declared"] == "3"
        assert "MANIFEST_DIR" in m["config"] and "experts gpu=" in m["resolved_line"]
        # $HOME embedded in a knob value (E35 EXTRA_DOCKER_ARGS), not only as a string prefix
        env = {**base_env(home), "EXTRA_DOCKER_ARGS": f"-v {home}/overlay/x.py:/sgl/x.py:ro -v {home}/y:/y:ro"}
        m = em.collect(env, container=None, collectors=fake_collectors(), now=lambda: FIXED_NOW,
                       recipe_dir=pathlib.Path(home) / "recipe")
        assert home not in json.dumps(m), "a $HOME path inside a knob value leaked into the manifest"
        assert m["config"]["EXTRA_DOCKER_ARGS"] == "-v ~/overlay/x.py:/sgl/x.py:ro -v ~/y:/y:ro"


def test_rescrub_leaked_manifest():
    """A manifest written by the prefix-only scrubber: rescrub fixes the value, keeps the manifest
    verifying, records the old hash, and re-stamps every sibling in place (formatting untouched)."""
    with tempfile.TemporaryDirectory() as home:
        d = pathlib.Path(home) / "run"
        d.mkdir()
        m = make_manifest(home)
        m["config"]["EXTRA_DOCKER_ARGS"] = f"-v {home}/overlay/x.py:/sgl/x.py:ro"
        m = em.finalize(m)  # a leaked but validly hashed manifest, as the old writer produced
        old = m["content_hash"]
        (d / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
        a = d / "a.json"
        a.write_text(json.dumps({"tps": 1, "evidence": {"tool": "t", "recorded_utc": "x", "manifest_hash": old}}, indent=2) + "\n")
        b = d / "b.json"  # stamped with another hash: left alone
        b.write_text(json.dumps({"evidence": {"tool": "t", "recorded_utc": "x", "manifest_hash": "f" * 64}}))
        got_old, new, files = em.rescrub(d, home)
        assert got_old == old and new != old and files == [a]
        nm = em.load(d / "manifest.json")
        assert nm["rescrubbed_from"] == old and home not in json.dumps(nm)
        assert nm["config"]["EXTRA_DOCKER_ARGS"] == "-v ~/overlay/x.py:/sgl/x.py:ro"
        assert json.loads(a.read_text())["evidence"]["manifest_hash"] == new and a.read_text().endswith("}\n")
        assert "  \"tps\": 1" in a.read_text()  # indent=2 preserved
        assert json.loads(b.read_text())["evidence"]["manifest_hash"] == "f" * 64
        problems = em.check_evidence(d.parent)
        assert [p for p in problems if "b.json" not in p] == [], problems  # only the deliberately foreign b.json
        # idempotent
        assert em.rescrub(d, home)[:2] == (new, new)


def _write_manifest(home: str, **overrides) -> tuple[pathlib.Path, dict]:
    d = pathlib.Path(home) / "mf"
    d.mkdir(exist_ok=True)
    m = make_manifest(home, **overrides)
    p = d / "current.json"
    p.write_text(json.dumps(m))
    return p, m


def test_stamp_sibling_rules():
    with tempfile.TemporaryDirectory() as home:
        p, m = _write_manifest(home)
        env = {**base_env(home), "EVIDENCE_MANIFEST": str(p)}
        live = lambda name: "c" * 64  # noqa: E731
        out = pathlib.Path(home) / "evidence/run/a.json"
        r = em.stamp({"x": 1}, out, "t.py", env=env, container_id=live, now=lambda: FIXED_NOW)
        ev = r["evidence"]
        assert ev["manifest_hash"] == m["content_hash"] and ev["container_check"] == "match"
        assert ev["tool"] == "t.py" and ev["manifest_file"] == "manifest.json" and "allowed_stale" not in ev
        sibling = out.parent / "manifest.json"
        assert em.load(sibling)["content_hash"] == m["content_hash"]
        # same manifest again in the same dir: fine (second tool of the same serve)
        em.stamp({}, out.parent / "b.json", "u.py", env=env, container_id=live, now=lambda: FIXED_NOW)
        # a different serve's manifest into the same dir: refused, sibling untouched
        p2, m2 = _write_manifest(home, hsmp_sample=lambda: {"fclk_mhz": 2000})
        assert m2["content_hash"] != m["content_hash"]
        try:
            em.stamp({}, out.parent / "c.json", "v.py", env={**env, "EVIDENCE_MANIFEST": str(p2)}, container_id=live,
                     now=lambda: FIXED_NOW)
            raise AssertionError("second manifest accepted into the same evidence dir")
        except em.ManifestError:
            pass
        assert em.load(sibling)["content_hash"] == m["content_hash"]
        # default lookup goes through MANIFEST_DIR; a missing file is an error with guidance, not a silent null stamp
        try:
            em.stamp({}, out, "t.py", env={**base_env(home), "MANIFEST_DIR": f"{home}/nowhere"}, container_id=live)
            raise AssertionError("missing manifest stamped")
        except em.ManifestError as e:
            assert "EVIDENCE_MANIFEST=none" in str(e)


def test_stamp_container_check():
    with tempfile.TemporaryDirectory() as home:
        p, m = _write_manifest(home)
        env = {**base_env(home), "EVIDENCE_MANIFEST": str(p)}
        out = pathlib.Path(home) / "ev/a.json"
        stale = lambda name: "d" * 64  # noqa: E731
        try:
            em.stamp({}, out, "t.py", env=env, container_id=stale)
            raise AssertionError("stale container stamped without ALLOW_STALE")
        except em.ManifestError as e:
            assert "ALLOW_STALE" in str(e)
        assert not (out.parent / "manifest.json").exists(), "refused stamp must not leave a sibling"
        r = em.stamp({}, out, "t.py", env={**env, "EVIDENCE_MANIFEST_ALLOW_STALE": "1"}, container_id=stale)
        assert r["evidence"]["container_check"] == "stale" and r["evidence"]["allowed_stale"] is True

        def gone(name):
            raise RuntimeError("docker: no such container")

        r = em.stamp({}, out, "t.py", env=env, container_id=gone)
        assert r["evidence"]["container_check"].startswith("unavailable")
        # a manifest written without a container (bench against a hand-run server) skips the check
        p0, _ = _write_manifest(home, container=None)
        r = em.stamp({}, pathlib.Path(home) / "ev0/a.json", "t.py", env={**env, "EVIDENCE_MANIFEST": str(p0)},
                     container_id=stale)
        assert r["evidence"]["container_check"] == "skipped" and r["evidence"]["container_id"] is None


def test_stamp_none_is_explicit():
    with tempfile.TemporaryDirectory() as home:
        out = pathlib.Path(home) / "ev/a.json"
        r = em.stamp({}, out, "t.py", env={**base_env(home), "EVIDENCE_MANIFEST": "none"}, now=lambda: FIXED_NOW)
        assert r["evidence"]["manifest_hash"] is None and r["evidence"]["unmanifested"] == "EVIDENCE_MANIFEST=none"
        assert "recorded_utc" in r["evidence"] and not (out.parent / "manifest.json").exists()


def test_check_evidence():
    with tempfile.TemporaryDirectory() as home:
        root = pathlib.Path(home) / "evidence"
        p, m = _write_manifest(home)
        env = {**base_env(home), "EVIDENCE_MANIFEST": str(p)}
        live = lambda name: "c" * 64  # noqa: E731
        # legacy file (no `evidence` key) and a non-dict JSON: exempt
        (root / "old").mkdir(parents=True)
        (root / "old/run.json").write_text(json.dumps({"tps": 40.3}))
        (root / "old/list.json").write_text("[1, 2]")
        # stamped + sibling
        good = root / "good/a.json"
        r = em.stamp({"tps": 1}, good, "t.py", env=env, container_id=live)
        good.write_text(json.dumps(r))
        # explicitly unmanifested
        (root / "none").mkdir()
        rn = em.stamp({}, root / "none/a.json", "t.py", env={**env, "EVIDENCE_MANIFEST": "none"})
        (root / "none/a.json").write_text(json.dumps(rn))
        assert em.check_evidence(root) == []

        # null hash with no reason
        bad = json.loads(json.dumps(rn))
        del bad["evidence"]["unmanifested"]
        (root / "none/b.json").write_text(json.dumps(bad))
        # stamped but sibling missing
        (root / "orphan").mkdir()
        (root / "orphan/a.json").write_text(json.dumps(r))
        # sibling tampered after the fact
        (root / "tamper").mkdir()
        (root / "tamper/a.json").write_text(json.dumps(r))
        t = json.loads((good.parent / "manifest.json").read_text())
        t["config"]["EXPERTS_GPU"] = "1"
        (root / "tamper/manifest.json").write_text(json.dumps(t))
        # stamped with a hash that is not the (valid) sibling's
        (root / "mismatch").mkdir()
        (root / "mismatch/manifest.json").write_text(json.dumps(make_manifest(home, hsmp_sample=lambda: None)))
        (root / "mismatch/a.json").write_text(json.dumps(r))
        problems = em.check_evidence(root)
        hit = {k for k in ("none/b.json", "orphan/a.json", "tamper/manifest.json", "mismatch/a.json") if any(k in s for s in problems)}
        assert hit == {"none/b.json", "orphan/a.json", "tamper/manifest.json", "mismatch/a.json"}, problems
        assert not any("good" in s or "old" in s for s in problems), problems


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"ok: {len(fns)} evidence-manifest tests")
    sys.exit(0)

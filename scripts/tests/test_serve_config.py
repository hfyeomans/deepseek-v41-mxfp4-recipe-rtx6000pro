#!/usr/bin/env python3
"""Tests for the typed knob registry (scripts/serve_config.py, T3.19e) and the launcher
that evals it (scripts/serve-hybrid.sh).

No docker, no GPU: every launcher case runs with DRY_RUN=1 or --print. The wrong
implementations these are written against:
  - the registry drifting from the launcher the goldens were captured from (before any
    launcher change, 2026-10-08): the `resolved:` line and the `--print` docker command
    must stay byte-identical for the qualified profiles -> test_goldens.
  - a mistyped knob (KT_FOO) silently ignored, so an experiment runs the default while
    its log claims otherwise -> test_unknown_knob_refused.
  - a removed knob accepted again -> test_removed_knob_refused.
  - an unparsable value reaching docker (SGLang fails minutes later) -> test_mistyped_value.
  - a diagnostic knob (KT_TIMELINE) set on the host but never forwarded, so the overlay
    never sees it -> test_diagnostic_forwarded; and the converse: forwarded even when
    unset, which would change every default launch -> test_goldens (print-default).
  - the launcher forwarding env the registry does not know, or the overlay reading env
    the registry does not declare -> test_overlay_reads_declared (skipped without the overlay).
  - (T3.24a) a profile value silently losing to the registry default, or winning over the
    environment, so a serve runs a shape its log does not name -> test_profile_resolution_order;
    a profile carrying a mistyped / unknown / removed / unsafe knob -> test_profile_refusals;
    the shipped qualified.env drifting from the registry default, so `PROFILE=qualified` and
    an empty environment stop meaning the same serve -> test_shipped_profiles_match_registry.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVE = ROOT / "scripts/serve-hybrid.sh"
GOLDEN = ROOT / "scripts/tests/golden"
OVERLAY = Path(os.environ.get("OVERLAY_DIR", str(Path.home() / "dev/src/sglang-dsv41-overlay")))

sys.path.insert(0, str(ROOT / "scripts"))
import serve_config as sc  # noqa: E402

BASE_ENV = {k: v for k, v in os.environ.items()
            if not (k.startswith(sc.OWNED_PREFIXES) or k in sc.BY_NAME)}
BASE_ENV.setdefault("HOME", str(Path.home()))


def launcher(args: list[str], **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SERVE), *args], env={**BASE_ENV, **env},
                          capture_output=True, text=True, check=False)


def portable(text: str) -> str:
    return text.replace(str(ROOT), "<repo>").replace(BASE_ENV["HOME"], "~")


def test_goldens() -> None:
    cases = {
        "resolved-default.txt": ([], {"DRY_RUN": "1"}),
        "resolved-512k-1slot.txt": ([], {"DRY_RUN": "1", "CONTEXT_LENGTH": "524288",
                                         "MAX_RUNNING_REQUESTS": "1", "MEM_FRACTION_STATIC": "0.962"}),
        "resolved-262k-226-4096.txt": ([], {"DRY_RUN": "1", "CONTEXT_LENGTH": "262144", "MAX_RUNNING_REQUESTS": "1",
                                            "KT_NUM_GPU_EXPERTS": "226", "CHUNKED_PREFILL_SIZE": "4096"}),
        "resolved-dspark-208.txt": ([], {"DRY_RUN": "1", "DSPARK": "1", "KT_NUM_GPU_EXPERTS": "208",
                                         "MEM_FRACTION_STATIC": "0.935"}),
        "resolved-ngram.txt": ([], {"DRY_RUN": "1", "NGRAM": "1"}),
        "print-default.txt": (["--print"], {}),
        "print-empties-extras.txt": (["--print"], {"PROFILE": "developer", "CUDA_GRAPH_BS": "", "MAX_TOTAL_TOKENS": "",
                                                   "EXTRA_DOCKER_ENV": "CUDA_LAUNCH_BLOCKING=1 KT_TIMELINE=1",
                                                   "EXTRA_SERVER_ARGS": "--log-level debug"}),
    }
    for name, (args, env) in cases.items():
        p = launcher(args, **env)
        assert p.returncode == 0, (name, p.returncode, p.stderr)
        want = (GOLDEN / name).read_text()
        got = portable(p.stdout)
        assert got == want, f"{name} drifted:\n--- want\n{want}\n--- got\n{got}"


def test_unknown_knob_refused() -> None:
    p = launcher([], DRY_RUN="1", KT_FOO="1")
    assert p.returncode == 2 and "FAIL config: unknown knob(s) KT_FOO" in p.stderr, (p.returncode, p.stderr)
    p = launcher([], DRY_RUN="1", KT_FOO="1", ALLOW_UNKNOWN_KNOBS="1", PROFILE="developer")  # T3.24b: developer-only
    assert p.returncode == 0 and p.stdout.startswith("resolved:"), (p.returncode, p.stderr)


def test_removed_knob_refused() -> None:
    for name in ("KT_B2_UVA_EXPERTS", "KT_B2_MAX_TOKENS"):
        p = launcher([], DRY_RUN="1", **{name: "1"})
        assert p.returncode == 2 and "removed at T3.16j" in p.stderr, (name, p.returncode, p.stderr)


def test_mistyped_value() -> None:
    for name, bad in (("KT_CPUINFER", "sixteen"), ("MEM_FRACTION_STATIC", "0.96x"),
                      ("DSPARK", "yes"), ("CUDA_GRAPH_BS", "1 two")):
        p = launcher([], DRY_RUN="1", **{name: bad})
        assert p.returncode == 2 and f"FAIL config: {name}=" in p.stderr, (name, p.returncode, p.stderr)


def test_empty_disables() -> None:
    r = sc.resolve({"CUDA_GRAPH_BS": "", "MAX_TOTAL_TOKENS": ""})
    assert r["CUDA_GRAPH_BS"] == "" and r["MAX_TOTAL_TOKENS"] == ""
    assert "capture_bs=[derived<=4]" in sc.resolved_line(r) and "max_total_tokens=fraction-sized" in sc.resolved_line(r)
    r = sc.resolve({"MAX_RUNNING_REQUESTS": "1", "CONTEXT_LENGTH": "262144"})
    assert r["MAX_TOTAL_TOKENS"] == str(262144 + 16384)


def test_expert_map_knob() -> None:
    # T3.8b: a map is mounted at /expert-map.json and named to the overlay. Default (owner flip
    # 2026-10-10) = the repo's el8 mix map; KT_EXPERT_MAP="" = checkpoint order, and then neither
    # the mount nor the server arg appears and the resolved line says `positional`.
    p = launcher(["--print"], KT_EXPERT_MAP="/tmp/some-map.json")
    assert p.returncode == 0, p.stderr
    line, cmd = p.stdout.splitlines()[:2]
    assert " expert_map=some-map.json " in line, line
    assert "-v /tmp/some-map.json:/expert-map.json:ro" in cmd and "--kt-expert-map /expert-map.json" in cmd, cmd
    default = (GOLDEN / "print-default.txt").read_text().splitlines()
    assert " expert_map=expert-map-el8-mix-219.json " in default[0], default[0]
    assert "-v <repo>/config/expert-map-el8-mix-219.json:/expert-map.json:ro" in default[1], default[1]
    assert (ROOT / "config/expert-map-el8-mix-219.json").is_file()
    p = launcher(["--print"], KT_EXPERT_MAP="")
    assert p.returncode == 0, p.stderr
    line, cmd = p.stdout.splitlines()[:2]
    assert " expert_map=positional " in line and "expert-map" not in cmd, (line, cmd)
    r = sc.resolve({"KT_EXPERT_MAP": ""})
    assert r["KT_EXPERT_MAP"] == "" and "expert_map=positional" in sc.resolved_line(r)


def test_diagnostic_forwarded() -> None:
    p = launcher(["--print"], PROFILE="developer", KT_TIMELINE="1", KT_TIMELINE_DIR="/tmp/tl",
                 SGLANG_BCG_DEBUG_SYNC="1", KT_NAN_PROBE_MIN_TOKENS="1")
    assert p.returncode == 0, p.stderr
    cmd = p.stdout.splitlines()[1]
    for kv in ("KT_TIMELINE=1", "KT_TIMELINE_DIR=/tmp/tl", "SGLANG_BCG_DEBUG_SYNC=1", "KT_NAN_PROBE_MIN_TOKENS=1"):
        assert f"-e {kv} " in cmd, (kv, cmd)
    # the diagnostics sit after the registry env and before EXTRA_DOCKER_ENV
    assert cmd.index("SGLANG_DSV41_REJECT_NAN_ROWS=1") < cmd.index("KT_TIMELINE=1")
    default_cmd = (GOLDEN / "print-default.txt").read_text().splitlines()[1]
    # T3.24b: the NaN probe is a diagnostic, forwarded only when set (the default docker command does not carry it)
    assert "KT_TIMELINE" not in default_cmd and "KT_NAN_PROBE_MIN_TOKENS" not in default_cmd


def test_developer_only_knobs_in_the_environment() -> None:
    # T3.24b: unsafe flags, EXTRA_* and diagnostics set to a non-default value are refused under
    # any profile but developer, with the message naming PROFILE=developer; at their default they
    # are not an experiment, and under developer they resolve and are listed as overrides
    for env, needle in (({"ALLOW_UNTESTED_FIT": "1"}, "ALLOW_UNTESTED_FIT is unsafe"),
                        ({"KT_ALLOW_UNSAFE_DEFERRAL": "1"}, "KT_ALLOW_UNSAFE_DEFERRAL is unsafe"),
                        ({"ALLOW_UNKNOWN_KNOBS": "1", "KT_FOO": "1"}, "ALLOW_UNKNOWN_KNOBS is unsafe"),
                        ({"EXTRA_SERVER_ARGS": "--log-level debug"}, "EXTRA_SERVER_ARGS is an escape hatch"),
                        ({"KT_NAN_PROBE_MIN_TOKENS": "1"}, "KT_NAN_PROBE_MIN_TOKENS is a diagnostic"),
                        ({"KT_NAN_INJECT": "row=0,bs=1,step=3,layer=5"}, "KT_NAN_INJECT is a diagnostic"),
                        ({"KT_TIMELINE": "1"}, "KT_TIMELINE is a diagnostic")):
        try:
            sc.resolve(env)
        except sc.ConfigError as e:
            assert needle in str(e) and "PROFILE=developer" in str(e), (env, e)
        else:
            raise AssertionError(f"{env}: accepted under the qualified profile")
    r = sc.resolve({"ALLOW_UNTESTED_FIT": "0", "KT_TIMELINE": "0", "EXTRA_SERVER_ARGS": ""})
    assert r["PROFILE"] == "qualified" and r["PROFILE_OVERRIDES"] == "EXTRA_SERVER_ARGS ALLOW_UNTESTED_FIT KT_TIMELINE", r
    r = sc.resolve({"PROFILE": "developer", "ALLOW_UNTESTED_FIT": "1", "KT_NAN_PROBE_MIN_TOKENS": "1"})
    assert r["ALLOW_UNTESTED_FIT"] == "1" and r["KT_NAN_PROBE_MIN_TOKENS"] == "1"
    assert r["PROFILE_OVERRIDES"] == "KT_NAN_PROBE_MIN_TOKENS ALLOW_UNTESTED_FIT", r
    # a mistyped value is still a type error, not a profile error
    try:
        sc.resolve({"ALLOW_UNTESTED_FIT": "yes"})
    except sc.ConfigError as e:
        assert "PROFILE=developer" not in str(e) and "expected flag" in str(e), e
    else:
        raise AssertionError("mistyped flag accepted")


def test_guards_match_launcher_messages() -> None:
    # the registry raises the exact messages test-serve-config.sh pins; probe two directly
    for env, needle in (({"KT_MAX_DEFERRED": "5"}, "D>=5 fails the NLL gate"),
                        ({"CONTEXT_LENGTH": "524288", "MAX_RUNNING_REQUESTS": "2", "MEM_FRACTION_STATIC": "0.962"},
                         "Larger multi-slot pools were never run"),
                        ({"NGRAM": "1", "DSPARK": "1", "KT_NUM_GPU_EXPERTS": "208"}, "two speculative algorithms"),
                        ({"NGRAM": "1", "NGRAM_DRAFT_TOKENS": "13"}, "outside 2..12")):
        try:
            sc.guards(sc.resolve(env))
        except sc.ConfigError as e:
            assert needle in str(e), (env, e)
        else:
            raise AssertionError(f"{env} not refused")


def test_registry_declares_everything_launcher_uses() -> None:
    text = SERVE.read_text()
    for k in sc.REGISTRY:
        if k.removed:
            assert k.name not in text, f"removed knob {k.name} still in the launcher"
    for k in sc.REGISTRY:
        if k.sink == sc.CONTAINER_ENV:
            assert (k.container_name or k.name) in " ".join(sc.container_env(sc.resolve({})))
    # every env the launcher forwards with -e comes from the registry, not a hard-coded line
    assert "-e SGLANG_" not in text and '-e "KT_' not in text


def _profile(tmp: str, name: str, body: str) -> str:
    path = Path(tmp) / f"{name}.env"
    path.write_text(body)
    return str(path)


def test_profile_resolution_order() -> None:
    # registry default (chunk 8192, 4 slots) < profile (chunk 4096, 1 slot) < environment (slots 2):
    # a wrong implementation reads the profile over the env, or the default over the profile.
    with tempfile.TemporaryDirectory() as tmp:
        p = _profile(tmp, "exp", "# comment\n\nKT_NUM_GPU_EXPERTS=226\nCHUNKED_PREFILL_SIZE=4096\n"
                                 "MAX_RUNNING_REQUESTS='1'\nKT_EXPERT_MAP={repo}/config/expert-map-el8-mix-219.json\n")
        r = sc.resolve({"PROFILE": p, "MAX_RUNNING_REQUESTS": "2", "DRY_RUN": "1"})
        assert r["CHUNKED_PREFILL_SIZE"] == "4096" and r["MAX_RUNNING_REQUESTS"] == "2", r
        assert r["KT_CPUINFER"] == "16"                              # untouched registry default
        assert r["KT_EXPERT_MAP"] == str(ROOT / "config/expert-map-el8-mix-219.json")   # {repo} expanded
        assert r["PROFILE"] == p and r["PROFILE_OVERRIDES"] == "MAX_RUNNING_REQUESTS", r  # DRY_RUN never listed
        assert sc.resolved_line(r).endswith(f"profile={p} overrides=[MAX_RUNNING_REQUESTS]")
        # through the launcher: the resolved line names the profile and the overrides (226 / 4096 x 1 is a
        # qualified rung above 64K, E34b-ii) ...
        q = launcher([], DRY_RUN="1", PROFILE=p, CONTEXT_LENGTH="262144", MAX_RUNNING_REQUESTS="1")
        assert q.returncode == 0 and q.stdout.rstrip().endswith(
            f"profile={p} overrides=[CONTEXT_LENGTH MAX_RUNNING_REQUESTS]"), (q.returncode, q.stdout, q.stderr)
        # ... and the guards judge the profile's values, not the registry defaults (226 at chunk 8192 is refused)
        q = launcher([], DRY_RUN="1", PROFILE=p, CHUNKED_PREFILL_SIZE="8192")
        assert q.returncode == 2 and "KT_NUM_GPU_EXPERTS<=219" in q.stderr, (q.returncode, q.stderr)
    # the shipped profiles resolve by bare name; an empty env reports no overrides
    r = sc.resolve({})
    assert r["PROFILE"] == "qualified" and r["PROFILE_OVERRIDES"] == ""
    r = sc.resolve({"PROFILE": "developer"})
    assert r["PROFILE"] == "developer" and r["CHUNKED_PREFILL_SIZE"] == "8192"


def test_profile_refusals() -> None:
    cases = {
        "unknown key": ("KT_FOO=1\n", "KT_FOO is not a knob"),
        "removed knob": ("KT_B2_UVA_EXPERTS=1\n", "removed at T3.16j"),
        "mistyped value": ("KT_CPUINFER=sixteen\n", "KT_CPUINFER='sixteen': expected int"),
        "malformed line": ("export KT_CPUINFER=16\n", "expected KNOB=value"),
        "duplicate key": ("KT_CPUINFER=16\nKT_CPUINFER=14\n", "KT_CPUINFER set twice"),
        "profile sets PROFILE": ("PROFILE=qualified\n", "PROFILE cannot be set from a profile"),
        "unsafe knob outside developer": ("ALLOW_UNTESTED_FIT=1\n", "only PROFILE=developer (a file named developer.env) may carry it"),
        "escape hatch outside developer": ("EXTRA_SERVER_ARGS=--log-level debug\n", "only PROFILE=developer"),
        "diagnostic outside developer": ("KT_NAN_PROBE_MIN_TOKENS=1\n", "is a diagnostic; only PROFILE=developer"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, (body, needle) in cases.items():
            p = _profile(tmp, "bad", body)
            try:
                sc.resolve({"PROFILE": p})
            except sc.ConfigError as e:
                assert needle in str(e), (name, e)
            else:
                raise AssertionError(f"{name}: profile accepted")
        # the same knobs are accepted by a file named developer.env, wherever it lives (a local copy)
        p = _profile(tmp, "developer", "ALLOW_UNTESTED_FIT=1\nEXTRA_SERVER_ARGS=--log-level debug\n")
        r = sc.resolve({"PROFILE": p})
        assert r["ALLOW_UNTESTED_FIT"] == "1" and r["EXTRA_SERVER_ARGS"] == "--log-level debug", r
    # missing profile, empty name, derived knob set by hand: all refused at the launcher
    for env, needle in (({"PROFILE": "nope"}, "cannot read"), ({"PROFILE": ""}, "name a profile"),
                        ({"PROFILE_OVERRIDES": "x"}, "derived from the environment")):
        p = launcher([], DRY_RUN="1", **env)
        assert p.returncode == 2 and needle in p.stderr, (env, p.returncode, p.stderr)


def test_shipped_profiles_match_registry() -> None:
    names = sorted(p.stem for p in (ROOT / "config/profiles").glob("*.env"))
    assert names == ["developer", "qualified"], names
    for name in names:
        assert sc.profile_drift(name) == [], (name, sc.profile_drift(name))
        prof = sc.load_profile(name)
        # the profile states the serve shape explicitly rather than inheriting it silently
        for knob in ("KT_NUM_GPU_EXPERTS", "KT_EXPERT_MAP", "KT_MAX_DEFERRED", "KT_CPUINFER", "CHUNKED_PREFILL_SIZE",
                     "CONTEXT_LENGTH", "MAX_RUNNING_REQUESTS", "MEM_FRACTION_STATIC", "KT_REJECT_NAN_ROWS"):
            assert knob in prof, (name, knob)
    # drift is detected: a profile that disagrees with the registry names the knob and both values
    with tempfile.TemporaryDirectory() as tmp:
        p = _profile(tmp, "drift", "KT_CPUINFER=14\n")
        assert sc.profile_drift(p) == ["KT_CPUINFER: profile='14' registry='16'"]
    cp = subprocess.run([sys.executable, str(ROOT / "scripts/serve_config.py"), "check-profiles"],
                        capture_output=True, text=True, check=False)
    assert cp.returncode == 0 and cp.stdout.startswith("ok profiles developer qualified"), (cp.stdout, cp.stderr)


def test_overlay_reads_declared() -> None:
    if not (OVERLAY / "python/sglang").is_dir():
        print(f"     overlay {OVERLAY} not found (set OVERLAY_DIR); overlay read check skipped")
        return
    missing = sc.undeclared_overlay_reads(OVERLAY)
    assert not missing, f"overlay reads env the registry does not declare: {sorted(missing)}"
    reads = sc.overlay_env_reads(OVERLAY)
    assert "KT_GPU_PREFILL_MIN_TOKENS" in reads and "KT_TIMELINE" in reads, sorted(reads)


def main() -> int:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok   {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

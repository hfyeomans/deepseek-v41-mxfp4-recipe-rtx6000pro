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
"""
from __future__ import annotations

import os
import subprocess
import sys
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
        "print-empties-extras.txt": (["--print"], {"CUDA_GRAPH_BS": "", "MAX_TOTAL_TOKENS": "",
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
    p = launcher([], DRY_RUN="1", KT_FOO="1", ALLOW_UNKNOWN_KNOBS="1")
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
    p = launcher(["--print"], KT_TIMELINE="1", KT_TIMELINE_DIR="/tmp/tl", SGLANG_BCG_DEBUG_SYNC="1")
    assert p.returncode == 0, p.stderr
    cmd = p.stdout.splitlines()[1]
    for kv in ("KT_TIMELINE=1", "KT_TIMELINE_DIR=/tmp/tl", "SGLANG_BCG_DEBUG_SYNC=1"):
        assert f"-e {kv} " in cmd, (kv, cmd)
    # the diagnostics sit after the registry env and before EXTRA_DOCKER_ENV
    assert cmd.index("KT_NAN_PROBE_MIN_TOKENS=0") < cmd.index("KT_TIMELINE=1")
    assert "KT_TIMELINE" not in (GOLDEN / "print-default.txt").read_text().splitlines()[1]


def test_guards_match_launcher_messages() -> None:
    # the registry raises the exact messages test-serve-config.sh pins; probe two directly
    for env, needle in (({"KT_MAX_DEFERRED": "5"}, "D>=5 fails the NLL gate"),
                        ({"CONTEXT_LENGTH": "262144"}, "above 64K only single-slot rungs"),
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

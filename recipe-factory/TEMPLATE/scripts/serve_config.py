#!/usr/bin/env python3
"""Typed knob registry template for a new serving recipe (model- and engine-agnostic).

Copy this file to scripts/serve_config.py and adapt REGISTRY. The pattern is lifted from the
DeepSeek-V4.1 hybrid recipe's scripts/serve_config.py (T3.19e) — read that file for the fully
worked version: knob kinds, sinks, profile layering, guard messages with measured numbers.

Commands:
    python3 scripts/serve_config.py shell          # KEY='value' lines for `eval`, or rc 2
    python3 scripts/serve_config.py resolved       # the one-line resolved config
    python3 scripts/serve_config.py markdown       # knob table for docs/CONFIG.md
    python3 scripts/serve_config.py check-profiles # every config/profiles/*.env loads and equals the defaults

Rules the registry enforces before any launch:
- an environment variable in an owned name family (OWNED_PREFIXES) that is not a declared knob is refused
  (rc 2), so a mistyped override cannot launch the default while the log claims otherwise;
- an `unsafe` knob, an EXTRA_* escape hatch or a diagnostic set to a non-default value is refused under
  every profile but `developer`;
- a removed knob is refused with the experiment that removed it;
- a value that does not parse as its type is refused;
- GUARDS refuse combinations the evidence condemned, each naming its override.

Profiles: PROFILE=<name> (default `qualified`) loads config/profiles/<name>.env, layered:
    registry default  <  profile file  <  environment
so an env override is an experiment on top of a named profile, and the resolved line + launch manifest
record the profile and every override. The CI test keeps the shipped `qualified` profile equal to the
registry defaults, so the two cannot drift (the registry is the single source of truth).
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- sinks: where the resolved value is consumed -----------------------------------------
SERVER_ARG = "server-arg"        # becomes an engine CLI flag
CONTAINER_ENV = "container-env"  # always forwarded into the runtime
DIAGNOSTIC_ENV = "diagnostic-env"  # forwarded only when the caller sets it
LAUNCHER = "launcher"            # used by serve.sh itself (paths, guards, docker run)
GUARD = "guard"                  # exists only to bypass a refusal


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str                       # int | float | str | flag | list
    default: Any                    # literal, or callable(resolved: dict) -> str
    doc: str
    sink: str = LAUNCHER
    evidence: str = ""              # evidence/<dir>/ that set this default
    aliases: tuple[str, ...] = ()
    unsafe: bool = False            # bypasses a guard; never a deployment value
    empty_disables: bool = False    # VAR="" means off / "let the engine derive it"
    container_name: str = ""        # env name inside the container when it differs
    removed: str = ""               # non-empty: refuse with this message


def _repo(p: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), p)


PROFILE_DIR = _repo("config/profiles")
DEVELOPER_PROFILE = "developer"
PROFILE_ONLY_DEVELOPER = ("EXTRA_SERVER_ARGS", "EXTRA_DOCKER_ENV", "EXTRA_DOCKER_ARGS")
NOT_OVERRIDES = ("PROFILE", "PROFILE_OVERRIDES", "DRY_RUN")


def developer_only(knob: Knob) -> str:
    """Why `knob` may only be set under PROFILE=developer ("" when any profile may set it)."""
    if knob.unsafe:
        return "unsafe"
    if knob.name in PROFILE_ONLY_DEVELOPER:
        return "an escape hatch"
    if knob.sink == DIAGNOSTIC_ENV:
        return "a diagnostic"
    return ""


# Name families this registry owns: an env var that starts with one of these and is not a declared knob is a
# typo (refused). Explicit families, not "every prefix a knob has", so host variables the launcher must pass
# through untouched are never caught. Customize per engine: ENGINE_*, MEM_*, SPEC_*, KV_* ...
OWNED_PREFIXES = ("ENGINE_", "MEM_", "SPEC_", "KV_", "CTX_", "EXTRA_", "ALLOW_", "MODEL_DIR", "SNAPSHOT")


# ------------------------------------------------------------------------------------------
# REGISTRY — declare every knob once. The entries below are the generic families every recipe
# needs; replace the defaults and add the engine-specific knobs. Every doc line must name the
# evidence directory that set the default, or say "to be measured" until it exists.
# ------------------------------------------------------------------------------------------
REGISTRY: tuple[Knob, ...] = (
    # ---- profile -----------------------------------------------------------------------
    Knob("PROFILE", "str", "qualified",
         "Profile file to load under the environment (config/profiles/<name>.env, or a path). "
         "Registry default < profile < environment. `qualified` = the serve shape every headline "
         "number was measured on; `developer` = the same shape and the only profile allowed to "
         "carry EXTRA_* / ALLOW_* knobs.", evidence="config/profiles/"),
    Knob("PROFILE_OVERRIDES", "str", "",
         "Derived, not settable: the knobs the environment set on top of the profile."),
    # ---- identity / paths --------------------------------------------------------------
    Knob("MODEL_DIR", "str", "/models/<org>/<model>",
         "Checkpoint directory (bind-mounted read-only). Pin the snapshot below.", sink=LAUNCHER),
    Knob("SNAPSHOT", "str", "<40-hex-sha>",
         "Immutable checkpoint snapshot under MODEL_DIR/snapshots.", evidence="PROVENANCE.md"),
    Knob("SERVED_MODEL_NAME", "str", "<model>-local", "Name the API serves.", sink=SERVER_ARG),
    Knob("BIND_ADDRESS", "str", "127.0.0.1",
         "Host interface the port publishes on. A non-loopback bind without an API key must refuse."),
    Knob("API_KEY_FILE", "str", "",
         "Path to a file holding one bearer token; non-empty mounts it read-only and the server "
         "requires it. A path, not the key, so the manifest never carries the secret."),
    Knob("PORT", "int", "8000", "Host port the API publishes on."),
    # ---- memory / context / slots (the KV-cache triangle) --------------------------------
    Knob("MEM_FRACTION_STATIC", "float", "0.95",
         "Engine memory fraction. Cap the KV pool explicitly (MAX_TOTAL_TOKENS), do not let the "
         "fraction size it: engines oversize pools by large factors. Evidence: to be measured."),
    Knob("CONTEXT_LENGTH", "int", "32768",
         "Served context. The ladder rungs (32K x 4 -> ... -> max x 1) each get qualified at their "
         "own fraction; values above the qualified max must refuse."),
    Knob("MAX_RUNNING_REQUESTS", "int", "4",
         "Decode slots. Above some context, only 1 slot was ever qualified; guard it."),
    Knob("MAX_TOTAL_TOKENS", "int",
         lambda r: str(int(r["MAX_RUNNING_REQUESTS"]) * int(r["CONTEXT_LENGTH"]) + 16384),
         "KV pool cap = slots x context + one prefill chunk. Empty = let the engine derive it.",
         sink=SERVER_ARG, empty_disables=True),
    Knob("CHUNKED_PREFILL_SIZE", "int", "4096",
         "Prefill chunk. Bigger chunks buy TTFT at a workspace cost; measure the trade per chunk "
         "size and encode the winning pairs in GUARDS. Evidence: to be measured."),
    Knob("CUDA_GRAPH_MAX_BS", "int", "4", "Largest captured decode batch.", sink=SERVER_ARG),
    Knob("CUDA_GRAPH_BS", "list", "1 2 3 4",
         "Explicit decode capture list (derived lists pad odd batches; capture them explicitly). "
         "Empty = derived.", sink=SERVER_ARG, empty_disables=True),
    # ---- speculative decoding (default OFF in every recipe until gated) -------------------
    Knob("SPEC_ENABLE", "flag", "0",
         "Speculative decoding. On a CPU-hybrid MoE, verify rows multiply offload traffic; every "
         "spec path measured in the reference project lost at C3/C4. Gate any retry with the "
         "spec-consistency probe, not just NLL. Evidence: see TRIED-OPTIONS-LEDGER.md section 2.",
         sink=SERVER_ARG),
    Knob("SPEC_DRAFT_TOKENS", "int", "5", "Draft window (verify rows per step).", sink=SERVER_ARG),
    # ---- host threads -------------------------------------------------------------------
    Knob("HOST_THREAD_CPUS", "str", "auto",
         "cpulist (taskset syntax) the server process tree starts on. If the recipe has CPU-side "
         "workers (offload/hybrid), the server's other threads must stay off the workers' cores: "
         "floating threads caused a bimodal C4 in the reference project. `auto` = the workers' SMT "
         "siblings from sysfs. Empty = floating."),
    # ---- escape hatches / guard bypasses --------------------------------------------------
    Knob("EXTRA_SERVER_ARGS", "str", "", "Appended verbatim (word-split) to the engine command."),
    Knob("EXTRA_DOCKER_ENV", "str", "", "Space-separated KEY=VALUE pairs passed to the runtime."),
    Knob("EXTRA_DOCKER_ARGS", "str", "", "Extra runtime launch flags, word-split."),
    Knob("DRY_RUN", "flag", "0", "Stop after the resolved line, before any launch.", sink=GUARD),
    Knob("ALLOW_UNTESTED_FIT", "flag", "0", "Bypass the fit refusals to run an experiment on purpose.",
         sink=GUARD, unsafe=True),
    Knob("ALLOW_UNKNOWN_KNOBS", "flag", "0",
         "Accept variables in an owned name family this registry does not declare.", sink=GUARD, unsafe=True),
    # ---- removed (the pattern: keep the refusal forever) ----------------------------------
    Knob("ENGINE_OLD_LEVER", "str", "", "", sink=GUARD,
         removed="ENGINE_OLD_LEVER was removed at <experiment> (<why>); unset it."),
)

BY_NAME: dict[str, Knob] = {k.name: k for k in REGISTRY}
for _k in REGISTRY:
    for _a in _k.aliases:
        assert _a not in BY_NAME, _a
        BY_NAME[_a] = _k


class ConfigError(Exception):
    pass


def _parse(knob: Knob, raw: str) -> str:
    """Validate `raw` as knob.kind; return it unchanged (the shell consumes strings)."""
    if raw == "" and knob.empty_disables:
        return ""
    try:
        if knob.kind == "int":
            int(raw)
        elif knob.kind == "float":
            float(raw)
        elif knob.kind == "flag":
            if raw not in ("0", "1"):
                raise ValueError
        elif knob.kind == "list":
            for tok in raw.split():
                int(tok)
        elif knob.kind != "str":
            raise AssertionError(knob.kind)
    except ValueError:
        raise ConfigError(f"{knob.name}={raw!r}: expected {knob.kind}") from None
    return raw


_PROFILE_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.*)$")


def profile_path(name: str) -> str:
    """`PROFILE` value -> file: a bare name maps into config/profiles/, anything with a separator is a path."""
    return name if os.sep in name else os.path.join(PROFILE_DIR, f"{name}.env")


def load_profile(name: str) -> dict[str, str]:
    """Parse config/profiles/<name>.env into knob -> raw value; raise ConfigError on a missing file, a
    malformed line, a key that is not a (non-removed, canonical-name) knob, a duplicate, PROFILE itself,
    or an unsafe / EXTRA_* knob outside `developer`."""
    path = profile_path(name)
    try:
        text = Path(path).read_text()
    except OSError as e:
        known = sorted(p.stem for p in Path(PROFILE_DIR).glob("*.env")) if os.path.isdir(PROFILE_DIR) else []
        raise ConfigError(f"PROFILE={name!r}: cannot read {path} ({e.strerror}); "
                          f"profiles in config/profiles/: {' '.join(known) or 'none'}") from None
    repo_root = _repo("")[:-1]
    developer = Path(path).stem == DEVELOPER_PROFILE   # by file name, so a local copy keeps its status
    out: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROFILE_LINE.match(line)
        if not m:
            raise ConfigError(f"{path}:{lineno}: expected KNOB=value, got {line!r}")
        key, value = m.group(1), m.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        value = value.replace("{repo}", repo_root)
        knob = BY_NAME.get(key)
        if knob is None:
            raise ConfigError(f"{path}:{lineno}: {key} is not a knob declared in scripts/serve_config.py")
        if knob.removed:
            raise ConfigError(f"{path}:{lineno}: {knob.removed}")
        if key != knob.name:
            raise ConfigError(f"{path}:{lineno}: {key} is an alias; profiles use the canonical name {knob.name}")
        if key in NOT_OVERRIDES:
            raise ConfigError(f"{path}:{lineno}: {key} cannot be set from a profile")
        if key in out:
            raise ConfigError(f"{path}:{lineno}: {key} set twice")
        if developer_only(knob) and not developer:
            raise ConfigError(f"{path}:{lineno}: {key} is {developer_only(knob)}; "
                              f"only PROFILE={DEVELOPER_PROFILE} may carry it")
        out[key] = _parse(knob, value)
    return out


def resolve(env: dict[str, str]) -> dict[str, str]:
    """Resolve every knob: registry default < profile file < `env`; raise ConfigError on an unknown,
    removed or mistyped knob, or an unsafe knob set under a non-developer profile."""
    for k in REGISTRY:
        if k.removed and any(n in env for n in (k.name, *k.aliases)):
            raise ConfigError(k.removed)
    if "PROFILE_OVERRIDES" in env:
        raise ConfigError("PROFILE_OVERRIDES is derived from the environment, not set in it")
    profile_name = _parse(BY_NAME["PROFILE"], env.get("PROFILE", BY_NAME["PROFILE"].default))
    if not profile_name:
        raise ConfigError("PROFILE='': name a profile (config/profiles/*.env) or leave it unset for `qualified`")
    profile = load_profile(profile_name)
    if Path(profile_path(profile_name)).stem != DEVELOPER_PROFILE:
        for k in REGISTRY:
            if k.removed or not developer_only(k):
                continue
            src = next((n for n in (k.name, *k.aliases) if n in env), None)
            if src is not None and not callable(k.default) and _parse(k, env[src]) != k.default:
                raise ConfigError(
                    f"{src}={env[src]}: {k.name} is {developer_only(k)}; PROFILE={profile_name} refuses it. "
                    f"Run the experiment as PROFILE={DEVELOPER_PROFILE}, which the resolved line and the "
                    f"manifest then record.")
    if env.get("ALLOW_UNKNOWN_KNOBS", "0") != "1":
        unknown = sorted(n for n in env if n.startswith(OWNED_PREFIXES) and n not in BY_NAME)
        if unknown:
            raise ConfigError(
                f"unknown knob(s) {' '.join(unknown)}: not declared in scripts/serve_config.py "
                f"(declare it there, or ALLOW_UNKNOWN_KNOBS=1 under PROFILE={DEVELOPER_PROFILE})")
    out: dict[str, str] = {}
    overrides: list[str] = []
    for k in REGISTRY:
        if k.removed:
            continue
        src = next((n for n in (k.name, *k.aliases) if n in env), None)
        if src is not None:
            out[k.name] = _parse(k, env[src])
            if k.name not in NOT_OVERRIDES:
                overrides.append(k.name)
        elif k.name in profile:
            out[k.name] = profile[k.name]
        else:
            out[k.name] = k.default(out) if callable(k.default) else k.default
    out["PROFILE"] = profile_name
    out["PROFILE_OVERRIDES"] = " ".join(overrides)
    if not out["BIND_ADDRESS"]:
        out["BIND_ADDRESS"] = "127.0.0.1"   # empty must never mean "all interfaces"
    return out


def guards(r: dict[str, str]) -> None:
    """Refuse combinations the evidence condemned, each with an override. Every message names the
    experiment and the measured consequence. Extend with your own measured refusals; this is how the
    tried-options ledger stays enforced."""
    ctx = int(r["CONTEXT_LENGTH"]); slots = int(r["MAX_RUNNING_REQUESTS"])
    if ctx > 1_048_576:
        raise ConfigError(
            f"CONTEXT_LENGTH={ctx}: no rung above 1M x 1 was ever qualified. No override: a larger "
            "pool is a new experiment.")
    if ctx > 262144 and abs(float(r["MEM_FRACTION_STATIC"]) - 0.95) < 1e-9:
        raise ConfigError(
            f"CONTEXT_LENGTH={ctx} with MEM_FRACTION_STATIC={r['MEM_FRACTION_STATIC']}: long-context rungs "
            "were qualified at a higher fraction; at this one the pool caps below the context. "
            "Set the qualified fraction, or ALLOW_UNTESTED_FIT=1 to try.")
    if ctx > 65536 and slots > 1 and not r["ALLOW_UNTESTED_FIT"] == "1":
        raise ConfigError(
            f"CONTEXT_LENGTH={ctx} with MAX_RUNNING_REQUESTS={slots}: above 64K only single-slot rungs "
            "were ever qualified. Set MAX_RUNNING_REQUESTS=1, or ALLOW_UNTESTED_FIT=1 to try (then run "
            "the stress gate at that concurrency).")
    if r["SPEC_ENABLE"] == "1" and not 1 <= int(r["SPEC_DRAFT_TOKENS"]) <= 8:
        raise ConfigError(f"SPEC_DRAFT_TOKENS={r['SPEC_DRAFT_TOKENS']} outside the tested 1..8.")
    if r["BIND_ADDRESS"] not in ("127.0.0.1", "localhost", "::1") and not r["API_KEY_FILE"] \
            and r["PROFILE"] != DEVELOPER_PROFILE:
        raise ConfigError(
            f"BIND_ADDRESS={r['BIND_ADDRESS']} without API_KEY_FILE: a non-loopback bind publishes the "
            "model to the network unkeyed. Set API_KEY_FILE, bind to loopback, or PROFILE=developer.")


def resolved_line(r: dict[str, str]) -> str:
    """The one line a launch log carries so every knob that decides the result is named."""
    mtt = r["MAX_TOTAL_TOKENS"] or "derived"
    cap = r["CUDA_GRAPH_BS"] or f"derived<={r['CUDA_GRAPH_MAX_BS']}"
    return (f"resolved: model={r['SNAPSHOT']} fraction={r['MEM_FRACTION_STATIC']} "
            f"context={r['CONTEXT_LENGTH']} slots={r['MAX_RUNNING_REQUESTS']} pool={mtt} "
            f"chunk={r['CHUNKED_PREFILL_SIZE']} capture=[{cap}] spec={r['SPEC_ENABLE']} "
            f"host_thread_cpus={r['HOST_THREAD_CPUS'] or 'floating'} profile={r['PROFILE']} "
            f"overrides=[{r['PROFILE_OVERRIDES']}]")


def shell_fragment(env: dict[str, str]) -> str:
    r = resolve(env)
    guards(r)
    lines = [f"{k.name}={shlex.quote(r[k.name])}" for k in REGISTRY if not k.removed]
    lines.append(f"RESOLVED_LINE={shlex.quote(resolved_line(r))}")
    return "\n".join(lines) + "\n"


def markdown_table() -> str:
    rows = ["| knob | type | default | sink | doc | evidence |", "|---|---|---|---|---|---|"]
    for k in REGISTRY:
        default = "derived" if callable(k.default) else (k.default or '""')
        tag = " **(unsafe)**" if k.unsafe else (" *(removed)*" if k.removed else "")
        doc = (k.removed or k.doc).replace("|", "\\|")
        rows.append(f"| `{k.name}`{tag} | {k.kind} | `{default}` | {k.sink} | {doc} | {k.evidence} |")
    return "\n".join(rows) + "\n"


def profile_drift(name: str) -> list[str]:
    """Knobs a shipped profile sets to something other than the registry default. The registry is the
    single source of truth; a shipped profile restates the serve shape and must agree with it."""
    profile = load_profile(name)
    pure: dict[str, str] = {}
    for k in REGISTRY:
        if not k.removed:
            pure[k.name] = k.default(pure) if callable(k.default) else k.default
    return [f"{k}: profile={v!r} registry={pure[k]!r}" for k, v in profile.items() if v != pure[k]]


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "shell"
    env = dict(os.environ)
    try:
        if cmd == "shell":
            sys.stdout.write(shell_fragment(env))
        elif cmd == "resolved":
            r = resolve(env); guards(r); print(resolved_line(r))
        elif cmd == "markdown":
            sys.stdout.write(markdown_table())
        elif cmd == "check-profiles":
            names = sorted(p.stem for p in Path(PROFILE_DIR).glob("*.env"))
            if not names:
                print(f"FAIL no profiles under {PROFILE_DIR}", file=sys.stderr); return 1
            bad = {n: profile_drift(n) for n in names}
            for n, drift in bad.items():
                for line in drift:
                    print(f"FAIL config/profiles/{n}.env drifts from the registry: {line}", file=sys.stderr)
            if any(bad.values()):
                return 1
            print(f"ok profiles {' '.join(names)} load and agree with the registry defaults")
        else:
            print(__doc__, file=sys.stderr); return 2
    except ConfigError as e:
        print(f"FAIL config: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python3
"""Immutable evidence manifests (T3.19f).

Every `scripts/serve-hybrid.sh` launch writes one manifest: what served (recipe SHA, overlay SHA,
image id, container id, every resolved knob), on what (BIOS slot as declared plus the HSMP-measured
FCLK/MCLK, GPU power caps, kernel) and in what box state (`/proc/meminfo`, `/proc/pressure/*`,
page-cache residency of the Engram shards). The manifest carries a content hash over everything
else in it. The evidence scripts (`bench-decode.py`, `bench-prefill.py`, `nll-score.py`,
`verify-batch-consistency.py`, `stress-kv-full.py`) stamp every JSON they write with that hash,
copy the manifest beside it as `manifest.json`, and refuse a manifest whose container is no longer
the one running. `ci.sh` then checks every stamped file under `evidence/` against its sibling.

    python3 scripts/evidence_manifest.py write --container dsv41-kt     # after docker run (the launcher does this)
    python3 scripts/evidence_manifest.py show [PATH]                    # summary of a manifest (default: current)
    python3 scripts/evidence_manifest.py verify PATH                    # recompute the hash; rc 1 on mismatch
    python3 scripts/evidence_manifest.py check-evidence evidence/       # CI: stamped files match their sibling manifest
    python3 scripts/evidence_manifest.py rescrub evidence/<run>/        # re-scrub $HOME from a leaked manifest, re-stamp siblings

Lookup from the evidence scripts: `EVIDENCE_MANIFEST=<path>` names the manifest, `EVIDENCE_MANIFEST=none`
records an explicitly unmanifested run (the file says so), unset means `<MANIFEST_DIR>/current.json`.
`EVIDENCE_MANIFEST_ALLOW_STALE=1` accepts a manifest whose container id differs from the running
container (recorded in the stamp). Files written before the cut-over (2026-10-08, no `evidence` key) are exempt.

Everything a collector cannot read is recorded as null plus an entry in `errors`; a manifest is
written even on a box without docker, HSMP or nvidia-smi. Paths are stored with `$HOME` as `~` so a
committed manifest passes the identifier scan in ci.sh.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_config  # noqa: E402

SCHEMA = 1
CUTOVER = "2026-10-08"
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Dirty", "Active(file)", "Inactive(file)",
                "Shmem", "SwapTotal", "SwapFree")


class ManifestError(RuntimeError):
    pass


# ---- helpers --------------------------------------------------------------------------------

def _tilde(value: Any, home: str | None) -> Any:
    """Replace $HOME with ~ in strings (recursively) so manifests carry no account name. Anywhere in
    the string, not only at the start: a knob such as EXTRA_DOCKER_ARGS holds `-v $HOME/...:...`."""
    if isinstance(value, str) and home:
        home = home.rstrip("/")
        if value == home:
            return "~"
        return value.replace(home + "/", "~/")
    if isinstance(value, dict):
        return {k: _tilde(v, home) for k, v in value.items()}
    if isinstance(value, list):
        return [_tilde(v, home) for v in value]
    return value


def _untilde(p: str) -> str:
    return os.path.expanduser(p) if p.startswith("~") else p


def _run(cmd: list[str], timeout: float = 20) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def canonical(manifest: dict) -> bytes:
    body = {k: v for k, v in manifest.items() if k != "content_hash"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def content_hash(manifest: dict) -> str:
    return hashlib.sha256(canonical(manifest)).hexdigest()


def finalize(manifest: dict) -> dict:
    out = dict(manifest)
    out.pop("content_hash", None)
    out["content_hash"] = content_hash(out)
    return out


def verify(manifest: dict) -> bool:
    return manifest.get("content_hash") == content_hash(manifest)


# ---- collectors (each returns a JSON value; failures are recorded by collect()) ---------------

def git_state(repo: Path) -> dict:
    sha = _run(["git", "-C", str(repo), "rev-parse", "HEAD"])
    dirty = bool(_run(["git", "-C", str(repo), "status", "--porcelain"]))
    branch = _run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"])
    return {"dir": str(repo), "sha": sha, "short": sha[:8], "branch": branch, "dirty": dirty}


def docker_image_id(image: str) -> str:
    return _run(["docker", "image", "inspect", image, "--format", "{{.Id}}"])


def docker_container(name: str) -> dict:
    out = _run(["docker", "inspect", name, "--format", "{{.Id}}\t{{.Created}}\t{{.Image}}"])
    cid, created, image_id = out.split("\t")
    return {"name": name, "id": cid, "created": created, "image_id": image_id}


def docker_container_id(name: str) -> str:
    return _run(["docker", "inspect", name, "--format", "{{.Id}}"])


def gpu_state() -> list[dict]:
    out = _run(["nvidia-smi", "--query-gpu=index,name,power.limit,driver_version,memory.total",
                "--format=csv,noheader,nounits"])
    rows = []
    for line in out.splitlines():
        idx, name, plimit, drv, mem = [x.strip() for x in line.split(",")]
        rows.append({"index": int(idx), "name": name, "power_limit_w": float(plimit), "driver": drv,
                     "memory_total_mib": int(float(mem))})
    return rows


def hsmp_sample() -> dict:
    spec = importlib.util.spec_from_file_location("hsmp_clocks", HERE / "hsmp-clocks.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    if not os.path.exists(mod.DEV):
        raise FileNotFoundError(mod.DEV)
    s = mod.sample()
    s.pop("t", None)
    return s


def meminfo() -> dict:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        if key in MEMINFO_KEYS:
            out[key] = int(rest.split()[0])
    out["SwapUsed"] = out["SwapTotal"] - out["SwapFree"]
    return {"unit": "kB", **out}


def pressure() -> dict:
    out = {}
    for res in ("cpu", "io", "memory"):
        p = Path("/proc/pressure") / res
        if not p.exists():
            continue
        out[res] = {}
        for line in p.read_text().splitlines():
            kind, *fields = line.split()
            out[res][kind] = {k: float(v) if k != "total" else int(v) for k, v in (f.split("=") for f in fields)}
    return out


def engram_residency(model_dir: str, snapshot: str) -> list[dict]:
    """Page-cache residency of the two checkpoint shards that hold the Engram tables (E18 layout `file`)."""
    snap = Path(_untilde(model_dir)) / "snapshots" / snapshot
    paths = [snap / s for s in ENGRAM_SHARDS]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(str(p))
    out = _run(["fincore", "-b", "--noheadings", "--output", "RES,SIZE,FILE", *map(str, paths)])
    rows = []
    for line in out.splitlines():
        res, size, path = line.split(maxsplit=2)
        rows.append({"shard": Path(path).name, "resident_bytes": int(res), "size_bytes": int(size),
                     "resident_pct": round(100.0 * int(res) / int(size), 2) if int(size) else None})
    return rows


DEFAULT_COLLECTORS: dict[str, Callable[..., Any]] = {
    "git_state": git_state, "docker_image_id": docker_image_id, "docker_container": docker_container,
    "gpu_state": gpu_state, "hsmp_sample": hsmp_sample, "meminfo": meminfo, "pressure": pressure,
    "engram_residency": engram_residency,
}


def collect(env: dict[str, str], *, container: str | None = None, collectors: dict | None = None,
            now: Callable[[], float] = time.time, recipe_dir: Path | None = None) -> dict:
    """Build an unhashed manifest from the environment the launcher ran with. Never raises for a
    collector failure: the field is null and `errors` names it."""
    c = {**DEFAULT_COLLECTORS, **(collectors or {})}
    errors: list[str] = []

    def attempt(name: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001  - any failure becomes a recorded null
            errors.append(f"{name}: {type(e).__name__}: {str(e).strip()[:200]}")
            return None

    resolved = serve_config.resolve(env)
    config = {k.name: resolved[k.name] for k in serve_config.REGISTRY if not k.removed and k.sink != serve_config.GUARD}
    recipe = recipe_dir or HERE.parent
    m = {
        "schema": SCHEMA,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now())),
        "recipe": attempt("recipe", lambda: git_state_with(c, recipe)),
        "overlay": attempt("overlay", lambda: git_state_with(c, Path(resolved["OVERLAY_DIR"]))),
        "image": {"tag": resolved["IMAGE"], "id": attempt("image", lambda: c["docker_image_id"](resolved["IMAGE"]))},
        "container": attempt("container", lambda: c["docker_container"](container)) if container else None,
        "config": config,
        "resolved_line": serve_config.resolved_line(resolved),
        "bios": {"slot_declared": resolved["BIOS_SLOT"], "hsmp": attempt("hsmp", c["hsmp_sample"])},
        "gpus": attempt("gpus", c["gpu_state"]),
        "kernel": platform.release(),
        "box": {
            "note": "sampled when docker run returned, before the server loaded weights",
            "meminfo": attempt("meminfo", c["meminfo"]),
            "pressure": attempt("pressure", c["pressure"]),
            "engram_residency": attempt("engram", lambda: c["engram_residency"](resolved["HF_MODEL_DIR"], resolved["SNAPSHOT"])),
        },
        "errors": errors,
    }
    return _tilde(m, env.get("HOME"))  # one pass over everything: no collector may leak an account name


def git_state_with(c: dict, repo: Path) -> dict:
    return c["git_state"](repo)


# ---- files -----------------------------------------------------------------------------------

def manifest_dir(env: dict[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    return Path(_untilde(serve_config.resolve(env)["MANIFEST_DIR"]))


def write(manifest: dict, out_dir: Path, container: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = finalize(manifest)
    stamp = m["written_utc"].replace(":", "").replace("-", "")
    path = out_dir / f"{stamp}-{container or 'no-container'}.json"
    text = json.dumps(m, indent=1) + "\n"
    path.write_text(text)
    (out_dir / "current.json").write_text(text)
    return path


def load(path: Path) -> dict:
    m = json.loads(path.read_text())
    if not verify(m):
        raise ManifestError(f"{path}: content hash does not match its body (edited after writing?)")
    return m


def locate(env: dict[str, str] | None = None) -> Path | None:
    """The manifest the evidence scripts should stamp with; None means EVIDENCE_MANIFEST=none."""
    env = os.environ if env is None else env
    choice = env.get("EVIDENCE_MANIFEST", "")
    if choice == "none":
        return None
    if choice:
        return Path(_untilde(choice))
    return manifest_dir(env) / "current.json"


# ---- stamping evidence ----------------------------------------------------------------------

def stamp(report: dict, output: Path, tool: str, *, env: dict[str, str] | None = None,
          container_id: Callable[[str], str] = docker_container_id,
          now: Callable[[], float] = time.time) -> dict:
    """Attach the manifest hash to a report that is about to be written at `output`, and copy the
    manifest beside it as manifest.json (an existing sibling must be the same manifest)."""
    env = os.environ if env is None else env
    recorded = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now()))
    path = locate(env)
    if path is None:
        report["evidence"] = {"tool": tool, "recorded_utc": recorded, "manifest_hash": None,
                              "unmanifested": "EVIDENCE_MANIFEST=none"}
        return report
    if not path.exists():
        raise ManifestError(f"no manifest at {path}: serve with scripts/serve-hybrid.sh (it writes one), point "
                            f"EVIDENCE_MANIFEST at the right file, or set EVIDENCE_MANIFEST=none to record that on purpose")
    m = load(path)
    check = "skipped"
    allow_stale = env.get("EVIDENCE_MANIFEST_ALLOW_STALE", "0") == "1"
    if m.get("container"):
        try:
            live = container_id(m["container"]["name"])
            check = "match" if live == m["container"]["id"] else "stale"
        except Exception as e:  # noqa: BLE001 - docker absent or container gone
            check = f"unavailable: {type(e).__name__}"
        if check == "stale" and not allow_stale:
            raise ManifestError(f"manifest {path} describes container {m['container']['id'][:12]} but "
                                f"{m['container']['name']} is now a different container; serve through the launcher "
                                f"again or set EVIDENCE_MANIFEST_ALLOW_STALE=1 to record it anyway")
    sibling = output.parent / "manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if sibling.exists():
        existing = load(sibling)
        if existing["content_hash"] != m["content_hash"]:
            raise ManifestError(f"{sibling} is a different manifest ({existing['content_hash'][:12]} vs "
                                f"{m['content_hash'][:12]}); one evidence directory holds one serve")
    else:
        sibling.write_text(json.dumps(m, indent=1) + "\n")
    report["evidence"] = {"tool": tool, "recorded_utc": recorded, "manifest_hash": m["content_hash"],
                          "manifest_file": "manifest.json", "container_check": check,
                          "container_id": (m.get("container") or {}).get("id")}
    if check == "stale":
        report["evidence"]["allowed_stale"] = True
    return report


def stamp_or_exit(report: dict, output: Path, tool: str) -> dict:
    try:
        return stamp(report, output, tool)
    except ManifestError as e:
        raise SystemExit(f"FAIL evidence manifest: {e}")


# ---- CI check -------------------------------------------------------------------------------

def check_evidence(root: Path) -> list[str]:
    """Every stamped JSON under root matches its sibling manifest.json, whose hash recomputes."""
    problems = []
    for path in sorted(root.rglob("*.json")):
        if path.name == "manifest.json":
            try:
                load(path)
            except (ManifestError, ValueError) as e:
                problems.append(str(e))
            continue
        try:
            doc = json.loads(path.read_text())
        except ValueError:
            continue  # legacy text capture under a .json name (E34a soak); a stamped file is always JSON
        ev = doc.get("evidence") if isinstance(doc, dict) else None
        if not isinstance(ev, dict) or "recorded_utc" not in ev:
            continue  # pre-cut-over file
        if ev.get("manifest_hash") is None:
            if not ev.get("unmanifested"):
                problems.append(f"{path}: stamped without a manifest hash and without an `unmanifested` reason")
            continue
        sibling = path.parent / "manifest.json"
        if not sibling.exists():
            problems.append(f"{path}: stamped with {ev['manifest_hash'][:12]} but no sibling manifest.json")
            continue
        try:
            m = load(sibling)
        except (ManifestError, ValueError) as e:
            problems.append(str(e))
            continue
        if m["content_hash"] != ev["manifest_hash"]:
            problems.append(f"{path}: hash {ev['manifest_hash'][:12]} != sibling manifest {m['content_hash'][:12]}")
    return problems


def rescrub(evidence_dir: Path, home: str | None) -> tuple[str, str, list[Path]]:
    """Re-apply `_tilde` to an already-written `DIR/manifest.json` and re-stamp its siblings.

    For a manifest whose writer leaked `$HOME` inside a knob value (the `_tilde` before 2026-10-09
    scrubbed string prefixes only). The manifest must still verify: this is the only sanctioned way
    to edit one, and it records the old hash as `rescrubbed_from`. Every sibling JSON stamped with
    the old hash has the hash string replaced in place, so its formatting is untouched. Returns
    (old_hash, new_hash, restamped files); a no-op (nothing to scrub) returns old == new.
    """
    path = evidence_dir / "manifest.json"
    m = load(path)
    old = m["content_hash"]
    scrubbed = _tilde({k: v for k, v in m.items() if k != "content_hash"}, home)
    if scrubbed == {k: v for k, v in m.items() if k != "content_hash"}:
        return old, old, []
    scrubbed["rescrubbed_from"] = old
    new_m = finalize(scrubbed)
    path.write_text(json.dumps(new_m, indent=1) + "\n")
    restamped = []
    for sib in sorted(evidence_dir.glob("*.json")):
        if sib.name == "manifest.json":
            continue
        text = sib.read_text()
        if f'"manifest_hash": "{old}"' not in text:
            continue
        sib.write_text(text.replace(f'"manifest_hash": "{old}"', f'"manifest_hash": "{new_m["content_hash"]}"'))
        restamped.append(sib)
    return old, new_m["content_hash"], restamped


# ---- CLI ------------------------------------------------------------------------------------

def summary(m: dict) -> str:
    rec, ov, ct = m.get("recipe") or {}, m.get("overlay") or {}, m.get("container") or {}
    eng = (m.get("box") or {}).get("engram_residency") or []
    mi = (m.get("box") or {}).get("meminfo") or {}
    hs = (m.get("bios") or {}).get("hsmp") or {}
    lines = [
        f"manifest {m.get('content_hash', '')[:12]} written {m.get('written_utc')}",
        f"recipe {rec.get('short')}{'+dirty' if rec.get('dirty') else ''}  overlay {ov.get('short')}"
        f"{'+dirty' if ov.get('dirty') else ''} ({ov.get('branch')})  image {(m.get('image') or {}).get('tag')} "
        f"{((m.get('image') or {}).get('id') or '')[:19]}",
        f"container {ct.get('name')} {str(ct.get('id') or '')[:12]}",
        m.get("resolved_line", ""),
        f"bios slot {(m.get('bios') or {}).get('slot_declared')}  FCLK {hs.get('fclk_mhz')} MCLK {hs.get('mclk_mhz')}  "
        f"gpus {[g.get('power_limit_w') for g in (m.get('gpus') or [])]} W  kernel {m.get('kernel')}",
        f"box: free {mi.get('MemFree', 0) // 1048576} GB avail {mi.get('MemAvailable', 0) // 1048576} GB "
        f"inactive(file) {mi.get('Inactive(file)', 0) // 1048576} GB swap used {mi.get('SwapUsed', 0) // 1024} MB; "
        f"engram resident " + ", ".join(f"{e['shard'][6:11]} {e['resident_pct']} %" for e in eng),
    ]
    if m.get("errors"):
        lines.append("errors: " + "; ".join(m["errors"]))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "write":
        container, out_dir = None, None
        it = iter(rest)
        for a in it:
            if a == "--container":
                container = next(it)
            elif a == "--out-dir":
                out_dir = Path(next(it))
            else:
                print(f"unknown argument {a}", file=sys.stderr)
                return 2
        m = collect(dict(os.environ), container=container)
        path = write(m, out_dir or manifest_dir(), container)
        print(_tilde(str(path), os.environ.get("HOME")))  # launch.txt is captured into evidence; no account name
        for e in m["errors"]:
            print(f"warning: {e}", file=sys.stderr)
        return 0
    if cmd == "show":
        path = Path(rest[0]) if rest else manifest_dir() / "current.json"
        print(summary(load(path)))
        return 0
    if cmd == "verify":
        try:
            load(Path(rest[0]))
        except ManifestError as e:
            print(e)
            return 1
        print("ok")
        return 0
    if cmd == "check-evidence":
        problems = check_evidence(Path(rest[0]))
        for p in problems:
            print(p)
        return 1 if problems else 0
    if cmd == "rescrub":
        old, new, files = rescrub(Path(rest[0]), os.environ.get("HOME"))
        if old == new:
            print(f"{rest[0]}/manifest.json: nothing to scrub ({old[:12]})")
        else:
            print(f"{rest[0]}/manifest.json: {old[:12]} -> {new[:12]}; restamped {len(files)} file(s)")
            for f in files:
                print(f"  {f.name}")
        return 0
    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

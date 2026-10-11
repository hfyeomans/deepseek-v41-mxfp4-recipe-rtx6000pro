# Review C4-1 — hybrid serving launch path (serve-hybrid.sh / serve_config.py / launcher scripts)

- **Requested by:** Amp thread `T-01a0c10b-a5f8-77fc-b238-7147d12ccf29` (owner's T3.35 code-review exercise), executed in thread `T-01a12805-975e-77cf-bffb-89419896841d`.
- **Date:** 2026-10-10. **Revision:** `fdc2101` plus the pre-existing uncommitted worktree changes by others (`README.md`, `docs/CONFIG.md`, `PROVENANCE.md`, `patches/*`), which were reviewed as-is and **not modified**. No tracked file was changed by this review.
- **Scope:** `scripts/serve-hybrid.sh`, `scripts/serve_config.py` and its tests (`scripts/tests/test-serve-config.sh`, `scripts/tests/test_serve_config.py`), `scripts/load-env.sh`, `scripts/serve.sh`, `scripts/stop.sh`, `scripts/wait-ready.py`, `scripts/preflight.sh`, `config/`, `systemd/`.
- **Method:** full read of all of the above; both config suites run; `shellcheck -x -P .`; the GitHub Actions log of the latest `master` run; targeted live probes (4-core `lscpu` mimic, guard probes with `DRY_RUN`/`--print`, `docker inspect` of a throwaway keyed container, a throwaway `-p :PORT` publish, `docker ps` against the running `dsv41-kt`). Exact commands and outputs in the appendix.

## Summary (ranked)

| # | Severity | Finding | Where |
|---|----------|---------|-------|
| F1 | **High** | The bash guard suite merges stderr into the asserted stream, so it fails on any host with fewer than 16 logical CPUs. CI is red on exactly this and `ci.sh` exits there, so **every validation step after line 61 never runs**. | `scripts/tests/test-serve-config.sh:15`, `scripts/ci.sh:61` |
| F2 | **High** | `stop.sh` cannot stop the hybrid: it only removes the legacy container name, while the qualified deployment is `dsv41-kt`. The unit's `ExecStartPre` and `serve.sh`'s "run stop.sh first" hint inherit the gap, and nothing else checks GPU occupancy. (Known: T3.38.) | `scripts/load-env.sh:25`, `scripts/stop.sh:6-10`, `scripts/serve_config.py:107` |
| F3 | **High** | The systemd unit's readiness probe hard-codes `--url http://127.0.0.1:8010` and omits `--container`, although the unit advertises `.env` via `Environment=ENV_FILE`. A `.env` override makes the unit fail after 3600 s on a healthy server; the KV-cap scan silently reads the wrong container; a keyed serve 401s the probe. | `systemd/dsv41-flash.service.in:20` |
| F4 | Medium | The API key is passed on the container argv, so it is visible to any local user in `ps` / `/proc/*/cmdline` and in `docker inspect`. The registry doc recommends this key for exposure beyond the host. | `scripts/serve-hybrid.sh:76` |
| F5 | Medium | `BIND_ADDRESS=""` silently publishes the port on **all** interfaces (legacy `load-env.sh` maps empty to loopback), and no guard requires `API_KEY_FILE` for a non-loopback bind. | `scripts/serve_config.py:109`, `scripts/serve-hybrid.sh:132` |
| F6 | Medium | The hybrid runs with `--gpus all`: `GPU_DEVICES` (documented in `config/example.env`) is ignored, the hybrid cannot be pinned to a free pair, and nothing checks that the cards are idle. | `scripts/serve-hybrid.sh:127`, `:154`, `:157` |
| F7 | Medium | `API_KEY_FILE`: `-r` accepts a directory (raw `tr` error, exit 1), and `tr -d '[:space:]'` silently joins a multi-line key into a value that differs from what every client sends → server up, all evidence scripts 401. | `scripts/serve-hybrid.sh:73-76`, `scripts/openai_client.py:23-25` |
| F8 | Low | The guards have no upper bound on `CONTEXT_LENGTH`: 2M and 4M × 1 at 0.962 are accepted although the documented ladder stops at 1M; the pool then silently caps (or the serve OOMs) minutes into the load. | `scripts/serve_config.py:503`, `:509` |
| F9 | Low | `load-env.sh` exports `DSPARK_BLOCK_SIZE`, which the hybrid registry refuses as an unknown `DSPARK*` knob — a hard refusal for an operator who sources the documented env. | `scripts/load-env.sh:34,43`, `config/example.env:27` |
| F10 | Low | `preflight.sh` runs the feasibility gate with `--sequences 1` while the serve it gates runs `MAX_RUNNING_REQUESTS=8`: ~1.5 GiB of KV is not counted. No verdict flips today. | `scripts/preflight.sh:74` |
| F11 | Low | The P2P check is fail-open: if `nvidia-smi` errors, `grep` sees no match and the script prints "ok P2P read matrix OK". | `scripts/preflight.sh:28-31` |
| F12 | Minor | `load-env.sh`'s value regex admits `#`, so an inline comment becomes part of a value; the hybrid has no restart policy and its name check counts exited containers, so a stale `dsv41-kt` blocks relaunch after a reboot (documented workaround exists). | `scripts/load-env.sh:10`, `scripts/serve-hybrid.sh:154` |

---

## F1 (High) — The guard suite asserts on merged stderr; it fails on any <16-CPU host, and CI has stopped there

**Where:** `scripts/tests/test-serve-config.sh:15`

```bash
out="$(env "$@" DRY_RUN=1 bash "$SERVE" 2>&1)"; rc=$?
if [[ "$want" == ok && $rc -eq 0 && "$out" == resolved:* ]]; then
```

**What happens.** `serve-hybrid.sh:34` runs `cfg="$(python3 .../serve_config.py shell)"`, and `serve_config.py:611-612` prints advisory warnings to **stderr** before writing the fragment to stdout. The warning is host-dependent: `warnings()` (`serve_config.py:453-465`) fires whenever `KT_CPUINFER` (registry default **16**, `serve_config.py:147`) exceeds the core count it derives from `lscpu -p=CORE` (`:458-459`). On a 4-CPU host (GitHub's `ubuntu-latest`, `.github/workflows/ci.yml:9`), every accepted case's `out` starts with `WARNING: KT_CPUINFER=16 ...`, not `resolved:`, so the `expect_ok` assertions fail; the refusals still pass by coincidence (`FAIL config:` contains the warning too). On this 32-logical-CPU host the warning does not fire and the suite is green — which is why it looks healthy locally.

**Why it is wrong.** The suite is supposed to pin the *resolved* line and the refusal contract, but it asserts on a stream that also carries advisory text; the pass/fail therefore depends on the developer's CPU count. The consequence is larger than one test: `scripts/ci.sh:61` runs this suite with output redirected and `exit 1`s on failure, and CI logs confirm it fails there:

```
2026-10-10T20:21:41.9750251Z FAIL serve config tests        (gh run 38083416481, master)
```

Because `ci.sh` is `set -euo pipefail` and exits at the first failure, **everything after line 61 — the registry suite (:64), profile-drift check (:67), `docs/CONFIG.md` freshness (:68), patch-series re-apply, evidence manifests, the README headline check — never runs in CI**. The whole validation pipeline is down, not one test.

**What I would change.**

1. Split the streams in `run_case` (the Python suite already does this: `test_serve_config.py:46-48` uses `capture_output=True` and asserts on `p.stdout`):

```bash
run_case() {
  local want="$1" name="$2" pattern="$3"; shift 3
  local out err rc
  err="$(mktemp)"
  out="$(env "$@" DRY_RUN=1 bash "$SERVE" 2>"$err")"; rc=$?
  err="$(<"$err")"; rm -f "$err"
  if [[ "$want" == ok && $rc -eq 0 && "$out" == resolved:* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  elif [[ "$want" == refuse && $rc -eq 2 && "$err" == "FAIL config:"* && "$err" == *"$pattern"* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  else
    echo "FAIL $name (rc=$rc): ${out%%$'\n'*}${err:+ / ${err%%$'\n'*}}"; fail=$((fail + 1))
  fi
}
```

   Assert `resolved:` on stdout and `FAIL config:` on stderr — that is the actual contract (`serve_config.py:638` prints refusals to stderr).
2. While there: `warnings()` claims "physical cores" but counts **distinct lines** of `lscpu -p=CORE`, whose first column is the CPU number — i.e. logical CPUs (32 on this 16-core/32-thread host, verified). On SMT hosts the warning under-fires (KT_CPUINFER=24 on 16 cores would not warn). Deduplicate on core+socket (`cut -d, -f2,3 | sort -u`) if the E26 cliff is about physical cores.

---

## F2 (High) — `stop.sh` cannot stop the qualified deployment (the hybrid), and the gap also disables the unit's pre-start cleanup

**Where:** `scripts/load-env.sh:25` (`CONTAINER_NAME=dsv41-flash-rtxpro6000`) vs `scripts/serve_config.py:107` (`CONTAINER_NAME=dsv41-kt`), `scripts/stop.sh:6-10`, `scripts/serve-hybrid.sh:13`, `scripts/serve.sh:87-88`, `systemd/dsv41-flash.service.in:18`.

**What happens.** `stop.sh` sources `load-env.sh` and only removes the legacy name. Live, during this review:

```
$ bash scripts/stop.sh
no container named dsv41-flash-rtxpro6000
$ docker ps
dsv41-kt  Up 3 hours  127.0.0.1:8000->30000/tcp
```

The hybrid's own header tells you to use `docker rm -f dsv41-kt` (`serve-hybrid.sh:13`), and the docs repeat it (`docs/BIOS-OPTIMIZATION-GUIDE.md:412`, `docs/TROUBLESHOOTING.md:111`). This is registered as **T3.38** (`plan/TASKS.md:131`) and is still unfixed.

**Why it is wrong.** The canonical stop path silently no-ops on the only deployment that is currently qualified. Worse, the systemd unit's `ExecStartPre=-@REPO@/scripts/stop.sh` (unit:18, `-` = failure tolerated) is exactly the "clear the field" step before `serve.sh`; it cannot clear a running hybrid. `serve.sh:87-88` only refuses when *its own* name exists, and `preflight.sh:18-25` checks GPU *count* and *total* memory, never free memory or running GPU processes. So: hybrid up (cards 0,1 held) → `systemctl start dsv41-flash` → stop.sh no-ops → preflight passes → a second 4-GPU serve launches onto occupied cards. Nothing in the chain catches it.

**What I would change.** Make the registry the single source of truth: add a tiny `serve_config.py` subcommand (e.g. `container-name`) that prints the resolved default, and have `stop.sh` remove whichever of `$CONTAINER_NAME` (legacy env) and the registry default exists — or accept an explicit `NAME`/`--all` argument. Update `serve.sh`'s message and the unit's `ExecStartPre` accordingly. Independently (see F6), add a GPU-idle check to `serve.sh`/`preflight.sh` so a second serve is refused even when the names do not match.

---

## F3 (High) — The systemd readiness probe bypasses `.env`: wrong URL, wrong container, and no key for a keyed serve

**Where:** `systemd/dsv41-flash.service.in:20`

```
Environment=ENV_FILE=@REPO@/.env                                   # :17
ExecStart=@REPO@/scripts/serve.sh                                  # :19
ExecStartPost=@REPO@/scripts/wait-ready.py --url http://127.0.0.1:8010 --timeout 3600   # :20
```

**What happens.** `Environment=ENV_FILE=...` only tells the unit's *children* where `.env` is; nothing reads it for `ExecStartPost` (`wait-ready.py` never sources `load-env.sh`). `serve.sh` sources it and honors `BIND_ADDRESS`/`PORT` (its follow-up hint at `serve.sh:93` uses them), but the probe does not:

- **`.env` override of `PORT`/`BIND_ADDRESS`** → the probe polls the wrong address. `wait-ready.py:65-66` retries on any exception until the 3600 s deadline, prints `timeout`, exits 1; a `Type=oneshot` unit with a failing `ExecStartPost` is marked **failed** while the server is healthy.
- **`--container` absent** → `wait-ready.py:50` defaults to `dsv41-kt` (the *hybrid* name). The legacy container this unit just started is `dsv41-flash-rtxpro6000` (`load-env.sh:25`), so the KV-pool-cap scan (`wait-ready.py:29-41`) reads the wrong container or nothing — the exact warning the scan exists to surface (found on the 512K rung, `wait-ready.py:7-11`) is silently skipped for every unit-served launch. The README chain has the same hole (`README.md:121` runs `scripts/wait-ready.py` bare).
- **Keyed serve** → `wait-ready.py:23` + `openai_client.py:17-26` read `API_KEY`/`API_KEY_FILE` from the *probe's* environment, which never sees `.env`; `/v1/models` returns 401, the loop retries for an hour, and the unit fails. Today `serve.sh` has no key support (`grep API_KEY scripts/serve.sh` → none), but the hybrid does (`serve-hybrid.sh:72-76`) and the registry doc (`serve_config.py:110-114`) recommends the key for the T3.35 tailscale-funnel exposure — so the trap is one line away, on the exact path the repo is about to take.

**What I would change.** Run the probe through the same env loader the serve uses:

```
ExecStartPost=/bin/bash -c '. @REPO@/scripts/load-env.sh; exec @REPO@/scripts/wait-ready.py --url "http://${BIND_ADDRESS}:${PORT}" --container "${CONTAINER_NAME}" --timeout 3600'
```

(`ENV_FILE` is already in the unit's `Environment`, so `load-env.sh` reads the right file and fails loudly on a malformed one.) Pass `--container` in `README.md:121` and `serve.sh:93`'s hints too, or make `wait-ready.py` default to skipping the scan unless `--container` is given — a default name is only ever right for one of the two launchers. Secondary: `wait-ready.py` should fail fast on HTTP 401/403 instead of burning the full timeout on a key that can never start working.

---

## F4 (Medium) — The API key travels on the container argv

**Where:** `scripts/serve-hybrid.sh:76` (`server_args+=(--api-key "$api_key")`).

**What happens.** The key becomes part of `docker run`'s command, so it is stored in the container config (`docker inspect` → `.Config.Cmd`) and visible on the host to any local user in `ps` / `/proc/<pid>/cmdline` for the life of the container. The overlay offers no environment-variable alternative: `--api-key` is declared only as a CLI argument (`sglang/srt/server_args.py:1325`), and `environ.py` has no `API_KEY` (only unrelated `EXA_API_KEY`).

**Why it is wrong.** The registry doc (`serve_config.py:110-114`) recommends `API_KEY_FILE` precisely when the port is exposed beyond the host (tailscale funnel, T3.35) — the moment a leaked bearer token matters most. `docker inspect` access is already root-equivalent (docker group), but `ps` is not, and the container config persists in `docker inspect` long after the launch shell is gone.

**What I would change.** Patch the overlay's argument handling to accept the key from the container environment (`SGLANG_API_KEY`, passed with `-e`) and prefer it over the flag; the overlay already reads a large env registry, so this is small. If that is out of scope, document the exposure in the knob's doc text and steer remote exposure to loopback + a reverse proxy that injects the header, rather than `--api-key` on the argv.

---

## F5 (Medium) — `BIND_ADDRESS=""` publishes on all interfaces, and nothing requires a key for a non-loopback bind

**Where:** `scripts/serve_config.py:109` (plain `str` knob), `scripts/serve-hybrid.sh:132` (`-p "${BIND_ADDRESS}:${PORT}:30000"`).

**What happens.** In `load-env.sh:23`, `: "${BIND_ADDRESS:=127.0.0.1}"` treats an empty value as unset and yields loopback. In the hybrid registry, an empty string passes through unchanged, so `-p :8000:30000` is emitted — Docker's "empty host IP" form, which binds all interfaces. Verified live with a throwaway container:

```
LISTEN 0 4096 0.0.0.0:45699 0.0.0.0:*
LISTEN 0 4096    [::]:45699    [::]:*
```

There is also no guard tying a non-loopback `BIND_ADDRESS` to `API_KEY_FILE`; the key's doc text is advisory only, and `guards()` contains no network check.

**Why it is wrong.** The two launchers disagree about the meaning of `BIND_ADDRESS=` (legacy: loopback; hybrid: the whole LAN), and an operator who "clears" the variable to reset it gets the opposite of what the legacy scripts taught them. Combined with F4/F7, a typo or an over-broad tailscale bind can serve an open endpoint.

**What I would change.** Treat empty as loopback (or refuse it) in the registry, matching `load-env.sh`; and add a guard — non-loopback `BIND_ADDRESS` without `API_KEY_FILE` → `ConfigError` unless `PROFILE=developer` (same override style as the other guards, `serve_config.py:468-517`).

---

## F6 (Medium) — `--gpus all`, `GPU_DEVICES` ignored, no idle-GPU check

**Where:** `scripts/serve-hybrid.sh:127` (`--gpus all`), checks at `:154` (name) and `:157` (port). Contrast `scripts/serve.sh:68` (`--gpus "\"device=${GPU_DEVICES}\""`).

**What happens.** The hybrid never sources `load-env.sh` and the registry has no `GPU_DEVICES` knob, so the variable documented in `config/example.env:8-9` is a no-op for the qualified deployment:

```
$ GPU_DEVICES=2,3 bash scripts/serve-hybrid.sh --print | grep -o -- '--gpus all'
--gpus all
```

On this 4-GPU host the hybrid container therefore sees all four cards; SGLang's `--tp 2` uses devices 0,1. You cannot pin the hybrid to a free pair (e.g. while the GLM container holds 0,1), and there is no check that the cards are idle — the pre-launch checks cover the container name and the port only.

**Why it is wrong.** A documented knob silently does nothing, and on a multi-tenant box the launch either collides with whatever holds devices 0,1 or (with `--tp 2` on busy cards) OOMs inside the container after a multi-minute load. This is also what makes F2's collision scenario reachable without any error.

**What I would change.** Add `GPU_DEVICES` (list, default `0,1`) to the registry and emit `--gpus "\"device=${GPU_DEVICES}\""` like `serve.sh:68` (device ordinals remap inside the container, so `--tp 2` still works); record it in the resolved line/manifest. Add an occupancy check on the selected devices (`nvidia-smi --query-compute-apps` or `memory.used` per index) before launching, at least warning unless `PROFILE=developer`.

---

## F7 (Medium) — `API_KEY_FILE` handling: directory accepted, multi-line key silently mangled vs every client

**Where:** `scripts/serve-hybrid.sh:73-76`, `scripts/openai_client.py:23-25`.

**What happens.** Two defects in five lines:

1. `[[ -r "$API_KEY_FILE" ]]` accepts a **directory** (readable ≠ regular file). Verified: `API_KEY_FILE=/tmp/rev-keydir bash scripts/serve-hybrid.sh --print` → rc 1 with a raw `tr: read error: Is a directory`, no `FAIL ...` message of its own.
2. `tr -d '[:space:]'` strips *all* whitespace and silently concatenates a multi-line file. With `printf 'sk-aaa\nsk-bbb\n'`, the launcher passes `sk-aaask-bbb` while `openai_client.auth_headers()` (`openai_client.py:25`, `.strip()`) sends `Bearer sk-aaa\nsk-bbb`. Verified both values. Result: server starts healthy, every evidence script 401s, and the launcher reports nothing.

**Why it is wrong.** The two sides of the same knob disagree on the value whenever the file is not exactly one whitespace-free line — the failure is silent at launch and looks like a server problem later. The launcher is the right place to be strict, since it is the side that can refuse to start.

**What I would change.** Test `-f`; read one line (`IFS= read -r key < "$API_KEY_FILE"`); fail if the key is empty or if a second non-empty line exists ("API_KEY_FILE must hold exactly one line"); then share that reader with `openai_client` so launcher and client cannot drift (a small helper in `scripts/`, imported by the client).

---

## F8 (Low) — No upper bound on `CONTEXT_LENGTH`

**Where:** `scripts/serve_config.py:503-508` (fraction guard) and `:509-513` (placement-ladder guard), both inside `if r["ALLOW_UNTESTED_FIT"] != "1":` (`:476`). The knob doc stops at 1M (`:195-199`).

**What happens.** Verified accepted (rc 0) with the qualified shape:

```
CONTEXT_LENGTH=2097152 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962 → rc=0
CONTEXT_LENGTH=4194304 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962 → rc=0
```

The KV pool cannot hold those contexts at the qualified fraction, so SGLang silently caps `max_total_tokens` (the warning `wait-ready.py` exists to surface) and inputs over the cap return 400 — or the serve OOMs in capture. Either way the operator learns after a multi-minute load, not at launch.

**Why it is wrong.** Every other rung boundary is guarded; the top of the ladder is not, so the launcher accepts shapes whose only evidence is "never run" while presenting them like the qualified rungs.

**What I would change.** Add a sibling inside the same `ALLOW_UNTESTED_FIT` block: `if ctx > 1048576: raise ConfigError("CONTEXT_LENGTH=... the qualified ladder tops out at 1M x 1 (E34d-v (ii), T3.29); set ALLOW_UNTESTED_FIT=1 to try (then stress-kv-full.py ...)")`.

---

## F9 (Low) — `load-env.sh`'s `DSPARK_BLOCK_SIZE` is an unknown knob to the hybrid registry

**Where:** `scripts/load-env.sh:34,43`; `config/example.env:27`; consumed by `serve.sh:63`. The hybrid's knob is `DSPARK_K` (`serve_config.py`).

**What happens.** Verified: `(source scripts/load-env.sh; DSPARK_BLOCK_SIZE=5 bash scripts/serve-hybrid.sh --print)` → rc 2, refused as an unknown `DSPARK*` knob (`serve_config.py:428-433`). Any flow that loads the documented `.env` into the environment (e.g. `source scripts/load-env.sh`, or `set -a; . .env`) and then launches the hybrid hits a hard refusal on a variable the repo itself documents — the same SGLang flag under two names.

**What I would change.** Alias `DSPARK_BLOCK_SIZE` → `DSPARK_K` in the registry (one line), or rename the legacy knob, or at minimum make the unknown-knob message name the legacy spelling as a hint.

---

## F10 (Low) — `preflight.sh` gates with `--sequences 1` while the serve runs 8 slots

**Where:** `scripts/preflight.sh:74` (`--context-tokens "$MAX_MODEL_LEN" --sequences 1`), `scripts/serve.sh:58` (`--max-running-requests "$MAX_RUNNING_REQUESTS"`, default 8 from `load-env.sh:33`).

**What happens.** The KV term is `890 B/token/sequence × ctx × sequences` (`scripts/feasibility-calc.py:39,106`), so the gate under-counts 7 × 262,144 × 890 B ≈ 1.5 GiB. Measured on this host (RTX PRO 6000 total 97,887 MiB, host 246.8 GiB, util 0.90, ctx 262,144, `engram-host`):

```
--sequences 1 → margin 42.4 GiB, FITS
--sequences 8 → margin 40.9 GiB, FITS
```

No verdict flips today, so this is Low — but on a host nearer the margin the gate would pass a shape that then OOMs at C8.

**What I would change.** Pass `--sequences "$MAX_RUNNING_REQUESTS"`. (The hybrid deliberately skips `preflight.sh` — `serve-hybrid.sh:8-9` — so this is the legacy path's gate; its own config guards are the hybrid's equivalent.)

---

## F11 (Low) — The P2P check fails open

**Where:** `scripts/preflight.sh:28-31`.

```bash
if nvidia-smi topo -p2p r -i "$GPU_DEVICES" 2>/dev/null | grep -Eq '\b(NS|CNS|GNS|TNS|DR)\b'; then
  fail "P2P read is not OK between all selected GPUs"
fi
ok "P2P read matrix OK"
```

**What happens.** If `nvidia-smi` errors (bad device list, unsupported `topo -p2p`), stderr is discarded, `grep` sees empty input, returns 1, and the script prints **"ok P2P read matrix OK"** without having checked anything. `set -euo pipefail` does not help: the pipeline is in an `if` condition, so its status is consumed.

**What I would change.** Capture the output, check `nvidia-smi`'s exit status first (`|| fail "nvidia-smi topo failed"`), then grep the captured text for the bad states.

---

## F12 (Minor)

- **Inline comments become values.** `scripts/load-env.sh:10`'s whitelist regex admits `#` in the value, so `PORT=8010  # http` exports `8010  # http` (later a docker "invalid port" or a bogus knob value). Strip a trailing unquoted ` #...` comment or reject `#`.
- **Stale exited container blocks relaunch.** `scripts/serve-hybrid.sh:154` matches with `docker ps -a` (exited containers included) and the hybrid passes no `--restart` policy (contrast `serve.sh:77`), so after a reboot `serve-hybrid.sh` fails with "already exists" until `docker rm -f dsv41-kt` — a workaround the docs already carry (`docs/BIOS-OPTIMIZATION-GUIDE.md:412`). Either mention the `rm` in the error text or clean up exited containers on launch.
- **README chain omits `--container`.** `README.md:121` (and `serve.sh:93`'s hint) run `wait-ready.py` bare, so the KV-cap scan targets `dsv41-kt`; correct for the hybrid, wrong for the legacy chain. Covered by F3's fix.

---

## Checked and found correct (not findings)

- **`eval "$cfg"` is safe.** Every emitted value is `shlex.quote`d (`serve_config.py:555-562`), including `CONTAINER_ENV`/`DIAG_ENV`; no unquoted interpolation path found.
- **`--print` redaction holds.** `serve-hybrid.sh:144`'s `s/--api-key [^ ]*/.../` cannot leak a fragment: the key element is whitespace-free by construction (`:74` strips it) and `printf %q` only escapes with backslashes for other characters, so `[^ ]*` always consumes the whole token.
- **Expert map is structurally sound.** `config/expert-map-el8-mix-219.json`: 40 per-layer lists × 219 unique expert ids, max id < 384 (`num_experts`); matches the checkpoint's `text_config.num_hidden_layers=40`, `n_routed_experts=384`.
- **Overlay mount is effective.** The bind at `serve-hybrid.sh:134` targets the live package of the image's editable install (`/sgl-workspace/sglang/python/sglang`), so the overlay's patched sources are what loads.
- **Per-profile URL defaults are internally consistent.** Legacy: 8010 (`load-env.sh:24`, `serve.sh:93`); hybrid: 8000 (`serve_config.py:115`, `serve-hybrid.sh:170`).
- **Hybrid skipping `preflight.sh` is deliberate and documented** (`serve-hybrid.sh:8-9`) — the feasibility gate is written for the all-GPU profiles.
- **`DRY_RUN` is honored after every guard** (`serve-hybrid.sh:103-108`), so the config suites exercise the real refusals without docker (the T3.29 incident's fix).
- **Suites and lint on this host:** `test-serve-config.sh` 21/21, `test_serve_config.py` green, `shellcheck -x -P .` clean.

---

## Appendix — verification log (exact commands and observed results)

All probes are read-only or `DRY_RUN`/`--print`; the two throwaway containers were removed; no tracked file was modified. Scratch files under `/tmp/rev-*` and the fake `lscpu` in `/tmp/rev-fb4/`.

**CI (latest `master` run, 2026-10-10T20:21Z, `gh run 38083416481`):**

```
$ gh run view 38083416481   # conclusion: failure, step "Validate repository"
$ grep -n 'FAIL serve config' /tmp/rev-ci.log
214:2026-10-10T20:21:41.9750251Z FAIL serve config tests      # scripts/ci.sh:61
```

**F1 reproduction (fake `lscpu` reporting 4 cores):**

```
$ PATH=/tmp/rev-fb4:$PATH bash scripts/tests/test-serve-config.sh 2>&1 | tail -2
FAIL unsafe flag at its default is not an experiment (T3.24b) (rc=0): WARNING: KT_CPUINFER=16 exceeds the 4 physical cores; ...
4/21 passed
$ PATH=/tmp/rev-fb4:$PATH KT_CPUINFER=4 bash scripts/tests/test-serve-config.sh 2>&1 | tail -2
21/21 passed                       # isolates the warning as the cause
$ bash scripts/tests/test-serve-config.sh | tail -1
21/21 passed                       # real host: 32 logical CPUs, no warning
$ lscpu -p=CORE | grep -v '^#' | cut -d, -f2,3 | sort -u | wc -l   # physical cores
16
$ lscpu -p=CORE | grep -vc '^#'                                    # what warnings() counts
32
```

**F2 (live):**

```
$ bash scripts/stop.sh
no container named dsv41-flash-rtxpro6000
$ docker ps --format '{{.Names}}  {{.Status}}  {{.Ports}}'
dsv41-kt  Up 3 hours  127.0.0.1:8000->30000/tcp
```

**F4/F5 (throwaway container, `dsv41-kt:recipe`, removed afterwards):**

```
$ docker run -d --rm --name rev-bind-probe -p :45699:30000 --entrypoint python3 dsv41-kt:recipe -c 'import time; time.sleep(25)'
$ ss -ltn '( sport = :45699 )'
LISTEN 0 4096 0.0.0.0:45699 0.0.0.0:*
LISTEN 0 4096    [::]:45699    [::]:*
```

(Key-on-argv was confirmed earlier in this review by creating a container with `--api-key SECRET123`: `docker inspect` shows `.Config.Cmd = ["--api-key","SECRET123"]`.)

**F6/F7/F8/F9 guard probes:**

```
$ GPU_DEVICES=2,3 bash scripts/serve-hybrid.sh --print | grep -o -- '--gpus all'
--gpus all
$ BIND_ADDRESS= bash scripts/serve-hybrid.sh --print | grep -o -- '-p [^ ]*:8000:30000'
-p :8000:30000
$ API_KEY_FILE=/tmp/rev-keydir bash scripts/serve-hybrid.sh --print ; echo rc=$?
tr: read error: Is a directory
rc=1
$ printf 'sk-aaa\nsk-bbb\n' > /tmp/rev-2line.key
$ tr -d '[:space:]' < /tmp/rev-2line.key   → sk-aaask-bbb
$ python3 -c "...read_text().strip()"       → 'sk-aaa\nsk-bbb'
$ CONTEXT_LENGTH=2097152 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962 bash scripts/serve-hybrid.sh --print >/dev/null 2>&1; echo rc=$?
rc=0
$ CONTEXT_LENGTH=4194304 ... (same shape) → rc=0
$ (source scripts/load-env.sh; DSPARK_BLOCK_SIZE=5 bash scripts/serve-hybrid.sh --print >/dev/null 2>&1; echo rc=$?)
rc=2
```

**F10 (feasibility calculator, this host's GPU/RAM):**

```
$ python3 scripts/feasibility-calc.py evidence/budget-upstream-bf16fp8.json --gpus 4 --gpu-gib 95.593 \
    --host-gib 246.8 --util 0.90 --gpu-reserve-gib 4 --host-reserve-gib 24 \
    --context-tokens 262144 --sequences 1 --layout engram-host    → margin 42.4  FITS
    (… --sequences 8 …)                                           → margin 40.9  FITS
```
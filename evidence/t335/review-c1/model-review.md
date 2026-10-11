# Code review — DeepSeek-V4.1-Flash two-GPU hybrid serving recipe

Scope: `scripts/`, `config/`, `systemd/`, `docs/`, and the CI in `scripts/ci.sh`, at HEAD `fdc2101`. Read-only review; no tracked files modified. All findings below were confirmed by reading the code and, where marked "verified", by running the relevant command on this checkout (`bash -n` and `shellcheck -x` on every shell file, the Python unit tests, `serve_config.py check-profiles`, `evidence_manifest.py check-evidence`, `benchmarks_headline.py check`, and `test-load-env.sh` all pass, so these are behavioral bugs, not syntax or test failures).

| # | Severity | Where | Finding |
|---|---|---|---|
| F1 | High | `scripts/stop.sh:5-6` | Stops only the legacy container; the hybrid (`dsv41-kt`) keeps both GPUs |
| F2 | High | `scripts/wait-ready.py:46,50`, `scripts/release-smoke.sh:23,25` | Two incompatible default families (port 8000/8010, container `dsv41-kt`/`dsv41-flash-rtxpro6000`); wait-ready.py mixes them; release-smoke can scan the wrong container |
| F3 | Medium | `scripts/load-env.sh:10` | Accepts trailing comments and spaces into values, contradicting its own contract |
| F4 | Medium | `scripts/serve_config.py:42,420-425` | Unknown-knob guard covers only `KT_`/`DSPARK`/`NGRAM`; typos in `ENGRAM_`, `SGLANG_`, `MEM_`, `CUDA_GRAPH_`, `EXTRA_`… are silently ignored |
| F5 | Medium-low | `CONTRIBUTING.md:40`, `scripts/release-smoke.sh:70` | `--context` is a gate parameter, not a deployment knob; release-smoke never checks it against the running server |
| F6 | Low | `scripts/release-smoke.sh:77-78` | Same-prompt gate omits `--flush-cache` despite its own tool's radix-cache caveat |
| F7 | Low | `scripts/serve-hybrid.sh:157` | Port-in-use preflight silently skipped when `ss` is missing |
| F8 | Low | `scripts/release-smoke.sh:110` | `xid-count.txt` gets `0\n0` when the kernel log exists with no Xid lines |
| F9 | Low | `scripts/release-smoke.sh:106` | Health grep `' nan'` misses a NaN line at column 0 |
| F10 | Low | `scripts/summarize-power.py:80,83` | Unguarded `sys.argv[1]` → IndexError traceback instead of usage |
| F11 | Low | `scripts/nll-score.py:171-173` | `ZeroDivisionError` on a text that scores no tokens (e.g. `--max-chars 1`) |
| F12 | Low | `scripts/verify.py:40` | `--needles 7` plants only 6 but requires 7 → guaranteed FAIL |
| F13 | Low | `scripts/bench-engram-gather.py:42-44,159-160` | `--path` can overwrite and then delete a caller's file; README's own example walks into it |
| F14 | Low | `scripts/serve-hybrid.sh` (whole path) | The hybrid launcher never reads `.env`, but README/CONTRIBUTING point users at it |

---

## F1 — High — `stop.sh` cannot stop the hybrid deployment

`scripts/stop.sh:5-6` sources `load-env.sh`, so `CONTAINER_NAME` defaults to `dsv41-flash-rtxpro6000` (`load-env.sh:25`) — the legacy `serve.sh` container. The hybrid launcher's container is `dsv41-kt` (`serve_config.py:107`, documented in `docs/CONFIG.md:75`). So after `bash scripts/serve-hybrid.sh`, `scripts/stop.sh` prints `no container named dsv41-flash-rtxpro6000` (`stop.sh:10`) and removes nothing; the server keeps running and both GPUs stay busy.

This is not hypothetical: `scripts/serve.sh:88` tells the user to run `scripts/stop.sh first`, and `stop.sh` is the only script named "stop" — while `docs/TROUBLESHOOTING.md:111` and the launcher header tell the hybrid user to `docker rm -f dsv41-kt` by hand. Two different stop stories for two deployments that share one `stop.sh`. Already registered as **T3.38** (`plan/TASKS.md:131`) and still unfixed.

**Change:** resolve the hybrid container name from the same source of truth as the launcher (read it from `serve_config.py`, e.g. `python3 scripts/serve_config.py resolved` or a small import), then remove whichever of the two names exists (legacy + hybrid), or accept a `--container` argument and default to checking both. Add a `DRY_RUN`-style test so this cannot regress.

## F2 — High — two incompatible default families for URL/port/container, and `wait-ready.py` mixes them

The scripts hard-code two different deployments:

| Defaults | Port | Container | Model |
|---|---|---|---|
| Hybrid: `serve_config.py:107,108,115`; `release-smoke.sh:23,25`; `nll-score.py:183`; `stress-kv-full.py:56`; `verify-batch-consistency.py:113`; `docs/CONFIG.md:75-79` | 8000 | `dsv41-kt` | `dsv41-flash-hybrid` |
| Legacy: `load-env.sh:24-26`; `config/example.env:13-15`; `verify.py:55`; `bench-decode.py:157`; `bench-prefill.py:77`; `systemd/dsv41-flash.service.in:20`; README:87/121 chain | 8010 | `dsv41-flash-rtxpro6000` | `deepseek-v4.1-flash-local` |

Concrete failures:

1. **`wait-ready.py` matches no deployment.** `scripts/wait-ready.py:46` defaults `--url http://127.0.0.1:8010` (legacy) while `:50` defaults `--container dsv41-kt` (hybrid). On the legacy chain — which is exactly what README:121 and `systemd/dsv41-flash.service.in:20` run — it polls the right port but scans a container that does not exist, so the KV-pool-cap WARNING (`wait-ready.py:37-41`, the check that catches the silent HTTP 400 above the profiled `max_total_tokens`) is silently lost. On the hybrid it polls the wrong port and times out after an hour. Nothing validates that the (URL, container) pair belongs to one deployment.
2. **Bench/verify defaults target the other server.** After `bash scripts/serve-hybrid.sh`, `bench-decode.py`, `bench-prefill.py` and `verify.py` all default to `:8010` and fail with connection refused until the user finds `--url`; the 8000-default scripts fail the same way against a legacy serve. README:121-123 shows the legacy chain with no `--url` at all.
3. **`release-smoke.sh` can scan the wrong container.** `release-smoke.sh:25` is `CONTAINER="${CONTAINER_NAME:-dsv41-kt}"`, reading the ambient environment. `load-env.sh` *exports* `CONTAINER_NAME` (`load-env.sh:42`), so a shell that has sourced it (its docstring says "Source this") with the legacy value gives: `units` mode failing to see the running `dsv41-kt` and proceeding with the cards busy (`release-smoke.sh:53-54`), and `serve` mode running `docker logs dsv41-flash-rtxpro6000` (`:105`) whose *error text* lands in `launch-container.txt`; the health grep (`:106-109`) finds no error lines in that text, so `health=0` and the "0 error lines in the container log" gate **passes while the real log was never scanned**. The same leak makes `wait-ready`'s KV-cap scan vacuous.

**Why wrong:** the deployment identity is duplicated per script instead of derived from one description, and nothing cross-checks it. `docs/CONFIG.md:75-79` and `config/example.env:13-15` even show the two value sets side by side without saying which launcher owns which.

**Change:** make the launcher's resolved config the single source (it already emits a shell fragment: `python3 scripts/serve_config.py shell`), and have `wait-ready.py`, `release-smoke.sh` and the benches default from it — or at minimum (a) set `wait-ready.py`'s `--url` and `--container` defaults to the same deployment, (b) stop `release-smoke.sh` from honoring an ambient legacy `CONTAINER_NAME` (explicit `--container` flag, or derive it), and (c) fail the health gate when `docker logs` fails instead of grepping docker's error text.

## F3 — Medium — `load-env.sh` accepts trailing comments and spaces into values

`scripts/load-env.sh:2-4` promises "Only KEY=VALUE lines are accepted". The regex at `:10` (`^[A-Za-z_][A-Za-z0-9_]*=[^\`\$\(\)\;\&\|]*$`) admits spaces, `#` and quotes, so **verified**: `PORT=8010 # comment` produces `PORT` = `8010 # comment`, which then reaches `docker run -p "${PORT}:30000"` (invalid port) or an `API_KEY_FILE=/path # note` that can never be opened. A conventionally quoted `MODEL_DIR="/mnt/x"` keeps the literal quotes. The safety property the regex buys (no command substitution, no `;`/`|`/`&`/backticks) is real and worth keeping — the bug is that well-formed-but-commented lines are silently mis-parsed instead of stripped or rejected.

`scripts/test-load-env.sh:9-14` only covers a *full-line* comment (`# comment`), so CI cannot catch this.

**Change:** strip a trailing ` #...` (or `\t#...`) before validation, and either unquote `"..."`/`'...'` values deliberately or reject them; then add a trailing-comment case to `test-load-env.sh`. Keep the no-eval, fail-closed design.

## F4 — Medium — the unknown-knob guard covers only three of ~twenty families

`scripts/serve_config.py:42` sets `OWNED_PREFIXES = ("KT_", "DSPARK", "NGRAM")`; the check at `:420-425` only refuses undeclared names in those namespaces. But `README.md:91` says every knob — with the example list "`KT_*`, `DSPARK*`, `MEM_FRACTION_STATIC`, `CUDA_GRAPH_BS`, `EXTRA_*`, …" — "is declared once in the typed registry … unknown or removed knobs are refused before docker starts". **Verified** with `DRY_RUN=1`: `SGLANG_DSV41_REJECT_NAN_ROWSS=0`, `ENGRAM_FILE_PREFETCHH=ahead`, `MEM_FRACTION_STATICC=0.5`, `CUDA_GRAPH_BSS=8`, `EXTRA_SERVER_ARGSS=x` and `CONTEXT_LENGTHH=65536` all launch (rc 0) while a `KT_` typo is refused (rc 2). The registry declares 60 knobs across ~20 first-word families, so most of the documented surface is unguarded.

**Why wrong:** the failure the guard exists for — `docs/TROUBLESHOOTING.md:180`: "a guard probe once launched a real server because the typo was silently ignored" — applies equally to `ENGRAM_`, `SGLANG_` and `EXTRA_` typos: the user believes an experiment knob was applied, the resolved line and manifest do not show it, and the serve silently uses the default. `docs/CONFIG.md:13` and `docs/TROUBLESHOOTING.md:180` describe the narrow scope, so the README is the overstatement — but the gap itself is real.

**Change:** extend the guard to the other project-owned families. Do *not* derive every prefix from the registry blindly: `CUDA_`/`HF_` prefixes would start refusing ambient variables the tooling does not own (`CUDA_VISIBLE_DEVICES`, `HF_TOKEN`) and make the launcher hostile. Use an explicit list of owned families (`KT_`, `DSPARK`, `NGRAM`, `ENGRAM_`, `SGLANG_`, `EXTRA_`, `MEM_FRACTION_`, `CUDA_GRAPH_`, `CONTEXT_LENGTH`, `MAX_RUNNING_`, `CHUNKED_`, …), or a near-miss check (refuse an undeclared name within edit distance 1–2 of a declared knob/alias). Add a test with an `ENGRAM_`/`SGLANG_` typo; if any family stays unguarded, soften `README.md:91` to say which prefixes are checked.

## F5 — Medium-low — `--context` is a gate parameter, not a deployment knob, and release-smoke never cross-checks it

`CONTRIBUTING.md:40` reads "for a ladder rung add `--context 262144` etc.", which invites the reading that this reconfigures the server. It does not: `--context` only sizes `verify.py`'s needle prompt (`verify.py:57`) and the stress prompt (`release-smoke.sh:73-74`). The real context is launch-time (`CONTEXT_LENGTH`, plus `MAX_RUNNING_REQUESTS` for slots); the rung is launched with e.g. `CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 bash scripts/serve-hybrid.sh`, and `CONTRIBUTING.md:40` itself says the resolved line must record those overrides.

`release-smoke.sh` records its own args (`:70`) but never compares them with the running server: it reads neither `/v1/models` `max_model_len` nor the launch manifest. So a 262K × 4 rung launched with the flags forgotten runs every gate at 64K while the evidence directory claims the rung (silent under-testing — the dangerous direction), and a 64K server with `--context 262144` dies at `verify.py:79` (`context_ceiling`) with a message that does not say "the server was not launched at this context".

**Change:** have `release-smoke.sh` read `max_model_len` from `/v1/models` (and `CONTEXT_LENGTH`/`MAX_RUNNING_REQUESTS` from the launch manifest) and refuse when `--context`/`--slots` disagree, the same way `benchmarks_headline.py` refuses a mismatched stamp; rewrite `CONTRIBUTING.md:40` to show the launch command plus the matching smoke flags.

## F6 — Low — the same-prompt gate runs without `--flush-cache`

`release-smoke.sh:77-78` runs `verify-batch-consistency.py --same-prompt` with no `--flush-cache`, although that tool's own docstring (`verify-batch-consistency.py:29-35`) says the radix cache makes even `--concurrency 1 --same-prompt` diverge for cache-path reasons and to pass `--flush-cache` when the question is determinism. The committed evidence used `flush_cache:false` and passed, so this is a latent flake, not an observed failure: a future kernel that changes the cache-hit path would fail the release gate for a reason unrelated to the change under test.

**Change:** add `--flush-cache` to the gate, or say in the smoke header why it is deliberately off.

## F7 — Low — the port-in-use preflight is silently skipped without `ss`

`serve-hybrid.sh:157` is `if ss -ltn "( sport = :${PORT} )" | grep -q LISTEN; then`. If `ss` is not installed (minimal host/container), the command-not-found failure is masked by the `if` condition (set -e does not apply) and `grep` sees EOF, so the launch proceeds with no preflight at all. The check also ignores `BIND_ADDRESS` (it catches any listener on the port, which is stricter, so harmless).

**Change:** probe for `ss` first and warn/fail, or use a stdlib Python bind probe (`socket.bind`) as the fallback — always available next to the launcher.

## F8 — Low — `xid-count.txt` gets `0\n0` on a clean kernel log

`release-smoke.sh:110`: `{ grep -c -iE 'xid' /var/log/kern.log 2>/dev/null || echo 0; }`. When the file exists with no matches, `grep -c` prints `0` *and* exits 1, so the `||` appends a second `0`. **Verified** with `od -c`: the file contains `0 \n 0 \n`. The common case (clean box) produces a malformed count.

**Change:** `n=$(grep -c -iE 'xid' /var/log/kern.log 2>/dev/null) || n=0; echo "$n"`.

## F9 — Low — health grep misses a NaN line at column 0

`release-smoke.sh:106` matches `' nan'` with a leading space, so a log line that starts with `nan`/`NaN` (or has it after a `[`/`{`) is not counted, even though `-i` is used. The gate could pass with a real non-finite complaint in the log.

**Change:** match `(^|[^A-Za-z])nan` (or `[^a-z]nan` with `-i`), and add the server's explicit non-finite marker (`docs/TROUBLESHOOTING.md` names `[non-finite-logits]`) to the pattern set.

## F10 — Low — `summarize-power.py` crashes instead of printing usage

`scripts/summarize-power.py:80` and `:83` index `sys.argv[1]` (and `:81` `sys.argv[2]`) unguarded. **Verified**: `python3 scripts/summarize-power.py` exits 1 with `IndexError: list index out of range` and a traceback; `--smoke` alone fails the same way.

**Change:** use `argparse` with a required `path` and `--smoke DIR`, matching every other script in `scripts/`.

## F11 — Low — `nll-score.py` divides by zero on a text that scores no tokens

`scripts/nll-score.py:171-173`: `summarize()` computes `-sum(lps) / n` with `n = len(lps)`; the first logprob is always dropped (`:80`, "position 0 has no prediction"), so a text that tokenizes to one token yields `lps == []`. **Verified** by calling `summarize([], 1024)` → `ZeroDivisionError`. Reachable from the CLI via `--max-chars 1` (`:185`). Related: `pooled_stats` (`:131-139`) only sets `se_cluster`/`z_cluster` when `n_tot > 0`, so an empty corpus makes the release-smoke NLL gate (`release-smoke.sh:84`, `c["z_cluster"]`) die with a `KeyError` traceback instead of a named failure.

**Change:** validate `n > 0` in `summarize`/`score_text` with a clear message (and skip/reject texts that score nothing); make the gate print "no paired tokens" when `z_cluster` is absent.

## F12 — Low — `verify.py --needles 7` can never pass

`scripts/verify.py:40`: `depths = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99][:needles]` caps at 6, while `:39` builds `needles` keys and `:44` zips them with `depths`, silently planting only 6. The check at `:109-110` then requires `found == args.needles`, so `--needles 7` is a guaranteed FAIL with a confusing "found=6 needles=7". There is no upper bound on the flag (default 6 at `:58`).

**Change:** either validate `--needles <= len(depths)` with a clear error, or generate the depths arithmetically (`[(i + 0.5) / needles for i in range(needles)]`) so any count works.

## F13 — Low — `bench-engram-gather.py` can overwrite and delete a caller's file

`scripts/bench-engram-gather.py:42-44`: an existing `--path` of the same size is silently reused, a different size is overwritten in place; `:159-160` then unlinks the file unless `--keep-file`. Nothing in the docstring or `--help` says the file is deleted (`--keep-file` at `:85` has no help text), and `README.md:110` tells users to pass `--path results/engram-nvme0.bin`. Point `--path` at anything real, or re-run with a different `--rows`, and the file is gone at the end of the benchmark.

**Change:** unlink only when this run created the file; require `--force` to reuse/overwrite an existing path; document `--keep-file` and the deletion in the docstring and help.

## F14 — Low — `.env` is read only by the legacy chain; the hybrid launcher ignores it

`serve-hybrid.sh` never sources `load-env.sh`, and `serve_config.py` reads the process environment plus `config/profiles/<name>.env` only — nothing reads the repo `.env`. Yet `README.md:116` (`cp config/example.env .env`), `CONTRIBUTING.md:10` ("image digests in `.env`/`PROVENANCE.md`") and the legacy scripts make `.env` the obvious place for settings, and `config/example.env:13-15` carries the *legacy* port/container/model. A user who puts `API_KEY_FILE=` or `PORT=` in `.env` and runs `bash scripts/serve-hybrid.sh` gets an unkeyed serve on 8000 with no warning — a real risk given `docs/TROUBLESHOOTING.md` recommends `API_KEY_FILE` for exposed ports (T3.35).

**Change:** state in `README.md` and `docs/CONFIG.md` that the hybrid path reads the environment and profiles, not `.env`; or source `.env` in `serve-hybrid.sh` through the same fail-closed loader (after fixing F3) so both chains agree.

---

## Checked, no finding

- **API-key redaction** in `serve-hybrid.sh:144` is effective: `printf '%q '` quotes the whole docker command and the `sed` then rewrites the single `--api-key <value>` token; whitespace in a key file is stripped when read (`:74`), so a key cannot split the token. No leak into `DRY_RUN` output. (Verified by construction and a run.)
- **No eval injection:** every value the launcher `eval`s goes through `shlex.quote` (`serve_config.py:544-555`).
- **`openai_client.py`** is sound: SSE framing, `stream_options.include_usage`, `thinking` pinned false by default (`:75`, matching `docs/TROUBLESHOOTING.md:250`), `decode_tps` preferring server-reported usage. Two minor notes, neither a bug: a missing `API_KEY_FILE` surfaces as a bare `FileNotFoundError` from the request path rather than a clear message; and the `API_KEY` literal it accepts (`:22`) is script-side only — the launcher has no such knob, so exporting only `API_KEY` leaves the server keyless (an unused `Authorization` header is ignored, harmless) while using it against a differently-keyed serve gives 401.
- **`source scripts/load-env.sh; bash scripts/serve-hybrid.sh`** fails closed with `unknown profile 4gpu-sglang` (`load-env.sh:28` + `serve_config.py:407-408`, pinned by a test) — correct, though a line in `docs/TROUBLESHOOTING.md` would save the first user who tries it.
- `release-smoke.sh`'s default baseline corpus (`evidence/t328/default-219-8192/corpus`) and the NLL fixture (`scripts/fixtures/nll-corpus`) exist in-repo, so the documented smoke path works on a fresh clone.
- `bench-engram-gather.py`'s O_DIRECT path works on this host (E3 numbers reproduced); no bug found there.
- `ci.sh` runs clean under `shellcheck -x`; the identifier scan, JSON checks and test census all pass.

## Suggested fix order

1. **F1** (one-line container resolution) and **F2** (single source of defaults + stop honoring ambient `CONTAINER_NAME` + fail the health gate on a missing log) — these are the two that leave a user with a running server they cannot manage and gates that pass without looking.
2. **F3** and **F4** — both are fail-silent configuration bugs with small, testable fixes.
3. **F5** — doc/CI contract fix (rung launch command + smoke cross-check).
4. F6–F14 as a cleanup pass; F6 and F12 are the two most likely to bite during a release week.
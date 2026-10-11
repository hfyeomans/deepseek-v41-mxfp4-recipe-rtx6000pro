# Review C4-2 — evidence and benchmark tooling

Scope: `scripts/release-smoke.sh`, `openai_client.py`, `evidence_manifest.py`, `bench-decode.py`, `bench-prefill.py`,
`nll-score.py`, `verify.py`, `verify-batch-consistency.py`, `stress-kv-full.py`, `gpu-power-sample.sh`,
`summarize-power.py`, `benchmarks_headline.py`, `ci.sh`, checked against what `docs/BENCHMARKS.md` and
`CONTRIBUTING.md` claim they measure.

Method: read all 13 files in full, plus the docs claims, `config/benchmarks-headline.json`, the stamped JSONs under
`evidence/`, and `scripts/tests/test_evidence_manifest.py` / `test_nll_stats.py`. Statistical and shell edge cases
below were reproduced by executing the modules directly (scratch harness under `.amp/in/t335/scratch/`). No tracked
files were modified. Confirmed today: `python3 scripts/benchmarks_headline.py check` → rc 0 and
`python3 scripts/evidence_manifest.py check-evidence evidence/` → rc 0, so the generated-artifact checks themselves
re-derive correctly from the JSONs.

Findings are ranked by severity; each says why it is wrong and what I would change.

---

## High

### H1. The release NLL gate can pass on a stale `pooled.json`

[`release-smoke.sh:79-87`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L79-L87)

```sh
python3 "$ROOT/scripts/nll-score.py" --url ... --corpus ... \
  --output-dir "$OUT/corpus" --compare-dir "$BASELINE_CORPUS" > "$OUT/corpus.txt" 2>&1
gate "NLL corpus |z| <= $NLL_Z_MAX" python3 - "$OUT/corpus/pooled.json" "$NLL_Z_MAX" <<'PY'
```

The corpus run's exit status is never captured, and the gate then reads `$OUT/corpus/pooled.json` unconditionally.
`nll-score.py` only writes `pooled.json` at the end of a successful corpus run (`nll-score.py:256-257`), so if the
run dies mid-way — server error, a missing baseline text (`:231-232`), a kill, an OOM — and a previous run into the
same `--output-dir` already wrote `pooled.json`, the gate reads the *old* file and can report PASS for a run that
never happened. Rerunning `release-smoke.sh serve --output-dir evidence/<release>/64k-4slot` after a partial failure
is exactly the workflow where this happens. This is the only gate in the script that reads a file produced by a
command whose status was not checked (every other gate invokes the tool through `gate()` directly).

Change: capture the rc and make it a gate of its own, e.g.

```sh
if ! python3 "$ROOT/scripts/nll-score.py" ... > "$OUT/corpus.txt" 2>&1; then
  echo "FAIL NLL corpus run (see $OUT/corpus.txt)"; fails+=("NLL corpus run")
else
  gate "NLL corpus |z| <= $NLL_Z_MAX" python3 - ... <<'PY'
  ...
  PY
fi
```

and/or `rm -rf "$OUT/corpus"` before the run, and have the gate also assert the file's
`evidence.recorded_utc` is from this run.

### H2. `z_cluster` is 0.0 whenever `se_cluster` is 0, so the `|z| ≤ 2` gate passes vacuously

[`nll-score.py:136-137`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/nll-score.py#L136-L137)
(the same pattern for `z_paired` at [`nll-score.py:113`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/nll-score.py#L113))

```python
st["se_cluster"] = math.sqrt(acc) / n_tot
st["z_cluster"] = mean_d / st["se_cluster"] if st["se_cluster"] else 0.0
```

For a single text, `acc = (Σdᵢ − n·mean_d)² = 0` exactly, so `se_cluster` is 0 for **any** delta, and `z_cluster`
becomes `0.0`. The release gate checks only `abs(z) <= NLL_Z_MAX` (`release-smoke.sh:81-87`), so a corpus of one
text passes no matter how bad the candidate is. Verified: a one-text pair with ΔNLL = +0.9 nats/token (450× the
+0.002 gate the script's own docstring documents) yields `se_cluster 0.0, z_cluster +0.00`, and the gate passes.
The 1M rung's `Δ -0.00000 ± 0.00000 (z +0.00, 0 of 65,653 moved)` is the benign version of the same arithmetic
(all deltas exactly 0); the single-text case is not benign.

Change: when `se_cluster == 0`, report `z_cluster = 0.0` only if `mean_d == 0.0`, else `±inf` (or add an explicit
`se_zero` flag); and make the release gate require `c["texts"] >= 2` and a minimum `n` before trusting `z`.
Add a unit case to `test_nll_stats.py`.

### H3. `ci.sh`'s "2 GPUs must not fit" check treats any calculator error as "does not fit"

[`ci.sh:89-91`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/ci.sh#L89-L91)

```sh
if python3 "$ROOT/scripts/feasibility-calc.py" "$ROOT/evidence/budget-upstream-bf16fp8.json" --gpus 2 --layout engram-host >/dev/null; then
  echo 'FAIL calculator says 2 GPUs fit; README verdict is stale'; exit 1
fi
```

`feasibility-calc.py` exits 0 = fits, 2 = usage / missing budget file (`:243`, `:250`), 3 = does not fit (`:292`),
1 = unhandled exception. This gate only fails on 0, so rc 2 (budget JSON renamed/moved) and rc 1 (traceback) are
silently read as "does not fit" and the README verdict is "confirmed" without the calculator having worked.
Verified: a nonexistent budget path exits 2 and the gate passes. (The companion `hybrid-cpu` check at `ci.sh:92-94`
is fail-closed and correct.)

Change: `rc=0; python3 ... || rc=$?; [[ $rc -eq 3 ]] || { echo "FAIL calculator did not return 'does not fit' (rc $rc)"; exit 1; }`.

### H4. `--full` can silently lose its power evidence, and the sampler's failure is invisible

[`release-smoke.sh:103`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L103) ·
[`summarize-power.py:65-76`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/summarize-power.py#L65-L76) ·
[`gpu-power-sample.sh:17-24`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/gpu-power-sample.sh#L17-L24)

`summarize_smoke()` computes `max((v["temp_max_c"] or 0) for v in power.values())` with no guard for an empty
`power` dict. An empty or header-only `power-*.csv` therefore raises `ValueError: max() iterable argument is empty`
(verified: rc 1 with a header-only CSV), and `release-smoke.sh:103` swallows it with `|| true`. A `--full` run can
exit 0 with no `power-summary.json` at all, while `BENCHMARKS.md:3`/`:61` presents the power condition and cites
`power-summary.json` as the evidence for the default rung. Compounding it, `gpu-power-sample.sh` never checks that
the sampler produced any rows: if `nvidia-smi` is missing or dies immediately, the wrapper still returns the bench's
exit status and leaves an empty CSV, which then triggers the crash above.

Change: in `summarize_smoke`, skip/record empty files (`if not power: report[label] = {"error": "no samples"}; continue`);
in `release-smoke.sh --full`, gate on `power-summary.json` existing and containing one entry per expected label;
in `gpu-power-sample.sh`, fail (or at least warn loudly) when the CSV has no data rows after the command returns.

---

## Medium

### M1. The needle prompt is only ~84.5 % of the configured context, and nothing asserts it

[`verify.py:31-50`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/verify.py#L31-L50) ·
`BENCHMARKS.md:10` ("N-needle exact retrieval **at the configured context**")

`filler()` assumes "~1.3 tokens per short English word" and emits `int(n_tokens / 1.3)` words, but the 20 filler
words average ~1.1 tokens each, so the body lands at ~0.77×n tokens; with the 600-token reserve the measured
prompts are 55,206 / 65,536 (84 %), 221,756 / 262,144 (85 %), 887,387 / 1,048,576 (85 %). The `multi_needle` check
(`verify.py:104-112`) asserts only `found == needles` and `finish_reason == "stop"`; it never compares
`prompt_tokens` with `--context`, so the docs claim is unenforced (the headline table itself reports the true
55,206-token number, which is where the mismatch is visible).

Change: calibrate the filler (≈ /1.0, or measure once and scale) so the prompt actually reaches the target, or
assert `prompt_tokens >= 0.97 * args.context` as part of the check and correct the Method text to state the real
fill fraction.

### M2. `context_ceiling` is silently skipped when the server does not report `max_model_len`

[`verify.py:76-79`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/verify.py#L76-L79)

```python
max_len = (m or {}).get("max_model_len")
record("models_alias", m is not None, model=model, max_model_len=max_len)
if max_len is not None:
    record("context_ceiling", int(max_len) >= args.context, ...)
```

A server that lists the alias but omits `max_model_len` passes with the ceiling check simply not run, and the
release-smoke gate "needles at $CONTEXT" sees exit 0. Since `models_alias` already requires the model to be listed,
a missing `max_model_len` is abnormal and should fail — at least when `--context` was given explicitly.

Change: `record("context_ceiling", max_len is not None and int(max_len) >= args.context, ...)`.

### M3. The "container is no longer the one running" invariant is not enforced when the container is gone

[`evidence_manifest.py:294-303`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/evidence_manifest.py#L294-L303)
vs the module docstring [`evidence_manifest.py:10-11`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/evidence_manifest.py#L10-L11)

On any docker failure (`docker` absent, or `docker inspect` → "no such container"), `check` becomes
`"unavailable: …"` and the stamp proceeds. Only a *different live container with the same name* is refused. So:
launch via the launcher, `docker rm -f` the container (or reboot), hand-start a different server, run an evidence
script — the new JSON is stamped with the old manifest as if that serve produced it. The test suite encodes this as
intended (`scripts/tests/test_evidence_manifest.py:212-215`), so what is wrong is the docstring's promise (and
`check_evidence` never looks at `container_check` at all). The hand-run-server case is already covered by
`EVIDENCE_MANIFEST=none`.

Change: either refuse `unavailable` for a manifested run unless `EVIDENCE_MANIFEST_ALLOW_STALE=1` (matching the
docstring), or soften the docstring and have `check_evidence` flag post-cut-over files whose
`container_check != "match"` (or `"skipped"` with an explicit `unmanifested` reason).

### M4. Docs and the generated caption disagree with the harness on runs, warmups and max tokens

`BENCHMARKS.md:7` ("256 forced output tokens per sequence, **2 warmups then 5 measured runs** per concurrency
level") vs [`release-smoke.sh:95-98`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L95-L98)
(`--max-tokens 512 --runs 6 --warmup 1` for C1, `--max-tokens 512 --runs 3 --warmup 1` for C4) vs
[`benchmarks_headline.py:168`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/benchmarks_headline.py#L168)
("over **5 runs** of 512 forced tokens").

All three headline rungs' JSONs record `runs` 6 (C1) / 3 (C4), `warmup` 1, `max_tokens` 512. So the Method bullet
is stale on all three counts, and the generated caption contains a hand-typed "5 runs" (and a hand-typed
"65,653-token corpus" at `:170`) that contradicts the JSONs it is generated from. `ci.sh:83` only checks the block
for byte-equality with the generator, so a wrong-but-stable sentence survives forever.

Change: derive the caption's run counts and corpus size from the stamped JSONs (`doc["runs"]`,
`pooled["n"]`/`tokens_scored_total`), and fix `BENCHMARKS.md:7` to the actual invocation (512 tokens, 1 warmup,
6 runs at C1 / 3 at C4 — or change the invocation and regenerate).

### M5. `nll-score.py`'s own documented gate is not the gate the harness enforces

[`nll-score.py:38-39`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/nll-score.py#L38-L39) ·
[`release-smoke.sh:81-87`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L81-L87)

The docstring documents "same kernel shapes → every token bit-identical (`tokens_changed == 0`); shape-changing →
pooled `delta_mean_nll <= +0.002`". The release gate checks only `abs(z_cluster) <= 2`: it never checks
`tokens_changed`, and never enforces the absolute +0.002 threshold. A z-gate is sample-size dependent — the corpus
size sets the effective ΔNLL threshold (≈ 0.0026 at 65K tokens with this `se_cluster`), and nothing pins it if the
corpus or baseline changes. A large-but-noisy regression can pass; a tiny-but-consistent one can fail.

Change: enforce both in the release gate (`delta_mean_nll <= 0.002` and `abs(z_cluster) <= 2`, plus
`tokens_changed == 0` when the token shapes match), or rewrite the docstring to state the |z| gate as the real one.

### M6. `decode_tps` can report ~10¹⁰ tok/s

[`openai_client.py:42-56`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/openai_client.py#L42-L56)
with [`openai_client.py:84-131`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/openai_client.py#L84-L131)

If a stream carries usage `completion_tokens` but no content/reasoning deltas (a tool-call-only response, or an
empty content stream), `ttft` is never set, so `ttft_s = total_s`, `decode_s = max(0, 1e-9)`, and
`decode_tps = (n−1)/1e-9`. Verified: 30 usage tokens → 2.9×10¹⁰ tok/s. The shared client feeds
`bench-decode.py`, `stress-kv-full.py` and `verify.py`, and the `verify.py` tool-call path constructs exactly such
a request.

Change: track whether any token event was observed and return `None` when there were none (the `n <= 1` guard
already covers the no-usage case).

### M7. `verify-batch-consistency.py`'s docstring promises byte-identity that the code does not gate on

[`verify-batch-consistency.py:25-27`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/verify-batch-consistency.py#L25-L27)
vs [`verify-batch-consistency.py:167-179`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/verify-batch-consistency.py#L167-L179)

The docstring says exit 0 requires "all concurrent rows are identical to each other"; the code requires only
`min_pair_prefix >= --min-prefix` (40 chars by default) and records `rows_identical` without gating on it. The
release-smoke header correctly documents the 40-char gate and E26 shows hundreds of chars of legitimate drift, so
the code is the deliberate choice — the docstring is stale.

Change: fix the docstring to match (`all concurrent rows share at least --min-prefix characters`).

---

## Low

### L1. The stress gate is only "all finished"

[`stress-kv-full.py:91-92`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/stress-kv-full.py#L91-L92)

`ok = all(r["finish_reason"] is not None for r in results)` passes even if a request prefilled far less than
`--prompt-tokens` or produced a single token — the "filling every slot" claim is checked only by the median
printed in the headline. `totals` at `:91` is also taken from `samples[0]`, which can be an error row
(`:49-50`), in which case `vram_total_mib` is empty.

Change: `ok = all(r["finish_reason"] is not None and (r["prompt_tokens"] or 0) >= 0.95*args.prompt_tokens
and (r["completion_tokens"] or 0) >= 1 for r in results)`, and take totals from the first non-error sample.

### L2. `rescrub` misses nested evidence directories

[`evidence_manifest.py:384`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/evidence_manifest.py#L384)

`for sib in sorted(evidence_dir.glob("*.json"))` only re-stamps the top level. The qualified rung directories have
a `corpus/` subdirectory with stamped per-text JSONs plus its own `manifest.json`; rescrubbing the parent leaves
them pointing at the old hash, so the next `check-evidence` (`ci.sh:79`) fails — or, if only the parent manifest
needed scrubbing, the subdirectory silently keeps a hash for a manifest that no longer exists.

Change: `rglob("*.json")` and skip only the top-level `manifest.json`.

### L3. `xid-count.txt` gets two lines (`0\n0`) when no Xid is found

[`release-smoke.sh:110`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L110)

`grep -c` prints `0` and exits 1 on no matches, so `|| echo 0` appends a second `0`. Only the "file unreadable"
path produces a single `0`.

Change: `count=$(grep -c -iE 'xid' /var/log/kern.log 2>/dev/null || true); echo "${count:-0}"`.

### L4. `bench-decode.py` defaults to same-prompt, against the stated method

[`bench-decode.py:165-167`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/bench-decode.py#L165-L167)
vs `BENCHMARKS.md:7` ("**Every concurrency figure (C2+) is measured with `--distinct-prompts`**")

The JSON records the flag and the headline appends "(same prompt)", so evidence stays self-describing, but a user
following the Method section and running the script directly gets numbers the docs say are not the standing
method. Change: default the flag on (add `--same-prompt` to opt out) or state in the help/README that
`--distinct-prompts` is required for reportable C2+ figures.

### L5. The "cold" prefill figure is the median of the cold pass, hiding the genuinely cold first prompt

[`bench-prefill.py:66-69`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/bench-prefill.py#L66-L69)
· `BENCHMARKS.md:23`/`:58`

`evidence/t339/window2/qual/prefill-16k.json` cold pass: TTFTs 24.12 / 5.47 / 5.24 s → median 5.5 s → headline
"2,934 cold (5.5 s)", while the first prompt after the serve ran at 662 tok/s (the docs' own Reading paragraph at
`BENCHMARKS.md:46` quotes 600–640 tok/s). The "cold" label is about pass ordering, not the first prompt, so the
headline number can be misread as the cost of the first prefill.

Change: report the first run separately (or label it "cold-pass median" and quote the first-run figure alongside).

### L6. `verify.py --needles > 6` can never pass; `--needles 0` passes vacuously

[`verify.py:38-50`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/verify.py#L38-L50)

`depths` has exactly 6 entries and `zip(keys.items(), depths)` silently truncates, so with `--needles 7` the extra
records are never placed in the text and `found < needles` forever; with `--needles 0`, `found == needles` is
trivially true. Change: assert `1 <= args.needles <= 6` (or extend `depths`).

---

## Minor

- [`summarize-power.py:79-86`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/summarize-power.py#L79-L86):
  no argparse — `sys.argv[1]`/`[2]` raise `IndexError` with no arguments; `read_samples` assumes the first line is
  the header.
- [`bench-prefill.py:71`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/bench-prefill.py#L71):
  `size['median_prefill_tps'] or 0` prints `0.00` for a missing median instead of `n/a`.
- [`ci.sh:118`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/ci.sh#L118): the
  executable-bit check exempts any path ending in `openai_client*`; scope it to `scripts/openai_client.py`.
- [`release-smoke.sh:112`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/release-smoke.sh#L112):
  the `sed -i ... "$OUT"/*.txt` redaction leaves the glob literal (and errors, ignored) when the directory has no
  `.txt` files.
- [`openai_client.py:24-25`](file://~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro/scripts/openai_client.py#L24-L25):
  `auth_headers()` raises a bare `FileNotFoundError` when `API_KEY_FILE` points at a missing file; a one-line
  message would save a confusing traceback in every evidence script.

## Checked and found sound

- `ci.sh` units mode: skip (rc 3) is recorded as a failure, matching `CONTRIBUTING.md:39` ("all three PASS").
- `check_evidence`/`benchmarks_headline.py check`: genuinely recompute hashes / regenerate the block from the JSONs
  and fail on drift (rc 0 today); pre-cutover exemption is deliberate.
- `nll-score.py` baseline pairing is fail-closed on a missing baseline text (`:231-232`), and the
  tokenization-differs path warns rather than silently scoring a shifted stream.
- `gpu-power-sample.sh` returns the wrapped command's exit status and kills the sampler on exit, as documented.
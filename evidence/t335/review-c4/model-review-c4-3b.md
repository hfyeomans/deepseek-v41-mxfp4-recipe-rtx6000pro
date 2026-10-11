# Docs-vs-code review C4-3 (read-only)

Compared README.md, docs/CONFIG.md, docs/BENCHMARKS.md, docs/TROUBLESHOOTING.md, CONTRIBUTING.md
against scripts/serve-hybrid.sh, scripts/serve_config.py, scripts/release-smoke.sh,
scripts/wait-ready.py, scripts/stop.sh, scripts/ci.sh, systemd/dsv41-flash.service.in.
No tracked file was modified; no serve/bench/docker/ci command was run; evidence/ and patches/ were
not read. Known items excluded (stop.sh stops only the legacy container; the 8000/8010 port and
dsv41-kt/legacy container-name split; the unknown-knob guard covering only KT_/DSPARK/NGRAM
prefixes; load-env.sh keeping trailing comments).

Everything below was verified by reading both the doc line and the code that implements it.

## 1. Medium — README.md:44 vs scripts/serve_config.py:493 (with CONFIG.md:98,100; BENCHMARKS.md:537)

- Doc says (in the "B. Three-tier hybrid" row): "above 64K the launcher allows one slot only
  without `ALLOW_UNTESTED_FIT=1`" and "**262K × 4 passes every gate at chunk 4096 since
  2026-10-09 (E34d-v (b) ...; OOMs at chunk 8192)**".
- Code does (guards(), serve_config.py:493): for `ctx > 65536` with `slots > 1`, the only accepted
  combination is `ctx <= 262144 and slots <= 4 and chunk == 8192 and experts == 219` with
  `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE == 1` (default on). So `CONTEXT_LENGTH=262144`
  with the default slots=4 launches; the README's stated passing shape (262K × 4 at chunk 4096) is
  refused without `ALLOW_UNTESTED_FIT=1`. docs/CONFIG.md:98/100 and docs/BENCHMARKS.md:537 already
  state the current rule correctly ("262K x 4 ... with the expandable lever", "≤ 4 slots, 219 /
  8192, lever on"), so README:44 is the stale side.
- Proposed fix: replace the parenthetical with the current rule — multi-slot above 64K is
  262K × ≤4 at 219/8192 with `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1` (default); 512K/1M
  are single-slot; the chunk-4096 pass was the pre-lever E34d-v (b) result.

## 2. Medium-low — docs/TROUBLESHOOTING.md:105 vs scripts/serve_config.py:269-272, 87-88, 418-427

- Doc says: "Verified with the injection hook: `KT_NAN_INJECT=row=1,bs=4,step=3,layer=20 bash
  scripts/serve-hybrid.sh` then `scripts/verify-nan-rejection.py`" — presented as a runnable
  reproducer.
- Code does: `KT_NAN_INJECT` has `sink=DIAGNOSTIC_ENV` (serve_config.py:269-272); `developer_only()`
  treats every DIAGNOSTIC_ENV knob as "a diagnostic" (line 87-88) and `resolve()` refuses a
  diagnostic set to a non-default value in the environment under any profile whose file is not
  `developer.env` (lines 418-427). Run as written (default `PROFILE=qualified`) the command fails
  with `FAIL config: KT_NAN_INJECT=...: KT_NAN_INJECT is a diagnostic; PROFILE=qualified refuses
  it. Run the experiment as PROFILE=developer ...`.
- Proposed fix: add `PROFILE=developer` to the example command (the same document explains the rule
  at line 182 and in the retired-workarounds table at line 194). Same class, same fix:
  TROUBLESHOOTING.md:117 tells the reader to "run soaks with `KT_NAN_PROBE_MIN_TOKENS=1`", also a
  diagnostic-env knob now.

## 3. Low — CONTRIBUTING.md:32 vs scripts/release-smoke.sh:72-75 and 90-103

- Doc says: `--full` adds "the stress, decode C1 / C4 and prefill 16K / 32K benches, **each** under
  a 500 ms nvidia-smi power / clock / temperature sampler (`gpu-power-sample.sh` →
  `power-<bench>.csv`, reduced to `power-summary.json` by `summarize-power.py --smoke`)".
- Code does: the stress run (`stress-kv-full.py`, lines 73-75) is invoked directly with no
  `sampled()` wrapper; only decode C1, decode C<slots>, prefill 16K and prefill 32K go through
  `sampled()` (lines 94-102). `summarize-power.py --smoke` consequently only sees
  `power-decode-c1/-c4/-prefill-16k/-prefill-32k.csv` (summarize-power.py:60-75), never a stress
  power CSV. (`gpu-power-sample.sh` does sample at `-lms 500`, so that part is accurate.)
- Proposed fix: drop "the stress," from the "each under a 500 ms ... sampler" clause, or wrap the
  stress invocation in `sampled()` too.

## 4. Low — docs/BENCHMARKS.md:15 vs scripts/serve_config.py:493, 503

- Doc says: "262K × 4 and 1M × 1 are per-rung overrides (`CONTEXT_LENGTH=…`, see CONFIG)".
- Code does: `CONTEXT_LENGTH` alone works for 262K × 4 (slots default 4, chunk 8192, 219 experts,
  lever on), but not for 1M × 1: `ctx > 65536 && slots > 1` is refused unless `ctx <= 262144`
  (line 493), and `ctx > 262144` additionally requires `MEM_FRACTION_STATIC=0.962` (line 503). A
  working 1M × 1 launch is `CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1
  MEM_FRACTION_STATIC=0.962`. (CONFIG.md:98/100 state the extra knobs; BENCHMARKS:15 does not.)
- Proposed fix: name the full 1M × 1 override set in this line (or say "see CONFIG for the per-rung
  override set" without implying `CONTEXT_LENGTH` alone).

## 5. Low — CONTRIBUTING.md:40 vs scripts/serve_config.py:503

- Doc says (release checklist step 3): the resolved line and manifest must read
  "`profile=qualified overrides=[]` (or only the rung's `CONTEXT_LENGTH` / `MAX_RUNNING_REQUESTS`)".
- Code does: the 512K × 1 and 1M × 1 rungs require `MEM_FRACTION_STATIC=0.962` (`ctx > 262144`,
  line 503), so those rungs' overrides lists necessarily contain `MEM_FRACTION_STATIC` as well; a
  rung launched with only the two named knobs is refused by `guards()`.
- Proposed fix: add `MEM_FRACTION_STATIC` to the allowed rung overrides (for 512K/1M).

## 6. Low — README.md:75 vs README.md:156 (doc-internal; patches/ not read per scope)

- Doc says: line 75 "51 patches on upstream `da64c5cb`"; line 156 "the overlay as a 50-patch
  series on upstream `da64c5cb`". Both cannot be right.
- Code side: `scripts/export-sglang-series.sh` writes `patches/sglang/SERIES.env` with a count field
  ("base, head, tree, count", line 7) and prints `exported $count patches` (line 49); `ci.sh:73`
  re-applies the series via `check-sglang-patch-series.sh`. I did not read `patches/` (out of
  scope), so the true count is unverified, but the two README numbers disagree.
- Proposed fix: reconcile both lines against `patches/sglang/SERIES.env` (count) and keep one
  number.

## 7. Low / borderline — docs/TROUBLESHOOTING.md:180 vs scripts/serve_config.py:482

- Doc says: the launcher refuses "`CHUNKED_PREFILL_SIZE ≥ 8192` above 208 experts (E22/E28)".
- Code does: the guard is `chunk >= 8192 and experts > 219` (line 482); 209-219 experts at chunk
  8192 pass the guards, 226 is refused. The same paragraph, two sentences later, states the
  correct rule ("chunk 8192 above 219 experts"), so the "208" figure reads as a leftover from the
  T3.16b era rather than a second, different rule.
- Proposed fix: change 208 → 219 in that sentence (or mark it explicitly as the historical rule).

## Checked and found consistent (no finding)

- CONFIG.md's knob table is generated from serve_config.py and ci.sh:68-70 enforces freshness; all
  defaults I spot-checked match (PORT 8000, CONTAINER_NAME dsv41-kt, SNAPSHOT dba1be0a…, context
  65536, slots 4, chunk 8192, experts 219, fraction 0.96, deferred 4, prefetch ahead, chunk-ahead 1,
  lever 1, reject-NaN 1, clamp 1, logprob chunk 512, CUDA_GRAPH_BS "1 2 3 4").
- CONFIG.md's claims about profiles (rc 2 with file:line, duplicate/unknown/removed refusal,
  canonical names, developer-only knobs by file name, `{repo}` expansion, env override precedence,
  `resolved:` line ending `profile=<name> overrides=[...]` with DRY_RUN excluded) match
  serve_config.py:354-450 and the shipped profiles under config/profiles/.
- CONFIG.md:21 (`MAX_TOTAL_TOKENS=""` / `CUDA_GRAPH_BS=""` = derive) matches empty_disables plus the
  `-n` guards in serve-hybrid.sh:91-97; derived cap slots × context + 16,384 matches BENCHMARKS
  pool figures (278,528 / 1,064,960).
- README:82 / CONTRIBUTING:31-32 gate set (same-prompt C4, NLL vs evidence/t328/default-219-8192/
  corpus with |cluster-robust z| ≤ 2, needles, health; `--full` adds stress + benches) matches
  release-smoke.sh, including `z_cluster` being the key read from pooled.json and the
  units-mode "refuses while dsv41-kt exists" check.
- wait-ready.py: default timeout 3600 s, pool-cap WARNING with exit code unchanged, container-log
  scan default dsv41-kt — matches TROUBLESHOOTING:290 and the 512K/0.962 advice in CONFIG.md:97.
- serve-hybrid.sh reads no env var that is not a registry knob or a registry-emitted variable
  (checked all `$VAR` reads), so CONFIG.md:3's "every knob ... declared once" holds.
- TROUBLESHOOTING:279's `--print` redaction claim holds: the key content is whitespace-stripped
  (serve-hybrid.sh:74) before `printf '%q'` + `sed`, so the `--api-key` token is always fully
  replaced.
- TROUBLESHOOTING:141's capture-list fix (`--cuda-graph-bs-decode 1 2 3 4`, default "1 2 3 4") and
  TROUBLESHOOTING:151's "serve-hybrid.sh prints a warning when KT_CPUINFER exceeds the core count"
  match serve_config.py:220-223 and 453-465.
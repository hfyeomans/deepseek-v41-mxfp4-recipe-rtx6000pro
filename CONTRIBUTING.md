# Contributing

## What this repository is

A feasibility study with a measured verdict, a phased plan, and scripts that fail closed. Contributions that change the verdict must change the evidence first.

## Rules

1. **Numbers come from scripts.** Any byte total, bandwidth or throughput in the docs must be reproducible by a script in `scripts/` and have its JSON in `evidence/`. If you change a number in prose, link the JSON that supports it.
2. **Pins are immutable.** Model revisions live in `scripts/load-env.sh`; image digests in `PROVENANCE.md` (`IMAGE` in `scripts/serve_config.py` for the hybrid; `.env` for the legacy chain). Changing a pin is a `PROVENANCE.md` change with the new header-derived byte total.
3. **No identifiers in tracked files.** `scripts/ci.sh` rejects home paths, LAN addresses, GPU UUIDs, PCI bus IDs, serial numbers and token-shaped strings. Scrub; do not whitelist.
4. **Unqualified stays labelled.** A profile is "qualified" only after `verify.py` passes and `bench-decode.py`/`bench-prefill.py` tables are in `docs/BENCHMARKS.md` from runs on the stated hardware and power condition.
5. **Negative results are recorded.** An experiment that fails is written up in `docs/BENCHMARKS.md` or `docs/TROUBLESHOOTING.md` with the failure mode, not deleted.
6. **Do not disturb the running deployment.** GPU experiments stop the GLM container first and restart it afterwards; say so in the PR.

## Before opening a PR

```bash
scripts/ci.sh
```

It runs syntax checks, unit tests, the 2-GPU verdict consistency check, shellcheck if installed, and the identifier scan.

## Test suites and where each runs (T3.24d census, 2026-10-10)

Three tiers. `ci.sh` is the CPU tier and must stay runnable on a host without a GPU, without the image, and without the overlay checkout (every in-image or overlay-dependent runner exits 3 = skipped, never fails, when its prerequisite is absent). The GPU tiers are run by hand before a release and their output lands under `evidence/`.

| Tier | Runner | Suites | Needs |
|---|---|---|---|
| CPU, every commit | `scripts/ci.sh` | `test-load-env.sh`, `test-checkpoint-budget.py`, `test-feasibility-calc.py`, `test-openai-client.py`; under `scripts/tests/`: `test_nll_stats.py`, `test_trace_prefix_memory.py`, `test-serve-config.sh`, `test_serve_config.py` (+ `serve_config.py check-profiles`, `docs/CONFIG.md` freshness), `test_evidence_manifest.py` (+ `evidence_manifest.py check-evidence`), `test_benchmarks_headline.py` (+ `benchmarks_headline.py check`, `docs/BENCHMARKS.md` headline freshness), `test_dspark_overlap_gate.py`, `check-sglang-patch-series.sh`; in-image CPU-torch runners `run-kt-buffer-tests.sh` → `test_kt_buffer_lifetime.py`, `run-kt-contract-tests.sh` → `test_kt_contract.py`, `run-kt-deferral-tests.sh` → `test_kt_deferral_adjacency.py`, `run-kt-provider-tests.sh` → `test_kt_provider.py`, `run-kt-parity-check.sh` → `kt-parity-replay.py`, `run-sm120-split-seal-tests.sh` → `test_sm120_split_seal.py`, `run-kt-timeline-tests.sh` → `test_kt_timeline.py`, `run-kt-runtime-tests.sh` → `test_kt_runtime.py`, `run-kt-nan-inject-tests.sh` → `test_kt_nan_inject.py`, `run-kt-remap-tests.sh` → `test_kt_remap.py`, `run-kt-prefill-tests.sh` → `test_kt_prefill.py`; then the feasibility verdict, shellcheck, required files, licence hash, executable bits, identifier scan | python3; docker + `dsv41-kt:recipe` + overlay checkout for the in-image runners (skipped otherwise) |
| GPU unit, before a release or after a kernel-path change | `scripts/release-smoke.sh units` | `run-indexer-tiling-tests.sh` → `test_indexer_row_tiling.py` (E34b), `run-moe-workspace-tests.sh` → `test_moe_workspace.py` (E34d-v (i)), `run-prefill-transient-tests.sh` → `test_indexer_k_gather.py` + `test_hc_combine_input.py` (E34d-v (ii)) | one free GPU for a few seconds each; refuses to run while the `dsv41-kt` container exists |
| Serve-level, the qualification gate set | `scripts/release-smoke.sh serve --output-dir evidence/<dir> [--context N --slots N --full]` against a server launched by `scripts/serve-hybrid.sh`; it refuses (rc 2) unless the launch manifest describes the running container and its `CONTEXT_LENGTH` / `MAX_RUNNING_REQUESTS` equal `--context` / `--slots` | same-prompt C4 (`verify-batch-consistency.py --same-prompt --flush-cache`), NLL corpus vs `evidence/t328/default-219-8192/corpus` (`nll-score.py --corpus --compare-dir`: the run itself, then pooled ΔNLL ≤ +0.002 and \|cluster-robust z\| ≤ 2 over ≥ 2 texts; and, by default, 0 tokens moved vs the same-map identity baseline `evidence/t343b/64k4/corpus` (`--identity-dir`; `NLL_REQUIRE_IDENTICAL=0` for a serve on another expert map / experts / chunk, or `IDENTITY_CORPUS=<corpus scored on that shape>`)), needles (`verify.py --context`, ≥ 95 % of the context by `prompt_tokens`), container-log health; `--full` adds the stress (every slot ≥ 95 % filled, ≥ 1 output token; not sampled) and the decode C1 / C4 and prefill 16K / 32K benches, each of the four under a 500 ms nvidia-smi power / clock / temperature sampler (`gpu-power-sample.sh` → `power-<bench>.csv`, reduced to `power-summary.json` by `summarize-power.py --smoke`, one sampled entry per bench or the smoke fails) | the qualified serve; every JSON is manifest-stamped by its tool (a manifest whose container cannot be inspected is refused unless `EVIDENCE_MANIFEST_ALLOW_STALE=1`), the `.txt` logs are redacted in place |

Every `scripts/tests/*` file is in exactly one tier (checked by hand 2026-10-10: 19 runners / tests in `ci.sh`, 3 GPU runners in `release-smoke.sh units`, `golden/` consumed by `test_serve_config.py`). A new test goes into `ci.sh` if it runs on CPU torch or plain Python, else into `release-smoke.sh units`.

## Release checklist

1. `scripts/ci.sh` → `repository checks passed`, rc 0 (includes the overlay patch-series tree-hash check).
2. GPUs free: `scripts/release-smoke.sh units` → all three PASS (evidence under `evidence/t324d/` or the release directory).
3. Launch the qualified profile (`bash scripts/serve-hybrid.sh`), then `scripts/release-smoke.sh serve --output-dir evidence/<release>/64k-4slot --full`. For a ladder rung the launch overrides and the smoke flags must agree, e.g. `CONTEXT_LENGTH=262144 bash scripts/serve-hybrid.sh` then `--context 262144 --slots 4`, or `CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962 bash scripts/serve-hybrid.sh` then `--context 1048576 --slots 1` (the smoke refuses a mismatch, R12). The resolved line and `manifest.json` must read `profile=qualified overrides=[]` (or only the rung's `CONTEXT_LENGTH` / `MAX_RUNNING_REQUESTS` / `MEM_FRACTION_STATIC`).
4. `PROVENANCE.md` "qualified stack at a glance" matches the manifest: base SHA, overlay head + tree (`patches/sglang/SERIES.env`), image id, kt-kernel wheel + patch list, snapshot revision, expert-map hash.
5. If a rung was re-qualified, point `config/benchmarks-headline.json` at the new evidence directory and run `python3 scripts/benchmarks_headline.py write` (the table is generated; never edit the block by hand), then tag the overlay commit in the overlay repo (`git tag -a recipe-<yyyy-mm> <sha>`; `recipe-2026-10` = `682fc6a0`) and record the tag in PROVENANCE.
6. Commit locally; push to the private remote `origin` (`github.com/hfyeomans/deepseek-v41-mxfp4-recipe-rtx6000pro`) only on an explicit owner instruction.

## Adding an experiment

Add it to `plan/EXPERIMENTS.md` with a question, a numeric hypothesis, a procedure that names the script, a pass criterion, and where the result lands. Add the task to `plan/TASKS.md` with its status. Write the script under `scripts/` with a `--output` JSON flag and standard-library-only dependencies unless it must run inside a torch container (say so in the docstring).

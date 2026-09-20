# Contributing

## What this repository is

A feasibility study with a measured verdict, a phased plan, and scripts that fail closed. Contributions that change the verdict must change the evidence first.

## Rules

1. **Numbers come from scripts.** Any byte total, bandwidth or throughput in the docs must be reproducible by a script in `scripts/` and have its JSON in `evidence/`. If you change a number in prose, link the JSON that supports it.
2. **Pins are immutable.** Model revisions live in `scripts/load-env.sh`; image digests in `.env`/`PROVENANCE.md`. Changing a pin is a `PROVENANCE.md` change with the new header-derived byte total.
3. **No identifiers in tracked files.** `scripts/ci.sh` rejects home paths, LAN addresses, GPU UUIDs, PCI bus IDs, serial numbers and token-shaped strings. Scrub; do not whitelist.
4. **Unqualified stays labelled.** A profile is "qualified" only after `verify.py` passes and `bench-decode.py`/`bench-prefill.py` tables are in `docs/BENCHMARKS.md` from runs on the stated hardware and power condition.
5. **Negative results are recorded.** An experiment that fails is written up in `docs/BENCHMARKS.md` or `docs/TROUBLESHOOTING.md` with the failure mode, not deleted.
6. **Do not disturb the running deployment.** GPU experiments stop the GLM container first and restart it afterwards; say so in the PR.

## Before opening a PR

```bash
scripts/ci.sh
```

It runs syntax checks, unit tests, the 2-GPU verdict consistency check, shellcheck if installed, and the identifier scan.

## Adding an experiment

Add it to `plan/EXPERIMENTS.md` with a question, a numeric hypothesis, a procedure that names the script, a pass criterion, and where the result lands. Add the task to `plan/TASKS.md` with its status. Write the script under `scripts/` with a `--output` JSON flag and standard-library-only dependencies unless it must run inside a torch container (say so in the docstring).

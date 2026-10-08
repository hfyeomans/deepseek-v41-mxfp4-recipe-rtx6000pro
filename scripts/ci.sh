#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash -n "$ROOT"/scripts/*.sh
python3 -m py_compile "$ROOT"/scripts/*.py
python3 -m json.tool "$ROOT/examples/openai-request.json" >/dev/null
for f in "$ROOT"/evidence/*.json; do python3 -m json.tool "$f" >/dev/null; done
"$ROOT/scripts/test-load-env.sh"
python3 "$ROOT/scripts/test-checkpoint-budget.py"
python3 "$ROOT/scripts/test-feasibility-calc.py"
python3 "$ROOT/scripts/test-openai-client.py"
# kt-kernel buffer-lifetime tests run inside the recipe image (no GPU); skipped where docker/image is absent.
python3 -m py_compile "$ROOT"/scripts/tests/*.py
bash -n "$ROOT"/scripts/tests/*.sh
# T3.28 shape-aware NLL gate: paired/pooled statistics in nll-score.py (pure Python, no server).
python3 "$ROOT/scripts/tests/test_nll_stats.py"
if bash "$ROOT/scripts/tests/run-kt-buffer-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt buffer-lifetime tests (exit $rc)"; exit 1; fi
fi
# SGLang->kt boundary/topology contract tests (T3.16d) run the same way against the overlay checkout.
if bash "$ROOT/scripts/tests/run-kt-contract-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt contract tests (exit $rc)"; exit 1; fi
fi
# kt expert-deferral adjacency tests (T3.16e): slot/incremental/sync decisions across adjacent layers.
if bash "$ROOT/scripts/tests/run-kt-deferral-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt deferral tests (exit $rc)"; exit 1; fi
fi
# CpuExpertProvider seam conformance (T3.19b): FakeProvider + KtKernelProvider over the deferral recorders; in-image.
if bash "$ROOT/scripts/tests/run-kt-provider-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt provider tests (exit $rc)"; exit 1; fi
fi
# T3.16e(3) kt parity: recorded serving-path kt_output vs standalone kt replay, bit-for-bit; in-image, CPU-only.
if bash "$ROOT/scripts/tests/run-kt-parity-check.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt parity check (exit $rc)"; exit 1; fi
fi
# SM120 page-split buffer seal (review C5): in-image against the overlay checkout; skipped without it.
if bash "$ROOT/scripts/tests/run-sm120-split-seal-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL sm120 split seal tests (exit $rc)"; exit 1; fi
fi
# T3.19a decode timeline recorder + report (scripted-clock, no GPU): in-image against the overlay checkout; skipped without it.
if bash "$ROOT/scripts/tests/run-kt-timeline-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt timeline tests (exit $rc)"; exit 1; fi
fi
if bash "$ROOT/scripts/tests/run-kt-runtime-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt runtime tests (exit $rc)"; exit 1; fi
fi
# T3.19d streamed-prefill state machine (PrefillPipeline, pure Python): in-image against the overlay checkout; skipped without it.
if bash "$ROOT/scripts/tests/run-kt-prefill-tests.sh"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL kt prefill tests (exit $rc)"; exit 1; fi
fi
# T3.29 launcher config guards (DRY_RUN=1, no docker/GPU): qualified profiles accepted, unqualified refused.
if bash "$ROOT/scripts/tests/test-serve-config.sh" >/dev/null; then :; else echo "FAIL serve config tests"; exit 1; fi
# DSpark multi-stream-overlap gate (review C4 / Bug A): pure Python against the overlay checkout; skipped without it.
if python3 "$ROOT/scripts/tests/test_dspark_overlap_gate.py"; then :; else
  rc=$?; if [[ $rc -ne 3 ]]; then echo "FAIL dspark overlap gate tests (exit $rc)"; exit 1; fi
fi
# The verdict in README must match the calculator: 2 GPUs must not fit GPU-resident, but must fit the hybrid with NVMe Engram.
if python3 "$ROOT/scripts/feasibility-calc.py" "$ROOT/evidence/budget-upstream-bf16fp8.json" --gpus 2 --layout engram-host >/dev/null; then
  echo 'FAIL calculator says 2 GPUs fit; README verdict is stale'; exit 1
fi
if ! python3 "$ROOT/scripts/feasibility-calc.py" "$ROOT/evidence/budget-upstream-bf16fp8.json" --gpus 2 --layout hybrid-cpu --engram nvme >/dev/null; then
  echo 'FAIL calculator says the 2-GPU hybrid does not fit; README verdict is stale'; exit 1
fi
if command -v shellcheck >/dev/null; then shellcheck -x -P "$ROOT" "$ROOT"/scripts/*.sh; else echo 'shellcheck not installed; skipped'; fi
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib, re, subprocess, sys
root = Path(sys.argv[1])
required = [
    'README.md', 'ATTRIBUTIONS.md', 'THIRD_PARTY_NOTICES.md', 'PROVENANCE.md', 'CONTRIBUTING.md',
    'NOTICE', 'LICENSE', 'THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt',
    'docs/FEASIBILITY.md', 'docs/HARDWARE.md', 'docs/BENCHMARKS.md', 'docs/TROUBLESHOOTING.md', 'docs/LEARNING.md',
    'plan/README.md', 'plan/EXPERIMENTS.md', 'plan/TASKS.md', 'plan/RESEARCH.md', 'plan/ROADMAP.md',
    'evidence/budget-upstream-bf16fp8.json', 'evidence/hardware-inventory.json',
]
for name in required:
    if not (root / name).is_file():
        raise SystemExit(f'missing {name}')
lic = root / 'THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt'
if hashlib.sha256(lic.read_bytes()).hexdigest() != 'f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040':
    raise SystemExit('DeepSeek MIT license hash mismatch')
git_tree = (root / '.git').exists()
if git_tree:
    staged = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', 'scripts'], text=True)
    for line in staged.splitlines():
        mode, _, _, path = line.split(maxsplit=3)
        if path.endswith(('.sh', '.py')) and not path.split('/')[-1].startswith('openai_client') and mode != '100755':
            raise SystemExit(f'script is not executable in Git: {path} ({mode})')
patterns = [
    re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),
    re.compile(r'hf_[A-Za-z0-9]{30,}'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
    re.compile(r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'/(?:home|media)/[A-Za-z0-9._-]+/'),
    re.compile(r'\b(?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b'),
    re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F-]{8,}'),
    re.compile(r'\b0000:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\b'),
    re.compile(r'"serial"|Serial Number: [A-Z0-9]{6,}'),
]
if git_tree:
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    scan = [root / p for p in files]
else:
    scan = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
for path in scan:
    if not path.is_file() or path.suffix in {'.png', '.jpg'} or path == root / 'scripts' / 'ci.sh':
        continue  # the scanner's own pattern list would match itself
    text = path.read_text(errors='ignore')
    for pattern in patterns:
        for m in pattern.finditer(text):
            raise SystemExit(f'possible identifier or secret in {path.relative_to(root)}: {pattern.pattern} -> {m.group(0)[:40]}')
print('repository checks passed')
PY

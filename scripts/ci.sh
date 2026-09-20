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
# The verdict in README must match the calculator: 2 GPUs must not fit.
if python3 "$ROOT/scripts/feasibility-calc.py" "$ROOT/evidence/budget-upstream-bf16fp8.json" --gpus 2 --layout engram-host >/dev/null; then
  echo 'FAIL calculator says 2 GPUs fit; README verdict is stale'; exit 1
fi
if command -v shellcheck >/dev/null; then shellcheck -x -P "$ROOT" "$ROOT"/scripts/*.sh; else echo 'shellcheck not installed; skipped'; fi
python3 - "$ROOT" <<'PY'
from pathlib import Path
import hashlib, re, subprocess, sys
root = Path(sys.argv[1])
required = [
    'README.md', 'ATTRIBUTIONS.md', 'THIRD_PARTY_NOTICES.md', 'PROVENANCE.md', 'CONTRIBUTING.md',
    'NOTICE', 'LICENSE', 'THIRD_PARTY_LICENSES/DeepSeek-V4.1-Flash-MIT.txt',
    'docs/FEASIBILITY.md', 'docs/HARDWARE.md', 'docs/BENCHMARKS.md', 'docs/TROUBLESHOOTING.md',
    'plan/README.md', 'plan/EXPERIMENTS.md', 'plan/TASKS.md', 'plan/RESEARCH.md',
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

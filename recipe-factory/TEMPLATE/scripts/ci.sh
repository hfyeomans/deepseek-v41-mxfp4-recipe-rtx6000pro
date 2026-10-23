#!/usr/bin/env bash
# CI template for a new recipe. Lifted from the reference repo's scripts/ci.sh, minus the
# model-specific gates. The gates every new recipe should keep, in order:
#   syntax → registry/profile consistency → evidence integrity → identifier scan → generated-doc
#   staleness → the verdict-vs-calculator pin (the P0 verdict must stay true in CI).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Syntax: every script parses.
bash -n "$ROOT"/scripts/*.sh
python3 -m py_compile "$ROOT"/scripts/*.py

# 2. Typed knob registry: profiles load through the registry and agree with its defaults
#    (the registry is the single source of truth), and the generated CONFIG table is not stale.
if python3 "$ROOT/scripts/serve_config.py" check-profiles >/dev/null; then :; else
  echo "FAIL config/profiles drift from scripts/serve_config.py"; exit 1; fi
if ! diff -q <(python3 "$ROOT/scripts/serve_config.py" markdown) \
     <(sed -n '/<!-- generated: python3 scripts\/serve_config.py markdown -->/,$p' "$ROOT/docs/CONFIG.md" | tail -n +2) >/dev/null; then
  echo "FAIL docs/CONFIG.md is stale: regenerate with python3 scripts/serve_config.py markdown"; exit 1; fi

# 3. Evidence integrity: every JSON under evidence/ parses, and every stamped file matches its
#    sibling manifest.json (evidence_manifest.py check-evidence; port it from the reference repo).
for f in "$ROOT"/evidence/*.json; do python3 -m json.tool "$f" >/dev/null; done
if python3 "$ROOT/scripts/evidence_manifest.py" check-evidence "$ROOT/evidence"; then :; else
  echo "FAIL evidence/ has stamped files that do not match their manifest.json"; exit 1; fi

# 4. Generated BENCHMARKS headline is not stale (benchmarks_headline.py check; port it).
if python3 "$ROOT/scripts/benchmarks_headline.py" check >/dev/null; then :; else
  echo "FAIL docs/BENCHMARKS.md headline block is stale"; exit 1; fi

# 5. The P0 verdict must stay true: re-run the calculator and require the documented
#    fits/does-not-fit outcome (rc 3 exactly = "does not fit", not a crash).
# python3 "$ROOT/scripts/feasibility-calc.py" "$ROOT/evidence/budget.json" --gpus 2 --layout hybrid || ...

# 6. Identifier / secret scan over every tracked file (the scanner's own patterns excluded).
python3 - "$ROOT" <<'PY'
from pathlib import Path
import re, subprocess, sys
root = Path(sys.argv[1])
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
if (root / '.git').exists():
    files = subprocess.check_output(['git', '-C', str(root), 'ls-files'], text=True).splitlines()
    scan = [root / p for p in files]
else:
    scan = [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts]
for path in scan:
    if not path.is_file() or path.suffix in {'.png', '.jpg'} or path == root / 'scripts' / 'ci.sh':
        continue
    text = path.read_text(errors='ignore')
    for pattern in patterns:
        for m in pattern.finditer(text):
            raise SystemExit(f'possible identifier or secret in {path.relative_to(root)}: {pattern.pattern} -> {m.group(0)[:40]}')
print('repository checks passed')
PY

# 7. Executable bits: every tracked script under scripts/ is executable in Git (mode 100755).
if [ -d "$ROOT/.git" ]; then
  python3 - "$ROOT" <<'PY'
import subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
staged = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--stage', 'scripts'], text=True)
for line in staged.splitlines():
    mode, _, _, path = line.split(maxsplit=3)
    if path.endswith(('.sh', '.py')) and mode != '100755':
        raise SystemExit(f'script is not executable in Git: {path} ({mode})')
PY
fi

echo "ci.sh: all gates green"

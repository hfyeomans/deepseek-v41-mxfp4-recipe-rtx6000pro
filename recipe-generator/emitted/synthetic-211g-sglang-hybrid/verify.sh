#!/usr/bin/env bash
# Generated smoke: endpoint up + one completion. The full gate suite
# (NLL identity, needles, stress, batch consistency, health) is the
# recipe-factory gate discipline - see README's checklist.
set -euo pipefail
curl -sf http://127.0.0.1:8000/v1/models
curl -sf http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}'

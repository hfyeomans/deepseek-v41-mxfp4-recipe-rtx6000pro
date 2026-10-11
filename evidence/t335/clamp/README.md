# T3.35 clamp reproduction (overlay patch 0051)

Serve: `CONTEXT_LENGTH=262144 scripts/serve-hybrid.sh` with the launcher default
`SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS=1` (container env), restarted 2026-10-11 00:06 UTC.
Client: Amp review thread C4-3 (docs-vs-code), which sends `max_completion_tokens: 128000`
on every turn and had been refused with HTTP 400 at 160,663 input tokens on the stock check.

`server-log-excerpt.txt` is the `rg 'clamped|longer than the model'` slice of `docker logs dsv41-kt`:
the first request after the restart clamps `max_new_tokens 128000 -> 101647` at 160,497 input
tokens and returns 200; 11 turns are clamped in total; the final line is the genuine over-length
refusal at 262,282 input tokens, reached because the harness never compacted (see LEARNING §11).

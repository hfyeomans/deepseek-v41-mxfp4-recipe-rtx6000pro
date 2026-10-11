# T3.35 — the served model reviews its own recipe (262K × 4, Amp harness)

Serve: `CONTEXT_LENGTH=262144 scripts/serve-hybrid.sh` (219 / 165, chunk 8192, D = 4, lever on,
`--api-key`), reached by Amp's servers through `tailscale funnel` → a logging proxy (127.0.0.1:8001)
→ the container; threads ran on a local Amp runner in this checkout, agent mode `deepseek-v4.1-flash`
(Custom URL provider mapping `deepseek/deepseek-v4.1-flash -> dsv41-flash-hybrid`). Reasoning on;
Amp sends `max_completion_tokens 128000` on every turn.

| Run | Threads | Brief | Requests | Wall | Cache hit | Decode (median tok/s) | TTFB med / p90 | Report |
|---|---|---|---|---|---|---|---|---|
| C1 | 1 | whole recipe, code-first | 80 (80 × 200) | 43 min | 92.3 % | 55.8 (1 running) | 5.4 s / 12 s | `review-c1/model-review.md` (14 findings) |
| C4 | 4 at once | C4-1 launch path, C4-2 evidence tooling, C4-3 docs vs code, C4-4 kernels / scripts | 198 (187 × 200, 11 × 400) | 70 min | 89.0 % | 55.6 / 85.2 / 106.7 / 124.4 at 1 / 2 / 3 / 4 running | 9.2 s / 14.5 s | `review-c4/model-review-c4-{1,2,4}.md`; C4-3 produced nothing (every turn 400 past 134,144 input tokens) |
| C4-3b | 1 | docs vs code, scoped (≤ 40 tool calls, no whole-file reads), on the clamped serve | 11 × 200 | 14 min | — | ≈ 54 | — | `review-c4/model-review-c4-3b.md` (7 findings) |

Power during C4: busy-median 193 / 205 W per GPU, max 85 °C, no thermal throttling (`review-c4/power-summary.json`).
`start.txt` / `end.txt` are the `nvidia-smi` + container state before and after; `server-log.txt` the
redacted `docker logs` slice; `clamp/` the overlay-0051 reproduction.

**Grading** (reviewer of the reviews: the orchestrating Amp thread, reading every cited file:line):
the four delivered reports hold 37 + 7 real findings, consolidated with status in
`plan/REVIEW-T335-FINDINGS.md` (fix pass T3.43). Invented or wrong: C4-1 called the host a "4-GPU host"
(it has 2); §5 of that document lists the rest. The highest-value catch was R1 — GitHub CI on
`8cc736c` is red because the serve-config test merges stderr and 4-CPU runners trip the
`KT_CPUINFER` warning — which no local run had shown.

**What the run taught the recipe:** stock SGLang refuses prompt + `max_new_tokens` > context, so a
client with a fixed 128K completion ceiling could use only 134K of the 262K window — overlay 0051
(`SGLANG_DSV41_CLAMP_MAX_NEW_TOKENS`, launcher default 1) clamps instead; with it on, the next wall is
the real one, and the harness has to compact before the prompt reaches 262,144 tokens (the resumed
C4-3 did not, and was unrecoverable). Four concurrent review threads gave ≈ 2.2 × the token throughput
of one at the same per-stream rate (31 tok/s each at 4 running); a whole-repository review costs about
one GPU-hour either way.

# E32 — HumanEval pass@1 greedy on the served hybrid: 161/164 = 0.9817 (2026-10-09)

Served on the 262K × 4 chunk-4096 rung serve (overlay `1eeb48cd`, 219 / 165 experts, D=4, spec off,
`CHUNKED_PREFILL_SIZE=4096`; same server as `../e34-indexer-memory/d-v-a/262k-4slot-chunk4096/`,
whose `manifest.json` this one copies). ~16 min generate + ~1 min score.

Question (EXPERIMENTS E32): the NLL gate is a relative instrument; what does the exact-weight hybrid
score on the one public test the other published V4.1-Flash recipe prints (mlx-stream, lossy 3.0 bpw
EXL3 experts: **158/164 = 0.9634**, failing 32, 83, 132, 140, 145, 163)?

## Result

| | measured | hypothesis | verdict |
|---|---|---|---|
| pass@1 greedy | **161/164 = 0.9817**; failed **132, 140, 145** | (1) ≥ 158 | **met** |
| vs mlx-stream's six failures | we pass 32, 83, 163 (3/6); our three failures are a subset of its six; excluding those six **158/158** | — | exact experts do not lose a problem the lossy repack passes |
| decode per request | median **42.23 tok/s** (min 41.04, max 43.36) | (2) ≥ 38 | **met** |
| TTFT | median **1.049 s**, max 2.01 s | (2) < 1 s | missed by 5 % (chat-template prompts of ~200 tokens through the chunk-4096 serve; the C1 review fixture on the same serve read 0.618 s) |
| finish | 164 × `stop`; 164 × full-function answers (0 fell back to prompt + completion); median 166 completion tokens, 33,197 total; 962 s wall | — | no truncation, no missing code block |

The three failures are the model's own reading of three ambiguous specs, not serving defects:
132 (`is_nested`: returns `max_balance >= 2`, misses the "valid subsequence" case), 140 (`fix_spaces`:
regex collapses the 3-space run to `-` correctly but maps 1–2 spaces to the same count of `_`),
145 (`order_by_points`: digit sum of `abs(n)`, the reference treats the sign on the leading digit).
Each is 146–207 tokens, finish `stop`, and asserts on the reference tests, not on an exception.

A score below 155 would have triggered a correctness investigation; 161 clears the bar, and the
score beside the T3.28 NLL gate gives a later config a public number to hold.

## Method

`scripts/bench-humaneval.py --url http://127.0.0.1:8000 --model dsv41-flash-hybrid --output-dir
evidence/e32-humaneval`: 164 problems from `HumanEval.jsonl.gz` (sha256 in `report.json`), one
`/v1/chat/completions` each, temperature 0, seed 0, thinking off, max_tokens 4096, concurrency 1;
the last fenced code block is the answer (the dataset prompt is prepended only when the block lacks
the entry point — 0 cases here). Scoring: each program plus the problem's `check()` runs in a
throwaway `dsv41-kt:recipe` container with `--network none`, 20 s per problem (`runner.py`,
`results.json`). The scorer was validated before the run on a canonical solution, a wrong one and a
hang.

## Files

`report.json` (stamped: generate + score summaries, dataset hash), `samples.jsonl` (per problem: raw
text, extracted program, mode, TTFT, decode tok/s, tokens, finish), `programs/NNN.py` (what was run),
`results.json` (per-program pass / return code / stderr tail), `runner.py`, `run.txt`,
`manifest.json`.

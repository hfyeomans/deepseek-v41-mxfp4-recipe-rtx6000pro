# T3.9h-b — n-gram speculation on DSV4, done properly (E43)

2026-10-11/12. Five developer-profile serves on the launcher default (219 / 165, chunk 8192, el8-mix, lever
on, both determinism levers, exact top-k, 262K × 4), overlay `0887a342` = patch 0059 (`ngram_worker.py`
chain-only verify: `chainify_drafts()` refuses breadth ≠ 1, skips `move_accept_tokens_to_target_kvcache`
because the DSV4 pool has no `move_kv_cache`, calls `clear_unaccepted_c128_draft_states` after verify as the
EAGLE / DSpark path does). Question (from E23 R6 and the T3.9h crash): does n-gram speculation pay on this
hybrid once it runs at all, and is its output correct?

## Answer

**No, and no.** NGRAM reaches the throughput bar only on prompt-copying work (+60 / +76 % C1 on copy-code /
copy-table), loses 26 % on the review prompt and 63 % at C4, and — the finding that closes it — **its decode
state is wrong whenever draft rows are rejected**: a teacher-forced re-score of its own greedy output finds a
token it emitted at ≈ 0 nats that the model, re-fed the same prefix, scores at −20.5 nats (copy-table, output
position 20: a seventh `|---` cell in a six-column table), plus 8 flips > 0.5 nats on review (max 3.1) and 6
on a 14.8K prompt. Spec-off reads 0 / 0 / 1 / 1 on the same probe (max gap 0.75 nats — the known decode-vs-
prefill reduction noise, E38). The defect is deterministic (3 serves, bit-identical numbers), disappears when
the corpus is warm (accept ≈ 5, 384 / 384, max |Δlogp| 0.000), and **reproduces with the target runner
forced eager** (`SGLANG_BCG_DEBUG_EAGER=target`, log line "debug-eager forced for target runner"), so it is a
verify-path state defect under rejected rows, not a CUDA-graph replay one. The NLL corpus gate cannot see it
(teacher-forced scoring never rejects a draft: bit-identical 1.4875254 on every serve), nor can `greedy-drift`
(bitwise identical: it compares the serve with itself), nor the needle gate (6 / 6). The `NGRAM` knob stays
off and is now documented as not recommended; the state fix is T3.9h-c.

## Serves

| dir | knobs | result |
|---|---|---|
| `a-spec-off/` | none (developer profile) | verify ALL PASS; C1 review 62.84 / 59.57 wall, copy-code 53.71 / 50.35, copy-table 51.76 / 51.27; C4 122.60 / 117.63; drift bitwise; NLL 1.4875254 |
| `b-ngram/` | `NGRAM=1 NGRAM_DRAFT_TOKENS=5 NGRAM_BREADTH=1` | verify ALL PASS; C1 review 46.58 / 44.53 (min 30.12), copy-code 85.86 / 77.07, copy-table 91.06 / 88.71; C4 45.90 / 39.29; drift bitwise; NLL 1.4875254; `consistency-{first,cold,warm}.json` |
| `c-ngram-eager/` | + `SGLANG_BCG_DEBUG_EAGER=1` | **no-op** (the overlay reads `draft` / `target`; the registry wrongly typed it `flag`, fixed in this commit); graph-mode replicate: cold probe bit-identical to `b-ngram` |
| `d-ngram-eager-target/` | + `SGLANG_BCG_DEBUG_EAGER=target` | target runner eager (both ranks logged it); cold probe bit-identical to `b-ngram`: copy-table 383 / 384, gap 20.5; review 354 / 384, 8 flips; long12000 368 / 384, 6 flips |
| `a2-spec-off/` | none | probe baseline: copy-code 384 / 384 (max 0.001), copy-table 384 / 384 (0.00015), review 366 / 384, 1 flip (0.625), long12000 374 / 384, 1 flip (0.75) |

Bench driver (serves A / B): `scripts/verify.py`, `bench-decode.py` C1 on the review, copy-code and copy-table
prompts and C4, `greedy-drift.py --long-tokens 12000`, `nll-score.py` corpus. Probe (serves B–D, A2):
`scripts/spec-consistency.py --flush-cache` (new, this task).

## The probe: `scripts/spec-consistency.py`

Greedy `/generate` (chat-templated, `</think>` closed so the model answers directly) with per-token logprobs,
then the prompt + its own output re-scored through `input_ids` with `logprob_start_len`; reports argmax
agreement, flips with a gap above `--margin` (0.5 nats), max |decode − forced| logprob, `spec_accept_length`.
`--flush-cache` (GET `/flush_cache`) resets the prefix cache **and the n-gram corpus**, so every serve's cold
probe sees the same acceptance history. `--compare <json>` prints the delta against a previous run. A spec
path that is correct must agree with teacher forcing to numerics noise (≤ 0.6 nats here, E38); a 20-nat gap
is a wrong KV / state, not a rounding difference.

## What the eager run excludes, and what is left

Excluded: CUDA-graph replay of the verify forward (eager target reproduces it), the multi-stream overlap race
(Bug A gate is active: "NGRAM on SM120: multi-stream overlap forced off"), the kt CPU tier (bit-reproducible,
E38), the C128 ring (cleared for rejected rows, patch 0059), the online-C128 path (off here), the numerics
levers (both on). Warm-corpus runs (accept ≈ 5, almost no rejected rows) are clean, so the defect needs a
rejected draft row in the step before.

Suspects, in order (`deepseek_v4_backend.py` raw-verify path, `deepseek_v4_memory_pool.py`): (1) the c4
indexer compress states written by `_low_ratio_compress_torch(..., fuse_index_store=True)` for the rejected
rows — position-keyed ring slots that the next real token should overwrite, unless the group pooling
(`_low_ratio_write_group` at `group_pos`) or the index-k store keeps the rejected contribution; (2) the SWA
paged ring (`storage=paged` under NGRAM, not `request_window`) for rejected positions; (3) the verify
metadata's `seq_lens` vs the accepted length when the bonus token is the only accepted one. DSpark's verify
never calls the C128 clear and E23 measured it at ≤ 0.6 nats, so whatever DSpark does for its rejected rows
is the first thing to diff against (T3.9h-c).

## Decision

- `NGRAM` stays `0`; knob doc says not recommended with the numbers; no profile change (owner call whether
  to make it developer-only or refuse it until T3.9h-c).
- Patch 0059 stays in the series (inert unless `NGRAM=1`; it turns a crash into a measurable path).
- `SGLANG_BCG_DEBUG_EAGER` is a `str` knob (`draft` / `target`) from this commit; `=1` was silently ignored.
- T3.9h-c registered (low): find and fix the rejected-row state defect; gate = `spec-consistency.py` cold
  probe within the spec-off baseline on all four prompts, then re-run the serve-B benches. Only then can the
  copy-workload win be offered as an opt-in.

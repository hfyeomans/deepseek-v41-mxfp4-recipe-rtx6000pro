# T3.13 / C10 — a NaN row fails one request, not the server (2026-10-10)

Closure of the T3.13 ledger criterion (G0/C10, 2026-09-24): **(a)** an injected-NaN test fails one
request of a C4 batch while the other three complete; **(b)** the E22c assert is bisected or pinned
as a reproducer. (a) is this run; (b) closes by attribution (below).

Deployment profile (`bash scripts/serve-hybrid.sh`: 219 / 165, D=4, chunk 8192, 64K × 4,
`MEM_FRACTION_STATIC=0.96`, spec off) plus the diagnostic hook
`KT_NAN_INJECT=row=1,bs=4,step=3,layer=20`. Overlay `682fc6a0` (served as `124ca1de+dirty`, the
same tree), image `dsv41-kt:recipe`, one serve 18:30–18:37 UTC (`manifest.json`, hash stamped into
both JSONs). Ready in ≈ 5 min, `wait-ready` TTFT 18.07 s.

## What the overlay does now (default on, `KT_REJECT_NAN_ROWS=1`)

The sampler runs outside the decode graph. Before sampling it flags every logits row that holds a
NaN or +inf, then rewrites those values (NaN → −1e30, +inf → 1e30; −inf is kept so grammar masks
still work) so the sampling kernels never see a non-finite row. Well-formed rows are untouched —
the rewrite is bit-identical on finite input (checked with `torch.equal` in-image). The flag rides
D2H with `next_token_ids`; the scheduler's batch-result processor aborts a flagged request with
`FINISH_ABORT(..., 500, "NonFiniteLogits")` before `update_finish_state`, logs one ERROR line
`[non-finite-logits] rid=… row=… mode=decode|prefill` per rank, and the tokenizer manager returns
HTTP 500 `{"object":"error","message":"non-finite logits (NaN/Inf) were produced for this request;
it was aborted, the server keeps serving", …}` to that one client. Nothing is retried or clamped
into a token: the request fails, the batch shrinks by one row, the others continue.

`KT_NAN_INJECT` (rank 0 only, fires once) adds a NaN to one row of the MoE output after the CPU-tier
add on the n-th bs=b decode replay — inside the captured graph, at the point a kt-kernel defect
would surface. It is a diagnostic; production leaves it unset. Unit tests:
`scripts/tests/test_kt_nan_inject.py` (3, in-image on CPU torch, wired into `ci.sh`).

## (a) injected C4: 1 aborted, 3 complete, server up — `verify-nan-rejection.{txt,json}`

`scripts/verify-nan-rejection.py` (bs=1 references → injected C4 → clean C4 → `/health`):

| phase | slot 0 | slot 1 | slot 2 | slot 3 |
|---|---|---|---|---|
| injected C4 (64 tok, greedy) | 200, 64 tok, prefix 290/363 | 200, 64 tok, gib 0.11 (ref 0.22) | **500 `non-finite logits …` (row 1 of the bs=4 replay)** | 200, 64 tok, prefix 244/307 |
| clean C4 afterwards | 200, 64 tok, prefix 323/363 | 200, 64 tok | 200, 64 tok, prefix 278/278 | 200, 64 tok |
| `/health` after | 200 | | | |

Server log (`server.log` lines 530–535, `health-hits.txt`): `[KT_NAN_INJECT] poisoning row 1 of the
bs=4 decode replay #3 after MoE layer 20` → `[non-finite-logits] rid=c0ea223e… row=1 mode=decode`
on TP0 and TP1 (both ranks flag the same row; the abort is idempotent) → one
`POST /v1/chat/completions … 500` → the next decode log line shows `#running-req: 3` → the clean
group prefills 4 and decodes at bs=4. **PASS**: exactly `--expect-aborted 1` 500 with the expected
message, every other row ran to `max_tokens` and reads no worse than its bs=1 reference
(gibberish within `--gibberish-slack 0.05` of the reference's own score). Shared prefixes are
reported, not gated: distinct prompts at bs=4 already drift from bs=1 within a few tokens on this
stack (LEARNING §11 "Distinct-prompt bs=1-vs-bs=4 consistency is not a gate"), and here the batch
also shrinks to 3 rows mid-decode.

A first run of the same serve (same injection, same outcome: 1 × 500, 3 × 200, clean 4 × 200,
health 200) was scored by an earlier version of the script that gated on an absolute gibberish
bound and a 40-char shared prefix; the code prompt scored 0.22 against its *own byte-identical*
reference (identifiers and operators count as non-words) and the Romeo prompt drifted at token 1.
The gate was changed to the reference-relative form above and the serve repeated; the run-1 raw
output is not kept as evidence.

House gate on the same serve: `verify-batch-consistency.py --same-prompt --concurrency 4` **PASS**
(min pairwise row prefix 290 chars, gate ≥ 40; `consistency-same-prompt-c4.{txt,json}`).
`journalctl -k`: 0 Xid during the serve.

## (b) E22c: attributed to kt bug C, no reproducer since the T3.15 fix

E22c (2026-09-22) was a NaN assert in a 4-sequence, 664-real-token batch on the pristine kt-kernel
build. E22d could not reproduce it offline or in 20 same-shape batches. On 2026-09-23 bug C was found
and patched (`config/kt-kernel-patches/0001-…`, T3.9g): the bs=4 decode graph had a *temporary*
pinned buffer baked in, so rows 1–3 of every bs=4 step read stale bytes — the same defect that
corrupted every pre-patch C3/C4 number (LEARNING §11, E23 R3b erratum). T3.15 then served D=5 on the
patched image through capture, the first bs=4 steps, C1–C4 verify, NLL and a 61K-token bench with
no NaN, closing the E21 D≥5 sighting as bug C and attributing E22c to it. Since then every
qualification and soak has scanned its server log for `NaN|Xid|Traceback|OOM` and found none:
T3.19 G1-exit soak 141,312 decoded tokens (`evidence/t319/g1-exit/`), T3.8b default soak ≈ 2.6 M
decode tokens plus HumanEval, reasoning, stress and needle sets (`evidence/t38b/soak-default/`,
health hits 0), T3.31 idea-8 soak (`evidence/t331/64k4-lever-soak/`), the E33 ladder, T3.26/T3.29
262K / 512K / 1M serves. E22c therefore closes as *attributed, not bisected*: the pristine build is
no longer served, the patched build has not produced a NaN in ≈ 3 M decoded tokens, and the per-request
rejection above now bounds the blast radius of any future one to the request that hit it.

## Files

- `verify-nan-rejection.{txt,json}` — (a), stamped with the manifest hash.
- `consistency-same-prompt-c4.{txt,json}` — house batch-consistency gate on the same serve.
- `server.log`, `health-hits.txt` — full container log and the NaN / inject / 500 lines.
- `manifest.json` — the launch manifest (`evidence_manifest.py show`).

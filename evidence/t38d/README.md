# T3.8d step 1 — can online placement beat the static el8 map? (2026-10-23, GPU-free)

Question: T3.8c closed static maps at a 10-15 % CPU-route floor (per-set oracle 1-3 %); is the
runtime-placement prize real, i.e. does a mask trained on live decode traces reach the oracle on
held-out traffic? Method: the PR #2093 co-activation planner + EMA tracker
(`scripts/expert_placement.py`, vendored, Apache-2.0, ktransformers PR #2093 head) trained on the
first 60 % of the 767 single-token decode passes from today's 1017 recorder dumps
(`evidence/t1017/recorder/decode-distinct/`, logical ids = post-el8-remap), scored on the held-out
40 % and on the prefill-text set (`scripts/t38d-mask-eval.py`, run in the dsv41-kt container).

| mask | held-out decode CPU share | held-out prefill CPU share |
|---|---|---|
| el8 (static, today's default) | **9.5 %** | **15.0 %** |
| co-activation (decode-trained) | **2.1 %** | 28.8 % |
| cluster (decode-trained) | **1.8 %** | 21.6 % |
| EMA (decode-trained, decay 0.9) | **2.2 %** | 32.4 % |

**The online decode masks reach the per-set oracle (1-3 %) — a 7-12 percentage-point decode
prize over the static map, measured on real 1017 traffic.** Decode-trained masks lose on prefill
(decode and prefill hot sets differ; T3.8c saw the same), so a runtime placement must be
**phase-aware**: an online decode mask for decode steps, the el8 (or a prefill-fitted) map for
prefill passes - exactly what the T3.8 recorder already separates. This upgrades T3.8d from
"blocked, worth doing" to "blocked with a measured 7-12 pp go signal"; the staged unblock plan is
`plan/T38D-UNBLOCK-PLAN.md`.

## Stage 2 (2026-10-23) - the live mask source is in the overlay, log-only

Overlay branch `t38d-stage2` (worktree `/tmp/sglang-t38d2`, commits `256736db5`, `f274bd954`,
`3ce59c14e` on `deec7a6f`): `SGLANG_DSV41_RUNTIME_EXPERT_MASK=1` makes every recorder dump feed
the single-token decode passes into the EMA tracker and log the tracker mask's GPU share next to
the static el8 mask's; `SGLANG_DSV41_RUNTIME_EXPERT_MASK_DIR` optionally appends the mask JSONL.
Launcher knobs `RUNTIME_EXPERT_MASK` / `RUNTIME_EXPERT_MASK_DIR` (defaults off). The hook loads
the just-dumped .pt (file-mode dump returns None), retries on the torn-read race between the two
ranks, and is exception-guarded - a hook failure can never take the server down (verified: the
first unguarded version took the scheduler down; the guarded version logged the failure and the
serve stayed up).

Live readout on the 1017 default (recorder serve, two windows of ~997 single-token decode passes):

| window | traffic | static el8 GPU share | tracker GPU share |
|---|---|---|---|
| 1 | distinct fixtures | 0.892 | 0.944 |
| 2 | copy fixtures (traffic shift) | **0.791** | **0.981** |

**The static map degrades when traffic shifts (89.2 % -> 79.1 % GPU share, i.e. 10.8 % -> 20.9 %
CPU routes); the live EMA mask tracks the shift and holds 98.1 %.** This is the stage-3 weight
swap's measured justification, on top of the offline held-out gate (step 1: 2.1 % CPU share vs
the static 9.5 %). The mask never remaps anything yet - stage 3 is the swap.

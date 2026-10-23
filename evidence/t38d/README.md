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

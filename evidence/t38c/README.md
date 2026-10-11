# T3.8c — refit the expert map on our own recorded counts (2026-10-11, no GPU)

Question: the el8 mix map (`config/expert-map-el8-mix-219.json`, the launcher default since
T3.8b) was fitted on el8's published counts; we now hold 88,940 decode and 195,079 prefill tokens
of our own recorded routes (the T3.8b histogram serve and the T3.8b soak, both on the el8 map,
raw `.pt` dumps gitignored under `evidence/t38b/*/`). Does a map fitted on our own traffic send
fewer routes to the CPU tier than el8's on traffic it has not seen?

Method: `analysis/fit.py` (torch in the image, no GPU). The recorder on a remapped serve counts
physical slots, so counts are un-permuted to checkpoint ids with the served map (`kt_remap.py`'s
rule: GPU list ascending, then the rest ascending); the sanity rows reproduce the serve's live CPU
share to the digit (`fit-round0.txt`, e.g. HumanEval C4 decode 15.0 %, reasoning C1 9.6 %) and
the el8 mix map rebuilt from `counts-A..E.json` matches the served file on 100 % of slots. Fit =
top-219 per layer of 0.6 × token-weighted decode mix + 0.4 × token-weighted prefill mix (el8's
decode-heavy recipe). Scoring is **leave-one-traffic-family-out**: every set that shares the
held-out set's prompts (the same HumanEval problems at C1 / C4 / thinking on and off, the same
NLL corpus on two serves, the same distinct fixtures) leaves the fit together. A first pass that
left out one *set* at a time read 9.2 % held-out decode — a leak, since the sibling copy of the
same prompts stayed in the fit; that pass is not in the table.

## Result: no refit. Every own-count weight makes the held-out share worse.

Token-weighted held-out CPU share of routes (lower is better; 219 GPU experts per layer):

| map | decode | prefill |
|---|---|---|
| el8 mix (today's default) | **13.3 %** | **15.0 %** |
| own fit, leave-family-out | 16.4 % | 26.8 % |
| 0.5 own + 0.5 el8 | 13.6 % | 17.8 % |
| 0.1 own + 0.9 el8 | 13.4 % | 15.4 % |
| oracle (top-219 of the held-out set's own counts) | 2.4 % | 3.3 % |

The blend sweep is monotonic (alpha 0 → 1: decode 13.3 → 16.4, prefill 15.0 → 26.8), so there
is no weight at which our counts help. Per family: the own fit wins only on reasoning (C4 10.7 →
8.7 %, C1 9.6 → 8.1 %, fitted on code + text) and loses on everything else — HumanEval decode
15.0 → 20.0 %, the distinct benchmark fixtures 14.7 → 26.1 %, the NLL corpus prefill 15.0 →
29.5 %. In-sample the same map reads 6 % on HumanEval and 9 % on the corpus: a 31K-token coding
set fits its own prompts and not the next coding set. Checked the other way, the own-fit map on
el8's prose-decode counts reads 29 % against the el8 map's 10 %.

Why el8's map generalises and ours does not: the oracle is 1–3 % on every set but the families'
hot sets differ (a map fitted on four families reads 20–30 % on the fifth), so the routing
concentration the T3.37 soak saw (decode spread at equal C follows the prompt set) is real and
content-specific; a static map can only capture the part of the hot set that is shared across
content, and el8's five workloads over a larger corpus estimate that shared part better than our
88K decode tokens from three prompt families. More tokens of the *same* families would not
change this (HumanEval full is 4× the T3.8b histogram set and reads the same 15 %); more
independent families might, and the bar for a retry is in `plan/TASKS.md` (T3.8c).

Bar from the T3.8c row: serve the refit only if its held-out share beats el8's 10–15 % by more
than the run-to-run spread. It does not; no serve was spent. The el8 mix map stays the default.

Files: `analysis/fit.py`, `analysis/fit-round0.txt` (full output, every set and both passes),
`analysis/maps.json` (held-out table and the fit-on-all GPU lists, not served).

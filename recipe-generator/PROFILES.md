# PROFILES.md — per-use-case presets generated from the tuning model

A profile is a named **delta over the tuned baseline** (the `qualified` shape the placement calculator emits),
layered the way the parent repo layers them: `registry default < profile < environment`. Each delta names the
levers it moves and cites the measured row in [LEVERS.md](LEVERS.md) that justifies it. Machine-readable form:
[`profiles/`](profiles/), schema-compatible with `recipe-factory/UI/JSON-SCHEMA.md`'s profile-diff idea.

Seeded from `recipe-factory/KNOBS-AND-PROFILES.md` §3 (the reference build's presets); a new recipe regenerates
each preset from its own measured numbers — the rule that keeps presets honest.

## `qualified` (the baseline, not a delta)

What the calculator emits for the reference host: 219/165 experts via the el8 map, D=4, `KT_CPUINFER=16`,
chunk 8192, fraction 0.96, 262K×4, capture bs 1–4, Engram `ahead` + chunk-ahead, expandable-after-capture,
NaN rejection, clamp, deterministic-greedy levers, `HOST_THREAD_CPUS=auto`, reasoning effort high, spec off.
Measured: C1 60.8/58.8 wall, C4 124.0/118.9, prefill 4,344/4,495 (16K/32K cold), needles 6/6, stress 278 s.

## `coding-agent` — one user, one agent, long sessions; TTFT matters, batch throughput does not

| Lever | Delta | Why (measured row) |
|---|---|---|
| KV slots | `MAX_RUNNING_REQUESTS=1–2` | single stream; fewer slots = less scheduler interference |
| Chunk | 8192 (keep) | chunk is what buys long-session first-turn TTFT; 16384 costs −4.4 % C1 (E34d-iv) |
| Engram admission prefetch | on | −0.2…−0.4 s off every cold first-request TTFT (T3.49) |
| Kernel cache | persistent dir | kills the +14–19 s first-request Triton compile (T3.47) |
| Reasoning effort | high (owner call) | pure prompt-prefix quality/latency knob (T3.54) |

## `multi-user` — several concurrent sessions; aggregate tok/s is the metric

| Lever | Delta | Why |
|---|---|---|
| KV slots | 4 (qualified max at 262K) | C4 is the metric |
| Host threads | `auto` (non-negotiable) | the C4 106/123 bimodality is a shape defect (T3.52) |
| Context | 64K×4 if sessions are short | trades context for headroom (C4 121.0/116.2) |
| Graph capture | "1 2 3 4" | bs=3 capture is +28 % at C3 (E26) |
| Speculation | off | every spec path measured loses at C4 (−63…−77 %) |

## `long-context-ladder` — single giant context

| Rung | Delta | Measured |
|---|---|---|
| 262K×1 | context 262144, slots 1 | needles 6/6 at 255,388 tok, TTFT 59 s |
| 512K×1 | + fraction 0.962 | needles 6/6 at 511,145 tok, TTFT 142 s |
| 1M×1 | context 1048576, fraction 0.962 | needles 6/6 at 1,022,456 tok, TTFT 448 s |

Guards encode all three: >262K with any other fraction is refused; >1,048,576 refused outright (never run).

## `power-capped`

Decode is cap-insensitive on this model (memory-bound step); prefill pins to the cap; GLM-5.3 was the opposite
(+9 % C16 at 325 W) — **the sweep protocol transfers, the number does not**. Preset: keep 300–325 W for
decode-heavy traffic; raise only for prefill-heavy traffic; lower only for heat/cost — measure the prefill
curve first (power-sweep bench pattern, `docs/BENCHMARKS.md`).

## `quality-first`

| Lever | Delta | Why |
|---|---|---|
| Reasoning effort | max (100) | benchmark-grade reasoning budget |
| Deferral | 4, never 5 | D=5 fails NLL — the gate *is* the quality floor |
| GPU experts | never 230 | 230 fails the NLL gate (E29) |
| Determinism levers | on | bit-reproducible greedy is itself a quality property |
| HumanEval spot-check | per checkpoint | 161/164 baseline, 158/164 on 1017 — re-run per model update |

## Regeneration rule

`profiles/*.json` are deltas over the lever dictionary; CI (when wired in) fails if a profile names a lever id
that is not in `levers.json`, or cites a measured row that is not in the lever's `measured` list. The generator
emits the profile files from the tuning model; they are never hand-written.

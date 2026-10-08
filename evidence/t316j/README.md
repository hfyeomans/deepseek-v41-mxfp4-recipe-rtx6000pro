# T3.16j: B2 direct-UVA residue removed (ledger G2, 2026-10-08)

Gate G2 of the migration roadmap, triggered by the T3.19c commit (`095789db`). One overlay commit on
`095789db`: 5 files changed, +14/-314 (`kt_ep_wrapper.py` 1,357 -> 1,074 lines). Recipe side: 4 files,
+17/-33. Behaviour-preserving by construction: every removed line was behind `KT_B2_UVA_EXPERTS > 0`,
which no profile has set since E30 rejected B2 (2026-09-24). The gate below proves the default path
did not move.

## What was removed

Overlay (`python/sglang/srt/layers/moe/`):

| File | Removed |
|---|---|
| `kt_ep_wrapper.py` | `KT_B2_UVA_EXPERTS` / `KT_B2_MAX_TOKENS` knob parsing (replaced by a 4-line comment pointing at `evidence/t318-b2-prototype/`); `remap_topk_ids_for_b2`; `_uva_view`; `_B2HostPool` (class state `_buf/_off`, the last class global); `_B2UvaExperts` (`run` / `fork` / `join`); `self.b2` and `self.num_kt_experts`; the B2 block in `process_weights_after_loading`; the `num_kt_experts` parameter of `submit()` (now `submit(layer, dispatch_output)` using `self.num_cpu_experts`); the `use_b2` branch and the `b2_out` join in `apply` |
| `kt_contract.py` | `check_moe_topology(..., b2_experts, has_streamer)` parameters and the two B2 checks (B2 needs a streamer; GPU + B2 + CPU tiers must sum to the routed count) |
| `kt_runtime.py` | `_side_streams` and `side_stream(device)` (the T3.19c landing spot for the B2 stream; nothing else used it) |
| `kt_timeline.py`, `kt_provider.py` | docstring mentions of the B2 join |

Recipe:

| File | Change |
|---|---|
| `scripts/serve-hybrid.sh` | knob defaults, comments, `-e` passthroughs and `resolved:` fields gone; `fail_config` now **refuses** a launch if either knob is set in the environment (verified: `KT_B2_UVA_EXPERTS=40 bash scripts/serve-hybrid.sh` -> `FAIL config ... removed at T3.16j`, rc 2, no container) |
| `scripts/tests/test_kt_contract.py` | `topo()` fixture lost the two parameters; `test_b2_needs_streamer_and_fits_cpu_tier` replaced by `test_b2_knobs_are_gone` (asserts the parameters are absent from the signature) |
| `scripts/tests/test_kt_runtime.py` | side-stream isolation assertion replaced by `assert not hasattr(runtime, "_side_streams")` |
| `scripts/bench-b2-uva-moe.py` | kept as the E30 measurement record; header comment says the serving-path code is gone |

Kept: `remap_topk_ids_for_cpu(topk_ids, num_gpu_experts, num_cpu_experts)` with its 3-argument
signature (the parity replay harness calls it that way), the `numpy` import (`_HostSlice` uses it).

## Decisions taken without the owner, open to veto

1. The stream/UVA primitives (`KtRuntime.side_stream`, `_uva_view`) went too. G2's rule: they stay
   only if a T3.20 design note names them, and none exists. E30 (`evidence/t318-b2-prototype/`) is
   the record of what worked; re-adding ~40 lines when T3.20 wants a side stream is cheaper than
   carrying untested code through T3.19d-f.
2. `check_moe_topology` lost `has_streamer` as well as `b2_experts`: it only existed to say "B2
   needs a streamer". The streamer itself is unchanged.
3. The launcher refuses the old knobs instead of ignoring them, so a stale profile or shell
   environment cannot silently run a different configuration from the one it names.
4. `scripts/bench-b2-uva-moe.py` stays (it imports only FlashInfer and torch, not the overlay).

## Tests (CPU-only, in `ci.sh`, rc = 0)

Buffer lifetime 8/8, contract 14/14 (one test swapped), deferral adjacency 9/9, provider 17/17,
runtime 5/5, split seal 7/7, DSpark gate 7/7, timeline, parity replay check. The five kt files
compile inside the image.

## Gate serve (`KT_NAN_PROBE_MIN_TOKENS=1`, `KT_TIMELINE` unset)

`resolved: experts gpu=219 cpu=165 deferral D=4 cpuinfer=16 chunk=8192 mem_fraction=0.96
context=65536 slots=4 max_total_tokens=278528 capture_bs=[1 2 3 4] dspark=0` (`launch.txt`).
GPU power cap 300 W, BIOS slot 3 (FCLK 2100). 40 layers loaded 165 CPU experts each; capture-end
avail mem 5.56 / 5.50 GB (T3.19c and T3.28: 5.56 / 5.50). `docker logs | rg -c "KT B2"` = 0.
Health: 0 NaN / Xid / Traceback / KTContract lines; EDAC CE/UE 0 / 0.

| check | T3.28 comparator | T3.19c | T3.16j measured | gate | result |
|---|---:|---:|---:|---|---|
| consistency `--same-prompt` C1..C4 (`consistency-c*.json`) | PASS | PASS | **PASS**, min pairwise prefix 310 / 310 / 290 / 290 | PASS bs 1..4 | pass |
| C1 median / wall tok/s, TTFT (`c1-off6.json`) | 41.3 / 39.4 / 0.63 s | 41.56 / 39.58 / 0.634 s | **41.18 / 39.21 / 0.631 s** | wall >= 39.0 | pass (low end of band, -0.5 % vs T3.28) |
| C4 aggregate / wall, per stream (`c4-off0.json`) | 56.3 / 54.2 / 14.1 | 56.05 / 53.85 / 14.0 | **56.13 / 53.93 / 14.0** | >= 13.6 / stream | pass |
| NLL vs `evidence/t328/default-219-8192/corpus/` (`pooled.json`, 8 texts) | | 0 moved | **+0.00000, 0 of 65,653 tokens moved** | 0 moved | pass |

The three C1 wall readings since T3.28 (39.4, 39.58, 39.21) span 0.9 %, inside the run-to-run band
seen at E22/T3.28; C4 is within 0.1 tok/s of T3.19c. Nothing on the decode path changed, and the
NLL and consistency results say so bit-for-bit.

## Files

`launch.txt` (launcher), `launch-gate-container.txt` (container log), `consistency-c{1..4}.json`,
`c1-off6.json`, `c4-off0.json`, `0?-*.json` + `pooled.json` (NLL per text and pooled).

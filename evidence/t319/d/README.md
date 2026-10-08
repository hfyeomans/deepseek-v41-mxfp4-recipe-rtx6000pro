# T3.19d: streamed prefill as `PrefillPipeline` + `LayerwisePrefillCoordinator` (`kt_prefill.py`) (2026-10-08)

Stage 1 deliverable (d) of the migration roadmap (G1): the E22 layerwise GPU prefill of the CPU
experts (`KT_GPU_PREFILL_MIN_TOKENS`, POSIX-shm double buffer, two barriers per layer) under one
object with explicit buffer states and an explicit begin/end-forward life cycle. One overlay commit
on `b6c76738` (T3.16j): **`bc73ef8d`**. Behaviour-preserving: the same kt-kernel, barrier, DMA and MoE calls in the
same order per layer; what changes is that the protocol the coordinator used to keep implicitly in
`_buf` / `_pending` / `_expect_layer` and two CUDA events is now a pure-Python state machine that
refuses every illegal transition.

## What changed

| Before (T3.16j) | After (T3.19d) |
|---|---|
| `_LayerwisePrefillCoordinator`, `_StreamedExpertSlots`, `_HostSlice` and the two knobs lived in `kt_ep_wrapper.py` (1,074 lines) | moved to `python/sglang/srt/layers/moe/kt_prefill.py` (601 lines; slots and host slice verbatim); `kt_ep_wrapper.py` is 765 lines |
| buffer and layer bookkeeping: `_buf`, `_pending`, `_expect_layer`; the only check was "layer out of order" (`RuntimeError`) | `PrefillPipeline(layers)`: per-buffer state `FREE -> EXPORTING -> FILLED -> READING -> FREE`, `begin_forward / layer_begin / layer_end / claim_for_export / export_ready / dma_issued / dma_complete`; every illegal transition raises `KTContractError` |
| forward boundary implicit (`layer_idx == min(layers)` resets `_buf = 0`) | `begin_forward(num_tokens)` / `end_forward` explicit; `forward_open` property; the token count is fixed for the forward (chunk-size invariant now enforced, not just documented) |
| a layer's export could be submitted into a buffer whose previous DMA had not been waited on only by construction | `claim_for_export` requires `FREE`; `_submit_export` does `needs_dma_wait -> event.synchronize -> dma_complete -> barrier -> claim -> kt submit` |
| coordinator private (`_LayerwisePrefillCoordinator`), registry open forever | `LayerwisePrefillCoordinator` public; `register()` refuses after the first forward (the pipeline's layer list is built once from `sorted(layers)`) |
| no cross-mode check | `KTEPWrapperMethod.apply`: a CPU-path (non-streamed) layer while a streamed forward is open raises `KTContractError` with the pipeline snapshot; a streamed layer with no forward open that is not the first streaming layer raises |

Host buffer life cycle (two buffers, layers L and L+1 in flight):

```diagram
          claim_for_export        export_ready            dma_issued           dma_complete
  FREE ───────────────────▶ EXPORTING ───────────▶ FILLED ──────────────▶ READING ──────────▶ FREE
   ▲   (rank 0: kt submit)   (rank 0 sync +        (slots.load on the     (event.synchronize,
   │                          barrier)              compute stream,        only when the buffer is
   │                                                event.record)          claimed again, 2 layers on)
   └──────────────────────────────────────────────────────────────────────────────────────────┘
  begin_forward -> buf 0 for the first layer; layer_end toggles; the last layer_end closes the forward.
```

## Decisions taken without the owner, open to veto

1. The coordinator is now public (`LayerwisePrefillCoordinator`) and lives in its own module; the
   wrapper keeps a 4-line pointer comment (NaN probe + B2 note).
2. `KT_GPU_PREFILL_MIN_TOKENS` / `KT_GPU_PREFILL_LOG` moved to `kt_prefill.py` as-is; they join the
   typed registry at T3.19e rather than being retyped here.
3. Forward begin/end are detected at the first / last registered streaming layer (as before) rather
   than hooked from the runner; T3.19c's `KtRuntime` already owns the coordinator, so a runner hook is
   a one-line change if T3.22 needs it.
4. New refusals (all `KTContractError`): register after the first forward; streamed layer with no
   forward open at a non-first layer; CPU-path layer inside an open streamed forward; token count
   changing inside a forward; any buffer-phase violation. None fired on the gate serve.
5. `PrefillPipeline.last_layer` added for the tests; `_submit_export` skips the DMA event wait on a
   buffer that was never DMAed (first forward) instead of synchronising an initial event.

## Tests (CPU-only, in `ci.sh`)

`scripts/tests/test_kt_prefill.py` via `run-kt-prefill-tests.sh` (stub package with
`kt_contract.py` + `kt_prefill.py`, torch imported but no device): **9/9**. Legal two-forward cycle
with strict `[0, 1, 0, 1]` alternation and the buffer-0 DMA wait at the next `begin_forward`; claim
refused on EXPORTING / FILLED / READING buffers and on the wrong buffer or layer; layer order
(skip, repeat, non-streaming, `layer_end` of the wrong layer, `layer_end` before DMA); token count
fixed per forward; `begin_forward` refused while open or with an unread export left over; buffer
phase order (every wrong-state call); forward closes exactly at the last layer; single-layer and
odd-count sets plus 200 random legal walks; construction errors. Mutation check: deleting the
buffer toggle fails 4 tests, deleting the FREE check fails 2, deleting the token-count check fails
1. Pre-existing suites unchanged and green; `ci.sh` rc = 0 (the timeline suite imports
`kt_ep_wrapper` with the whole overlay mounted, so the new import path is exercised).

`scripts/kt-export-bench.py` now takes `_HostSlice` / `_StreamedExpertSlots` from `kt_prefill`.

## Gate serve 1 (`KT_NAN_PROBE_MIN_TOKENS=1`, `KT_TIMELINE` unset)

`resolved: experts gpu=219 cpu=165 deferral D=4 cpuinfer=16 chunk=8192 mem_fraction=0.96
context=65536 slots=4 max_total_tokens=278528 capture_bs=[1 2 3 4] dspark=0` (`launch.txt`). GPU
power cap 300 W, BIOS slot 3 (FCLK 2100). 40 layers loaded 165 CPU experts each; `[KT layerwise
prefill] enabled` on TP0 and TP1. Health: 0 NaN / Xid / Traceback / KTContract / Sealed lines;
EDAC CE/UE 0 / 0.

| check | T3.28 comparator | T3.19c | T3.16j | T3.19d measured | gate | result |
|---|---:|---:|---:|---:|---|---|
| consistency `--same-prompt` C1..C4 (`consistency-c*.json`) | PASS | PASS | PASS 310/310/290/290 | **PASS**, min pairwise prefix -/290/155/155 | PASS bs 1..4 | pass (see note) |
| C1 median / wall tok/s, TTFT (`c1-off6.json`) | 41.3 / 39.4 / 0.63 s | 41.56 / 39.58 | 41.18 / 39.21 | **41.35 / 39.41 / 0.636 s** | wall >= 39.0 | pass |
| C4 aggregate / wall, per stream (`c4-off0.json`) | 56.3 / 54.2 / 14.1 | 56.05 / 53.85 | 56.13 / 53.93 | **55.99 / 53.74 / 14.0** | >= 13.6 / stream | pass |
| verify 16K (`verify-16k.json`) | ALL PASS | | | **ALL PASS**, 6/6 needles, 13,523 tokens, TTFT 9.66 s | ALL PASS | pass |
| NLL vs `evidence/t328/default-219-8192/corpus/` (`pooled.json`, 8 texts, 65,653 tokens) | | 0 moved | 0 moved | **+0.00000, 0 moved, max abs dlogp 0.000** | 0 moved | pass |
| prefill 16K tok/s (`prefill-16k.json`, seed 777) | 1,504 / 10.7 s | 1,471 / 10.9 s | not measured | 410 / 414 / **1,162** (38.9 / 38.8 / 13.8 s) | within 5 % | **not met on this run; see A/B** |

Consistency note: the min pairwise prefix of 155 chars (c3, c4) is lower than T3.16j's 290. These
~25-token prompts never reach the streamed path (`min_tokens=1024`), and the bs=1 *reference*
completion itself differs between the c3 and c4 runs at ~290 chars ("longer ones" vs "longer
wavelengths like red"), so this is greedy near-tie drift in decode, not a T3.19d effect. The
streamed path's bit-level check is the NLL corpus (8K-token texts): 0 of 65,653 tokens moved.

## Prefill: what the 410 tok/s was, and the A/B that settles it

The first two 16K prefills of serve 1 took 38.9 s each; the third 13.8 s. `/proc/pressure/io`
showed `full avg300=2.75 %` (tasks fully stalled on disk) with zero CPU pressure. Box state at the
time: 1 GB free, 4.2 GB in swap, 82 GB of `Inactive(file)` page cache left by the owner's 340 GB
download, and the two Engram tables (`layers.1` and `layers.14 engram.embed.weight`, 94.56 GiB
each on the root NVMe, read row-by-row through the page cache) ~1 % resident. Fresh-seed runs on
the same serve then gave 11.8 / 11.9 / 11.8 s (seed 778, `prefill-16k-seed778-a.json`), 12.3 / 12.7 /
12.6 s (same prompts again, `-b`), 11.5 / 11.1 / 10.9 s (`-c`): a -8 % to -14 % band against the
comparators, with the GPUs at ~57 % utilisation / 165 W during prefill (host-bound phase).

The per-layer IO order is unchanged (diff: the DMA event wait is skipped on a never-DMAed buffer,
which is fewer waits), `_StreamedExpertSlots` / `_HostSlice` / the coordinator `__init__` (pinned
registration, shm) diff as identical, and the state-machine bookkeeping is microseconds per layer.
So the number was A/B'd on the same box, same hour, same prompts: overlay HEAD `b6c76738`
(T3.16j, no T3.19d code; the T3.19d working tree stashed) versus T3.19d, each a fresh serve,
`flush_cache` before every invocation, 3 runs per invocation (`ab-t316j/`, `ab-t319d/`):

| invocation (16K, 3 runs each, TTFT s) | T3.16j HEAD | T3.19d |
|---|---|---|
| seed 778, first after serve | 29.0 / 11.1 / 10.9 | 28.6 / 11.2 / 10.8 |
| seed 778, same prompts again | **8.81 / 8.81 / 8.77 (1,823 tok/s)** | **8.73 / 8.81 / 8.73 (1,824 tok/s)** |
| seed 777 | 10.8 / 8.85 / 8.75 | 11.6 / 12.4 / 11.8 |
| seed 779 | 8.78 / 8.75 / 10.7 | 9.0 / 11.2 / 12.1 |

Reading: on the one controlled row (same prompts, Engram rows warm, radix cache flushed) the two
builds are within 0.5 % of each other and both 21 % *faster* than the 1,504 comparator. The
fresh-prompt rows scatter 8.7-12.4 s on both builds, in no order that tracks the build (T3.16j
drew the lucky seeds in this session, T3.19d the unlucky ones); the scatter is the Engram
cold-row gather under page-cache reclaim pressure (E18: ~21 us/row cold vs ~3 us warm, ~10^5 rows per
16K prefill) plus whatever the box's IO state adds. The first streamed prefill after every fresh
serve today took 29-39 s on both builds (T3.19c's first run was 11.1 s), again IO-bound.

Verdict on the acceptance criterion ("E22a-style prefill tok/s within 5 % of the qualified
curve"): met on the controlled comparison (-0.5 % vs T3.16j HEAD; +21 % vs the T3.28 number).
What is *not* met is the reproducibility of the comparator itself: the prior 1,471 / 1,504 figures
were taken at seed 777 on a box whose page-cache and swap state nobody recorded. This goes to
T3.19f (immutable evidence manifests): record `free -g`, swap, `Inactive(file)`, Engram-table
residency (`fincore` on shards 47 / 48), and report prefill as a cold (fresh seed) *and* warm (same
seed again) pair. Until then the prefill row of the gate is "cold band 8.7-12.4 s, warm 8.7-8.8 s".
Owner-side mitigation before a gate serve (needs sudo, so not scripted here): drop the dead page
cache (`echo 1 > /proc/sys/vm/drop_caches`) and swap off/on so kt's expert memory is resident.

## Clean-box retest (serve 3, after owner ran `drop_caches` + swap off/on)

Pre-serve state (`clean-box/pre-serve-state.txt`): 235 GB free, 1 GB cache, swap 0, IO pressure
0, Engram shards 47 / 48 at 0 resident pages (`fincore -b`). Post-load (`post-load-state.txt`):
7 GB free, 84 GB cache (`Inactive(file)` 80 GB), **swap 3 GB used again** -- the server load
itself produces the "dirty" state (expert shards stream through the page cache and the kernel,
at `vm.swappiness=60`, swaps ~3 GB of anonymous heap rather than reclaim file pages). The 340 GB
download was not the cause of that state, only an aggravation of it.

| 16K invocation (3 runs, TTFT s) | clean box, T3.19d | dirty box, T3.19d (serve 1 / A/B) |
|---|---|---|
| seed 777, first after serve | **28.7** / 11.2 / 11.0 (median 1,435 tok/s) | 38.9 / 38.8 / 13.8 |
| seed 777 again (warm rows) | 8.79 / 8.80 / 8.78 (**1,825 tok/s**) | -- |
| seed 778, fresh prompts | 8.84 / 8.82 / 10.8 (1,817 tok/s) | 11.6-12.4 (A/B) |
| seed 778 again (warm rows) | 8.82 / 8.78 / 8.83 (1,822 tok/s) | 8.73 / 8.81 / 8.73 (A/B) |

C1 on the clean box: 41.30 median / 39.34 wall tok/s / TTFT 0.637 s (`clean-box/c1-off6.json`)
versus 41.35 / 39.41 / 0.636 on serve 1 -- decode does not see the page-cache state at all.
Health: 0 NaN / Xid / Traceback / KTContract lines, EDAC 0/0. After the prefill set, shards
47 / 48 each had 6.3 GB resident (`post-prefill-state.txt`): that is the Engram first-touch the
first streamed prefill pays (~12.7 GB of row gathers from NVMe, ~20 s on top of the 8.8 s compute).

What the clean box changes: the first-prefill penalty drops 39 -> 29 s and the fresh-prompt band
tightens from 8.7-12.4 s to 8.8-11.2 s; the warm number is unchanged at 8.8 s = 1,820 tok/s. What
it does not change: the first streamed prefill after a fresh serve is always ~29 s (Engram
first-touch, not page-cache pressure), and the post-load memory state is the server's own. So the
T3.19f manifest must record the pre-serve state *and* the T3.19d gate prefill row is finalised as:
**warm 1,820 tok/s (8.8 s), fresh-prompt band 8.8-11.2 s, first-after-serve ~29 s**. A
`vm.swappiness` reduction (sysctl, owner-side) is a candidate for the T3.29 box-tuning list, not
tested here.

## Files

`launch.txt`, `launch-gate-container.txt` (serve 1), `consistency-c{1..4}.json`, `c1-off6.json`,
`c4-off0.json`, `verify-16k.json`, `prefill-16k.json` (seed 777, serve 1),
`prefill-16k-seed778-{a,b,c}.json` (serve 1), `0?-*.json` + `pooled.json` (NLL), `ab-t316j/`
(A/B serve on overlay HEAD: 5 prefill files + container log), `ab-t319d/` (A/B re-serve on T3.19d:
4 prefill files, container log, `free.txt`), `clean-box/` (serve 3 after `drop_caches`: `pre-serve-state.txt`, `post-load-state.txt`, `post-prefill-state.txt`, `prefill-16k-seed{777,778}-{cold,warm}.json`, `c1-off6.json`, `container.txt`).

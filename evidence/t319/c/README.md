# T3.19c: runner-owned kt state (`KtRuntime`), explicit compact loader, no module registry (2026-10-08)

Stage 1 deliverable (c) of the migration roadmap (G1). One overlay commit on `d6811728`
(7 files changed, +57/-181, plus `kt_runtime.py` 199 lines and `kt_loader.py` 67 lines new).
Behaviour-preserving: the same kt-kernel calls in the same order; what moves is *who owns* the
provider registry, the leases, the forward ledger, the prefill coordinator, the timeline and the
loader. All of it now hangs off `ModelRunner.kt_runtime`, one object per runner / TP rank.

## What changed

| Before (T3.19b) | After (T3.19c) |
|---|---|
| `kt_provider.py` module globals `_PROVIDERS / _LEASES / _LEDGER`, `providers()`, `ledger()`, `reset_registry()`, `reserve_capture_sizes()` | `KtRuntime.providers / leases / ledger`, `KtRuntime.new_provider()`, `KtRuntime.reserve_capture_sizes()`; `KtKernelProvider` takes `ledger=` and `leases=` as required arguments |
| `_LayerwisePrefillCoordinator._instance` class singleton | `KtRuntime.prefill_coordinator` slot, reached through `KTConfig.runtime` (`for_method(method)`) |
| `kt_timeline._TIMELINE` global + `get_timeline()` | `KtRuntime.timeline` (built when `KT_TIMELINE` is set); the graph backend reads it from `model_runner.kt_runtime` |
| loader monkey-patch: `_install_kt_mxfp4_loader_compact()` rewrote `MXFP4SafeTensorLoader.load_experts` process-wide and passed the GPU mask through the module global `_KT_LOADER_GPU_MASK` | `kt_loader.CompactMXFP4Loader(weight_path, gpu_experts_mask)` subclass; `KtRuntime.load_cpu_experts(wrapper, cpu_ids, mask)` installs it in kt's `NativeMoEWrapper._native_loader_instance` slot for that one `load_weights` call and asserts kt used it |
| `_B2_STREAMS` module dict + `_b2_stream()` | `KtRuntime.side_stream(device)` (B2 is rejected; this and `_B2HostPool._buf/_off` are the only class/process state left and both go at T3.16j) |
| `KTMoEWrapper.set_capture_batch_sizes` called from two graph runners | `model_runner.kt_runtime.reserve_capture_sizes(sizes)` from both, skipped when the runner has no kt runtime |

The one process-level handle that remains is scoped, not global: `model_runner.initialize()` runs
`load_model()` inside `with kt_runtime.activate(self.kt_runtime):`, and
`create_kt_config_from_server_args` reads `kt_runtime.current()` (raises if none is active, nested
activation raises). SGLang's quantization-config factory is called from deep inside model loading with
no runner in scope, so this is the smallest hand-off that avoids threading a runner through every
loader signature; it is empty outside `load_model`.

## kt-kernel protocol detail found at the first gate serve

The first serve died at layer 0 with `KTContractError: kt still holds a loader
(MXFP4SafeTensorLoader)`. kt's `NativeMoEWrapper.__init__` (amx.py L702-704) fills the class slot with
a plain loader when the *first* wrapper is constructed, long before any `load_weights`; `load_weights`
reads the slot (refilling it if empty) and `_release_loader()` clears it after every layer. So the slot
is occupied only before layer 0, by kt's own pre-created plain loader. `load_cpu_experts` now releases
a plain `MXFP4SafeTensorLoader` found there and refuses only a leftover `CompactMXFP4Loader` or a
foreign object (which would mean kt stopped releasing per layer). `test_kt_runtime.py` covers both
cases. The CPU stub in the first version of the test created the wrapper with an empty slot and so
missed this; the test now mirrors kt's constructor.

## Decisions taken without the owner, open to veto

1. Scoped `activate()` / `current()` handle during `load_model` instead of plumbing the runner into
   SGLang's quant-config factory signature.
2. The compact loader is injected into kt's own loader slot per layer (the hook kt exposes) with a
   post-condition that `wrapper.loader is` our instance and `loaded_cpu_experts == len(cpu_ids)`.
3. `_B2HostPool` class state and `KtRuntime.side_stream` were left in place for T3.16j rather than
   half-removed here.
4. `CompactMXFP4Loader` prints the same `[MXFP4SafeTensorLoader] Loaded N CPU experts ...` line as
   the old monkey-patch so `rg -c "CPU experts from"` = 40 remains the load check.

## Tests (CPU-only, in `ci.sh`)

- New `scripts/tests/test_kt_runtime.py` via `run-kt-runtime-tests.sh` (overlay `python/sglang`
  mounted over the installed package): **5/5**. Two runtimes share no providers / leases / ledger /
  coordinator / timeline; `reserve_capture_sizes` contracts are per runtime; `activate` scope
  (nested and empty cases raise `KTContractError`); the compact loader loads exactly the CPU
  complement in order with correct ue8m0 -> bf16 scales against a 5-expert safetensors fixture
  (3 GPU-masked); `load_cpu_experts` protocol (kt's constructor-made plain loader replaced and its
  handles cleared, stale compact / foreign loader refused, wrapper that ignores the loader refused).
- `scripts/tests/test_kt_provider.py` adapted to build providers from a `KtRuntime`: **17/17**,
  same assertions as T3.19b (kt call sequence unchanged, lease identity, forward-end check).
- Pre-existing suites unchanged and green: timeline 59/0, buffer lifetime 8/8, contract 14/14,
  deferral adjacency 9/9, split seal 7/7, DSpark gate 7/7, parity replay check. `ci.sh` rc = 0.

## Per-step gate on the plain serve (`KT_TIMELINE` unset, `KT_NAN_PROBE_MIN_TOKENS=1`)

Current default profile (T3.28): `resolved: experts gpu=219 cpu=165 deferral D=4 cpuinfer=16
chunk=8192 mem_fraction=0.96 context=65536 slots=4 max_total_tokens=278528 capture_bs=[1 2 3 4]
dspark=0` (`launch.txt`). GPU power cap 300 W, BIOS slot 3 (FCLK 2100). All 40 layers logged
`Loaded 165 CPU experts ... as compact ids 0..164 (219 GPU experts not loaded)`; capture end avail
mem 5.56 / 5.50 GB (T3.28: 5.56 / 5.50).

The comparators are the T3.28 numbers, which supersede `baseline-pin.json` (pinned at 226/158,
chunk 4096, 325 W): C1 41.3 median / 39.4 wall, C4 56.3 / 54.2, prefill 16K 1,504 tok/s.

| check | T3.28 comparator | T3.19c measured | gate | result |
|---|---:|---:|---|---|
| C1 median / wall tok/s, TTFT (`c1-off6.json`) | 41.3 / 39.4 / 0.63 s | **41.56 / 39.58 / 0.634 s** | wall >= 39.0, TTFT <= 0.65 | pass |
| C4 aggregate / wall, per stream (`c4-off0.json`) | 56.3 / 54.2 / 14.1 | **56.05 / 53.85 / 14.0** | >= 13.6 / stream | pass |
| prefill 16K tok/s, TTFT (`prefill-16k.json`) | 1,504 / 10.7 s | **1,471 / 10.9 s** | informational | inside band (-2 %) |
| NLL vs `evidence/t328/default-219-8192/corpus/` (`pooled.json`, 8 texts) | | **+0.00000, 0 of 65,653 tokens moved, max abs dlogp 0.000** | 0 moved | pass |
| consistency `--same-prompt` C4 (`consistency.json`) | | PASS, min pairwise prefix 310 chars | PASS (>= 40) | pass |
| NaN / Xid / Traceback / `KTContract` / `Sealed` lines (`launch-gate-container.txt`) | | 0 | 0 | pass |
| EDAC ce / ue | | 0 / 0 | 0 / 0 | pass |

## Reading

1. No module or class singleton remains on the kt path except the two B2 leftovers scheduled for
   T3.16j; a second `ModelRunner` in the same process (the T3.22 multi-rank prototype's starting
   point) gets its own providers, leases, ledger, coordinator, timeline and loader by construction,
   and the CPU test proves it.
2. The loader is now an explicit object with an explicit mask instead of a process-wide
   monkey-patch; the gate serve shows kt loads exactly the 165-expert complement per layer and the
   NLL corpus is bit-identical, so the compact id mapping is unchanged.
3. Performance is inside the run-to-run band on every row, as expected for a pure ownership move.
4. G1 remaining: T3.16j (B2 residue removal, now safe), T3.19d (unified prefill coordinator),
   T3.19e (typed config registry), T3.19f (immutable evidence manifests).

Files: `c1-off6.json`, `c4-off0.json`, `prefill-16k.json`, `consistency.json`, `pooled.json` and
the eight per-text NLL files (`01-..08-*.json`), `launch.txt` (redacted launcher log),
`launch-gate-container.txt` (redacted container log of the gate serve).

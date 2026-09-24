# Maintainability review: DeepSeek-V4.1-Flash FP4 hybrid

**Measured:** Review date: 2026-09-23  
**Measured:** Repositories: recipe repository and `~/dev/src/sglang-dsv41-overlay`  
**Measured:** Overlay range: upstream base `da64c5cb` through `07f4aef0`

Claim labels used throughout:

- **Measured**: directly observed in source, Git history, checked-in experiment records, or supplied benchmark results.
- **Inferred**: a consequence of those observations that was not directly measured.
- **Proposed**: a recommendation or interface that does not exist yet.

## 1. Scope, method, and exclusions

### Scope

- **Measured:** This review covers the qualified two-GPU plus CPU path launched by `scripts/serve-hybrid.sh`: TP2, 208 GPU and 176 CPU experts, kt-kernel, breakable decode graphs, D=4, graph batch ceiling 4, streamed prefill threshold 1024, and speculation off (`scripts/serve-hybrid.sh:38-70`, `scripts/serve-hybrid.sh:84-110`).
- **Measured:** It covers the overlay commits after `da64c5cb`, the hot SGLang orchestration paths named below, kt-kernel 0.7.0.post4's installed and pristine `experts_base.py`, the staged E23 documentation/patch, and recipe experiment/evidence traceability (`PROVENANCE.md:40-50`, `plan/EXPERIMENTS.md:300-324`).
- **Measured:** The current recipe worktree already had staged E23 changes and one unrelated untracked evidence JSON before this review. They were read but not modified.

### Method

- **Measured:** Inspection used only `git status`, `git log`, `git show`, `git diff --cached`, `rg`, `find`, `nl`, `sed`, and `diff`; no server, container, GPU, network, benchmark, or live process was touched.
- **Measured:** Commit traceability was checked against exact SHAs and described behavior in `PROVENANCE.md`, `plan/`, `docs/`, and `evidence/`.
- **Measured:** Source citations use recipe-relative paths; overlay citations are prefixed `[overlay]`; installed dependency citations are prefixed `[kt installed]` or `[kt pristine]`.
- **Measured:** Staged E23 text was treated as part of the current investigation rather than as committed history. In particular, the reported bug-C fix and its regression are not conflated with overlay commits (`plan/EXPERIMENTS.md:302-314`; `config/kt-kernel-patches/0001-experts-base-temp-buffer-lifetime.patch:1-100`).

### Not inspected or verified

- **Measured:** No live GPU state, server health, kernel journal, container image, runtime allocation, or benchmark result was independently rechecked because the live-run constraints forbid it.
- **Measured:** No script that contacts a server was run; this includes `scripts/bench-decode.py`, whose requests and metrics reads are at `scripts/bench-decode.py:66-114`, and `scripts/bench-power-sweep.sh`, whose server/GPU operations are at `scripts/bench-power-sweep.sh:41-90`.
- **Measured:** No upstream web issue, PR, or current upstream branch was inspected. Statements about upstream are only those recorded locally, for example `docs/TROUBLESHOOTING.md:90`.
- **Measured:** Native kt-kernel C++ implementations behind `forward_task`, host callbacks, and `CPUInfer` were not present in the three requested Python files and were not audited; Python hands them raw pointers at `[kt pristine] /tmp/r3/kt/experts_base.orig.py:624-632` and `:658-670`.
- **Measured:** Overlay tests were not run. The review therefore does not certify the overlay as independently reproducible.
- **Measured:** The E20 JSON tuning corpus was not numerically revalidated; only its source gate, commit, and recipe evidence links were reviewed (`docs/research/BENCHMARK-EVOLUTION-AND-E20-FP8.md:82`, `:157`).
- **Inferred:** Findings are strongest for the qualified baseline and weaker for LoRA, alternate speculative algorithms, non-SM120 devices, non-TP2 layouts, and NPU paths.

## 2. Hotspot map

| # | Claim | Area and location | Why accidental complexity is high | Blast radius and coupling | Baseline load-bearing? |
|---:|---|---|---|---|---|
| 1 | **Measured** | kt pinned/static buffer lifetime: `[kt pristine] experts_base.orig.py:216-283`; patch `config/kt-kernel-patches/0001-experts-base-temp-buffer-lifetime.patch:3-97` | Class-global caches mix permanent capture tuples, one reusable temporary tuple, seven tensor families, and graph-captured raw pointers. Registration can happen after an autotune allocation. The patch adds promotion and event retirement, but its only behavioral delta currently correlates with the supplied C4 regression. | Every CPU-expert submit/sync, every layer, capture, speculative width, eager shape transition, and host allocator lifetime. Coupled to decode-runner registration at `[overlay] .../decode_cuda_graph_runner.py:363-376`. | **Yes.** CPU experts and graph replay are baseline paths; the patch itself is not yet qualified because C4 reportedly falls from 148 to 49 aggregate while C1 is unchanged. |
| 2 | **Measured** | SM120 page-split persistent buffers: `[overlay] python/sglang/kernels/ops/attention/flash_mla_sm120.py:520-677` | Graphs retain old pointers; growth, zero-initialization, touched-page masks, two pool tags, and pre-reservation must agree. Old tensors are intentionally process-lifetime leaks to preserve graph validity. | All SM120 sparse MLA prefill/decode using re-paged pools; corruption can hit unrelated graph inputs. Coupled to pool geometry and backend initialization (`flash_mla_sm120.py:625-644`). | **Yes.** Re-paging is required by the baseline's c1/c2 pools (`docs/TROUBLESHOOTING.md:56-60`). |
| 3 | **Measured** | Breakable CUDA graph segmentation: `[overlay] .../breakable_cuda_graph.py:14-22`, `:225-284`, `:291-365` | It relies on shared-pool lifetime, weak references, strong bridge outputs, recursive in-place copying, monkey-patched stream ordering, nested eager-break behavior, and optional sync diagnostics. | Model output structure, Engram, attention, side streams, every decode graph segment, and fault attribution. | **Yes.** File-backed Engram requires eager graph breaks (`plan/EXPERIMENTS.md:179`). |
| 4 | **Measured** | TARGET_VERIFY static metadata and `captured_req_width`: `[overlay] .../decode_cuda_graph_runner.py:315-410`, `:730-800`, `:1385-1418` | Graph keys are request counts while tensors may have `bs × width` rows. Uniform and ragged verify have different keys, padding, layouts, and admission rules. kt buffers key on token count, not request count. | DSpark/DFlash verification, graph output sizing, attention metadata, kt capture sizes, padding, TBO and LoRA compatibility. | **No today** because speculation is off; **yes if reopened**. |
| 5 | **Measured** | Ragged verify: `[overlay] .../decode_cuda_graph_runner.py:378-401`, `:802-809`, `:1387-1408`; `[overlay] python/sglang/srt/speculative/ragged_verify.py:219-319` | Captured per-tier layouts are refreshed in place. It forbids TBO, LoRA, and disabled graph padding, and has separate request-slot and token-tier geometry. | Scheduler admission, graph selection, attention backends, speculative metadata, and output row counts. | **No.** Qualified baseline has speculation off. |
| 6 | **Measured** | Layerwise streamed prefill coordinator: `[overlay] .../kt_ep_wrapper.py:159-235`, `:276-371`, `:804-809`, `:907-947` | Singleton process state, rank-0 export ownership, POSIX shared memory, pinned registration, barriers, two host buffers, one GPU staging set, event reuse, and strict layer order form a distributed state machine hidden in one wrapper. | Prefill throughput, TP rank liveness, host/GPU memory, FlashInfer layouts, kt export API, and all MoE layers. | **Yes for qualified prefill; no for decode.** Threshold 1024 is the launcher default (`scripts/serve-hybrid.sh:47-57`). |
| 7 | **Measured** | D=4 deferral state: `[kt pristine] experts_base.orig.py:372-449`, `:542-589`, `:618-671`, `:730-817` | `_layer_has_pending_deferred` is class-global; layer L writes the alternate ring slot and layer L+1 asks native code to accumulate it. The ring is keyed by `layer_idx % 2`, not a forward ID. | Numerical correctness, batch/shape transitions, overlap, graph replay, all CPU-expert layers, and NaN behavior. | **Yes.** D=4 is the qualified default (`scripts/serve-hybrid.sh:41-46`); D>=5 is prohibited (`plan/EXPERIMENTS.md:313`). |
| 8 | **Measured** | Compact CPU expert view and rank-0 ownership: `[overlay] .../kt_ep_wrapper.py:374-466`, `:717-756`, `:768-809`, `:826-881` | A global monkey patch and temporary global mask rewrite the loader contract; expert IDs are remapped to a compact contiguous suffix; only rank 0 owns kt while every rank runs GPU MoE. | Weight loading, EPLB, router IDs, TP reductions, scaling, clamp semantics, draft all-GPU layers, and kt API versions. | **Yes.** This makes host memory fit (`plan/EXPERIMENTS.md:169`). |
| 9 | **Measured** | Engram file-backed path and graph breaks: `[overlay] python/sglang/srt/layers/engram.py:693-727`, `:804-862`; `[overlay] python/sglang/srt/models/deepseek_v4.py:3795-3821`, `:3862-3886` | mmap/page-cache ownership, prefetch threads, hash IDs, file rows, two eager breaks, and optional side-stream KV prefetch cross OS, CPU, and CUDA lifetimes. | Decode graph shape, host memory budget, storage behavior, latency, and launch flags. | **Yes.** File layout and breakable graphs are baseline requirements (`scripts/serve-hybrid.sh:31-36`, `:100`, `:132-136`). |
| 10 | **Measured** | Engram/attention graph-break forward-mode gates: `[overlay] .../deepseek_v4.py:1973`, `:3813-3820`; `[overlay] .../deepseek_v4_backend.py:421-426`, `:1057-1086`, `:2812` | `TARGET_VERIFY` also reports extend-like behavior, so predicates must distinguish prefill from verify. Two commits fixed similar gates in different files. | Speculative graph capture, low-ratio projections, attention, Engram hash IDs, and BCG segment count. | **No today; yes for any spec retry.** |
| 11 | **Measured** | Multi-stream overlap under graph capture: `[overlay] .../deepseek_v4.py:1187`, `:2034`, `:3591-3617`; workaround evidence `docs/TROUBLESHOOTING.md:68` | A production-default global optimization is incompatible with captured verify replay, but the workaround is an environment override rather than a forward-mode capability gate. | Token correctness, acceptance rate, attention/dense overlap, DSpark target and draft execution. | **No today.** It is dangerous if speculation is re-enabled. |
| 12 | **Measured** | Launch and environment assembly: `scripts/serve-hybrid.sh:24-80`, `:84-146` | Shell defaults, diagnostic escape hatches, verbatim word splitting, container env, server args, bind mounts, and argparse-last-wins behavior collectively define the runtime. | Reproducibility, security, graph backend, memory fit, all feature combinations, and evidence comparability. | **Yes.** This is the canonical baseline launcher. |
| 13 | **Measured** | Evidence/config matrices: `plan/EXPERIMENTS.md:300-344`, `plan/TASKS.md:73-82`, `docs/BENCHMARKS.md` E23 tables | Results, defaults, workarounds, gates, and open defects are duplicated across roadmap, tasks, experiments, benchmarks, learning, troubleshooting, provenance, and raw evidence. Some staged E23 statements already lag today's supplied regression. | Operator decisions, release claims, future experiment design, and commit justification. | **Yes operationally.** Code may run without it, but qualification cannot be trusted without coherent records. |
| 14 | **Measured** | SM120 W8A8 tuning/config gate: overlay commit `32a25f37`; `docs/TROUBLESHOOTING.md:126-130`; `docs/research/BENCHMARK-EVOLUTION-AND-E20-FP8.md:82,157` | Seven device-name/shape-specific files depend on TP layout and an architecture gate permitting split-K/swap-AB. A shape change silently falls back with only warnings. | Dense projections, shared experts, TP changes, device naming, decode throughput, and numerical checks. | **Yes for performance, not functional correctness.** |
| 15 | **Measured** | NaN/Xid workarounds and diagnosis: `plan/EXPERIMENTS.md:348-354`; `docs/TROUBLESHOOTING.md:93-103` | Three sightings have different reproducibility and suspected boundaries; asynchronous faults surface at later syncs. Zero-fill removed one symptom; D>=5 and a D=4 multi-sequence sighting remain unresolved. | Sampling, attention, kt deferral, memory pressure, request survivability, and release safety. | **Yes as a release gate.** D=4 is load-bearing but not fully explained. |

**Erratum 2026-09-23:** the C4 “regression” in hotspot 1 was the correction of a pre-existing silent bug, not a patch-induced performance loss. On the pristine build, only row 0 of the bs=4 graph received correct CPU-expert output; rows 1–3 used stale/aliased buffers. Qualified patched performance is **50.8 summed / 47.0 wall tok/s (12.7 per stream)**, and all historical C3/C4 near-linear scaling is an artifact. See `docs/BENCHMARKS.md` “Erratum” and `docs/TROUBLESHOOTING.md` bug C.

### Concentration of risk

- **Inferred:** The dominant accidental-complexity pattern is implicit ownership: graph objects own pointers they did not allocate, native callbacks own pointers Python cannot see, the next layer owns prior deferred output, rank 0 owns CPU work, and process-global dictionaries own state across runners.
- **Inferred:** The second pattern is dimensional ambiguity: request batch, flattened token count, captured width, real token count, padded token count, and per-field output rows are all called “batch” in adjacent layers.
- **Inferred:** The qualified baseline is stable only inside a narrow compatibility envelope; enabling one nominally independent feature can cross graph, buffer, rank, and scheduling boundaries.

## 3. Debt register

### 3a. Immediate correctness debt

| ID | Severity | Claim and evidence | Smallest safe fix |
|---|---|---|---|
| C1 | Critical | **Measured:** The bug-C kt patch prevents a captured-temp use-after-free, but the supplied current result says its promotion changes C4 aggregate throughput from 148 to 49 while C1 is unaffected. The promotion is at `config/kt-kernel-patches/0001-experts-base-temp-buffer-lifetime.patch:7-21`; the patch also makes promoted tuples permanently owned at `:54-76`. | **Proposed:** Freeze architecture work. Create a CPU-only/unit harness around tuple identity and native callback sequencing, then run one controlled GPU A/B only after the live run ends. Split promotion from retirement into separately benchmarked patches. Do not ship or remove the patch on mechanism alone. |
| C2 | Critical | **Measured:** D>=5 deterministically produces NaN on first bs=4, and D=4 had one unreproduced multi-sequence sighting (`docs/TROUBLESHOOTING.md:101-103`). | **Proposed:** Add a hard startup gate rejecting `max_deferred_experts_per_token >= 5` for this hybrid, not just a comment. Keep D=4 under an explicit soak/release gate. |
| C3 | High | **Measured:** Class-global `_layer_has_pending_deferred` and static `KExpertsCPUBuffer` state have no forward or runner identity (`[kt pristine] experts_base.orig.py:223-227`, `:378`, `:449`). | **Proposed:** First add assertions that state is empty/consistent at forward boundaries and that adjacent layers share shape/tuple identity. Only later make state instance-scoped. |
| C4 | High | **Measured:** Bug A is worked around globally with `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`; source still defaults the optimization on (`[overlay] environ.py:1450`; `docs/TROUBLESHOOTING.md:68`). | **Proposed:** Gate overlap off in code only for captured speculative draft/verify forwards and add a regression test comparing eager and graph tokens. Keep the env workaround until that test passes. |
| C5 | High | **Measured:** SM120 grow-after-capture is survivable only because old tensors are never freed and reservation uses assumed-static pool geometry (`[overlay] flash_mla_sm120.py:541-555`, `:625-644`). | **Proposed:** Turn any post-capture growth into a hard error after all pools are reserved; retain retire-not-free only as a diagnostic fallback until capture-order tests exist. |
| C6 | High | **Measured:** The launcher permits incompatible combinations through untyped `EXTRA_SERVER_ARGS`, `EXTRA_DOCKER_ENV`, and `EXTRA_DOCKER_ARGS` with intentional word splitting (`scripts/serve-hybrid.sh:72-80`, `:112-120`). | **Proposed:** Add a read-only validation phase that prints and rejects known-bad combinations before invoking the container command; preserve existing defaults and escape hatches. |
| C7 | High | **Measured:** The current docs say the kt patch was verified and E23 closed, while today's supplied regression makes the patched baseline unqualified. Existing staged statements are at `plan/EXPERIMENTS.md:302-314`. | **Proposed:** Before merge, amend E23 records to distinguish “correctness crash fixed” from “patch performance qualification failed/pending.” Do not alter raw evidence. |
| C8 | Medium | **Measured:** kt capture sizes are registered globally from each decode runner (`[overlay] decode_cuda_graph_runner.py:368-376`), but a prior autotune can allocate before registration (`docs/TROUBLESHOOTING.md:86`). | **Proposed:** Register all capture token sizes before any model/autotune forward, then assert capture-time requests are already permanent. |
| C9 | Medium | **Measured:** Rank-0-only kt behavior returns zeros on other ranks and relies on later TP reduction (`[overlay] kt_ep_wrapper.py:844-881`, `:935-943`) without a local topology assertion. | **Proposed:** Assert TP rank, TP size, and combine semantics when constructing the wrapper; reject unsupported EP/PP layouts explicitly. |
| C10 | Medium | **Measured:** E24 calls for per-request NaN survival, but current failure is a device-side assert that can terminate serving (`plan/EXPERIMENTS.md:348-354`). | **Proposed:** Add test-build fault injection and prove request-level rejection before calling the recipe releasable; do not add retry/clamp behavior. |
| C11 | Critical | **Measured 2026-09-23:** no quality gate ran at batch size > 1 until 2026-09-23; `scripts/verify-batch-consistency.py --same-prompt` exposed bug C's silent corruption in rows 1–3. | **Required:** run `scripts/verify-batch-consistency.py --same-prompt` at every captured batch size in the release gate, including padded shapes, and compare all rows with the bs=1 greedy reference. |

### 3b. Later refactoring debt

| ID | Severity | Claim and evidence | Smallest safe refactor after experiments stabilize |
|---|---|---|---|
| R1 | High | **Measured:** `KTEPWrapperMethod` combines loader monkey-patching, configuration, routing, CPU submission, TP policy, streamed prefill, probes, and output combining (`[overlay] kt_ep_wrapper.py:374-980`). | **Proposed:** Extract protocols for CPU-expert execution and streamed-weight export without changing algorithms; migrate one call site at a time. |
| R2 | High | **Measured:** BCG recursively interprets arbitrary containers/dataclass-like objects and mutates outputs (`[overlay] breakable_cuda_graph.py:164-222`). | **Proposed:** Require an explicit graph-output adapter protocol with static tensor fields and row-count metadata. |
| R3 | High | **Measured:** Request-count and token-count graph dimensions cross several APIs (`[overlay] decode_cuda_graph_runner.py:315-410`). | **Proposed:** Introduce named types/dataclasses such as `GraphRequestShape` and `GraphTokenShape`; eliminate bare `bs` at interface boundaries. |
| R4 | Medium | **Measured:** Layerwise prefill is a singleton with hidden registration/order state (`[overlay] kt_ep_wrapper.py:167-189`, `:313-363`). | **Proposed:** Make a runner-owned coordinator with explicit `begin_forward`, `run_layer`, and `end_forward`, preserving the two-buffer schedule. |
| R5 | Medium | **Measured:** Environment flags are defined across shell, `environ.py`, direct `os.environ` reads, and kt (`scripts/serve-hybrid.sh:24-80`; `[overlay] kt_ep_wrapper.py:70-76`; `[kt pristine] experts_base.orig.py:75`). | **Proposed:** Generate a typed registry and launcher documentation from one schema; retain env compatibility aliases for at least one release cycle. |
| R6 | Medium | **Measured:** Evidence conclusions are duplicated across at least six documents (`plan/EXPERIMENTS.md:300-354`; `docs/TROUBLESHOOTING.md:62-103`; `PROVENANCE.md:40-50`). | **Proposed:** Make one experiment manifest per E-number containing config, commits, evidence paths, result, and qualification status; render summary tables from it later. |
| R7 | Medium | **Measured:** The compact loader uses process globals to communicate a temporary expert mask (`[overlay] kt_ep_wrapper.py:396-466`, `:796-802`). | **Proposed:** Pass an explicit expert-selection object into a loader adapter; do not monkey-patch the library class globally. |
| R8 | Low | **Measured:** Diagnostic flags and production flags share the same launch escape hatch (`scripts/serve-hybrid.sh:72-80`). | **Proposed:** Separate `diagnostics` from `runtime` config and print a loud “unqualified diagnostic mode” banner when any diagnostic is enabled. |

## 4. Undocumented or under-enforced invariants

**Measured:** The following register contains 24 invariants relied upon by code. “Undocumented” here includes invariants mentioned in local comments but not asserted at the boundary that depends on them.

| # | Claim | Invariant and evidence | Assert/document at |
|---:|---|---|---|
| 1 | **Inferred** | Every layer in one forward must resolve the same `KExpertsCPUBuffer` tuple for a given flattened token count; otherwise ring slots point into unrelated allocations (`[kt pristine] experts_base.orig.py:542-589`, `:730-757`). | **Proposed:** Assert tuple identity in `BaseMoEWrapper._prepare_forward_cpu_buffers`; document in kt buffer class. |
| 2 | **Measured** | `capture_bs` must be populated before first capture-time `get_buffer` for each token count; late registration caused bug C (`[overlay] decode_cuda_graph_runner.py:368-376`; `docs/TROUBLESHOOTING.md:86-87`). | **Proposed:** Assert in `KExpertsCPUBuffer.get_buffer` whenever the current stream captures. |
| 3 | **Inferred** | A tuple used by any graph must remain strongly reachable for that graph's full lifetime. The pristine implementation does not represent graph ownership (`[kt pristine] experts_base.orig.py:223-283`). | **Proposed:** Document in `KExpertsCPUBuffer`; track graph-owned tuple IDs explicitly. |
| 4 | **Inferred** | Native CPU tasks may outlive the Python call that supplied their raw pointers (`[kt pristine] experts_base.orig.py:624-670`). | **Proposed:** Document on `submit_forward`; expose a completion token from native code. |
| 5 | **Measured** | `buffer_depth == 2` is coupled to `layer_idx % 2` and “defer into next layer” semantics (`[kt pristine] experts_base.orig.py:227`, `:562-563`, `:664-671`). | **Proposed:** Assert depth 2 while deferral is enabled; document why depth is not a tuning knob. |
| 6 | **Inferred** | Layer L+1's immediate native task accumulates layer L's deferred output when `_layer_has_pending_deferred[L]` is true (`[kt pristine] experts_base.orig.py:623-632`). | **Proposed:** Document beside `_layer_has_pending_deferred`; assert adjacent layer IDs and matching shapes. |
| 7 | **Inferred** | No unrelated forward may interleave between adjacent layers sharing class-global deferred state. | **Proposed:** Add a monotonically increasing forward ID to deferred state; assert in submit/sync. |
| 8 | **Measured** | The final model layer cannot defer because no next layer can accumulate it (`[overlay] kt_ep_wrapper.py:685-691`). | **Proposed:** Keep code and add a unit assertion/test naming the invariant. |
| 9 | **Measured** | D must remain below 5 for the qualified system (`scripts/serve-hybrid.sh:41-46`; `docs/TROUBLESHOOTING.md:101`). | **Proposed:** Enforce in server-argument validation, with an explicit unsafe experimental override. |
| 10 | **Measured** | The compact view requires a contiguous split: GPU IDs first, CPU IDs as one suffix (`[overlay] kt_ep_wrapper.py:717-726`, `:547-555`). | **Proposed:** Assert mask equals that exact prefix; document in a CPU-expert protocol. |
| 11 | **Measured** | EPLB physical-to-logical mapping must be identity (`[overlay] kt_ep_wrapper.py:772-795`). | **Proposed:** Existing runtime rejection is good; also surface it in launch/config documentation. |
| 12 | **Measured** | Only TP rank 0 owns and executes kt CPU experts (`[overlay] kt_ep_wrapper.py:727-747`, `:844-881`). | **Proposed:** Assert supported TP/EP/PP topology at wrapper construction. |
| 13 | **Inferred** | Non-rank-0 zeros plus rank-0 CPU output are correct only if the enclosing MoE combine performs the expected TP sum (`[overlay] kt_ep_wrapper.py:875-881`, `:935-943`). | **Proposed:** Assert combine/reduction mode in `create_moe_runner`. |
| 14 | **Measured** | Routed scaling must be applied exactly once after CPU+GPU sum (`[overlay] kt_ep_wrapper.py:611-617`). | **Proposed:** Keep class override; add a numerical contract test around wrapper output. |
| 15 | **Measured** | Streamed prefill makes the same threshold decision on every rank and visits registered layers in increasing order (`[overlay] kt_ep_wrapper.py:280-323`, `:907-914`). | **Proposed:** Broadcast/compare token count at `begin_forward`; assert all ranks register identical layer keys. |
| 16 | **Measured** | Rank 0 must finish export before either rank reads a host buffer, and all ranks must finish DMA before overwrite (`[overlay] kt_ep_wrapper.py:283-333`). | **Proposed:** Document as buffer ownership states; add sequence counters around barriers in debug builds. |
| 17 | **Inferred** | Every registered streamed layer has the same CPU-expert count, hidden size, TP intermediate size, and compatible GPU method as the first layer used to construct the singleton (`[overlay] kt_ep_wrapper.py:177-188`, `:262-274`). | **Proposed:** Assert all fields in `register`. |
| 18 | **Measured** | A page-split pool's page count never changes after initialization (`[overlay] flash_mla_sm120.py:551-555`). | **Proposed:** Seal pool geometry after `reserve_split_buffers` and raise on growth. |
| 19 | **Measured** | SWA and compressed extra pools need distinct persistent output tags (`[overlay] flash_mla_sm120.py:662-677`). | **Proposed:** Replace free-form string tags with an enum and assert uniqueness. |
| 20 | **Measured** | Unwritten split pages must be finite because padding/invalid rows can still read them (`[overlay] flash_mla_sm120.py:590-604`). | **Proposed:** Keep zero allocation and add a focused padding test. |
| 21 | **Measured** | A uniform verify batch's runtime width must equal `captured_req_width`; otherwise it must execute eagerly (`[overlay] decode_cuda_graph_runner.py:740-749`). | **Proposed:** Preserve admission check and add a counter/log for eager fallback. |
| 22 | **Measured** | Compact ragged verify is incompatible with TBO, LoRA, and disabled graph padding (`[overlay] decode_cuda_graph_runner.py:392-401`). | **Proposed:** Move this to central configuration validation and document in the registry. |
| 23 | **Measured** | Qualified decode assumes graph request buckets include 1, 2, and 4, padding remains enabled, maximum running requests is 4, and capture max is 4 (`scripts/serve-hybrid.sh:69-70`, `:104-105`; `plan/EXPERIMENTS.md:328-331`). | **Proposed:** Emit the resolved capture list at launch and fail qualification mode if it differs. |
| 24 | **Measured** | Engram `file` lookup requires breakable decode graphs and stable shard paths (`plan/EXPERIMENTS.md:179`; `scripts/serve-hybrid.sh:31-36`, `:100`, `:132-136`). | **Proposed:** Validate layout/backend pairing before model load; document file/shard lifetime ownership. |

## 5. Overlay commit traceability

**Measured:** `git log --reverse --oneline da64c5cb..HEAD` contains 16 commits. “Linked” means a recipe document or evidence file names the SHA or uniquely records the behavior and experiment.

| SHA | Subject | Recipe justification | Traceability |
|---|---|---|---|
| `3805730e` | kt-ep hybrid: compact CPU-expert view, kt-kernel 0.7 API, V4.1 fixes | **Measured:** E17 result and one-layer evidence (`plan/EXPERIMENTS.md:169`; `evidence/t36b-kt-one-layer/`; `PROVENANCE.md:40`). | Linked |
| `a1ca246a` | Engram file layout | **Measured:** E18 implementation and timings (`plan/EXPERIMENTS.md:179`; `evidence/e18-engram-file/`; `PROVENANCE.md:41`). | Linked |
| `cd55166a` | BCG dataclass outputs; SM120 extra-pool re-page | **Measured:** E19 blockers and troubleshooting (`plan/EXPERIMENTS.md:204`; `docs/TROUBLESHOOTING.md:50-60`; `PROVENANCE.md:42`). | Linked |
| `32a25f37` | SM120 W8A8 configs and split-K/swap-AB gate | **Measured:** E20 tuning and correctness evidence (`plan/EXPERIMENTS.md:218`; `evidence/e20-dense-fp8-gemm/`; `PROVENANCE.md:43`). | Linked |
| `1d6a98d3` | Zero-fill SM120 page-split buffers | **Measured:** E20/T3.6d2 soak and finite checks (`plan/EXPERIMENTS.md:225`; `docs/TROUBLESHOOTING.md:99`; `PROVENANCE.md:44`). | Linked |
| `11e7f745` | Deferral with compact CPU view | **Measured:** E21 D=4 and selector comparison (`plan/EXPERIMENTS.md:249`; `evidence/e21-kt-fixed-floor/`; `PROVENANCE.md:45`). | Linked |
| `0138c6ad` | Layerwise streamed GPU prefill | **Measured:** E22 evidence and architecture record (`plan/EXPERIMENTS.md:291-298`; `evidence/e22-layerwise-prefill/`; `PROVENANCE.md:46`). | Linked |
| `670994ac` | Per-layer NaN probe | **Measured:** E22d probe and non-reproduction record (`plan/EXPERIMENTS.md:293-295`; `PROVENANCE.md:47`). | Linked |
| `6032dd44` | Keep prefill graph breaks out of TARGET_VERIFY | **Measured:** E23 fit/capture record (`plan/TASKS.md:74`; `evidence/e23-spec-decode/fit/k2-mfs0.92-oom.txt`; `PROVENANCE.md:48`). | Linked |
| `3b1fedce` | Gate low-ratio source projection break | **Measured:** Same E23 fit record and paired provenance entry (`plan/TASKS.md:74`; `PROVENANCE.md:48`). | Linked |
| `71645529` | Preserve per-field dataclass row counts | **Measured:** E23 fit attempt and open-source plan (`evidence/e23-spec-decode/fit/k2-mfs0.92-oom.txt`; `docs/research/OPEN-SOURCE-PLAN.md:28`; `PROVENANCE.md:49`). | Linked |
| `95f2fe3d` | Scale BCG output cap by captured request width | **Measured:** E23 fit evidence and task result (`evidence/e23-spec-decode/fit/k2-mfs0.92-oom.txt`; `plan/TASKS.md:74`; `PROVENANCE.md:50`). | Linked |
| `b457c4e6` | BCG segment/break diagnostics and nested eager behavior | **Measured:** Behavior is described in E23's troubleshooting diagnostic ladder (`docs/TROUBLESHOOTING.md:69-74`) but the SHA is not named anywhere in the recipe tree. | **No direct SHA-linked evidence** |
| `4685955e` | E23 eager-side diagnostics, mm skip, POST_REPLAY | **Measured:** Behaviors are described at `docs/TROUBLESHOOTING.md:72,74,78`, but the SHA is not named in the recipe tree. | **No direct SHA-linked evidence** |
| `838490a6` | Retain graph-captured SM120 buffers and pre-size | **Measured:** Bug-B diagnosis, fix, and successful matrix (`docs/TROUBLESHOOTING.md:66-70`; `plan/EXPERIMENTS.md:314`; `evidence/e23-spec-decode/`). | Linked |
| `07f4aef0` | Sync before segment 0 for fault attribution | **Measured:** The discriminator behavior is recorded (`docs/TROUBLESHOOTING.md:82`), but the SHA is not named in the recipe tree. | **No direct SHA-linked evidence** |

- **Measured:** 13 of 16 overlay commits have direct SHA-linked recipe evidence; 3 of 16 (`b457c4e6`, `4685955e`, `07f4aef0`) have behavior-level records but no direct SHA citation.
- **Proposed:** Add those three SHAs to `PROVENANCE.md` and link each to the exact E23 crash excerpt before publishing the overlay history.

## 6. Configuration inventory and consolidation proposal

### Qualified launcher and hybrid feature controls

| Name | Claim | Read/default | Purpose | Class | Documented at |
|---|---|---|---|---|---|
| `KT_NUM_GPU_EXPERTS` | **Measured** | launcher, `208` (`scripts/serve-hybrid.sh:39`) | GPU/CPU expert split | Production | launcher comments; E19/E21 plans |
| `KT_CPUINFER` | **Measured** | launcher, `16` (`scripts/serve-hybrid.sh:40`) | CPU worker threads | Production | launcher; E17/E21 |
| `KT_MAX_DEFERRED` | **Measured** | launcher, `4` (`scripts/serve-hybrid.sh:41-46`) | Lowest-score experts deferred one layer | Production, safety-sensitive | launcher; E21; troubleshooting |
| `KT_GPU_PREFILL_MIN_TOKENS` | **Measured** | launcher `1024`; overlay direct env default `0` (`scripts/serve-hybrid.sh:47-55`; `[overlay] kt_ep_wrapper.py:70`) | Switch large prefill to streamed GPU experts | Production | launcher; E22 |
| `KT_GPU_PREFILL_LOG` | **Measured** | launcher/overlay `0` (`scripts/serve-hybrid.sh:55`; `[overlay] kt_ep_wrapper.py:71`) | Per-chunk streamed timing | Debug | launcher comment |
| `KT_NAN_PROBE_MIN_TOKENS` | **Measured** | launcher/overlay `0` (`scripts/serve-hybrid.sh:56-57`; `[overlay] kt_ep_wrapper.py:72-76`) | Per-layer finite checks | Debug | launcher; E22d |
| `KT_FORCE_SYNC_SUBMIT` | **Measured** | kt direct env, off (`[kt pristine] experts_base.orig.py:40,75`) | Synchronous callback bypass | Debug/fallback | kt source only |
| `DSPARK` | **Measured** | launcher `0` (`scripts/serve-hybrid.sh:58-63`, `:109-111`) | Enable speculative algorithm | Experimental, off | launcher; E23 |
| `DSPARK_K` | **Measured** | launcher `5` (`scripts/serve-hybrid.sh:63`) | DSpark block size | Experimental | launcher; E23 |
| `MEM_FRACTION_STATIC` | **Measured** | launcher `0.92` (`scripts/serve-hybrid.sh:64`, `:101`) | Static GPU memory fraction | Production | launcher; E23 fit notes |
| `CHUNKED_PREFILL_SIZE` | **Measured** | launcher `4096` (`scripts/serve-hybrid.sh:66-68`, `:102`) | Scheduler chunk and kt native buffer ceiling | Production, memory-sensitive | launcher; E22 |
| `MAX_RUNNING_REQUESTS` | **Measured** | launcher `4` (`scripts/serve-hybrid.sh:69`, `:104`) | Scheduler concurrency | Production | launcher; E21/E23 |
| `CUDA_GRAPH_MAX_BS` | **Measured** | launcher `4` (`scripts/serve-hybrid.sh:70`, `:105`) | Decode capture ceiling | Production | launcher; E23 |
| `ENGRAM_FILE_THREADS` | **Measured** | launcher `16`; overlay typed default `16` (`scripts/serve-hybrid.sh:71`, `:136`; `[overlay] environ.py:1429`) | mmap prefetch workers | Production | launcher; E18 |
| `EXTRA_SERVER_ARGS` | **Measured** | empty, word-split (`scripts/serve-hybrid.sh:72`, `:112-113`) | Arbitrary server overrides | Diagnostic/escape hatch | launcher only |
| `EXTRA_DOCKER_ENV` | **Measured** | empty, word-split (`scripts/serve-hybrid.sh:73-76`, `:115-118`) | Arbitrary container env | Diagnostic/experimental | launcher only |
| `EXTRA_DOCKER_ARGS` | **Measured** | empty, word-split (`scripts/serve-hybrid.sh:77-80`, `:119-120`) | Mount instrumentation/overrides | Debug-only | launcher only |

### SGLang flags material to this overlay

| Name | Claim | Read/default | Purpose | Class | Documented at |
|---|---|---|---|---|---|
| `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | **Measured** | typed default false; launcher sets `1` (`[overlay] environ.py:1408`; `scripts/serve-hybrid.sh:133`) | Host/file Engram path | Production | launcher; E18 |
| `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT` | **Measured** | typed `auto`; launcher `file` (`[overlay] environ.py:1426`; `scripts/serve-hybrid.sh:134`) | Engram ownership/layout | Production | launcher; E18 |
| `SGLANG_DSV41_ENGRAM_FILE_DIR` | **Measured** | typed empty; launcher snapshot (`[overlay] environ.py:1427`; `scripts/serve-hybrid.sh:135`) | Stable shard directory | Production | launcher |
| `SGLANG_DSV41_ENGRAM_FILE_THREADS` | **Measured** | typed `16` (`[overlay] environ.py:1429`) | File prefetch workers | Production | launcher |
| `SGLANG_ENABLE_DSV41_ENGRAM_KV_PREFETCH` | **Measured** | typed false (`[overlay] environ.py:1410`; `[overlay] deepseek_v4.py:3677`) | Side-stream layer-14 prefetch | Experimental | source comments only |
| `SGLANG_OPT_USE_TILELANG_INDEXER` | **Measured** | launcher forces `0` (`scripts/serve-hybrid.sh:137-139`) | Avoid incompatible indexer default | Production workaround | troubleshooting `:46-48` |
| `SGLANG_FP8_PAGED_MQA_LOGITS_TORCH` | **Measured** | typed false; launcher forces `0` (`[overlay] environ.py:1435`; `scripts/serve-hybrid.sh:139`) | Avoid incompatible logits fallback | Production workaround | troubleshooting `:46-48` |
| `SGLANG_SM120_FLASHMLA_BACKEND` | **Measured** | typed `flashinfer` (`[overlay] environ.py:996`; `[overlay] flash_mla_sm120.py:206-208`) | Select flashinfer/triton/torch | Production fallback switch | source docstring; troubleshooting |
| `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` | **Measured** | typed true (`[overlay] environ.py:1450`) | Model side-stream overlap | Production default; spec workaround sets false | E23 troubleshooting `:68` |
| `SGLANG_BCG_DEBUG_SYNC` | **Measured** | direct env `0` (`[overlay] breakable_cuda_graph.py:35-38`) | Sync around segments/breaks | Debug-only | source; E23 troubleshooting |
| `SGLANG_BCG_DEBUG_EAGER` | **Measured** | direct env empty (`[overlay] runner_backend/utils.py:90-102`) | Force draft or target eager | Debug-only | source; E23 |
| `SGLANG_BCG_DEBUG_LAYER_BREAKS` | **Measured** | direct env empty (`[overlay] deepseek_v4.py:751-752`, `:3658-3667`) | Insert layer-granular breaks | Debug-only | source; troubleshooting `:69` |
| `SGLANG_BCG_DEBUG_INPUTS` | **Measured** | direct env `0` (`[overlay] deepseek_v4.py:854-863`) | Dump static graph inputs | Debug-only | source; troubleshooting `:69` |
| `SGLANG_SM120_IDX_GUARD` | **Measured** | direct env `0` (`[overlay] deepseek_v4.py:864-868`) | Count/clamp bad attention indices | Debug-only | source; troubleshooting `:69` |
| `SGLANG_DSPARK_ENABLE_MULTI_STREAM` | **Measured** | typed true (`[overlay] environ.py:530`) | DSpark-specific side streams | Experimental | source only |
| `SGLANG_DSPARK_FAST_KERNEL` | **Measured** | typed true (`[overlay] environ.py:520`) | DSpark optimized kernel | Experimental | source only |
| `SGLANG_DSPARK_FP32_LM_HEAD` | **Measured** | typed false (`[overlay] environ.py:521`) | DSpark diagnosis/precision switch | Experimental/debug | source only |

### Script variables affecting evidence comparability

| Name | Claim | Default/evidence | Risk |
|---|---|---|---|
| `--concurrency` | **Measured** | decode defaults `1,2,4,8` (`scripts/bench-decode.py:121`) | Qualified launcher allows only 4 running requests; higher levels may queue and summed rates mislead. Wall rate is separately computed (`:95-108`). |
| `--distinct-prompts` | **Measured** | off by default (`scripts/bench-decode.py:127-131`) | Same prompts share routing and can overstate CPU-tier scaling (`:39-42`). |
| `--max-tokens`, `--runs`, `--warmup` | **Measured** | 256, 5, 2 (`scripts/bench-decode.py:122-124`) | E23 overrides these; evidence is incomparable unless invocation is retained. |
| `CAPS`, `RUNS`, `PHASES`, `CONC` | **Measured** | power sweep shell defaults (`scripts/bench-power-sweep.sh:19-26`) | Power and throughput matrices can differ silently by shell override. |

### Consolidation

- **Proposed:** Define one `HybridRuntimeConfig` schema in Python with typed fields, ranges, production/diagnostic classification, compatibility rules, and provenance text. Have the shell launcher invoke a CPU-only renderer that outputs server args and container env.
- **Proposed:** Keep environment names as compatibility inputs; resolve once, print a redacted canonical config, and pass explicit values downstream. Do not teach model code to read more direct `os.environ` values.
- **Proposed:** Add a generated flag registry with columns matching the tables above: owner, default, valid values, mode, incompatibilities, evidence, and removal criteria.
- **Proposed:** Treat `EXTRA_*` as an “unqualified mode.” It should remain available for diagnosis, but canonical evidence should record its resolved tokens.
- **Proposed:** Defer this consolidation until C1/C2 are understood; changing launch plumbing during buffer-lifetime experiments would add an uncontrolled variable.

## 7. Proposed assertions

**Proposed:** These snippets are proposals only. They were not inserted.

### 7.1 kt boundary tensor contract — low risk

**Proposed insertion site:** `[overlay] python/sglang/srt/layers/moe/kt_ep_wrapper.py`, in `KTEPWrapperMethod.submit`, immediately after lines 847-849.

```python
assert x.ndim == 2 and x.shape[1] == self.hidden_size, x.shape
assert topk_ids.shape == topk_weights.shape, (
    topk_ids.shape,
    topk_weights.shape,
)
assert topk_ids.ndim == 2 and topk_ids.shape[0] == x.shape[0]
assert topk_ids.shape[1] == self.wrapper.num_experts_per_tok
assert topk_ids.dtype in (torch.int32, torch.int64)
assert topk_weights.is_floating_point()
assert x.device == topk_ids.device == topk_weights.device
assert bool(((topk_ids >= 0) & (topk_ids < self.global_num_experts)).all())
```

- **Proposed:** Enable in debug/CI first; the final device reduction can synchronize and therefore needs a benchmark before production.

### 7.2 Capture-time buffer membership and identity — low risk if Python-only checks

**Proposed insertion site:** `[kt installed] /tmp/r3/kt/experts_base.py`, `KExpertsCPUBuffer.get_buffer`, before returning any tuple at lines 236-253.

```python
capturing = torch.cuda.is_current_stream_capturing()
if capturing:
    assert batch_size in cls.capture_bs, (
        f"capturing unregistered kt buffer size {batch_size}; "
        f"registered={sorted(cls.capture_bs)}"
    )
    assert batch_size in cls.capture_buffers, (
        f"capturing temporary kt tuple for size {batch_size}"
    )
    buf = cls.capture_buffers[batch_size]
    assert buf is cls.capture_buffers[batch_size]
```

- **Proposed:** The checks are low risk, but ordering must change so promotion happens before the assertion. Do not allocate or log stack traces while capturing.

### 7.3 No temporary allocation during capture — low risk, fail-fast

**Proposed insertion site:** `[kt installed] experts_base.py`, immediately before allocation at current line 281.

```python
assert not torch.cuda.is_current_stream_capturing(), (
    f"kt temporary allocation during CUDA graph capture: bs={batch_size}, "
    f"capture_bs={sorted(cls.capture_bs)}, "
    f"cached={sorted(cls.capture_buffers)}"
)
```

- **Proposed:** Enable immediately in diagnostic builds. It converts memory corruption into startup failure and should not affect successful paths.

### 7.4 Captured tuple is never retired — low risk

**Proposed insertion site:** `[kt installed] experts_base.py`, first lines of `_retire_temp_buffer` after `old = cls.temp_buffer`.

```python
captured_ids = {id(buf) for buf in cls.capture_buffers.values()}
assert id(old) not in captured_ids or cls.temp_bs in cls.capture_buffers
if cls.temp_bs in cls.capture_buffers:
    assert cls.capture_buffers[cls.temp_bs] is old
    return
```

- **Proposed:** Add a unit test that retains weak references and cycles eager sizes; benchmark only after promotion and retirement are tested separately.

### 7.5 Per-forward tuple identity across layers — needs benchmark before enabling

**Proposed insertion site:** `[kt pristine] BaseMoEWrapper._prepare_forward_cpu_buffers`, after `get_buffer` at line 560.

```python
key = (flat_hidden_states.shape[0], flat_hidden_states.shape[1])
buf_id = id(input_tensor_cpu)
prev = getattr(BaseMoEWrapper, "_debug_forward_buffer", None)
if self.layer_idx == 0:
    BaseMoEWrapper._debug_forward_buffer = (key, buf_id)
elif prev is not None:
    assert prev == (key, buf_id), (self.layer_idx, prev, key, buf_id)
```

- **Proposed:** The concept is valuable, but layer numbering may not begin at zero and concurrent/draft forwards need a real forward ID. Keep behind a debug flag until modeled correctly.

### 7.6 Deferred-state adjacency — needs benchmark and design review

**Proposed insertion site:** `[kt pristine] BaseMoEWrapper.submit_forward`, before computing `incremental` near lines 775-783.

```python
if BaseMoEWrapper._layer_has_pending_deferred.get(self.layer_idx - 1, False):
    assert self.layer_idx > 0
    assert current_slot == self.layer_idx % KExpertsCPUBuffer.buffer_depth
    assert next_slot == (current_slot + 1) % KExpertsCPUBuffer.buffer_depth
    assert KExpertsCPUBuffer.buffer_depth == 2
```

- **Proposed:** Low runtime cost, but it does not solve cross-forward contamination. Add a forward ID before treating it as a complete guard.

### 7.7 Forward-mode compatibility — low risk

**Proposed insertion site:** `[overlay] KTEPWrapperMethod.apply`, before line 907.

```python
mode = get_forward_context().forward_batch.forward_mode
if self.streamer is not None and self.streamer.wants(x.shape[0]):
    assert mode.is_extend_without_speculative(), (
        f"streamed kt prefill unsupported for forward mode {mode}"
    )
```

- **Proposed:** Use the repository's actual forward-context accessor. Enable after a unit test covers prefill, decode, and TARGET_VERIFY.

### 7.8 Unsupported combinations at startup — low risk

**Proposed insertion site:** central server-argument validation after all hooks resolve values, before model load.

```python
if cfg.kt_weight_path:
    if cfg.kt_max_deferred_experts_per_token >= 5:
        raise ValueError("KT deferral >=5 is blocked pending E24 NaN root cause")
    if cfg.disable_cuda_graph_padding:
        raise ValueError("KT hybrid requires CUDA graph padding for qualified capture sizes")
    if cfg.speculative_algorithm and cfg.speculative_algorithm != "DSPARK":
        raise ValueError("KT hybrid speculative mode is only audited for DSPARK")
    if cfg.speculative_algorithm and cfg.cuda_graph_backend_decode != "breakable":
        raise ValueError("file-backed Engram + speculative KT requires breakable decode graphs")
```

- **Proposed:** Low risk for qualification mode; needs a separate `--allow-unqualified-hybrid` escape hatch for experiments.

### 7.9 Spec width and kt capture keys — low risk

**Proposed insertion site:** `[overlay] DecodeCudaGraphRunner.__init__`, immediately after lines 374-376.

```python
kt_sizes = {bs * self.captured_req_width for bs in self.capture_bs}
assert min(kt_sizes) > 0
assert self.max_bs * self.captured_req_width in kt_sizes
logger.info("KT permanent capture token sizes: %s", sorted(kt_sizes))
```

- **Proposed:** Also expose a kt API that returns registered sizes and assert equality after registration.

### 7.10 Page-pool sealing — needs benchmark before enabling

**Proposed insertion site:** `[overlay] flash_mla_sm120.py`, `_grow_persistent`, before creating a larger tensor at line 557.

```python
if prev is not None and buffers.get("flash_mla_sm120_pools_sealed", False):
    raise RuntimeError(
        f"SM120 page-split buffer {key} grew after pool reservation/capture"
    )
```

- **Proposed:** Set the seal only after every pool is known and reservation completed. Benchmark startup/capture ordering before production.

## 8. Proposed stable interfaces

### Goals

- **Proposed:** SGLang should own scheduling, forward modes, graph admission, and TP combination.
- **Proposed:** The CPU-expert provider should own native tasks and all buffers handed to them until an explicit completion token retires.
- **Proposed:** Engram storage should own file mappings/page-cache hints; SGLang should own GPU staging and graph-visible tensors.
- **Proposed:** Neither provider should inspect arbitrary SGLang global state or direct environment variables after construction.

### CPU-expert protocol

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class ExpertBatchShape:
    request_count: int
    token_count: int
    hidden_size: int
    top_k: int
    forward_id: int
    layer_id: int
    mode: Literal["decode", "prefill", "target_verify"]

@dataclass(frozen=True)
class CaptureContract:
    token_buckets: tuple[int, ...]
    graph_padding: bool
    max_in_flight: int
    buffers_are_address_stable: bool

@dataclass
class ExpertSubmission:
    output: torch.Tensor
    completion: "CompletionToken"
    buffer_lease: "BufferLease"
    deferred_for_layer: int | None

class CpuExpertProvider(Protocol):
    def reserve(self, contract: CaptureContract) -> None: ...
    def validate(self, shape: ExpertBatchShape) -> None: ...
    def submit(
        self,
        shape: ExpertBatchShape,
        hidden: torch.Tensor,
        compact_ids: torch.Tensor,
        weights: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> ExpertSubmission: ...
    def wait(self, submission: ExpertSubmission, stream: torch.cuda.Stream) -> torch.Tensor: ...
```

- **Proposed:** `BufferLease` owns all pinned/device tensors referenced by native callbacks or graphs; it cannot be released until `CompletionToken` is complete and all graph owners are destroyed.
- **Proposed:** `reserve` must happen before autotune/capture and must reject an unregistered capture token count.
- **Proposed:** Deferral returns an explicit `deferred_for_layer` rather than mutating a class dictionary; SGLang carries it to the next layer with the same `forward_id`.
- **Proposed:** Rank ownership is a constructor field (`owner_rank=0`) and the protocol declares whether output is rank-local, replicated, or awaiting TP reduction.

### Streamed-weight export protocol

```python
@dataclass(frozen=True)
class ExportLayout:
    experts: int
    hidden_size: int
    intermediate_per_rank: int
    tp_size: int
    weight_format: Literal["mxfp4"]
    scale_format: Literal["bf16_e8m0_bits"]

class ExpertWeightExporter(Protocol):
    layout: ExportLayout
    def submit_layer(self, layer_id: int, host_slot: int) -> "CompletionToken": ...
    def wait_layer(self, token: "CompletionToken") -> None: ...
```

- **Proposed:** SGLang owns host-slot state transitions (`FREE -> EXPORTING -> READY -> DMA -> FREE`) and TP barriers; kt owns only writing the declared layout.

### Engram protocol

```python
@dataclass(frozen=True)
class EngramLookup:
    forward_id: int
    layer_id: int
    row_ids: torch.Tensor
    capture_mode: Literal["eager", "breakable_replay"]

class EngramProvider(Protocol):
    def prefetch(self, lookup: EngramLookup) -> "CompletionToken": ...
    def materialize(
        self, lookup: EngramLookup, stream: torch.cuda.Stream
    ) -> tuple[torch.Tensor, "BufferLease"]: ...
```

- **Proposed:** The graph contract says `materialize` runs only at an eager break and copies into a stable bridge tensor owned by BCG; the provider never returns a graph-captured mmap-derived pointer.

### Ownership and ordering diagram

```diagram
┌──────────────────────────────┐
│ SGLang forward orchestrator  │
│ shape + mode + forward_id    │
└──────────────┬───────────────┘
               │ explicit contracts
      ┌────────┴────────┐
      │                 │
┌─────▼────────────┐  ┌─▼────────────────┐
│ CPU expert       │  │ Engram provider  │
│ provider         │  │ mmap + prefetch  │
│ native tasks     │  └─┬────────────────┘
│ pinned leases    │    │ materialized rows
└─────┬────────────┘    │ at eager break
      │ output +        │
      │ completion      │
┌─────▼─────────────────▼──────┐
│ SGLang-owned stable GPU      │
│ buffers / BCG bridge tensors│
└──────────────┬───────────────┘
               │ stream events + TP combine
┌──────────────▼───────────────┐
│ next layer / graph segment   │
└──────────────────────────────┘
```

- **Proposed:** CUDA stream semantics are “producer records completion; consumer stream waits.” Host callbacks may use raw pointers only while a lease is live.
- **Proposed:** Graph semantics are “all addresses captured by a graph are immutable and strongly owned until graph destruction.” Growth after sealing is an error, not an allocator policy.
- **Proposed:** This boundary can first wrap the current implementation without changing kernels; native lowering should come only after behavior and performance match.

## 9. What to do now versus later

### Do now: correctness before more architecture work

1. **Proposed:** Stop new speculative, routing, or native-migration features until the bug-C patch's C4 regression is isolated. Benchmark promotion and retirement independently after the live run ends.
2. **Proposed:** Enforce the D>=5 prohibition in startup validation and keep D=4 behind E24 finite/soak gates (`docs/TROUBLESHOOTING.md:101-103`).
3. **Proposed:** Add the low-risk capture-time assertions: no temp allocation while capturing, every capture size permanent, and no captured tuple retirement.
4. **Proposed:** Move kt capture-size reservation before any autotune forward; log and compare requested, registered, and captured token sizes.
5. **Proposed:** Add exact shape/range assertions at the SGLang-to-kt boundary and explicit topology assertions for rank-0 ownership.
6. **Proposed:** Correct staged E23 qualification language so bug C is “correctness fix with unresolved performance regression,” not closed production readiness.
7. **Proposed:** Add direct provenance links for overlay commits `b457c4e6`, `4685955e`, and `07f4aef0`; retain their behavior-level E23 evidence.
8. **Proposed:** Build deterministic, non-server tests for buffer identity/lifetime, deferred-state adjacency, page-pool sealing, and graph/eager output parity.
9. **Proposed:** Finish E24 with a deterministic D>=5 reproducer, a D=4 mixed-shape soak, and request-level NaN failure handling before broad release claims.

**Erratum 2026-09-23:** item 1 and the related qualification wording above predate the same-prompt gate. The lower patched C4 rate is the correct cost of computing all four rows, not an unresolved regression. Keep architecture projections frozen until C3/C4 assumptions are recomputed, and make the every-captured-bs same-prompt gate immediate work.

### Do later: refactor only after experiments stabilize

1. **Proposed:** Introduce the explicit `CpuExpertProvider`, `BufferLease`, and shape/capture dataclasses as adapters around current code.
2. **Proposed:** Replace class globals and loader monkey-patching with runner-owned state and explicit loader arguments.
3. **Proposed:** Separate streamed prefill into a coordinator with visible buffer states and begin/end-forward lifecycle.
4. **Proposed:** Consolidate flags into one typed registry while preserving current environment aliases.
5. **Proposed:** Normalize experiment metadata into manifests and generate duplicated summary tables.
6. **Proposed:** Replace BCG's arbitrary object recursion with an explicit graph-output protocol.
7. **Proposed:** Revisit rank-balanced CPU-expert/Engram ownership only after the current rank-0 protocol has assertions and stable measurements.

### Explicit non-recommendation

- **Proposed:** Do **not** perform a broad rewrite while E24 and bug-C performance are active investigations. The present system has narrow but measured behavior; a rewrite would simultaneously change ownership, graph capture, native callbacks, rank semantics, and performance, destroying the discriminators needed to find the current defects.
- **Inferred:** The safest path is “make contracts executable, then extract interfaces, then change ownership.” Reversing that order risks reproducing the same bugs behind cleaner abstractions.

## Review bottom line

- **Measured:** The highest-risk code is not the large kernels but the small lifetime and coordination layers around them: kt static pinned tuples, SM120 persistent page buffers, BCG bridge tensors, class-global deferral, and rank-0 orchestration.
- **Measured:** The qualified baseline depends on several non-default or overlay-specific choices: breakable graphs, file-backed Engram, rank-0 kt, compact experts, D=4, streamed prefill, SM120 tuning, graph padding, and fixed capture/concurrency ceilings.
- **Measured:** The overlay has 16 commits; 13 have direct SHA-linked recipe evidence and 3 have only behavior-level evidence.
- **Measured:** This review identified 24 invariants requiring stronger assertions or boundary documentation.
- **Inferred:** Maintainability will improve most from explicit ownership, dimensioned shape types, startup compatibility validation, and deterministic boundary tests—not from immediately moving more Python into native code.

## 10. Review pass 1 — after the T3.18 / E30 verdict (2026-09-24)

Standing task T3.16i: refresh the hotspot map and debt register after each stage. Scope of this pass: overlay commits since the review (`cad73914`, `8ad514ed`, `7e41317e`), kt patch `0002-experts-base-capture-assertions.patch`, `scripts/serve-hybrid.sh` validation, and the B2 prototype's residue. Method as §1 (read the diffs, `git log da64c5cb..HEAD`, `scripts/tests/run-kt-buffer-tests.sh`, the launcher's `fail_config` block). Labels as §1.

### 10a. Debt register status

| ID | Status | Evidence |
|---|---|---|
| C1 | **Closed (reclassified)** | Erratum 2026-09-23: the C4 drop was the correction of silent row corruption, not a regression. Qualified C4 55–58 summed after E28/E29 tiers. |
| C2 | **Closed** | T3.16b: `serve-hybrid.sh` refuses `KT_MAX_DEFERRED ≥ 5` unless `KT_ALLOW_UNSAFE_DEFERRAL=1`. D=5 no longer crashes (T3.15) but fails the NLL gate, so the guard now protects the quality gates. |
| C3 | **Half closed** | kt patch 0002 asserts no allocation during capture and cached-tuple shape on every hit (§7.3, §7.2). Forward-boundary emptiness and adjacent-layer identity (§7.5, §7.6) are still unasserted — carried into T3.19. |
| C4 | Open | `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` remains the launcher workaround; no forward-mode gate in code. Speculation is off by default (E23 closed), which lowers the exposure but not the debt. |
| C5 | **Half closed** | `838490a6` never frees captured SM120 buffers and pre-sizes from pool geometry. Post-capture growth is still survived, not rejected. |
| C6 | **Closed for known-bad combinations** | T3.16b: fit refusals (experts/chunk/DSpark), `DSPARK_K` range, B2 constraints, one `resolved:` line per launch. `EXTRA_*` word splitting is unchanged by design (escape hatch). |
| C7 | **Closed** | E23 records amended; erratum rows in BENCHMARKS, EXPERIMENTS, this review. |
| C8 | **Closed** | T3.16c (`8ad514ed`): capture sizes registered before the autotune forward; captured ⊆ registered ⊆ cached asserted after decode capture; 8/8 `scripts/tests/` pass. |
| C9 | ~~Open — **now the gating item for T3.19b**~~ **Closed 2026-09-24 (T3.16d)** — `check_moe_topology` asserts kt wrapper present iff `tp_rank == 0`, MoE-TP == model-TP, `moe_ep == 1`, and the B2 subset bounds, once per layer after weight load; the zeros path is derived from that (no wrapper off rank 0) rather than checked per step. T3.19b no longer blocked. Original text: Rank-0 zeros path still unasserted. B2 (`7e41317e`) added a *second* per-rank contract (every rank computes its own B2 slice, only rank 0 runs kt), so the topology assertion must now name both. |
| C10 | Open | E24 per-request NaN rejection not proven; T3.13. |
| C11 | **Closed** | `verify-batch-consistency.py --same-prompt` runs at every captured width in every serve since 2026-09-23 (E28, E29, E30 all PASS). |
| R1 | **Worse** | `KTEPWrapperMethod` grew by ≈ 300 lines (B2: `_B2HostPool`, `_B2UvaExperts`, `_uva_view`, `remap_topk_ids_for_b2`, `_b2_stream`, `submit(num_kt_experts)`). See R9. |
| R2–R4, R6–R8 | Open, unchanged | — |
| R5 | **Partly addressed** | The launcher's `resolved:` line is the first single-source view of every knob; the typed registry is still T3.19d. |

### 10b. Hotspot map additions

| # | Claim | Area and location | Why accidental complexity is high | Blast radius | Baseline load-bearing? |
|---:|---|---|---|---|---|
| 16 | **Measured** | B2 direct-UVA experts: `[overlay] kt_ep_wrapper.py` (`7e41317e`, +304/−10) | A third stream per device held in a process-global dict (`_B2_STREAMS`), a host pool sized by `kt_config.num_layers`, a UVA device-pointer view built through a private `__cuda_array_interface__` class, a token-count gate (`KT_B2_MAX_TOKENS`) that changes *which* engine computes an expert per forward, and a `submit()` signature that now carries `num_kt_experts`. Correctness was proven only at bs = 1 decode and bs = 4 with B2 gated off. | Off by default → zero blast radius when `KT_B2_UVA_EXPERTS=0`; when on, expert-ID remapping, the combine point, the streamed-prefill export path (which it reuses), and graph capture of a forked stream. | **No.** Rejected as a lever (E30). |

Concentration of risk (update): hotspot 16 adds a fourth instance of the §2 "implicit ownership" pattern (a stream owned by a module dict, not the runner) and a second instance of "which engine owns this expert is decided per forward" (after D=4's per-layer deferral). Neither is load-bearing today; both are exactly the kind of state T3.19c (runner-owned graph/stream state) is meant to absorb.

### 10c. Decision — B2 code: debt or template?

**Decision (this pass): template-then-retire, not keep-as-feature.**

- Keep in the tree until T3.19 lands: `_b2_stream` / `fork` / `join` (the per-device side-stream pattern, proven graph-replayable in E30 and needed by T3.20 Engram staging) and `_uva_view` (stock grouped GEMM on pinned host memory — the shortest path to a staging-free Engram gather experiment).
- Retire in the T3.19 close-out unless T3.20 has adopted them by then: `_B2HostPool`, `_B2UvaExperts.run`, `remap_topk_ids_for_b2`, the `use_b2` branch in `apply`, the `num_kt_experts` parameter on `submit()`, and the launcher knobs `KT_B2_UVA_EXPERTS` / `KT_B2_MAX_TOKENS` with their three `fail_config` checks. Git history (`7e41317e`) and `evidence/t318-b2-prototype/` remain the durable record; a rejected lever does not need a live code path.
- Rationale: R1 is already the highest-severity refactoring debt; carrying 300 off-by-default lines through the T3.19 extraction of `CpuExpertProvider` means writing an adapter for a path no configuration uses. The reusable parts are ≈ 60 lines and belong in the runner-owned stream state T3.19c introduces, not in the wrapper.
- New register entry **R9 (Medium):** B2 residue as above; smallest safe fix = the retire list, executed as one overlay commit after T3.19c, with `run-kt-buffer-tests.sh` and one serve + `verify-batch-consistency` run.

### 10d. Overlay commit traceability (update)

`git log --reverse --oneline da64c5cb..HEAD` now contains 19 commits. The three previously unlinked SHAs (`b457c4e6`, `4685955e`, `07f4aef0`) are named in `PROVENANCE.md` with their E23 R1/R3 roles (closes "Do now" 7 / T3.16f). New since the review, all linked: `cad73914` (T3.16a, TROUBLESHOOTING "operation not permitted when stream is capturing"), `8ad514ed` (T3.16c, `scripts/tests/`), `7e41317e` (T3.18, `evidence/t318-b2-prototype/`, BENCHMARKS E30). **19 of 19 linked.**

### 10e. "Do now" scorecard

| § 9 item | Status |
|---|---|
| 1 freeze until C4 isolated | Closed by erratum (reclassified) |
| 2 D ≥ 5 startup gate | Done (T3.16b) |
| 3 capture-time assertions | Done (kt patch 0002; §7.2–7.4); §7.5/7.6 → T3.19 |
| 4 capture-size reservation before autotune | Done (`8ad514ed`) |
| 5 boundary shape/topology assertions | Done (T3.16d, overlay `b93cb915`: `kt_contract.py`, 14 CPU tests, serve gate bit-identical NLL) |
| 6 E23 wording | Done |
| 7 provenance SHAs | Done |
| 8 deterministic non-server tests | Partly: 8 kt-buffer tests exist; deferred-adjacency and graph/eager parity → T3.16e |
| 9 E24 close | Open (T3.13) |

### Pass 1 bottom line

- **Measured:** 7 of 11 immediate-correctness items are closed or reclassified; the open ones (C4, C9, C10 and the C3/C5 halves) are all "assert or gate a boundary" items, and C9 is the direct prerequisite for T3.19b/T3.22. *(Update 2026-09-24: C9 closed by T3.16d; remaining open: C4, C10, halves of C3/C5.)*
- **Measured:** the only new complexity since the review is the B2 prototype, and it is off by default. It goes out with T3.19 except for two reusable pieces.
- **Inferred:** the sequencing rule from §9 still holds — contracts (T3.16d, done 2026-09-24) → seams (T3.19) → ownership (T3.22). Nothing measured since the review argues for moving native work earlier.

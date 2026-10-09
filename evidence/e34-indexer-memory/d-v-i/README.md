# E34d-v (i) — persistent FlashInfer CUTLASS fused-MoE workspace (2026-10-09)

Overlay `2323052b` (this change, on top of e-i's `b6dc5173`), image `dsv41-kt:recipe`, launcher
default 219 / 165 + chunk 8192, BIOS slot 3, 300 W, D=4, spec off. Three serves, ~45 min:
64K × 4 at the launcher defaults (gates), then the two T3.29 rungs that failed on this binder —
1M × 1 (`1m/`, `MEM_FRACTION_STATIC=0.962`) and 262K × 4 (`262k-4slot/`, `ALLOW_UNTESTED_FIT=1`).
Comparators: `evidence/t328/default-219-8192/` (64K × 4 on the same default, NLL corpus and 16K
prefill), `evidence/e34-indexer-memory/d-ii/` (64K × 4 stress), `evidence/t329/1m-1slot/` and
`evidence/t329/262k-4slot/` (the failed rungs, overlay `b6dc5173` / `bc73ef8d`).

Question (EXPERIMENTS Part d-v (i)): the 1M × 1 and 262K × 4 rungs died on a 980 MiB
`cutlass_fused_moe` workspace that FlashInfer allocates per call from the caching allocator — the
1M failure with 1.87 GiB reserved-but-unallocated, i.e. fragmentation, not bytes. Does allocating
that workspace once per device before graph capture and handing it to every call remove the
binder, bit-exactly and at no decode or prefill cost?

## Finding that changed the protocol

The pinned FlashInfer 0.6.18 `cutlass_fused_moe` accepts `workspace_buffer=` and documents
`cutlass_fused_moe_workspace_size(...)` as monotonic in the row count, and the binding checks the
caller's buffer (int8/uint8, 1-D, contiguous, 128-byte aligned, `size >= need`; a short buffer is a
Python `RuntimeError`, not a kernel fault). So **no row tiling is needed**: one buffer sized for the
chunk ceiling serves every smaller call. Measured: the size is independent of `inter_size` (same
bytes at 144 / 1152 / 2304) and depends on rows, experts and `top_k`; the DSV4 TP2 slice (hidden
5120, inter 1152, 219 experts, top_k 6, 8192 rows) needs **979.6 MiB**, the crash's "980 MiB".

## Change (overlay `2323052b`)

`moe_runner/flashinfer_cutlass.py`: a module-level `_MoeWorkspace` per indexed `torch.device`,
allocated at the first `fused_experts_none_to_flashinfer_mxfp4` call (this happens during
FlashInfer autotune at startup, before capture — `launch-64k4.txt` 04:07:02 on both ranks) and never
resized; raises if first called under stream capture. Row ceiling = env override or
`max(max_prefill_buffer_tokens() or schedule.max_prefill_tokens, schedule.max_running_requests,
rows)` (the `n_gram_embedding.py` precedent). Calls with more rows than the ceiling, or a second
(hidden, experts, top_k) shape that does not fit, fall back to FlashInfer's per-call allocation
with one logged warning — chosen over raising so a mis-sized ceiling degrades to the old behaviour
instead of killing the server. Knobs in `environ.py`: `SGLANG_FLASHINFER_MOE_PERSISTENT_WORKSPACE`
(default on; kill switch) and `SGLANG_FLASHINFER_MOE_WORKSPACE_MAX_ROWS`. Decode runners and the
A2A path are untouched.

`scripts/tests/test_moe_workspace.py` 4/4 in the image (`run-moe-workspace-tests.sh`, GPU 0):
workspace size monotonic in rows; size matches the binding including a 128-byte-short buffer
being rejected; outputs bit-equal with and without the buffer at rows 1 / 5 / 64 / 777 / 2048 plus
an overflow call at 2049 that warns once and leaves the buffer's `data_ptr` unchanged, and the
kill switch; capture guard via a monkeypatched `is_current_stream_capturing`. Mutant (size the
buffer for `top_k = 1`): the equality tests fail with FlashInfer's "workspace_buffer too small".

## Results

### 64K × 4 on the launcher defaults (`launch-64k4.txt`, `launch-container-64k4.txt`)

| measure | t328 / d-ii (`bc73ef8d`) | d-v (i) (`2323052b`) | gate |
|---|---|---|---|
| "Capture ... end" avail mem | 5.56 / 5.51 GB | **4.61 / 4.55 GB** (−0.95 GB: the buffer is now resident at rest instead of cached after the first prefill) | — |
| prefill 16K / 32K (`bench-prefill.py --runs 3 --seed 777`) | 1,504 / — (t328 16K) | **1,512 / 1,580** | ≥ comparator: met |
| NLL, 65,653-token corpus vs `evidence/t328/default-219-8192/corpus/` | — | **0 of 65,653 moved** (bit-identical) | 0 moved: met |
| C1 (`bench-decode.py` off6, 6 runs) | 41.5 / 39.6 wall | **41.50 / 39.59 wall**, TTFT 0.619 s | within noise: met |
| stress 4 × 62K (`stress-kv-full.py --seed 31`) | min free 2,339 / 1,302 MiB (d-ii) | all 4 finished, wall 181.7 s, min free **1,119 / 1,062 MiB** | ≥ 1 GB: met |
| needles (`verify.py --context 65536`) | 6/6 | **6/6 at 55,206**, TTFT 32.6 s; ALL PASS | met |
| health (`health-hits.txt`), EDAC (`edac.txt`) | 0, 0/0 | **0 NaN / Xid / Traceback, 0/0** | met |

The first 16K run on the fresh server took 28.1 s end to end (the other two 10.6 / 10.4 s): the
scheduler's own log shows that prefill at the normal ~10 s (04:10:29 → 04:10:40), so the extra
~18 s was spent before the scheduler saw the request (first-request tokenizer path), not in the
MoE. The median, not the first run, is the reported figure, as on every prior rung.

### 1M × 1 retry (`1m/`, fraction 0.962, pool 1,064,960 tokens) — **fail, on a new binder**

Capture-end avail 3.25 / 3.17 GB (was 4.19 / 4.11). Prefill ran at 1,500 tok/s per chunk to ~360K,
1,330 at ~480K, 1,150 at ~640K — **through and well past the ~306K prefix where the old run died** —
and OOMed at the ~0.64 pool mark (≈ 680K prefix, 04:32:33) on a **332 MiB** allocation in
`deepseek_v4_memory_pool.get_index_k_fp4` (`crash.log`; called from
`_low_ratio_index_topk_dense` → `get_low_ratio_index_k_fp4`), with 301 MiB free, 92.42 GiB
allocated and 1.20 GiB reserved-but-unallocated. `vram-trace.txt` (30 s samples) shows free VRAM
at 10–30 MiB on both cards from ~40K of prefix onward — the run lived inside the allocator's
cache the whole way.

The 332 MiB is exactly the int64 index tensor the gather builds: `payload_cols = (off * 64) +
arange(64)` is `[n, 64]` int64 = 512 B per prefix token (348,127,232 B / 512 = 679,936 rows), and
`scale_cols` adds `[n, 4]` int64; the gathered payload itself is 64 B per token. Per low-ratio
layer and per chunk this materialises ≈ 610 B × prefix of transient (0.6 GiB at 1M) to read 68 B ×
prefix of keys. It is T3.31 idea 9's "(num_tokens, 1, 512) gather workspace", struck on 2026-10-08
pending exactly this evidence; the fix (index the `[pages, page_size, 64]` view with two `[n]`
index tensors, ~16 B per token, bit-exact by construction) is one function. Registered as
**E34d-v (ii)**.

### 262K × 4 retry (`262k-4slot/`, fraction 0.96, same 1,064,960-token pool) — **fail, earlier**

Capture-end avail **2.92 / 2.86 GB** (was 3.88 GB; bs-4 capture costs 0.24 GB and the buffer is
resident). The very first 8192-token chunk OOMed (04:38:52 – 04:39:00, before the first
"Prefill batch" line): the allocator retried 320 / 400 / 640 MiB requests after flushing its cache
and then failed a 234 MiB `int32` temporary in `kt_prefill.py:329 load` (the CPU-tier GPU-prefill
scale conversion), with 183 MiB free and 818 MiB reserved-but-unallocated. The old run reached
~190K prefix because the workspace was cached, not resident; the total bytes are the same, the
buffer is just paid before the first chunk instead of at it. **The chunk-8192 transient set minus
the workspace is ≈ 3 GB and does not fit in 2.9 GB** — the "other chunk-proportional transients"
E34d-iv counted (3.2 GB peak at 8192) and that Part d-v's timeline step was written to name.

## Verdict — **the fragmentation binder is gone; 1M × 1 and 262K × 4 each now fail on the next one**

- Bit-exact and free at 64K × 4: NLL 0 moved, prefill +0.5 % / 32K 1,580, C1 unchanged, stress and
  needles pass, health 0. The cost is visible rather than new: 0.95 GB per rank moves from
  "cached after the first large prefill" to "resident after autotune", so capture-end avail mem
  reads 0.95 GB lower on every profile and the stress min free at 64K × 4 reads ~1.2 / 0.24 GB
  lower than d-ii (the d-ii peak was taken mid-prefill with the workspace cached; same bytes).
- 1M × 1 moved from ~306K to ~680K of prefix and from a fragmentation failure to a prefix-linear
  one. The projected total transient at 1M with E34d-v (ii) in place is the 512K high-water
  (≈ 4.2 GB) against 3.25 GB after capture **plus** the ~1.2 GiB the allocator still holds as
  reserved-but-unallocated at the OOM, so (ii) is a coin flip again, not a sure pass; idea 8
  (`expandable_segments` enabled after capture, now registered as T3.31's fallback) is the lever
  for that last gigabyte if (ii) is not enough.
- 262K × 4 cannot be reached by removing per-call allocations alone: with the buffer resident the
  rung needs ≈ 3 GB of chunk-8192 transient headroom and has 2.9. Options, none chosen here:
  the Part d-v timeline (name the 320 / 400 / 640 MiB transients and shrink or tile the largest),
  chunk 4096 for the rung only (halves the transient set, −30 % prefill, needs its own
  qualification), or 212 / 172 placement (+2.3 GB, owner's named fallback, −2 % decode).

Not run: idle C4 at 64K × 4 (unchanged code path on decode; C1 is the decode gate), 128K / 262K /
512K × 1 (the buffer makes them 0.95 GB tighter at rest but their prefill peaks are unchanged —
the 512K rung's 1,091 / 1,018 MiB min free was measured with the workspace cached in the same
bytes; re-run on the next rung pass, not speculatively).

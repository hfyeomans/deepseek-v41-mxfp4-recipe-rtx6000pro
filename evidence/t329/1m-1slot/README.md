# 1M × 1 rung on the 219 / 165 + chunk 8192 default — FAIL (2026-10-09)

Overlay `b6dc5173` (E34e e-i), image `dsv41-kt:recipe`, launcher default 219 / 165, chunk 8192,
`CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962` (`launch.txt`,
`manifest.json`), BIOS slot 3, 300 W, D=4, spec off. One serve, ~10 min. Run straight after
E34e e-i, which had made the rung a ~0.1 GB coin flip (`../../e34-indexer-memory/e-i/README.md`).

## What happened

- Pool 1,064,960 tokens; **4.19 / 4.11 GB available after graph capture** (rank 0 / 1; nvidia-smi
  4,007 / 3,930 MiB free). 512K × 1 on the same default had 5.32 GB after capture and a min free of
  1,091 MiB during its stress, i.e. a prefill high-water of ≈ 4.2 GB — hence the coin flip.
- `stress-kv-full.py --prompt-tokens 1046528 --seed 31`: prefill ran at 1,600–1,694 tok/s per
  chunk (`crash.log`, `input throughput`), free VRAM was down to 383 / 306 MiB by ~257K of prefix,
  and the **37th chunk (~306K of prefix) raised `CUDA out of memory. Tried to allocate 980.00 MiB`**
  from `flashinfer/fused_moe/core.py cutlass_fused_moe → FusedMoeRunner::getWorkspaceInfo`
  via `moe_runner/flashinfer_cutlass.py:412 fused_experts_none_to_flashinfer_mxfp4` — the same
  per-call `cutlass_fused_moe` workspace that killed T3.29's 262K × 4 at ~190K prefix. Both ranks,
  same chunk. The scheduler exited; the stress script got HTTP 500 (`stress-kv.txt`).
- At the OOM: **499 MiB free, 91.56 GiB allocated by PyTorch, 1.87 GiB reserved but unallocated,
  48 MiB in graph pools.** So 2.36 GiB were nominally available and the 980 MiB request still
  failed: the cached blocks were fragmented below 980 MiB contiguous. This is a fragmentation
  failure of a per-call workspace, not a shortfall of total VRAM.
- EDAC ce/ue 0/0; 0 Xid; no NaN before the OOM.

## Reading

The rung is blocked by the chunk-proportional transient set that e-ii / e-i identified as the
peak owner, and specifically by its largest member being re-allocated from the caching allocator
every chunk. Two fixes already registered apply directly:

1. **E34d-v (i)** — caller-owned fused-MoE workspace (`cutlass_fused_moe(workspace_buffer=)`,
   present in the image's FlashInfer 0.6.18), one persistent buffer per device allocated *before*
   graph capture with an independent row tile. Removes both the per-chunk 980 MiB allocation and
   its fragmentation exposure; also the T3.29 262K × 4 binder. This is the fix to pull forward.
2. **T3.31 idea 8** — `expandable_segments` toggled on *after* capture (E34d showed it breaks the
   custom all-reduce IPC *at* capture). Now admissible by its own gate: the failing allocation is
   outside the graph pool. A fallback if (1) stalls.

The placement fallback for the rung alone is 212 / 172 with `ALLOW_UNTESTED_FIT=1` (+2.3 GB,
~2.5 % C1); owner decision, not taken here.

Files: `launch.txt`, `manifest.json` (the launcher's, copied from the manifest cache; no stamped
evidence was produced), `crash.log` (full container log, redacted), `stress-kv.txt` (client side).

# E34d — prefill chunk size at the 64K x 4 deployment profile (2026-10-08)

Protocol: [`plan/EXPERIMENTS.md` E34](../../../plan/EXPERIMENTS.md), part d. Task T3.26.
Overlay `d6811728` (E34b-ii), 226 / 158, D=4, spec off, `MEM_FRACTION_STATIC=0.96`,
`CONTEXT_LENGTH=65536`, 4 slots. Ladder registered: `CHUNKED_PREFILL_SIZE=8192`, then 16384, go bar
>= 10 % prefill gain at <= 0.5 GB of stress margin. **Result: 8192 does not fit at 226 / 158; the
ladder ends at its first rung. 4096 stays the default.** 16384 was not run (strictly more memory).

The launcher's fit guard (`serve-hybrid.sh`: chunk >= 8192 beside > 208 GPU experts) refused the
run; both serves used `ALLOW_UNTESTED_FIT=1` on purpose, because the guard's "+2.2 GB over 4096"
estimate predates the indexer tiling (E34b) and half of the old transient was the `[chunk, prefix]`
indexer logits, which no longer scale with chunk. The other half does.

## Serve 1 — chunk 8192 (`8192/`)

Ready in the normal time; free after capture **3.25 / 3.19 GB** (4096: 3.48 / 3.42), so the static
cost of the larger chunk is only ~ 0.23 GB. The first request, the 16K prefill bench, killed both
ranks: **CUDA OOM, 980 MiB** requested with 535 MiB free (GPU0) / 476 MiB (GPU1), 913 MiB reserved
but unallocated on each. The allocation is the FlashInfer CUTLASS fused-MoE **workspace**
(`FusedMoeRunner::getWorkspaceInfo` via `cutlass_fused_moe`, called from
`fused_experts_none_to_flashinfer_mxfp4` for the GPU-resident experts), which scales linearly with
the chunk's token count — not the indexer, not KV. `prefill-16k.txt` / `prefill-32k.txt` hold the
client-side HTTP 500 / connection reset.

*Evidence gap:* `8192/launch-container.txt` was copied at ready time and ends before the crash; the
OOM figures above (980 MiB request, 535 / 476 MiB free, 913 MiB reserved-unallocated,
`FusedMoeRunner::getWorkspaceInfo` in the traceback) were read from the live container log, which was
not re-captured before the container was removed. Treat them as operator-transcribed, not archived.

Reading of the budget at chunk 8192: 3.25 GB after capture minus the prefill activations of an
8192-row chunk (hyper-connection residual streams, attention and MoE intermediates, ~ 2x the 4096
figure) leaves ~ 1.4 GB, of which 913 MiB was fragmented into blocks too small for one 980 MiB
request. Even unfragmented, the 4 x 62K stress (which at 4096 leaves 1.54 GB) would have been
~ 0.5 GB short, so the go bar was unreachable at this placement regardless of the OOM.

## Serve 2 — chunk 8192 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`8192-expandable/`)

Tried because 535 + 913 MiB of free-plus-fragmented memory would have covered the 980 MiB
request. The server died **during CUDA-graph capture**: `get_graph_buffer_ipc_meta` in
`sgl_kernel.allreduce` raised `RuntimeError: invalid argument` from the custom all-reduce's
`register_graph_buffers`. Expandable segments are VMM-mapped (`cuMemCreate` / `cuMemMap`) and
cannot be exported as legacy CUDA IPC handles, which the TP custom all-reduce needs to share its
graph buffers between the two ranks. The workaround — disabling the custom all-reduce and using
NCCL for every decode step — taxes the thing the box is tuned for (C1 wall), so it was not tried.
`expandable_segments` is therefore **incompatible with this serving stack** as long as the custom
all-reduce is on; recorded so nobody reaches for it again.

## What would make 8192 fit, and what it would cost (projected, not measured)

The E22 measurement (BENCHMARKS, 208 / 176 placement, pre-tiling indexer): chunk 8192 gave
**1,375 / 1,404 tok/s at 16K / 32K vs 837 / 811 at 4096 (+64 %)** and cost ~ 2.5 GB of free VRAM
after a 32K prefill. Today's 4096 baseline is 1,040 / 1,040 (E34b). Buying ~ 2.2 GB at 0.34 GB per
GPU expert per rank (E28) means ~ 7 fewer GPU experts, i.e. **219 / 165**, projected decode cost
~ 3-4 % of C1 wall (E28: 222 -> 226 was +2 % for 4 experts) for a projected prefill gain of
+30-60 % (the E22 ratio, discounted because the tiled indexer already removed part of what the
larger chunk used to amortise). That is a placement-vs-chunk trade that changes the deployment
default and the owner's call; registered as E34d-ii, not run.

## Health

Both serves: EDAC 0/0, 0 Xid; serve 1 died on a Python-level OOM (clean SIGQUIT), serve 2 on a
capture-time RuntimeError before any request. Containers removed; GPUs 2 / 80 MiB.

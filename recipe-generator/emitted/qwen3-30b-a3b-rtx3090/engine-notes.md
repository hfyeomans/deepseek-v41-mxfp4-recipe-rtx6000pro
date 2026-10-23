# Engine notes - llama.cpp

- Placement: `-ngl/--gpu-layers`, mmap, `-ot/--override-tensor` regex per-tensor placement, `--n-cpu-moe/--cpu-moe` MoE offload, auto-fit `--fit on`.
- Expert caching: draft `--moe-expert-cache N` LRU VRAM cache over host experts (+31% decode; LRU-64 hits 67% - temporal locality beats static hot lists).
- KV cache: `-c` context is the whole lever; FlashAttention `-fa 1` shrinks the KV footprint; KV offload tiers do not exist here - KV is GPU or the request fails.
- Deep-lever carry-over: CUDA-graph-like capture is internal; there is no deferral (no CPU-expert compute path on the GPU engine), no Engram, no DSpark; determinism is upstream's. The CSA2/Engram deepseek41 branch is blocked on the upstream CSA2 merge (ggml-org/llama.cpp#28696) - the parked route (T3.7).
- Speculation: n-gram/DSpark-style drafts are not wired to the MoE offload path; treat any speculative flags as unvalidated on this engine.

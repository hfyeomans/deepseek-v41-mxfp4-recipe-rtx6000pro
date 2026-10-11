# T3.10 / T3.47 — Engram working set, the pinned page cache, and the first-request stall (E40)

2026-10-11, launcher default (219 / 165 experts, el8-mix map, chunk 8192, 262K × 4, D=4, lever on,
`FUSED_FINALIZE=0` + `SORT_DECODE_TOPK=1`), overlay `743f8fa9`, image `dsv41-kt:recipe`.
Question: why did prefill throughput swing 20–35 % between serves of one shape (T3.47), and would a
re-quantized Engram table (E11) help the NVMe-backed row fetch (T3.10, GitHub #2)?

## Files

| path | what |
|---|---|
| `corpus-only.json` | `engram-working-set.py` over the smoke corpus: distinct rows, rows per weight page, 2-/4-gram distinctness, page cache per first-touched token |
| `smoke-replay-262k-4.{json,txt}` | the same tool replaying every prompt the 262K × 4 release smoke sends: pages per layer, total, request-level LRU hit rate at 16 / 32 / 64 / 86 GiB |
| `warmth/` | first serve: `warmth-run1.txt` (first request 23.8 s, Engram cache plateau at ~10 GiB with evictions), `fincore` before/after `posix_fadvise(DONTNEED)`, `warmth-run2-after-dontneed.txt` |
| `smoke-after-dontneed/` | release smoke on that serve after the manual drop (prefill 16K 4,401 / 32K 4,278, stress 262 s, 12/12 PASS) |
| `default-wait-ready/` | fresh plain-launcher serve with the new `wait-ready.py` (drop + prime) — the headline row: prefill 16K 4,248 (TTFT 3.76 s) / 32K 4,232 (7.55 s), stress 4 × 260,096 in 257.2 s, C1 57.36 / C4 124.35 tok/s, needles 6/6 at 255,388 tokens, corpus bit-identical to `t315c/default-flip`, Xid 0, prefill power 487–494 W total |
| `default-full/` | the **committed launcher** (overlay `743f8fa9`, no overrides, no worktree) re-run after `wait-ready.py` landed — the qualification of the shipped shape: 12/12 PASS, C1 59.83 / C4 122.03, prefill 16K 3,849 (TTFT 4.16 s) / 32K 4,050 (7.90 s), stress 279.6 s, needles 6/6, corpus bit-identical, Xid 0. Prefill is ~10 % under `default-wait-ready/` because this serve started with a cold Engram cache (the shard stream evicted it before `wait-ready.py` could drop the shard pages; the drop found 0.0 GiB resident), see T3.50 — the shape is qualified, the headline keeps the warmer serve |
| `prime-serve-a/` | `KERNEL_CACHE_DIR` mount with an **empty** cache: `wait.txt` (ready probe 13.9 s, prime 22.3 s, wait-ready 5m29 wall), `warmth.txt` (first user request 7.8 s) |
| `prime-serve-b/` | the next serve on the **populated** cache: `wait.txt` (ready probe 0.23 s, prime 7.3 s, wait-ready 3m50 wall), `warmth.txt` |

## Findings

1. **Working set is pages, not bytes.** Real text touches ~1.02 Engram rows per 4 KiB weight page, so each
   first-touched token costs ~190 KiB of page cache (12 GiB for 66K tokens); 2-gram columns are 44 %
   distinct, 4-gram 82 %. The full 262K × 4 smoke touches 37.8 GiB per layer, 75.6 GiB both; a request-level
   LRU hits 38 % at 16 GiB of cache and 55 % at 86 GiB. A smaller row encoding (E11 FP4 / q3_K) cannot change
   the page count, which is why E11's A/B is closed negative.
2. **The page cache was pinned by the weight load.** All 48 shards (475 GiB) stream through the page cache
   at load; the ~73 GiB of non-Engram tensor pages (now in VRAM, never read again) stayed resident beside the
   136 GiB shm and held the Engram tables to ~10 GiB with evictions during the first smoke. `fadvise(DONTNEED)`
   on those shards freed 73 GiB in 2.8 s without root; Engram residency then grew monotonically (27 GiB after
   two warm runs, 80 GiB after the stress). Effect on the gates: prefill 16K 3,311 → 4,401 / 4,248 tok/s
   (two serves), 32K 3,480 → 4,278 / 4,232, stress 320 → 262 / 257 s, prefill power ~200–235 → 245–250 W per
   GPU. Shipped as the `wait-ready.py` default (`--keep-shard-cache` opts out). RssShmem was already
   142.7 GB per rank at ready, which rules out shm faults as the cause.
3. **Cold vs warm Engram is a flat +1.7–3.1 s per request** (16K real text 6.9–7.9 s cold vs 4.7–4.8 s
   warm; 32K 9.9–10.7 vs 8.3), independent of prompt length — the first chunk's NVMe fetch that the
   chunk-ahead prefetch cannot hide (registered as T3.49).
4. **The first request of a serve was 23.8 / 25.1 s, not 7 s.** The overlay's `triton_load_watch` logs
   `_hc_mix_reduce_sinkhorn_kernel took 12.2–14.2 s to compile after serving started`, the CPU-tier
   `dedicated runner for 165 experts` is built on that request, and a dozen Triton kernels device-load at
   0.25–0.7 GiB free — all triggered by the first chunk-sized prefill, which the server's 256-token warmup
   never reaches, on a `/root/.cache/sglang` that died with the container. Two fixes:
   - `KERNEL_CACHE_DIR` (registry knob, default `~/.cache/dsv41-recipe/kernel-cache`, empty = ephemeral)
     bind-mounts the Triton (30 MB), CUDA JIT (390 MB), DeepGEMM, FlashInfer-autotune and inductor caches.
     Serve A filled it; serve B reached ready **3m36 after launch vs 5m04** (same 164–169 s weight load, the
     saving is in capture/warmup JIT), its ready probe answered in 0.23 s vs 13.9 s, and its prime ran in
     7.3 s vs 22.3 s.
   - `wait-ready.py` sends one synthetic chunk + 1,000-token prefill after the shard drop (`--no-prime`
     opts out), so the compile lands before the first user request: serve A's first user request (16K real
     text, cold Engram) was 7.8 s instead of 23.8–25.1 s. The prime does not cover every device-load —
     other prompt lengths still trigger `_w8a8_block_fp8_matmul` loads for new tile configs (logged as
     warnings, sub-second, no OOM at 0.16 GiB free) — so the headline "cold" prefill is the cold-Engram
     figure, not the compile.

The container writes the kernel cache as root; clear it with
`docker run --rm -v ~/.cache/dsv41-recipe/kernel-cache:/c alpine sh -c 'rm -rf /c/*'`.

# E36 (1) — what an Engram row lookup costs, by issue mechanism (2026-10-09)

`scripts/bench-engram-advise.py` on the idle box (no server; 246 GB DRAM, layer-1 shard
50–76 % resident across runs as the bench itself warmed rows), inside the recipe image, layer 1's
table (`model-00047`, 384,006,168 rows × 264 B). Per repetition: fresh random row ids, `mincore`
residency, then *issue* (the advise mechanism) and *gather* (`index_select` from the mapping), cold
(first touch of those ids) and warm (the same ids again). Three repetitions, medians. 48 rows per
token = one layer's lookup for the token.

| tokens | mechanism | cold issue | cold gather | cold total | warm issue | warm gather | file |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | pool-16 (served path until T3.20) | 0.69 | 0.36–0.43 | **1.05–1.12 ms** | 0.65–0.68 | 0.008 | `advise-pool-serial-none.json`, `advise-ctypes-bg.json` |
| 1 | one-thread (`mmap.madvise` per page) | 0.06–0.08 | 0.30–0.39 | **0.36–0.47 ms** | 0.021–0.023 | 0.003 | same |
| 1 | ctypes (libc `madvise`, GIL released) | 0.09 | 0.30–0.38 | 0.40–0.47 | 0.05 | 0.003 | `advise-ctypes-bg.json` |
| 1 | none (page faults only) | 0 | 2.2–3.2 | 2.2–3.2 | 0 | 0.003 | all |
| 16 | pool-16 | 1.6–2.7 | 2.6–2.9 | 4.4–5.6 | 0.84–1.16 | 0.02–0.03 | |
| 16 | one-thread | 1.1–1.6 | 2.4–2.6 | 3.5–4.0 | 0.37–0.41 | 0.01–0.02 | |
| 8192 (one prefill chunk, one layer) | pool-16 | 480–650 | 103–129 | **583–780 ms** | **204–212** | 8.9–9.5 | |
| 8192 | one-thread | 422–501 | 88–114 | **510–615 ms** | 189–198 | 8.3–9.4 | |
| 8192 | ctypes | 584–726 | 90–97 | 674–823 | 397–419 | 8.6 | |
| 8192 | bg (ctypes on a helper thread, gather at once, join) | 6 | 551–643 | 557–650 | 5.5 | 394–413 (the join) | |
| 8192 | pmadv (`process_madvise`, 1024 pages per syscall) | 344 | 72 | 416 | **190** | 8.2 | `advise-pmadv-seccomp-unconfined.json` |
| 8192 | none | 0 | 777–1,345 | 777–1,345 | 0 | 8.2–9.6 | |

What it says:

1. **The thread pool was the cost, not the help.** CPython's `mmap.madvise` holds the GIL, so 16
   threads serialise on it and add hand-offs: at 48 rows the pool pays 0.65 ms *warm* for rows that
   are already in the page cache (gather 0.008 ms), one thread pays 0.02 ms. Cold, one thread is
   2.4–3× faster than sixteen at every shape. E18's "16 threads beat 64" was true; "16 beat 1"
   was never measured.
2. **WILLNEED's cost is per page and kernel-side**: ≈ 0.24 µs per page warm, ≈ 0.5 µs cold
   (`madvise_willneed` → `force_page_cache_readahead` per range), the same through `mmap.madvise`,
   through libc via ctypes (2× slower: ctypes call overhead), or batched 1024 pages per
   `process_madvise` syscall (190 ms warm at 8192 tokens — identical to one-thread). Syscall entry is
   not what you pay, so batching does not help; `process_madvise` also needs the container's
   seccomp profile opened (`EPERM` under Docker's default even with `CAP_SYS_NICE`), so it is not
   worth a launcher flag.
3. **One 8192-token prefill chunk touches ≈ 393K rows ≈ 786K distinct pages per layer**: ≈ 0.5–0.6 s
   cold and ≈ 0.2 s warm per layer with one-thread issue, 1.0–1.6 s cold for the two layers ≈ 25–35 %
   of a ≈ 4.3 s chunk, and 0.4 s *warm* ≈ 9 % — the latter pure WILLNEED overhead on rows that are
   resident. Running the issue on a worker thread (`bg`) hides nothing by itself because the
   gather then faults the same pages before the advise reaches them; it only helps when there is
   model compute to hide behind, which is what the served `ahead` mode does (layers 2–13).
4. Page faults alone (`none`) are 2–6× worse than advised reads at every cold shape (the E18
   finding stands), and 0.003 ms warm — the madvise is only worth issuing for pages that are not
   resident, which the kernel knows and we do not without a per-page `mincore` that costs the
   same syscall.

Harness: `docker run --rm --ipc=host -v <hf-cache>/models--deepseek-ai--DeepSeek-V4.1-Flash:/hfm:ro -v $PWD/scripts:/recipe:ro -v <out>:/out --entrypoint python dsv41-kt:recipe /recipe/bench-engram-advise.py --snapshot /hfm/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 --tokens 1,16,8192 --reps 3 --json /out/<name>.json`; the `pmadv` row needs `--security-opt seccomp=unconfined`. No server, no manifest (idle-box microbench).

# T3.36 NUMA / THP / thread placement for the kt CPU tier, with T3.40 (`-march=native` rebuild) in the same window (2026-10-11)

Registered 2026-10-09 from the RonanLabsAI sidecar assessment (#7): E25 measured the fabric roof
(62-63 GB/s per CCD, 125 socket) but not TLB or allocation-policy effects on the *served* kt loop.
Go bar: >= +3 % C1 wall. T3.40 (registered 2026-10-10 from the K2 build): rebuild the `avx512_bf16`
extension with `-march=native` on the same source + patches, gate = NLL bit-identical, then speed.

**Result: both closed, nothing adopted.** No page-size, thread-placement or compiler-flag setting
moves the CPU tier at one layer or in a served A/B; two placements are large *losses*; the
`-march=native` build is bit-identical and no faster. Launcher defaults unchanged.

## Host facts (verified on this box)

- One NUMA node (`numactl -H`: node 0 only), so "NUMA" reduces to CCD placement: 16 cores / 32
  threads, CCD0 = cores 0-7, CCD1 = 8-15, SMT siblings 16-31 (`lscpu -e`).
- `transparent_hugepage/enabled = madvise`, `shmem_enabled = never`. kt's expert arenas are 4 KiB
  pages (`AnonHugePages 0`, `ShmemHugePages 0` on every serve so far).
- No-root THP lever for malloc'd memory: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (glibc 2.39 in the
  image madvises `MADV_HUGEPAGE` on every mmap'd malloc chunk). Verified on the one-layer harness:
  2.79 of the 3.5 GiB arena on huge pages (`anon_huge_gib_after` in the `load` report,
  `test-kt-one-layer.py` now records it).
- **The served arena cannot take THP without root.** Since K2 (T3.39 window 2) the 136 GiB of CPU
  experts live in one `memfd` per layer (`kt_prefill.py`, `MOEConfig.weight_arenas`) mapped by both
  ranks - `Shmem 136 GiB` in `/proc/meminfo` while a serve is up - and shmem huge pages are
  governed by `shmem_enabled`, which is `never` here (root: `echo advise > ...shmem_enabled`
  + `MADV_HUGEPAGE`, or `memfd_create(MFD_HUGETLB)` against `vm.nr_hugepages`). The glibc tunable
  therefore reaches only the malloc'd side of the server (kt scratch / activation buffers, torch
  host allocations): `AnonHugePages` 5.4-6.3 GiB during the `b-thp` serve vs 0 on the others.
  The one-layer matrix (stock allocator, so THP *does* cover the arena there) is the direct test of
  arena THP, and it is null (below).

## One-layer matrix (`bench.sh`, `one-layer/`): page size and thread placement

`test-kt-one-layer.py --bench-cold`, layer 5, 219 GPU / 165 CPU experts, 3 of 6 slots on the CPU
tier, distinct cold experts per token, 40 iters, two reps per setting, medians in ms per layer.
Every token routes its CPU slots to CPU experts (worst case). `one-layer-log.txt` has the run order.

| setting | threads | cpuset | THP GiB (arena) | T=1 | T=2 | T=4 | T=8 | T=64 | T=256 |
|---|---|---|---|---|---|---|---|---|---|
| baseline r1 / r2 | 16 | all | 0 | 0.58 / 0.57 | 1.12 / 1.08 | 2.07 / 2.08 | 3.92 / 3.96 | 23.28 / 23.33 | 43.31 / 43.39 |
| thp r1 / r2 (`glibc.malloc.hugetlb=1`) | 16 | all | 2.79 | 0.57 / 0.57 | 1.09 / 1.06 | 2.01 / 2.03 | 3.97 / 3.91 | 23.22 / 23.15 | 42.91 / 43.31 |
| phys16 r1 / r2 (physical cores only) | 16 | 0-15 | 0 | **7.00 / 6.00** | 6.97 / 7.02 | 6.59 / 9.14 | 11.00 / 7.01 | 40.02 / 39.21 | 57.02 / 56.99 |
| thp-phys16 r1 / r2 | 16 | 0-15 | 2.78 | 6.61 / 6.20 | 8.99 / 6.84 | 6.00 / 6.19 | 6.12 / 6.52 | 39.28 / 39.80 | 57.08 / 56.98 |
| ccd0-8 r1 / r2 (one CCD) | 8 | 0-7,16-23 | 0 | **1.01 / 1.03** | 2.00 / 2.00 | 3.87 / 3.90 | 7.62 / 7.72 | 44.37 / 44.53 | 84.04 / 83.80 |
| ccd0-8-thp r1 / r2 | 8 | 0-7,16-23 | 2.79 | 1.00 / 1.01 | 2.03 / 2.01 | 3.95 / 3.88 | 7.67 / 7.64 | 44.35 / 44.23 | 83.92 / 83.77 |
| T3.40 recipe image r1 / r2 (`../t340/one-layer/`) | 16 | all | 0 | 0.58 / 0.58 | 1.07 / 1.08 | 2.04 / 2.03 | 3.97 / 3.87 | 23.37 / 23.36 | 43.20 / 43.28 |
| T3.40 native image r1 / r2 | 16 | all | 0 | 0.58 / 0.59 | 1.09 / 1.12 | 2.04 / 2.07 | 3.93 / 3.99 | 23.50 / 23.34 | 43.43 / 43.47 |

- **THP on the arena: null** (<= 1 % at every T, inside the rep-to-rep spread). The tier streams
  21.7 MiB per expert sequentially; the TLB walk per 4 KiB page is already hidden behind the DRAM
  stream, and 2 MiB pages change nothing.
- **Pinning the 16 workers to the 16 physical cores (`--cpuset-cpus 0-15`) is 10x worse at T=1
  (6-7 ms vs 0.58)** and still +30 % at T=256. kt's workers busy-spin, and with no SMT sibling to
  float to, the dispatch / launch thread (and the harness's own main thread) is starved for a
  scheduler tick (~6 ms = the visible latency). Same mechanism as T3.39 window 1's H2 (the Engram
  worker out-scheduled by the export threads). The server is not pinned and must not be.
- **One CCD with 8 workers is exactly 2x slower at every T** (1.01 ms T=1, 84 ms T=256): the tier
  is per-CCD-fabric-bound (62-63 GB/s per CCD, E25); both CCDs are needed and already used.
  There is no "local vs interleave" choice on a one-node box; `numactl` has nothing to bind.
- **`-march=native` (znver5) vs the wheel's explicit AVX-512 flag set: identical** (within 1 %).
  The hot loops are hand-written AVX-512 intrinsics and the bound is bandwidth, not codegen.

## Served A/B (`serve-ab.sh`, `serve-ab/`): four serves, same hour, launcher defaults (262K x 4)

Per serve: `wait-ready`, one warm 32K prefill (seed 777), decode C1 (6 runs x 512, distinct
prompts) and C4 (3 runs), prefill 16K / 32K (2 runs, seed 1277), `kt-cpu-per-step.sh` at C1 / C4,
NLL corpus (8 texts, 65,653 tokens) against the 219/8192 anchor, health counts, manifest.
`a2-baseline` is the order control: `b` and `c` both read prefill below `a`, and `a2` (the baseline
again, last) reads exactly what `b` and `c` read, so `a`'s prefill was the first-serve-of-the-day
box state (T3.37's "fresh serve vs warm box" effect, here in the other direction), not a cost of
either variable.

| serve | image / env | C1 tok/s (min-max) | C4 | prefill 16K / 32K tok/s | TTFT 16K / 32K s | kt cpu-s/step C1 / C4 | NLL vs `a` | health / mapping failures |
|---|---|---|---|---|---|---|---|---|
| a-baseline (09:50) | `dsv41-kt:recipe` | 59.74 (58.15-60.49) | 118.67 | 3,683 / 4,292 | 4.34 / 7.45 | 0.326 / 0.734 | - (vs anchor \|z\| 0.26; **0 of 65,653 moved vs `t337/262k4-smoke`**) | 0 / 0 |
| b-thp (10:00) | recipe + `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` (developer profile) | 61.25 (60.63-61.59) | 121.35 | 3,197 / 3,435 | 5.00 / 9.31 | 0.312 / 0.676 | **0 moved** | 0 / 0 |
| c-native (10:13) | `dsv41-kt:native` (`KT_CPU_INSTRUCT=NATIVE`) | 60.43 (60.07-60.94) | 119.94 | 3,147 / 3,414 | 5.08 / 9.37 | 0.330 / 0.705 | **0 moved** | 0 / 0 |
| a2-baseline (10:24) | `dsv41-kt:recipe` | 60.96 (60.76-61.27) | 119.20 | 3,197 / 3,404 | 5.00 / 9.39 | 0.322 / 0.676 | **0 moved** | 0 / 0 |

Against the order control `a2`: THP C1 +0.5 %, C4 +1.8 %, prefill 0 / +1 %; native C1 -0.9 %,
C4 +0.6 %, prefill -1.6 / +0.3 %. The four serves' C1 medians span 59.7-61.3 (+-1.3 %) and C4
118.7-121.4; both variables sit inside that band, under the +3 % go bar, and the kt CPU-seconds
per step are the same to the third digit. Both candidates are bit-identical on the corpus
(`analysis/identity.py`, per-token logprobs, 0 of 65,653 moved), so the NLL gate for T3.40 passes
and there is simply nothing to buy.

## Decisions

- T3.36 closed: no THP / pinning / CCD setting helps; two of them are the largest CPU-tier losses
  measured in the programme (keep the server unpinned, never `--cpuset-cpus`). The served arena is
  on 4 KiB shmem pages and stays there; a root-only `shmem_enabled` lever is not worth a request
  because the one-layer arena-THP test (where THP *did* cover the arena) is null. T1.6 (hugepage
  reservation) is absorbed and closed with it.
- T3.40 closed: `KT_CPU_INSTRUCT=NATIVE` is bit-identical and no faster; the image keeps the
  wheel's flag set (`AVX512`). The build ARG stays in `Dockerfile.dsv41-kt` (one line) so the A/B is
  reproducible; `dsv41-kt:native` is a local tag only. Note for the record: kt-kernel's `setup.py`
  documents a `CPUINFER_NATIVE` env var that nothing reads at the pinned commit; the real switch
  is `CPUINFER_CPU_INSTRUCT=NATIVE` (`-DLLAMA_NATIVE=ON` -> `-march=native`, explicit
  `LLAMA_AVX512_*` flags still forwarded; `../t340/build-native.log` line "ARCH_FLAGS").

## Files

- `bench.sh`, `bench-log.txt`, `one-layer/<setting>-r<rep>.{json,txt}`, `one-layer/meminfo-after.txt`
- `serve-ab.sh`, `serve-ab-log.txt` (a, b, c), `serve-ab-log2.txt` (a2), `serve-ab/<serve>/`
  (launch / ready / decode / prefill / kt-cpu / corpus / health / manifest; `b-thp` and `c-native`
  also `meminfo-during.txt`)
- `analysis/identity.py` - offline per-token identity check between two corpus directories
- `../t340/` - native image build log, `one-layer.sh`, `one-layer/`, `one-layer-log.txt`

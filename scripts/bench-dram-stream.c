// E25: STREAM-style host DRAM bandwidth probe in C/OpenMP (revalidates E1b's torch number).
//
// Build:  gcc -O3 -march=native -fopenmp -o /tmp/bench-dram-stream scripts/bench-dram-stream.c
// Run:    OMP_NUM_THREADS=16 GOMP_CPU_AFFINITY="0 8 1 9 2 10 3 11 4 12 5 13 6 14 7 15"
//           /tmp/bench-dram-stream --pattern read --size-gib 8 --iters 7 --hugepage --json
//
// Patterns and the byte accounting used for GB/s (decimal, 1e9):
//   read         sum over a uint64 array                     bytes = S
//   write        fill a uint64 array (--nt: streaming stores) bytes = S
//   copy         b[i] = a[i]                                  bytes = 2 S  (read + write, like STREAM Copy)
//   triad        c[i] = a[i] + k*b[i], doubles                bytes = 3 S  (STREAM Triad)
//   expert-chunk pool of --chunk-mib chunks; each iteration reads --active-chunks random chunks,
//                one thread per chunk (omp dynamic)           bytes = active * chunk
//   expert-rows  same chunk selection, but every thread splits every chunk (kt-kernel style:
//                one expert's rows spread over all threads), with an OpenMP barrier every
//                --chunks-per-sync chunks (kt's per-layer sync boundary) bytes = active * chunk
//
// S is the array size (--size-gib) and must be far larger than the 2 x 32 MiB L3. All arrays are
// mmap'd anonymous memory, first-touched in parallel by the same threads that later read them,
// and madvise'd MADV_HUGEPAGE (--hugepage) or MADV_NOHUGEPAGE (default; host THP mode is
// "madvise", so the default is 4 KiB pages, which is also what torch got in E1b).
// Every iteration's wall time is reported so the driver can show distributions; the first
// iteration is a warm-up and is flagged as such.
#define _GNU_SOURCE
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void *alloc(size_t bytes, int hugepage) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); exit(2); }
    if (madvise(p, bytes, hugepage ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) perror("madvise");
    return p;
}

static void first_touch_u64(uint64_t *a, size_t n, uint64_t v) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = v + (i & 7);
}

static void first_touch_f64(double *a, size_t n, double v) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) a[i] = v;
}

static uint64_t sum_range(const uint64_t *a, size_t lo, size_t hi) {
    uint64_t s = 0;
#pragma omp simd reduction(+ : s)
    for (size_t i = lo; i < hi; i++) s += a[i];
    return s;
}

static long anon_hugepages_kib(void) {
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return -1;
    char line[256]; long kib = -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "AnonHugePages: %ld kB", &kib) == 1) break;
    fclose(f);
    return kib;
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;
static uint64_t rng(void) {  // splitmix64
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static void shuffle(size_t *ids, size_t n) {
    for (size_t i = n - 1; i > 0; i--) { size_t j = rng() % (i + 1); size_t t = ids[i]; ids[i] = ids[j]; ids[j] = t; }
}

int main(int argc, char **argv) {
    const char *pattern = "read";
    double size_gib = 8.0, chunk_mib = 21.8;
    int iters = 7, hugepage = 0, nt = 0, json = 0, active = 100, per_sync = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--pattern")) pattern = argv[++i];
        else if (!strcmp(argv[i], "--size-gib")) size_gib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--chunk-mib")) chunk_mib = atof(argv[++i]);
        else if (!strcmp(argv[i], "--active-chunks")) active = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chunks-per-sync")) per_sync = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hugepage")) hugepage = 1;
        else if (!strcmp(argv[i], "--nt")) nt = 1;
        else if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--seed")) rng_state = strtoull(argv[++i], NULL, 10);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    size_t S = (size_t)(size_gib * 1073741824.0);
    S &= ~(size_t)4095;
    size_t n = S / 8, n8 = n & ~(size_t)7;
    int threads = omp_get_max_threads();
    cpu_set_t cs; CPU_ZERO(&cs); sched_getaffinity(0, sizeof cs, &cs);
    int allowed = CPU_COUNT(&cs);

    uint64_t *a = NULL, *b = NULL; double *da = NULL, *db = NULL, *dc = NULL;
    double bytes_per_iter;
    size_t chunk = (size_t)(chunk_mib * 1048576.0) & ~(size_t)63, nchunks = 0, *ids = NULL;
    int is_expert = !strncmp(pattern, "expert", 6);
    if (!strcmp(pattern, "triad")) {
        da = alloc(S, hugepage); db = alloc(S, hugepage); dc = alloc(S, hugepage);
        first_touch_f64(da, n, 1.0); first_touch_f64(db, n, 2.0); first_touch_f64(dc, n, 0.0);
        bytes_per_iter = 3.0 * S;
    } else if (!strcmp(pattern, "copy")) {
        a = alloc(S, hugepage); b = alloc(S, hugepage);
        first_touch_u64(a, n, 1); first_touch_u64(b, n, 0);
        bytes_per_iter = 2.0 * S;
    } else {
        a = alloc(S, hugepage); first_touch_u64(a, n, 1);
        bytes_per_iter = (double)S;
        if (is_expert) {
            nchunks = S / chunk;
            if (active <= 0 || (size_t)active > nchunks) active = (int)nchunks;
            ids = malloc(nchunks * sizeof *ids);
            for (size_t i = 0; i < nchunks; i++) ids[i] = i;
            bytes_per_iter = (double)active * chunk;
        }
    }
    long huge_kib = anon_hugepages_kib();

    double *secs = malloc(iters * sizeof *secs);
    volatile uint64_t sink = 0;
    for (int it = 0; it < iters; it++) {
        uint64_t total = 0;
        double t0 = omp_get_wtime();
        if (!strcmp(pattern, "read")) {
#pragma omp parallel reduction(+ : total)
            {
                int t = omp_get_thread_num(), T = omp_get_num_threads();
                size_t lo = n * t / T, hi = n * (t + 1) / T;
                total += sum_range(a, lo, hi);
            }
        } else if (!strcmp(pattern, "write")) {
            uint64_t v = (uint64_t)it + 3;
            if (nt) {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n8; i += 8)
                    _mm512_stream_si512((__m512i *)(a + i), _mm512_set1_epi64((long long)v));
                _mm_sfence();
            } else {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) a[i] = v;
            }
        } else if (!strcmp(pattern, "copy")) {
            if (nt) {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n8; i += 8)
                    _mm512_stream_si512((__m512i *)(b + i), _mm512_loadu_si512((const void *)(a + i)));
                _mm_sfence();
            } else {
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; i++) b[i] = a[i];
            }
        } else if (!strcmp(pattern, "triad")) {
            double k = 3.0 + it;
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; i++) dc[i] = da[i] + k * db[i];
        } else if (!strcmp(pattern, "expert-chunk")) {
            shuffle(ids, nchunks);
#pragma omp parallel for schedule(dynamic, 1) reduction(+ : total)
            for (int c = 0; c < active; c++) {
                size_t lo = ids[c] * (chunk / 8);
                total += sum_range(a, lo, lo + chunk / 8);
            }
        } else if (!strcmp(pattern, "expert-rows")) {
            shuffle(ids, nchunks);
            for (int g = 0; g < active; g += per_sync) {
                int gend = g + per_sync < active ? g + per_sync : active;
#pragma omp parallel reduction(+ : total)
                {
                    int t = omp_get_thread_num(), T = omp_get_num_threads();
                    size_t rows = chunk / 8;
                    for (int c = g; c < gend; c++) {
                        size_t base = ids[c] * rows;
                        total += sum_range(a, base + rows * t / T, base + rows * (t + 1) / T);
                    }
                }
            }
        } else { fprintf(stderr, "unknown pattern %s\n", pattern); return 2; }
        secs[it] = omp_get_wtime() - t0;
        sink += total;
    }

    if (json) {
        printf("{\"pattern\":\"%s\",\"threads\":%d,\"allowed_cpus\":%d,\"size_bytes\":%zu,\"bytes_per_iter\":%.0f,"
               "\"hugepage\":%s,\"anon_hugepages_kib\":%ld,\"nt\":%s,\"chunk_bytes\":%zu,\"active_chunks\":%d,"
               "\"chunks_per_sync\":%d,\"iters_seconds\":[",
               pattern, threads, allowed, S, bytes_per_iter, hugepage ? "true" : "false", huge_kib,
               nt ? "true" : "false", is_expert ? chunk : 0, is_expert ? active : 0, is_expert ? per_sync : 0);
        for (int it = 0; it < iters; it++) printf("%s%.6f", it ? "," : "", secs[it]);
        printf("],\"gbps\":[");
        for (int it = 0; it < iters; it++) printf("%s%.2f", it ? "," : "", bytes_per_iter / secs[it] / 1e9);
        printf("],\"sink\":%llu}\n", (unsigned long long)sink);
    } else {
        for (int it = 0; it < iters; it++)
            printf("%-12s thr=%2d iter %d%s %.4f s  %.1f GB/s\n", pattern, threads, it, it == 0 ? " (warmup)" : "",
                   secs[it], bytes_per_iter / secs[it] / 1e9);
    }
    return 0;
}

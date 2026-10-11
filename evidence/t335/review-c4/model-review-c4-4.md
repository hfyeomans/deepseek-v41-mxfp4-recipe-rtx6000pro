# Code review — kernel experiments and tooling (C4.4)

Scope: everything under `kernels/` (e13, e20, t321), the named scripts (`bench-engram-gather.py`,
`feasibility-calc.py`, `trace-prefix-memory.py`, `greedy-logprobs.py`, `verify-nan-rejection.py`,
`download-model.sh`), and other `scripts/` files that touch CUDA/torch/the model files. Read at HEAD
`fdc2101`. No tracked files were modified.

Method: full read of the named files plus every non-test script that imports torch or touches the
checkpoint, cross-checked against the local checkpoint (`model-00047-of-00048.safetensors` header:
`layers.1.engram.embed.weight` F8_E4M3 `(384006168, 256)` + F8_E8M0 scale `(384006168, 8)` =
264 B/row), the SGLang overlay, and the evidence JSONs the docs cite. Findings are ranked; style
nits are excluded. "Verified correct" items at the end record what was checked and cleared.

---

## High

### 1. `scripts/bench-engram-gather.py` destroys the file it is pointed at

`build_table` L42–52 and `main` L159–160.

```python
if path.exists() and path.stat().st_size == total:
    return path, total          # L42-43: silently reuses ANY same-sized file
with path.open("wb") as fh:     # L44: truncates a foreign file of the wrong size
...
if args.backend == "nvme" and not args.keep_file:
    args.path.unlink(missing_ok=True)   # L159-160: deletes it afterwards
```

Why it is wrong: `--path` is documented as "file path on the NVMe to test", not a scratch path, but
the script treats it as its own scratch file. Point it at a real engram table (or any file) of
exactly `rows * row_bytes` bytes and it will read it, report it as synthetic results, and then
**unlink it** — unless the user happens to pass `--keep-file`. Point it at any other existing file
and it truncates and overwrites it first. The evidence workflow shows this is not hypothetical: the
recorded run used `"path": "results/engram-nvme0.bin"` (`evidence/e3-engram-gather-nvme0.json`).
The `--keep-file` flag only protects the file for the run that knows to use it; the default is
destructive.

What I would change: track whether this run created the file
(`created = not (path.exists() and path.stat().st_size == total)`), and only `unlink` when
`created`. If the path exists with the wrong size, refuse to overwrite unless a new `--force` is
given; if it exists with the right size, either verify it is ours (marker/header/checksum) or
require `--reuse` before trusting it. Never delete a file the run did not create.

### 2. `scripts/greedy-logprobs.py` reports "same tokens" for an empty or truncated completion

`ask` L66–68, `compare` L71–81, `main` L119–126.

```python
if not logprobs or not ch.get("logprobs"):
    return []                                   # L66-68
...
first, mx, mean = compare(base[p], toks)        # L122
worst = max(worst, mx)                          # L123
status = "same tokens" if first is None else ...  # L124
```

Why it is wrong: `compare([], base)` and `compare(base, [])` both return `(None, 0.0, 0.0)` — zero
common tokens, "no divergence" — so a completion that came back empty (server error surfaced as an
empty choice, `logprobs` ignored/omitted, `content: []`) prints `same tokens`, contributes `0.0` to
`worst`, and the run exits 0. A strict-prefix truncation is equally invisible: if the new run
returns the first 3 of 32 baseline tokens and they match, it also reports "same tokens". This is a
drift gate that passes on a broken run. (L68 also raises `TypeError` if a server returns
`logprobs: {"content": null}`.)

What I would change: in `compare`, treat unequal lengths as a divergence (return the shorter index
and a flag) and make `main` fail loudly — non-zero exit or an explicit `INCOMPLETE` status — when
either side has zero tokens or lengths differ. The module caveat (L16–21) already says token
divergence is not quality evidence, but that is an argument for not using this as a gate, not for
letting an empty response pass it.

---

## Medium

### 3. `bench-engram-gather.py` `gather_dram` measures 2 bytes of each 264-byte row; the headline number is not a row-gather rate

`gather_dram` L56–62, derived fields L149–152.

```python
for off in offsets:
    acc += mv[off]                     # L60
    acc += mv[off + row_bytes - 1]     # L61
```

Why it is wrong: the loop touches the first and last byte of each row — 2 bytes and at most two
cache lines out of the 5–6 that a 264-byte row occupies. Nothing in the timed region moves the
263 bytes in between, yet `rows_per_s` (L149), `tokens_per_s_equiv`, and
`prefill_64k_estimate_s = 3_100_000 / rows_per_s` (L152) are computed as if whole rows were read.
The recorded result (`evidence/e3-engram-gather-dram.json`) is 4,148,704 rows/s = **1.1 TB/s at
264 B/row**, roughly 8× the host's measured ~131 GB/s read ceiling documented in
`feasibility-calc.py` L57–60 — an impossible number that itself proves the traffic is not happening.
It also means the DRAM and NVMe rows/s are not comparable (the NVMe path really does read 4 KiB per
row), and `docs/HARDWARE.md` L46 quotes "4.15 M rows/s ... Random 264 B row gather" as a measured
result. The docstring claim that both numbers are "lower bounds on what a C/CUDA gather path
achieves" (L14–15) is inverted for DRAM: it is an upper bound, because a native gather must read
the missing ~262 bytes.

What I would change: make the DRAM path read every byte of the row (e.g. a strided
`memoryview` sum, or `mv[off:off+row_bytes]` into a byte accumulator) so the number is a real
traffic-bound gather, and report `bytes_moved / wall_s` explicitly so it can be sanity-checked
against the DRAM ceiling. If the 2-byte touch is deliberate (a TLB/line-fetch probe), rename the
metric (`row_start_touches_per_s`) and drop the prefill extrapolation. The `prefill_64k_estimate_s`
field should be derived from measured bytes, not from `rows_per_s`, and the docs table re-labelled.

### 4. `scripts/trace-prefix-memory.py` — blocks in `active_pending_free` are counted nowhere, so `reserved ≠ allocated + inactive`

`summarise_snapshot` L85–94.

Why it is wrong: PyTorch's memory snapshot block states are `active_allocated`,
`active_pending_free` (freed but not yet returned to the allocator/CUDA), and `inactive`. The loop
only adds `active_allocated` to `allocated_bytes` (L86–90) and `inactive` to `inactive_bytes`
(L91–94). `reserved_bytes` (L83) is the sum of segment `total_size`, so for any segment with
pending-free blocks the reported identity `reserved = allocated + inactive + fragmentation` is off
by the pending-free bytes, and `diff_devices` (L144) diffs these same three numbers. A reader
tracking "inactive" growth as cache that can be reused will under-report what is actually held.

What I would change: add a `pending_free_bytes` bucket (and count) for `active_pending_free`, print
it next to `inactive`, and either assert or document
`reserved = allocated + pending_free + inactive + slack`. Minor, same file: the driver sets
`stop.set()` at L249 but never `sampler.join()`s before building `report` at L250–251, so one final
VRAM sample can be appended after the run and even after the last snapshot; join the sampler
(small timeout) before serializing.

### 5. `scripts/feasibility-calc.py` — Engram fractions are hard-coded to one budget file, and the exit code contradicts the documented contract

`ENGRAM_FRACTION` L49–51; exit at L292 with `--layout` at L228.

Why it is wrong: `dram-fp4` and `dram-q3k` are ratios of a **hard-coded 188.83 GiB** FP8-table size,
while the actual denominator is `budget["engram_tables"]` (printed at L271). Run against a
different budget (the repo ships `budget-libertai-nvfp4.json`, `budget-nvidia-nvfp4.json`,
`budget-upstream-bf16fp8.json`) and the host-footprint estimate is silently wrong with no warning.
Second: the docstring says "Exit status: 0 if the requested layout fits, 3 if it does not" (L22),
but with the default `--layout all` (L228) the process returns 3 if *any* of the four layouts fails
(L292), even when the layout the user cares about fits. Any caller using `$?` as a gate gets a
false negative.

What I would change: compute the fractions from `budget["engram_tables"]` (keep the quoted
FP4/GGUF numbers as per-mode byte counts, not as ratios of a frozen denominator), and make the
exit code depend only on the requested layout(s): return 0 when every *requested* row fits, 3 only
when a requested row does not, and consider a distinct code for "some non-requested layout fails"
under `--layout all`.

---

## Low

### 6. `kernels/e13/mxf4_grouped_gemm.cu` L874 — device properties are re-fetched for device 0 and the SM check validates the wrong GPU

```cuda
CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));  // L873
cudaError_t error = cudaGetDeviceProperties(&props, 0);          // L874  <-- clobbers props
if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {  // L875
```

Why it is wrong: L873 correctly fills `props` for the active device, then L874 overwrites `props`
with device 0's properties (and its return value is assigned but never checked). On a host where
the active device is SM120 but device 0 is not (or vice versa), the guard admits or rejects the
wrong GPU. In this repo's own workflow the script is run under `CUDA_VISIBLE_DEVICES`, so the
clobber is masked; on a mixed host it is a real wrong-device check.

What I would change: delete L874 (L873 already did the job); if the intent was an error check,
make it `CUDA_CHECK(error)` on the L873 result instead.

### 7. `kernels/t321/bench_fp8_prefill_gemm.py` — `--iters 1` crashes, and some candidates allocate inside the timed region

`statistics.quantiles(times, n=10)` L200; `--iters` default 20 L158; `triton_tuned` L109–110;
`scaled_mm` L148–149.

Why it is wrong: (a) `statistics.quantiles` raises `StatisticsError` for fewer than two data
points, and the call is outside the `try` at L190–198, so `--iters 1` kills the whole run after the
first candidate. (b) `triton_tuned` and `scaled_mm` return a freshly allocated output each call,
while `cutile_32`/`cutile_128`/`cutlass_128` write into the preallocated `out_bf16` (L121, L133,
L140) — the timed region for the allocating candidates includes `torch.empty`/allocator work that
the others avoid, biasing the comparison. (The `Timer.flush.zero_()` at L90 is fine: it is
stream-ordered before `s.record()`, so the flush kernel is excluded and only its L2 effect remains,
which is the intent.)

What I would change: guard the quantile call (`if len(times) < 2`, or require `--iters >= 2`), and
either pass `out=` for every candidate or state in the header that allocating candidates include
allocation. Nothing else in the harness (percentile indexing, TFLOPS math, `rel_err` reference)
looks wrong.

### 8. `scripts/verify.py` L40 with L109–110 — `--needles > 6` can never pass

```python
depths = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99][:needles]   # L40
...
record("multi_needle", found == args.needles and ...)      # L110
```

Why it is wrong: `zip(keys.items(), depths)` (L44) plants only `len(depths)` needles, but the
question (L49) lists all `needles` keys and the check requires all `needles` codes to appear.
With `--needles 8`, two codes are never in the prompt, so `found` can never reach 8 and the run
reports a failure that is an artifact of the harness. Default is 4, so this only bites when someone
raises the count.

What I would change: either extend `depths` (or generate them evenly) when `needles > 6`, or cap
`--needles` at `len(depths)` with an explicit argument error.

### 9. `scripts/bench-engram-file.py` L126–129 and `scripts/bench-engram-advise.py` L102–106 — `MADV_WILLNEED` misses the tail page of straddling rows

Why it is wrong: both compute a single page-aligned address from the row's *start*
(`(w_off + r * DIM) // page * page`) and advise exactly one page. A 256-byte weight row crosses a
4 KiB boundary ~6% of the time (and the 8-byte scale row ~0.2%), so its tail lives on a page that is
never advised. `bench-engram-file.py`'s docstring claims "every row's page" (L17) and
`bench-engram-advise.py`'s `pages()` claims "the rows' weight and scale bytes touch" (L103);
`resident_fraction` (L108) is therefore computed over a page set that is slightly too small. The
effect on the cold-miss timings is small but systematic.

What I would change: advise `(a, min(page, end - a))` for the row's end too — i.e. emit both the
start page and the page containing `w_off + r*DIM + DIM - 1` (dedup via `np.unique`) — or state in
the docstring that only the start page is advised.

### 10. Miscellaneous file handling and robustness

- `scripts/bench-dram-read.py` L19: writes to a hard-coded `open("/w/results/e1b.json","w")` with no
  directory creation and no `--output`; fails (or writes into the wrong tree) anywhere that path
  does not exist. Make it an argument, `mkdir(parents=True, exist_ok=True)`.
- `scripts/bench-b2-uva-moe.py` L425: `open(args.output, "w")` with no parent `mkdir`, unlike the
  other harnesses. Add `Path(args.output).parent.mkdir(parents=True, exist_ok=True)`.
- `scripts/checkpoint-budget.py` L106–111: a local directory without
  `model.safetensors.index.json` raises an uncaught `FileNotFoundError` (the caller's handler at
  `feasibility-calc.py` L241 catches `OSError`, but standalone use crashes); raise a clear error
  naming the missing file.
- `scripts/gap-context.py` L48: `gpu[0]` raises `IndexError` when the trace has no kernel/memcpy
  events; guard and print "no GPU events".
- `scripts/stress-kv-full.py` L91: `totals` is taken from `samples[0]`, which may be the error row
  appended at L50 (no `*_total_mib` keys) → empty `vram_total_mib` and the "min free" line
  suppressed. Pick the first sample that actually has the keys.
- `scripts/bench-engram-gather.py` L69: `os.pread` with `O_DIRECT` requires a page-aligned buffer
  and length; this relies on CPython returning 4 KiB-aligned bytes objects for reads ≥ 4 KiB. It
  worked in the recorded NVMe runs, but it is an implementation detail, not a guarantee — use
  `mmap`/`ctypes` aligned buffers or drop `O_DIRECT` when alignment cannot be guaranteed.
- `kernels/e20/tune_w8a8_block32.py` L156–157: `partials = torch.empty((split_k, M, N), ...)` is
  allocated inside `run_config`, which is the timed function (L196), so split-K configs pay an
  allocator call in every rep; allocate once outside `bench`. Also L143 compares with
  `rtol=atol=2e-2`, loose enough that a config with a partially wrong scale path could pass — tighten
  to the level the bf16 output actually justifies (the E20 evidence shows max_rel ~1e-2 for correct
  configs, so this tolerance is doing little).
- `scripts/bench-dram-suite.py` L158: the DMA helper always runs `--gpus all`, so a `--gpu-dma 0`
  run still copies on every visible GPU (contaminating the measurement); pass the requested device
  set. L232's fixed `time.sleep(18)` assumes container start + `pin_memory` fits in 18 s — on a cold
  image pull it does not, and the "DMA running by now" comment becomes false; poll the helper's
  stdout for the first `h2d` line (with a timeout) instead.

---

## Verified correct (checked, no finding)

- **Engram row layout and offsets.** Checked against the real shard header: `embed.weight` is
  F8_E4M3 with 256 columns and `scale` F8_E8M0 with 8 columns → 264 B/row; the scripts' offset
  math (`w_off`, `s_off`, `base = 8 + n`, `rows` derivation) matches, and `bench-engram-file.py`'s
  bit-exactness check compares mmap reads to safetensors reads, so a layout error would be caught.
- **`kernels/e13` (decode/grouped GEMM):** host reference vs device output compared with exact
  `TensorEquals`, alpha/beta pointer ordering correct, launch/verify loop sound.
- **`kernels/e20` (`check_sglang_w8a8_path.py`):** positional call matches the overlay's
  `w8a8_block_fp8_matmul_triton` signature; the FP32 dequantised reference and its block-scale
  indexing (`b_s[n0//BLOCK[0] : n1//BLOCK[0]]`) are correct for the tested shapes.
- **`verify-nan-rejection.py`:** read in full; the inject/serve/assert flow is correct.
- **`download-model.sh`:** pinned-revision check and resume behavior correct; only nit is that the
  free-space precheck is skipped when `RECIPE_PIN.txt` exists.
- **`scripts/hsmp-clocks.py`:** ioctl struct size (44 B, packed) and the DDR bandwidth bit decode
  match the kernel UAPI header.
- **`bench-decode.py` / `bench-prefill.py` / `sched-wait-per-step.py`:** the `steps = max_tokens`
  assumption in the latter is guaranteed by the former's `min_tokens`/`ignore_eos`, so it is not a
  bug. `nll-score.py`, `verify-batch-consistency.py`, `summarize-power.py`,
  `benchmarks_headline.py`, `kernel-census.py`, `evidence_manifest.py`, `kt-timeline-report.py`,
  `wait-ready.py`, `load-env.sh`: no material findings.
- **`bench-cpu-expert.py`, `bench-pcie-h2d.py`, `bench-dram-contention.py`:** measurement math
  (bytes/iteration, pinned vs pageable, additivity ratio) is correct; the int8-deq variant is
  already labelled an invalid compute proxy.
- **`kt-export-bench.py`:** declared historical and unmaintained for the current overlay; not
  reviewed for correctness against the replaced path.

Coverage note: test files under `scripts/tests/` were spot-checked (layout-sensitive ones:
`test_moe_workspace.py`, `test_indexer_k_gather.py`, `test_indexer_row_tiling.py`,
`test_hc_combine_input.py`) but not reviewed line-by-line; they are test harnesses, not the
measurement tooling this review targets.
# Benchmark evolution E19 → E23 R0, and the E20 dense FP8 GEMM tuning in detail

Written 2026-09-22. §1 consolidates the hybrid's numbers in one place with the aggregate / per-stream / wall distinctions spelled out; §2 records E20 at the level of module names, config selection and search space so it can be reproduced or upstreamed ([OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md)). Every number is **measured** unless labelled; sources are `docs/BENCHMARKS.md`, `plan/EXPERIMENTS.md` E19–E23 and `evidence/`.

## 1. Benchmark evolution

### 1.1 Three throughput definitions

`scripts/bench-decode.py` reports all three; they are not interchangeable.

| Term | Definition | Use |
|---|---|---|
| **summed stream rate** ("aggregate") | Σ over concurrent streams of (tokens ÷ that stream's own generation time) | comparable across runs; overstates real throughput |
| **per stream** | summed ÷ concurrency | what one user sees while decoding |
| **wall** | total tokens ÷ wall-clock from first request to last token, including TTFT and ramp | what a queue of jobs actually gets; **the number to quote** |

At E23 R0 wall is 7 % (C1) to 11 % (C4) below the summed rate.

**Erratum 2026-09-23:** the E23 R0 C4 comparison is an artifact of kt-kernel bug C. The qualified patched C4 is **50.8 summed / 47.0 wall tok/s (12.7 per stream)**; the old percentage is retained only as historical context.

### 1.2 Decode

| Metric | E19 first serve | E20 tuned FP8 config | E21 deferral D=4 | E22 streamed prefill resident | E23 R0 (512-tok outputs) |
|---|---:|---:|---:|---:|---:|
| C1 median, summed | 13.98 tok/s | 31.60 (31.42–31.75) | **38.53** | 38.41 | 38.73; **wall 36.01**, TTFT 1.02 s |
| C2 aggregate (per stream) | 28.0 (14.0) | ≈ 38 (19) | 45.44 (22.7) | 47.81 (23.9) | not run |
| C3 aggregate (per stream) | 64.0 (21.3; 64-token run) | — | 111.71 (37.2) | — | 111.71; wall 98.24 (37.2), TTFT 1.45 s |
| C4 aggregate (per stream) | 86.7 (21.7); wall 76 distinct / 72 same-prompt | ≈ 120 (30) | **146.96** (36.7) | 144.97 | 148.33; wall 131.89 (37.0), TTFT 1.63 s |
| Step time at bs=1 | ≈ 71 ms | ≈ 32 ms | ≈ 26 ms | unchanged | — |
| GPU power during C1 / C4 | 144/159 W / 172/189 W | — | never near 300 W | — | — |
| Teacher-forced NLL | — | 1.48741 (D=0 baseline) | ΔNLL −0.0061, z +1.44 | ΔNLL −0.0013, z +0.35 | speculation off: no change |

Improvements: E19 → E20 **2.26×** (dense FP8 GEMM configs, §2); E20 → E21 **+22 % C1, +22 % C4** (expert deferral D=4); E21 → E22 decode unchanged by design (prefill-only change); E22 → E23 R0 reproduces within +0.5 %. Overall C1: 13.98 → 38.7 tok/s, **2.77×**; C4 aggregate 86.7 → 148.3, **1.71×**.

The C2 dip (per-stream ≈ 19–24 tok/s against ≈ 37 at C1/C3/C4) has been present since E20 and is unexplained. D=5 gives ≈ +4 % more but asserts at bs ≥ 4; D=6 also costs +0.23 nats (E21) — D=4 is the ceiling until E24 resolves the NaN.

**Erratum 2026-09-23:** every C3/C4 decode figure in this table, and the improvement/scaling conclusions drawn from them, is an artifact of kt-kernel bug C: the unpatched bs=4 graph computed correct CPU-expert output only for row 0, and bs=3 padded into it. The patched-image references are C3 **38.3 summed / 35.8 wall tok/s (12.8 per stream)** and C4 **50.8 / 47.0 (12.7 per stream)**. C2 was correct on both builds. C4 now buys ≈ +30% aggregate over C1 at about 3× lower per-stream speed. See `docs/BENCHMARKS.md` “Erratum” and `docs/TROUBLESHOOTING.md` bug C.

### 1.3 Prefill and TTFT by context length

| Prompt | E19–E21 CPU path | E22 chunk 4,096 (default) | E22 chunk 8,192 (`CHUNKED_PREFILL_SIZE=8192`) |
|---|---:|---:|---:|
| 16K | 215.6 tok/s, TTFT 75 s | 837 tok/s, TTFT 19.1 s | **1,375 tok/s**, TTFT 11.6 s (first post-launch run 547) |
| 32K | ≈ 210 tok/s, TTFT ≈ 2.5 min (**projected**) | 811 tok/s, TTFT 38.4 s | **1,404 tok/s**, TTFT 22.8 s |
| `verify.py` 16K, 3/3 needles | pass, TTFT 57–74 s | pass, TTFT 35.5 s | pass, TTFT 9.8 s |
| free VRAM after 32K prefill | — | ≈ 3.7 GiB | ≈ 1.2 GiB |
| 262K | unqualified | unqualified | unqualified |

Streaming cost per chunk is nearly flat in tokens (4.7–5.1 s at 4K, 5.4–5.8 s at 8K ⇒ ≈ 4.6 s fixed export/barrier), so 8K chunks are 1.65–1.73× faster; 16K chunks do not fit. The 4K chunk stays the default because 8K leaves ≈ 1.2 GiB and cannot coexist with the DSpark draft (3.85 GB/rank). Percent improvements over E19: prefill **3.9×** (4K) / **6.4–6.5×** (8K); 16K TTFT **−75 % / −85 %**.

### 1.4 Against the 2-GPU incumbents (E14, T2.9, measured)

| | GLM-5.3-Flash (EXL3, 16 slots, 325 W) | V4-Flash NVFP4 (vLLM, 16 slots, 1M) | V4.1-Flash hybrid (this recipe, 4 slots, 32K) |
|---|---:|---:|---:|
| C1 decode | 132 tok/s (40 % DFlash acceptance) | 207–239 (K=5 DSpark, 40–63 %); 89 without speculation | 38.5 (speculation off) |
| C16 / C4 decode wall | 642 (C16) | 1,213–1,258 (C16) | 131.9 (C4) |
| 32K prefill | 4,429 tok/s | 8,256–8,376 | 811 (4K chunk) / 1,404 (8K) |
| 1M prefill | 3.8K tok/s | 1.9K | not applicable (32K context) |

The hybrid's case on two GPUs is quality and privacy at 32K with four slots, not speed; the 3-GPU layout ([notes §2](HETEROGENEOUS-ROUTING-AND-OPTIMIZATION-NOTES.md#2-third-rtx-pro-6000--back-of-envelope)) is the projected route to 42–60 tok/s C1.

**Erratum 2026-09-23:** the retained 131.9 C4 wall value is a bug-C artifact; the qualified value is **47.0 wall tok/s**. Four slots remain supported, but they do not preserve C1 per-stream speed.

### 1.5 Current qualified operating point (2026-09-22)

`scripts/serve-hybrid.sh` defaults: `KT_NUM_GPU_EXPERTS=208`, `KT_MAX_DEFERRED=4`, `KT_GPU_PREFILL_MIN_TOKENS=1024`, `CHUNKED_PREFILL_SIZE=4096`, `MEM_FRACTION_STATIC=0.92`, `MAX_RUNNING_REQUESTS=4`, `CUDA_GRAPH_MAX_BS=4`, `DSPARK=0`, 32K context, 325 W. Qualified by `verify.py` 16K all-pass, ΔNLL within gate, and the E20 attempt-8 35-minute / 96-mixed-batch soak; the E22c NaN sighting (E24) means the ≥ 20K-token zero-Xid soak is still owed for the streamed-prefill default.

### 1.6 Speculation status (E23)

| Round | Config | Result | Label |
|---|---|---|---|
| R0 | D=4, off | C1 36.01 wall / C3 98.24 / C4 131.89; go bar C1 wall ≥ 39.6 | measured |
| fit | K=2, `MEM_FRACTION_STATIC=0.92` | no KV budget (main 82.97 + draft 3.85 GB) | failed |
| fit | K=2, 0.935 | KV 279,808 tokens, 5.74 GB free after capture; three overlay fixes needed to capture and run | measured |
| R1 | K=2 | ≈ 500 tokens: accept length 1.8–2.15 of 3, acceptance 0.41–0.57, ≈ 30–34 tok/s summed (below baseline); then CUDA illegal memory access in target-verify replay | failed (in diagnosis) |
| R1 | K=5 | — | pending |
| R2 | D=0 × {off, K=2, K=5} | — | pending |

**Erratum 2026-09-23:** the R0 C3/C4 values are artifacts of bug C. The corrected speculation-off wall rates are **35.8 (C3)** and **47.0 (C4)**. Because CPU decode cost scales per row (≈ 8 ms + 17.7 ms × rows), this strengthens the E23 verdict that verification loses when it multiplies target rows.

## 2. E20 — dense W8A8 block-FP8 GEMM tuning for SM120 (T3.6d4)

### 2.1 What the problem was — and was not

The T3.6d3 step profile (`evidence/t36d3-step-profile/`) put **42.9 of 70 ms** per bs=1 decode step in **202 calls** to SGLang's Triton dense block-FP8 GEMM running its *default* configuration at 60–80 GB/s on a ≈ 1,600 GB/s card. This is the **dense W8A8 FP8** work — attention projections, shared expert, dense-layer MLPs — with V4.1's `block_shape=[32,32]` scaling. It is **not** the routed-expert path: the MXFP4 experts run in FlashInfer `flashinfer_mxfp4` (GPU tier) and kt-kernel (CPU tier), were already at the bandwidth roofline (E12/E13), and were untouched by E20.

### 2.2 Code path and config selection

- Runtime kernel: `sglang.kernels.ops.quantization.fp8_kernel::_w8a8_block_fp8_matmul` (public `w8a8_block_fp8_matmul`).
- Overlay change: `python/sglang/kernels/ops/quantization/fp8_kernel.py::w8a8_block_fp8_matmul_triton` (overlay `32a25f37`).
- Config lookup: `get_w8a8_block_fp8_configs(N, K, block_n, block_k)` opens `python/sglang/kernels/ops/quantization/configs/N=<N>,K=<K>,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[32, 32].json`; the JSON maps M (batch-size) keys to configurations and the runtime picks the nearest M key.
- Fallback when no file matches: `BLOCK_SIZE_M=64, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32, GROUP_SIZE_M=32, num_warps=4, num_stages=3`, with one WARNING per shape — `Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!` — 14 lines at startup that are easy to miss.
- Why nothing matched: upstream ships `[32,32]` configs **only for H200**; the H100/B200 files are for `[128,128]`. Nothing exists for any SM120 device name.
- Why the stock tuner could not make them: `benchmark/kernels/quantization/tuning_block_wise_kernel.py` searches `BLOCK_SIZE_K ∈ {64, 128}` and filters for divisibility into the scale block, which is 32 wide — an empty search space — and it does not search split-K or swap-AB at all.

### 2.3 The SM120 gate

The tuned configs use `SPLIT_K` and `SWAP_AB`, which stock SGLang honours only on SM90:

```python
hopper_tuned = get_platform().is_sm90 and (
    config.get("SWAP_AB", False) or config.get("SPLIT_K", 1) > 1
)
```

The overlay lifts it to SM120 as well:

```python
platform = get_platform()
hopper_tuned = (platform.is_sm90 or platform.is_sm120) and (
    config.get("SWAP_AB", False) or config.get("SPLIT_K", 1) > 1
)
```

The alternate kernel is plain Triton, not Hopper-intrinsic code; the SM90 restriction reflected where it had been tuned (inferred from reading the kernel).

### 2.4 The seven TP=2 dense shapes

N is sharded by TP, so a `--tp` change changes the shapes and the files must be re-tuned; the warning names the exact missing shape.

```text
N=16384,K=1280
N=1792,K=5120
N=2304,K=5120
N=25600,K=6144
N=4096,K=1280
N=5120,K=1152
N=5120,K=4096
```

### 2.5 Tuner and search space (`kernels/e20/`)

Files: `tune_w8a8_block32.py` (search), `merge_configs.py` (best per shape/M across passes), `check_sglang_w8a8_path.py` (validation through the real SGLang call path); evidence in `evidence/e20-dense-fp8-gemm/`. Ran inside `dsv41-kt` on GPU1 while GPU0 was serving. Timing with `triton.testing.do_bench` (CUDA events, L2 flush).

| M range | Dimensions searched |
|---|---|
| Small M (decode) | `BLOCK_M=16`; `BLOCK_N ∈ {16,32,64,128}`; `SPLIT_K ∈ {1,2,4,8,16}`; `SWAP_AB` on/off; warps 2/4/8; stages 2–5 |
| Larger M (prefill) | `BLOCK_M ∈ {32,64,128}`; `BLOCK_N ∈ {32,64,128,256}`; warps 4/8; stages 2–4; `GROUP_SIZE_M ∈ {1,32}` |
| Pass 2 | extended split-K to M = 32/64/128 after an intermediate-M cliff (e.g. `1792×5120` M=32: 68 → 27 µs) |

`BLOCK_K` is fixed at 32 by the scale layout. Winner pattern at decode M: `BLOCK_M=16`, `BLOCK_N` 32–128, 2 warps, 4–5 stages, `SPLIT_K=8` for K ≥ 4096, `SWAP_AB` at mid-M. One recorded M=128 winner for `5120×4096`: `BLOCK_M=16, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=32`, 2 warps, 5 stages, `SWAP_AB=True`. One M=4096 winner: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=32`, 4 warps, 2 stages. The full JSON per shape/M key is in the overlay's configs directory, not reproduced here.

### 2.6 Validation

- Each selected config vs FP32 dequantise-then-matmul at `rtol = atol = 2e-2` during tuning.
- Installed runtime path (`check_sglang_w8a8_path.py`): seven shapes × M ∈ {1, 2, 3, 4, 5, 7, 768, 769, 1024, 1025} — **70/70 finite**, max relative error 0.0037–0.0043.
- Server: `verify.py --context 16384 --needles 3` all pass; startup shows 7 `Using configuration from … for W8A8 Block FP8 kernel.` lines and 0 default warnings (was 14).

### 2.7 Results

Per-kernel at M=1 (measured, `evidence/e20-dense-fp8-gemm/`):

| N×K | Default → tuned | Speed-up |
|---|---:|---:|
| 16384×1280 | 371 → 30.7 µs | 12.1× |
| 1792×5120 | 119 → 14.3 µs | 8.3× |
| 2304×5120 | 150 → 16.4 µs | 9.1× |
| 5120×1152 | 115 → 12.3 µs | 9.3× |
| 5120×4096 | 258 → 30.7 µs | 8.4× |
| 4096×1280 | 102 → 12.3 µs | ≈ 8.3× |
| 25600×6144 | 1,220 → 217 µs | 5.6× |

Tuned decode cases reach 650–750 GB/s with a 12–14 µs launch floor. At M=4096 the gain is only 2.2–3.0× (≈ 230 TFLOP/s, ≈ 15 % of FP8 peak): prefill-side dense GEMM wants CUTLASS/DeepGEMM block-scaled kernels, not more Triton tuning.

Server level: dense FP8 time per bs=1 step 42.9 → ≈ 3.4–4.5 ms; step ≈ 70 → ≈ 32 ms; **C1 13.98 → 31.60 tok/s (2.26×)**. Recipe commit `aaf9be7`; overlay `32a25f37`.

### 2.8 The scratch-buffer fix is a separate change

Overlay `1d6a98d3` (`flash_mla_sm120.py::_split_kv_pages_to_64`: `torch.empty` → `torch.zeros`) is a NaN probe, not tuning. The E20 C4 NaN persisted after tuning; the 70/70 check ruled the dense kernels out; stale bytes in an unwritten FP8 scratch page (e4m3 `0x7F`/`0xFF` decode to NaN) fit a non-deterministic, first-mixed-batch, both-ranks failure, and zero-fill was followed by the clean 35-minute / 96-batch soak (attempt 8). It is the third candidate upstream PR in [OPEN-SOURCE-PLAN.md](OPEN-SOURCE-PLAN.md) and carries the caveat that "stale bytes" is an explanation, not a bisected root cause.

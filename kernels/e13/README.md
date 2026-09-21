# E13 stage 1 — MXFP4 × MXFP4 grouped GEMM for SM120

A native block-scaled grouped GEMM for DeepSeek-V4.1-Flash's routed experts on the RTX PRO 6000
Blackwell (SM120): A and B in **MXFP4** (E2M1 data, UE8M0 scales, block 32 — the checkpoint's own
format, no re-quantisation), BF16 output, one problem per expert (PtrArray / grouped). Derived from
CUTLASS 4.7.1 example `79d_blackwell_geforce_nvfp4_grouped_gemm.cu`; `diff` against it to see the
changes (type swap, plain `LinearCombination` epilogue, benchmark-file M rounding, `--no_reinit`).

Protocol and results: [plan/EXPERIMENTS.md § E13](../../plan/EXPERIMENTS.md); readout in
[plan/ROADMAP.md](../../plan/ROADMAP.md); background reading in [docs/LEARNING.md §4](../../docs/LEARNING.md).

## Build (GPU-free, ~35 s)

Needs the CUDA 13.0 devel image `dsv4-sm120:cuda-devel-https` and a CUTLASS checkout
(`git clone -b v4.7.1 https://github.com/NVIDIA/cutlass ~/dev/src/cutlass`; override with `CUTLASS_DIR`).

```bash
kernels/e13/build.sh                       # W4A4, tile 128x128x128 -> kernels/e13/mxf4_grouped_gemm_n128k128
E13_A8=1 kernels/e13/build.sh              # W4A8: A = MXFP8 E4M3 (kind::mxf8f6f4), B = MXFP4
TILE_N=64 TILE_K=256 kernels/e13/build.sh  # tile variants (K3)
```

The W4A4 path uses `mma.sync … kind::mxf4nvf4`, which the PTX ISA restricts to `sm_120a`/`sm_121a`;
hence `-gencode arch=compute_120a,code=sm_120a`. W4A8 (`kind::mxf8f6f4`) would also build for `sm_120f`.

## Run

```bash
# V4.1 expert shapes, both schedules, decode verified against CUTLASS's host block-scaled reference
kernels/e13/run.py --verify-decode                         # -> results/e13-stage1.json
kernels/e13/run.py --skip-prefill --no-reinit --iterations 200 --output results/e13-decode.json

# the binary directly
docker run --rm --gpus all -v "$PWD:/w" -w /w --entrypoint ./kernels/e13/mxf4_grouped_gemm_n128k128 \
    dsv4-sm120:cuda-devel-https --m=8 --n=4608 --k=5120 --groups=6 --iterations=100 --no_reinit
```

`run.py` writes per-case `avg_ms`, `tflops`, `weight_GBps` (MXFP4 weight + scale bytes ÷ time; the
decode metric, since decode is weight-bandwidth bound) and `verified` for both schedules.

## Results on this host (2026-09-21, GPU 0 at 300 W)

| Case | M/group | Groups | µs (pingpong) | Weight GB/s | TFLOPS |
|---|---:|---:|---:|---:|---:|
| gate/up 4608×5120 | 1 | 6 | 53.8 | 1,397 | 5.3 |
| gate/up | 8 | 6 | 54.7 | 1,374 | 41 |
| down 5120×2304 | 1 | 6 | 39.6 | 949 | 3.6 |
| gate/up prefill | 64 | 384 | 3,462 (coop) | 1,390 | 335 |
| gate/up dense proxy | 4,096 | 6 | 1,094 | — | 1,060 |

Measured GDDR7 copy rate on the same card is ≈1,470 GB/s (E2), so decode gate/up runs at 93 % of
the memory roofline. Smaller N tiles (128×64×128, 128×64×256) were 30–35 % slower at every decode M.
W4A8 (`E13_A8=1`) verifies but is ~2× slower at decode (gate/up M=1 103.8 µs, 724 GB/s): the 128-row
M tile issues M=128 of MMA for an M=1 group, and the FP8-rate `mxf8f6f4` instruction is compute-bound
there. `TILE_M=64` does not compile (SM120 TMA warp-specialized mainloop asserts a 128-row CTA tile);
a decode-shaped kernel is a separate mainloop (roadmap K10).

## Files

- `mxf4_grouped_gemm.cu` — kernel + host test/bench (CUTLASS example structure)
- `build.sh` — nvcc in the devel container; `E13_A8`, `TILE_N`, `TILE_K`, `CUTLASS_DIR`, `IMAGE` env knobs
- `run.py` — stdlib driver: V4.1 expert-shape cases → JSON
- binaries `mxf4_grouped_gemm_*` are build products and gitignored

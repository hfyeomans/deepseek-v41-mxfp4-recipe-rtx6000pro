# kt-kernel patches carried by the hybrid image

Every change this recipe makes to `kt-kernel` (PyPI `kt-kernel==0.7.0.post4`, source
`kvcache-ai/ktransformers` tag `v0.7.0.post4` = commit `5632f7197e7a9898f0723755c2fc187a946a85a6`)
lives here as a `git`-style patch and is applied by `config/Dockerfile.dsv41-kt` at image build.
Nothing is patched at serve time. PROVENANCE.md has the pins and image ids; this file is the index
and the runbook.

| patch | kind | applied to | what / why | registered |
|---|---|---|---|---|
| `0001-experts-base-temp-buffer-lifetime.patch` | Python | `site-packages/kt_kernel/experts_base.py` | E23 bug C: a temp pinned buffer baked into a captured CUDA graph was freed by a later resize → `cuGraphLaunch` SIGSEGV. Promote a pre-registered temp into `capture_buffers`; retire replaced temps behind a CUDA event instead of freeing | 2026-09-23, `docs/TROUBLESHOOTING.md` "Bug C", `evidence/e23-spec-decode/r3/` |
| `0002-experts-base-capture-assertions.patch` | Python | `site-packages/kt_kernel/experts_base.py` | T3.16a: assert at capture time that every buffer the graph will see is pre-registered and large enough, so a grow-after-capture fails loudly instead of as an Xid | 2026-09-24, `evidence/t316a/` |
| `0003-weight-arena.patch` | C++ (`operators/common.hpp`, `operators/amx/moe_base.hpp`, `ext_bindings.cpp`) | the `_kt_kernel_ext_avx512_bf16` extension, rebuilt from source in the `ktbuild` stage | T3.39 window 2 K2: caller-owned resident-weight arena. `MOEConfig.weight_arenas` (one pointer per CPU TP part); when set, `AMX_MOE_BASE::init` carves every expert's gate / up / down BufferB storage from it instead of `std::aligned_alloc`; static `weight_arena_bytes(cfg)`; post-load `weight_arena_layout()`. Load / forward / export untouched. Lets both TP ranks map one memfd per MoE layer and DMA resident experts directly, removing the per-chunk export | 2026-10-10, `evidence/t339/README.md` "Window 2, K2 build", `evidence/t339/window2/k2-build/` |

Verification of each patch: `scripts/tests/run-kt-{buffer,provider,deferral}-tests.sh` (0001 / 0002, applied
to a scratch copy of the wheel's Python), `config/kt_smoke.py` at build time (0003 bindings present, CUDA host-stream bindings kept),
`evidence/t339/window2/k2-build/smoke_arena.py` (0003 forward bit-identical to the stock allocator).

## How the build works

1. Stage `ktbuild` (`FROM ${BASE}`): clone the tag, assert `git rev-parse HEAD` = the pinned commit, init the
   `pybind11` + `llama.cpp` submodules, `git apply --check` then `git apply` every C++ patch, run
   `python3 setup.py build_ext --inplace` with **the wheel's own flag set** (`CPUINFER_CPU_INSTRUCT=AVX512`,
   AVX512 VNNI / BF16 / VBMI on, AMX / oneDNN off, ~~CUDA off~~ **`CPUINFER_USE_CUDA=1` with static cudart
   and archs `80;86;89;90;120`**, cpptrace off, `CPUINFER_NATIVE=OFF`), and keep the one resulting `.so`.
   The flag set is kt-kernel's `.github/workflows/release-pypi.yml` at the pinned commit. `USE_CUDA=1` compiles
   no device code into the extension (`kt-kernel/cuda` is a separate package); it links cudart for the
   `cudaLaunchHostFunc` bridge behind `CPUInfer.submit_with_cuda_stream` / `sync_with_cuda_stream`, which the
   hybrid's decode graphs and layerwise prefill submit through. The first K2 image (`sha256:8a7802d2…`, kept as
   `dsv41-kt:k2-cpuonly-ext`) was built with `CPUINFER_USE_CUDA=0`, which defines `KTRANSFORMERS_CPU_ONLY` and
   compiles those bindings out; it loaded and page-locked all 40 arenas but died at CUDA-graph warmup with
   `AttributeError: 'kt_kernel_ext.CPUInfer' object has no attribute 'submit_with_cuda_stream'`
   (`evidence/t339/window2/qual/attempt1-cpuonly-ext/`). `config/kt_smoke.py` now asserts both bindings at build time.
2. Main stage: install the pruned wheel, `patch -p1 --forward` the Python patches (`000[12]-*.patch`) into
   site-packages, then `COPY --from=ktbuild` the rebuilt `.so` over the wheel's `avx512_bf16` variant. The other
   five ISA variants and all of the wheel's Python stay as shipped.
3. `config/kt_smoke.py` runs at the end of the build and fails the build if the patched bindings are missing.

The flag set is the wheel's, not `-march=native`, so the only delta between the wheel's extension and ours is
the patch (one variable per experiment). A `CPUINFER_NATIVE=ON` A/B is registered separately (ROADMAP T3.40,
low) and is not expected to move the CPU tier: its hot loops are hand-written AVX-512 intrinsics and the
measured bound is DRAM / fabric bandwidth, not ALU.

## Runbook

Rebuild the image (≈ 2 min for the `ktbuild` stage on the 9955WX; the rest is cached):

```
docker tag dsv41-kt:recipe dsv41-kt:pre-<name>      # keep the previous image until the new one qualifies
docker build -f config/Dockerfile.dsv41-kt -t dsv41-kt:recipe config/
```

Write or refresh a C++ patch:

```
git clone --branch v0.7.0.post4 https://github.com/kvcache-ai/ktransformers .amp/in/ktsrc
git -C .amp/in/ktsrc submodule update --init --depth 1 third_party/pybind11 third_party/llama.cpp
# edit under .amp/in/ktsrc/kt-kernel/, build with the same env as the Dockerfile to check it compiles
git -C .amp/in/ktsrc diff > config/kt-kernel-patches/000N-<name>.patch
```

Then add the patch to the `ktbuild` stage's `git apply` list, extend `config/kt_smoke.py` to assert the new
surface, add a row here and in PROVENANCE.md, and tag the previous image before rebuilding.

Write or refresh a Python patch: diff against the wheel's `kt_kernel/` (`.amp/in/k2/wheel/`), name it
`000N-<file>-<what>.patch`, and widen the `000[12]` globs in the Dockerfile and the three
`scripts/tests/run-kt-*-tests.sh` scripts.

Roll back: `docker tag dsv41-kt:pre-<name> dsv41-kt:recipe`. The `pre-k2` tag is `sha256:0ddb14aa…` (image
before patch 0003).

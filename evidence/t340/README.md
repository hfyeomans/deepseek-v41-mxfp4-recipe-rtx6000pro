# T3.40 kt-kernel compiler-tuning A/B: `-march=native` rebuild vs the wheel's flag set (2026-10-11)

Closed, nothing adopted. Full write-up with the served A/B is in `../t336/README.md` (shared window).

- Build: `docker build -f config/Dockerfile.dsv41-kt --build-arg KT_CPU_INSTRUCT=NATIVE -t dsv41-kt:native config/`
  (`build-native.log`; `ARCH_FLAGS` shows `-march=native` plus the explicit AVX-512 F/BW/DQ/VL/VBMI/VNNI/BF16
  flags). `CPUINFER_NATIVE`, which kt-kernel's `setup.py` docstring lists, is never read at the pinned commit
  `5632f71`; `CPUINFER_CPU_INSTRUCT=NATIVE` is the real switch (`-DLLAMA_NATIVE=ON`).
- One layer (`one-layer.sh`, `one-layer/`): recipe 0.58 / 0.58 ms at T=1, 43.20 / 43.28 at T=256;
  native 0.58 / 0.59 and 43.43 / 43.47 - identical within 1 %.
- Served (`../t336/serve-ab/c-native/` vs `a2-baseline/`): C1 60.43 vs 60.96, C4 119.94 vs 119.20,
  prefill 16K / 32K 3,147 / 3,414 vs 3,197 / 3,404, kt CPU-s per step 0.330 / 0.705 vs 0.322 / 0.676,
  NLL **0 of 65,653 logprobs moved**, health 0.
- The image keeps `KT_CPU_INSTRUCT=AVX512`; `dsv41-kt:native` is a local tag only.

# Provenance (generated 2026-10-23T20:06:36+00:00)

- Model: Qwen/Qwen3-30B-A3B@main  (Qwen/Qwen3-30B-A3B-GGUF:Q4_K_M)
- Hardware: RTX 3090 (24 GiB, Ampere) - published reference numbers, not local measurements
- Quantization: gguf-q4_k_m
- Plan: emitted by recipe-generator/emit_recipe.py v0.1.0 from `placement_calc.py examples/qwen3-30b-a3b/budget.json --hardware examples/qwen3-30b-a3b/hardware-rtx3090.json --json`
- Trust: calibrated
- Calibration anchor: Hardware Corner gpu-llm-benchmarks RTX 3090 row, published via https://smeltcore.com/recipes/qwen3-30b-a3b-rtx-3090/ (llama.cpp, -fa 1, Q4_K)
- Engine pins (sglang-hybrid): base da64c5cb, overlay head deec7a6f (tree adfc1b0f),
  image dsv41-kt:recipe; see engine-notes.md for the patch inventory.
- k-factors: {'decode': 1.0, 'c4': None, 'prefill': 0.21497185529114948}

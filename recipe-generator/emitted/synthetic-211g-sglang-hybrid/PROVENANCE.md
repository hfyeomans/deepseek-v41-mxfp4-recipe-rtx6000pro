# Provenance (generated 2026-10-23T20:06:36+00:00)

- Model: synthetic/211GB-V4-class-MoE@synthetic  (synthetic/211GB-V4-class-MoE)
- Hardware: SYNTHETIC run: the same quorra box (2x RTX PRO 6000, 192 GiB VRAM, measured bandwidths) serving the synthetic 211 GiB model
- Quantization: mxfp4
- Plan: emitted by recipe-generator/emit_recipe.py v0.1.0 from `placement_calc.py examples/synthetic-211g/budget.json --hardware examples/synthetic-211g/hardware.json --json`
- Trust: calibrated-model (k carried from another example)
- Calibration anchor: host/engine factors carried over from the DSV4.1 example (same box, same SGLang-kt engine family); the synthetic gpu_ms is NOT validated
- Engine pins (sglang-hybrid): base da64c5cb, overlay head deec7a6f (tree adfc1b0f),
  image dsv41-kt:recipe; see engine-notes.md for the patch inventory.
- k-factors: {'decode': 1.0, 'c4': 1.02, 'prefill': 0.789}

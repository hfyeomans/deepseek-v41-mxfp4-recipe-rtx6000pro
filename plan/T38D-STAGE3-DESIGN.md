# T3.8d stage 3 — the runtime expert weight swap: design (2026-10-23)

**Prize (measured):** the live mask holds 98.1 % GPU share on shifted traffic vs the static map's
79.1 % (stage 2); held-out decode CPU share 2.1 % vs 9.5 % static (step 1). The swap turns those
mask wins into served expert placement.

## The id space (verified against the wrapper)

- kt_remap.py permutes each layer's experts at LOAD so the el8 GPU set occupies physical ids
  0..num_gpu_experts-1 (219); the CPU set occupies 219..383. The router emits physical ids
  directly (the T3.8 recorder sees post-remap ids), so a runtime swap needs no gate change: it is
  a second permutation applied to the router's topk_ids between the router and the tiers.
- kt_ep_wrapper.apply(): topk_ids feed the GPU MoE runner positionally (< 219 = GPU slot) and
  remap_topk_ids_for_cpu() maps >= 219 to the compact CPU view (-1 = GPU-resident slot).

## The swap, per layer, per moved pair (promoted P from CPU, demoted D from GPU)

1. **Id swap:** `swap_table[384]` (identity by default) applied to topk_ids once per layer:
   `topk_ids = swap_table[topk_ids]` (torch gather; skipped entirely while identity). P takes D's
   physical slot (< 219 -> GPU), D takes P's slot (>= 219 -> CPU tier). Graph-safe: same class of
   op as remap_topk_ids_for_cpu (data-dependent on ids, fixed shapes).
2. **GPU-side row copy (promote direction):** write P's MXFP4 rows into the layer's resident MoE
   tensors at slot D. The bytes come from the checkpoint shard and the loader's own GPU packing
   (Mxfp4FlashinferCutlassMoEMethod.process_weights_after_loading path: w13 int8-packed
   [E, 2N, K/2] with the SM120 scale interleave, s13/s2 E8M0). Same conversion as the resident
   loader -> byte-identical rows; validated offline by packing a resident expert from the
   checkpoint and comparing bytes against the loaded tensor (the stage-3a unit test).
3. **kt-side arena write (demote direction):** write D's MXFP4 rows into P's arena slot. kt's
   arena content is written once by CompactMXFP4Loader (kt_loader.py) at load; the swap needs a
   minimal write path - **patch 0004**: a `swap_cpu_expert(layer, slot, packed_rows)` binding on
   the existing memfd arena (the arena is caller-owned since patch 0003, both ranks page-lock and
   DMA from it). The bytes come from the checkpoint via the same compact packing the loader uses.
   Until patch 0004 lands, the swap can only be simulated; a promote-only swap without it would
   strand D's routes (D is neither GPU-resident nor in the arena).
4. **The swap driver (stage 4 territory):** a scheduler-side trigger consuming the stage-2 EMA
   mask: at each trigger, apply up to K pair moves (LFRU policy, K small e.g. 4), then re-run
   steps 1-3. Trigger cadence + K are the stage-4 tuning knobs.

## Correctness gates (the rounding note from T3.8b)

A tier move changes the expert's rounding (GPU path vs CPU path), so the swap is routing-exact
but NOT bit-exact. The gates are: NLL delta <= 0.002 (cluster-robust, the T3.28 gate), held-out
CPU share < 8 % on the T3.8c families, C1/C4 up, stress/needles/health, and the identity-swap
control (swap table = identity must hold NLL identity = 0 tokens moved - this validates the
machinery is inert until a move happens).

## Staged execution

- **Stage 3a (this branch):** the swap table + the wrapper wiring (inert at identity), the
  promote-direction packing function, the identity-serve validation (NLL identity + C1 flat),
  and the offline packing byte-compare unit test.
- **Stage 3b:** patch 0004 (kt arena expert write) + image rebuild; then the first real swap A/B
  with the stage-2 mask driver, under the gates above.
- **Stage 3c (stage 4 of the plan):** the cadence/K tuning + the default decision.

## References

- kvcache-ai/sglang fork: _update_gpu_experts_from_batch + copy_experts_weights_int4
  (kt_ep_wrapper.py L6051-6130; the MXFP4 variant to mirror: _prepare_weight_mxfp4 /
  mxfp4_deepseek.py) - the reference swap driver, INT4-only today.
- vLLM #48505: MXFP4 expert cache (multi-dim scale handling, modular kernel for externally
  remapped topk_ids, LFRU policy) - borrow the LFRU + the remap-kernel idea.
- Our overlay: kt_ep_wrapper.py (apply(), remap_topk_ids_for_cpu), kt_remap.py (load perm),
  kt_loader.py (CompactMXFP4Loader - the packing to reuse for the demote bytes), the loader's
  Mxfp4FlashinferCutlassMoEMethod (the packing to reuse for the promote bytes).

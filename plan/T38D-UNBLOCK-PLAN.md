# T3.8d — runtime expert placement: unblock plan (2026-10-23, from the researched state of the art)

**Go signal (measured, evidence/t38d/):** decode-trained online masks read 1.8-2.2 % CPU share on
held-out 1017 decode traffic vs the static el8 map's 9.5 % (14.4-15 % on the full sets). The prize
is 7-12 percentage points of decode CPU routes - the T3.8b map bought C1 +37 % for a 27 pp
reduction, so this is the only remaining software lever in the +5-15 % C1 class.

**Why it is blocked today** (unchanged since 2026-10-11, each with its researched fix):
1. Upstream SGLang's kt path is positional (`topk_ids >= num_gpu_experts`); the fork's dynamic
   update machinery (`kvcache-ai/sglang` `kt_ep_wrapper.py` L4817-4920, L6051-6130) is the
   reference implementation but **skips MXFP4** (in-code comment L5845-5860: `copy_experts_weights_int4`
   hardcodes INT4 Marlin names). Fix: add `copy_experts_weights_mxfp4` (mirror `_prepare_weight_mxfp4` /
   `mxfp4_deepseek.py`) and remove the skip.
2. kt-kernel 0.7.0.post4 exposes no resident-weight pointer; our T3.39 patch 0003 memfd arenas are
   caller-owned, so a tier swap needs an arena-content swap path on the kt side (patch 0004-style).
3. On OUR overlay the runtime machinery does not exist (the 393-line wrapper + kt_remap permute at
   load only).

**Staged plan** (each stage independently committable; stages 1-2 are GPU-free):
- **Stage 1 (done, this commit):** vendor the PR #2093 planner/tracker (`scripts/expert_placement.py`)
  + the mask eval (`scripts/t38d-mask-eval.py`) and measure the go signal on real traffic. Result:
  decode masks hit 1.8-2.2 %; the design must be **phase-aware** (online decode mask + static
  prefill map).
- **Stage 2:** integrate the EMA tracker into the overlay as an opt-in residency-mask source
  (consume the T3.8 recorder's per-phase counts; emit `KT_EXPERT_MAP`-style masks live; log-only
  first, `SGLANG_DSV41_RUNTIME_EXPERT_MASK=1`). Gate: mask quality vs el8 on held-out traffic
  (t38d-mask-eval), no serve-path change.
- **Stage 3:** the weight swap. GPU side: a runtime re-permutation of the resident expert tensors
  (the loader tensors live per layer; a swap moves the promoted expert's weights GPU<-kt). kt side:
  arena-content swap (patch 0004) so the demoted expert's MXFP4 bytes leave the GPU tier and the
  promoted one's arrive. Reference implementations: kvcache-ai/sglang
  `_update_gpu_experts_from_batch` + a new `copy_experts_weights_mxfp4`; vLLM #48505 (MXFP4 expert
  cache: multi-dim scales, modular kernel for externally remapped topk_ids, LFRU policy).
- **Stage 4:** gates - held-out CPU share < 8 % on the T3.8c families, NLL identity, C1/C4 up,
  stress/needles, then the default flip.

**Out of scope / rejected:** BINDER idea 12 staging-slot prefetch (E31 closed it: consecutive-layer
route overlap 2.2 %); router-before-attention predictors (same); SGLang Paged Experts until it
grows an MXFP4 format (PR #42557).

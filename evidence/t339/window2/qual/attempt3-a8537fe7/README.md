# Attempt 3 (12:27–12:41) — clean qualification of overlay `a8537fe7`: every gate but prefill

One driver, image `dsv41-kt:recipe` = `sha256:314749b5…` (CUDA bindings present), launcher
defaults (64K × 4, chunk 8192, `KT_CPUINFER=16`, el8 mix map). Against the qualified profile
`9d481b0b` (`evidence/t320-engram-prefetch/chunk-ahead/on/`):

| gate | `a8537fe7` | `9d481b0b` | read |
|---|---|---|---|
| 16K prefill cold / warm | 2,136 / 2,166 | 2,195 / 2,345 | −3 / −8 % |
| 32K prefill cold / warm | 2,253 / 2,235 | 2,499 / 2,612 | −10 / −14 % |
| C1 summed / wall | 58.93 / 57.06 | 57.89 / 56.08 | flat |
| C4 summed / wall | 122.37 / 117.37 | 122.46 / 117.57 | flat |
| NLL vs `9d481b0b` corpus | 0 of 65,653 logprobs moved | — | bit-identical (the `corpus.txt` "changed 65226" is vs the positional-map comparator `t328/default-219-8192`, same as T3.8b's reading, z +0.33) |
| 4 × 62K stress | 4/4, wall 114.8 s, min free 2,553 / 2,496 MiB | 103.6 s, 2,171 / 2,042 | slower; +380 MiB trough (one 1.44 GiB staging set instead of the double buffer) |
| needles | 6/6 at 55K, TTFT 24.4 s | 6/6, 19.1 s | slower |
| same-prompt C4 | PASS (388 / 224 post) | PASS | |
| health | 0 | 0 | |
| VRAM after capture | 92,334 / 92,483 MiB | 92,778 / 92,835 | −0.4 GiB |

Container log: `8192 tokens x 40 layers streamed in 3.2–3.8 s (host side)` = 80–95 ms per
layer-chunk, *above* the 65 ms export-bound period the arenas replaced. So the arena path is
correct (NLL, needles, same-prompt) but its copies are slower than the export + serialised copy
they replaced.

## Diagnosis (`stage_dma.py` / `stage_dma.txt`, idle box, rank 0's slice of one real-size arena)

`_stage` as committed in `a8537fe7` issues four copy classes per layer:

| class | calls | bytes | median | GB/s |
|---|---|---|---|---|
| w13 nibbles | 2 × 2D (one per gate / up, 165 rows of 2.9 MB) | 928 MiB | 17.08 ms | 57.0 |
| s13 scales | 2 × 2D | 58 MiB | 1.08 ms | 56.2 |
| w2 nibbles | 165 × 2D, 576-byte rows at pitch 1,152 | 464 MiB | 9.18 ms | 53.0 |
| **s2 scales** | **165 × 2D, 36-byte rows at pitch 72** | 29 MiB | **44.08 ms** | **0.7** |
| total | | | 71.4 ms/layer → 2.86 s per 8192-token chunk | |

The down-scale slice copy is the whole regression: `cudaMemcpy2DAsync` pays per row, and
5,120 × 165 = 844,800 rows of 36 bytes run at 0.7 GB/s. `window2/dma2d.py` had only measured
the nibble rows (then thought to be 720 bytes) and never the scale rows.

Alternates, all bit-identical to the as-issued result:

| alternate | median | GB/s |
|---|---|---|
| s2: whole rows of every expert in one 2D call (row = one expert's contiguous 368,640-byte E8M0 block) + device slice `[:, :, r*36:(r+1)*36]` | 1.12 ms | 54.3 |
| w2: whole down block in one call + device slice | 18.04 ms | 54.0 (2× the bytes, slower than the 576-byte rows) |
| w2: grouped ×16 whole blocks + slice | 18.15 ms | 53.6 |

Chosen: whole-row s2 + per-expert w2 → **28.5 ms/layer (1.14 s per chunk per rank)**, +58 MiB
VRAM per rank for the `s2_rows` scratch. Overlay `124ca1de` carries it; attempt 4 (`../`)
qualifies it.

Dims used here are the real ones (H = 5120, I = 2304, I_tp = 1152, E = 165, arena stride
22,118,400 B with FP32-sized scale regions compacted in place to E8M0); see
`evidence/t339/README.md` for the H = 4096 / I = 2880 correction.

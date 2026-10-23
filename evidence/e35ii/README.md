# E35-ii — exact decoder SWA bounded replay (2,560-row tail): rejected (2026-10-23)

Overlay branch `e35ii` (worktree `/tmp/sglang-e35ii`): `SGLANG_DSV41_BOUNDED_REPLAY_TAIL=2560`
(the 128-row SWA window x (19 late layers + 1) = the full receptive field), served through
`DECODER_SWA_BOUNDED_REPLAY=1` + `--enable-decoder-swa-bounded-replay`. Two implementation
attempts are in the branch history: the abandon-to-CPU tail (regressed prefill ~50 % - the CPU
tier at 2,560 rows/layer is the cost) and the shrink-through-streamed tail (this measurement:
`layer_begin` updates `forward_tokens` mid-forward, the tail's GEMM runs at the smaller M).

## Numbers (replay2 = shrink-through tail; control = the 1017 headline serve)

| instrument | control | replay (exact tail) | verdict |
|---|---|---|---|
| prefill 16K cold / warm, tok/s | 4,370 / 4,513 (headline) | 2,962 / 3,459 | **-32 % / -23 %** |
| prefill 32K cold / warm, tok/s | 4,370 / 4,513 | 3,377 / 3,560 | **-25 % / -21 %** |
| C1 median / wall tok/s | 59.6 / 57.7 | 61.98 / 55.33 | flat |
| C4 median / wall tok/s | 123.8 / 118.7 | 110.76 / 101.85 | -6..-10 % (2 runs; within the C4 band) |
| verify 16K + needles 18K | ALL PASS | ALL PASS (6/6 @ 17,149, TTFT 9.85 s) | gate held |
| same-prompt C4 | PASS | PASS | gate held |
| stress 4 x 260,096 | wall 276.7 s | wall 410.7 s, all finished, min free 552 / 611 MiB | **-48 %** |
| tf-score fresh (158 positions) | baseline | **+0.0003 +/- 0.0031 nats, greedy 157/158** | **EXACT** (E35's 128-tail: +1.13) |
| tf-score extends (2,400 positions) | baseline | **+0.0514 +/- 0.0069 nats, greedy 2274/2400** | outside the gate (E35: +0.0553) |
| health | 0 | 0 | - |

## Why it loses

The tail layers' streamed DMA copies each layer's CPU-expert weights once per chunk and is
row-count independent, so running those layers at 2,560 rows instead of 8,192 saves GEMM time
but not the DMA - and the dual-count pipeline (8,192 then 2,560 per chunk) plus the tail's KV
writes cost more than the GEMM saves. E35's +39 % was specific to the near-free 128-row tail on
the CPU path, which costs +1.13 nats. The exactness is real but only where the receptive field
is inside the tail: fresh long prompts are bit-exact; short extends (< 2,560 rows) still floor
the window at the extend start, where the prefix's late-layer KV was never written - +0.051
nats, the same mechanism as E35's extend row.

**Bounded replay is closed on this architecture: no tail size gives both exactness and a
prefill win.** The knobs (`DECODER_SWA_BOUNDED_REPLAY` / `BOUNDED_REPLAY_TAIL`) and the
shrink-through-streamed support stay available on the e35ii branch for any future architecture
where the per-layer DMA shrinks with rows; the E35 HTTP-400 refusal fix remains unwritten
(never needed for an experiment that is rejected).

`replay/` = the abandoned-CPU-tail attempt (stamped); `replay2/` = the shrink-through attempt
(prefill, c1-c4, verify, same-prompt, stress, tf.json); `control/` = the control tf-score.

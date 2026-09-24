# T3.16a/b/c — review intake: kt capture-time assertions, launcher validation, early kt size registration (2026-09-24)

Server: launcher defaults (226 / 158, D=4, spec off, chunk 4096, fraction 0.96, capture 1 2 3 4,
`SGLANG_LOGPROB_CHUNK_SIZE=512`, `KT_NAN_PROBE_MIN_TOKENS=1`), image `dsv41-kt:recipe` rebuilt with
`config/kt-kernel-patches/0002-experts-base-capture-assertions.patch`. Owner's 340 GB download to
the separate `/mnt/data` NVMe was running (≈ 0.6 % C1 per E28).

| file | what |
|---|---|
| `launch-patch0002-defaults.txt` | container log (Decode/Prefill batch lines stripped, bridge IP redacted). Shows `KT_BUF capture sizes registered: requested=[1, 2, 3, 4] previously=[] now=[1, 2, 3, 4]` on both ranks and `KT_BUF promote temp buffer bs=4` on TP0 — bs=4 was requested before registration (warmup), the bug-C shape T3.16c removes. Capture end on both ranks; no assertion fired. |
| `verify-c4-patch0002.json` | `verify-batch-consistency --same-prompt --concurrency 4`: PASS |
| `c1-off6-patch0002.json` | C1, 6 × 512 tokens: 41.46 summed / 39.50 wall tok/s, TTFT 0.610 s (E29 control 41.76 / 39.83 / 0.617 — within noise) |
| `nll-patch0002.json` | ΔNLL +0.00182 (z −0.48, 143/148) — bit-identical to the default's E28/E29 reading |

Health: 0 NaN / Xid / Traceback lines, 0 kernel hardware errors, EDAC 0/0, no phantom-busy GPU after teardown.
Launcher validation checked offline with `--print`: D=5 refused (override `KT_ALLOW_UNSAFE_DEFERRAL=1` accepted);
230 at chunk 4096 refused, 230 at chunk 2048 accepted; chunk 8192 at 226 refused; DSpark at 226 refused, at 208 accepted.

## T3.16c — register kt sizes before the autotune forward (overlay `8ad514ed`, same image)

| file | what |
|---|---|
| `launch-t316c.txt` | container log (same redaction). `KT_BUF capture sizes registered … previously=[]` on both ranks at 01:56:56, `FlashInfer autotune completed` at 01:57:57, then the decode runner's re-registration with `cached=[4]` on TP0 (the autotune tuple is graph-owned from birth) and **no `promote` line**. `KT_BUF post-capture: captured=[1, 2, 3, 4] registered=[1, 2, 3, 4] cached=[1, 2, 3, 4]` on TP0 (`cached=[]` on TP1, which runs no CPU experts). |
| `verify-c4-t316c.json` | `verify-batch-consistency --same-prompt --concurrency 4`: PASS |
| `c1-off6-t316c.json` | C1, 6 × 512 tokens: 41.92 summed / 39.86 wall tok/s, TTFT 0.609 s (patch-0002 run above 41.46 / 39.50 / 0.610; E29 control 41.76 / 39.83 / 0.617) |
| `nll-t316c.json` | ΔNLL +0.00182 (z −0.48, 143/148) — bit-identical again |

Health: 0 NaN / Xid / Traceback lines, 0 kernel hardware errors, EDAC 0/0, GPUs idle after teardown. Owner's `/mnt/data` download may still have been running.

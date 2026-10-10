# Attempt 2 (12:07–12:22) — INVALID: two `qual.sh` drivers ran concurrently

Attempt 1's backgrounded `qual.sh` (launched 11:57, image without CUDA bindings) was still
inside `wait-ready.py --timeout 1800` when its container was removed by hand; it kept polling
127.0.0.1:8000, saw attempt 2's server come up at 12:11:42 and ran its full bench sequence
concurrently with attempt 2's, writing into the same directory. Every number here is therefore
a 2x-load measurement, not a window-2 result:

- container log: each C1 run prefilled `#new-seq: 2` (512 tokens = 2 x 256), C4 ran with
  `#running-req: 4, #queue-req: 4`;
- `c4.txt`: median TTFT 18.2 s (1b: 1.5 s); per-seq 29.7 tok/s x 4 reported as 119;
- `stress-kv.txt`: all four streams TTFT 115.7 s (1b staggers 32/59/83/106 s) -> queued behind
  the other driver's four 62K streams;
- `verify.txt`: arithmetic TTFT 7.98 s (1b: 0.47 s), needle TTFT 27.7 s (1b: 20.6 s);
- `prefill-16k/32k`: 1,476 / 1,820 tok/s = two interleaved prefills sharing the GPUs;
- `corpus.txt`: "changed 65211" tokens = different prefill batch composition, not a kernel change.

Rule added to LEARNING §11: `pgrep -af "qual.sh|wait-ready"` must be empty before relaunching a
qualification; a driver's `wait-ready` outlives a hand-removed container.

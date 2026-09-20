# Troubleshooting

## `preflight.sh` fails with "checkpoint does not fit"

This is the expected outcome on two RTX PRO 6000. It is not a configuration problem. The calculator output above the failure line shows the margin (about −112 GiB with Engram in host DRAM). Options: see [FEASIBILITY.md §4](FEASIBILITY.md). To run experiment E5 anyway, set `ALLOW_UNQUALIFIED=1` and expect the container to die during weight load with a CUDA OOM at roughly 140 GiB demanded per GPU.

## Hugepages warning from `preflight.sh`

SGLang's Engram host table (`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`) allocates the ~189 GiB FP8 tables in hugepage-backed memory. Reserve at runtime (reversible; fails if memory is fragmented, so do it soon after boot or while nothing large is running):

```bash
# 2 MiB pages: 190 GiB = 97,280 pages
sudo sysctl -w vm.nr_hugepages=97280
grep HugePages_ /proc/meminfo
```

Or at boot in the kernel command line: `hugepagesz=1G hugepages=190`. This host has 246.8 GiB; after 190 GiB of hugepages about 56 GiB remains for the OS and the serving process, which is enough for SGLang's host-side buffers but not for a second large model. Revert with `vm.nr_hugepages=0`.

If the GLM container is running, `docker stop glm53-flash-2x-rtxpro6000` first; it pins host memory.

## SM120 pitfalls reported by the community for DeepSeek-V4 / V4.1

From vLLM issues #47266 and #56700 and the 0xSero SGLang recipe. Not all reproduce on every build; check before applying.

| Symptom | Cause | Mitigation |
|---|---|---|
| Server hangs at startup with no log progress | DeepGEMM / FlashInfer autotune on SM120 | Disable autotune for the affected backend; SGLang cookbook says leave backends on auto |
| Crash during CUDA graph capture | Capture pool exceeds memory or unsupported kernel in graph | `--enforce-eager` (vLLM) / `--disable-cuda-graph` (SGLang); expect lower decode throughput |
| Wrong answers only on prompts ≥ 100K tokens | FP8 KV scale layout bug | Use `KV_CACHE_DTYPE=auto` (BF16) or a build that includes the fix; `verify.py --context 262144` catches it |
| MXFP4 experts run slowly | Marlin fallback instead of native FP4 grouped GEMM (vLLM #30135) | Check the kernel selection in the server log; compare against SGLang's FlashInfer MXFP4 path |
| All-reduce errors or NCCL hang on PCIe boxes | Custom all-reduce on non-NVLink topology | `--disable-custom-all-reduce` (vLLM); confirm `nvidia-smi topo -p2p r` shows OK |
| Sparse MLA dispatch error | FlashInfer SM120 dispatch gap for some head configurations | Use the SGLang preview image, which the 4-GPU recipe proved on SM120 |

## Thinking mode surprises

DeepSeek-V4.1 reasoning is on by default at effort 50 in both engines. Every client in this repo sends `chat_template_kwargs: {"thinking": false}` unless asked otherwise. If your own client sees long, slow answers, that is why. The recipe's smoke test is `17*19` → `323`.

## Weight load is slow

475 GiB from NVMe at ~14 GB/s is at least 33 s of pure I/O; the engine's dequant and layout passes dominate, and the Engram host-table copy adds a pass over 189 GiB. Ten to thirty minutes is normal on the 4-GPU profile. `wait-ready.py` defaults to a one-hour timeout.

## `download-model.sh` refuses to start

It needs ~500 GiB free at `MODEL_DIR`. On this host nvme0 has 1.7 TB free; make sure `MODEL_DIR` is on it and not on a smaller volume. The checkpoint interleaves vision, DSpark and text tensors across all 48 shards, so a partial download cannot save space.

## `ci.sh` reports "possible identifier or secret"

The scan catches home paths, LAN addresses, GPU UUIDs, PCI bus IDs and token-shaped strings in tracked files. Scrub the file (replace with placeholders) rather than whitelisting it; evidence JSON is meant to be publishable.

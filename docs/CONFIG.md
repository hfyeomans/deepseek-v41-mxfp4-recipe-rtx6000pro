# Launcher configuration registry

Every knob `scripts/serve-hybrid.sh` honours is declared once in
[`scripts/serve_config.py`](../scripts/serve_config.py) (T3.19e): type, default, where the value
goes (`server-arg` → an `sglang.launch_server` flag, `container-env` → always passed with `-e`,
`diagnostic-env` → passed with `-e` only when you set it, `launcher` → used by the script itself,
`guard` → exists only to bypass a refusal) and the evidence that set the default. The launcher
evals the registry's output, so the `resolved:` line and the `FAIL config:` refusals come from
one place; `scripts/tests/test_serve_config.py` pins both to goldens.

Rules the registry enforces before any docker call:

- an unknown `KT_*` / `DSPARK*` variable in the environment is refused (a mistyped knob used to
  run the default while the log claimed otherwise); `ALLOW_UNKNOWN_KNOBS=1` passes it through;
- a removed knob is refused with the experiment that removed it;
- a value that does not parse as its type is refused;
- the T3.16b / E34d / T3.29 fit and quality refusals apply (each names its override);
- `MAX_TOTAL_TOKENS=""` and `CUDA_GRAPH_BS=""` mean "let SGLang derive it".

Probe without starting anything: `DRY_RUN=1 scripts/serve-hybrid.sh` (resolved line only) or
`scripts/serve-hybrid.sh --print` (resolved line + the docker command). Regenerate the table
below with `python3 scripts/serve_config.py markdown`; `scripts/ci.sh` fails when it is stale.

<!-- generated: python3 scripts/serve_config.py markdown -->
| knob | type | default | sink | doc | evidence |
|---|---|---|---|---|---|
| `IMAGE` | str | `dsv41-kt:recipe` | launcher | Image built by config/Dockerfile.dsv41-kt (dev-dsv41 + pruned kt-kernel). |  |
| `CONTAINER_NAME` | str | `dsv41-kt` | launcher | docker container name; `docker rm -f` it to stop. |  |
| `SERVED_MODEL_NAME` | str | `dsv41-flash-hybrid` | server-arg | Name the OpenAI-compatible API serves. |  |
| `BIND_ADDRESS` | str | `127.0.0.1` | launcher | Host interface the port is published on. |  |
| `PORT` | int | `8000` | launcher | Host port mapped to the container's 30000. |  |
| `HF_MODEL_DIR` | str | `derived` | launcher | HF hub model dir (bind-mounted read-only at /hfm). |  |
| `SNAPSHOT` | str | `dba1be0a40aa45a94ad051997016db3960a90277` | launcher | Checkpoint snapshot under HF_MODEL_DIR/snapshots. | PROVENANCE.md |
| `OVERLAY_DIR` | str | `derived` | launcher | SGLang overlay checkout; its python/sglang is bind-mounted over the image's. |  |
| `MANIFEST_DIR` | str | `derived` | launcher | Where each launch writes its evidence manifest (scripts/evidence_manifest.py); current.json is the one the evidence scripts stamp with. |  |
| `BIOS_SLOT` | str | `3` | launcher | BIOS profile the operator booted, as declared (not read from firmware): 3 = FCLK 2100 / VSOC 1.05 V / VDDG 950 mV / EXPO 6400 (BIOS guide T3-F1v, Boot 12). The manifest records it beside the HSMP-measured FCLK/MCLK so a mismatch is visible. | docs/BIOS-OPTIMIZATION-GUIDE.md |
| `KT_NUM_GPU_EXPERTS` | int | `219` | server-arg | Routed experts per layer resident on the GPUs (contiguous IDs 0..N-1 of 384; the rest run on the CPU through kt-kernel). ~0.33 GB per expert per rank. 208 -> 226 at T3.6g once the KV pool was capped; 230 rejected (E29: needs chunk 2048 + fraction 0.975, fails the NLL gate); 226 -> 219 on 2026-10-08 (E34d-ii + T3.28) to afford CHUNKED_PREFILL_SIZE=8192: prefill +38/+47 % at 16K/32K, C1 wall -1.9 %, pooled NLL +0.00075 +- 0.00098. 226/158 + 4096 is the measured fallback. | evidence/t328/README.md, evidence/t36g-placement/, evidence/t36g2-placement-r2/ |
| `KT_CPUINFER` | int | `16` | server-arg | kt-kernel worker threads; one per physical core. E26 A1 on the 16-core 9955WX: 14 and 16 are a plateau, 24 costs -12 %, 30 collapses to 5 tok/s (workers beyond the core count float on SMT siblings and starve the two TP-rank launch threads). Never above the physical cores (warned at resolve time). | evidence/e26-single-user/ |
| `KT_MAX_DEFERRED` | int | `4` | server-arg | Expert deferral D (E21): each token's D lowest-scored routed experts are computed during the next layer and added one layer late. D=4: +22 % C1 over D=0, NLL within gate (T3.28: +0.00066 +- 0.00103 on the forced CPU path). D=5 fails the NLL gate (+0.00287, T3.15) and greedy consistency; D=6 loses 0.23 nats. 0 disables. | evidence/e21-deferral/, evidence/t315-d5/, evidence/t328/ |
| `KT_GPU_PREFILL_MIN_TOKENS` | int | `1024` | container-env | Layerwise GPU prefill (E22): chunks with at least this many tokens stream each layer's CPU experts to the GPUs (kt export -> pinned buffer -> H2D -> FlashInfer MoE). 0 keeps every chunk on the CPU path. Streamed prefill is NLL-neutral vs the CPU path (T3.28: -0.00025 +- 0.00104). | evidence/e22-gpu-prefill/, evidence/t328/ |
| `KT_GPU_PREFILL_LOG` | flag | `0` | container-env | Log per-chunk streamed-prefill timing. |  |
| `KT_NAN_PROBE_MIN_TOKENS` | int | `0` | container-env | Per-layer non-finite check on forwards of >= N MoE tokens (E22d/E24 diagnosis; 0 = off; 1 in every qualification serve). | evidence/e24-nan/ |
| `DSPARK` | flag | `0` | server-arg | DSPARK=1 adds the three MTP draft stages (--speculative-algorithm DSPARK). Off: E23 measured it a net loss under the CPU-bound MoE; drafts cost ~3.85 GB per rank. | evidence/e23-spec-decode/ |
| `DSPARK_K` | int | `5` | server-arg | DSPARK block size; tested 1..5 (E23 R0-R3). | evidence/e23-spec-decode/ |
| `NGRAM` | flag | `0` | server-arg | NGRAM=1 turns on SGLang n-gram speculation (--speculative-algorithm NGRAM: drafts come from a suffix cache of the request's own tokens, no draft model, no extra weights). E23 R6: pays only where the output copies spans of the prompt; a verify of N rows routes N rows' worth of CPU experts. Exclusive with DSPARK. | evidence/e23-spec-decode/r6-ngram/ |
| `NGRAM_DRAFT_TOKENS` | int | `5` | server-arg | --speculative-num-draft-tokens for NGRAM (verify window incl. the root; SGLang's default is 12, which would route 12 rows of CPU experts per step; 5 matches DSpark's K=4 window). | evidence/e23-spec-decode/r6-ngram/ |
| `NGRAM_BREADTH` | int | `1` | server-arg | --speculative-ngram-max-bfs-breadth for NGRAM (1 = one chain per step; SGLang's default 10 builds a tree, more verify rows). | evidence/e23-spec-decode/r6-ngram/ |
| `MEM_FRACTION_STATIC` | float | `0.96` | server-arg | SGLang static fraction. 0.92 -> 0.96 at T3.6g (the pool is capped by MAX_TOTAL_TOKENS, so the fraction only has to leave room for prefill transients and capture). 0.962 is the qualified value for the 512K x 1 rung, where 0.96 silently caps the pool at 505,856 tokens and 400s longer inputs. | evidence/e34-indexer-memory/b-ii/512k/, evidence/t329/ |
| `CONTEXT_LENGTH` | int | `65536` | server-arg | Served context. 32K -> 64K on 2026-10-07 (E34a + soak). 128K/262K/512K x 1 are qualified rungs (T3.29 on the 219/8192 default; 512K needs MEM_FRACTION_STATIC=0.962); above 64K only one slot fits. | evidence/e34-indexer-memory/, evidence/t329/README.md |
| `CHUNKED_PREFILL_SIZE` | int | `8192` | server-arg | Prefill chunk. 4096 -> 8192 on 2026-10-08 together with 226 -> 219 experts (E34d-ii, T3.28): first-turn TTFT is what chunk buys for long coding sessions. 16384 is +22 % prefill but costs 14 experts (E34d-iv); 2048 frees ~1.5 GB per rank but costs 40 % of prefill (E29). The fused-MoE prefill workspace scales with the chunk (E34d), so 8192 only fits beside <= 219 experts. | evidence/e34-indexer-memory/d-ii/, evidence/t328/README.md |
| `MAX_RUNNING_REQUESTS` | int | `4` | server-arg | Decode slots. Above 64K context only 1 is qualified (T3.29). | evidence/t329/README.md |
| `SGLANG_LOGPROB_CHUNK_SIZE` | int | `512` | container-env | A return_logprob request materialises chunk x 129,280-vocab fp32 logits (1.06 GB at SGLang's 2048) and a CUDA OOM there kills the server, not the request (E29). 512 is bit-identical and no slower. | evidence/t36g2-placement-r2/ |
| `MAX_TOTAL_TOKENS` | int | `derived` | server-arg | KV pool cap (T3.6g): slots x context + one 16K chunk. Left to the fraction SGLang sized the pool 15x too large (E26 A4). MAX_TOTAL_TOKENS="" lets the fraction size it again. | evidence/e26-single-user/ |
| `CUDA_GRAPH_MAX_BS` | int | `4` | server-arg | Largest captured decode batch. |  |
| `CUDA_GRAPH_BS` | list | `1 2 3 4` | server-arg | Explicit decode capture list. SGLang derives [1, 2, 4] and pads bs=3 to 4, which cost C3 22 % (E26 B1); capturing 3 costs ~nothing. Qualified by the E27 served soak. CUDA_GRAPH_BS="" falls back to the derived list. | evidence/e26-single-user/, evidence/e27-soak/ |
| `ENGRAM_FILE_PREFETCH` | str | `ahead` | container-env | How the Engram rows left in the checkpoint shard are fetched (T3.20/E36): ahead = layer 14's rows advised and gathered on a worker thread from layer 1's break; sync = serial MADV_WILLNEED per distinct page in the break; pool = the E18 per-row thread pool (A/B comparator). | evidence/t320-engram-prefetch/ |
| `ENGRAM_FILE_THREADS` | int | `16` | container-env | Threads of ENGRAM_FILE_PREFETCH=pool only (E18). | evidence/e18-engram-file/ |
| `EXTRA_SERVER_ARGS` | str | `""` | launcher | Appended verbatim (word-split) to the sglang.launch_server command. |  |
| `EXTRA_DOCKER_ENV` | str | `""` | launcher | Space-separated KEY=VALUE pairs passed as -e (e.g. CUDA_LAUNCH_BLOCKING=1). |  |
| `EXTRA_DOCKER_ARGS` | str | `""` | launcher | Extra `docker run` flags, word-split (diagnosis: bind-mount an instrumented module). |  |
| `DRY_RUN` | flag | `0` | guard | Stop after the resolved line, before any docker call (guard probes, config tests). |  |
| `ALLOW_UNTESTED_FIT` **(unsafe)** | flag | `0` | guard | Bypass the VRAM-fit refusals below to run an experiment on purpose. |  |
| `KT_ALLOW_UNSAFE_DEFERRAL` **(unsafe)** | flag | `0` | guard | Bypass the D>=5 refusal (fails the NLL gate, T3.15). |  |
| `ALLOW_UNKNOWN_KNOBS` **(unsafe)** | flag | `0` | guard | Accept KT_* / DSPARK* variables this registry does not declare. |  |
| `SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE` | flag | `1` | container-env | T3.31 idea 8: after CUDA-graph capture, empty the cache and switch the allocator to expandable segments so chunk-sized prefill temporaries stop fragmenting (262K x 4 at chunk 8192 OOMed on a 320 MiB request with 1.05 GiB reserved-but-unallocated). Graph pools, weights and the KV pool stay ordinary segments. Default since 2026-10-09 on the T3.31 64K x 4 soak (NLL 0 moved, C1/C4/prefill within noise, +390 MiB min free under stress); set 0 for the pre-flip allocator. | evidence/t331/64k4-lever-soak/, evidence/e34-indexer-memory/d-v-a/262k-4slot-idea8/ |
| `KT_TIMELINE` | flag | `0` | diagnostic-env | T3.19a per-layer decode timeline recorder (kt_timeline.py). | evidence/t319/a/ |
| `KT_TIMELINE_DIR` | str | `""` | diagnostic-env | Where the timeline recorder writes (container path). |  |
| `KT_TIMELINE_SAMPLE` | int | `0` | diagnostic-env | Record every Nth step. |  |
| `KT_TIMELINE_MAX_RECORDS` | int | `0` | diagnostic-env | Stop recording after N records. |  |
| `KT_TIMELINE_DUMP_MAX` | int | `0` | diagnostic-env | Max tensor dumps. |  |
| `KT_TIMELINE_DUMP_MAX_TOKENS` | int | `0` | diagnostic-env | Only dump forwards of <= N tokens. |  |
| `KT_TIMELINE_DUMP_LAYERS` | str | `""` | diagnostic-env | Comma-separated layer ids to dump. |  |
| `KT_FORCE_SYNC_SUBMIT` | flag | `0` | diagnostic-env | kt_provider: submit CPU expert work synchronously (parity debugging). |  |
| `SGLANG_BCG_DEBUG_SYNC` | flag | `0` | diagnostic-env | Breakable CUDA graph: synchronise after every segment. |  |
| `SGLANG_BCG_DEBUG_EAGER` | flag | `0` | diagnostic-env | Breakable CUDA graph: run segments eagerly. |  |
| `SGLANG_BCG_DEBUG_INPUTS` | flag | `0` | diagnostic-env | Breakable CUDA graph: log segment inputs. |  |
| `SGLANG_BCG_DEBUG_LAYER_BREAKS` | flag | `0` | diagnostic-env | Breakable CUDA graph: log where the graph breaks. |  |
| `KT_B2_UVA_EXPERTS` *(removed)* | str | `""` | guard | KT_B2_UVA_EXPERTS / KT_B2_MAX_TOKENS were removed at T3.16j (B2 rejected, E30); unset them. |  |
| `KT_B2_MAX_TOKENS` *(removed)* | str | `""` | guard | KT_B2_UVA_EXPERTS / KT_B2_MAX_TOKENS were removed at T3.16j (B2 rejected, E30); unset them. |  |

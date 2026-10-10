#!/usr/bin/env python3
"""Typed knob registry for scripts/serve-hybrid.sh (T3.19e).

Every environment knob the launcher or the overlay reads is declared once here: type,
default, aliases, where it goes (server arg / container env / launcher / diagnostic) and
the evidence that set its default. The launcher evals the output of `shell` instead of
carrying its own `${VAR:-default}` lines and `fail_config` block; the guards below are the
T3.16b / E34d / T3.29 refusals, moved verbatim, and `scripts/tests/test-serve-config.sh`
plus `scripts/tests/test_serve_config.py` pin their messages and the `resolved:` line.

    python3 scripts/serve_config.py shell          # KEY='value' lines for `eval`, or rc 2
    python3 scripts/serve_config.py resolved       # the one-line resolved config
    python3 scripts/serve_config.py markdown       # knob table for the docs
    python3 scripts/serve_config.py check-overlay DIR   # every env the overlay reads is declared

Rules: an unknown `KT_*` / `DSPARK*` / `NGRAM*` variable in the environment is refused (rc 2) unless
`ALLOW_UNKNOWN_KNOBS=1`; a removed knob is refused with the experiment that removed it;
a value that does not parse as its type is refused. No docker, no GPU, no side effects.
"""
from __future__ import annotations

import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

EXPERTS_PER_LAYER = 384
OWNED_PREFIXES = ("KT_", "DSPARK", "NGRAM")

# sinks: where the resolved value is consumed
SERVER_ARG = "server-arg"      # becomes an sglang.launch_server flag
CONTAINER_ENV = "container-env"  # always forwarded with -e
DIAGNOSTIC_ENV = "diagnostic-env"  # forwarded with -e only when set by the caller
LAUNCHER = "launcher"          # used by serve-hybrid.sh itself (docker run, paths, guards)
GUARD = "guard"                # exists only to bypass a refusal


@dataclass(frozen=True)
class Knob:
    name: str
    kind: str                       # int | float | str | flag | list
    default: Any                    # literal, or callable(resolved: dict) -> str
    doc: str
    sink: str = LAUNCHER
    evidence: str = ""
    aliases: tuple[str, ...] = ()
    unsafe: bool = False            # bypasses a guard; never a deployment value
    empty_disables: bool = False    # `${VAR-default}` semantics: VAR="" means off
    container_name: str = ""        # env name inside the container when it differs
    removed: str = ""               # non-empty: refuse with this message


def _home(p: str) -> str:
    return os.path.join(os.environ.get("HOME", "~"), p)


REGISTRY: tuple[Knob, ...] = (
    # ---- image / paths / network --------------------------------------------------------
    Knob("IMAGE", "str", "dsv41-kt:recipe", "Image built by config/Dockerfile.dsv41-kt (dev-dsv41 + pruned kt-kernel)."),
    Knob("CONTAINER_NAME", "str", "dsv41-kt", "docker container name; `docker rm -f` it to stop."),
    Knob("SERVED_MODEL_NAME", "str", "dsv41-flash-hybrid", "Name the OpenAI-compatible API serves.", sink=SERVER_ARG),
    Knob("BIND_ADDRESS", "str", "127.0.0.1", "Host interface the port is published on."),
    Knob("PORT", "int", "8000", "Host port mapped to the container's 30000."),
    Knob("HF_MODEL_DIR", "str", lambda r: _home(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash"),
         "HF hub model dir (bind-mounted read-only at /hfm)."),
    Knob("SNAPSHOT", "str", "dba1be0a40aa45a94ad051997016db3960a90277", "Checkpoint snapshot under HF_MODEL_DIR/snapshots.",
         evidence="PROVENANCE.md"),
    Knob("OVERLAY_DIR", "str", lambda r: _home("dev/src/sglang-dsv41-overlay"),
         "SGLang overlay checkout; its python/sglang is bind-mounted over the image's."),
    # ---- evidence manifest (T3.19f) ----------------------------------------------------
    Knob("MANIFEST_DIR", "str", lambda r: _home(".cache/dsv41-recipe/manifests"),
         "Where each launch writes its evidence manifest (scripts/evidence_manifest.py); current.json is the "
         "one the evidence scripts stamp with."),
    Knob("BIOS_SLOT", "str", "3",
         "BIOS profile the operator booted, as declared (not read from firmware): 3 = FCLK 2100 / VSOC 1.05 V / "
         "VDDG 950 mV / EXPO 6400 (BIOS guide T3-F1v, Boot 12). The manifest records it beside the HSMP-measured "
         "FCLK/MCLK so a mismatch is visible.", evidence="docs/BIOS-OPTIMIZATION-GUIDE.md"),
    # ---- expert placement and CPU tier --------------------------------------------------
    Knob("KT_NUM_GPU_EXPERTS", "int", "219",
         "Routed experts per layer resident on the GPUs (contiguous IDs 0..N-1 of 384; the rest run on the "
         "CPU through kt-kernel). ~0.33 GB per expert per rank. 208 -> 226 at T3.6g once the KV pool was capped; "
         "230 rejected (E29: needs chunk 2048 + fraction 0.975, fails the NLL gate); 226 -> 219 on 2026-10-08 "
         "(E34d-ii + T3.28) to afford CHUNKED_PREFILL_SIZE=8192: prefill +38/+47 % at 16K/32K, C1 wall -1.9 %, "
         "pooled NLL +0.00075 +- 0.00098. 226/158 + 4096 is the measured fallback.",
         sink=SERVER_ARG, evidence="evidence/t328/README.md, evidence/t36g-placement/, evidence/t36g2-placement-r2/"),
    Knob("KT_CPUINFER", "int", "16",
         "kt-kernel worker threads; one per physical core. E26 A1 on the 16-core 9955WX: 14 and 16 are a plateau, "
         "24 costs -12 %, 30 collapses to 5 tok/s (workers beyond the core count float on SMT siblings and starve "
         "the two TP-rank launch threads). Never above the physical cores (warned at resolve time). T3.39: 14 read +9 % on "
         "16K prefill while the FlashInfer workspace stall held the layer period at 90 ms (window 1); on the fixed "
         "overlay 99128a7e the soak reads 14 at -11 to -25 % prefill (the core-bound kt export is now exposed), decode "
         "within noise. 16 stays.",
         sink=SERVER_ARG, evidence="evidence/e26-single-user/, evidence/t339/"),
    Knob("KT_MAX_DEFERRED", "int", "4",
         "Expert deferral D (E21): each token's D lowest-scored routed experts are computed during the next layer "
         "and added one layer late. D=4: +22 % C1 over D=0, NLL within gate (T3.28: +0.00066 +- 0.00103 on the "
         "forced CPU path). D=5 fails the NLL gate (+0.00287, T3.15) and greedy consistency; D=6 loses 0.23 nats. "
         "0 disables.", sink=SERVER_ARG, evidence="evidence/e21-deferral/, evidence/t315-d5/, evidence/t328/"),
    Knob("KT_GPU_PREFILL_MIN_TOKENS", "int", "1024",
         "Layerwise GPU prefill (E22): chunks with at least this many tokens stream each layer's CPU experts to "
         "the GPUs (kt export -> pinned buffer -> H2D -> FlashInfer MoE). 0 keeps every chunk on the CPU path. "
         "Streamed prefill is NLL-neutral vs the CPU path (T3.28: -0.00025 +- 0.00104).",
         sink=CONTAINER_ENV, evidence="evidence/e22-gpu-prefill/, evidence/t328/"),
    Knob("KT_GPU_PREFILL_LOG", "flag", "0", "Log per-chunk streamed-prefill timing.", sink=CONTAINER_ENV),
    Knob("KT_NAN_PROBE_MIN_TOKENS", "int", "0",
         "Per-layer non-finite check on forwards of >= N MoE tokens (E22d/E24 diagnosis; 0 = off; 1 in every "
         "qualification serve).", sink=CONTAINER_ENV, evidence="evidence/e24-nan/"),
    # ---- speculative decoding -----------------------------------------------------------
    Knob("DSPARK", "flag", "0",
         "DSPARK=1 adds the three MTP draft stages (--speculative-algorithm DSPARK). Off: E23 measured it a net "
         "loss under the CPU-bound MoE; drafts cost ~3.85 GB per rank.", sink=SERVER_ARG,
         evidence="evidence/e23-spec-decode/"),
    Knob("DSPARK_K", "int", "5", "DSPARK block size; tested 1..5 (E23 R0-R3).", sink=SERVER_ARG,
         evidence="evidence/e23-spec-decode/"),
    Knob("NGRAM", "flag", "0",
         "NGRAM=1 turns on SGLang n-gram speculation (--speculative-algorithm NGRAM: drafts come from a suffix "
         "cache of the request's own tokens, no draft model, no extra weights). E23 R6: pays only where the "
         "output copies spans of the prompt; a verify of N rows routes N rows' worth of CPU experts. Exclusive "
         "with DSPARK.", sink=SERVER_ARG, evidence="evidence/e23-spec-decode/r6-ngram/"),
    Knob("NGRAM_DRAFT_TOKENS", "int", "5",
         "--speculative-num-draft-tokens for NGRAM (verify window incl. the root; SGLang's default is 12, which "
         "would route 12 rows of CPU experts per step; 5 matches DSpark's K=4 window).", sink=SERVER_ARG,
         evidence="evidence/e23-spec-decode/r6-ngram/"),
    Knob("NGRAM_BREADTH", "int", "1",
         "--speculative-ngram-max-bfs-breadth for NGRAM (1 = one chain per step; SGLang's default 10 builds a "
         "tree, more verify rows).", sink=SERVER_ARG, evidence="evidence/e23-spec-decode/r6-ngram/"),
    # ---- memory, context, prefill -------------------------------------------------------
    Knob("MEM_FRACTION_STATIC", "float", "0.96",
         "SGLang static fraction. 0.92 -> 0.96 at T3.6g (the pool is capped by MAX_TOTAL_TOKENS, so the fraction "
         "only has to leave room for prefill transients and capture). 0.962 is the qualified value for the 512K x 1 "
         "rung, where 0.96 silently caps the pool at 505,856 tokens and 400s longer inputs.",
         sink=SERVER_ARG, evidence="evidence/e34-indexer-memory/b-ii/512k/, evidence/t329/"),
    Knob("CONTEXT_LENGTH", "int", "65536",
         "Served context. 32K -> 64K on 2026-10-07 (E34a + soak). 128K/262K/512K x 1 are qualified rungs "
         "(T3.29 on the 219/8192 default; 512K needs MEM_FRACTION_STATIC=0.962); above 64K only one slot fits.",
         sink=SERVER_ARG, evidence="evidence/e34-indexer-memory/, evidence/t329/README.md"),
    Knob("CHUNKED_PREFILL_SIZE", "int", "8192",
         "Prefill chunk. 4096 -> 8192 on 2026-10-08 together with 226 -> 219 experts (E34d-ii, T3.28): first-turn "
         "TTFT is what chunk buys for long coding sessions. 16384 is +22 % prefill but costs 14 experts (E34d-iv); "
         "2048 frees ~1.5 GB per rank but costs 40 % of prefill (E29). The fused-MoE prefill workspace scales with "
         "the chunk (E34d), so 8192 only fits beside <= 219 experts.",
         sink=SERVER_ARG, evidence="evidence/e34-indexer-memory/d-ii/, evidence/t328/README.md"),
    Knob("MAX_RUNNING_REQUESTS", "int", "4", "Decode slots. Above 64K context only 1 is qualified (T3.29).",
         sink=SERVER_ARG, evidence="evidence/t329/README.md"),
    Knob("SGLANG_LOGPROB_CHUNK_SIZE", "int", "512",
         "A return_logprob request materialises chunk x 129,280-vocab fp32 logits (1.06 GB at SGLang's 2048) and "
         "a CUDA OOM there kills the server, not the request (E29). 512 is bit-identical and no slower.",
         sink=CONTAINER_ENV, evidence="evidence/t36g2-placement-r2/"),
    Knob("MAX_TOTAL_TOKENS", "int",
         lambda r: str(int(r["MAX_RUNNING_REQUESTS"]) * int(r["CONTEXT_LENGTH"]) + 16384),
         "KV pool cap (T3.6g): slots x context + one 16K chunk. Left to the fraction SGLang sized the pool 15x too "
         "large (E26 A4). MAX_TOTAL_TOKENS=\"\" lets the fraction size it again.",
         sink=SERVER_ARG, empty_disables=True, evidence="evidence/e26-single-user/"),
    Knob("CUDA_GRAPH_MAX_BS", "int", "4", "Largest captured decode batch.", sink=SERVER_ARG),
    Knob("CUDA_GRAPH_BS", "list", "1 2 3 4",
         "Explicit decode capture list. SGLang derives [1, 2, 4] and pads bs=3 to 4, which cost C3 22 % (E26 B1); "
         "capturing 3 costs ~nothing. Qualified by the E27 served soak. CUDA_GRAPH_BS=\"\" falls back to the derived list.",
         sink=SERVER_ARG, empty_disables=True, evidence="evidence/e26-single-user/, evidence/e27-soak/"),
    Knob("ENGRAM_FILE_PREFETCH", "str", "ahead",
         "How the Engram rows left in the checkpoint shard are fetched (T3.20/E36): ahead = layer 14's rows advised and "
         "gathered on a worker thread from layer 1's break; sync = serial MADV_WILLNEED per distinct page in the break; "
         "pool = the E18 per-row thread pool (A/B comparator).",
         sink=CONTAINER_ENV, container_name="SGLANG_DSV41_ENGRAM_FILE_PREFETCH", evidence="evidence/t320-engram-prefetch/"),
    Knob("ENGRAM_FILE_THREADS", "int", "16", "Threads of ENGRAM_FILE_PREFETCH=pool only (E18).",
         sink=CONTAINER_ENV, container_name="SGLANG_DSV41_ENGRAM_FILE_THREADS", evidence="evidence/e18-engram-file/"),
    # ---- escape hatches -----------------------------------------------------------------
    Knob("EXTRA_SERVER_ARGS", "str", "", "Appended verbatim (word-split) to the sglang.launch_server command."),
    Knob("EXTRA_DOCKER_ENV", "str", "", "Space-separated KEY=VALUE pairs passed as -e (e.g. CUDA_LAUNCH_BLOCKING=1)."),
    Knob("EXTRA_DOCKER_ARGS", "str", "", "Extra `docker run` flags, word-split (diagnosis: bind-mount an instrumented module)."),
    Knob("DRY_RUN", "flag", "0", "Stop after the resolved line, before any docker call (guard probes, config tests).", sink=GUARD),
    # ---- guard bypasses (never deployment values) ---------------------------------------
    Knob("ALLOW_UNTESTED_FIT", "flag", "0", "Bypass the VRAM-fit refusals below to run an experiment on purpose.",
         sink=GUARD, unsafe=True),
    Knob("KT_ALLOW_UNSAFE_DEFERRAL", "flag", "0", "Bypass the D>=5 refusal (fails the NLL gate, T3.15).", sink=GUARD, unsafe=True),
    Knob("ALLOW_UNKNOWN_KNOBS", "flag", "0", "Accept KT_* / DSPARK* variables this registry does not declare.", sink=GUARD, unsafe=True),
    # ---- overlay allocator lever (always forwarded; default-on since 2026-10-09) ---------
    Knob("SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE", "flag", "1",
         "T3.31 idea 8: after CUDA-graph capture, empty the cache and switch the allocator to expandable segments "
         "so chunk-sized prefill temporaries stop fragmenting (262K x 4 at chunk 8192 OOMed on a 320 MiB request "
         "with 1.05 GiB reserved-but-unallocated). Graph pools, weights and the KV pool stay ordinary segments. "
         "Default since 2026-10-09 on the T3.31 64K x 4 soak (NLL 0 moved, C1/C4/prefill within noise, "
         "+390 MiB min free under stress); set 0 for the pre-flip allocator.",
         sink=CONTAINER_ENV, evidence="evidence/t331/64k4-lever-soak/, evidence/e34-indexer-memory/d-v-a/262k-4slot-idea8/"),
    # ---- overlay diagnostics (forwarded into the container only when set) ---------------
    Knob("KT_TIMELINE", "flag", "0", "T3.19a per-layer decode timeline recorder (kt_timeline.py).", sink=DIAGNOSTIC_ENV,
         evidence="evidence/t319/a/"),
    Knob("KT_TIMELINE_DIR", "str", "", "Where the timeline recorder writes (container path).", sink=DIAGNOSTIC_ENV),
    Knob("KT_TIMELINE_SAMPLE", "int", "0", "Record every Nth step.", sink=DIAGNOSTIC_ENV),
    Knob("KT_TIMELINE_MAX_RECORDS", "int", "0", "Stop recording after N records.", sink=DIAGNOSTIC_ENV),
    Knob("KT_TIMELINE_DUMP_MAX", "int", "0", "Max tensor dumps.", sink=DIAGNOSTIC_ENV),
    Knob("KT_TIMELINE_DUMP_MAX_TOKENS", "int", "0", "Only dump forwards of <= N tokens.", sink=DIAGNOSTIC_ENV),
    Knob("KT_TIMELINE_DUMP_LAYERS", "str", "", "Comma-separated layer ids to dump.", sink=DIAGNOSTIC_ENV),
    Knob("KT_FORCE_SYNC_SUBMIT", "flag", "0", "kt_provider: submit CPU expert work synchronously (parity debugging).", sink=DIAGNOSTIC_ENV),
    Knob("SGLANG_BCG_DEBUG_SYNC", "flag", "0", "Breakable CUDA graph: synchronise after every segment.", sink=DIAGNOSTIC_ENV),
    Knob("SGLANG_BCG_DEBUG_EAGER", "flag", "0", "Breakable CUDA graph: run segments eagerly.", sink=DIAGNOSTIC_ENV),
    Knob("SGLANG_BCG_DEBUG_INPUTS", "flag", "0", "Breakable CUDA graph: log segment inputs.", sink=DIAGNOSTIC_ENV),
    Knob("SGLANG_BCG_DEBUG_LAYER_BREAKS", "flag", "0", "Breakable CUDA graph: log where the graph breaks.", sink=DIAGNOSTIC_ENV),
    # ---- removed ------------------------------------------------------------------------
    Knob("KT_B2_UVA_EXPERTS", "str", "", "", sink=GUARD,
         removed="KT_B2_UVA_EXPERTS / KT_B2_MAX_TOKENS were removed at T3.16j (B2 rejected, E30); unset them."),
    Knob("KT_B2_MAX_TOKENS", "str", "", "", sink=GUARD,
         removed="KT_B2_UVA_EXPERTS / KT_B2_MAX_TOKENS were removed at T3.16j (B2 rejected, E30); unset them."),
)

BY_NAME: dict[str, Knob] = {k.name: k for k in REGISTRY}
for _k in REGISTRY:
    for _a in _k.aliases:
        assert _a not in BY_NAME, _a
        BY_NAME[_a] = _k

# Container env the launcher always sets, in `docker run` order (the --print goldens pin it).
# Fixed values are not knobs: the Engram tables stay in the checkpoint shard on NVMe (E18,
# overlay a1ca246a) and the SM120 indexer defaults pick paths V4.1 cannot take (TROUBLESHOOTING).
def container_env(r: dict[str, str]) -> list[str]:
    snapshot_dir = f"/hfm/snapshots/{r['SNAPSHOT']}"
    return [
        "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1",
        "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=file",
        f"SGLANG_DSV41_ENGRAM_FILE_DIR={snapshot_dir}",
        f"SGLANG_DSV41_ENGRAM_FILE_PREFETCH={r['ENGRAM_FILE_PREFETCH']}",
        f"SGLANG_DSV41_ENGRAM_FILE_THREADS={r['ENGRAM_FILE_THREADS']}",
        "SGLANG_OPT_USE_TILELANG_INDEXER=0",
        "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0",
        f"SGLANG_LOGPROB_CHUNK_SIZE={r['SGLANG_LOGPROB_CHUNK_SIZE']}",
        f"KT_GPU_PREFILL_MIN_TOKENS={r['KT_GPU_PREFILL_MIN_TOKENS']}",
        f"KT_GPU_PREFILL_LOG={r['KT_GPU_PREFILL_LOG']}",
        f"KT_NAN_PROBE_MIN_TOKENS={r['KT_NAN_PROBE_MIN_TOKENS']}",
        f"SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE={r['SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE']}",
    ]


FIXED_CONTAINER_ENV: tuple[str, ...] = (
    "SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE", "SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT",
    "SGLANG_DSV41_ENGRAM_FILE_DIR", "SGLANG_OPT_USE_TILELANG_INDEXER", "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH",
)


class ConfigError(Exception):
    pass


def _parse(knob: Knob, raw: str) -> str:
    """Validate `raw` as knob.kind; return it unchanged (the shell consumes strings)."""
    if raw == "" and knob.empty_disables:
        return ""
    try:
        if knob.kind == "int":
            int(raw)
        elif knob.kind == "float":
            float(raw)
        elif knob.kind == "flag":
            if raw not in ("0", "1"):
                raise ValueError
        elif knob.kind == "list":
            for tok in raw.split():
                int(tok)
        elif knob.kind != "str":
            raise AssertionError(knob.kind)
    except ValueError:
        raise ConfigError(f"{knob.name}={raw!r}: expected {knob.kind}") from None
    return raw


def resolve(env: dict[str, str]) -> dict[str, str]:
    """Resolve every knob from `env`; raise ConfigError on an unknown, removed or mistyped knob."""
    if env.get("ALLOW_UNKNOWN_KNOBS", "0") != "1":
        unknown = sorted(n for n in env if n.startswith(OWNED_PREFIXES) and n not in BY_NAME)
        if unknown:
            raise ConfigError(
                f"unknown knob(s) {' '.join(unknown)}: not declared in scripts/serve_config.py "
                "(declare it there, or ALLOW_UNKNOWN_KNOBS=1 to pass it through unchecked)")
    for k in REGISTRY:
        if k.removed and any(n in env for n in (k.name, *k.aliases)):
            raise ConfigError(k.removed)
    out: dict[str, str] = {}
    for k in REGISTRY:
        if k.removed:
            continue
        src = next((n for n in (k.name, *k.aliases) if n in env), None)
        if src is not None:
            out[k.name] = _parse(k, env[src])
        else:
            out[k.name] = k.default(out) if callable(k.default) else k.default
    return out


def warnings(r: dict[str, str]) -> list[str]:
    """Non-fatal advice (printed to stderr)."""
    msgs = []
    try:
        import subprocess
        cores = subprocess.run(["lscpu", "-p=CORE"], capture_output=True, text=True, check=False).stdout
        phys = len({ln for ln in cores.splitlines() if ln and not ln.startswith("#")})
    except OSError:
        phys = 0
    if phys and int(r["KT_CPUINFER"]) > phys:
        msgs.append(f"WARNING: KT_CPUINFER={r['KT_CPUINFER']} exceeds the {phys} physical cores; "
                    "E26 A1 measured -12 % at 24 and a 7x collapse at 30 threads")
    return msgs


def guards(r: dict[str, str]) -> None:
    """Refuse combinations the evidence condemned (T3.16b / E34d / T3.29), each with an override."""
    experts = int(r["KT_NUM_GPU_EXPERTS"]); chunk = int(r["CHUNKED_PREFILL_SIZE"])
    ctx = int(r["CONTEXT_LENGTH"]); slots = int(r["MAX_RUNNING_REQUESTS"]); d = int(r["KT_MAX_DEFERRED"])
    if d >= 5 and r["KT_ALLOW_UNSAFE_DEFERRAL"] != "1":
        raise ConfigError(
            f"KT_MAX_DEFERRED={d}: D>=5 fails the NLL gate (+0.00287, T3.15) and the greedy "
            "consistency gate; D=6 loses 0.23 nats/token (E21). Set KT_ALLOW_UNSAFE_DEFERRAL=1 to run it anyway.")
    if r["ALLOW_UNTESTED_FIT"] != "1":
        if experts > 226 and chunk >= 4096:
            raise ConfigError(
                f"KT_NUM_GPU_EXPERTS={experts} with CHUNKED_PREFILL_SIZE={chunk}: "
                "226 is the fit ceiling for 4 x 32K slots (E28; 230 needed chunk 2048 + fraction 0.975 and failed "
                "the NLL gate, E29). Set ALLOW_UNTESTED_FIT=1 to try, then run stress-kv-full.py followed by nll-score.py.")
        if chunk >= 8192 and experts > 219:
            raise ConfigError(
                f"CHUNKED_PREFILL_SIZE={chunk} needs ~2.2 GB more than 4096 and only fits "
                f"beside KT_NUM_GPU_EXPERTS<=219 (E22/E28/E34d-ii; at 226 the fused-MoE prefill workspace OOMs on the "
                "first 16K prefill, E34d, and expandable_segments breaks the custom all-reduce). Set ALLOW_UNTESTED_FIT=1 "
                "to try, then run stress-kv-full.py followed by nll-score.py --corpus.")
        # T3.29: above 64K only single-slot rungs are qualified (262K x 4 OOMs in the fused-MoE
        # prefill workspace at ~190K prefix of the first request: pool-size limit, not concurrency).
        if ctx > 65536 and slots > 1:
            raise ConfigError(
                f"CONTEXT_LENGTH={ctx} with MAX_RUNNING_REQUESTS={slots}: "
                "above 64K only single-slot rungs are qualified (262K / 512K x 1 at 219 / 8192, T3.29; 512K needs "
                "MEM_FRACTION_STATIC=0.962). 262K x 4 OOMs in the fused-MoE prefill workspace at ~190K prefix. "
                "Set MAX_RUNNING_REQUESTS=1, or ALLOW_UNTESTED_FIT=1 to try (then stress-kv-full.py at that "
                "concurrency, nll-score.py --corpus).")
        if ctx > 65536 and (chunk != 8192 or experts != 219) and not (chunk == 4096 and experts == 226):
            raise ConfigError(
                f"CONTEXT_LENGTH={ctx} with KT_NUM_GPU_EXPERTS={experts} CHUNKED_PREFILL_SIZE={chunk}: "
                "the single-slot ladder rungs are qualified at 219 / 8192 (T3.29) and at 226 / 4096 (E33/E34); "
                "other placements above 64K need ALLOW_UNTESTED_FIT=1 and the E33 gates a-f + nll-score.py --corpus.")
        if r["DSPARK"] == "1" and experts > 208:
            raise ConfigError(
                "DSPARK=1 adds 3.85 GB/rank of draft stages; it was only ever fitted at KT_NUM_GPU_EXPERTS=208 "
                "with MEM_FRACTION_STATIC=0.935 (E23). Set KT_NUM_GPU_EXPERTS=208 or ALLOW_UNTESTED_FIT=1.")
    if r["DSPARK"] == "1" and not 1 <= int(r["DSPARK_K"]) <= 5:
        raise ConfigError(f"DSPARK_K={r['DSPARK_K']} outside the tested 1..5 (E23 R0-R3).")
    if r["NGRAM"] == "1" and r["DSPARK"] == "1":
        raise ConfigError("NGRAM=1 and DSPARK=1 are two speculative algorithms; SGLang runs one. Pick one.")
    if r["NGRAM"] == "1" and not 2 <= int(r["NGRAM_DRAFT_TOKENS"]) <= 12:
        raise ConfigError(f"NGRAM_DRAFT_TOKENS={r['NGRAM_DRAFT_TOKENS']} outside 2..12 (SGLang's default is 12; "
                          "every row above 1 is CPU-expert bytes per step).")
    if r["NGRAM"] == "1" and not 1 <= int(r["NGRAM_BREADTH"]) <= 10:
        raise ConfigError(f"NGRAM_BREADTH={r['NGRAM_BREADTH']} outside 1..10 (SGLang's default is 10).")


def resolved_line(r: dict[str, str]) -> str:
    """The one line a launch log carries so every knob that decides the result is named."""
    mtt = r["MAX_TOTAL_TOKENS"] or "fraction-sized"
    cap = r["CUDA_GRAPH_BS"] or f"derived<={r['CUDA_GRAPH_MAX_BS']}"
    return (f"resolved: experts gpu={r['KT_NUM_GPU_EXPERTS']} cpu={EXPERTS_PER_LAYER - int(r['KT_NUM_GPU_EXPERTS'])} "
            f"deferral D={r['KT_MAX_DEFERRED']} cpuinfer={r['KT_CPUINFER']} chunk={r['CHUNKED_PREFILL_SIZE']} "
            f"mem_fraction={r['MEM_FRACTION_STATIC']} context={r['CONTEXT_LENGTH']} slots={r['MAX_RUNNING_REQUESTS']} "
            f"max_total_tokens={mtt} capture_bs=[{cap}] dspark={r['DSPARK']} dspark_k={r['DSPARK_K']} "
            f"ngram={r['NGRAM']} ngram_draft={r['NGRAM_DRAFT_TOKENS']} ngram_breadth={r['NGRAM_BREADTH']} "
            f"gpu_prefill_min={r['KT_GPU_PREFILL_MIN_TOKENS']} logprob_chunk={r['SGLANG_LOGPROB_CHUNK_SIZE']} "
            f"nan_probe_min={r['KT_NAN_PROBE_MIN_TOKENS']} "
            f"expandable_after_capture={r['SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE']} "
            f"engram_prefetch={r['ENGRAM_FILE_PREFETCH']} image={r['IMAGE']}")


def diagnostic_env(r: dict[str, str], env: dict[str, str]) -> list[str]:
    """KEY=VALUE pairs for -e: diagnostics only when the caller set them (keeps the default docker cmd stable)."""
    return [f"{k.name}={r[k.name]}" for k in REGISTRY
            if k.sink == DIAGNOSTIC_ENV and any(n in env for n in (k.name, *k.aliases))]


def shell_fragment(env: dict[str, str]) -> str:
    r = resolve(env)
    guards(r)
    lines = [f"{k.name}={shlex.quote(r[k.name])}" for k in REGISTRY if not k.removed]
    lines.append(f"RESOLVED_LINE={shlex.quote(resolved_line(r))}")
    emitted = {p.split("=", 1)[0] for p in container_env(r)}
    for k in REGISTRY:
        if k.sink == CONTAINER_ENV:
            assert (k.container_name or k.name) in emitted, f"{k.name} declared container-env but not emitted"
    lines.append("CONTAINER_ENV=" + shlex.quote(" ".join(container_env(r))))
    lines.append("DIAG_ENV=" + shlex.quote(" ".join(diagnostic_env(r, env))))
    return "\n".join(lines) + "\n"


def markdown_table() -> str:
    rows = ["| knob | type | default | sink | doc | evidence |", "|---|---|---|---|---|---|"]
    for k in REGISTRY:
        default = "derived" if callable(k.default) else (k.default or '""')
        tag = " **(unsafe)**" if k.unsafe else (" *(removed)*" if k.removed else "")
        doc = (k.removed or k.doc).replace("|", "\\|")
        rows.append(f"| `{k.name}`{tag} | {k.kind} | `{default}` | {k.sink} | {doc} | {k.evidence} |")
    return "\n".join(rows) + "\n"


_ENV_READ = re.compile(r'(?:environ(?:\.get)?\(?\[?|get_bool_env_var\(|get_int_env_var\()\s*"((?:KT_|DSPARK|NGRAM|SGLANG_DSV41|SGLANG_BCG|SGLANG_LOGPROB)[A-Z_0-9]*)"')


def overlay_env_reads(overlay_dir: Path) -> set[str]:
    names: set[str] = set()
    for p in (overlay_dir / "python" / "sglang").rglob("*.py"):
        try:
            names.update(_ENV_READ.findall(p.read_text(errors="replace")))
        except OSError:
            continue
    return names


def undeclared_overlay_reads(overlay_dir: Path) -> set[str]:
    declared = set(BY_NAME) | {k.container_name for k in REGISTRY if k.container_name} | set(FIXED_CONTAINER_ENV)
    return {n for n in overlay_env_reads(overlay_dir) if n not in declared}


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "shell"
    env = dict(os.environ)
    try:
        if cmd == "shell":
            for w in warnings(resolve(env)):
                print(w, file=sys.stderr)
            sys.stdout.write(shell_fragment(env))
        elif cmd == "resolved":
            r = resolve(env); guards(r); print(resolved_line(r))
        elif cmd == "markdown":
            sys.stdout.write(markdown_table())
        elif cmd == "check-overlay":
            missing = undeclared_overlay_reads(Path(argv[2]))
            if missing:
                print("FAIL undeclared overlay env reads: " + " ".join(sorted(missing)), file=sys.stderr)
                return 1
            print("ok every overlay env read is declared")
        else:
            print(__doc__, file=sys.stderr); return 2
    except ConfigError as e:
        print(f"FAIL config: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

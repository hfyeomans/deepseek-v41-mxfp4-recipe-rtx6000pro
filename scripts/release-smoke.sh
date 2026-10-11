#!/usr/bin/env bash
# The GPU pre-release smoke (recipe T3.24d). Two modes, run in this order on a quiet host:
#
#   scripts/release-smoke.sh units
#       The three GPU unit suites that ci.sh cannot run (they need a card for a few seconds):
#       run-indexer-tiling-tests.sh, run-moe-workspace-tests.sh, run-prefill-transient-tests.sh.
#       Refuses to run while the dsv41-kt container exists (it owns both GPUs).
#
#   scripts/release-smoke.sh serve --output-dir evidence/<dir> [--context 65536] [--slots 4] [--full]
#                            [--url URL] [--model NAME] [--container NAME]   (defaults: serve_config.py endpoint)
#       Against a server already launched by scripts/serve-hybrid.sh: waits for ready, refuses to
#       start unless the launch manifest describes the running container and its CONTEXT_LENGTH /
#       MAX_RUNNING_REQUESTS equal --context / --slots and /v1/models reports max_model_len ==
#       --context (R12: an evidence directory claims the shape it was gated at), then the
#       qualification gates in the order every ladder rung used (evidence/t329/, t38b/, e34-*):
#         same-prompt C<slots>   verify-batch-consistency.py --same-prompt --flush-cache (rows share >= 40 chars)
#         NLL corpus run         nll-score.py --corpus vs --baseline-corpus exits 0 (a fresh corpus/ dir)
#         NLL corpus gate        pooled dNLL <= NLL_DELTA_MAX, |cluster-robust z| <= NLL_Z_MAX, >= 2 texts, written
#                                by this run; NLL_REQUIRE_IDENTICAL=1 adds tokens_changed == 0
#         needles                verify.py --context <context> (6/6 at >= 95 % of context + arithmetic / thinking / tool call)
#         health                 container log free of NaN / Xid / Traceback / OOM / contract errors
#       --full adds the stress (slots x (context - 2048) tokens), decode C1 / C<slots> and prefill
#       16K / 32K benches that BENCHMARKS tables are built from, each under the power sampler, and a
#       power-summary gate (one sampled entry per bench). Every JSON is stamped by its tool with the
#       launch manifest (evidence_manifest.py); the .txt logs are redacted in place.
#
# Exit 0 = every gate passed; 1 = a gate failed (the summary names it); 2 = usage / not ready / shape mismatch.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# URL / model / container default from the registry (serve_config.py endpoint: the serve this
# environment names), overridable with --url / --model / --container. Before 2026-10-11 an ambient
# legacy CONTAINER_NAME (exported by load-env.sh) made `units` miss the running hybrid and `serve`
# scan docker's "no such container" text as a clean log (T3.43a / R3).
eval "$(python3 "$ROOT/scripts/serve_config.py" endpoint)" || { echo "serve_config.py endpoint failed; pass --url/--model/--container" >&2; exit 2; }
BASELINE_CORPUS="${BASELINE_CORPUS:-$ROOT/evidence/t328/default-219-8192/corpus}"
NLL_Z_MAX="${NLL_Z_MAX:-2.0}"
NLL_DELTA_MAX="${NLL_DELTA_MAX:-0.002}"
NLL_REQUIRE_IDENTICAL="${NLL_REQUIRE_IDENTICAL:-0}"
CONTEXT=65536; SLOTS=4; FULL=0; OUT=""

usage() { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; exit 2; }
mode="${1:-}"; shift || true
case "$mode" in units|serve) ;; *) usage ;; esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir) OUT="$2"; shift 2 ;;
    --context) CONTEXT="$2"; shift 2 ;;
    --slots) SLOTS="$2"; shift 2 ;;
    --url) URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --baseline-corpus) BASELINE_CORPUS="$2"; shift 2 ;;
    --full) FULL=1; shift ;;
    *) echo "unknown argument $1" >&2; usage ;;
  esac
done

fails=()
exec 3>&1  # the script's own stdout: a gate's `> file` redirect must not swallow its PASS/FAIL line
gate() {  # gate <name> <command...>: run, record pass/fail, keep going
  local name="$1"; shift
  if "$@"; then echo "PASS $name" >&3; else echo "FAIL $name" >&3; fails+=("$name"); fi
}

if [[ "$mode" == units ]]; then
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then
    echo "release-smoke units: container $CONTAINER exists; the GPU suites need the cards free (docker rm -f $CONTAINER)" >&2
    exit 2
  fi
  for suite in run-indexer-tiling-tests.sh run-moe-workspace-tests.sh run-prefill-transient-tests.sh; do
    bash "$ROOT/scripts/tests/$suite"; rc=$?
    if [[ $rc -eq 0 ]]; then echo "PASS $suite"
    elif [[ $rc -eq 3 ]]; then echo "SKIP $suite (docker, image, overlay or GPU unavailable)"; fails+=("$suite (skipped)")
    else echo "FAIL $suite (exit $rc)"; fails+=("$suite"); fi
  done
else
  [[ -n "$OUT" ]] || { echo "serve mode needs --output-dir" >&2; usage; }
  [[ -d "$BASELINE_CORPUS" ]] || { echo "baseline corpus $BASELINE_CORPUS not found" >&2; exit 2; }
  mkdir -p "$OUT"
  python3 "$ROOT/scripts/wait-ready.py" --url "$URL" --container "$CONTAINER" --timeout "${WAIT_TIMEOUT:-1800}" > "$OUT/wait.txt" 2>&1 \
    || { echo "server at $URL not ready (see $OUT/wait.txt)" >&2; docker logs "$CONTAINER" > "$OUT/launch-container.txt" 2>&1; exit 2; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$OUT/vram-after-capture.txt" 2>&1 || true
  # The shape this directory will claim must be the shape that is serving (R12): the launch manifest
  # must describe the running container, its CONTEXT_LENGTH / MAX_RUNNING_REQUESTS must equal
  # --context / --slots, and the server must report that ceiling. Anything else is a mislabelled
  # evidence directory, so it is a usage error (rc 2) before any gate runs.
  eval "$(python3 "$ROOT/scripts/evidence_manifest.py" serve-shape)" || { echo "no readable launch manifest (serve through scripts/serve-hybrid.sh)" >&2; exit 2; }
  live_id="$(docker inspect "$CONTAINER" --format '{{.Id}}' 2>/dev/null || true)"
  model_len="$(python3 -c '
import sys; sys.path.insert(0, sys.argv[3])
from openai_client import get_json
m = next((x for x in get_json(sys.argv[1] + "/v1/models")["data"] if x["id"] == sys.argv[2]), {})
print(m.get("max_model_len", ""))' "$URL" "$MODEL" "$ROOT/scripts" 2>/dev/null || true)"
  shape_ok=1
  [[ "$live_id" == "$MF_CONTAINER_ID" ]] || { echo "manifest describes container ${MF_CONTAINER_ID:0:12} but $CONTAINER is ${live_id:0:12}" >&2; shape_ok=0; }
  [[ "$MF_CONTEXT" == "$CONTEXT" ]] || { echo "manifest CONTEXT_LENGTH=$MF_CONTEXT but --context $CONTEXT" >&2; shape_ok=0; }
  [[ "$MF_SLOTS" == "$SLOTS" ]] || { echo "manifest MAX_RUNNING_REQUESTS=$MF_SLOTS but --slots $SLOTS" >&2; shape_ok=0; }
  [[ "$model_len" == "$CONTEXT" ]] || { echo "/v1/models max_model_len='$model_len' but --context $CONTEXT" >&2; shape_ok=0; }
  [[ $shape_ok -eq 1 ]] || { echo "release-smoke serve: shape mismatch; pass the launch's --context/--slots or serve the rung you mean" >&2; exit 2; }
  echo "context=$CONTEXT slots=$SLOTS full=$FULL url=$URL model=$MODEL out=$OUT experts=$MF_EXPERTS chunk=$MF_CHUNK expert_map=${MF_EXPERT_MAP##*/} manifest=${MF_HASH:0:12}" > "$OUT/smoke-args.txt"
  started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  if [[ $FULL -eq 1 ]]; then
    gate "stress ${SLOTS}x$((CONTEXT - 2048))" python3 "$ROOT/scripts/stress-kv-full.py" --url "$URL" --model "$MODEL" \
      --concurrency "$SLOTS" --prompt-tokens $((CONTEXT - 2048)) --max-tokens 256 --seed 31 \
      --output "$OUT/stress-kv.json" > "$OUT/stress-kv.txt" 2>&1
  fi
  # --flush-cache (R17): with the radix cache on, the bs=1 reference prefills the whole prompt while
  # the identical concurrent prompts hit the cache and run a different kernel path for their last
  # token; the gate asks whether bs>1 matches bs=1, so every request must prefill from scratch.
  gate "same-prompt C$SLOTS" python3 "$ROOT/scripts/verify-batch-consistency.py" --url "$URL" --model "$MODEL" \
    --same-prompt --flush-cache --concurrency "$SLOTS" --output "$OUT/same-prompt-c$SLOTS.json" > "$OUT/same-prompt-c$SLOTS.txt" 2>&1
  # The corpus run is a gate of its own and writes into a fresh directory (R6): a crashed run used to
  # leave an earlier pooled.json for the gate below to read.
  rm -rf "$OUT/corpus"
  gate "NLL corpus run" python3 "$ROOT/scripts/nll-score.py" --url "$URL" --corpus "$ROOT/scripts/fixtures/nll-corpus" \
    --output-dir "$OUT/corpus" --compare-dir "$BASELINE_CORPUS" > "$OUT/corpus.txt" 2>&1
  gate "NLL corpus dNLL <= $NLL_DELTA_MAX, |z| <= $NLL_Z_MAX$([[ "$NLL_REQUIRE_IDENTICAL" == 1 ]] && echo ', identical')" \
    python3 - "$OUT/corpus/pooled.json" "$NLL_Z_MAX" "$NLL_DELTA_MAX" "$NLL_REQUIRE_IDENTICAL" "$started_utc" <<'PY'
import json, sys
path, z_max, d_max, identical, started = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4] == "1", sys.argv[5]
try:
    doc = json.load(open(path))
except (OSError, ValueError) as e:
    print(f"  no pooled.json from this run: {e}"); sys.exit(1)
c = doc["compare"]
z = c["z_cluster"]
print(f"  pooled dNLL {c['delta_mean_nll']:+.5f} +/- {c['se']:.5f} nats/tok, z {c['z_paired']:+.2f}, cluster-robust z {z:+.2f}, "
      f"n {c['n']}, texts {c.get('texts')}, moved {c['tokens_changed']}, recorded {doc.get('evidence', {}).get('recorded_utc')}")
bad = []
if doc.get("evidence", {}).get("recorded_utc", "") < started: bad.append("pooled.json predates this smoke")
if c.get("texts", 0) < 2: bad.append("fewer than 2 texts (cluster-robust z needs >= 2)")
if not abs(z) <= z_max: bad.append(f"|z| {abs(z):.2f} > {z_max}")
if not c["delta_mean_nll"] <= d_max: bad.append(f"dNLL {c['delta_mean_nll']:+.5f} > +{d_max}")
if identical and c["tokens_changed"] != 0: bad.append(f"{c['tokens_changed']} tokens moved but NLL_REQUIRE_IDENTICAL=1")
for b in bad: print(f"  FAIL {b}")
sys.exit(1 if bad else 0)
PY
  gate "needles at $CONTEXT" python3 "$ROOT/scripts/verify.py" --url "$URL" --model "$MODEL" --context "$CONTEXT" \
    --output "$OUT/verify.json" > "$OUT/verify.txt" 2>&1
  if [[ $FULL -eq 1 ]]; then
    # Each bench runs under a 500 ms nvidia-smi sampler (power, clocks, temperature, throttle
    # reasons); summarize-power.py reduces the CSVs to busy-median watts / max temperature.
    # shellcheck disable=SC2329  # invoked through gate()
    sampled() { bash "$ROOT/scripts/gpu-power-sample.sh" "$OUT/power-$1.csv" -- "${@:2}"; }
    gate "decode C1" sampled decode-c1 python3 "$ROOT/scripts/bench-decode.py" --url "$URL" --model "$MODEL" --concurrency 1 --fixture-offset 6 \
      --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output "$OUT/c1.json" > "$OUT/c1.txt" 2>&1
    gate "decode C$SLOTS" sampled "decode-c$SLOTS" python3 "$ROOT/scripts/bench-decode.py" --url "$URL" --model "$MODEL" --concurrency "$SLOTS" --fixture-offset 6 \
      --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output "$OUT/c$SLOTS.json" > "$OUT/c$SLOTS.txt" 2>&1
    gate "prefill 16K" sampled prefill-16k python3 "$ROOT/scripts/bench-prefill.py" --url "$URL" --model "$MODEL" --tokens 16384 --runs 3 --seed 777 \
      --output "$OUT/prefill-16k.json" > "$OUT/prefill-16k.txt" 2>&1
    gate "prefill 32K" sampled prefill-32k python3 "$ROOT/scripts/bench-prefill.py" --url "$URL" --model "$MODEL" --tokens 32768 --runs 3 --seed 777 \
      --output "$OUT/prefill-32k.json" > "$OUT/prefill-32k.txt" 2>&1
    # One sampled entry per bench or the smoke fails (R10): before T3.43b this was `|| true`.
    gate "power summary (4 sampled benches)" python3 "$ROOT/scripts/summarize-power.py" --smoke "$OUT" \
      --expect-labels "decode-c1,decode-c$SLOTS,prefill-16k,prefill-32k" > "$OUT/power.txt" 2>&1
  fi
  # The health gate reads the real container log or fails: a missing container is a failed gate, not a
  # clean one (R3).
  # shellcheck disable=SC2016  # $1/$2 are the inner bash's positionals
  gate "container log readable ($CONTAINER)" bash -c 'docker logs "$1" > "$2" 2>&1' _ "$CONTAINER" "$OUT/launch-container.txt"
  health="$(grep -iE -e ' nan' -e 'probability tensor' -e 'Xid' -e 'Traceback' -e 'KTContract' -e 'Sealed.*Error' -e 'OutOfMemory' -e 'NonFiniteLogits' \
             "$OUT/launch-container.txt" | grep -vc 'any later grow raises')"
  echo "health=$health" > "$OUT/health.txt"
  gate "health (0 error lines in the container log)" test "$health" -eq 0
  { grep -c -iE 'xid' /var/log/kern.log 2>/dev/null || echo 0; } > "$OUT/xid-count.txt"
  # Redact the logs the way every evidence directory is (ci.sh refuses home paths and bridge addresses).
  sed -i 's#/home/linuxbre[w]/#/<brew>/#g; s#/home/[A-Za-z0-9._-]*/#~/#g; s#172\.17\.[0-9]*\.[0-9]*#<docker-bridge-ip>#g' "$OUT"/*.txt
fi

if [[ ${#fails[@]} -eq 0 ]]; then echo "release-smoke $mode: all gates passed"; exit 0; fi
echo "release-smoke $mode: ${#fails[@]} gate(s) failed: ${fails[*]}"; exit 1

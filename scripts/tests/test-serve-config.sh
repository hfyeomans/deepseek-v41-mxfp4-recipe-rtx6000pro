#!/usr/bin/env bash
# Launcher config-guard tests for scripts/serve-hybrid.sh, run with DRY_RUN=1 so nothing is
# started. Each case pins a fail_config refusal or an accepted qualified profile to the
# evidence that set it (E28/E29, E34d, T3.29). No docker, no GPU; safe while a server runs.
set -uo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVE="$ROOT/scripts/serve-hybrid.sh"
pass=0; fail=0

# expect_ok NAME ENV...   -> rc 0 and a "resolved:" line
# expect_refuse NAME PATTERN ENV... -> rc 2 and "FAIL config:" mentioning PATTERN
run_case() {
  local want="$1" name="$2" pattern="$3"; shift 3
  local out rc
  out="$(env "$@" DRY_RUN=1 bash "$SERVE" 2>&1)"; rc=$?
  if [[ "$want" == ok && $rc -eq 0 && "$out" == resolved:* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  elif [[ "$want" == refuse && $rc -eq 2 && "$out" == "FAIL config:"* && "$out" == *"$pattern"* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  else
    echo "FAIL $name (rc=$rc): ${out%%$'\n'*}"; fail=$((fail + 1))
  fi
}

# Qualified profiles (accepted)
run_case ok "default 64K x 4 at 219/8192" "" CONTEXT_LENGTH=65536
run_case ok "262K x 1 at 219/8192 (T3.29)" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1
run_case ok "512K x 1 at 219/8192, fraction 0.962 (T3.29)" "" CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962
run_case ok "262K x 1 at 226/4096 (E34b-ii)" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1 KT_NUM_GPU_EXPERTS=226 CHUNKED_PREFILL_SIZE=4096

# Refusals
run_case refuse "262K x 4 refused (T3.29 OOM at ~190K prefix)" "MAX_RUNNING_REQUESTS=4" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4
run_case refuse "unqualified placement above 64K refused" "KT_NUM_GPU_EXPERTS=212" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1 KT_NUM_GPU_EXPERTS=212
run_case refuse "chunk 8192 beside 226 refused (E34d)" "CHUNKED_PREFILL_SIZE=8192" KT_NUM_GPU_EXPERTS=226 CHUNKED_PREFILL_SIZE=8192
run_case refuse "230 experts refused (E29)" "KT_NUM_GPU_EXPERTS=230" KT_NUM_GPU_EXPERTS=230
run_case refuse "D=5 refused without the unsafe flag (T3.15)" "KT_MAX_DEFERRED=5" KT_MAX_DEFERRED=5
run_case refuse "B2 knobs refused (T3.16j)" "KT_B2_UVA_EXPERTS" KT_B2_UVA_EXPERTS=1

# Override escape hatch still works
run_case ok "ALLOW_UNTESTED_FIT=1 lets 262K x 4 through" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 ALLOW_UNTESTED_FIT=1

echo "$pass/$((pass + fail)) passed"
[[ $fail -eq 0 ]]

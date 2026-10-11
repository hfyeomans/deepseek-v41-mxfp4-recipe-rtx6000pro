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
  # stdout and stderr are asserted separately: `resolved:` goes to stdout, `FAIL config:` to stderr,
  # and host-dependent advisories (e.g. KT_CPUINFER above the host's core count on a 4-CPU CI runner)
  # are stderr lines that must not shift the asserted stream (T3.43a / R1).
  local out err rc errfile
  errfile="$(mktemp)"
  out="$(env "$@" DRY_RUN=1 bash "$SERVE" 2>"$errfile")"; rc=$?
  err="$(cat "$errfile")"; rm -f "$errfile"
  if [[ "$want" == ok && $rc -eq 0 && "$out" == resolved:* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  elif [[ "$want" == refuse && $rc -eq 2 && "$err" == *"FAIL config:"* && "$err" == *"$pattern"* ]]; then
    echo "ok   $name"; pass=$((pass + 1))
  else
    echo "FAIL $name (rc=$rc): ${out%%$'\n'*} ${err%%$'\n'*}"; fail=$((fail + 1))
  fi
}

# Qualified profiles (accepted)
run_case ok "default 64K x 4 at 219/8192" "" CONTEXT_LENGTH=65536
run_case ok "262K x 1 at 219/8192 (T3.29)" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1
run_case ok "512K x 1 at 219/8192, fraction 0.962 (T3.29)" "" CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962
run_case ok "1M x 1 at 219/8192, fraction 0.962 (E34d-v (ii))" "" CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1 MEM_FRACTION_STATIC=0.962
run_case ok "262K x 1 at 226/4096 (E34b-ii)" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1 KT_NUM_GPU_EXPERTS=226 CHUNKED_PREFILL_SIZE=4096
run_case ok "PROFILE=developer is the qualified shape (T3.24a)" "" PROFILE=developer
run_case ok "262K x 4 at 219/8192 with the lever (T3.31 idea 8, re-qualified on 682fc6a0)" "" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4

# Refusals
run_case refuse "262K x 4 without the expandable lever refused (idea 8 (a) OOM)" "SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=1" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 SGLANG_DSV41_EXPANDABLE_SEGMENTS_AFTER_CAPTURE=0
run_case refuse "512K x 2 refused (never run)" "MAX_RUNNING_REQUESTS=2" CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=2 MEM_FRACTION_STATIC=0.962
run_case refuse "1M x 1 at fraction 0.96 refused (pool caps below the context)" "MEM_FRACTION_STATIC=0.962" CONTEXT_LENGTH=1048576 MAX_RUNNING_REQUESTS=1
run_case refuse "512K x 1 at fraction 0.96 refused" "MEM_FRACTION_STATIC=0.962" CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=1
run_case refuse "unqualified placement above 64K refused" "KT_NUM_GPU_EXPERTS=212" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=1 KT_NUM_GPU_EXPERTS=212
run_case refuse "chunk 8192 beside 226 refused (E34d)" "CHUNKED_PREFILL_SIZE=8192" KT_NUM_GPU_EXPERTS=226 CHUNKED_PREFILL_SIZE=8192
run_case refuse "230 experts refused (E29)" "KT_NUM_GPU_EXPERTS=230" KT_NUM_GPU_EXPERTS=230
run_case refuse "D=5 refused without the unsafe flag (T3.15)" "KT_MAX_DEFERRED=5" KT_MAX_DEFERRED=5
run_case refuse "B2 knobs refused (T3.16j)" "KT_B2_UVA_EXPERTS" KT_B2_UVA_EXPERTS=1
run_case refuse "unknown profile refused (T3.24a)" "profiles in config/profiles/" PROFILE=4gpu-sglang

run_case refuse "unsafe flag under the qualified profile refused (T3.24b)" "PROFILE=developer" CONTEXT_LENGTH=262144 MAX_RUNNING_REQUESTS=4 ALLOW_UNTESTED_FIT=1
run_case refuse "diagnostic under the qualified profile refused (T3.24b)" "KT_NAN_PROBE_MIN_TOKENS is a diagnostic" KT_NAN_PROBE_MIN_TOKENS=1

# Override escape hatch still works under the developer profile
run_case ok "PROFILE=developer ALLOW_UNTESTED_FIT=1 lets 512K x 2 through" "" PROFILE=developer CONTEXT_LENGTH=524288 MAX_RUNNING_REQUESTS=2 MEM_FRACTION_STATIC=0.962 ALLOW_UNTESTED_FIT=1
run_case ok "unsafe flag at its default is not an experiment (T3.24b)" "" ALLOW_UNTESTED_FIT=0 KT_TIMELINE=0

echo "$pass/$((pass + fail)) passed"
[[ $fail -eq 0 ]]

#!/usr/bin/env bash
# T3.52 open question: is the −1…−2 % 16K prefill of E44's pinned passes real? Paired A/B/A/B on the
# qualification serve (Engram warm after the full smoke): pinned (16–31, the serve's own state) /
# floating (0–31) / pinned / floating, one fresh seed per pass, 3 runs each.
set -u
D=evidence/t352
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
pin() {  # pin <mask>: every non-numa_* thread in the container
  docker exec dsv41-kt sh -c 'n=0; for t in /proc/[0-9]*/task/*; do c=$(cat $t/comm 2>/dev/null) || continue; case "$c" in numa_*) ;; *) taskset -p '"$1"' ${t##*/} >/dev/null 2>&1 && n=$((n+1));; esac; done; echo "pinned $n threads to '"$1"'"'
}
run() {  # run <label> <seed> <tokens>
  python3 scripts/bench-prefill.py --url $U --model $M --tokens $3 --runs 3 --seed $2 --output $D/aba-$1.json > $D/aba-$1.txt 2>&1
  echo "$1 seed=$2 tokens=$3: $(python3 -c "import json;print(json.load(open('$D/aba-$1.json'))['median_prefill_tok_s'])" 2>/dev/null || tail -1 $D/aba-$1.txt)"
}
pin 0xffff0000; run pinned1-16k 5201 16384
pin 0xffffffff; run floating1-16k 5211 16384
pin 0xffff0000; run pinned2-16k 5221 16384
pin 0xffffffff; run floating2-16k 5231 16384
pin 0xffff0000; run pinned1-32k 5241 32768
pin 0xffffffff; run floating1-32k 5251 32768
pin 0xffff0000
echo aba-done

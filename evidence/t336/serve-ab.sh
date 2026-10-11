#!/usr/bin/env bash
# T3.36 + T3.40 serve-level A/B, one window, same day, launcher defaults (262K x 4) everywhere:
#   a-baseline : dsv41-kt:recipe, no extra env                       (comparator)
#   b-thp      : dsv41-kt:recipe + GLIBC_TUNABLES=glibc.malloc.hugetlb=1 (T3.36: kt arena on 2 MiB THP)
#   c-native   : dsv41-kt:native (KT_CPU_INSTRUCT=NATIVE, -march=native) (T3.40)
#   a2-baseline: the baseline again, after c-native (order control: b and c both read prefill below a)
# Per serve: warm 32K prefill, decode C1 x6 / C4 x3 (512 tokens, distinct prompts), prefill 16K/32K x2,
# kt-cpu-per-step at C1/C4, NLL corpus vs the 219/8192 anchor, health counts, manifest.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
ROOT=evidence/t336/serve-ab
ANCHOR=evidence/t328/default-219-8192/corpus
mkdir -p $ROOT
for S in ${SERVES:-a-baseline b-thp c-native}; do
  D=$ROOT/$S; mkdir -p $D
  echo "== serve $S $(date +%T)"
  case $S in
    a-baseline|a2-baseline) bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1;;
    b-thp)      PROFILE=developer EXTRA_DOCKER_ENV="GLIBC_TUNABLES=glibc.malloc.hugetlb=1" bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1;;
    c-native)   IMAGE=dsv41-kt:native bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1;;
  esac || { echo LAUNCHFAIL $S; continue; }
  python3 scripts/wait-ready.py --url $U --container dsv41-kt --timeout 1800 > $D/ready.txt 2>&1 \
    || { echo READYFAIL $S; docker logs dsv41-kt > $D/launch-container.txt 2>&1; docker rm -f dsv41-kt; continue; }
  # THP check on the TP0 scheduler process (the one owning kt's numa_* threads)
  CPID=$(docker inspect -f '{{.State.Pid}}' dsv41-kt)
  for p in $(pgrep -P "$CPID"; pgrep -P "$(pgrep -P "$CPID" | head -1)" || true); do
    if grep -qs '^numa_' /proc/"$p"/task/*/comm 2>/dev/null; then grep -E 'AnonHugePages|Rss' /proc/$p/smaps_rollup > $D/tp0-smaps.txt; break; fi
  done
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 1 --seed 777 --output $D/prefill-32k-warm.json > $D/prefill-32k-warm.txt 2>&1; tail -1 $D/prefill-32k-warm.txt
  python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/decode-c1.json > $D/decode-c1.txt 2>&1; sed -n 3p $D/decode-c1.txt
  python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/decode-c4.json > $D/decode-c4.txt 2>&1; sed -n 3p $D/decode-c4.txt
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 2 --seed 1277 --output $D/prefill-16k.json > $D/prefill-16k.txt 2>&1; tail -1 $D/prefill-16k.txt
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 1277 --output $D/prefill-32k.json > $D/prefill-32k.txt 2>&1; tail -1 $D/prefill-32k.txt
  bash scripts/kt-cpu-per-step.sh --url $U --model $M --concurrency 1,4 --max-tokens 200 --output $D/kt-cpu > $D/kt-cpu-per-step.txt 2>&1; cat $D/kt-cpu-per-step.txt
  python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus --compare-dir $ANCHOR > $D/nll.txt 2>&1; tail -4 $D/nll.txt
  docker logs dsv41-kt 2>&1 | grep -E -i -c '\bnan\b|xid|traceback|out of memory|contract' > $D/health-hits.txt
  docker logs dsv41-kt 2>&1 | grep -c 'memory mapping failed' > $D/mapping-failures.txt
  nvidia-smi --query-gpu=memory.used,temperature.gpu,power.draw --format=csv > $D/nvidia-smi.txt
  cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
  docker rm -f dsv41-kt > /dev/null 2>&1; sleep 10
  echo "== done $S $(date +%T)"
done
echo "== all done $(date +%T)"

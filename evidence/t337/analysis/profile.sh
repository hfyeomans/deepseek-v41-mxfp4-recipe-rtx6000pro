#!/usr/bin/env bash
# T3.37: CPU+GPU torch-profiler trace of one 32K prefill at two KV pools on the same prompt seeds.
# pool 64k4  = launcher default (64K x 4, max_total_tokens 278,528)
# pool 262k4 = CONTEXT_LENGTH=262144 (262K x 4, max_total_tokens 1,064,960; lever on by default)
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
for POOL in ${POOLS:-64k4 262k4}; do
  D=evidence/t337/$POOL
  mkdir -p $D
  echo "== serve pool=$POOL $(date +%T)"
  if [[ $POOL == 262k4 ]]; then export CONTEXT_LENGTH=262144; else unset CONTEXT_LENGTH; fi
  bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
  python3 scripts/wait-ready.py --url $U --container dsv41-kt --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 1 --seed 777 --output $D/prefill-32k-warm.json > $D/prefill-32k-warm.txt 2>&1; tail -2 $D/prefill-32k-warm.txt
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 1277 --output $D/prefill-32k-unprofiled.json > $D/prefill-32k-unprofiled.txt 2>&1; tail -2 $D/prefill-32k-unprofiled.txt
  docker exec dsv41-kt mkdir -p /tmp/prof
  curl -s -X POST $U/start_profile -H 'content-type: application/json' -d '{"output_dir":"/tmp/prof","num_steps":6,"activities":["CPU","GPU"],"with_stack":false}' > $D/start-profile.txt; cat $D/start-profile.txt; echo
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 1 --seed 1777 --output $D/prefill-32k-profiled.json > $D/prefill-32k-profiled.txt 2>&1; tail -2 $D/prefill-32k-profiled.txt
  sleep 5
  curl -s -X POST $U/stop_profile > $D/stop-profile.txt 2>&1; echo
  sleep 90
  docker cp dsv41-kt:/tmp/prof/. $D/ 2>&1 | tail -1
  ls -la $D
  docker logs dsv41-kt 2>&1 | rg -e "Prefill batch|layerwise prefill" | tail -16 > $D/prefill-log.txt
  cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
  bash scripts/stop.sh > /dev/null 2>&1; for i in $(seq 30); do docker ps -a --format "{{.Names}}" | grep -qx dsv41-kt || break; sleep 2; done
  echo "== done pool=$POOL $(date +%T)"
done

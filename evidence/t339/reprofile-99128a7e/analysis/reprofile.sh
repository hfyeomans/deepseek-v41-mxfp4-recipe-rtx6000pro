#!/usr/bin/env bash
# T3.39 re-profile on overlay 99128a7e (qualified profile, launcher defaults, KT_CPUINFER=16):
# CPU+GPU torch-profiler trace of one 16K prefill, single serve. Sizes window 2.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
D=evidence/t339/reprofile-99128a7e
mkdir -p $D
echo "== serve $(date +%T)"
bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 1 --seed 777 --output $D/prefill-16k-warm.json > $D/prefill-16k-warm.txt 2>&1; tail -2 $D/prefill-16k-warm.txt
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 2 --seed 1277 --output $D/prefill-16k-unprofiled.json > $D/prefill-16k-unprofiled.txt 2>&1; tail -2 $D/prefill-16k-unprofiled.txt
docker exec dsv41-kt mkdir -p /tmp/prof
curl -s -X POST $U/start_profile -H 'content-type: application/json' -d '{"output_dir":"/tmp/prof","num_steps":4,"activities":["CPU","GPU"],"with_stack":false}' > $D/start-profile.txt; cat $D/start-profile.txt; echo
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 1 --seed 1777 --output $D/prefill-16k-profiled.json > $D/prefill-16k-profiled.txt 2>&1; tail -2 $D/prefill-16k-profiled.txt
sleep 5
curl -s -X POST $U/stop_profile > $D/stop-profile.txt 2>&1; echo
sleep 60
docker cp dsv41-kt:/tmp/prof/. $D/ 2>&1 | tail -1
ls -la $D
docker logs dsv41-kt 2>&1 | rg -e "Prefill batch|layerwise prefill" | tail -12 > $D/prefill-log.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== done $(date +%T)"

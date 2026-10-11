#!/usr/bin/env bash
# T3.37 step 2: does the per-chunk drop appear only once the pool has been filled by earlier traffic?
# One 262K x 4 serve: fresh 32K prefill -> fill the pool (stress 4 x 260K) -> 32K prefill -> /flush_cache -> 32K prefill.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
D=evidence/t337/262k4-filled
mkdir -p $D
echo "== serve 262k4 $(date +%T)"
CONTEXT_LENGTH=262144 bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --container dsv41-kt --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; exit 2; }
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 1 --seed 777 --output $D/prefill-32k-warm.json > $D/prefill-32k-warm.txt 2>&1; tail -2 $D/prefill-32k-warm.txt
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 1277 --output $D/prefill-32k-fresh.json > $D/prefill-32k-fresh.txt 2>&1; tail -2 $D/prefill-32k-fresh.txt
echo "== fill $(date +%T)"
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 260096 --max-tokens 64 --output $D/stress-kv.json > $D/stress-kv.txt 2>&1; tail -3 $D/stress-kv.txt
echo "== filled $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 1277 --output $D/prefill-32k-filled.json > $D/prefill-32k-filled.txt 2>&1; tail -2 $D/prefill-32k-filled.txt
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 2277 --output $D/prefill-32k-filled-newseed.json > $D/prefill-32k-filled-newseed.txt 2>&1; tail -2 $D/prefill-32k-filled-newseed.txt
curl -s -X POST $U/flush_cache > $D/flush.txt; echo
echo "== flushed $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 2 --seed 1277 --output $D/prefill-32k-flushed.json > $D/prefill-32k-flushed.txt 2>&1; tail -2 $D/prefill-32k-flushed.txt
docker logs dsv41-kt 2>&1 | rg -e "Prefill batch|layerwise prefill|flush" > $D/prefill-log.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
echo "== done $(date +%T) (server left running)"

#!/usr/bin/env bash
# T3.8 round-2 diagnostic: why the stat recorder dumped zeros. One serve, debug hooks, one request each phase.
set -u
U=http://127.0.0.1:8000; M=dsv41-flash-hybrid; R=.amp/in/t38/diag3; mkdir -p $R
echo "== serve $(date +%T)"
EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" EXTRA_DOCKER_ENV="SGLANG_EXPERT_DIST_DEBUG=1" bash scripts/serve-hybrid.sh > $R/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $R/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $R/container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 32 --output $R/warm.json > $R/warm.txt 2>&1; tail -1 $R/warm.txt
docker exec dsv41-kt sh -c "rm -f /tmp/expert_distribution_recorder_*.pt"
curl -s -X POST $U/start_expert_distribution_record; echo
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 0 --max-tokens 64 --output $R/dec.json > $R/dec.txt 2>&1; tail -1 $R/dec.txt
curl -s -X POST $U/dump_expert_distribution_record; echo
python3 scripts/bench-prefill.py --url $U --model $M --tokens 4096 --runs 1 --seed 777 --output $R/pre.json > $R/pre.txt 2>&1; tail -1 $R/pre.txt
curl -s -X POST $U/dump_expert_distribution_record; echo
curl -s -X POST $U/stop_expert_distribution_record; echo
sleep 2
docker cp dsv41-kt:/tmp/. $R/tmp/ 2>/dev/null; ls $R/tmp | grep expert_distribution
docker logs dsv41-kt > $R/container.txt 2>&1; rg "T38DBG|expert distribution|Write expert" $R/container.txt > $R/dbg.txt; wc -l $R/dbg.txt
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== done $(date +%T)"

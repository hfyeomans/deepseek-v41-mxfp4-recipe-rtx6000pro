#!/usr/bin/env bash
# T3.8b soak, part 2: long-reasoning sets (bench-reasoning.py, thinking on, up to 16K tokens) with the recorder on.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
R=evidence/t38b/soak-default
mkdir -p $R
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
echo "== serve2 $(date +%T)"
EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" bash scripts/serve-hybrid.sh > $R/launch2.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $R/ready2.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $R/launch2-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $R/manifest2.json
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 64 --output $R/warm2.json > $R/warm2.txt 2>&1

dumper() { while [ -f /tmp/t38b-soak-dumping ]; do curl -s -X POST $U/dump_expert_distribution_record > /dev/null; sleep 10; done; }
run_set() {
  local name=$1; shift
  mkdir -p $R/$name
  docker exec dsv41-kt sh -c "rm -f /tmp/expert_distribution_recorder_*.pt"
  curl -s -X POST $U/start_expert_distribution_record > $R/$name/start.txt
  touch /tmp/t38b-soak-dumping; dumper & local dp=$!
  echo "== set $name $(date +%T): $*"
  "$@"
  rm -f /tmp/t38b-soak-dumping; wait $dp
  curl -s -X POST $U/dump_expert_distribution_record > $R/$name/dump.txt
  curl -s -X POST $U/stop_expert_distribution_record > $R/$name/stop.txt
  sleep 2
  docker exec dsv41-kt sh -c "mkdir -p /tmp/s-$name && mv /tmp/expert_distribution_recorder_*.pt /tmp/s-$name/"
  docker cp dsv41-kt:/tmp/s-$name/. $R/$name/ | tail -1
  echo "dumps: $(ls $R/$name/*.pt | wc -l)  health=$(health)"
}

# 6. long reasoning C4: all 8 hard prompts, thinking on, up to 16K tokens each
run_set reasoning-c4 bash -c "python3 scripts/bench-reasoning.py --url $U --model $M --concurrency 4 --max-tokens 16384 --output $R/reasoning-c4/bench.json > $R/reasoning-c4/bench.txt 2>&1"
tail -4 $R/reasoning-c4/bench.txt
# 7. long reasoning C1: first 4 prompts
run_set reasoning-c1 bash -c "python3 scripts/bench-reasoning.py --url $U --model $M --concurrency 1 --limit 4 --max-tokens 16384 --output $R/reasoning-c1/bench.json > $R/reasoning-c1/bench.txt 2>&1"
tail -4 $R/reasoning-c1/bench.txt
echo "health=$(health)"
docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e OutOfMemory -e "expandable_segments" | rg -v "any later grow raises" > $R/health-hits2.txt
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== soak2 done $(date +%T)"

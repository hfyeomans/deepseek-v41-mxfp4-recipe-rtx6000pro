#!/usr/bin/env bash
# T3.8 round 2: per-(layer, expert) routing counts on real traffic, SGLang's stat recorder (vision_topk now reports), four traffic sets.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
R=evidence/t38/round2
mkdir -p $R
echo "== serve $(date +%T)"
EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" bash scripts/serve-hybrid.sh > $R/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $R/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $R/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $R/manifest.json
# warm (not recorded)
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 64 --output $R/warm.json > $R/warm.txt 2>&1; tail -1 $R/warm.txt

dumper() {  # dump every 10 s so the 1000-pass circular buffer never wraps (C1 is ~40 passes/s)
  while [ -f /tmp/t38-dumping ]; do curl -s -X POST $U/dump_expert_distribution_record > /dev/null; sleep 10; done
}
run_set() {  # name, then the command
  local name=$1; shift
  mkdir -p $R/$name
  docker exec dsv41-kt sh -c "rm -f /tmp/expert_distribution_recorder_*.pt"
  curl -s -X POST $U/start_expert_distribution_record > $R/$name/start.txt
  touch /tmp/t38-dumping; dumper & local dp=$!
  echo "== set $name $(date +%T): $*"
  "$@"
  rm -f /tmp/t38-dumping; wait $dp
  curl -s -X POST $U/dump_expert_distribution_record > $R/$name/dump.txt
  curl -s -X POST $U/stop_expert_distribution_record > $R/$name/stop.txt
  sleep 2
  docker exec dsv41-kt sh -c "mkdir -p /tmp/t38-$name && mv /tmp/expert_distribution_recorder_*.pt /tmp/t38-$name/"
  docker cp dsv41-kt:/tmp/t38-$name/. $R/$name/ | tail -1
  ls $R/$name | wc -l
}

# A (fit): HumanEval generation, 40 problems at C4 -- real code decode traffic
run_set humaneval-c4 python3 scripts/bench-humaneval.py --url $U --model $M --phase generate --limit 40 --concurrency 4 --max-tokens 2048 --output-dir $R/humaneval-c4/bench
# B (held-out): the distinct prose/code fixtures at C1 and C4, 256 tokens
run_set decode-distinct python3 scripts/bench-decode.py --url $U --model $M --concurrency 1,4 --runs 3 --warmup 0 --max-tokens 256 --distinct-prompts --fixture-set distinct --output $R/decode-distinct/bench.json
# C (prefill, pseudo-random words; el8 set C analogue): one 16K prefill
run_set prefill-random python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 1 --seed 777 --output $R/prefill-random/bench-16k.json
# D (prefill, real text; el8 sets A/B analogue): the NLL corpus, ~60K tokens of in-house docs/scripts, prefill-only passes
run_set prefill-text python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $R/prefill-text/bench
docker logs dsv41-kt 2>&1 | rg -i "expert distribution|Write expert" | tail -5 > $R/recorder-log.txt
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== done $(date +%T)"

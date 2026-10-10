#!/usr/bin/env bash
# T3.8b: qualify the el8-mix expert map (overlay c777da2d, --kt-expert-map) at launcher defaults.
# Serve 1: the T3.39 qualification set + kt CPU-seconds/step.  Serve 2: recorder histogram
# (realised CPU share on remapped ids).  Serve 3: positional comparator for kt CPU-seconds/step.
set -u
MAP=$PWD/config/expert-map-el8-mix-219.json
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }

D=evidence/t38b/qual-el8-mix
mkdir -p $D
echo "== qual serve $(date +%T)"
free -g > $D/pre-serve-state.txt
KT_EXPERT_MAP=$MAP bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; exit 2; }
docker logs dsv41-kt 2>&1 | rg -e "Expert remap" -e "remapped order" -e "engram host table" > $D/remap-log.txt; wc -l $D/remap-log.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
echo "== benches $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 3 --seed 777 --warm-pass --output $D/prefill-16k.json > $D/prefill-16k.txt 2>&1; tail -3 $D/prefill-16k.txt
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 3 --seed 1277 --warm-pass --output $D/prefill-32k.json > $D/prefill-32k.txt 2>&1; tail -3 $D/prefill-32k.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1.json > $D/c1.txt 2>&1; tail -1 $D/c1.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4.json > $D/c4.txt 2>&1; tail -1 $D/c4.txt
echo "== kt cpu/step $(date +%T)"
bash scripts/kt-cpu-per-step.sh --url $U --model $M --output $D/kt-cpu-per-step > $D/kt-cpu-per-step.txt 2>&1; tail -4 $D/kt-cpu-per-step.txt
echo "== nll $(date +%T)"
python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus --compare-dir evidence/t328/default-219-8192/corpus > $D/corpus.txt 2>&1; tail -2 $D/corpus.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t38b-qual --output $D/consistency.json > $D/consistency.txt 2>&1; tail -2 $D/consistency.txt
echo "== stress $(date +%T)"
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --seed 31 --output $D/stress-kv.json > $D/stress-kv.txt 2>&1; tail -3 $D/stress-kv.txt
python3 scripts/verify.py --url $U --model $M --context 65536 --output $D/verify.json > $D/verify.txt 2>&1; tail -2 $D/verify.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t38b-qual-post --output $D/consistency-post.json > $D/consistency-post.txt 2>&1; tail -1 $D/consistency-post.txt
echo "health=$(health)"
docker logs dsv41-kt 2>&1 | rg -e "dedicated runner|FlashInfer MoE" > $D/runner-log.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== qual done $(date +%T)"

# ---- serve 2: recorder histogram on remapped ids (same four sets as T3.8 round 2)
R=evidence/t38b/histogram-el8-mix
mkdir -p $R
echo "== histogram serve $(date +%T)"
KT_EXPERT_MAP=$MAP EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" bash scripts/serve-hybrid.sh > $R/launch.txt 2>&1 || { echo LAUNCHFAIL2; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $R/ready.txt 2>&1 || { echo READYFAIL2; docker logs dsv41-kt > $R/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $R/manifest.json
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 64 --output $R/warm.json > $R/warm.txt 2>&1; tail -1 $R/warm.txt
dumper() { while [ -f /tmp/t38b-dumping ]; do curl -s -X POST $U/dump_expert_distribution_record > /dev/null; sleep 10; done; }
run_set() {
  local name=$1; shift
  mkdir -p $R/$name
  docker exec dsv41-kt sh -c "rm -f /tmp/expert_distribution_recorder_*.pt"
  curl -s -X POST $U/start_expert_distribution_record > $R/$name/start.txt
  touch /tmp/t38b-dumping; dumper & local dp=$!
  echo "== set $name $(date +%T): $*"
  "$@"
  rm -f /tmp/t38b-dumping; wait $dp
  curl -s -X POST $U/dump_expert_distribution_record > $R/$name/dump.txt
  curl -s -X POST $U/stop_expert_distribution_record > $R/$name/stop.txt
  sleep 2
  docker exec dsv41-kt sh -c "mkdir -p /tmp/t38b-$name && mv /tmp/expert_distribution_recorder_*.pt /tmp/t38b-$name/"
  docker cp dsv41-kt:/tmp/t38b-$name/. $R/$name/ | tail -1
  ls $R/$name | wc -l
}
run_set humaneval-c4 python3 scripts/bench-humaneval.py --url $U --model $M --phase generate --limit 40 --concurrency 4 --max-tokens 2048 --output-dir $R/humaneval-c4/bench
run_set decode-distinct python3 scripts/bench-decode.py --url $U --model $M --concurrency 1,4 --runs 3 --warmup 0 --max-tokens 256 --distinct-prompts --fixture-set distinct --output $R/decode-distinct/bench.json
run_set prefill-random python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 1 --seed 777 --output $R/prefill-random/bench-16k.json
run_set prefill-text python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $R/prefill-text/bench
docker logs dsv41-kt 2>&1 | rg -i "expert distribution|Write expert" | tail -5 > $R/recorder-log.txt
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== histogram done $(date +%T)"

# ---- serve 3: positional comparator for kt CPU-seconds/step (same overlay, flag unset)
P=evidence/t38b/kt-cpu-positional
mkdir -p $P
echo "== positional serve $(date +%T)"
bash scripts/serve-hybrid.sh > $P/launch.txt 2>&1 || { echo LAUNCHFAIL3; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $P/ready.txt 2>&1 || { echo READYFAIL3; docker logs dsv41-kt > $P/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $P/manifest.json
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 64 --output $P/warm.json > $P/warm.txt 2>&1
bash scripts/kt-cpu-per-step.sh --url $U --model $M --output $P/kt-cpu-per-step > $P/kt-cpu-per-step.txt 2>&1; tail -4 $P/kt-cpu-per-step.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $P/c1.json > $P/c1.txt 2>&1; tail -1 $P/c1.txt
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== all done $(date +%T)"

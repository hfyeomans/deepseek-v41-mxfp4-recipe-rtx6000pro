#!/usr/bin/env bash
# T3.8b soak of the new default (KT_EXPERT_MAP = el8 mix map) on coding + long-reasoning traffic,
# recorder on for every set (realised CPU-tier share per traffic kind), then the E33 gates.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
R=evidence/t38b/soak-default
mkdir -p $R
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
edac() { for f in /sys/devices/system/edac/mc/mc*/ce_count /sys/devices/system/edac/mc/mc*/ue_count; do [ -f "$f" ] && echo "$f $(cat $f)"; done; }
echo "== serve $(date +%T)"
free -g > $R/pre-serve-state.txt; edac > $R/edac-before.txt
EXTRA_SERVER_ARGS="--expert-distribution-recorder-mode stat" bash scripts/serve-hybrid.sh > $R/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $R/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $R/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $R/manifest.json
docker logs dsv41-kt 2>&1 | rg -e "Expert remap" -e "remapped order" | wc -l > $R/remap-lines.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $R/vram-after-capture.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --runs 1 --warmup 1 --max-tokens 64 --output $R/warm.json > $R/warm.txt 2>&1

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

echo "== decode gate (start) $(date +%T)"
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $R/c1-start.json > $R/c1-start.txt 2>&1; tail -1 $R/c1-start.txt

# 1. coding, thinking off: the full HumanEval at C4 (pass@1 vs E32's 161/164 on checkpoint order)
run_set humaneval-full-c4 python3 scripts/bench-humaneval.py --url $U --model $M --phase generate --concurrency 4 --max-tokens 2048 --output-dir $R/humaneval-full-c4/bench
# 2. long reasoning, thinking on, C4: 32 problems, up to 8K tokens each
run_set humaneval-think-c4 python3 scripts/bench-humaneval.py --url $U --model $M --phase generate --thinking --limit 32 --concurrency 4 --max-tokens 8192 --output-dir $R/humaneval-think-c4/bench
# 3. long reasoning, thinking on, C1: 6 problems (single-stream wall under reasoning)
run_set humaneval-think-c1 python3 scripts/bench-humaneval.py --url $U --model $M --phase generate --thinking --limit 6 --seed 1 --concurrency 1 --max-tokens 8192 --output-dir $R/humaneval-think-c1/bench
# 4. the distinct fixtures, C1 and C4 (comparable with qual / T3.8)
run_set decode-distinct python3 scripts/bench-decode.py --url $U --model $M --concurrency 1,4 --runs 3 --warmup 0 --max-tokens 256 --distinct-prompts --fixture-set distinct --output $R/decode-distinct/bench.json
# 5. prefill text: the NLL corpus (also the NLL gate vs t328)
run_set prefill-text bash -c "python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $R/prefill-text/corpus --compare-dir evidence/t328/default-219-8192/corpus > $R/prefill-text/corpus.txt 2>&1"
tail -2 $R/prefill-text/corpus.txt | head -1

echo "== gates $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 3 --seed 777 --warm-pass --output $R/prefill-16k.json > $R/prefill-16k.txt 2>&1; tail -3 $R/prefill-16k.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t38b-soak --output $R/consistency.json > $R/consistency.txt 2>&1; tail -1 $R/consistency.txt
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --seed 31 --output $R/stress-kv.json > $R/stress-kv.txt 2>&1; tail -3 $R/stress-kv.txt
python3 scripts/verify.py --url $U --model $M --context 65536 --output $R/verify.json > $R/verify.txt 2>&1; tail -2 $R/verify.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $R/c1-end.json > $R/c1-end.txt 2>&1; tail -1 $R/c1-end.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $R/c4-end.json > $R/c4-end.txt 2>&1; tail -1 $R/c4-end.txt
echo "health=$(health)"
docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e OutOfMemory -e "expandable_segments" | rg -v "any later grow raises" > $R/health-hits.txt
docker logs dsv41-kt 2>&1 | rg -i "expert distribution|Write expert" | tail -3 > $R/recorder-log.txt
docker rm -f dsv41-kt > /dev/null 2>&1
edac > $R/edac-after.txt
echo "== score $(date +%T)"
python3 scripts/bench-humaneval.py --phase score --output-dir $R/humaneval-full-c4/bench > $R/humaneval-full-c4/score.txt 2>&1; tail -1 $R/humaneval-full-c4/score.txt
python3 scripts/bench-humaneval.py --phase score --output-dir $R/humaneval-think-c4/bench > $R/humaneval-think-c4/score.txt 2>&1; tail -1 $R/humaneval-think-c4/score.txt
python3 scripts/bench-humaneval.py --phase score --output-dir $R/humaneval-think-c1/bench > $R/humaneval-think-c1/score.txt 2>&1; tail -1 $R/humaneval-think-c1/score.txt
echo "== soak done $(date +%T)"

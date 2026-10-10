#!/usr/bin/env bash
# T3.20 (c): A/B of the chunk-ahead Engram fetch (overlay 9d481b0b, SGLANG_DSV41_ENGRAM_CHUNK_AHEAD)
# at launcher defaults. Serve "off" = ENGRAM_CHUNK_AHEAD=0 comparator, serve "on" = default.
# Each serve starts with the checkpoint's page cache dropped (loader fadvise), so the first
# pass of the 16K bench is the cold-box reading; --warm-pass adds the warm one.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }

serve_and_bench() {
  local name=$1 D=evidence/t320-engram-prefetch/chunk-ahead/$1; shift
  mkdir -p $D
  echo "== $name serve $(date +%T)"
  free -g > $D/pre-serve-state.txt
  env "$@" bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL-$name; return 1; }
  python3 scripts/wait-ready.py --url $U --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL-$name; docker logs dsv41-kt > $D/launch-container.txt 2>&1; docker rm -f dsv41-kt; return 2; }
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
  echo "== benches $(date +%T)"
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 3 --seed 777 --warm-pass --output $D/prefill-16k.json > $D/prefill-16k.txt 2>&1; tail -5 $D/prefill-16k.txt
  python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 3 --seed 1277 --warm-pass --output $D/prefill-32k.json > $D/prefill-32k.txt 2>&1; tail -5 $D/prefill-32k.txt
  docker logs dsv41-kt 2>&1 | rg -e "engram file table" > $D/engram-stats-after-prefill.txt
  python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1.json > $D/c1.txt 2>&1; tail -1 $D/c1.txt
  python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4.json > $D/c4.txt 2>&1; tail -1 $D/c4.txt
  echo "== nll $(date +%T)"
  python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus --compare-dir evidence/t328/default-219-8192/corpus > $D/corpus.txt 2>&1; tail -2 $D/corpus.txt
  python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t320c-$name --output $D/consistency.json > $D/consistency.txt 2>&1; tail -2 $D/consistency.txt
  echo "== stress $(date +%T)"
  python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --seed 31 --output $D/stress-kv.json > $D/stress-kv.txt 2>&1; tail -3 $D/stress-kv.txt
  python3 scripts/verify.py --url $U --model $M --context 65536 --output $D/verify.json > $D/verify.txt 2>&1; tail -2 $D/verify.txt
  python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t320c-$name-post --output $D/consistency-post.json > $D/consistency-post.txt 2>&1; tail -1 $D/consistency-post.txt
  echo "health=$(health)" | tee $D/health.txt
  docker logs dsv41-kt 2>&1 | rg -e "engram file table" -e "fetched ahead of use" > $D/engram-stats.txt
  cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
  docker rm -f dsv41-kt > /dev/null 2>&1
  echo "== $name done $(date +%T)"
}

serve_and_bench off ENGRAM_CHUNK_AHEAD=0
serve_and_bench on
echo "== all done $(date +%T)"

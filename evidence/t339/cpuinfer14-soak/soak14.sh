#!/usr/bin/env bash
# T3.39: KT_CPUINFER=14 soak on the qualified overlay 99128a7e, same gates as qual.sh (owner-approved default candidate).
# benches + NLL + same-prompt, then 4 x 62K stress + needles, one serve.
set -u
D=evidence/t339/cpuinfer14-soak
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
mkdir -p $D
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
echo "== qual serve $(date +%T)"
free -g > $D/pre-serve-state.txt; fincore ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/model-0004[78]*.safetensors >> $D/pre-serve-state.txt
KT_CPUINFER=14 bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; exit 2; }
docker logs dsv41-kt 2>&1 | rg "engram host table" > $D/engram-describe.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
echo "== benches $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 3 --seed 777 --warm-pass --output $D/prefill-16k.json > $D/prefill-16k.txt 2>&1; tail -3 $D/prefill-16k.txt
python3 scripts/bench-prefill.py --url $U --model $M --tokens 32768 --runs 3 --seed 1277 --warm-pass --output $D/prefill-32k.json > $D/prefill-32k.txt 2>&1; tail -3 $D/prefill-32k.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1.json > $D/c1.txt 2>&1; tail -1 $D/c1.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4.json > $D/c4.txt 2>&1; tail -1 $D/c4.txt
echo "== nll $(date +%T)"
python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus --compare-dir evidence/t328/default-219-8192/corpus > $D/corpus.txt 2>&1; tail -2 $D/corpus.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t339-cpuinfer14 --output $D/consistency.json > $D/consistency.txt 2>&1; tail -2 $D/consistency.txt
echo "== stress $(date +%T)"
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --seed 31 --output $D/stress-kv.json > $D/stress-kv.txt 2>&1; tail -3 $D/stress-kv.txt
python3 scripts/verify.py --url $U --model $M --context 65536 --output $D/verify.json > $D/verify.txt 2>&1; tail -2 $D/verify.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t339-cpuinfer14-post --output $D/consistency-post.json > $D/consistency-post.txt 2>&1; tail -1 $D/consistency-post.txt
echo "health=$(health)"
docker logs dsv41-kt 2>&1 | rg -e "dedicated runner|FlashInfer MoE" > $D/runner-log.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== qual done $(date +%T)"

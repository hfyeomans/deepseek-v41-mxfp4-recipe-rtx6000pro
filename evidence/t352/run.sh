#!/usr/bin/env bash
# T3.52: qualified serve at launcher defaults + HOST_THREAD_CPUS=auto (host threads pinned to the SMT
# siblings 16-31; kt workers keep cores 0-15). Full release gate, then second C1/C4 pass and spec consistency.
set -u
D=evidence/t352
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
echo "== smoke $(date +%T)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
IDENTITY_CORPUS=evidence/t341/serve-gemv/corpus bash scripts/release-smoke.sh serve --output-dir $D --context 262144 --slots 4 --full > $D/smoke-log.txt 2>&1; echo "smoke rc=$?"
echo "== reruns $(date +%T)"
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1-rerun.json > $D/c1-rerun.txt 2>&1; tail -1 $D/c1-rerun.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4-rerun.json > $D/c4-rerun.txt 2>&1; tail -1 $D/c4-rerun.txt
echo "== spec-consistency $(date +%T)"
python3 scripts/spec-consistency.py --url $U --output $D/spec-consistency.json --compare evidence/t341/serve-gemv/spec-consistency.json --label t352 > $D/spec-consistency.txt 2>&1; tail -3 $D/spec-consistency.txt
echo "health=$(health)"
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
echo "== done $(date +%T)"

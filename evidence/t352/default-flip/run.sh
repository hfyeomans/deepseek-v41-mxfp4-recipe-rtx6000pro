#!/usr/bin/env bash
# T3.52 default flip: the plain launcher (no overrides) after HOST_THREAD_CPUS defaulted to `auto`.
# Full release gate (the ladder rule: one default serve per default flip), then a second C1 / C4 pass.
set -u
D=evidence/t352/default-flip
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
docker exec dsv41-kt sh -c 'for t in /proc/[0-9]*/task/*; do c=$(cat $t/comm 2>/dev/null)||continue; echo "$c $(grep Cpus_allowed_list $t/status|cut -f2)"; done' | sort | uniq -c | sort -rn > $D/thread-affinity.txt
IDENTITY_CORPUS=evidence/t352/corpus bash scripts/release-smoke.sh serve --output-dir $D --context 262144 --slots 4 --full > $D/smoke-log.txt 2>&1; echo "smoke rc=$?" > $D/smoke-rc.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1-rerun.json > $D/c1-rerun.txt 2>&1
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4-rerun.json > $D/c4-rerun.txt 2>&1
docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l > $D/health.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
echo done > $D/run-done.txt

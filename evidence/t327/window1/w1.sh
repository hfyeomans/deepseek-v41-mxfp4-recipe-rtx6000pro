#!/usr/bin/env bash
# T3.27 window 1 — entry-condition timeline on the qualified profile (overlay 124ca1de, image dsv41-kt:recipe,
# launcher defaults 64K x 4 / chunk 8192 / KT_CPUINFER=16 / el8 map). One serve:
#   KT_TIMELINE=1 (in-graph per-layer buckets, T3.19a method) under the standard C1 and C4 recipes,
#   then one torch-profiler trace (CPU+GPU, 8 steps) of a C1 decode and one of a C4 decode
#   (E20 method, scripts/step-profile.py) to split "other_gpu" into dense FP8 GEMM / attention / all-reduce.
# Answers: what share of the current decode step is the dense FP8 GEMMs and the resident MXFP4 grouped GEMM,
# i.e. whether T3.27's entry condition (>= 10 % of the targeted path) holds after T3.8b and T3.39.
set -u
D=evidence/t327/window1
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
mkdir -p $D
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
echo "== serve $(date +%T)"
docker rm -f dsv41-kt > /dev/null 2>&1; pgrep -f "wait-ready.py" > /dev/null && { echo STRAYFAIL; exit 3; }
EXTRA_DOCKER_ENV="KT_TIMELINE=1 KT_TIMELINE_DIR=/tmp/kt-timeline KT_TIMELINE_DUMP_MAX=0" bash scripts/serve-hybrid.sh > $D/launch.txt 2>&1 || { echo LAUNCHFAIL; exit 1; }
python3 scripts/wait-ready.py --url $U --timeout 1800 > $D/ready.txt 2>&1 || { echo READYFAIL; docker logs dsv41-kt > $D/launch-container.txt 2>&1; docker rm -f dsv41-kt; exit 2; }
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
echo "== timeline benches $(date +%T)"
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1.json > $D/c1.txt 2>&1; tail -1 $D/c1.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4.json > $D/c4.txt 2>&1; tail -1 $D/c4.txt
echo "== profiler C1 $(date +%T)"
docker exec dsv41-kt mkdir -p /tmp/prof-c1 /tmp/prof-c4
curl -s -X POST $U/start_profile -H 'content-type: application/json' -d '{"output_dir":"/tmp/prof-c1","num_steps":8,"activities":["CPU","GPU"],"with_stack":false}' > $D/start-profile-c1.txt; cat $D/start-profile-c1.txt; echo
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 400 --runs 1 --warmup 0 --distinct-prompts --seed 4242 --output $D/prof-c1-bench.json > $D/prof-c1-bench.txt 2>&1; tail -1 $D/prof-c1-bench.txt
sleep 5; curl -s -X POST $U/stop_profile > $D/stop-profile-c1.txt 2>&1; sleep 45
echo "== profiler C4 $(date +%T)"
curl -s -X POST $U/start_profile -H 'content-type: application/json' -d '{"output_dir":"/tmp/prof-c4","num_steps":8,"activities":["CPU","GPU"],"with_stack":false}' > $D/start-profile-c4.txt; cat $D/start-profile-c4.txt; echo
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 400 --runs 1 --warmup 0 --distinct-prompts --seed 4242 --output $D/prof-c4-bench.json > $D/prof-c4-bench.txt 2>&1; tail -1 $D/prof-c4-bench.txt
sleep 5; curl -s -X POST $U/stop_profile > $D/stop-profile-c4.txt 2>&1; sleep 45
echo "== collect $(date +%T)"
mkdir -p $D/prof-c1 $D/prof-c4 $D/timeline
docker cp dsv41-kt:/tmp/prof-c1/. $D/prof-c1/ 2>&1 | tail -1
docker cp dsv41-kt:/tmp/prof-c4/. $D/prof-c4/ 2>&1 | tail -1
docker cp dsv41-kt:/tmp/kt-timeline/. $D/timeline/ 2>&1 | tail -1
ls -la $D/prof-c1 $D/prof-c4 $D/timeline
echo "health=$(health)"
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
docker rm -f dsv41-kt > /dev/null 2>&1
echo "== done $(date +%T)"

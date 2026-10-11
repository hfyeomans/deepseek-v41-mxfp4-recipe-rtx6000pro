#!/usr/bin/env bash
# E27-shape soak on the running 262K x 4 serve (default-flip precondition): 8 blocks x 12 runs x 512 tokens,
# C4/C3/C2/C1 at distinct fixture offsets, health check after every block.
set -u
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
D=evidence/t337/262k4-soak
mkdir -p $D
for spec in "4 0" "3 2" "2 4" "4 8" "1 6" "4 12" "3 9" "2 0"; do
  set -- $spec; C=$1; OFF=$2
  echo "== c$C-off$OFF $(date +%T)"
  python3 scripts/bench-decode.py --url $U --model $M --concurrency "$C" --fixture-offset "$OFF" --max-tokens 512 --runs 12 --warmup 0 --distinct-prompts --seed 4242 --output "$D/c$C-off$OFF.json" > "$D/c$C-off$OFF.txt" 2>&1
  sed -n 3p "$D/c$C-off$OFF.txt"
  docker logs dsv41-kt 2>&1 | grep -E -i -c '\bnan\b|xid|traceback|out of memory|contract' >> $D/health-hits.txt
done
docker logs dsv41-kt 2>&1 | grep -c 'memory mapping failed' > $D/mapping-failures.txt
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
echo "== done $(date +%T)"

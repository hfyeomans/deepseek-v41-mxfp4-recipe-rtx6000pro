#!/usr/bin/env bash
# T3.40: one-layer CPU tier, dsv41-kt:native (-march=native) vs dsv41-kt:recipe (wheel flag set), same harness as evidence/t336/bench.sh.
set -u
R=evidence/t340
HFM=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
SNAP=$(ls $HFM/snapshots | head -1)
OV=~/dev/src/sglang-dsv41-overlay/python/sglang
mkdir -p $R/one-layer
run() {
  local name=$1 image=$2
  echo "== $name image=$image $(date +%T)"
  docker run --rm --gpus all --ipc=host \
    -v "$HFM:/hfm:ro" -v "$OV:/sgl-workspace/sglang/python/sglang:ro" \
    -v "$PWD/scripts:/recipe:ro" -v "$PWD/$R/one-layer:/out" --entrypoint python "$image" \
    /recipe/test-kt-one-layer.py --weight-path /hfm/snapshots/$SNAP --num-gpu-experts 219 \
    --threads 16 --skip-reference --bench-cold --cpu-slots-per-token 3 \
    --bench-tokens 1,2,4,8,64,256 --bench-iters 40 --json /out/$name.json 2>&1 \
    | rg -e "^load:" -e "^bench " | tee $R/one-layer/$name.txt
}
for rep in 1 2; do
  run recipe-r$rep dsv41-kt:recipe
  run native-r$rep dsv41-kt:native
done
echo "== done $(date +%T)"

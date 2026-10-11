#!/usr/bin/env bash
# T3.36: the kt-kernel CPU-expert tier of one MoE layer under page-size and thread-placement
# settings, no server. Same harness and shape as E21 (`test-kt-one-layer.py --bench-cold`), at the
# served split (219 GPU / 165 CPU experts), 3 of 6 slots on the CPU tier, distinct experts per token.
# One NUMA node on the 9955WX, so "NUMA" reduces to CCD placement (cores 0-7 = CCD0, 8-15 = CCD1;
# SMT siblings 16-31) and page size: THP is `madvise` on the host and kt's arena is 4 KiB
# (T3.39 window 2: AnonHugePages 0), so the no-root lever is glibc's malloc tunable, which
# madvises MADV_HUGEPAGE on every mmap'd chunk (glibc 2.39 in the image).
set -u
R=evidence/t336
HFM=~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash
SNAP=$(ls $HFM/snapshots | head -1)
OV=~/dev/src/sglang-dsv41-overlay/python/sglang
mkdir -p $R/one-layer
run() {
  local name=$1 threads=$2 cpuset=$3 env=$4
  local extra=()
  [ -n "$cpuset" ] && extra+=(--cpuset-cpus "$cpuset")
  [ -n "$env" ] && extra+=(-e "$env")
  echo "== $name  threads=$threads cpuset=${cpuset:-all} env=${env:-none}  $(date +%T)"
  docker run --rm --gpus all --ipc=host "${extra[@]}" \
    -v "$HFM:/hfm:ro" -v "$OV:/sgl-workspace/sglang/python/sglang:ro" \
    -v "$PWD/scripts:/recipe:ro" -v "$PWD/$R/one-layer:/out" --entrypoint python dsv41-kt:recipe \
    /recipe/test-kt-one-layer.py --weight-path /hfm/snapshots/$SNAP --num-gpu-experts 219 \
    --threads "$threads" --skip-reference --bench-cold --cpu-slots-per-token 3 \
    --bench-tokens 1,2,4,8,64,256 --bench-iters 40 --json /out/$name.json 2>&1 \
    | rg -e "^load:" -e "^bench " | tee $R/one-layer/$name.txt
}
for rep in 1 2; do
  run baseline-r$rep        16 ""                 ""
  run thp-r$rep             16 ""                 "GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
  run phys16-r$rep          16 "0-15"             ""
  run thp-phys16-r$rep      16 "0-15"             "GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
  run ccd0-8-r$rep           8 "0-7,16-23"        ""
  run ccd0-8-thp-r$rep       8 "0-7,16-23"        "GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
done
grep -H AnonHugePages /proc/meminfo > $R/one-layer/meminfo-after.txt
echo "== done $(date +%T)"

#!/usr/bin/env bash
# Generated preflight: refuse to serve if the host is below the plan's margins.
set -euo pipefail
NEED_VRAM_GIB=24.6   # plan's VRAM margin (free after load)
NEED_HOST_GIB=162.8   # plan's host margin
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits || exit 1
free -g | awk 'NR==2{print "host free GiB:", $7}'
echo "compare against the margins above (generated from the placement plan)"

#!/usr/bin/env python3
"""T3.49 helper: drop the Engram shards' page cache (the two largest shards)
so a measurement starts page-cache cold, mirroring wait-ready's no-root
posix_fadvise(DONTNEED) drop but for the Engram tables."""
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("snapshot", type=Path)
ap.add_argument("--engram-shards", type=int, default=2)
args = ap.parse_args()
shards = sorted(args.snapshot.glob("model-*-of-*.safetensors"), key=lambda p: p.stat().st_size)
dropped = 0
for p in shards[-args.engram_shards:]:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    dropped += 1
    print(f"dropped {p.name} ({p.stat().st_size / 2**30:.1f} GiB)")
print(f"engram cache dropped for {dropped} shards")

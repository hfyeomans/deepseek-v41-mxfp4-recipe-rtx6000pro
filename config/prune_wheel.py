"""Strip vendored third-party packages out of a wheel before installing it.

kt-kernel 0.7.0.post4 ships ~1,000 files that are not kt-kernel: a cu12 copy of
deep_gemm, sgl_kernel 0.3.21, triton_kernels, flash_attn_origin, mscclpp.
`pip install --no-deps` unpacks all of them over the base image's newer copies
(TROUBLESHOOTING.md: "libnvrtc.so.12 at decode graph capture"). This keeps only
the entries whose top-level directory belongs to the named distribution and
filters RECORD to match.

Usage: prune_wheel.py in.whl out_dir [dist_name]
"""

import re
import sys
import zipfile
from pathlib import Path


def keep_prefixes(wheel: Path, dist: str) -> tuple[str, ...]:
    m = re.match(rf"({re.escape(dist)})-([^-]+)-", wheel.name)
    if not m:
        sys.exit(f"cannot parse distribution/version from {wheel.name}")
    version = m.group(2)
    return (
        f"{dist}/",
        f"{dist}.libs/",
        f"{dist}-{version}.dist-info/",
        f"{dist}-{version}.data/",
    )


def main() -> int:
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    dist = sys.argv[3] if len(sys.argv) > 3 else "kt_kernel"
    prefixes = keep_prefixes(src, dist)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name

    kept = dropped = 0
    dropped_tops: set[str] = set()
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if not info.filename.startswith(prefixes):
                dropped += 1
                dropped_tops.add(info.filename.split("/")[0])
                continue
            data = zin.read(info)
            if info.filename.endswith(".dist-info/RECORD"):
                lines = data.decode().splitlines()
                data = "\n".join(l for l in lines if l.split(",")[0].startswith(prefixes)).encode() + b"\n"
            zout.writestr(info, data)
            kept += 1

    print(f"{src.name}: kept {kept} entries, dropped {dropped} from {sorted(dropped_tops)}")
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

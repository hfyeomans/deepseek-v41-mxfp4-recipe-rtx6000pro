#!/usr/bin/env python3
"""Read the live Infinity Fabric clock (FCLK), memory clock (MCLK) and DDR bandwidth from the
AMD HSMP mailbox (BIOS Tier 3 baseline "T3-0", docs/BIOS-OPTIMIZATION-GUIDE.md §6).

Uses only read-only GET messages through /dev/hsmp (module hsmp_acpi, ACPI id AMDI0097), which
this kernel exposes world-readable — no root, no third-party SMU driver, no reboot. UCLK is not
exposed by HSMP (protocol 7 has no UCLK message); it stays "not observable from Linux".

Message IDs and the struct hsmp_message layout follow arch/x86/include/uapi/asm/amd_hsmp.h:
  0x0F GET_FCLK_MCLK      -> [fclk MHz, mclk MHz]  (DDR rate = 2 x MCLK)
  0x14 GET_DDR_BANDWIDTH  -> theoretical GB/s << 20 | utilised GB/s << 8 | utilisation %
  0x04 GET_SOCKET_POWER   -> mW
  0x10 GET_CCLK_THROTTLE  -> core clock limit MHz

  python3 scripts/hsmp-clocks.py                 # one sample
  python3 scripts/hsmp-clocks.py --watch 1 --n 30   # 30 samples one second apart (run a load alongside)
  python3 scripts/hsmp-clocks.py --json out.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import struct
import sys
import time

DEV = "/dev/hsmp"
# struct hsmp_message { u32 msg_id; u16 num_args; u16 response_sz; u32 args[8]; u16 sock_ind; }
FMT = "<IHH8IH2x"
SIZE = struct.calcsize(FMT)
IOCTL = (3 << 30) | (SIZE << 16) | (0xF8 << 8)  # _IOWR(0xF8, 0, struct hsmp_message)

GET_SOCKET_POWER = 0x04
GET_FCLK_MCLK = 0x0F
GET_CCLK_THROTTLE_LIMIT = 0x10
GET_DDR_BANDWIDTH = 0x14


def query(msg_id: int, response_sz: int, sock: int = 0) -> tuple[int, ...]:
    buf = bytearray(struct.pack(FMT, msg_id, 0, response_sz, *([0] * 8), sock))
    fd = os.open(DEV, os.O_RDONLY)
    try:
        fcntl.ioctl(fd, IOCTL, buf)
    finally:
        os.close(fd)
    return struct.unpack(FMT, buf)[3 : 3 + response_sz]


def sample() -> dict:
    fclk, mclk = query(GET_FCLK_MCLK, 2)
    (bw,) = query(GET_DDR_BANDWIDTH, 1)
    (power_mw,) = query(GET_SOCKET_POWER, 1)
    (cclk_limit,) = query(GET_CCLK_THROTTLE_LIMIT, 1)
    return {
        "t": time.time(),
        "fclk_mhz": fclk,
        "mclk_mhz": mclk,
        "ddr_rate_mts": 2 * mclk,
        "ddr_theoretical_gbps": bw >> 20,
        "ddr_utilised_gbps": (bw >> 8) & 0xFFF,
        "ddr_utilisation_pct": bw & 0xFF,
        "socket_power_w": power_mw / 1000.0,
        "cclk_limit_mhz": cclk_limit,
    }


def fmt(s: dict) -> str:
    return (
        f"FCLK {s['fclk_mhz']} MHz  MCLK {s['mclk_mhz']} MHz (DDR5-{s['ddr_rate_mts']})  "
        f"DDR {s['ddr_utilised_gbps']}/{s['ddr_theoretical_gbps']} GB/s ({s['ddr_utilisation_pct']} %)  "
        f"socket {s['socket_power_w']:.1f} W  CCLK limit {s['cclk_limit_mhz']} MHz"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=float, default=0, metavar="SEC", help="sample every SEC seconds")
    ap.add_argument("--n", type=int, default=0, help="number of samples in --watch mode (0 = until Ctrl-C)")
    ap.add_argument("--json", help="write all samples to this JSON file")
    args = ap.parse_args()

    if not os.path.exists(DEV):
        print(f"{DEV} missing: hsmp_acpi not loaded or firmware exposes no HSMP", file=sys.stderr)
        return 2

    samples = []
    try:
        while True:
            s = sample()
            samples.append(s)
            print(fmt(s), flush=True)
            if args.watch <= 0 or (args.n and len(samples) >= args.n):
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": DEV, "samples": samples}, f, indent=1)
        print(f"wrote {args.json}")
    if len(samples) > 1:
        f = sorted(x["fclk_mhz"] for x in samples)
        m = sorted(x["mclk_mhz"] for x in samples)
        print(f"FCLK min/max {f[0]}/{f[-1]} MHz  MCLK min/max {m[0]}/{m[-1]} MHz over {len(samples)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())

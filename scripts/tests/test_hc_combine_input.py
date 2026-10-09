#!/usr/bin/env python3
"""GPU bit-equality test: `hc_combine` on a bf16 input equals `hc_combine` on its fp32
copy (recipe E34d-v (ii)). Runs inside the recipe image via
scripts/tests/run-prefill-transient-tests.sh.

Background: the model tail (`DeepseekV4Model.forward`, `hc_pre_from_prev_sublayer`)
called `hc_combine(pre_hc_head.float(), ...)` on the `[rows, hc_mult * hidden]` bf16
hyper-connection stream: a 640 MiB fp32 copy per 8192-row chunk (hc_mult 4, hidden 5120),
the allocation the 1M x 1 prefill OOMed on at a ~300K prefix once the indexer gather no
longer did. The Triton kernel casts every load to fp32 itself, so passing bf16 is the same
arithmetic (bf16 -> fp32 is exact) and the copy is gone. The two other `hc_combine`
callers already pass bf16.

What a wrong implementation would get wrong: a kernel that multiplied in the input dtype
(bf16 x bf16 products rounded before the sum) would differ from the fp32-input run at
almost every element; a kernel reading the input with the wrong element stride would
differ in shape or values. Compared at the served shape (8192 rows) and at 1 row.
"""
from __future__ import annotations

import sys

import torch

from sglang.kernels.ops.layernorm.mhc import hc_combine

DEV = torch.device("cuda")
HC, HIDDEN = 4, 5120  # DeepSeek-V4.1-Flash: hc_mult 4, hidden 5120 (x_flat width 20480)


def test_bf16_input_equals_fp32_input() -> None:
    g = torch.Generator(device=DEV).manual_seed(20261009)
    for rows in (1, 37, 8192):
        x = torch.randn((rows, HC * HIDDEN), generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        pre = torch.randn((rows, HC), generator=g, device=DEV, dtype=torch.float32).to(torch.bfloat16)
        want = hc_combine(x.float(), pre, HC, torch.bfloat16)
        got = hc_combine(x, pre, HC, torch.bfloat16)
        assert got.shape == want.shape == (rows, HIDDEN), (rows, got.shape)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), f"rows={rows}"


def main() -> int:
    if not torch.cuda.is_available():
        print("hc_combine input tests: no GPU; skipped")
        return 3
    try:
        test_bf16_input_equals_fp32_input()
        print("PASS hc_combine bf16 input == fp32 input")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL hc_combine bf16 input == fp32 input: {type(e).__name__}: {e}")
        print("FAILED: 1 failure(s)")
        return 1
    print("OK: 0 failure(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

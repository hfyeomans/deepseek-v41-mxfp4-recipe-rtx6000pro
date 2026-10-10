"""K2 window 2 smoke (no GPU): the overlay's KtRuntime arena path against real checkpoint experts.

1. KtRuntime.weight_arena_bytes on a real compact KTMoEWrapper (layer L, 165 CPU experts)
   equals the BufferB formula.
2. KtRuntime.load_cpu_experts(..., weight_arena=ptr) makes kt carve the experts into a
   memfd arena (MOEConfig subclass injection), restores kt_amx.MOEConfig afterwards, and
   the post-load weight_arena_layout check passes; ArenaLayout.from_kt accepts it.
3. CPU forwards (decode + prefill sizes) through the arena-loaded MoE are bit-identical
   to a second wrapper loaded the stock way (no arena) from the same checkpoint experts.
4. A forked child opens /proc/<pid>/fd/<fd>, maps the arena MAP_SHARED and sees the same
   bytes at the layout offsets (what TP rank 1 does in attach_layer).
"""
import ctypes
import hashlib
import mmap
import os
import sys
import time

import torch

sys.path.insert(0, "/stub")
from sglang.srt.layers.moe.kt_runtime import KtRuntime  # noqa: E402
from sglang.srt.layers.moe.kt_prefill import ArenaLayout  # noqa: E402

import kt_kernel  # noqa: E402,F401
import kt_kernel.utils.amx as kt_amx  # noqa: E402
from kt_kernel import KTMoEWrapper  # noqa: E402

LAYER = int(os.environ.get("SMOKE_LAYER", "1"))
E_TOTAL, N_GPU, K, H, I = 384, 219, 6, 5120, 2304  # config.json text_config: hidden_size 5120, moe_intermediate_size 2304
N_CPU = E_TOTAL - N_GPU
WEIGHT_PATH = os.environ["SMOKE_WEIGHT_PATH"]

mask = torch.zeros(E_TOTAL, dtype=torch.bool)
mask[:N_GPU] = True  # CPU experts = checkpoint ids 219..383 in order (remap is orthogonal)


def make_wrapper(layer_idx):
    return KTMoEWrapper(
        layer_idx=layer_idx,
        num_experts=N_CPU,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=I,
        gpu_experts_mask=None,
        cpuinfer_threads=16,
        threadpool_count=1,
        weight_path=WEIGHT_PATH,
        chunked_prefill_size=8192,
        method="MXFP4",
        max_deferred_experts_per_token=0,
        swiglu_limit=0.0,
    )


rt = KtRuntime(tp_rank=0)
stock_cfg = kt_amx.MOEConfig

# --- 1. size
w_arena = make_wrapper(LAYER)
nbytes = rt.weight_arena_bytes(w_arena)


def al(x):
    return (x + 63) // 64 * 64


expect = N_CPU * (2 * al(I * H // 2 + 4 * I * (H // 32)) + al(H * I // 2 + 4 * H * (I // 32)))
print(f"weight_arena_bytes={nbytes} ({nbytes / 2**30:.3f} GiB) expected={expect}")
assert nbytes == expect
assert kt_amx.MOEConfig is stock_cfg

# --- 2. arena load through the overlay runtime
fd = os.memfd_create(f"kt_arena_L{LAYER}", getattr(os, "MFD_ALLOW_SEALING", 2))
os.ftruncate(fd, nbytes)
mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
assert base % 64 == 0
t0 = time.perf_counter()
rt.load_cpu_experts(w_arena, torch.arange(N_CPU), mask, None, weight_arena=base)
print(f"arena load: {time.perf_counter() - t0:.1f} s; MOEConfig restored: {kt_amx.MOEConfig is stock_cfg}")
assert kt_amx.MOEConfig is stock_cfg
layout = w_arena.moe.weight_arena_layout()
lay = ArenaLayout.from_kt(layout, nbytes, N_CPU)
print(f"layout: gate0={lay.gate0} up0={lay.up0} down0={lay.down0} stride={lay.stride}")
assert lay.gate0 == 0 and lay.stride == 2 * al(I * H // 2 + 4 * I * (H // 32)) + al(H * I // 2 + 4 * H * (I // 32))
assert lay.stride * (N_CPU - 1) + lay.down0 + al(H * I // 2 + 4 * H * (I // 32)) <= nbytes

# --- 3. bit-identical forwards vs the stock (no arena) load
w_stock = make_wrapper(LAYER)
t0 = time.perf_counter()
rt.load_cpu_experts(w_stock, torch.arange(N_CPU), mask, None)
print(f"stock load: {time.perf_counter() - t0:.1f} s")
try:
    w_stock.moe.weight_arena_layout()
    raise SystemExit("stock-loaded moe reports an arena layout")
except RuntimeError:
    pass


def fwd(w, q):
    g = torch.Generator().manual_seed(q)
    x = torch.randn(q, H, generator=g).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(N_CPU, generator=g)[:K] for _ in range(q)]).to(torch.int64)
    wt = torch.rand(q, K, generator=g)
    out = torch.zeros(q, H, dtype=torch.bfloat16)
    bsz = torch.tensor([q], dtype=torch.int32)
    w.cpu_infer.submit(w.moe.forward_task(bsz.data_ptr(), K, ids.data_ptr(), wt.data_ptr(), x.data_ptr(), out.data_ptr(), False))
    w.cpu_infer.sync()
    return out


for q in (1, 4, 64, 2048):
    a, b = fwd(w_stock, q), fwd(w_arena, q)
    assert torch.isfinite(a.float()).all() and a.abs().sum() > 0, "stock output degenerate"
    assert torch.equal(a, b), f"forward mismatch at qlen={q}: max |d|={(a.float() - b.float()).abs().max()}"
    print(f"qlen={q}: arena forward bit-identical to stock load (|out| mean {a.float().abs().mean():.4f})")

# --- 4. cross-process mapping via /proc/<pid>/fd/<fd>
view = torch.frombuffer(mm, dtype=torch.uint8)
probe = [(lay.gate0, 1 << 20), (lay.up0 + 7 * lay.stride, 1 << 20), (lay.down0 + (N_CPU - 1) * lay.stride, 1 << 20)]
want = [hashlib.sha256(view[o : o + n].numpy().tobytes()).hexdigest() for o, n in probe]
r, w = os.pipe()
pid = os.fork()
if pid == 0:
    os.close(r)
    try:
        cfd = os.open(f"/proc/{os.getppid()}/fd/{fd}", os.O_RDWR)
        assert os.fstat(cfd).st_size == nbytes
        cmm = mmap.mmap(cfd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        cv = torch.frombuffer(cmm, dtype=torch.uint8)
        got = [hashlib.sha256(cv[o : o + n].numpy().tobytes()).hexdigest() for o, n in probe]
        os.write(w, ("\n".join(got) + "\n").encode())
        os._exit(0)
    except Exception as e:  # noqa: BLE001
        os.write(w, f"ERR {type(e).__name__}: {e}\n".encode())
        os._exit(1)
os.close(w)
child = b""
while True:
    chunk = os.read(r, 65536)
    if not chunk:
        break
    child += chunk
_, status = os.waitpid(pid, 0)
lines = child.decode().strip().split("\n")
print("child:", "ok" if status == 0 else lines)
assert status == 0 and lines == want, (lines, want)
print("child sees identical arena bytes at 3 layout offsets through /proc/<pid>/fd")
print("SMOKE OK")

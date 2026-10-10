"""K2 smoke: AMXFP4_KGroup_MOE on a caller-owned memfd arena vs the stock aligned_alloc path.

Checks: weight_arena_bytes matches the BufferB formula, weight_arena_layout offsets are
64-aligned and non-overlapping within the arena, the arena bytes at the layout offsets
are the packed weights (nibbles) and compacted E8 scales, and decode + prefill forwards
are bit-identical between arena and non-arena instances.
"""
import ctypes
import mmap
import os
import sys

import torch
import kt_kernel  # noqa: F401 (selects the ISA variant)
import kt_kernel.kt_kernel_ext as ext

E, K, H, I, G = 4, 2, 4096, 2880, 32
torch.manual_seed(0)

wc = ext.WorkerPoolConfig()
wc.subpool_count = 1
wc.subpool_numa_map = [0]
wc.subpool_thread_count = [8]
infer = ext.CPUInfer(wc)

def rand_scales(n, k):
    # ue8m0 exponents in [-6, 1] -> exact powers of two, bf16-representable, e8 compactable
    e = torch.randint(-6, 2, (n, k // G))
    return torch.ldexp(torch.ones(n, k // G), e).to(torch.bfloat16).contiguous()

gate = [torch.randint(0, 256, (I, H // 2), dtype=torch.uint8) for _ in range(E)]
up = [torch.randint(0, 256, (I, H // 2), dtype=torch.uint8) for _ in range(E)]
down = [torch.randint(0, 256, (H, I // 2), dtype=torch.uint8) for _ in range(E)]
gate_s = [rand_scales(I, H) for _ in range(E)]
up_s = [rand_scales(I, H) for _ in range(E)]
down_s = [rand_scales(H, I) for _ in range(E)]
mask = torch.zeros(E, dtype=torch.bool)  # all experts on CPU
p2l = torch.arange(E, dtype=torch.int64)

def make_cfg():
    c = ext.moe.MOEConfig(E, K, H, I, mask.data_ptr())
    c.layer_idx = 0
    c.pool = infer.backend_
    c.max_len = 8192
    c.quant_config.bits = 4
    c.quant_config.group_size = G
    c.quant_config.zero_point = False
    c.gate_projs = [[t.data_ptr() for t in gate]]
    c.up_projs = [[t.data_ptr() for t in up]]
    c.down_projs = [[t.data_ptr() for t in down]]
    c.gate_scales = [[t.data_ptr() for t in gate_s]]
    c.up_scales = [[t.data_ptr() for t in up_s]]
    c.down_scales = [[t.data_ptr() for t in down_s]]
    return c

M = ext.moe.AMXFP4_KGroup_MOE
nbytes = M.weight_arena_bytes(make_cfg())
def al(x): return (x + 63) // 64 * 64
expect = E * (2 * al(I * H // 2 + 4 * I * (H // G)) + al(H * I // 2 + 4 * H * (I // G)))
print(f"weight_arena_bytes={nbytes} ({nbytes/2**20:.1f} MiB) expected={expect}")
assert nbytes == expect

fd = os.memfd_create("k2-arena", 0)
os.ftruncate(fd, nbytes)
mm = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
assert base % 64 == 0, base
arena_view = torch.frombuffer(mm, dtype=torch.uint8)

def load(c):
    m = M(c)
    infer.submit(m.load_weights_task(p2l.data_ptr()))
    infer.sync()
    return m

m_ref = load(make_cfg())
c2 = make_cfg()
c2.weight_arenas = [base]
m_arena = load(c2)

layout = m_arena.weight_arena_layout()
assert len(layout) == 1 and len(layout[0]) == E, layout
spans = []
for e, (g, u, d, ge8, ue8, de8) in enumerate(layout[0]):
    assert ge8 and ue8 and de8, (e, ge8, ue8, de8)
    for off in (g, u, d):
        assert off % 64 == 0 and off < nbytes, (e, off)
    spans += [(g, al(I * H // 2 + 4 * I * (H // G))), (u, al(I * H // 2 + 4 * I * (H // G))), (d, al(H * I // 2 + 4 * H * (I // G)))]
spans.sort()
for (a, la), (b, _) in zip(spans, spans[1:]):
    assert a + la <= b, ("overlap", a, la, b)
assert spans[-1][0] + spans[-1][1] <= nbytes
print("layout:", layout[0][0], "... non-overlapping, 64-aligned")

# Arena content at expert 0: packed nibbles then compacted e8 scales (ue8m0 byte = bf16 bits >> 7).
g0 = layout[0][0][0]
nib = arena_view[g0 : g0 + I * H // 2]
assert torch.equal(nib, gate[0].reshape(-1)), "gate nibbles differ from source"
e8 = arena_view[g0 + I * H // 2 : g0 + I * H // 2 + I * (H // G)]
want_e8 = (gate_s[0].view(torch.int16).to(torch.int32) & 0xFFFF).reshape(-1) >> 7
assert torch.equal(e8.to(torch.int32), want_e8), "gate e8 scales differ"
d0 = layout[0][0][2]
assert torch.equal(arena_view[d0 : d0 + H * I // 2], down[0].reshape(-1)), "down nibbles differ"
print("arena bytes at layout offsets == source nibbles + e8(scales)")

def fwd(m, q):
    x = torch.randn(q, H, generator=torch.Generator().manual_seed(q)).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(E, generator=torch.Generator().manual_seed(q + i))[:K] for i in range(q)]).to(torch.int64)
    w = torch.rand(q, K, generator=torch.Generator().manual_seed(7 + q))
    out = torch.zeros(q, H, dtype=torch.bfloat16)
    bsz = torch.tensor([q], dtype=torch.int32)
    infer.submit(m.forward_task(bsz.data_ptr(), K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), out.data_ptr(), False))
    infer.sync()
    return out

for q in (1, 4, 64, 2048):
    a, b = fwd(m_ref, q), fwd(m_arena, q)
    assert torch.isfinite(a.float()).all() and a.abs().sum() > 0, "ref output degenerate"
    assert torch.equal(a, b), f"forward mismatch at qlen={q}: max |d|={(a.float()-b.float()).abs().max()}"
    print(f"qlen={q}: arena forward bit-identical to aligned_alloc path (|out| mean {a.float().abs().mean():.4f})")

# Non-arena instance must refuse weight_arena_layout().
try:
    m_ref.weight_arena_layout()
    raise SystemExit("expected weight_arena_layout() to throw without an arena")
except RuntimeError as ex:
    print("no-arena layout() raises:", str(ex)[:80])
print("SMOKE OK")

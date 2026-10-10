"""Window 2 diagnosis: time each copy class of kt_prefill._stage exactly as issued
(real dims H=5120, I=2304, It=1152, E=165, kt arena layout, memfd + cudaHostRegister
source, one TP rank's slice), per layer. dma2d.py only sized the 720-B down rows;
_stage also issues 36-B scale rows (s2) which that test never covered.
"""
import mmap
import os
import time

import torch
import cuda.bindings.runtime as rt

H, I, TP, E, G = 5120, 2304, 2, 165, 32
It = I // TP
r = int(os.environ.get("RANK", "0"))
dev = torch.device("cuda", 0)
stream = torch.cuda.Stream(device=dev)
st = stream.cuda_stream


def check(res):
    err = res[0] if isinstance(res, tuple) else res
    assert int(err) == 0, err


# kt arena layout for one layer: per expert [gate | up | down] each = nibbles then e8 scales
# scale regions are sized for FP32 per-32 scales; kt compacts them in place to
# uint8 E8M0 at the start of the region (patch 0003 scale_e8_valid), so copies read
# the first H*I/32 bytes of each region.
gate_nib, gate_sc = I * (H // 2), I * (H // G) * 4
down_nib, down_sc = H * (I // 2), H * (I // G) * 4
stride = 2 * (gate_nib + gate_sc) + down_nib + down_sc
gate0, up0, down0 = 0, gate_nib + gate_sc, 2 * (gate_nib + gate_sc)
nbytes = stride * E
assert stride == 22_118_400 and nbytes == 3_649_536_000, (stride, nbytes)

fd = os.memfd_create("arena_diag")
os.ftruncate(fd, nbytes)
mm = mmap.mmap(fd, nbytes, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
import ctypes

base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
# touch every page (first-touch like kt's load) with a pattern
mv = memoryview(mm)
blk = bytes(range(256)) * (1 << 16)
for off in range(0, nbytes, len(blk)):
    mv[off : off + min(len(blk), nbytes - off)] = blk[: min(len(blk), nbytes - off)]
t0 = time.perf_counter()
check(rt.cudaHostRegister(base, nbytes, rt.cudaHostRegisterPortable))
print(f"cudaHostRegister {nbytes/2**30:.2f} GiB in {time.perf_counter()-t0:.2f} s")

# device staging like _StreamedExpertSlots (this rank's TP slice)
w13 = torch.empty((E, 2 * It, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.empty((E, 2 * It, H // G), dtype=torch.uint8, device=dev)
w2 = torch.empty((E, H, It // 2), dtype=torch.uint8, device=dev)
s2 = torch.empty((E, H, It // G), dtype=torch.uint8, device=dev)
s2_full = torch.empty((E, H, I // G), dtype=torch.uint8, device=dev)  # alt: whole scales, slice on device
w2_full = torch.empty((E, H, I // 2), dtype=torch.uint8, device=dev)  # alt: whole down, slice on device


def copy2d(dst, dpitch, src, spitch, width, height):
    check(rt.cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, st))


def copy1d(dst, src, n):
    check(rt.cudaMemcpyAsync(dst, src, n, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, st))


nib_rows, e8_rows = H // 2, H // G
w13_pitch = w13[1].data_ptr() - w13[0].data_ptr()
s13_pitch = s13[1].data_ptr() - s13[0].data_ptr()


def stage_w13():
    for off, row0 in ((up0, 0), (gate0, It)):
        copy2d(w13.data_ptr() + row0 * nib_rows, w13_pitch, base + off + r * It * nib_rows, stride, It * nib_rows, E)


def stage_s13():
    for off, row0 in ((up0, 0), (gate0, It)):
        copy2d(s13.data_ptr() + row0 * e8_rows, s13_pitch, base + off + I * nib_rows + r * It * e8_rows, stride, It * e8_rows, E)


w2_pitch, s2_pitch = It // 2, It // G


def stage_w2():
    for e in range(E):
        blk = base + down0 + e * stride
        copy2d(w2[e].data_ptr(), w2_pitch, blk + r * w2_pitch, I // 2, w2_pitch, H)


def stage_s2():
    for e in range(E):
        blk = base + down0 + e * stride
        copy2d(s2[e].data_ptr(), s2_pitch, blk + H * (I // 2) + r * s2_pitch, I // G, s2_pitch, H)


def stage_s2_full_then_slice():
    # one 2D call over all experts: row = one expert's whole down scales (contiguous H*I/G bytes)
    copy2d(s2_full.data_ptr(), H * (I // G), base + down0 + H * (I // 2), stride, H * (I // G), E)
    with torch.cuda.stream(stream):
        s2.copy_(s2_full[:, :, r * s2_pitch : (r + 1) * s2_pitch])


def stage_w2_full_then_slice():
    copy2d(w2_full.data_ptr(), H * (I // 2), base + down0, stride, H * (I // 2), E)
    with torch.cuda.stream(stream):
        w2.copy_(w2_full[:, :, r * w2_pitch : (r + 1) * w2_pitch])


GRP = int(os.environ.get("GRP", "16"))
w2_grp = torch.empty((GRP, H, I // 2), dtype=torch.uint8, device=dev)  # alt: grouped whole-down scratch


def stage_w2_grouped():
    # whole down blocks of GRP consecutive experts per call into a small scratch, slice on device
    for g0 in range(0, E, GRP):
        g = min(GRP, E - g0)
        copy2d(w2_grp.data_ptr(), H * (I // 2), base + down0 + g0 * stride, stride, H * (I // 2), g)
        with torch.cuda.stream(stream):
            w2[g0 : g0 + g].copy_(w2_grp[:g, :, r * w2_pitch : (r + 1) * w2_pitch])


def timeit(fn, nbytes_moved, label, reps=5):
    with torch.cuda.stream(stream):
        fn()
    stream.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            fn()
        stream.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"{label:34s} {nbytes_moved/2**20:8.1f} MiB  median {med*1e3:8.2f} ms  {nbytes_moved/med/1e9:6.1f} GB/s  (min {ts[0]*1e3:.2f})")
    return med


t_w13 = timeit(stage_w13, w13.numel(), "w13 nibbles (2 x 2D, 1 call each)")
t_s13 = timeit(stage_s13, s13.numel(), "s13 scales  (2 x 2D, 1 call each)")
t_w2 = timeit(stage_w2, w2.numel(), f"w2 nibbles  ({E} x 2D, {w2_pitch}-B rows)")
t_s2 = timeit(stage_s2, s2.numel(), f"s2 scales   ({E} x 2D, {s2_pitch}-B rows)")
tot = t_w13 + t_s13 + t_w2 + t_s2
print(f"-- _stage as issued: {tot*1e3:.1f} ms/layer -> {tot*40:.2f} s per 8192-token chunk per rank")
a_s2 = timeit(stage_s2_full_then_slice, s2_full.numel(), "ALT s2: whole scales 1 call + slice")
a_w2 = timeit(stage_w2_full_then_slice, w2_full.numel(), "ALT w2: whole down 1 call + slice")
a_w2g = timeit(stage_w2_grouped, w2_full.numel(), f"ALT w2: grouped x{GRP} whole down + slice")
for lbl, tw2 in (("as-issued w2", t_w2), ("whole w2", a_w2), (f"grouped x{GRP} w2", a_w2g)):
    t = t_w13 + t_s13 + a_s2 + tw2
    print(f"-- ALT stage (w13 + s13 + ALT s2 + {lbl}): {t*1e3:.1f} ms/layer -> {t*40:.2f} s per chunk")
# correctness of the alternates vs as-issued
ref_w2 = w2.clone()
ref_s2 = s2.clone()
with torch.cuda.stream(stream):
    stage_w2()
    stage_s2()
stream.synchronize()
w2_ref, s2_ref = w2.clone(), s2.clone()
with torch.cuda.stream(stream):
    stage_w2_full_then_slice()
    stage_s2_full_then_slice()
stream.synchronize()
print("ALT bit-identical to as-issued: w2", torch.equal(w2, w2_ref), "s2", torch.equal(s2, s2_ref))
w2.zero_()
with torch.cuda.stream(stream):
    stage_w2_grouped()
stream.synchronize()
print("ALT grouped w2 bit-identical to as-issued:", torch.equal(w2, w2_ref))
check(rt.cudaHostUnregister(base))

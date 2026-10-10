"""Window 2 sizing: can the DMA engine fetch kt's down-projection column slice
(720-byte rows, pitch 1440, 4096 rows per expert, 165 experts) straight from
pinned host memory at a useful rate, vs. the contiguous copies we do today?
"""
import time
import torch
import cuda.bindings.runtime as rt

H, I, N, TP = 4096, 2880, 165, 2
width = I // 2 // TP          # 720 B per row for this rank
pitch = I // 2                # 1440 B source pitch
dev = torch.device("cuda", 0)

host_down = torch.empty((N, H, pitch), dtype=torch.uint8, pin_memory=True)
host_down.random_(0, 255)
host_w13 = torch.empty((N, 2 * I // TP, H // 2), dtype=torch.uint8, pin_memory=True)
dst_down = torch.empty((N, H, width), dtype=torch.uint8, device=dev)
dst_w13 = torch.empty_like(host_w13, device=dev)
s = torch.cuda.Stream()

def check(r):
    err = r[0] if isinstance(r, tuple) else r
    assert int(err) == 0, err

def run_2d(rank):
    for e in range(N):
        check(rt.cudaMemcpy2DAsync(
            dst_down[e].data_ptr(), width,
            host_down[e].data_ptr() + rank * width, pitch,
            width, H, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, s.cuda_stream))

def run_2d_one(rank):
    # one 2D copy for all experts: rows = N*H, same pitch
    check(rt.cudaMemcpy2DAsync(
        dst_down.data_ptr(), width,
        host_down.data_ptr() + rank * width, pitch,
        width, N * H, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, s.cuda_stream))

def run_contig():
    dst_w13.copy_(host_w13, non_blocking=True)

def timeit(fn, nbytes, label, reps=5):
    with torch.cuda.stream(s):
        fn()
    s.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        with torch.cuda.stream(s):
            fn()
        s.synchronize()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    med = ts[len(ts) // 2]
    print(f"{label:28s} {nbytes/2**20:8.1f} MiB  median {med*1e3:7.2f} ms  {nbytes/med/1e9:6.1f} GB/s  (min {ts[0]*1e3:.2f})")

timeit(run_contig, host_w13.numel(), "contig w13 slice (today)")
timeit(lambda: run_2d(0), N * H * width, "2D down slice, per expert")
timeit(lambda: run_2d(1), N * H * width, "2D down slice rank1, per e")
timeit(lambda: run_2d_one(0), N * H * width, "2D down slice, one call")
# correctness of the 2D path
ref = host_down[:, :, width:2 * width].contiguous().to(dev)
with torch.cuda.stream(s):
    run_2d_one(1)
s.synchronize()
print("2D one-call rank1 bit-identical:", torch.equal(ref, dst_down))

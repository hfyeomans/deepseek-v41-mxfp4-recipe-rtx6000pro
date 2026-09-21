import torch, time, statistics
n = 8 * 2**30 // 4
out = []
for th in (8, 16, 32):
    torch.set_num_threads(th)
    x = torch.ones(n, dtype=torch.float32)
    def bench(fn, byts):
        ts = []
        for _ in range(4):
            t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
        return byts / statistics.median(ts[1:]) / 1e9
    r1 = bench(lambda: x.sum(), 8*2**30)
    y = x.view(torch.int64)
    r2 = bench(lambda: y.sum(), 8*2**30)
    z = torch.empty_like(x)
    r3 = bench(lambda: z.copy_(x), 2*8*2**30)
    line = f"threads={th} f32_sum={r1:.1f} i64_sum={r2:.1f} f32_copy_rw={r3:.1f} GB/s"
    print(line, flush=True); out.append({"threads": th, "f32_sum_gbps": round(r1,1), "i64_sum_gbps": round(r2,1), "f32_copy_rw_gbps": round(r3,1)})
import json; json.dump({"experiment":"E1b-torch-dram-read","size_gib":8,"note":"read-only sum vs copy(read+write counted); bf16 sum in bench-cpu-expert.py is conversion-bound and not a DRAM probe","results":out}, open("/w/results/e1b.json","w"), indent=1)

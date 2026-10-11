import sys, torch
sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged_v2, plan_topk_v2
torch.manual_seed(0)
dev = torch.device("cuda:0")
TOPK = 512
for L in [100, 511, 512, 513, 600, 1000, 2048, 4096, 20000, 65536, 262144]:
    score = torch.randn(1, (max(L, TOPK)+3)//4*4, device=dev, dtype=torch.float32)
    lengths = torch.tensor([L], device=dev, dtype=torch.int32)
    meta = plan_topk_v2(lengths)
    outs = []
    for i in range(30):
        out = torch.full((1, TOPK), -7, device=dev, dtype=torch.int32)
        topk_transform_paged_v2(score, lengths, None, out, 64, meta)
        torch.cuda.synchronize()
        outs.append(out.clone())
    first = outs[0]
    order_diff = sum(1 for o in outs[1:] if not torch.equal(o, first))
    set_diff = sum(1 for o in outs[1:] if not torch.equal(torch.sort(o)[0], torch.sort(first)[0]))
    valid = first[first >= 0]
    print(f"L={L:6d}: order differs {order_diff}/29, set differs {set_diff}/29, sorted={bool(torch.equal(valid, torch.sort(valid)[0]))}, n_valid={valid.numel()}")

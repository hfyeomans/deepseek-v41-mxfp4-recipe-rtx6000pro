import torch, sys
from sgl_kernel import fast_topk_v2, fast_topk_transform_fused
torch.manual_seed(0)
dev = torch.device("cuda:0")
res = {}
for L in [37, 300, 1500, 2047, 2048, 2049, 4096, 20000]:
    for topk in [2048]:
        score = torch.randn(1, max(L, 2048), device=dev, dtype=torch.float32)
        lengths = torch.tensor([L], device=dev, dtype=torch.int32)
        outs = []
        for i in range(30):
            o = fast_topk_v2(score, lengths, topk)
            torch.cuda.synchronize()
            outs.append(o.clone())
        first = outs[0]
        order_diff = sum(1 for o in outs[1:] if not torch.equal(o, first))
        set_diff = sum(1 for o in outs[1:] if not torch.equal(torch.sort(o)[0], torch.sort(first)[0]))
        sorted_out = bool(torch.equal(first, torch.sort(first)[0]))
        print(f"L={L:6d} topk={topk}: order differs {order_diff}/29, set differs {set_diff}/29, first sorted={sorted_out}, shape={tuple(first.shape)}, n_valid={(first>=0).sum().item()}")

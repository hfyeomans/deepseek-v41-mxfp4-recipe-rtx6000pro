import sys, torch
sys.path.insert(0, "/sgl-workspace/sglang/python")
from sglang.kernels.ops.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120

torch.manual_seed(0)
dev = torch.device("cuda:0")
PBS, BPT, PAGE_BYTES = 64, 584, 37440
NOPE, ROPE, SCALE_OFF = 448, 64, 64 * 576
H, D, DV, TOPK = 64, 512, 512, 512

def make_kv(num_pages):
    raw = torch.zeros(num_pages, PAGE_BYTES, dtype=torch.uint8, device=dev)
    # data: 64 tokens x 576 bytes (448 fp8 nope + 64 bf16 rope)
    nope = (torch.randn(num_pages, PBS, NOPE, device=dev) * 0.5).to(torch.float8_e4m3fn).view(torch.uint8)
    rope = (torch.randn(num_pages, PBS, ROPE, device=dev) * 0.5).to(torch.bfloat16).view(torch.uint8)
    data = torch.cat([nope, rope], dim=-1).reshape(num_pages, PBS * 576)
    raw[:, :PBS * 576] = data
    # footer: 64 tokens x 8 scale bytes (7 ue8m0-ish or fp8 scales + pad); use a mild constant
    scales = torch.full((num_pages, PBS, 8), 0x38, dtype=torch.uint8, device=dev)  # fp8 e4m3 0x38 = 1.0
    scales[..., 7] = 0
    raw[:, SCALE_OFF:SCALE_OFF + PBS * 8] = scales.reshape(num_pages, PBS * 8)
    return raw.as_strided((num_pages, PBS, 1, BPT), (PAGE_BYTES, BPT, BPT, 1))

def run(q, kv, idx, topk_len, sink):
    o = flash_mla_with_kvcache_sm120(q=q, k_cache=kv, head_dim_v=DV, softmax_scale=D ** -0.5,
                                     indices=idx, topk_length=topk_len, attn_sink=sink,
                                     extra_k_cache=None, extra_indices_in_kvcache=None, extra_topk_length=None)[0]
    torch.cuda.synchronize()
    return o.clone()

for L in [100, 300, 512, 600, 2048, 20000, 65536]:
    num_pages = (L + PBS - 1) // PBS + 2
    kv = make_kv(num_pages)
    q = (torch.randn(1, 1, H, D, device=dev) * 0.5).to(torch.bfloat16)
    sink = torch.zeros(H, device=dev, dtype=torch.float32)
    n_sel = min(L, TOPK)
    sel = torch.randperm(L, device=dev)[:n_sel].to(torch.int32)
    idx = torch.full((1, TOPK), -1, dtype=torch.int32, device=dev)
    idx[0, :n_sel] = torch.sort(sel)[0]
    topk_len = torch.tensor([n_sel], dtype=torch.int32, device=dev)
    outs = [run(q, kv, idx, topk_len, sink) for _ in range(20)]
    first = outs[0]
    rep = sum(1 for o in outs[1:] if not torch.equal(o, first))
    # same set, shuffled order
    idx2 = idx.clone(); idx2[0, :n_sel] = sel[torch.randperm(n_sel, device=dev)]
    o2 = run(q, kv, idx2, topk_len, sink)
    d = (o2.float() - first.float()).abs()
    nz = int((o2 != first).sum())
    print(f"L={L:6d} n_sel={n_sel}: repeats differ {rep}/19; shuffled order: {nz} elems differ, max|d| {float(d.max()):.3g}, rel {float(d.max()/first.float().abs().max()):.3g}, finite={bool(torch.isfinite(first.float()).all())}, |o|max {float(first.float().abs().max()):.3g}")

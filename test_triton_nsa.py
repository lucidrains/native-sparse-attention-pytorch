import torch
from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend

import einx
from einops import rearrange, einsum

assert torch.cuda.is_available()

def exists(v):
    return v is not None

def regular_attend(
    q, k, v,
    indices,
    mask,
    block_size = None,
):
    if exists(block_size):
        w = q.shape[-2] // block_size
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> b (h w) n d', n = block_size) for t in (q, k, v))

    seq_len, device = q.shape[-2], q.device
    scale = q.shape[-1] ** -0.5

    sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * scale
    causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)
    sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
    attn = sim.softmax(dim = -1)

    out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

    if exists(block_size):
        out = rearrange(out, 'b (h w) n d -> b h (w n) d', w = w)

    return out

# mock inputs

fine_block_size = 64

q = torch.randn(1, 2, 512, 64).cuda()
k = torch.randn(1, 2, 512, 64).cuda()
v = torch.randn(1, 2, 512, 64).cuda()

indices = torch.zeros(1, 2, 512, 1).long().cuda()
mask = torch.zeros(1, 2, 512, 1).bool().cuda()

# both regular and nsa pathways `r` and `n`

rq, rk, rv = tuple(t.clone().requires_grad_() for t in (q, k, v))
nq, nk, nv = tuple(t.clone().requires_grad_() for t in (q, k, v))

# regular forwards and backwards

out = regular_attend(rq, rk, rv, indices, mask, block_size = fine_block_size)
out.sum().backward()

# triton nsa forwards and backwards

nsa_out = native_sparse_attend(nq, nk, nv, fine_block_size, indices, mask, 1)
nsa_out.sum().backward()

# asserts

assert torch.allclose(out, nsa_out, atol = 1e-2)

assert torch.allclose(nq.grad, rq.grad, atol = 1e-2)
assert torch.allclose(nk.grad, rk.grad, atol = 1e-2)
assert torch.allclose(nv.grad, rv.grad, atol = 1e-2)

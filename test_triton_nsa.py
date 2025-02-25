import torch
from native_sparse_attention_pytorch.triton_native_sparse_attention import native_sparse_attend

import einx
from einops import rearrange, einsum, repeat

assert torch.cuda.is_available()

def exists(v):
    return v is not None

def regular_attend(
    q, k, v,
    indices,
    mask,
    block_size = None,
):
    q_heads, kv_heads = q.shape[1], k.shape[1]

    if exists(block_size):
        w = q.shape[-2] // block_size
        q, k, v = tuple(rearrange(t, 'b h (w n) d -> b (h w) n d', n = block_size) for t in (q, k, v))

    seq_len, device = q.shape[-2], q.device
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # block causal diagonal

    sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
    causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).triu(1)
    sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

    # rest of the indices

    num_sel_kv_blocks = indices.shape[-1]
    has_sel_kv_blocks = num_sel_kv_blocks > 0

    if has_sel_kv_blocks:
        bk, bv = tuple(rearrange(t, 'b (h w) n d -> b h w n d', h = kv_heads) for t in (k, v))
        sel_bk = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bk, indices)
        sel_bv = einx.get_at('b h [w] n d, b h i sel -> b h i (sel n) d', bv, indices)

        q = rearrange(q, 'b (h w) n d -> b h (w n) d', h = q_heads)
        bsim = einsum(q, sel_bk, 'b h i d, b h i j d -> b h i j')

        bsim = rearrange(bsim, 'b h (w i) (sel j) -> b h w i sel j', sel = num_sel_kv_blocks, i = fine_block_size)

        mask = rearrange(mask, 'b h (w i) sel -> b h w i sel', i = fine_block_size)
        bsim = torch.where(mask[..., None], bsim, -torch.finfo(bsim.dtype).max)

        sim = rearrange(sim, 'b (h w) i j -> b h w i 1 j', h = q_heads)

        sim = torch.cat((sim, bsim), dim = -2)
        sim = rearrange(sim, 'b h w i causal_and_sel j -> b h (w i) (causal_and_sel j)')

        sel_bv = rearrange(sel_bv, 'b h (w i) j d -> b h w i j d', i = fine_block_size)

        v = repeat(v, 'b (h w) j d -> b h w i j d', h = kv_heads, i = fine_block_size)
        v = torch.cat((v, sel_bv), dim = -2)
        v = rearrange(v, 'b h w i j d -> b h (w i) j d')

    # attend

    attn = sim.softmax(dim = -1)

    if has_sel_kv_blocks:
        out = einsum(attn, v, 'b h i j, b h i j d -> b h i d')
    else:
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        if exists(block_size):
            out = rearrange(out, 'b (h w) n d -> b h (w n) d', w = w)

    return out

# mock inputs

fine_block_size = 16

q = torch.randn(1, 2, 512, 64).cuda()
k = torch.randn(1, 2, 512, 64).cuda()
v = torch.randn(1, 2, 512, 64).cuda()

indices = torch.zeros(1, 2, 512, 1).long().cuda()
mask = torch.ones(1, 2, 512, 1).bool().cuda()

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

assert torch.allclose(nv.grad, rv.grad, atol = 1e-2)
assert torch.allclose(nk.grad, rk.grad, atol = 1e-2)
assert torch.allclose(nq.grad, rq.grad, atol = 1e-2)

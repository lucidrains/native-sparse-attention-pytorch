from __future__ import annotations

from math import ceil

import torch
from torch import nn, arange, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from local_attention import LocalAttention

# einstein notation

import einx
from einops import einsum, repeat, rearrange
from einops.layers.torch import Rearrange

# b - batch
# h - heads
# n - sequence (token level or compressed)
# w - windows, for fine or compressed
# i, j - query / key sequence
# d - feature dimension
# s - strategies

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# flex attn sliding attention mask

def create_sliding_mask(seq_len, window_size):
    def sliding_mask(_, __, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx

        sliding_mask = (q_idx - kv_idx) <= window_size
        causal_mask = causal_mask & sliding_mask

        return causal_mask

    block_mask = create_block_mask(sliding_mask, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# classes

class SparseAttention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        selection_block_size,
        num_selected_blocks,
        num_compressed_mem_kv = 4,
        norm = True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        assert compress_block_size == selection_block_size, 'start off with compressed being equal to selection block sizes'

        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # qkv

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        # sliding window strategy

        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = True,
            exact_windowsize = True,
            autopad = True
        )

        # compress strategy

        self.compress_block_size = compress_block_size

        assert num_compressed_mem_kv > 0

        self.compress_mem_kv = nn.Parameter(torch.zeros(2, heads, num_compressed_mem_kv, dim_head))
        self.k_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(heads, compress_block_size, dim_head))

        self.k_compress = nn.Sequential(
            Rearrange('b h n d -> b (h d) n'),
            nn.Conv1d(dim_head * heads, dim_head * heads, compress_block_size, stride = compress_block_size, groups = heads),
            Rearrange('b (h d) nc -> b h nc d', h = heads)
        )

        self.v_compress = nn.Sequential(
            Rearrange('b h n d -> b (h d) n'),
            nn.Conv1d(dim_head * heads, dim_head * heads, compress_block_size, stride = compress_block_size, groups = heads),
            Rearrange('b (h d) nc -> b h nc d', h = heads)
        )

        # selection related

        self.selection_block_size = selection_block_size
        self.num_selected_blocks = num_selected_blocks

        # they combine the three sparse branches through a learned combine with sigmoid activation

        self.to_strategy_combine = nn.Sequential(
            nn.Linear(dim, 3 * heads),
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads

        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        inp
    ):
        batch, seq_len, scale, device = *inp.shape[:2], self.scale, inp.device

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm

        inp = self.norm(inp)

        # queries, keys, values

        q, k, v = self.to_qkv(inp).chunk(3, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # compressed key / values

        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)

        ck = self.k_compress(k[..., :compress_divisible_seq_len, :] + k_pos)
        cv = self.v_compress(v[..., :compress_divisible_seq_len, :] + v_pos)

        # 1. coarse attention over compressed

        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)

        num_mem_compress_kv = mem_ck.shape[-2]

        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        csim = einsum(q, ck, 'b h i d, b h j d -> b h i j') * self.scale

        cq_seq = arange(seq_len, device = device)

        ck_seq = ((arange(num_compress_blocks) + 1) * self.compress_block_size) - 1
        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value = -1)

        cmask = einx.less('j, i -> i j', ck_seq, cq_seq)

        mask_value = -torch.finfo(csim.dtype).max

        csim = csim.masked_fill(cmask, mask_value)

        cattn = csim.softmax(dim = -1)

        compressed_attn_out = einsum(cattn, cv, 'b h i j, b h j d -> b h i d')

        # 2. fine attention over selected based on compressed attention logits

        importance_scores = csim[..., num_mem_compress_kv:]

        selected_importance_values, selected_block_indices = importance_scores.topk(self.num_selected_blocks, dim = -1)

        fmask = selected_importance_values > mask_value

        fq = q
        fk = k
        fv = v

        if seq_len < fine_divisible_seq_len:
            remainder = fine_divisible_seq_len - seq_len
            fk = pad_at_dim(fk, (0, remainder), value = 0., dim = -2)
            fv = pad_at_dim(fv, (0, remainder), value = 0., dim = -2)
            fq = pad_at_dim(fq, (0, remainder), value = 0., dim = -2)

            fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

            selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

        fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
        fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
        fmask = repeat(fmask, 'b h i w -> b h i (w j)', j = self.selection_block_size)

        fk = einx.get_at('b h [w] j d, b h i selected -> b h i (selected j) d', fk, selected_block_indices)
        fv = einx.get_at('b h [w] j d, b h i selected -> b h i (selected j) d', fv, selected_block_indices)

        fsim = einsum(fq, fk, 'b h i d, b h i j d -> b h i j') * self.scale

        fsim = fsim.masked_fill(fmask, mask_value)

        fattn = fsim.softmax(dim = -1)

        fine_out = einsum(fattn, fv, 'b h i j, b h i j d -> b h i d')

        fine_out = fine_out[..., :seq_len, :]

        # 3. overlapping sliding window, this is unsurprising and expected

        local_attn_out = self.sliding_window(q, k, v)

        # combine strategies

        strategy_weighted_combine = self.to_strategy_combine(inp)

        out = einsum(strategy_weighted_combine, stack([compressed_attn_out, fine_out, local_attn_out]), 'b h n s, s b h n d -> b h n d')

        # merge heads and combine them

        out = self.merge_heads(out)

        return self.combine_heads(out)

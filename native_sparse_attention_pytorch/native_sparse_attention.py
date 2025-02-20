from __future__ import annotations

from copy import deepcopy
from math import ceil

import torch
from torch import nn, arange, stack, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from local_attention import LocalAttention

from rotary_embedding_torch import RotaryEmbedding

# einstein notation

import einx
from einops import einsum, repeat, rearrange, reduce
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

def divisible_by(num, den):
    return (num % den) == 0

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
        kv_heads = None,
        num_compressed_mem_kv = 4,
        norm = True,
        use_diff_topk = False,
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        strategy_combine_mlp: Module | None = None
    ):
        super().__init__()

        # attention heads
        # handling gqa if `kv_heads` is set

        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)

        self.heads = heads
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        # scale

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # rotary

        self.rotary_emb = RotaryEmbedding(dim_head)

        # qkv

        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)

        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)

        self.qkv_split = qkv_split

        # sliding window strategy

        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = True,
            exact_windowsize = True,
            autopad = True,
            use_rotary_pos_emb = False
        )

        self.sliding_window_size = sliding_window_size

        # compress strategy

        self.compress_block_size = compress_block_size

        assert num_compressed_mem_kv > 0

        self.split_compress_window = Rearrange('b h (w n) d -> b h w n d', n = compress_block_size)

        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, compress_block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # selection related

        self.use_diff_topk = use_diff_topk

        self.selection_block_size = selection_block_size

        assert num_selected_blocks > 0
        self.num_selected_blocks = num_selected_blocks

        # they combine the three sparse branches through a learned combine with sigmoid activation

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 3 * heads)

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads

        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        inp,
        sliding_window_flex_mask = None
    ):
        batch, seq_len, scale, heads, device = *inp.shape[:2], self.scale, self.heads, inp.device

        compress_divisible_seq_len = round_down_mult(seq_len, self.compress_block_size)
        num_compress_blocks = compress_divisible_seq_len // self.compress_block_size

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # maybe prenorm

        inp = self.norm(inp)

        # queries, keys, values

        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        # compressed key / values - variables prepended with `c` stands for compressed

        k_pos = repeat(self.k_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)
        v_pos = repeat(self.v_intrablock_positions, 'h n d -> h (r n) d', r = num_compress_blocks)

        k_compress_input = self.split_compress_window(k[..., :compress_divisible_seq_len, :] + k_pos)
        v_compress_input = self.split_compress_window(v[..., :compress_divisible_seq_len, :] + v_pos)

        ck = self.k_compress(k_compress_input)
        cv = self.v_compress(v_compress_input)

        # 1. coarse attention over compressed

        mem_ck, mem_cv = repeat(self.compress_mem_kv, 'kv ... -> kv b ...', b = batch)

        num_mem_compress_kv = mem_ck.shape[-2]

        ck = cat((mem_ck, ck), dim = -2)
        cv = cat((mem_cv, cv), dim = -2)

        ck, cv = tuple(repeat(t, 'b h ... -> b (num_grouped_queries h) ...', num_grouped_queries = self.num_grouped_queries) for t in (ck, cv))

        csim = einsum(q, ck, 'b h i d, b h j d -> b h i j') * self.scale

        cq_seq = arange(seq_len, device = device)

        ck_seq = ((arange(num_compress_blocks, device = device) + 1) * self.compress_block_size) - 1
        ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value = -1)

        cmask = einx.less_equal('j, i -> i j', ck_seq, cq_seq)

        mask_value = -torch.finfo(csim.dtype).max

        csim = csim.masked_fill(~cmask, mask_value)

        cattn = csim.softmax(dim = -1)

        compressed_attn_out = einsum(cattn, cv, 'b h i j, b h j d -> b h i d')

        # for 2. and 3., will give them relative positions with rotary - compressed needs to be handled separately (even if they already have intra block absolute positions)

        rotated_q, rotated_k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        # 2. fine attention over selected based on compressed attention logits - variables prepended with `f` stands for the fine attention pathway

        importance_scores = cattn[..., num_mem_compress_kv:]

        # for gqa, we will average the compressed attention across each grouped queries (per key / values)

        importance_scores = reduce(importance_scores, 'b (grouped_queries h) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)

        # handle if compress block size not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length, then average within each fine / selection block size - pad on the right to 0s, which should be fine as sliding window convers the local anyways

        if self.compress_block_size != self.selection_block_size:
            importance_scores = repeat(importance_scores, '... j -> ... (j block_size)', block_size = self.compress_block_size)
            padding = fine_divisible_seq_len - importance_scores.shape[-1]

            importance_scores = F.pad(importance_scores, (0, padding))
            importance_scores = reduce(importance_scores, '... (j block_size) -> ... j', 'mean', block_size = self.selection_block_size)

        # handle if number of total blocks is less than number to select for fine attention

        num_selected = min(self.num_selected_blocks, importance_scores.shape[-1])

        fq = rotated_q
        fk = rotated_k
        fv = v

        if num_selected > 0:
            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)

            if self.use_diff_topk:
                gates = selected_importance_values + (1. - selected_importance_values).detach()

            fmask = selected_importance_values > 1e-10

            if seq_len < fine_divisible_seq_len:
                remainder = fine_divisible_seq_len - seq_len
                fk = pad_at_dim(fk, (0, remainder), value = 0., dim = -2)
                fv = pad_at_dim(fv, (0, remainder), value = 0., dim = -2)
                fq = pad_at_dim(fq, (0, remainder), value = 0., dim = -2)

                fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

                selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

                if self.use_diff_topk:
                    gates = pad_at_dim(gates, (0, remainder), value = 1., dim = -2)

            # handle block causal diagonal in the diagram, but run experiments without to see

            fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
            fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = batch, h = self.kv_heads)
            selected_block_indices = cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

            fmask = repeat(fmask, 'b h i w -> b h i w j', j = self.selection_block_size)

            causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
            causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = batch, h = self.kv_heads)

            fmask = cat((fmask, causal_mask), dim = -2)
            fmask = rearrange(fmask, 'b h i w j -> b h i (w j)')

            # select out the spatial crops of keys / values for fine attention

            fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
            fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

            # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)

            fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
            fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])

            selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])

            fk = fk.gather(3, selected_block_indices)
            fv = fv.gather(3, selected_block_indices)

            # handle maybe gating

            if self.use_diff_topk:
                gates = F.pad(gates, (0, 1), value = 1.)

                fk = einx.multiply('b h i w, b h i w j d -> b h i w j d', gates, fk)
                fv = einx.multiply('b h i w, b h i w j d -> b h i w j d', gates, fv)

            fk = rearrange(fk, 'b h i w j d -> b h i (w j) d')
            fv = rearrange(fv, 'b h i w j d -> b h i (w j) d')

            # fine attention

            fk, fv, fmask = tuple(repeat(t, 'b h ... -> b (num_grouped_queries h) ...', num_grouped_queries = self.num_grouped_queries) for t in (fk, fv, fmask))

            fsim = einsum(fq, fk, 'b h i d, b h i j d -> b h i j') * self.scale

            fsim = fsim.masked_fill(~fmask, mask_value)

            fattn = fsim.softmax(dim = -1)

            fine_attn_out = einsum(fattn, fv, 'b h i j, b h i j d -> b h i d')

            fine_attn_out = fine_attn_out[..., :seq_len, :]
        else:
            # if only first block, just do a simple block causal

            seq_len = fk.shape[-2]
            fmask = causal_mask = torch.ones((seq_len, seq_len), device = device, dtype = torch.bool).tril()

            fk, fv = tuple(repeat(t, 'b h ... -> b (num_grouped_queries h) ...', num_grouped_queries = self.num_grouped_queries) for t in (fk, fv))

            fsim = einsum(fq, fk, 'b h i d, b h j d -> b h i j') * self.scale

            fsim = fsim.masked_fill(~fmask, mask_value)

            fattn = fsim.softmax(dim = -1)

            fine_attn_out = einsum(fattn, fv, 'b h i j, b h j d -> b h i d')

        # 3. overlapping sliding window, this is unsurprising and expected - `s` for sliding

        sq = rotated_q
        sk = rotated_k
        sv = v

        sk, sv = tuple(repeat(t, 'b h ... -> b (num_grouped_queries h) ...', num_grouped_queries = self.num_grouped_queries) for t in (sk, sv))

        if exists(sliding_window_flex_mask):
            sliding_window_attn_out = flex_attention(sq, sk, sv, block_mask = sliding_window_flex_mask)
        else:
            sliding_window_attn_out = self.sliding_window(sq, sk, sv)

        # combine strategies

        strategy_weighted_combine = self.to_strategy_combine(inp)

        out = einsum(strategy_weighted_combine, stack([compressed_attn_out, fine_attn_out, sliding_window_attn_out]), 'b h n s, s b h n d -> b h n d')

        # merge heads and combine them

        out = self.merge_heads(out)

        return self.combine_heads(out)

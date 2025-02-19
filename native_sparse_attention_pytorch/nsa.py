import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import einsum
from einops.layers.torch import Rearrange

from local_attention import LocalAttention

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        sliding_window_size,
        compress_block_size,
        norm = True,
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # qkv

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        # sliding window strategy

        self.sliding_window = LocalAttention(
            dim = dim_head,
            window_size = sliding_window_size,
            causal = True,
            exact_window_size = True
        )

        # compress strategy

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

        # they combine the three sparse branches through a learned combine with sigmoid activation

        self.to_strategy_combine = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Sigmoid()
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

        inp = self.norm(inp)

        q, k, v = self.to_qkv(inp).chunk(3, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        out = self.sliding_window(q, k, v)

        out = self.merge_heads(out)

        return self.combine_heads(out)

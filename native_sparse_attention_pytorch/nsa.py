import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops.layers.torch import Rearrange

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
        norm = True
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.norm = nn.RMSNorm(dim) if norm else nn.Identity()

        # qkv

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

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
        return inp

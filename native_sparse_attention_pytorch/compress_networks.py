import torch
from torch import nn
from torch.nn import Module, ModuleList

from einops import einsum, rearrange
from einops.layers.torch import EinMix as Mix

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# start accumulating some types of compression networks

class ConvLinearCompress(Module):
    """
    used successfully in an old google brain paper, https://github.com/lucidrains/memory-efficient-attention-pytorch
    grouped convolutions so each head get its own parameters
    """

    def __init__(
        self,
        heads,
        dim_head,
        compress_block_size
    ):
        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(heads * dim_head, heads * dim_head, compress_block_size, stride = compress_block_size, groups = heads)

    def forward(
        self,
        kv # Float['b h w n d']
    ):

        kv = rearrange(kv, 'b h w n d -> b (h d) (w n)')

        compressed = self.conv(kv)

        return rearrange(compressed, 'b (h d) n -> b h n d', h = self.heads)

# attention pool used by enformer, deepmind's genetic attention network

class AttentionPool(Module):
    def __init__(
        self,
        dim_head,
        compress_block_size
    ):
        super().__init__()
        self.to_attn_logits = nn.Linear(dim_head, dim_head, bias = False)
        self.to_attn_logits.weight.data.copy_(torch.eye(dim_head))

    def forward(
        self,
        kv
    ):

        attn_logits = self.to_attn_logits(kv)

        attn = attn_logits.softmax(dim = -2)

        compressed = einsum(kv, attn, 'b h w n d, b h w n d -> b h w d')

        return compressed

# mlp per head

class GroupedMLP(Module):
    def __init__(
        self,
        dim_head,
        compress_block_size,
        heads,
        expand_factor = 1.,
    ):
        super().__init__()

        dim = dim_head * compress_block_size
        dim_hidden = int(dim * expand_factor)
        dim_out = dim_head

        self.net = nn.Sequential(
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim, o = dim_hidden),
            nn.ReLU(),
            Mix('b h w i -> b h w o', weight_shape = 'h i o', bias_shape = 'h o', h = heads, i = dim_hidden, o = dim_out),
        )

    def forward(
        self,
        kv
    ):
        kv = rearrange(kv, 'b h w n d -> b h w (n d)')

        compressed = self.net(kv)

        return compressed

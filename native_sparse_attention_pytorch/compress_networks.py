import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import einsum, rearrange
from einops.layers.torch import EinMix as Mix, Rearrange

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
        compress_window_size
    ):
        super().__init__()
        self.heads = heads
        self.conv = nn.Conv1d(heads * dim_head, heads * dim_head, compress_window_size, stride = compress_window_size, groups = heads)

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
        compress_window_size
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
        compress_window_size,
        heads,
        expand_factor = 1.,
    ):
        super().__init__()

        dim = dim_head * compress_window_size
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

# single projection "mlp"

class SingleProjection(Module):
    def __init__(
        self,
        dim_head,
        compress_window_size,
        heads = 1
    ):
        super().__init__()
        dim = dim_head * compress_window_size
        dim_out = dim_head

        is_grouped = heads > 1

        if not is_grouped:
            self.compress = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(dim, dim_out, bias = False)
            )
        else:
            self.compress = Mix(
                'b h w n i -> b h w o',
                weight_shape = 'h i o',
                h = heads,
                i = dim_head,
                o = dim_head
            )

    def forward(
        self,
        kv
    ):
        return self.compress(kv)

# simple transformer compressor, pull requested by Eric Pasewark

class SimpleMultiheadSelfAttention(Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout_p = dropout

    def forward(self, x):
        B, L, D = x.shape 
        q = self.q_proj(x)  # (B, L, D)
        k = self.k_proj(x)  # (B, L, D)
        v = self.v_proj(x)  # (B, L, D)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # -> (B, num_heads, L, head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=False
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(attn_out)

class SimpleTransformerFeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        """Two-layer feed-forward network with GELU activation."""
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out

class SimpleTransformerLayer(Module):
    def __init__(self, dim, num_heads, ff_hidden_dim=None, dropout=0.0):
        """Single Transformer layer: RMSNorm + Multi-head attention + RMSNorm + FeedForward."""
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = dim * 4
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.attn = SimpleMultiheadSelfAttention(dim, num_heads, dropout=dropout)
        self.ff   = SimpleTransformerFeedForward(dim, ff_hidden_dim, dropout=dropout)

    def forward(self, x):
        a = self.attn(self.norm1(x))
        x = x + a
        f = self.ff(self.norm2(x))
        x = x + f
        return x

class CompressTransformer(Module):
    def __init__(self, num_layers, dim, num_heads, ff_hidden_dim=None, dropout=0.0):
        """
        Stacked Transformer encoder layers.
        Args:
          num_layers: number of TransformerLayer to stack.
          dim: hidden dimension of the model (and input embeddings).
          num_heads: number of attention heads.
          ff_hidden_dim: hidden size of feed-forward network (defaults to 4*dim).
          dropout: dropout rate for attention weights and feed-forward (if any).
        """
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(dim, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x):
        # x shape: [b, h, w, n, d]
        b, h, w, n, d = x.shape

        # Flatten so each window is treated like a batch element: [b*w, n, h*d]
        inp = x.permute(0, 2, 3, 1, 4).contiguous()
        inp = inp.view(b*w, n, h*d)

        for i in range(self.num_layers - 1):
            inp = self.layers[i](inp)

        last_layer = self.layers[-1]

        a = last_layer.attn(last_layer.norm1(inp))
        inp = inp + a

        # Extract the last token along the 'n' dimension
        last_token = inp[:, -1].unsqueeze(1)  # (bw, 1, hd)

        normed = last_layer.norm2(last_token)
        ff_out = last_layer.ff(normed)
        last_token = last_token + ff_out

        last_token = last_token.squeeze(1).view(b, w, h, d).permute(0, 2, 1, 3)

        return last_token

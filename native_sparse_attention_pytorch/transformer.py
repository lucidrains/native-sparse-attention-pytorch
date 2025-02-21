import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from native_sparse_attention_pytorch.native_sparse_attention import (
    SparseAttention,
    create_compress_mask,
    create_fine_mask,
    create_sliding_mask,
)

# flex attention
# https://pytorch.org/blog/flexattention/

flex_attention = None

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)
except ImportError:
    pass

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        kv_heads = None
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        dim_inner = heads * dim_head
        dim_kv_inner = kv_heads * dim_head

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, dim_inner, bias = False)
        self.to_k = nn.Linear(dim, dim_kv_inner, bias = False)
        self.to_v = nn.Linear(dim, dim_kv_inner, bias = False)

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = nn.Linear(dim_inner, dim, bias = False)

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(self.split_heads, (q, k, v))

        # relative positions

        q, k = self.rotary_embed.rotate_queries_with_cached_keys(q, k)

        # naive gqa

        k, v = tuple(repeat(t, 'b h ... -> b (g h) ...', g = self.heads // self.kv_heads) for t in (k, v))

        # attention branch

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal = True
        )

        out = self.merge_heads(out)

        return self.to_out(out)

# feedforward

def FeedForward(dim, expansion_factor = 4.):
    dim_hidden = int(dim * expansion_factor)

    return nn.Sequential(
        RMSNorm(dim),
        Linear(dim, dim_hidden),
        nn.GELU(),
        Linear(dim_hidden, dim)
    )

# classes

class Transformer(Module):
    def __init__(
        self,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        kv_heads = None,
        ff_expansion_factor = 4.,
        use_sparse_attn = False,
        use_flex_sliding_window = False,
        use_flex_fine_selection = False,
        sparse_attn_kwargs: dict = dict(
            sliding_window_size = 32,
            compress_block_size = 4,
            selection_block_size = 4,
            num_selected_blocks = 4,
        )
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        if use_flex_sliding_window or use_flex_fine_selection:
            assert exists(flex_attention), 'flex attention is not available on your current version of pytorch'

        self.use_sparse_attn = use_sparse_attn
        self.use_flex_sliding_window = use_sparse_attn & use_flex_sliding_window
        self.use_flex_fine_selection = use_sparse_attn & use_flex_fine_selection

        layers = []
        for _ in range(depth):

            if use_sparse_attn:
                attn = SparseAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    kv_heads = kv_heads,
                    **sparse_attn_kwargs
                )
            else:
                attn = Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    kv_heads = kv_heads
                )

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([attn, ff]))

        self.attn_sliding_window_size = getattr(attn, 'sliding_window_size', None)
        self.attn_fine_block_size = getattr(attn, 'selection_block_size', None)

        self.layers = ModuleList(layers)

        self.norm = RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens, bias = False)
 
    def forward(
        self,
        ids,
        return_loss = False,
        disable_flex = False
    ):
        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        seq_len = ids.shape[-1]

        # token embedding

        tokens = self.token_emb(ids)

        # prepare maybe flex attention masks

        attn_kwargs = dict()

        if not disable_flex and self.use_flex_sliding_window:
            attn_kwargs.update(
                sliding_window_flex_mask = create_sliding_mask(seq_len, self.attn_sliding_window_size)
            )

        if not disable_flex and self.use_flex_fine_selection:
            attn_kwargs.update(
                fine_selection_flex_mask = create_fine_mask(seq_len, self.attn_fine_block_size)
            )

        # layers

        for attn, ff in self.layers:
            attn_out = attn(
                tokens,
                **attn_kwargs
            )

            tokens = attn_out + tokens
            tokens = ff(tokens) + tokens

        embed = self.norm(tokens)

        logits = self.to_logits(embed)

        if not return_loss:
            return logits

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)

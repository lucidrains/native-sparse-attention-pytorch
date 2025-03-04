import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm

from math import ceil
from tqdm import tqdm

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

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        causal = True,
        kv_heads = None
    ):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.heads = heads
        self.kv_heads = default(kv_heads, heads)
        dim_inner = heads * dim_head
        dim_kv_inner = kv_heads * dim_head

        self.causal = causal

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
            is_causal = self.causal
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
        causal = True,
        use_flex_sliding_window = False,
        use_flex_fine_selection = False,
        use_triton_fine_selection = False,
        sparse_attn_kwargs: dict = dict(
            sliding_window_size = 32,
            compress_block_size = 4,
            selection_block_size = 4,
            num_selected_blocks = 4,
        )
    ):
        super().__init__()
        assert at_most_one_of(use_flex_fine_selection, use_triton_fine_selection), 'either using flex or custom triton kernel for fine attn, but not both'

        self.token_emb = nn.Embedding(num_tokens, dim)

        if use_flex_sliding_window or use_flex_fine_selection:
            assert exists(flex_attention), 'flex attention is not available on your current version of pytorch'

        self.causal = causal

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
                    causal = causal,
                    use_triton_kernel = use_triton_fine_selection,
                    **sparse_attn_kwargs
                )
            else:
                attn = Attention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    causal = causal,
                    kv_heads = kv_heads
                )

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor)

            layers.append(ModuleList([attn, ff]))

        self.attn_sliding_window_size = getattr(attn, 'sliding_window_size', None)
        self.attn_fine_block_size = getattr(attn, 'selection_block_size', None)

        self.layers = ModuleList(layers)

        self.norm = RMSNorm(dim)
        self.to_logits = Linear(dim, num_tokens, bias = False)
 
    @torch.no_grad()
    def sample(
        self,
        prompt: Tensor,
        seq_len: int,
        temperature = 1.,
        filter_thres = 0.9,
        use_cache_kv = False
    ):
        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        cache = None

        for ind in tqdm(range(sample_num_times)):
            is_first = ind == 0

            logits, next_cache = self.forward(
                out,
                cache = cache,
                return_cache = True,
                disable_flex = True,
                disable_triton_kernel = not is_first
            )

            if use_cache_kv:
                cache = next_cache

            logits = logits[:, -1]
            logits = top_k(logits, thres = filter_thres)
            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    def forward_inference(
        self,
        ids,
        cache = None
    ):
        return ids

    def forward(
        self,
        ids,
        return_loss = False,
        disable_flex = False,
        disable_triton_kernel = False,
        cache = None,
        return_cache = False
    ):
        is_inferencing = exists(cache)

        if is_inferencing:
            disable_flex |= True
            disable_triton_kernel |= True

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        seq_len = ids.shape[-1]

        # token embedding

        tokens = self.token_emb(ids)

        # prepare maybe flex attention masks

        attn_kwargs = dict(
            disable_triton_kernel = disable_triton_kernel
        )

        if not disable_flex and self.use_flex_sliding_window:
            attn_kwargs.update(
                sliding_window_flex_mask = create_sliding_mask(seq_len, self.attn_sliding_window_size, causal = self.causal)
            )

        if not disable_flex and self.use_flex_fine_selection:
            attn_kwargs.update(
                fine_selection_flex_mask = create_fine_mask(seq_len, self.attn_fine_block_size, causal = self.causal)
            )

        # cache

        cache = default(cache, [])
        iter_cache = iter(cache)

        next_cache = []

        if is_inferencing:
            tokens = tokens[:, -1:]

        # layers

        for attn, ff in self.layers:

            attn_out, layer_cache = attn(
                tokens,
                cache = next(iter_cache, None),
                return_cache = True,
                **attn_kwargs
            )

            next_cache.append(layer_cache)

            tokens = attn_out + tokens
            tokens = ff(tokens) + tokens

        embed = self.norm(tokens)

        logits = self.to_logits(embed)

        if not return_loss:
            if not return_cache:
                return logits

            return logits, next_cache

        return F.cross_entropy(rearrange(logits, 'b n l -> b l n'), labels)

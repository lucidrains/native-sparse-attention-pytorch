import pytest

import torch
from torch import nn
from einops.layers.torch import Rearrange

from native_sparse_attention_pytorch import SparseAttention

@pytest.mark.parametrize('use_diff_topk', (False, True))
@pytest.mark.parametrize('seq_len', (1, 4, 31, 32, 120))
@pytest.mark.parametrize('kv_heads', (8, 4))
@pytest.mark.parametrize('selection_block_size', (4, 2))
def test_sparse_attn(
    use_diff_topk,
    seq_len,
    kv_heads,
    selection_block_size
):
    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        kv_heads = kv_heads,
        sliding_window_size = 2,
        compress_block_size = 4,
        selection_block_size = selection_block_size,
        num_selected_blocks = 2,
        use_diff_topk = use_diff_topk
    )

    tokens = torch.randn(2, seq_len, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

def test_alternative_compress_mlp():

    dim_head = 64
    compress_dim = dim_head * 4

    compress_mlp = nn.Sequential(
        Rearrange('b h w n d -> b h w (n d)'),
        nn.Linear(compress_dim, compress_dim),
        nn.SiLU(),
        nn.Linear(compress_dim, compress_dim),
        nn.SiLU(),
        nn.Linear(compress_dim, dim_head),
    )

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        sliding_window_size = 2,
        compress_block_size = 4,
        selection_block_size = 4,
        num_selected_blocks = 2,
        compress_mlp = compress_mlp
    )

    tokens = torch.randn(2, 31, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape


def test_compress_networks():
    from native_sparse_attention_pytorch.compress_networks import AttentionPool

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        sliding_window_size = 2,
        compress_block_size = 4,
        selection_block_size = 4,
        num_selected_blocks = 2,
        compress_mlp = AttentionPool(64, 4)
    )

    tokens = torch.randn(2, 31, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

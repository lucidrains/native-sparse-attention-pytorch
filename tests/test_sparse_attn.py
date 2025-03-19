import pytest

import torch
from torch import nn
from einops.layers.torch import Rearrange

from native_sparse_attention_pytorch import SparseAttention

@pytest.mark.parametrize('use_diff_topk', (False, True))
@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('seq_len', (1, 4, 31, 32, 120))
@pytest.mark.parametrize('kv_heads', (8, 4))
@pytest.mark.parametrize('selection_block_size', (8, 4, 2))
@pytest.mark.parametrize('compress_block_size', (8, 4))
@pytest.mark.parametrize('compress_block_overlap_len', (0, 2))
@pytest.mark.parametrize('num_selected_block', (0, 2))
@pytest.mark.parametrize('query_heads_share_selected_kv', (False, True))
@pytest.mark.parametrize('interpolated_importance_score', (False, True))
def test_sparse_attn(
    use_diff_topk,
    causal,
    seq_len,
    kv_heads,
    selection_block_size,
    compress_block_size,
    compress_block_overlap_len,
    num_selected_block,
    query_heads_share_selected_kv,
    interpolated_importance_score
):
    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        kv_heads = kv_heads,
        causal = causal,
        sliding_window_size = 2,
        selection_block_size = selection_block_size,
        compress_block_size = compress_block_size,
        compress_block_overlap_len = compress_block_overlap_len,
        num_selected_blocks = num_selected_block,
        use_diff_topk = use_diff_topk,
        query_heads_share_selected_kv = query_heads_share_selected_kv,
        interpolated_importance_score = interpolated_importance_score
    )

    tokens = torch.randn(2, seq_len, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

@pytest.mark.parametrize('seq_len', (2, 8, 16))
def test_inference(seq_len):

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        causal = True,
        sliding_window_size = 2,
        compress_block_size = 5,
        selection_block_size = 10,
        num_selected_blocks = 0
    )

    tokens = torch.randn(2, seq_len, 512)

    parallel_out = attn(tokens)

    cache = None
    sequential_out = []

    for i in range(seq_len):
      one_out, cache = attn(tokens[:, i:(i + 1)], cache = cache, return_cache = True)
      sequential_out.append(one_out)

    sequential_out = torch.cat(sequential_out, dim = 1)

    assert torch.allclose(parallel_out, sequential_out, atol = 1e-5)

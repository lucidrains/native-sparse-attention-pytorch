import pytest

import torch
from torch import nn
from einops.layers.torch import Rearrange

from native_sparse_attention_pytorch import SparseAttention

@pytest.mark.parametrize('use_diff_topk', (False, True))
@pytest.mark.parametrize('causal', (False, True))
@pytest.mark.parametrize('seq_len', (1, 4, 31, 32, 120))
@pytest.mark.parametrize('kv_heads', (8, 4))
@pytest.mark.parametrize('selection_block_size', (8, 16, 32))
@pytest.mark.parametrize('compress_block_size', (8, 4))
@pytest.mark.parametrize('compress_block_sliding_stride', (2, 4))
@pytest.mark.parametrize('num_selected_block', (0, 2))
@pytest.mark.parametrize('query_heads_share_selected_kv', (False, True))
def test_sparse_attn(
    use_diff_topk,
    causal,
    seq_len,
    kv_heads,
    selection_block_size,
    compress_block_size,
    compress_block_sliding_stride,
    num_selected_block,
    query_heads_share_selected_kv,
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
        compress_block_sliding_stride = compress_block_sliding_stride,
        num_selected_blocks = num_selected_block,
        use_diff_topk = use_diff_topk,
        query_heads_share_selected_kv = query_heads_share_selected_kv,
    )

    tokens = torch.randn(2, seq_len, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

@pytest.mark.parametrize('seq_len', (2, 8, 16))
@pytest.mark.parametrize('num_selected_blocks', (0, 2))
@pytest.mark.parametrize('compress_block_sliding_stride', (2, 4))
@pytest.mark.parametrize('selection_block_size', (8, 16, 20))
def test_inference(
    seq_len,
    num_selected_blocks,
    compress_block_sliding_stride,
    selection_block_size
):

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        causal = True,
        sliding_window_size = 2,
        compress_block_size = 5,
        selection_block_size = selection_block_size,
        num_selected_blocks = num_selected_blocks,
        compress_block_sliding_stride = compress_block_sliding_stride
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

@pytest.mark.parametrize('selection_block_size', (4, 8))
def test_transformer_inference(
    selection_block_size
):
    from native_sparse_attention_pytorch.transformer import Transformer

    model = Transformer(
        num_tokens = 256,
        dim = 512,
        depth = 2,
        causal = True,
        use_sparse_attn = True,
        sparse_attn_kwargs = dict(
            sliding_window_size = 16,
            compress_block_size = 4,
            compress_block_sliding_stride = 2,
            selection_block_size = selection_block_size,
            num_selected_blocks = 2
        )
    )

    prompt = torch.randint(0, 256, (1, 1))

    sampled = model.sample(prompt, 128, temperature = 0., use_cache_kv = False)
    sampled_cached = model.sample(prompt, 128, temperature = 0., use_cache_kv = True)

    assert torch.allclose(sampled, sampled_cached)

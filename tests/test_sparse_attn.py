import pytest
import torch

def test_sparse_attn():
    from native_sparse_attention_pytorch import SparseAttention

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        sliding_window_size = 2,
        compress_block_size = 4,
        selection_block_size = 4,
        num_selected_blocks = 2
    )

    tokens = torch.randn(2, 31, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

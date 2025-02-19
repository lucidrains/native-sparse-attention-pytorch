import pytest
import torch

@pytest.mark.parametrize('use_diff_topk', (False, True))
def test_sparse_attn(
    use_diff_topk
):
    from native_sparse_attention_pytorch import SparseAttention

    attn = SparseAttention(
        dim = 512,
        dim_head = 64,
        heads = 8,
        sliding_window_size = 2,
        compress_block_size = 4,
        selection_block_size = 4,
        num_selected_blocks = 2,
        use_diff_topk = use_diff_topk
    )

    tokens = torch.randn(2, 31, 512)

    attended = attn(tokens)

    assert tokens.shape == attended.shape

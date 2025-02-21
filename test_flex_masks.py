import torch
from native_sparse_attention_pytorch.native_sparse_attention import (
    create_compress_mask,
    create_fine_mask,
    create_sliding_mask,
)

# compress

print('compress mask:', create_compress_mask(1024, 256, 4))

# fine

selected_blocks = torch.randint(0, 5, (1, 1, 1024, 2)) # select mostly first few blocks

fine_block_mask = create_fine_mask(1024, 64)(selected_blocks.cuda())

print('fine:', fine_block_mask)

# sliding

print('sliding:', create_sliding_mask(1024, 32))

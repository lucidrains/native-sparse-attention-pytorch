import math
import gzip
import random
from tqdm import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from native_sparse_attention_pytorch.transformer import Transformer

from native_sparse_attention_pytorch.compress_networks import (
    ConvLinearCompress,
    AttentionPool,
    GroupedMLP
)

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 64
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512
HEADS = 8
KV_HEADS = 4

USE_SPARSE_ATTN = True
USE_TRITON_NSA = True
USE_FLEX_FOR_FINE_SELECTION = False   # will push flex a bit, won't be efficient as each layer needs sparsity dynmically generated, but may be enough just to compare to full attention before going all-in on triton kernels
QUERY_HEADS_SHARE_SELECTION = True    # if set to False, each query head can look at a different segment of their corresponding key / value head in GQA

# sparse attention related

SLIDING_WINDOW_SIZE = 64
COMPRESS_BLOCK_SIZE = 16

FINE_BLOCK_SIZE = 16
NUM_FINE_SELECTED = 4

INTERPOLATED_IMPORTANCE_SCORE = False
USE_DIFF_TOPK = True

USE_EFFICIENT_INFERENCE = True # needs validation still

# experiment related

PROJECT_NAME = 'native-sparse-attention'
RUN_NAME = 'baseline' if not USE_SPARSE_ATTN else f'sparse-attn: compress size {COMPRESS_BLOCK_SIZE} | fine size {FINE_BLOCK_SIZE} | {NUM_FINE_SELECTED} selected'
WANDB_ONLINE = False # turn this on to pipe experiment to cloud

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# printing

if USE_TRITON_NSA:
    print('using custom triton kernel')
elif USE_FLEX_FOR_FINE_SELECTION:
    print('using flex attn')
else:
    print('sparse attn in regular pytorch')

# model

model = Transformer(
    num_tokens = 256,
    dim = 512,
    depth = 6,
    heads = HEADS,
    dim_head = 64,
    kv_heads = KV_HEADS,
    use_sparse_attn = USE_SPARSE_ATTN,
    use_flex_sliding_window = True,
    use_triton_fine_selection = USE_TRITON_NSA,
    use_flex_fine_selection = USE_FLEX_FOR_FINE_SELECTION,
    sparse_attn_kwargs = dict(
        sliding_window_size = SLIDING_WINDOW_SIZE,
        compress_block_size = COMPRESS_BLOCK_SIZE,
        compress_mlp = GroupedMLP(
            dim_head = 64,
            compress_block_size = COMPRESS_BLOCK_SIZE,
            heads = KV_HEADS,
        ),
        selection_block_size = FINE_BLOCK_SIZE,
        num_selected_blocks = NUM_FINE_SELECTED,
        use_diff_topk = USE_DIFF_TOPK,
        interpolated_importance_score = INTERPOLATED_IMPORTANCE_SCORE,
        query_heads_share_selected_kv = QUERY_HEADS_SHARE_SELECTION
    )
).cuda()

# prepare enwik8 data

with gzip.open('./data/enwik8.gz') as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq.cuda()

train_dataset = TextSamplerDataset(data_train, SEQ_LEN)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer

optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# wandb experiment tracker

import wandb
wandb.init(project = PROJECT_NAME, mode = 'disabled' if not WANDB_ONLINE else 'online')
wandb.run.name = RUN_NAME
wandb.run.save()

# training

for i in tqdm(range(NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        loss = model(data, return_loss = True)

        (loss / GRAD_ACCUM_EVERY).backward()

    wandb.log(dict(loss = loss.item()), step = i)
    print(f"training loss: {loss.item():.3f}")

    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optim.step()
    optim.zero_grad()

    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)

            loss = model(valid_data, return_loss = True)
            wandb.log(dict(valid_loss = loss.item()), step = i)
            print(f"validation loss: {loss.item():.3f}")

    if i % GENERATE_EVERY == 0:
        model.eval()

        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.cuda()

        prime = decode_tokens(inp)
        print(f"\n{prime}\n")

        prompt = inp[None, ...]

        sampled = model.sample(
            prompt,
            GENERATE_LENGTH,
            use_cache_kv = USE_EFFICIENT_INFERENCE
        )

        base_decode_output = decode_tokens(sampled[0])

        print(f"\n{base_decode_output}\n")

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
USE_FLEX_FOR_FINE_SELECTION = True   # will push flex a bit, won't be efficient as each layer needs sparsity dynmically generated, but may be enough just to compare to full attention before going all-in on triton kernels
QUERY_HEADS_SHARE_SELECTION = False  # if set to False, each query head can look at a different segment of their corresponding key / value head in GQA

# sparse attention related

SLIDING_WINDOW_SIZE = 64
COMPRESS_BLOCK_SIZE = 64

FINE_BLOCK_SIZE = 32
NUM_FINE_SELECTED = 0

INTERPOLATED_IMPORTANCE_SCORE = False
USE_DIFF_TOPK = True

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

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in tqdm(range(sample_num_times)):
        logits = net(out, disable_flex = True)

        logits = logits[:, -1]
        logits = top_k(logits, thres = filter_thres)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

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

        sampled = base_decoding(model, prompt, GENERATE_LENGTH)

        base_decode_output = decode_tokens(sampled[0])

        print(f"\n{base_decode_output}\n")

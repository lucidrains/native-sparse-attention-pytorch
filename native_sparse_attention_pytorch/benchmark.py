import time
import torch
import numpy as np

# Import your modules
from native_sparse_attention import SparseAttention
from transformer import Attention as StandardAttention


def benchmark_module(module, x, runs=50, warmups=5):
    """
    Benchmark forward and backward pass of a module.
    Returns forward and backward times as numpy arrays (in seconds).
    """
    # Warm-up to stabilize JIT/CUDA and trigger Triton compilation
    for _ in range(warmups):
        out = module(x)
        loss = out.sum()
        loss.backward()
        module.zero_grad()

    # Forward timing
    fwd_times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = module(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t0)

    # Backward timing
    bwd_times = []
    for _ in range(runs):
        out = module(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        bwd_times.append(time.perf_counter() - t0)
        module.zero_grad()

    return np.array(fwd_times), np.array(bwd_times)


def main():
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 128
    seq_len = 1024
    d_model = 512
    n_heads = 8
    d_head = d_model // n_heads

    # NSA-specific hyperparameters
    sliding_window_size = 32
    compress_block_size = 4
    compress_block_sliding_stride = 4
    selection_block_size = 16
    num_selected_blocks = 4

    # Create input tensor
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Instantiate modules with identical hyperparameters
    std_attn = StandardAttention(
        dim=d_model,
        dim_head=d_head,
        heads=n_heads,
        causal=True
    ).to(device)

    nsa_attn = SparseAttention(
        d_model,
        d_head,
        n_heads,
        sliding_window_size,
        compress_block_size,
        compress_block_sliding_stride,
        selection_block_size,
        num_selected_blocks,
        use_triton_kernel=True
    ).to(device)

    # Run benchmarks
    runs = 5000
    warmups = 500
    std_fwd, std_bwd = benchmark_module(std_attn, x, runs, warmups)
    nsa_fwd, nsa_bwd = benchmark_module(nsa_attn, x, runs, warmups)

    # Report results
    print(f"{'Module':<25}{'Fwd Mean (ms)':>15}{'Fwd Std (ms)':>15}{'Bwd Mean (ms)':>15}{'Bwd Std (ms)':>15}")
    for name, fwd, bwd in [
        ("StandardAttention", std_fwd, std_bwd),
        ("SparseAttention (NSA)", nsa_fwd, nsa_bwd),
    ]:
        print(f"{name:<25}{fwd.mean()*1000:>15.3f}{fwd.std()*1000:>15.3f}{bwd.mean()*1000:>15.3f}{bwd.std()*1000:>15.3f}")

    print(nsa_attn.timer)


if __name__ == "__main__":
    main()

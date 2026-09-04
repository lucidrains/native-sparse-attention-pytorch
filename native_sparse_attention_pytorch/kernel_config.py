"""Runtime configuration helpers for the Triton attention kernels.

The fine-grained attention kernel uses a fairly large head-dimension tile.  On
GPUs with a 96 KiB shared-memory limit (for example, RTX 30-series cards), the
default tile can exceed the hardware limit when the head dimension is 128.
These helpers keep the hardware check separate from the Triton launch code so
that the policy can also be tested on a CPU-only machine.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


# The 128-wide head tile with a 16-row query tile can require more than 136 KiB
# in recent Triton releases.  Leave a little headroom below the 144 KiB cutoff
# because the exact allocation depends on the Triton version and architecture.
SAFE_SHARED_MEMORY_CUTOFF = 144 * 1024


def get_shared_memory_limit(device) -> Optional[int]:
    """Return the largest shared-memory limit reported by the CUDA driver.

    CUDA exposes both the regular per-block limit and, on some architectures,
    an opt-in limit.  Triton uses the largest limit available to the compiled
    kernel, so considering all known property names avoids selecting a tile
    that is unnecessarily conservative.  Device probing is deliberately best
    effort: if a backend does not expose the properties, callers use the safe
    tile instead.
    """

    if getattr(device, 'type', None) != 'cuda':
        return None

    try:
        properties = torch.cuda.get_device_properties(device)
    except Exception:
        return None

    limits = []
    for name in (
        'shared_memory_per_block',
        'shared_memory_per_block_optin',
        'max_shared_memory_per_block',
        'max_shared_mem',
    ):
        value = getattr(properties, name, None)
        if value is None:
            continue
        try:
            value = int(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            limits.append(value)

    return max(limits, default=None)


def select_kernel_config(
    dim: int,
    shared_memory_limit: Optional[int],
) -> Tuple[int, int, int]:
    """Choose ``(query_block, num_warps, num_stages)`` for a head dimension.

    A missing limit is treated conservatively because the launch may otherwise
    fail during Triton's first compilation.  The smaller query tile preserves
    the kernel's 16-wide dot-product dimensions while substantially reducing
    intermediate tensors for 128-wide heads.
    """

    if dim <= 0:
        raise ValueError('head dimension must be positive')

    # Match triton.next_power_of_2(dim), with the kernel's minimum tile width.
    block_head_dim = max(16, 1 << (dim - 1).bit_length())
    needs_safe_tile = (
        block_head_dim >= 128 and
        (shared_memory_limit is None or shared_memory_limit < SAFE_SHARED_MEMORY_CUTOFF)
    )

    if needs_safe_tile:
        return 8, 4, 1

    return 16, (4 if dim <= 64 else 8), 1


def is_shared_memory_error(error: BaseException) -> bool:
    """Identify Triton's compile-time shared-memory resource exception."""

    return (
        error.__class__.__name__ == 'OutOfResources' and
        'shared memory' in str(error).lower()
    )


def launch_with_shared_memory_fallback(launch, config):
    """Launch a kernel and retry once with the conservative query tile.

    ``launch`` is kept as a callback so this policy can be exercised without a
    CUDA device.  Non-resource errors and an already-safe configuration are
    propagated unchanged.
    """

    try:
        return launch(*config)
    except Exception as error:
        if config[0] <= 8 or not is_shared_memory_error(error):
            raise
        return launch(8, 4, 1)

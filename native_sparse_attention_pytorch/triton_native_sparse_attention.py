from __future__ import annotations
from native_sparse_attention_pytorch.tensor_typing import Float, Int, Bool

# taken from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# with fixes for triton 2.3

from functools import partial
import math
from math import ceil

import torch
from torch import Tensor
import torch.nn.functional as F

from einops import repeat, rearrange, reduce

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(num, den):
    return (num % den) == 0

def round_up_multiple(n, mult):
    return ceil(n / mult) * mult

def pad_at_dim(t, pad: tuple[int, int], *, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_to_multiple(t, mult, *, dim):
    length = t.shape[dim]
    padded_length = round_up_multiple(length, mult)
    remainder = padded_length - length
    return pad_at_dim(t, (0, remainder), dim = dim)

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1

TRITON_BLOCK_SIZE = 128 # some block size that allows triton not to break, at least half a year ago

INSTALL_COMMAND = 'pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly'

# make sure triton 2.1+ is installed

import packaging.version as pkg_version

import importlib
from importlib.metadata import version

try:
    triton_version = version('triton')
except:
    print(f'latest triton must be installed. `{INSTALL_COMMAND}` first')
    exit()

assert pkg_version.parse(triton_version) >= pkg_version.parse('3.0.0'), f'triton must be version 3.0.0 or above. `{INSTALL_COMMAND}` to upgrade'

import triton
import triton.language as tl
from triton.language.extra import libdevice

# kernels

@triton.jit
def forward_kernel_causal_and_sparse(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)

    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads

    offs_qh = off_h * QUERY_HEAD_GROUPS + tl.arange(0, QUERY_HEAD_GROUPS)

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q +
        off_b * stride_qb +
        offs_qh[None, :, None] * stride_qh +
        offs_m[:, None, None] * stride_qm +
        offs_d[None, None, :]
    )

    # maximum

    m_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype = tl.float32) - float("inf")

    # lse

    lse_ptrs = (
        Lse +
        off_b * stride_lse_b +
        offs_qh[None, :] * seqlen_q_rounded +
        offs_m[:, None]
    )

    lse_i = tl.zeros([BLOCK, QUERY_HEAD_GROUPS], dtype = tl.float32) - float("inf")

    # output

    out_ptrs = (
        Out +
        off_b * stride_ob +
        offs_qh[None, :, None] * stride_oh +
        offs_m[:, None, None] * stride_om +
        offs_d[None, None, :]
    )

    acc_o = tl.zeros([BLOCK,  QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype = tl.float32)

    # load queries, keys, values

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(
                q_ptrs,
                mask = offs_d[None, None, :] < headdim,
                other = 0.0
            )
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs,
                mask = offs_m[:, None, None] < seqlen_q,
                other = 0.0
            )
        else:
            q = tl.load(
                q_ptrs,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other = 0.0
            )

    q = q.reshape(BLOCK * QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if INCLUDE_BLOCK_CAUSAL:

        if SLIDING:
            num_kv_blocks = 2
            offset = -BLOCK
        else:
            num_kv_blocks = 1
            offset = 0

        offs_n = start_m * BLOCK + tl.arange(0, BLOCK) + offset

        for _ in range(num_kv_blocks):

            k_ptrs = (
                K +
                off_b * stride_kb +
                off_h * stride_kh +
                offs_n[:, None] * stride_kn +
                offs_d[None, :]
            )

            v_ptrs = (
                V +
                off_b * stride_vb +
                off_h * stride_vh +
                offs_n[:, None] * stride_vn +
                offs_d[None, :]
            )

            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    k = tl.load(k_ptrs)
                else:
                    k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            else:
                if EVEN_HEADDIM:
                    k = tl.load(
                        k_ptrs,
                        mask = offs_n[:, None] < seqlen_k,
                        other = 0.0,
                    )
                else:
                    k = tl.load(
                        k_ptrs,
                        mask = (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other = 0.0,
                    )

            qk = tl.zeros([BLOCK * QUERY_HEAD_GROUPS, BLOCK], dtype=tl.float32)

            qk += tl.dot(q, tl.trans(k))

            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)

            if not EVEN_N:
                within_range_mask = offs_n[None, :] < seqlen_k

                if SLIDING:
                    within_range_mask &= offs_n[None, :] >= 0.

                qk += tl.where(within_range_mask, 0, float("-inf"))

            qk = qk.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)

            causal_mask = offs_m[:, None, None] >= offs_n[None, None, :]

            if SLIDING:
                causal_mask &= (offs_n[None, None, :] - offs_m[:, None, None]) <= BLOCK

            qk += tl.where(causal_mask, 0, float("-inf"))

            m_ij = tl.maximum(tl.max(qk, 2) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, :, None])

            l_ij = tl.sum(p, 2)

            acc_o_scale = tl.exp(m_i - m_ij)
            acc_o *= acc_o_scale[:, :, None]

            if EVEN_N & EVEN_M:
                if EVEN_HEADDIM:
                    v = tl.load(v_ptrs)
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = offs_d[None, :] < headdim,
                        other = 0.0
                    )
            else:
                if EVEN_HEADDIM:
                    v = tl.load(
                        v_ptrs,
                        mask = offs_n[:, None] < seqlen_k,
                        other = 0.0,
                    )
                else:
                    v = tl.load(
                        v_ptrs,
                        mask = (offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                        other = 0.0,
                    )

            p = p.reshape(BLOCK * QUERY_HEAD_GROUPS, BLOCK).to(v.dtype)

            causal_o = tl.dot(p, v)

            acc_o += causal_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

            # -- update statistics

            m_i = m_ij
            l_i_new = tl.exp(lse_i - m_ij) + l_ij
            lse_i = m_ij + tl.log(l_i_new)

    # # take care of the selected kv blocks

    kv_block_indices_ptrs = (
        kv_block_indices +
        off_b * stride_kvbl_b +
        off_h * stride_kvbl_h +
        offs_m * stride_kvbl_m
    )

    kv_block_mask_ptrs = (
        kv_block_mask +
        off_b * stride_kvbl_b +
        off_h * stride_kvbl_h +
        offs_m * stride_kvbl_m
    )

    q = q.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)
    q = tl.expand_dims(q, 2)
    q = tl.broadcast_to(q, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    q = q.reshape(BLOCK, 16, BLOCK_HEADDIM)

    for off_sel_kv_block in range(NUM_SEL_KV_BLOCKS):
        block_indices = tl.load(kv_block_indices_ptrs + off_sel_kv_block)
        block_masks = tl.load(kv_block_mask_ptrs + off_sel_kv_block)

        blocks_offs_n = block_indices[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]

        block_k_ptrs = (
            K +
            off_b * stride_kb +
            off_h * stride_kh +
            blocks_offs_n[:, :, None] * stride_kn +
            offs_d[None, None, :]
        )

        block_v_ptrs = (
            V +
            off_b * stride_vb +
            off_h * stride_vh + 
            blocks_offs_n[:, :, None] * stride_vn +
            offs_d[None, None, :]
        )

        # load k of shape (m, n, d), sparsely selected by each query

        k_block = tl.load(block_k_ptrs)

        # similarities

        block_qk = tl.zeros([BLOCK, 16, BLOCK], dtype = tl.float32)
        qk = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK], dtype = tl.float32)

        k_block = k_block.reshape(BLOCK, BLOCK, BLOCK_HEADDIM)
        k_block = k_block.permute(0, 2, 1)

        block_qk += tl.dot(q, k_block)
        block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
        block_qk = tl.sum(block_qk, 2) / QUERY_EXPAND_DIM

        qk += block_qk
        qk += tl.where(block_masks[:, None, None], 0, float("-inf"))

        # attention

        m_ij = tl.maximum(tl.max(qk, 2) * softmax_scale, lse_i)
        block_p = tl.exp(qk * softmax_scale - m_ij[:, :, None])

        l_ij = tl.sum(block_p, 2)

        # renormalize the running output

        acc_o_scale = tl.exp(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, :, None]

        # aggregate values

        v_block = tl.load(block_v_ptrs)
        v_block = tl.reshape(v_block, (BLOCK, BLOCK, BLOCK_HEADDIM))

        block_p = block_p.to(v_block.dtype)
        p_expanded = block_p.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK)
        p_expanded = tl.expand_dims(p_expanded, 2)
        p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
        p_expanded = p_expanded.reshape(BLOCK, 16, BLOCK)

        block_acc_o = tl.dot(p_expanded, v_block)
        block_acc_o = block_acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)
        block_acc_o = tl.sum(block_acc_o, 2) / QUERY_EXPAND_DIM

        acc_o += block_acc_o

        # -- update statistics

        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    # normalize accumulated out

    acc_o_scale = tl.exp(m_i - lse_i)
    acc_o *= acc_o_scale[:, :, None]

    # write back lse

    lse_i = lse_i.reshape(BLOCK, QUERY_HEAD_GROUPS)
    tl.store(lse_ptrs, lse_i)

    # write to output

    acc_o = acc_o.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_d[None, None, :] < headdim
            )
    else:
        if EVEN_HEADDIM:
            tl.store(
                out_ptrs,
                acc_o,
                mask = offs_m[:, None, None] < seqlen_q
            )
        else:
            tl.store(
                out_ptrs,
                acc_o,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim)
            )

@triton.heuristics(
    dict(
        EVEN_M = lambda args: divisible_by(args["seqlen_q"], args["BLOCK"]),
        EVEN_N = lambda args: divisible_by(args["seqlen_k"], args["BLOCK"]),
        EVEN_HEADDIM = lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def forward_kernel(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    Out,
    Lse,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr
):
    forward_kernel_causal_and_sparse(
        Q,
        K,
        V,
        kv_block_indices,
        kv_block_mask,
        Out,
        Lse,
        softmax_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_ob,
        stride_oh,
        stride_om,
        stride_kvbl_b,
        stride_kvbl_h,
        stride_kvbl_m,
        stride_lse_b,
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        headdim,
        CACHE_KEY_SEQLEN_Q,
        CACHE_KEY_SEQLEN_K,
        BLOCK_HEADDIM,
        EVEN_M,
        EVEN_N,
        EVEN_HEADDIM,
        BLOCK,
        QUERY_HEAD_GROUPS,
        QUERY_EXPAND_DIM,
        NUM_SEL_KV_BLOCKS,
        INCLUDE_BLOCK_CAUSAL,
        SLIDING
    )

def native_sparse_attn_forward(
    q,
    k,
    v,
    kv_block_indices,
    kv_block_mask,
    block_size = 128,
    include_block_causal = True
):
    q, k, v, kv_block_indices = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v, kv_block_indices)]

    batch, nheads, seqlen_q, dim, device = *q.shape, q.device
    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(nheads, kv_heads)
    head_groups = nheads // kv_heads

    num_selected_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    assert k.shape == (batch, kv_heads, seqlen_k, dim)
    assert v.shape == (batch, kv_heads, seqlen_k, dim)
    assert dim <= 128, "only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert all([t.is_cuda for t in (q, k, v)])

    softmax_scale = dim ** -0.5

    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device = device, dtype = torch.float32)

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps = 4 if dim <= 64 else 8

    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * kv_heads) # kv heads here, as grouped query heads all loaded, following the paper

    forward_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        o,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        kv_block_indices.stride(0),
        kv_block_indices.stride(1),
        kv_block_indices.stride(2),
        lse.stride(0),
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,
        BLOCK_HEADDIM,
        BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        NUM_SEL_KV_BLOCKS = num_selected_fine_blocks,
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        SLIDING = False,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, lse

@triton.jit
def backward_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    qheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // qheads
    off_h = off_hb % qheads

    # initialize offsets

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # load

    o = tl.load(
        Out +
        off_b * stride_ob +
        off_h * stride_oh +
        offs_m[:, None] * stride_om +
        offs_d[None, :],
        mask = (
            (offs_m[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim)
        ),
        other = 0.0,
    ).to(tl.float32)

    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask = (
            offs_m[:, None] < seqlen_q) &
            (offs_d[None, :] < headdim
        ),
        other = 0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    # write-back

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def backward_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, sem = 'relaxed')
        else:
            tl.atomic_add(dv_ptrs, dv, mask=offs_d[None, :] < headdim, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=offs_d[None, :] < headdim, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k, sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k, sem = 'relaxed')
        else:
            tl.atomic_add(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), sem = 'relaxed')
            tl.atomic_add(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), sem = 'relaxed')


@triton.jit
def backward_kernel_one_col_block_sparse(
    start_n,
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    OFF_SEL_KV_BLOCKS: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)

    begin_m = ((start_n * BLOCK) // BLOCK) * BLOCK

    # initialize row/col offsets

    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)

    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m

    # initialize pointers to value-like data

    q_ptrs = (
        Q +
        offs_g[None, :, None] * stride_qh +
        offs_qm[:, None, None] * stride_qm +
        offs_d[None, None, :]
    )

    do_ptrs = (
        DO +
        offs_g[None, :, None] * stride_doh +
        offs_qm[:, None, None] * stride_dom +
        offs_d[None, None, :]
    )

    dq_ptrs = (
        DQ +
        offs_g[None, :, None] * stride_dqh +
        offs_qm[:, None, None] * stride_dqm +
        offs_d[None, None, :]
    )

    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.

    if begin_m >= seqlen_q:
        return

    # same block for block causal diagonal

    # load q, k, v, do on-chip
    # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(
                q_ptrs,
                mask = offs_m[:, None, None] < seqlen_q,
                other = 0.0
            )
        else:
            q = tl.load(
                q_ptrs,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other = 0.0,
            )

    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = tl.trans(lse_i) # (m, h)

    # compute dv
    # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
    # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
    # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
    # the output is correct.
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
        do = tl.load(
            do_ptrs,
            mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
            other = 0.0,
        )

    # compute dp = dot(v, do)
    # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
    # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
    # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    # There's a race condition for headdim=48
    if not EVEN_HEADDIM:
        tl.debug_barrier()

    # compute ds = p * (dp - delta[:, None])
    # Putting the subtraction after the dp matmul (instead of before) is slightly faster

    Di = tl.load(D + offs_d_or_lse)
    Di = tl.trans(Di) # (m, h)

    # Converting ds to q.dtype here reduces register pressure and makes it much faster
    # for BLOCK_HEADDIM=128

    dq = tl.zeros([BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM], dtype = tl.float32)

    # handle kv block indices using atomic adds for starters, todo: swap dq and dk/dv loops at some point, semi big refactor

    kv_block_indices_ptrs = (
        kv_block_indices +
        offs_m * stride_kvbl_m
    )

    kv_block_mask_ptrs = (
        kv_block_mask +
        offs_m * stride_kvbl_m
    )

    block_indices = tl.load(kv_block_indices_ptrs + OFF_SEL_KV_BLOCKS)
    block_masks = tl.load(kv_block_mask_ptrs + OFF_SEL_KV_BLOCKS)

    blocks_offs_n = (
        block_indices[:, None] * BLOCK +
        tl.arange(0, BLOCK)[None, :]
    )

    block_k_ptrs = (
        K +
        blocks_offs_n[:, :, None] * stride_kn +
        offs_d[None, None, :]
    )

    block_v_ptrs = (
        V +
        blocks_offs_n[:, :, None] * stride_vn +
        offs_d[None, None, :]
    )

    block_dv_ptrs = (
        DV +
        blocks_offs_n[:, :, None] * stride_dvn +
        offs_d[None, None, :]
    )

    block_dk_ptrs = (
        DK +
        blocks_offs_n[:, :, None] * stride_dkn +
        offs_d[None, None, :]
    )

    block_k = tl.load(block_k_ptrs)
    block_v = tl.load(block_v_ptrs)

    q_expanded = tl.expand_dims(q, 2)
    q_expanded = tl.broadcast_to(q_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    q_expanded = q_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)

    block_k_permuted = tl.permute(block_k, (0, 2, 1))
    block_qk = tl.dot(q_expanded, block_k_permuted)

    block_qk = block_qk.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
    qk = tl.sum(block_qk, 2) / QUERY_EXPAND_DIM

    masked_qk = qk + tl.where(block_masks[:, None, None], 0, float("-inf"))

    p = tl.exp(masked_qk * softmax_scale - lse_i[:, :, None])

    # prepare do

    do_expanded = tl.expand_dims(do, 2)
    do_expanded = tl.broadcast_to(do_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
    do_expanded = do_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)

    # take care of block dv

    if not BLOCK_DV_USE_DOT:
        p_expanded = p.to(do.dtype)
        p_expanded = tl.expand_dims(p_expanded, 2)
        p_expanded = tl.broadcast_to(p_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
        p_expanded = p_expanded.reshape(BLOCK, QUERY_HEAD_GROUPS * QUERY_EXPAND_DIM, BLOCK)
        p_expanded = tl.permute(p_expanded, (0, 2, 1))

        block_dv = tl.dot(p_expanded, do_expanded) / QUERY_EXPAND_DIM
    else:
        block_dv = p.to(do.dtype)[:, :, :, None] * do[:, :, None, :]
        block_dv = tl.sum(block_dv, 1)

    tl.atomic_add(block_dv_ptrs, block_dv, mask = block_masks[:, None, None], sem = 'relaxed')

    # get dp

    block_v = tl.permute(block_v, (0, 2, 1))

    dp = tl.dot(do_expanded, block_v)

    dp = dp.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK)
    dp = tl.sum(dp, 2) / QUERY_EXPAND_DIM

    # ds

    ds = (p * (dp - Di[:, :, None]) * softmax_scale)

    # maybe return gradients for better differentiable topk

    if RETURN_SEL_GRADS:

        kv_block_grads_ptrs = (
            kv_block_grads +
            offs_m * stride_kvbl_m
        )

        sel_grads = ds * qk
        sel_grads = tl.where(block_masks[:, None, None], sel_grads, 0.)
        sel_grads = sel_grads.reshape(BLOCK, QUERY_HEAD_GROUPS * BLOCK)
        sel_grads = tl.sum(sel_grads, 1)

        tl.atomic_add(
            kv_block_grads_ptrs + OFF_SEL_KV_BLOCKS,
            sel_grads,
            sem = 'relaxed'
        )

    # ds

    ds_expanded = tl.expand_dims(ds, 2)
    ds_expanded = tl.broadcast_to(ds_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK))
    ds_expanded = ds_expanded.reshape(BLOCK, 16, BLOCK)

    # block dk

    if BLOCK_DK_USE_DOT:
        q_expanded = tl.expand_dims(q, 2)
        q_expanded = tl.broadcast_to(q_expanded, (BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM))
        q_expanded = q_expanded.reshape(BLOCK, 16, BLOCK_HEADDIM)

        block_dk = tl.dot(tl.permute(ds_expanded, (0, 2, 1)), q_expanded.to(ds.dtype)) / QUERY_EXPAND_DIM
    else:
        block_dk = ds[:, :, :, None] * q[:, :, None, :].to(ds.dtype)
        block_dk = tl.sum(block_dk, 1)

    tl.atomic_add(
        block_dk_ptrs,
        block_dk,
        mask = block_masks[:, None, None] & (blocks_offs_n[:, :, None] < seqlen_k),
        sem = 'relaxed'
    )

    # block dq

    ds_expanded = ds_expanded.to(block_k.dtype)
    block_dq = tl.dot(ds_expanded, block_k)

    block_dq = block_dq.reshape(BLOCK, QUERY_HEAD_GROUPS, QUERY_EXPAND_DIM, BLOCK_HEADDIM)
    block_dq = tl.sum(block_dq, 2) / QUERY_EXPAND_DIM

    dq += block_dq

    # update dq

    dq = dq.reshape(BLOCK, QUERY_HEAD_GROUPS, BLOCK_HEADDIM)

    if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
        tl.atomic_add(dq_ptrs, dq, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = offs_m[:, None, None] < seqlen_q,
                sem = 'relaxed'
            )
        else:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = (offs_m[:, None, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                sem = 'relaxed',
            )

@triton.jit
def backward_kernel_one_col_block_causal(
    start_n,
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    stride_kvbl_m,
    stride_qh,
    stride_doh,
    stride_dqh,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    SLIDING: tl.constexpr
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)

    begin_m = ((start_n * BLOCK) // BLOCK) * BLOCK

    # initialize row/col offsets

    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    offs_g = tl.arange(0, QUERY_HEAD_GROUPS)

    offs_d_or_lse = seqlen_q_rounded * offs_g[:, None] + offs_m

    # initialize pointers to value-like data

    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    q_ptrs = (
        Q +
        offs_g[:, None, None] * stride_qh +
        offs_qm[None, :, None] * stride_qm +
        offs_d[None, None, :]
    )

    do_ptrs = (
        DO +
        offs_g[:, None, None] * stride_doh +
        offs_qm[None, :, None] * stride_dom +
        offs_d[None, None, :]
    )

    dq_ptrs = (
        DQ +
        offs_g[:, None, None] * stride_dqh +
        offs_qm[None, :, None] * stride_dqm +
        offs_d[None, None, :]
    )

    # initialize dv and dk

    dv = tl.zeros([BLOCK, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK, BLOCK_HEADDIM], dtype=tl.float32)

    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.

    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

        backward_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )

    # same block for block causal diagonal

    # load q, k, v, do on-chip
    # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
    if EVEN_M & EVEN_HEADDIM:
        q = tl.load(q_ptrs)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[None, :, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                other=0.0,
            )
    # recompute p = softmax(qk, dim=-1).T

    q = q.reshape([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM])

    qk = tl.dot(q, tl.trans(k))

    qk = qk.reshape(QUERY_HEAD_GROUPS, BLOCK, BLOCK)

    mask = offs_m[:, None] >= offs_n[None, :]

    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        mask &= offs_n[None, :] < seqlen_k

    if SLIDING:
        mask &= (offs_n[None, :] - offs_m[:, None]) < BLOCK

    qk = tl.where(mask, qk, float("-inf"))

    qk = qk.reshape(QUERY_HEAD_GROUPS * BLOCK, BLOCK)

    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    lse_i = tl.load(LSE + offs_d_or_lse)
    lse_i = lse_i.reshape(QUERY_HEAD_GROUPS * BLOCK)

    p = tl.exp(qk * softmax_scale - lse_i[:, None])

    # compute dv
    # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
    # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
    # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
    # the output is correct.
    if EVEN_M & EVEN_HEADDIM:
        do = tl.load(do_ptrs)
    else:
        # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
        do = tl.load(
            do_ptrs,
            mask = (offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
            other = 0.0,
        )

    do = do.reshape(QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM)

    dv += tl.dot(tl.trans(p.to(do.dtype)), do)

    # compute dp = dot(v, do)
    # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
    # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
    # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    dp = tl.dot(do, tl.trans(v))

    # There's a race condition for headdim=48
    if not EVEN_HEADDIM:
        tl.debug_barrier()

    # compute ds = p * (dp - delta[:, None])
    # Putting the subtraction after the dp matmul (instead of before) is slightly faster

    Di = tl.load(D + offs_d_or_lse)
    Di = Di.reshape(QUERY_HEAD_GROUPS * BLOCK)

    # Converting ds to q.dtype here reduces register pressure and makes it much faster
    # for BLOCK_HEADDIM=128

    ds = (p * (dp - Di[:, None]) * softmax_scale)

    ds = ds.to(q.dtype)

    # compute dk = dot(ds.T, q)

    dk += tl.dot(tl.trans(ds), q)

    # compute dq

    if not (
        EVEN_M & EVEN_HEADDIM
    ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
        tl.debug_barrier()

    dq = tl.zeros([QUERY_HEAD_GROUPS * BLOCK, BLOCK_HEADDIM], dtype = tl.float32)

    dq += tl.dot(ds, k)

    # update dq

    dq = dq.reshape(QUERY_HEAD_GROUPS, BLOCK, BLOCK_HEADDIM)

    if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
        tl.atomic_add(dq_ptrs, dq, sem = 'relaxed')
    else:
        if EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq, mask=offs_m[None, :, None] < seqlen_q, sem = 'relaxed')
        else:
            tl.atomic_add(
                dq_ptrs,
                dq,
                mask = (offs_m[None, :, None] < seqlen_q) & (offs_d[None, None, :] < headdim),
                sem = 'relaxed',
            )

    # write-back

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])

    backward_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M = EVEN_M,
        EVEN_N = EVEN_N,
        EVEN_HEADDIM = EVEN_HEADDIM,
    )

@triton.heuristics(
    dict(
        QUERY_EXPAND_DIM = lambda args: 16 // args['QUERY_HEAD_GROUPS']
    )
)
@triton.jit
def backward_kernel(
    Q,
    K,
    V,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    stride_kvbl_b,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_lse_b,
    stride_D_b,
    kv_heads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    QUERY_EXPAND_DIM: tl.constexpr,
    RETURN_SEL_GRADS: tl.constexpr,
    INCLUDE_BLOCK_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    BLOCK_DV_USE_DOT: tl.constexpr,
    BLOCK_DK_USE_DOT: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // kv_heads
    off_h = off_hb % kv_heads
    off_qh = off_h * QUERY_HEAD_GROUPS

    OFF_SEL_KV_BLOCKS = tl.program_id(0) - int(INCLUDE_BLOCK_CAUSAL)
    IS_CAUSAL = INCLUDE_BLOCK_CAUSAL and tl.program_id(0) == 0

    # offset pointers for batch/head

    Q += off_b * stride_qb + off_qh * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_qh * stride_doh
    DQ += off_b * stride_dqb + off_qh * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh

    # offset pointers for batch/head for selected kv block related

    kv_block_indices += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_mask += off_b * stride_kvbl_b + off_h * stride_kvbl_h
    kv_block_grads += off_b * stride_kvbl_b + off_h * stride_kvbl_h

    # pointer to row-wise quantities in value-like data

    D += (
        off_b * stride_D_b +
        off_qh * seqlen_q_rounded
    )

    LSE += (
        off_b * stride_lse_b +
        off_qh * seqlen_q_rounded
    )

    num_block_n = tl.cdiv(seqlen_k, BLOCK)

    if IS_CAUSAL:
        for start_n in range(0, num_block_n):
            backward_kernel_one_col_block_causal(
                start_n,
                Q,
                K,
                V,
                kv_block_indices,
                kv_block_mask,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                stride_kvbl_m,
                stride_qh,
                stride_doh,
                stride_dqh,
                seqlen_q,
                seqlen_k,
                seqlen_q_rounded,
                headdim,
                BLOCK_HEADDIM = BLOCK_HEADDIM,
                EVEN_M = EVEN_M,
                EVEN_N = EVEN_N,
                EVEN_HEADDIM = EVEN_HEADDIM,
                BLOCK = BLOCK,
                QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
                SLIDING = SLIDING
            )
    else:
        for start_n in range(0, num_block_n):
            backward_kernel_one_col_block_sparse(
                start_n,
                Q,
                K,
                V,
                kv_block_indices,
                kv_block_mask,
                kv_block_grads,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                stride_kvbl_m,
                stride_qh,
                stride_doh,
                stride_dqh,
                seqlen_q,
                seqlen_k,
                seqlen_q_rounded,
                headdim,
                BLOCK_HEADDIM = BLOCK_HEADDIM,
                EVEN_M = EVEN_M,
                EVEN_N = EVEN_N,
                EVEN_HEADDIM = EVEN_HEADDIM,
                BLOCK = BLOCK,
                QUERY_HEAD_GROUPS = QUERY_HEAD_GROUPS,
                QUERY_EXPAND_DIM = QUERY_EXPAND_DIM,
                RETURN_SEL_GRADS = RETURN_SEL_GRADS,
                OFF_SEL_KV_BLOCKS = OFF_SEL_KV_BLOCKS,
                BLOCK_DV_USE_DOT = BLOCK_DV_USE_DOT,
                BLOCK_DK_USE_DOT = BLOCK_DK_USE_DOT,
            )

def native_sparse_attn_backward(
    do,
    q, k, v,
    kv_block_indices,
    kv_block_mask,
    kv_block_grads,
    o,
    lse,
    dq, dk, dv,
    block_size = 128,
    include_block_causal = True,
    return_sel_grads = False,
    sliding = False,
    block_dk_dv_use_dot = None
):
    device = do.device

    # Make sure that the last dimension is contiguous
    if not is_contiguous(do):
        do = do.contiguous()

    batch, q_heads, seqlen_q, dim = q.shape

    _, kv_heads, seqlen_k, _ = k.shape
    assert divisible_by(q_heads, kv_heads)
    head_groups = q_heads // kv_heads
    assert divisible_by(16, head_groups)

    num_sel_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    # assert d in {16, 32, 64, 128}
    assert dim <= 128
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    assert lse.shape == (batch, q_heads, seqlen_q_rounded)
    assert all([is_contiguous(t) for t in (q, k, v, o, dq, dk, dv)])

    softmax_scale = dim ** -0.5

    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)

    delta = torch.empty_like(lse)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * q_heads)

    backward_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        q_heads,
        seqlen_q,
        seqlen_q_rounded,
        dim,
        BLOCK = block_size,
        BLOCK_HEADDIM = BLOCK_HEADDIM,
    )

    grid = lambda META: (
        num_sel_fine_blocks + int(include_block_causal),
        batch * kv_heads
    )

    backward_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        kv_block_grads,
        do,
        dq,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        kv_block_indices.stride(0),
        kv_block_indices.stride(1),
        kv_block_indices.stride(2),
        lse.stride(0),
        delta.stride(0),
        kv_heads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        BLOCK_HEADDIM,
        BLOCK = block_size,
        QUERY_HEAD_GROUPS = head_groups,
        EVEN_M = divisible_by(seqlen_q, block_size),
        EVEN_N = divisible_by(seqlen_k, block_size),
        EVEN_HEADDIM = BLOCK_HEADDIM == dim,
        RETURN_SEL_GRADS = return_sel_grads,
        INCLUDE_BLOCK_CAUSAL = include_block_causal,
        SLIDING = sliding,
        BLOCK_DV_USE_DOT = default(block_dk_dv_use_dot, head_groups > 1),
        BLOCK_DK_USE_DOT = default(block_dk_dv_use_dot, head_groups > 1)
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )

    return delta

# native sparse attention function

from torch.autograd import Function

class NSA(Function):

    @classmethod
    def forward(
        self,
        ctx,
        fq, fk, fv,
        block_size,
        selected_block_indices,
        fmask,
        sel_scale,
        include_block_causal,
        block_dk_dv_use_dot
    ):
        dtype = fq.dtype

        q_heads, kv_heads = fq.shape[1], fk.shape[1]
        assert divisible_by(q_heads, kv_heads)
        head_groups = q_heads // kv_heads

        fq, fk, fv = tuple(t.half() for t in (fq, fk, fv))

        out, lse = native_sparse_attn_forward(
            fq, fk, fv,
            selected_block_indices,
            fmask,
            block_size = block_size,
            include_block_causal = include_block_causal,
        )

        ctx.save_for_backward(fq, fk, fv, selected_block_indices, fmask, out, lse)

        return_sel_grads = exists(sel_scale)

        if return_sel_grads:
            assert (sel_scale == 1.).all(), 'for now, must be straight through as multiplier of 1.'

        ctx._saved_variables = (
            block_size,
            head_groups,
            return_sel_grads,
            include_block_causal,
            block_dk_dv_use_dot
        )

        return out.type(dtype), lse

    @classmethod
    def backward(self, ctx, do, _):
        device = do.device

        q, k, v, sel_block_indices, mask, out, lse = ctx.saved_tensors

        (
            block_size,
            head_groups,
            return_sel_grads,
            include_block_causal,
            block_dk_dv_use_dot
        ) = ctx._saved_variables

        do = do.half()
        dq = torch.zeros(q.shape, dtype = torch.float32, device = device)
        dk = torch.zeros(k.shape, dtype = torch.float32, device = device)
        dv = torch.zeros(v.shape, dtype = torch.float32, device = device)

        sel_grads = torch.zeros_like(sel_block_indices).float()

        native_sparse_attn_backward(
            do, q, k, v,
            sel_block_indices, mask, sel_grads,
            out, lse, dq, dk, dv,
            block_size = block_size,
            include_block_causal = include_block_causal,
            return_sel_grads = return_sel_grads,
            block_dk_dv_use_dot = block_dk_dv_use_dot
        )
    
        ret_sel_grads = None

        if return_sel_grads:
            ret_sel_grads = sel_grads

        return dq, dk, dv, None, None, None, ret_sel_grads, None, None

_native_sparse_attend = NSA.apply

# ein notation

# b - batch
# qh - query heads
# kh - key / value heads
# n - token sequence
# d - attention head dimension
# sel - selected indices

def native_sparse_attend(
    fq: Float['b qh n d'],
    fk: Float['b kh n d'],
    fv: Float['b kh n d'],
    block_size: int,
    selected_block_indices: Int['b qh n sel'] | Int['b kh n sel'],
    fmask: Bool['b qh n sel'] | Bool['b kh n sel'],
    sel_scale: Float['b kh n sel'] | Float['b qh n sel'] | None = None,
    include_block_causal = True,
    return_lse = False,
    block_dk_dv_use_dot = False
):
    seq_len = fq.shape[-2]
    q_heads, kv_heads, sel_heads = fq.shape[1], fk.shape[1], selected_block_indices.shape[1]

    assert divisible_by(q_heads, kv_heads)
    assert sel_heads in (q_heads, kv_heads)

    assert block_size >= 16, 'fine selection block size must be 16 or greater for now'

    # query heads within each group to attend to different segments

    if kv_heads != sel_heads:
        fk, fv = tuple(repeat(t, 'b h ... -> b (h gh) ...', gh = q_heads // kv_heads) for t in (fk, fv))

    out, lse = _native_sparse_attend(
        fq, fk, fv,
        block_size,
        selected_block_indices,
        fmask,
        sel_scale,
        include_block_causal,
        block_dk_dv_use_dot
    )

    if not return_lse:
        return out
    
    return out, lse[..., :seq_len]

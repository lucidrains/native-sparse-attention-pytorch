# taken from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# with fixes for triton 2.3

from functools import partial
import math
from math import ceil

import torch
from torch import Tensor

from einops import repeat, rearrange

def exists(v):
    return v is not None

def default(val, d):
    return val if exists(val) else d

def round_up_multiple(n, mult):
    return ceil(n / mult) * mult

def is_contiguous(x: Tensor):
    return x.stride(-1) == 1

TRITON_BLOCK_SIZE = 128 # some block size that allows triton not to break, at least half a year ago

INSTALL_COMMAND = 'pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly'

# make sure triton 2.1+ is installed

import packaging.version as pkg_version

import importlib
from importlib.metadata import version

try:
    triton_version = version('triton-nightly')
except:
    print(f'latest triton must be installed. `{INSTALL_COMMAND}` first')
    exit()

assert pkg_version.parse(triton_version) >= pkg_version.parse('3.0.0'), f'triton must be version 3.0.0 or above. `{INSTALL_COMMAND}` to upgrade'

import triton
import triton.language as tl
from triton.language.extra import libdevice

# kernels

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    KV_block_indices,
    KV_block_mask,
    Out,
    M,
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
    nheads,
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
    NUM_SEL_KV_BLOCKS: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])
    )

    # maximum

    m_ptrs = M + off_hb * seqlen_q_rounded + offs_m

    m_i = tl.zeros([BLOCK], dtype = tl.float32) - float("inf")

    # lse

    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m

    lse_i = tl.zeros([BLOCK], dtype = tl.float32) - float("inf")

    # output

    offs_d = tl.arange(0, BLOCK_HEADDIM)

    out_ptrs = (
        Out
        + off_b * stride_ob
        + off_h * stride_oh
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )

    acc_o = tl.zeros([BLOCK, BLOCK_HEADDIM], dtype = tl.float32)

    # load queries, keys, values

    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
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
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
            )
        else:
            k = tl.load(
                k_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    qk = tl.zeros([BLOCK, BLOCK], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))

    if not EVEN_N:
        qk += tl.where(offs_n[None, :] < seqlen_k, 0, float("-inf"))

    qk += tl.where(offs_m[:, None] >= offs_n[None, :], 0, float("-inf"))

    m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
    p = tl.exp(qk * softmax_scale - m_ij[:, None])

    l_ij = tl.sum(p, 1)

    acc_o_scale = tl.exp(m_i - m_ij)
    acc_o = acc_o * acc_o_scale[:, None]

    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            v = tl.load(v_ptrs)
        else:
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            v = tl.load(
                v_ptrs,
                mask=offs_n[:, None] < seqlen_k,
                other=0.0,
            )
        else:
            v = tl.load(
                v_ptrs,
                mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                other=0.0,
            )

    p = p.to(v.dtype)
    acc_o += tl.dot(p, v)

    # -- update statistics

    m_i = m_ij
    l_i_new = tl.exp(lse_i - m_ij) + l_ij
    lse_i = m_ij + tl.log(l_i_new)

    # normalize accumulated out

    acc_o_scale = tl.exp(m_i - lse_i)
    acc_o = acc_o * acc_o_scale[:, None]

    # offsets for m and lse

    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)

    # write back lse and m

    tl.store(lse_ptrs, lse_i)

    # write to output

    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

def flash_attn_forward(
    q,
    k,
    v,
    kv_block_indices,
    kv_block_mask,
    block_size = 128
):
    q, k, v = [x if is_contiguous(x) else x.contiguous() for x in (q, k, v)]

    batch, seqlen_q, nheads, dim = q.shape
    _, seqlen_k, _, _ = k.shape

    num_selected_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    assert k.shape == (batch, seqlen_k, nheads, dim)
    assert v.shape == (batch, seqlen_k, nheads, dim)
    assert dim <= 128, "only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    softmax_scale = dim ** -0.5

    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    m = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)

    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)
    num_warps = 4 if dim <= 64 else 8
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        o,
        m,
        lse,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        kv_block_indices.stride(0),
        kv_block_indices.stride(2),
        kv_block_indices.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        dim,
        seqlen_q // 32,
        seqlen_k // 32,
        BLOCK_HEADDIM,
        BLOCK = block_size,
        NUM_SEL_KV_BLOCKS = num_selected_fine_blocks,
        num_warps = num_warps,
        num_stages = 1,
    )

    return o, lse

@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # initialize offsets

    offs_m = start_m * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # load

    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)

    # write-back

    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)

@triton.jit
def _bwd_store_dk_dv(
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
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
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
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = ((start_n * BLOCK) // BLOCK) * BLOCK
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK)
    offs_n = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_m = start_n * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])

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
        _bwd_store_dk_dv(
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
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs,
                mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
    # recompute p = softmax(qk, dim=-1).T
    qk = tl.dot(q, tl.trans(k))

    # Trying to combine the two masks seem to make the result wrong
    if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
        qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))

    qk = tl.where(offs_m[:, None] >= (offs_n[None, :]), qk, float("-inf"))

    # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
    # Also wrong for headdim=64.

    if not (EVEN_M & EVEN_HEADDIM):
        tl.debug_barrier()

    lse_i = tl.load(LSE + offs_m)

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
            mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
            other=0.0,
        )
    # if EVEN_M:
    #     if EVEN_HEADDIM:
    #         do = tl.load(do_ptrs)
    #     else:
    #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    # else:
    #     if EVEN_HEADDIM:
    #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
    #     else:
    #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
    #                                    & (offs_d[None, :] < headdim), other=0.0)
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

    Di = tl.load(D + offs_m)

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
    if not ATOMIC_ADD:
        if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
            dq = tl.load(dq_ptrs, eviction_policy="evict_last")
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq, eviction_policy="evict_last")
        else:
            if EVEN_HEADDIM:
                dq = tl.load(
                    dq_ptrs,
                    mask=offs_m[:, None] < seqlen_q,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                dq += tl.dot(ds, k)
                tl.store(
                    dq_ptrs,
                    dq,
                    mask=offs_m[:, None] < seqlen_q,
                    eviction_policy="evict_last",
                )
            else:
                dq = tl.load(
                    dq_ptrs,
                    mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                    eviction_policy="evict_last",
                )
                dq += tl.dot(ds, k)
                tl.store(
                    dq_ptrs,
                    dq,
                    mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    eviction_policy="evict_last",
                )
    else:  # If we're parallelizing across the seqlen_k dimension
        dq = tl.dot(ds, k)
        if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
            tl.atomic_add(dq_ptrs, dq, sem = 'relaxed')
        else:
            if EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq, mask=offs_m[:, None] < seqlen_q, sem = 'relaxed')
            else:
                tl.atomic_add(
                    dq_ptrs,
                    dq,
                    mask = (offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    sem = 'relaxed',
                )

    # # increment pointers
    # dq_ptrs += BLOCK * stride_dqm
    # q_ptrs += BLOCK * stride_qm
    # do_ptrs += BLOCK * stride_dom

    # write-back

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
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

@triton.jit
def _bwd_kernel(
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
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_SEL_KV_BLOCKS: tl.constexpr
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
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
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD = False,
                BLOCK_HEADDIM = BLOCK_HEADDIM,
                EVEN_M = EVEN_M,
                EVEN_N = EVEN_N,
                EVEN_HEADDIM = EVEN_HEADDIM,
                BLOCK = BLOCK,
                NUM_SEL_KV_BLOCKS = NUM_SEL_KV_BLOCKS
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
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
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD = True,
            BLOCK_HEADDIM = BLOCK_HEADDIM,
            EVEN_M = EVEN_M,
            EVEN_N = EVEN_N,
            EVEN_HEADDIM = EVEN_HEADDIM,
            BLOCK = BLOCK,
            NUM_SEL_KV_BLOCKS = NUM_SEL_KV_BLOCKS
        )

def flash_attn_backward(
    do,
    q, k, v,
    kv_block_indices,
    kv_block_mask,
    o,
    lse,
    dq, dk, dv,
    block_size = 128
):
    # Make sure that the last dimension is contiguous
    if not is_contiguous(do):
        do = do.contiguous()

    batch, seqlen_q, nheads, dim = q.shape
    _, seqlen_k, _, _ = k.shape

    num_sel_fine_blocks = kv_block_indices.shape[-1]
    assert kv_block_indices.shape == kv_block_mask.shape

    # assert d in {16, 32, 64, 128}
    assert dim <= 128
    seqlen_q_rounded = round_up_multiple(seqlen_q, TRITON_BLOCK_SIZE)

    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert all([is_contiguous(t) for t in (q, k, v, o, dq, dk, dv)])

    softmax_scale = dim ** -0.5

    dq_accum = torch.empty_like(q, dtype = torch.float32)

    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(dim), 16)

    delta = torch.empty_like(lse)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK"]), batch * nheads)

    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        dim,
        BLOCK = block_size,
        BLOCK_HEADDIM = BLOCK_HEADDIM,
    )

    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        kv_block_indices,
        kv_block_mask,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
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
        NUM_SEL_KV_BLOCKS = num_sel_fine_blocks,
        SEQUENCE_PARALLEL = False,
        EVEN_M = (seqlen_q % block_size) == 0,
        EVEN_N = (seqlen_k % block_size) == 0,
        EVEN_HEADDIM = BLOCK_HEADDIM == dim
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)

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
        num_grouped_queries
    ):
        selected_block_indices, fmask = tuple(rearrange(t, 'b h i sel -> b i h sel') for t in (selected_block_indices, fmask))

        fq, fk, fv = tuple(rearrange(t, 'b h n d -> b n h d') for t in (fq, fk, fv))

        dtype = fq.dtype

        fq, fk, fv = tuple(t.half() for t in (fq, fk, fv))

        out, lse = flash_attn_forward(
            fq, fk, fv,
            selected_block_indices,
            fmask,
            block_size = block_size
        )

        ctx.save_for_backward(fq, fk, fv, selected_block_indices, fmask, out, lse)
        ctx._saved_variables = (block_size,)

        out = rearrange(out, 'b n h d -> b h n d')
        return out.type(dtype)

    @classmethod
    def backward(self, ctx, do):
        do = rearrange(do, 'b h n d -> b n h d')

        q, k, v, sel_block_indices, mask, out, lse = ctx.saved_tensors

        (
            block_size,
        ) = ctx._saved_variables

        do = do.half()
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        flash_attn_backward(
            do, q, k, v,
            sel_block_indices, mask,
            out, lse, dq, dk, dv,
            block_size = block_size
        )

        dq, dk, dv = tuple(rearrange(t, 'b n h d -> b h n d') for t in (dq, dk, dv))
        return dq, dk, dv, None, None, None, None

native_sparse_attend = NSA.apply

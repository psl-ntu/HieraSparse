import argparse
import itertools
from typing import List, Optional, Tuple, Union

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import triton
from tilelang.autotuner import *
from tilelang.engine.param import KernelParam
from tilelang.intrinsics.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter
from torch.nn.attention import SDPBackend, sdpa_kernel

from hierasparse.kernels.configs import (
    acc_t_B_mapping_inv,
    acc_t_C_mapping,
    flashattn_sp_tune_configs,
    pass_configs,
)
from hierasparse.utils import round


def get_configs(params):
    return [dict(zip(params, values)) for values in itertools.product(*params.values())]


tune_inputs: Optional[Tuple[torch.Tensor]] = None


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    return tune_inputs


@autotune(
    configs=get_configs(
        {
            "block_M": [32 * i for i in range(1, 5)],
            "threads": [128],
            "use_movmatrix": [False],
        }
    ),
    warmup=250,
    rep=1000,
    supply_prog=supply_prog,
)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: False,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def blockattn_sp_mk_mv_unoptimized(
    batch,
    heads,
    groups,
    dim,
    block_N,
    is_causal,
    block_M,
    threads,
    use_movmatrix,
    chunked_prefill: bool = False,
):
    seq_q = T.dynamic("seq_q")
    dummy0 = T.dynamic("dummy0")
    dummy1 = T.dynamic("dummy1")
    k_dense_blocks = T.dynamic("k_dense_blocks")
    k_sparse_blocks = T.dynamic("k_sparse_blocks")
    v_dense_blocks = T.dynamic("v_dense_blocks")
    v_sparse_blocks = T.dynamic("v_sparse_blocks")

    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    idx_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    assert heads % groups == 0

    warps = threads // 32
    mma_policy = T.GemmWarpPolicy.FullCol
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_M if mma_policy == T.GemmWarpPolicy.FullRow else block_M // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]
    # Key
    k_dense_blocks_shape = [batch, groups, k_dense_blocks, block_N, dim]
    k_sparse_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // 2]
    k_meta_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // e_factor]
    k_shared_shape = [block_N, dim]
    k_sp_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]
    # key page idx map
    k_page_idx_shape = [batch, groups, dummy0]
    # Value
    v_dense_blocks_shape = [batch, groups, v_dense_blocks, block_N, dim]
    v_sparse_blocks_shape = [batch, groups, v_sparse_blocks, block_N // 2, dim]
    v_meta_blocks_shape = [batch, groups, v_sparse_blocks, block_N // e_factor, dim]
    v_dense_shared_shape = [block_N, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]
    # value page idx map
    v_page_idx_shape = [batch, groups, dummy1]

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_dense_blocks_shape=} {k_sparse_blocks_shape=} {k_meta_blocks_shape=}")
    print(f"{v_dense_blocks_shape=} {v_sparse_blocks_shape=} {v_meta_blocks_shape=}")

    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def LOAD_K_DENSE(
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], K_dense_shared)

    @T.macro
    def LOAD_K_SP(
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], K_SP_shared)
        T.copy(K_E_blocks[bz, cur_kv_head, k_page_idx, :, :], K_E_shared)

    @T.macro
    def MMA0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm(K_dense_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

    @T.macro
    def MMASP0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            K_SP_shared,
            K_E_shared,
            Q_shared,
            acc_s_T,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=True,
        )

    @T.macro
    def LOAD_V_DENSE(
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], V_dense_shared)

    @T.macro
    def LOAD_V_SP(
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], V_SP_shared)
        T.copy(V_E_blocks[bz, cur_kv_head, k_page_idx, :, :], V_E_shared)

    @T.macro
    def MMA1_DENSE(
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm(
            V_dense_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def MMA1_SP(
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            V_SP_shared,
            V_E_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            transpose_E=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_M], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o_T[j, i] *= scores_scale[i]

    @T.prim_func
    def blockattn_sp_mk_mv_kernel(
        Q: T.Tensor(q_shape, dtype),
        seq_kv: T.int32,
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_dense_shared = T.alloc_shared(k_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)
            acc_s_T = T.alloc_fragment([block_N, block_M], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_M], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_M], dtype)
            acc_o_T = T.alloc_fragment([dim, block_M], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            next_k_page_idx = K_page_idx[bz, cur_kv_head, 0]
            # with T.attr("default", "async_scope", 1):
            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            if next_k_page_idx > 0:
                LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
            else:
                LOAD_K_SP(K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bz)
            # T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if chunked_prefill:
                assert is_causal
                loop_range = T.ceildiv(seq_kv - seq_q, block_N) + T.min(
                    T.ceildiv(seq_q, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
            else:
                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

            for k in T.serial(loop_range):
                k_page_idx = K_page_idx[bz, cur_kv_head, k]
                v_page_idx = V_page_idx[bz, cur_kv_head, k]

                # T.ptx_wait_group(0)
                # with T.attr("default", "async_scope", 1):
                if v_page_idx > 0:
                    LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bz)
                else:
                    LOAD_V_SP(V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bz)
                # T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(Q_shared, K_SP_shared, K_E_shared, acc_s_T)

                # T.ptx_wait_group(0)

                if k < loop_range - 1:
                    next_k_page_idx = K_page_idx[bz, cur_kv_head, k + 1]
                    # with T.attr("default", "async_scope", 1):
                    if next_k_page_idx > 0:
                        LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                    else:
                        LOAD_K_SP(
                            K_sparse_blocks,
                            K_E_blocks,
                            K_SP_shared,
                            K_E_shared,
                            -next_k_page_idx - 1,
                            cur_kv_head,
                            bz,
                        )
                    # T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s_T[j, i] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s_T[j, i], -T.infinity(acc_s_T.dtype)
                            )
                        else:
                            acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o_T, scores_scale)
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(V_SP_shared, V_E_shared, acc_s_cast_T, acc_o_T)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return blockattn_sp_mk_mv_kernel


@autotune(
    configs=get_configs(
        {
            "block_M": [32 * i for i in range(1, 5)],
            "threads": [128],
            "use_movmatrix": [False],
        }
    ),
    warmup=250,
    rep=1000,
    supply_prog=supply_prog,
)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: False,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def blockattn_sp_mk_mv_pipelined(
    batch,
    heads,
    groups,
    dim,
    block_N,
    is_causal,
    block_M,
    threads,
    use_movmatrix,
    chunked_prefill: bool = False,
):
    seq_q = T.dynamic("seq_q")
    dummy0 = T.dynamic("dummy0")
    dummy1 = T.dynamic("dummy1")
    k_dense_blocks = T.dynamic("k_dense_blocks")
    k_sparse_blocks = T.dynamic("k_sparse_blocks")
    v_dense_blocks = T.dynamic("v_dense_blocks")
    v_sparse_blocks = T.dynamic("v_sparse_blocks")

    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    idx_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    assert heads % groups == 0

    warps = threads // 32
    mma_policy = T.GemmWarpPolicy.FullCol
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_M if mma_policy == T.GemmWarpPolicy.FullRow else block_M // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]
    # Key
    k_dense_blocks_shape = [batch, groups, k_dense_blocks, block_N, dim]
    k_sparse_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // 2]
    k_meta_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // e_factor]
    k_shared_shape = [block_N, dim]
    k_sp_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]
    # key page idx map
    k_page_idx_shape = [batch, groups, dummy0]
    # Value
    v_dense_blocks_shape = [batch, groups, v_dense_blocks, block_N, dim]
    v_sparse_blocks_shape = [batch, groups, v_sparse_blocks, block_N // 2, dim]
    v_meta_blocks_shape = [batch, groups, v_sparse_blocks, block_N // e_factor, dim]
    v_dense_shared_shape = [block_N, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]
    # value page idx map
    v_page_idx_shape = [batch, groups, dummy1]

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_dense_blocks_shape=} {k_sparse_blocks_shape=} {k_meta_blocks_shape=}")
    print(f"{v_dense_blocks_shape=} {v_sparse_blocks_shape=} {v_meta_blocks_shape=}")

    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def LOAD_K_DENSE(
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], K_dense_shared)

    @T.macro
    def LOAD_K_SP(
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], K_SP_shared)
        T.copy(K_E_blocks[bz, cur_kv_head, k_page_idx, :, :], K_E_shared)

    @T.macro
    def MMA0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm(K_dense_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

    @T.macro
    def MMASP0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            K_SP_shared,
            K_E_shared,
            Q_shared,
            acc_s_T,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=True,
        )

    @T.macro
    def LOAD_V_DENSE(
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], V_dense_shared)

    @T.macro
    def LOAD_V_SP(
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], V_SP_shared)
        T.copy(V_E_blocks[bz, cur_kv_head, k_page_idx, :, :], V_E_shared)

    @T.macro
    def MMA1_DENSE(
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm(
            V_dense_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def MMA1_SP(
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            V_SP_shared,
            V_E_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            transpose_E=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_M], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o_T[j, i] *= scores_scale[i]

    @T.prim_func
    def blockattn_sp_mk_mv_kernel(
        Q: T.Tensor(q_shape, dtype),
        seq_kv: T.int32,
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_dense_shared = T.alloc_shared(k_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)
            acc_s_T = T.alloc_fragment([block_N, block_M], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_M], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_M], dtype)
            acc_o_T = T.alloc_fragment([dim, block_M], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            next_k_page_idx = K_page_idx[bz, cur_kv_head, 0]
            with T.attr("default", "async_scope", 1):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                if next_k_page_idx > 0:
                    LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                else:
                    LOAD_K_SP(
                        K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bz
                    )
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if chunked_prefill:
                assert is_causal
                loop_range = T.ceildiv(seq_kv - seq_q, block_N) + T.min(
                    T.ceildiv(seq_q, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
            else:
                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

            for k in T.serial(loop_range):
                k_page_idx = K_page_idx[bz, cur_kv_head, k]
                v_page_idx = V_page_idx[bz, cur_kv_head, k]

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    if v_page_idx > 0:
                        LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bz)
                    else:
                        LOAD_V_SP(
                            V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bz
                        )
                T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(Q_shared, K_SP_shared, K_E_shared, acc_s_T)

                T.ptx_wait_group(0)

                if k < loop_range - 1:
                    next_k_page_idx = K_page_idx[bz, cur_kv_head, k + 1]
                    with T.attr("default", "async_scope", 1):
                        if next_k_page_idx > 0:
                            LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                        else:
                            LOAD_K_SP(
                                K_sparse_blocks,
                                K_E_blocks,
                                K_SP_shared,
                                K_E_shared,
                                -next_k_page_idx - 1,
                                cur_kv_head,
                                bz,
                            )
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s_T[j, i] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s_T[j, i], -T.infinity(acc_s_T.dtype)
                            )
                        else:
                            acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o_T, scores_scale)
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(V_SP_shared, V_E_shared, acc_s_cast_T, acc_o_T)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return blockattn_sp_mk_mv_kernel


@autotune(
    configs=get_configs(
        {
            "block_M": [32 * i for i in range(1, 5)],
            "threads": [128],
            "use_movmatrix": [True],
        }
    ),
    warmup=250,
    rep=1000,
    supply_prog=supply_prog,
)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: False,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: False,
    },
)
def blockattn_sp_mk_mv_pipelined_movmatrix(
    batch,
    heads,
    groups,
    dim,
    block_N,
    is_causal,
    block_M,
    threads,
    use_movmatrix,
    chunked_prefill: bool = False,
):
    seq_q = T.dynamic("seq_q")
    dummy0 = T.dynamic("dummy0")
    dummy1 = T.dynamic("dummy1")
    k_dense_blocks = T.dynamic("k_dense_blocks")
    k_sparse_blocks = T.dynamic("k_sparse_blocks")
    v_dense_blocks = T.dynamic("v_dense_blocks")
    v_sparse_blocks = T.dynamic("v_sparse_blocks")

    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    idx_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    assert heads % groups == 0

    warps = threads // 32
    mma_policy = T.GemmWarpPolicy.FullCol
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_M if mma_policy == T.GemmWarpPolicy.FullRow else block_M // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]
    # Key
    k_dense_blocks_shape = [batch, groups, k_dense_blocks, block_N, dim]
    k_sparse_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // 2]
    k_meta_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // e_factor]
    k_shared_shape = [block_N, dim]
    k_sp_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]
    # key page idx map
    k_page_idx_shape = [batch, groups, dummy0]
    # Value
    v_dense_blocks_shape = [batch, groups, v_dense_blocks, block_N, dim]
    v_sparse_blocks_shape = [batch, groups, v_sparse_blocks, block_N // 2, dim]
    v_meta_blocks_shape = [batch, groups, v_sparse_blocks, block_N // e_factor, dim]
    v_dense_shared_shape = [block_N, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]
    # value page idx map
    v_page_idx_shape = [batch, groups, dummy1]

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_dense_blocks_shape=} {k_sparse_blocks_shape=} {k_meta_blocks_shape=}")
    print(f"{v_dense_blocks_shape=} {v_sparse_blocks_shape=} {v_meta_blocks_shape=}")

    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def LOAD_K_DENSE(
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], K_dense_shared)

    @T.macro
    def LOAD_K_SP(
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], K_SP_shared)
        T.copy(K_E_blocks[bz, cur_kv_head, k_page_idx, :, :], K_E_shared)

    @T.macro
    def MMA0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm(K_dense_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

    @T.macro
    def MMASP0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            K_SP_shared,
            K_E_shared,
            Q_shared,
            acc_s_T,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=True,
        )

    @T.macro
    def LOAD_V_DENSE(
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], V_dense_shared)

    @T.macro
    def LOAD_V_SP(
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], V_SP_shared)
        T.copy(V_E_blocks[bz, cur_kv_head, k_page_idx, :, :], V_E_shared)

    @T.macro
    def MMA1_DENSE(
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm(
            V_dense_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def MMA1_SP(
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            V_SP_shared,
            V_E_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            transpose_E=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_M], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o_T[j, i] *= scores_scale[i]

    @T.prim_func
    def blockattn_sp_mk_mv_kernel(
        Q: T.Tensor(q_shape, dtype),
        seq_kv: T.int32,
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_dense_shared = T.alloc_shared(k_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)
            acc_s_T = T.alloc_fragment([block_N, block_M], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_M], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_M], dtype)
            acc_o_T = T.alloc_fragment([dim, block_M], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            next_k_page_idx = K_page_idx[bz, cur_kv_head, 0]
            with T.attr("default", "async_scope", 1):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                if next_k_page_idx > 0:
                    LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                else:
                    LOAD_K_SP(
                        K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bz
                    )
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if chunked_prefill:
                assert is_causal
                loop_range = T.ceildiv(seq_kv - seq_q, block_N) + T.min(
                    T.ceildiv(seq_q, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
            else:
                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

            for k in T.serial(loop_range):
                k_page_idx = K_page_idx[bz, cur_kv_head, k]
                v_page_idx = V_page_idx[bz, cur_kv_head, k]

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    if v_page_idx > 0:
                        LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bz)
                    else:
                        LOAD_V_SP(
                            V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bz
                        )
                T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(Q_shared, K_SP_shared, K_E_shared, acc_s_T)

                T.ptx_wait_group(0)

                if k < loop_range - 1:
                    next_k_page_idx = K_page_idx[bz, cur_kv_head, k + 1]
                    with T.attr("default", "async_scope", 1):
                        if next_k_page_idx > 0:
                            LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                        else:
                            LOAD_K_SP(
                                K_sparse_blocks,
                                K_E_blocks,
                                K_SP_shared,
                                K_E_shared,
                                -next_k_page_idx - 1,
                                cur_kv_head,
                                bz,
                            )
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s_T[j, i] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s_T[j, i], -T.infinity(acc_s_T.dtype)
                            )
                        else:
                            acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o_T, scores_scale)
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(V_SP_shared, V_E_shared, acc_s_cast_T, acc_o_T)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return blockattn_sp_mk_mv_kernel


@autotune(
    configs=get_configs(
        {
            "block_M": [32 * i for i in range(1, 5)],
            "threads": [128],
            "use_movmatrix": [True],
        }
    ),
    warmup=250,
    rep=1000,
    supply_prog=supply_prog,
)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: False,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def blockattn_sp_mk_mv_pipelined_movmatrix_merge(
    batch,
    heads,
    groups,
    dim,
    block_N,
    is_causal,
    block_M,
    threads,
    use_movmatrix,
    chunked_prefill: bool = False,
):
    seq_q = T.dynamic("seq_q")
    dummy0 = T.dynamic("dummy0")
    dummy1 = T.dynamic("dummy1")
    k_dense_blocks = T.dynamic("k_dense_blocks")
    k_sparse_blocks = T.dynamic("k_sparse_blocks")
    v_dense_blocks = T.dynamic("v_dense_blocks")
    v_sparse_blocks = T.dynamic("v_sparse_blocks")

    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    idx_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    assert heads % groups == 0

    warps = threads // 32
    mma_policy = T.GemmWarpPolicy.FullCol
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_M if mma_policy == T.GemmWarpPolicy.FullRow else block_M // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]
    # Key
    k_dense_blocks_shape = [batch, groups, k_dense_blocks, block_N, dim]
    k_sparse_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // 2]
    k_meta_blocks_shape = [batch, groups, k_sparse_blocks, block_N, dim // e_factor]
    k_shared_shape = [block_N, dim]
    k_sp_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]
    # key page idx map
    k_page_idx_shape = [batch, groups, dummy0]
    # Value
    v_dense_blocks_shape = [batch, groups, v_dense_blocks, block_N, dim]
    v_sparse_blocks_shape = [batch, groups, v_sparse_blocks, block_N // 2, dim]
    v_meta_blocks_shape = [batch, groups, v_sparse_blocks, block_N // e_factor, dim]
    v_dense_shared_shape = [block_N, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]
    # value page idx map
    v_page_idx_shape = [batch, groups, dummy1]

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_dense_blocks_shape=} {k_sparse_blocks_shape=} {k_meta_blocks_shape=}")
    print(f"{v_dense_blocks_shape=} {v_sparse_blocks_shape=} {v_meta_blocks_shape=}")

    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def LOAD_K_DENSE(
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], K_dense_shared)

    @T.macro
    def LOAD_K_SP(
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(K_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], K_SP_shared)
        T.copy(K_E_blocks[bz, cur_kv_head, k_page_idx, :, :], K_E_shared)

    @T.macro
    def MMA0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_shared_shape, dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm(K_dense_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

    @T.macro
    def MMASP0(
        Q_shared: T.SharedBuffer(q_shared_shape, dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            K_SP_shared,
            K_E_shared,
            Q_shared,
            acc_s_T,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullCol,
            clear_accum=True,
        )

    @T.macro
    def LOAD_V_DENSE(
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_dense_blocks[bz, cur_kv_head, k_page_idx, :, :], V_dense_shared)

    @T.macro
    def LOAD_V_SP(
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.copy(V_sparse_blocks[bz, cur_kv_head, k_page_idx, :, :], V_SP_shared)
        T.copy(V_E_blocks[bz, cur_kv_head, k_page_idx, :, :], V_E_shared)

    @T.macro
    def MMA1_DENSE(
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm(
            V_dense_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def MMA1_SP(
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
    ):
        T.gemm_sp_v2(
            V_SP_shared,
            V_E_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            transpose_E=True,
            policy=T.GemmWarpPolicy.FullCol,
        )

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_M], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_M], dtype)",
            "T.SharedBuffer([block_N, block_M], dtype)",
        ],
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o_T[j, i] *= scores_scale[i]

    @T.prim_func
    def blockattn_sp_mk_mv_kernel(
        Q: T.Tensor(q_shape, dtype),
        seq_kv: T.int32,
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_dense_shared = T.alloc_shared(k_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)
            acc_s_T = T.alloc_fragment([block_N, block_M], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_M], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_M], dtype)
            acc_o_T = T.alloc_fragment([dim, block_M], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            next_k_page_idx = K_page_idx[bz, cur_kv_head, 0]
            with T.attr("default", "async_scope", 1):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                if next_k_page_idx > 0:
                    LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                else:
                    LOAD_K_SP(
                        K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bz
                    )
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if chunked_prefill:
                assert is_causal
                loop_range = T.ceildiv(seq_kv - seq_q, block_N) + T.min(
                    T.ceildiv(seq_q, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
            else:
                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

            for k in T.serial(loop_range):
                k_page_idx = K_page_idx[bz, cur_kv_head, k]
                v_page_idx = V_page_idx[bz, cur_kv_head, k]

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    if v_page_idx > 0:
                        LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bz)
                    else:
                        LOAD_V_SP(
                            V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bz
                        )
                T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(Q_shared, K_SP_shared, K_E_shared, acc_s_T)

                T.ptx_wait_group(0)

                if k < loop_range - 1:
                    next_k_page_idx = K_page_idx[bz, cur_kv_head, k + 1]
                    with T.attr("default", "async_scope", 1):
                        if next_k_page_idx > 0:
                            LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bz)
                        else:
                            LOAD_K_SP(
                                K_sparse_blocks,
                                K_E_blocks,
                                K_SP_shared,
                                K_E_shared,
                                -next_k_page_idx - 1,
                                cur_kv_head,
                                bz,
                            )
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s_T[j, i] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s_T[j, i], -T.infinity(acc_s_T.dtype)
                            )
                        else:
                            acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o_T, scores_scale)
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(V_SP_shared, V_E_shared, acc_s_cast_T, acc_o_T)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return blockattn_sp_mk_mv_kernel


@autotune(configs=flashattn_sp_tune_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs=pass_configs(),
)
def flashattn_sp_kv(
    batch, heads, groups, dim, is_causal, block_M, block_N, threads, use_movmatrix, chunked_prefill=False
):
    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    seq_q = T.dynamic("seq_q")
    seq_kv = T.dynamic("seq_kv")
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    # seq_q = T.dynamic("seq_q")
    # seq_kv = T.dynamic("seq_kv")
    warps = threads // 32
    assert heads % groups == 0
    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]
    # Key
    k_shape = [batch, groups, seq_kv, dim // 2]
    k_e_shape = [batch, groups, seq_kv, dim // e_factor]
    k_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]
    # Value
    v_shape = [batch, groups, seq_kv // 2, dim]
    v_e_shape = [batch, groups, seq_kv // e_factor, dim]
    v_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]
    mma_policy = T.GemmWarpPolicy.FullCol

    assert mma_policy in [T.GemmWarpPolicy.FullCol, T.GemmWarpPolicy.FullRow]

    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_M if mma_policy == T.GemmWarpPolicy.FullRow else block_M // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    assert atom_row > 0 and atom_col > 0, f"{atom_row=} {atom_col=}"

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_shape=} {k_e_shape=} {k_shared_shape=} {k_e_shared_shape=}")
    print(f"{v_shape=} {v_e_shape=} {v_shared_shape=} {v_e_shared_shape=}")

    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def MMASP1(
        V: T.Tensor(v_shape, dtype),
        V_E: T.Tensor(v_e_shape, e_dtype),
        V_shared: T.SharedBuffer(v_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        acc_s_cast_T: T.SharedBuffer([block_N, block_M], dtype),
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        k: T.int32,
        cur_kv_head: T.int32,
        bz: T.int32,
    ):
        T.gemm_sp_v2(
            V_shared,
            V_E_shared,
            acc_s_cast_T,
            acc_o_T,
            transpose_A=True,
            transpose_E=True,
            policy=mma_policy,
        )

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union["T.FragmentBuffer([block_N, block_M], dtype)", "T.SharedBuffer([block_N, block_M], dtype)"],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_M], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(
        acc_s_T: T.FragmentBuffer([block_N, block_M], accum_dtype),
        acc_s_cast_T: Union["T.FragmentBuffer([block_N, block_M], dtype)", "T.SharedBuffer([block_N, block_M], dtype)"],
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        # To do causal softmax, we need to set the scores_max to 0 if it is -inf
        # This process is called Check_inf in FlashAttention3 code, and it only need to be done
        # in the first ceil_div(kBlockM, kBlockN) steps.
        # for i in T.Parallel(block_M):
        #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
            # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            # max * log_2(e)) This allows the compiler to use the ffma
            # instruction instead of fadd and fmul separately.
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def Rescale(
        acc_o_T: T.FragmentBuffer([dim, block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o_T[j, i] *= scores_scale[i]

    @T.prim_func
    def flashattn_sp_kv_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        K_E: T.Tensor(k_e_shape, e_dtype),
        V: T.Tensor(v_shape, dtype),
        V_E: T.Tensor(v_e_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_shared = T.alloc_shared(k_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
            V_shared = T.alloc_shared(v_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)
            acc_s_T = T.alloc_fragment([block_N, block_M], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_M], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_M], dtype)
            acc_o_T = T.alloc_fragment([dim, block_M], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            with T.attr("default", "async_scope", 1):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.copy(K[bz, cur_kv_head, 0:block_N, :], K_shared)
                T.copy(K_E[bz, cur_kv_head, 0:block_N, :], K_E_shared)
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            if chunked_prefill:
                assert is_causal
                loop_range = T.ceildiv(seq_kv - seq_q, block_N) + T.min(
                    T.ceildiv(seq_q, block_N), T.ceildiv((bx + 1) * block_M, block_N)
                )
            else:
                loop_range = (
                    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                    if is_causal
                    else T.ceildiv(seq_kv, block_N)
                )

            for k in T.serial(loop_range):
                T.ptx_wait_group(0)

                with T.attr("default", "async_scope", 1):
                    T.copy(V[bz, cur_kv_head, k * block_N // 2 : (k + 1) * block_N // 2, :], V_shared)
                    T.copy(V_E[bz, cur_kv_head, k * block_N // e_factor : (k + 1) * block_N // e_factor, :], V_E_shared)
                T.ptx_commit_group()

                T.gemm_sp_v2(
                    K_shared, K_E_shared, Q_shared, acc_s_T, transpose_B=True, policy=mma_policy, clear_accum=True
                )

                T.ptx_wait_group(0)
                if k < loop_range - 1:
                    with T.attr("default", "async_scope", 1):
                        T.copy(K[bz, cur_kv_head, (k + 1) * block_N : (k + 2) * block_N, :], K_shared)
                        T.copy(K_E[bz, cur_kv_head, (k + 1) * block_N : (k + 2) * block_N, :], K_E_shared)
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s_T[j, i] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s_T[j, i], -T.infinity(acc_s_T.dtype)
                            )
                        else:
                            acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o_T, scores_scale)
                MMASP1(V, V_E, V_shared, V_E_shared, acc_s_cast_T, acc_o_T, k, cur_kv_head, bz)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return flashattn_sp_kv_kernel


def main(
    args,
):
    from hierasparse.compress_method import (
        torch_block_compress_key,
        torch_block_compress_value,
        torch_compress_key,
        torch_compress_value,
    )
    from hierasparse.prune_method import prune_block_key, prune_block_value, prune_topk

    torch.cuda.get_device_name(0)

    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_q = args.seq_q
    seq_kv = args.seq_kv
    dim = args.dim
    is_causal = args.is_causal
    block_N = args.block_N
    key_prune_ratio = args.key_prune_ratio
    value_prune_ratio = args.value_prune_ratio
    seq_kv = round(seq_kv, block_N)

    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    Q = torch.randn([batch, heads, seq_q, dim], dtype=torch.float16, device="cuda")
    K_unpruned = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")
    V_unpruned = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        latency_fa2 = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K_unpruned, V_unpruned, is_causal=is_causal, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
    print(f"FLASH_ATTENTION: {latency_fa2:.2f} ms {total_flops / latency_fa2 * 1e-9:.2f} TFlops")

    k_page_idx, k_dense_blocks, k_sparse_blocks, k_meta_blocks = torch_block_compress_key(
        prune_block_key(
            K_unpruned,
            block_seq_size=block_N,
            prune_ratio=key_prune_ratio,
        ),
        prune_ratio=key_prune_ratio,
        block_s=block_N,
    )
    v_page_idx, v_dense_blocks, v_sparse_blocks, v_meta_blocks = torch_block_compress_value(
        prune_block_value(
            V_unpruned,
            block_seq_size=block_N,
            prune_ratio=value_prune_ratio,
        ),
        prune_ratio=value_prune_ratio,
        block_s=block_N,
    )

    global tune_inputs
    tune_inputs = (
        Q,
        seq_kv,
        k_page_idx,
        v_page_idx,
        k_dense_blocks,
        k_sparse_blocks,
        k_meta_blocks,
        v_dense_blocks,
        v_sparse_blocks,
        v_meta_blocks,
    )

    kernel_unoptimized = blockattn_sp_mk_mv_unoptimized(
        batch,
        heads,
        groups,
        dim,
        block_N,
        is_causal,
    )

    latency_unoptimized = triton.testing.do_bench(
        lambda: kernel_unoptimized(
            Q,
            seq_kv,
            k_page_idx,
            v_page_idx,
            k_dense_blocks,
            k_sparse_blocks,
            k_meta_blocks,
            v_dense_blocks,
            v_sparse_blocks,
            v_meta_blocks,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"Unoptimized: {latency_unoptimized:.2f} ms {total_flops / latency_unoptimized * 1e-9:.2f} TFlops")

    kernel_pipelined = blockattn_sp_mk_mv_pipelined(
        batch,
        heads,
        groups,
        dim,
        block_N,
        is_causal,
    )
    latency_pipelined = triton.testing.do_bench(
        lambda: kernel_pipelined(
            Q,
            seq_kv,
            k_page_idx,
            v_page_idx,
            k_dense_blocks,
            k_sparse_blocks,
            k_meta_blocks,
            v_dense_blocks,
            v_sparse_blocks,
            v_meta_blocks,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"+pipelined: {latency_pipelined:.2f} ms {total_flops / latency_pipelined * 1e-9:.2f} TFlops")

    kernel_pipelined_movmatrix = blockattn_sp_mk_mv_pipelined_movmatrix(
        batch,
        heads,
        groups,
        dim,
        block_N,
        is_causal,
    )

    latency_pipelined_movmatrix = triton.testing.do_bench(
        lambda: kernel_pipelined_movmatrix(
            Q,
            seq_kv,
            k_page_idx,
            v_page_idx,
            k_dense_blocks,
            k_sparse_blocks,
            k_meta_blocks,
            v_dense_blocks,
            v_sparse_blocks,
            v_meta_blocks,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(
        f"+pipe+mov: {latency_pipelined_movmatrix:.2f} ms {total_flops / latency_pipelined_movmatrix * 1e-9:.2f} TFlops"
    )

    kernel_pipelined_movmatrix_merge = blockattn_sp_mk_mv_pipelined_movmatrix_merge(
        batch,
        heads,
        groups,
        dim,
        block_N,
        is_causal,
    )

    latency_merge_pipelined_movmatrix_merge = triton.testing.do_bench(
        lambda: kernel_pipelined_movmatrix_merge(
            Q,
            seq_kv,
            k_page_idx,
            v_page_idx,
            k_dense_blocks,
            k_sparse_blocks,
            k_meta_blocks,
            v_dense_blocks,
            v_sparse_blocks,
            v_meta_blocks,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(
        f"+pipe+mov+merge: {latency_merge_pipelined_movmatrix_merge:.2f} ms {total_flops / latency_merge_pipelined_movmatrix_merge * 1e-9:.2f} TFlops"
    )

    K_sp, K_E = torch_compress_key(prune_topk(K_unpruned, prune_dim=-1))
    V_sp, V_E = torch_compress_value(prune_topk(V_unpruned, prune_dim=-2))

    tune_inputs = (Q, K_sp, K_E, V_sp, V_E)
    kernel_final = flashattn_sp_kv(
        batch,
        heads,
        groups,
        dim,
        is_causal,
    )
    latency_final = triton.testing.do_bench(
        lambda: kernel_final(
            Q,
            K_sp,
            K_E,
            V_sp,
            V_E,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"all: {latency_final:.2f} ms {total_flops / latency_final * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--groups", type=int, default=8, help="gqa kv heads num")
    parser.add_argument("--seq_q", type=int, default=32768, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=32768, help="key/value sequence length")
    parser.add_argument("--block_N", type=int, required=True, help="block_N")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--rep", type=int, default=2_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run_naive", action="store_true")
    parser.add_argument("--key_prune_ratio", type=float, default=0.5)
    parser.add_argument("--value_prune_ratio", type=float, default=0.5)
    args = parser.parse_args()
    main(args)

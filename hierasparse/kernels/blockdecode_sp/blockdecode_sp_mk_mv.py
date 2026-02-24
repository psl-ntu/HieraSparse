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
from tilelang.utils.tensor import torch_assert_close
from torch.nn.attention import SDPBackend, sdpa_kernel

from hierasparse.kernels.configs import acc_t_B_mapping_inv, acc_t_C_mapping

torch.random.manual_seed(0)

BEST_CONFIGS = {
    "NVIDIA L40S": {
        # TileLang: 0.91 ms, 4.74 tflops
        # CUDNN: 1.57 ms 2.73 TFlops 1.74x
        # FLASH_ATTENTION: 1.54 ms 2.78 TFlops 1.70x
        (8, 32, 8): {"block_H": 8, "num_split": 2, "threads": 32, "use_movmatrix": False},
        # TileLang: 0.15 ms, 3.63 tflops
        # CUDNN: 0.26 ms 2.03 TFlops 1.79x
        # FLASH_ATTENTION: 0.26 ms 2.04 TFlops 1.78x
        (1, 32, 8): {"block_H": 8, "num_split": 32, "threads": 32, "use_movmatrix": False},
    },
}


def get_configs():
    block_H = [8]
    num_split = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]
    threads = [32]
    use_movmatrix = [False, True]
    _configs = list(itertools.product(block_H, num_split, threads, use_movmatrix))
    configs = [{"block_H": c[0], "num_split": c[1], "threads": c[2], "use_movmatrix": c[3]} for c in _configs]
    return configs


def get_pass_configs():
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: False,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,  # when no manual pipelining, tl can handle overlapping
    }


tune_inputs: Optional[Tuple] = None


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    workspace = []
    for param in params[len(tune_inputs) :]:
        workspace.append(torch.empty(tuple(param.shape), dtype=tune_inputs[0].dtype, device="cuda"))
    return tune_inputs + tuple(workspace)


@autotune(configs=get_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
@tilelang.jit(out_idx=[-2, -1], pass_configs=get_pass_configs())
def blockdecode_sp_mk_mv(
    batch,
    heads,
    groups,
    block_N,
    dim,
    block_H,
    num_split,
    threads,
    use_movmatrix,
):
    k_dense_blocks_len = T.dynamic("k_dense_blocks_len")
    k_sparse_blocks_len = T.dynamic("k_sparse_blocks_len")
    v_dense_blocks_len = T.dynamic("v_dense_blocks_len")
    v_sparse_blocks_len = T.dynamic("v_sparse_blocks_len")
    dummy0 = T.dynamic("dummy0")
    dummy1 = T.dynamic("dummy1")
    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    idx_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e  # log2(e)

    warps = threads // 32
    mma_policy = T.GemmWarpPolicy.FullCol
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_H = block_H if mma_policy == T.GemmWarpPolicy.FullRow else block_H // warps
    atom_col = warp_N // 8
    atom_row = warp_H // 8

    # Query
    shape_q = [batch, heads, dim]

    # Key
    k_dense_blocks_shape = [batch, groups, k_dense_blocks_len, block_N, dim]
    k_sparse_blocks_shape = [batch, groups, k_sparse_blocks_len, block_N, dim // 2]
    k_meta_blocks_shape = [batch, groups, k_sparse_blocks_len, block_N, dim // e_factor]

    k_dense_shared_shape = [block_N, dim]
    k_sp_shared_shape = [block_N, dim // 2]
    k_e_shared_shape = [block_N, dim // e_factor]

    # key page idx map
    k_page_idx_shape = [batch, groups, dummy0]

    # Value
    v_dense_blocks_shape = [batch, groups, v_dense_blocks_len, block_N, dim]
    v_sparse_blocks_shape = [batch, groups, v_sparse_blocks_len, block_N // 2, dim]
    v_meta_blocks_shape = [batch, groups, v_sparse_blocks_len, block_N // e_factor, dim]

    v_dense_shared_shape = [block_N, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]

    # value page idx map
    v_page_idx_shape = [batch, groups, dummy1]

    shape_o = [batch, heads, dim]
    lse_shape = [batch, heads, num_split]
    lse_combined_shape = [batch, heads]
    part_shape = [batch, heads, num_split, dim]

    kv_group_num = heads // groups
    valid_block_H = min(block_H, kv_group_num)

    masked_blocks = max(1, (block_H + block_N - 1) // block_N + 1)

    @T.macro
    def LOAD_K_DENSE(
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_dense_shared: T.SharedBuffer(k_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bid: T.int32,
    ):
        T.copy(K_dense_blocks[bid, cur_kv_head, k_page_idx, :, :], K_dense_shared)

    @T.macro
    def LOAD_K_SP(
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_SP_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bid: T.int32,
    ):
        T.copy(K_sparse_blocks[bid, cur_kv_head, k_page_idx, :, :], K_SP_shared)
        T.copy(K_E_blocks[bid, cur_kv_head, k_page_idx, :, :], K_E_shared)

    @T.macro
    def MMA0(
        Q_shared: T.SharedBuffer([block_H, dim], dtype),
        K_shared: T.SharedBuffer(k_dense_shared_shape, dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_H], accum_dtype),
    ):
        T.gemm(K_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

    @T.macro
    def LOAD_V_DENSE(
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_dense_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bid: T.int32,
    ):
        T.copy(V_dense_blocks[bid, cur_kv_head, k_page_idx, :, :], V_dense_shared)

    @T.macro
    def LOAD_V_SP(
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_SP_shared: T.SharedBuffer(v_sp_shared_shape, dtype),
        V_E_shared: T.SharedBuffer(v_e_shared_shape, e_dtype),
        k_page_idx: T.int32,
        cur_kv_head: T.int32,
        bid: T.int32,
    ):
        T.copy(V_sparse_blocks[bid, cur_kv_head, k_page_idx, :, :], V_SP_shared)
        T.copy(V_E_blocks[bid, cur_kv_head, k_page_idx, :, :], V_E_shared)

    @T.macro
    def MMASP0(
        Q_shared: T.SharedBuffer([block_H, dim], dtype),
        K_shared: T.SharedBuffer(k_sp_shared_shape, dtype),
        K_E_shared: T.SharedBuffer(k_e_shared_shape, e_dtype),
        acc_s_T: T.FragmentBuffer([block_N, block_H], accum_dtype),
    ):
        T.gemm_sp_v2(
            K_shared, K_E_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
        )

    @T.macro
    def MMA1_DENSE(
        V_shared: T.SharedBuffer(v_dense_shared_shape, dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_H], dtype)",
            "T.SharedBuffer([block_N, block_H], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_H], accum_dtype),
    ):
        T.gemm(
            V_shared,
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
            "T.FragmentBuffer([block_N, block_H], dtype)",
            "T.SharedBuffer([block_N, block_H], dtype)",
        ],
        acc_o_T: T.FragmentBuffer([dim, block_H], accum_dtype),
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
        acc_s_T: T.FragmentBuffer([block_N, block_H], accum_dtype),
        acc_s_cast_T: Union[
            "T.FragmentBuffer([block_N, block_H], dtype)",
            "T.SharedBuffer([block_N, block_H], dtype)",
        ],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_row)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_row)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_H], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_row)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_row)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
        for i in T.Parallel(block_H):
            scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
        for i in T.Parallel(block_H):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_H, block_N):
            acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s_T, scores_sum, dim=0)
        ReLayout(acc_s_T, acc_s_cast_T)
        for i in T.Parallel(block_H):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

    @T.macro
    def flash_attn(
        Q: T.Tensor(shape_q, dtype),
        seq_kv: T.int32,
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_dense_shared = T.alloc_shared(k_dense_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)

            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)

            acc_s_T = T.alloc_fragment([block_N, block_H], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_H], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_H], dtype)
            acc_o_T = T.alloc_fragment([dim, block_H], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            next_k_page_idx = K_page_idx[bid, cur_kv_head, 0]
            with T.attr("default", "async_scope", 1):
                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                if next_k_page_idx > 0:
                    LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bid)
                else:
                    LOAD_K_SP(
                        K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bid
                    )
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seq_kv, block_N)
            for k in T.serial(loop_range):
                k_page_idx = K_page_idx[bid, cur_kv_head, k]
                v_page_idx = V_page_idx[bid, cur_kv_head, k]

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    if v_page_idx > 0:
                        LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bid)
                    else:
                        LOAD_V_SP(
                            V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bid
                        )
                T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(
                        Q_shared,
                        K_SP_shared,
                        K_E_shared,
                        acc_s_T,
                    )
                T.ptx_wait_group(0)

                if k < loop_range - 1:
                    next_k_page_idx = K_page_idx[bid, cur_kv_head, k + 1]
                    with T.attr("default", "async_scope", 1):
                        if next_k_page_idx > 0:
                            LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bid)
                        else:
                            LOAD_K_SP(
                                K_sparse_blocks,
                                K_E_blocks,
                                K_SP_shared,
                                K_E_shared,
                                -next_k_page_idx - 1,
                                cur_kv_head,
                                bid,
                            )
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_H, block_N):
                        k_idx = k * block_N + j
                        acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)

                for i, j in T.Parallel(block_H, dim):
                    acc_o_T[j, i] *= scores_scale[i]

                v_page_idx = V_page_idx[bid, cur_kv_head, k]
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(
                        V_SP_shared,
                        V_E_shared,
                        acc_s_cast_T,
                        acc_o_T,
                    )

            for i, j in T.Parallel(block_H, dim):
                acc_o_T[j, i] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)
            for i in T.Parallel(block_H):
                if i < valid_block_H:
                    lse_combined[bid, hid * valid_block_H + i] = logsum[
                        i
                    ]  # NOTE: don't scale log2e here, scale after combination
            for i, j in T.Parallel(block_H, dim):
                if i < valid_block_H:
                    Q_shared[i, j] = acc_o_T[j, i]
            T.copy(
                Q_shared[:valid_block_H, :],
                Output[bid, hid * valid_block_H : (hid + 1) * valid_block_H, :],
            )

    @T.macro
    def flash_attn_split(
        Q: T.Tensor(shape_q, dtype),
        seq_kv: T.int32,
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
            num_N_tiles = T.ceildiv(seq_kv, block_N)
            tiles_per_split = num_N_tiles // num_split

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_dense_shared = T.alloc_shared(k_dense_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)

            V_dense_shared = T.alloc_shared(v_dense_shared_shape, dtype)
            V_SP_shared = T.alloc_shared(v_sp_shared_shape, dtype)
            V_E_shared = T.alloc_shared(v_e_shared_shape, e_dtype)

            acc_s_T = T.alloc_fragment([block_N, block_H], accum_dtype)
            if use_movmatrix:
                acc_s_cast_T = T.alloc_fragment([block_N, block_H], dtype)
            else:
                acc_s_cast_T = T.alloc_shared([block_N, block_H], dtype)
            acc_o_T = T.alloc_fragment([dim, block_H], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)
            k_page_idx = T.alloc_var(idx_dtype)
            next_k_page_idx = T.alloc_var(idx_dtype)
            v_page_idx = T.alloc_var(idx_dtype)

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            start_tile = tiles_per_split * sid

            next_k_page_idx = K_page_idx[bid, cur_kv_head, start_tile]
            with T.attr("default", "async_scope", 1):
                T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
                if next_k_page_idx > 0:
                    LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bid)
                else:
                    LOAD_K_SP(
                        K_sparse_blocks, K_E_blocks, K_SP_shared, K_E_shared, -next_k_page_idx - 1, cur_kv_head, bid
                    )
            T.ptx_commit_group()

            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            tiles_this_split = T.if_then_else(
                sid == num_split - 1,
                num_N_tiles - tiles_per_split * (num_split - 1),
                tiles_per_split,
            )

            for k_local in T.serial(tiles_this_split):
                k = start_tile + k_local
                k_page_idx = K_page_idx[bid, cur_kv_head, k]
                v_page_idx = V_page_idx[bid, cur_kv_head, k]

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    if v_page_idx > 0:
                        LOAD_V_DENSE(V_dense_blocks, V_dense_shared, v_page_idx - 1, cur_kv_head, bid)
                    else:
                        LOAD_V_SP(
                            V_sparse_blocks, V_E_blocks, V_SP_shared, V_E_shared, -v_page_idx - 1, cur_kv_head, bid
                        )
                T.ptx_commit_group()

                if k_page_idx > 0:
                    MMA0(Q_shared, K_dense_shared, acc_s_T)
                else:
                    MMASP0(
                        Q_shared,
                        K_SP_shared,
                        K_E_shared,
                        acc_s_T,
                    )
                T.ptx_wait_group(0)

                if k_local < tiles_this_split - 1:
                    next_k_page_idx = K_page_idx[bid, cur_kv_head, k + 1]
                    with T.attr("default", "async_scope", 1):
                        if next_k_page_idx > 0:
                            LOAD_K_DENSE(K_dense_blocks, K_dense_shared, next_k_page_idx - 1, cur_kv_head, bid)
                        else:
                            LOAD_K_SP(
                                K_sparse_blocks,
                                K_E_blocks,
                                K_SP_shared,
                                K_E_shared,
                                -next_k_page_idx - 1,
                                cur_kv_head,
                                bid,
                            )
                    T.ptx_commit_group()

                if tiles_this_split - masked_blocks <= k_local:
                    for i, j in T.Parallel(block_H, block_N):
                        k_idx = k * block_N + j
                        acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, acc_s_T[j, i], -T.infinity(acc_s_T.dtype))

                Softmax(acc_s_T, acc_s_cast_T, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)

                for i, j in T.Parallel(block_H, dim):
                    acc_o_T[j, i] *= scores_scale[i]

                v_page_idx = V_page_idx[bid, cur_kv_head, k]
                if v_page_idx > 0:
                    MMA1_DENSE(V_dense_shared, acc_s_cast_T, acc_o_T)
                else:
                    MMA1_SP(
                        V_SP_shared,
                        V_E_shared,
                        acc_s_cast_T,
                        acc_o_T,
                    )

            for i, j in T.Parallel(block_H, dim):
                acc_o_T[j, i] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            for i in T.Parallel(block_H):
                if i < valid_block_H:
                    glse[bid, hid * valid_block_H + i, sid] = logsum[i]
            for i, j in T.Parallel(block_H, dim):
                if i < valid_block_H:
                    Q_shared[i, j] = acc_o_T[j, i]
            T.copy(
                Q_shared[:valid_block_H, :],
                Output_partial[bid, hid * valid_block_H : (hid + 1) * valid_block_H, sid, :],
            )

    @T.macro
    def combine(
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (hid, bz):
            po_local = T.alloc_fragment([dim], dtype)
            o_accum_local = T.alloc_fragment([dim], accum_dtype)
            lse_local_split = T.alloc_local([1], accum_dtype)
            lse_logsum_local = T.alloc_local([1], accum_dtype)
            lse_max_local = T.alloc_local([1], accum_dtype)
            scale_local = T.alloc_local([1], accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            lse_max_local[0] = -T.infinity(accum_dtype)
            for k in T.serial(num_split):
                lse_max_local[0] = T.max(lse_max_local[0], glse[bz, hid, k])
            for k in T.Pipelined(num_split, num_stages=1):
                lse_local_split[0] = glse[bz, hid, k]
                lse_logsum_local[0] += T.exp2(lse_local_split[0] - lse_max_local[0])
            lse_logsum_local[0] = T.log2(lse_logsum_local[0]) + lse_max_local[0]
            for k in T.serial(num_split):
                for i in T.Parallel(dim):
                    po_local[i] = Output_partial[bz, hid, k, i]
                lse_local_split[0] = glse[bz, hid, k]
                scale_local[0] = T.exp2(lse_local_split[0] - lse_logsum_local[0])
                for i in T.Parallel(dim):
                    o_accum_local[i] += po_local[i] * scale_local[0]
            for i in T.Parallel(dim):
                Output[bz, hid, i] = o_accum_local[i]
            lse_combined[bz, hid] = lse_logsum_local[0] * (1.0 / log2e)

    @T.prim_func
    def blockdecode_sp_mk_mv_split_kernel(
        Q: T.Tensor(shape_q, dtype),
        seq_kv: T.int32,
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        flash_attn_split(
            Q,
            seq_kv,
            K_dense_blocks,
            K_sparse_blocks,
            K_E_blocks,
            K_page_idx,
            V_dense_blocks,
            V_sparse_blocks,
            V_E_blocks,
            V_page_idx,
            glse,
            Output_partial,
        )
        combine(glse, Output_partial, Output, lse_combined)

    @T.prim_func
    def blockdecode_sp_mk_mv_no_split_kernel(
        Q: T.Tensor(shape_q, dtype),
        seq_kv: T.int32,
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        V_dense_blocks: T.Tensor(v_dense_blocks_shape, dtype),
        V_sparse_blocks: T.Tensor(v_sparse_blocks_shape, dtype),
        V_E_blocks: T.Tensor(v_meta_blocks_shape, e_dtype),
        V_page_idx: T.Tensor(v_page_idx_shape, idx_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        flash_attn(
            Q,
            seq_kv,
            K_dense_blocks,
            K_sparse_blocks,
            K_E_blocks,
            K_page_idx,
            V_dense_blocks,
            V_sparse_blocks,
            V_E_blocks,
            V_page_idx,
            Output,
            lse_combined,
        )

    if num_split > 1:
        return blockdecode_sp_mk_mv_split_kernel
    else:
        return blockdecode_sp_mk_mv_no_split_kernel


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def ref_program(Q, K, V):
    Q = Q.unsqueeze(2)
    batch, heads, seq_len, dim = Q.shape
    _, groups, _, _ = K.shape
    K = repeat_kv(K, heads // groups)
    V = repeat_kv(V, heads // groups)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    lse = torch.logsumexp(scores, dim=-1)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output.squeeze(2), lse.squeeze(2)


def main(args):
    from hierasparse.compress_method import (
        torch_block_compress_key,
        torch_block_compress_value,
    )
    from hierasparse.prune_method import prune_block_key, prune_block_value

    device_name = torch.cuda.get_device_name()

    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_kv = args.seq_kv
    dim = args.dim
    tune = args.tune
    block_N = args.block_N
    key_prune_ratio = args.key_prune_ratio
    value_prune_ratio = args.value_prune_ratio

    qk_flops = 2 * batch * heads * seq_kv * dim
    pv_flops = 2 * batch * heads * seq_kv * dim
    total_flops = qk_flops + pv_flops

    Q = torch.randn([batch, heads, dim], dtype=torch.float16, device="cuda")
    K = prune_block_key(
        torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"),
        block_seq_size=block_N,
        prune_ratio=key_prune_ratio,
    )
    V = prune_block_value(
        torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"),
        block_seq_size=block_N,
        prune_ratio=value_prune_ratio,
    )

    k_page_idx, k_dense_blocks, k_sparse_blocks, k_meta_blocks = torch_block_compress_key(
        K, prune_ratio=key_prune_ratio, block_s=block_N
    )
    v_page_idx, v_dense_blocks, v_sparse_blocks, v_meta_blocks = torch_block_compress_value(
        V, prune_ratio=value_prune_ratio, block_s=block_N
    )

    global tune_inputs
    tune_inputs = (
        Q,
        seq_kv,
        k_dense_blocks,
        k_sparse_blocks,
        k_meta_blocks,
        k_page_idx,
        v_dense_blocks,
        v_sparse_blocks,
        v_meta_blocks,
        v_page_idx,
    )
    print(f"{seq_kv=}, {k_page_idx.shape=}, {v_page_idx.shape=}")

    if not tune:
        config = BEST_CONFIGS[device_name][(batch, heads, groups)]
        kernel = blockdecode_sp_mk_mv(batch, heads, groups, block_N, dim, **config)
    else:
        kernel = blockdecode_sp_mk_mv(
            batch,
            heads,
            groups,
            block_N,
            dim,
        )
        best_latency = kernel.latency
        config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {config} for {seq_kv=}")
        print(f"Ref latency: {ref_latency}")

    split = config["num_split"]
    glse = torch.empty(batch, heads, split, device="cuda", dtype=torch.float16)
    Output_partial = torch.empty(batch, heads, split, dim, device="cuda", dtype=torch.float16)

    args_list = [
        Q,
        seq_kv,
        k_dense_blocks,
        k_sparse_blocks,
        k_meta_blocks,
        k_page_idx,
        v_dense_blocks,
        v_sparse_blocks,
        v_meta_blocks,
        v_page_idx,
        glse,
        Output_partial,
    ]
    O_tl, lse_tl = kernel(*args_list)

    tl_latency = triton.testing.do_bench(
        lambda: kernel(*args_list),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"TileLang: {tl_latency:.2f} ms, {total_flops / tl_latency * 1e-9:.2f} tflops")
    if args.check:
        O_ref, lse_ref = ref_program(Q, K, V)
        torch_assert_close(O_tl, O_ref, base_name="tl", ref_name="ref")
        torch_assert_close(lse_tl, lse_ref, base_name="tl_lse", ref_name="ref_lse")

    Q = Q.unsqueeze(2)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"CUDNN: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops {latency / tl_latency:.2f}x")
        if args.check:
            O_sdpa = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
            torch_assert_close(
                O_sdpa.squeeze(2),
                O_tl,
                base_name="sdpa",
                ref_name="tilelang",
            )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(
            f"FLASH_ATTENTION: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops {latency / tl_latency:.2f}x"
        )
        if args.check:
            O_fa2 = F.scaled_dot_product_attention(Q, K, V, enable_gqa=True)
            torch_assert_close(
                O_fa2.squeeze(2),
                O_tl,
                base_name="fa2",
                ref_name="tilelang",
            )

    if args.check:
        print("Precision check passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--groups", type=int, default=8, help="groups")
    parser.add_argument("--seq_kv", type=int, default=32768, help="kv sequence length")
    parser.add_argument("--block_N", type=int, required=True, help="block_N")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--disable_cache", action="store_true", help="disable compiler cache")
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--rep", type=int, default=1_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--key_prune_ratio", type=float, default=0.5)
    parser.add_argument("--value_prune_ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.disable_cache:
        tilelang.clear_cache()
    main(args)

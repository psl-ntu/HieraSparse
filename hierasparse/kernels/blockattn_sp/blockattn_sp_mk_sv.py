import argparse
import itertools
from functools import partial
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

from hierasparse.kernels.configs import acc_t_B_mapping_inv, acc_t_C_mapping
from hierasparse.utils import round

BEST_CONFIGS = {
    "NVIDIA L40S": {
        # Tile-lang: 261.53 ms 269.07 TFlops
        # CUDNN: 408.15 ms 172.41 TFlops 1.56x
        # FLASH_ATTENTION: 413.72 ms 170.09 TFlops 1.58x
        (8, 32, 8, True): {"block_M": 96, "threads": 128, "use_movmatrix": True},
        # Tile-lang: 31.92 ms 275.57 TFlops
        # CUDNN: 48.58 ms 181.05 TFlops 1.52x
        # FLASH_ATTENTION: 49.08 ms 179.21 TFlops 1.54x
        (1, 32, 8, True): {"block_M": 96, "threads": 128, "use_movmatrix": True},
    },
    "NVIDIA A100 80GB PCIe": {
        (1, 32, 8, True): {"block_M": 96, "threads": 128, "use_movmatrix": True},
        (8, 32, 8, True): {"block_M": 128, "threads": 128, "use_movmatrix": True},
    },
}


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


def get_configs():
    iter_params = dict(
        block_M=[32 * i for i in range(1, 5)],
        threads=[128, 256],
        use_movmatrix=[True, False],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


tune_inputs: Optional[Tuple[torch.Tensor]] = None


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    return tune_inputs


@autotune(configs=get_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
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
def blockattn_sp_mk_sv(
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
    dummy2 = T.dynamic("dummy2")
    k_dense_blocks = T.dynamic("k_dense_blocks")
    k_sparse_blocks = T.dynamic("k_sparse_blocks")

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
    v_shape = [batch, groups, dummy1, dim]
    v_e_shape = [batch, groups, dummy2, dim]
    v_sp_shared_shape = [block_N // 2, dim]
    v_e_shared_shape = [block_N // e_factor, dim]

    print(f"{q_shape=} {q_shared_shape=}")
    print(f"{k_dense_blocks_shape=} {k_sparse_blocks_shape=} {k_meta_blocks_shape=}")
    print(f"{v_shape=} {v_sp_shared_shape=} {v_e_shared_shape=}")

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
        T.gemm_v2(
            K_dense_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True
        )

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
    def blockattn_sp_mk_sv_kernel(
        Q: T.Tensor(q_shape, dtype),
        seq_kv: T.int32,
        K_page_idx: T.Tensor(k_page_idx_shape, idx_dtype),
        K_dense_blocks: T.Tensor(k_dense_blocks_shape, dtype),
        K_sparse_blocks: T.Tensor(k_sparse_blocks_shape, dtype),
        K_E_blocks: T.Tensor(k_meta_blocks_shape, e_dtype),
        V_SP: T.Tensor(v_shape, dtype),
        V_E: T.Tensor(v_e_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_dense_shared = T.alloc_shared(k_shared_shape, dtype)
            K_SP_shared = T.alloc_shared(k_sp_shared_shape, dtype)
            K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
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

                T.ptx_wait_group(0)
                with T.attr("default", "async_scope", 1):
                    T.copy(V_SP[bz, cur_kv_head, k * block_N // 2 : (k + 1) * block_N // 2, :], V_SP_shared)
                    T.copy(V_E[bz, cur_kv_head, k * block_N // e_factor : (k + 1) * block_N // e_factor, :], V_E_shared)
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
                MMA1_SP(V_SP_shared, V_E_shared, acc_s_cast_T, acc_o_T)

            for i, j in T.Parallel(block_M, dim):
                acc_o_T[j, i] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            for i, j in T.Parallel(block_M, dim):
                Q_shared[i, j] = acc_o_T[j, i]

            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])
            T.copy(Q_shared, O[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return blockattn_sp_mk_sv_kernel


def ref_program(Q, K, V, is_causal):
    batch, heads, seq_len, dim = Q.shape
    _, groups, _, _ = K.shape
    K = repeat_kv(K, heads // groups)
    V = repeat_kv(V, heads // groups)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output, lse


def main(
    args,
):
    from tilelang.utils.tensor import torch_assert_close

    from hierasparse.compress_method import (
        torch_block_compress_key,
        torch_compress_value,
    )
    from hierasparse.prune_method import prune_block_key, prune_topk

    device_name = torch.cuda.get_device_name(0)

    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_q = args.seq_q
    seq_kv = args.seq_kv
    dim = args.dim
    is_causal = args.is_causal
    args.tune
    block_N = args.block_N
    key_prune_ratio = args.key_prune_ratio
    seq_kv = round(seq_kv, block_N)

    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    Q = torch.randn([batch, heads, seq_q, dim], dtype=torch.float16, device="cuda")
    K = prune_block_key(
        torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"),
        block_seq_size=block_N,
        prune_ratio=key_prune_ratio,
    )
    V = prune_topk(torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"), prune_dim=-2)

    k_page_idx, k_dense_blocks, k_sparse_blocks, k_meta_blocks = torch_block_compress_key(
        K, prune_ratio=key_prune_ratio, block_s=block_N
    )
    V_sp, V_E = torch_compress_value(V)

    k_page_idx = k_page_idx.to(torch.int16)

    global tune_inputs
    tune_inputs = (
        Q,
        seq_kv,
        k_page_idx,
        k_dense_blocks,
        k_sparse_blocks,
        k_meta_blocks,
        V_sp,
        V_E,
    )

    if not args.tune:
        kernel = blockattn_sp_mk_sv(
            batch,
            heads,
            groups,
            dim,
            block_N,
            is_causal,
            **BEST_CONFIGS[device_name][(batch, heads, groups, is_causal)],
        )
    else:
        kernel = blockattn_sp_mk_sv(
            batch,
            heads,
            groups,
            dim,
            block_N,
            is_causal,
        )
        best_latency = kernel.latency
        best_config = kernel.config
        ref_latency = kernel.ref_latency
        print(f"Best latency: {best_latency}")
        print(f"Best TFlops: {total_flops / best_latency * 1e-9}")
        print(f"Best config: {best_config}")
        print(f"Ref latency: {ref_latency}")

    if args.run_naive:
        ref_program_processed = partial(ref_program, is_causal=is_causal)
        O_ref, lse_ref = ref_program_processed(Q, K, V)
    O_tl, lse_tl = kernel(
        Q,
        seq_kv,
        k_page_idx,
        k_dense_blocks,
        k_sparse_blocks,
        k_meta_blocks,
        V_sp,
        V_E,
    )

    if args.run_naive:
        latency = triton.testing.do_bench(
            lambda: ref_program_processed(Q, K, V),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"Unfused: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
    latency_tl = triton.testing.do_bench(
        lambda: kernel(
            Q,
            seq_kv,
            k_page_idx,
            k_dense_blocks,
            k_sparse_blocks,
            k_meta_blocks,
            V_sp,
            V_E,
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"Tile-lang: {latency_tl:.2f} ms {total_flops / latency_tl * 1e-9:.2f} TFlops")
    if args.check and args.run_naive:
        torch_assert_close(O_tl, O_ref, base_name="tilelang", ref_name="unfused")
        torch_assert_close(lse_tl, lse_ref, base_name="tilelang_lse", ref_name="unfused_lse")

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"CUDNN: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops {latency / latency_tl:.2f}x")
        if args.check:
            O_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)
            if args.run_naive:
                torch_assert_close(O_sdpa, O_ref, base_name="sdpa", ref_name="unfused")
            torch_assert_close(
                O_sdpa,
                O_tl,
                base_name="sdpa",
                ref_name="tilelang",
            )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(
            f"FLASH_ATTENTION: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops {latency / latency_tl:.2f}x"
        )
        if args.check:
            O_fa2 = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)
            if args.run_naive:
                torch_assert_close(O_fa2, O_ref, base_name="fa2", ref_name="unfused")
            torch_assert_close(
                O_fa2,
                O_tl,
                base_name="fa2",
                ref_name="tilelang",
            )

    if args.check:
        print("Precision checking passed")

    print(f"Peak mem: {torch.cuda.max_memory_allocated() / 1024 ** 3} GiB")


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
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--disable_cache", action="store_true", help="disable compiler cache")
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--rep", type=int, default=2_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run_naive", action="store_true")
    parser.add_argument("--key_prune_ratio", type=float, default=0.5)
    args = parser.parse_args()
    if args.disable_cache:
        tilelang.disable_cache()
    main(args)

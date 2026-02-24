# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import argparse
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
from tilelang.utils.tensor import torch_assert_close
from torch.nn.attention import SDPBackend, sdpa_kernel

from hierasparse.kernels.compress_torch import torch_compress_value
from hierasparse.kernels.configs import (
    acc_t_B_mapping_inv,
    acc_t_C_mapping,
    flashattn_sp_tune_configs,
    pass_configs,
)

BEST_CONFIGS = {
    "NVIDIA L40S": {
        # Tile-lang: 314.72 ms 223.59 TFlops
        # CUDNN: 427.71 ms 164.52 TFlops 1.36x
        # FLASH_ATTENTION: 431.83 ms 162.96 TFlops 1.37x
        (8, 32, 8, True): {"block_M": 128, "block_N": 128, "threads": 256, "use_movmatrix": True},
        # Tile-lang: 38.27 ms 229.82 TFlops
        # CUDNN: 50.76 ms 173.30 TFlops 1.33x
        # FLASH_ATTENTION: 51.14 ms 171.99 TFlops 1.34x
        (1, 32, 8, True): {"block_M": 96, "block_N": 128, "threads": 128, "use_movmatrix": True},
    },
}


tune_inputs: Optional[Tuple[torch.Tensor]] = None


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    return tune_inputs


@autotune(configs=flashattn_sp_tune_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs=pass_configs(),
)
def flashattn_sp_v(
    batch, heads, groups, dim, is_causal, block_M, block_N, threads, use_movmatrix, chunked_prefill=False
):
    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    seq_q = T.dynamic("seq_q")
    seq_kv = T.dynamic("seq_kv")
    warps = threads // 32
    assert heads % groups == 0
    kv_groups_per_head = heads // groups
    # Query
    q_shape = [batch, heads, seq_q, dim]
    q_shared_shape = [block_M, dim]
    lse_shape = [batch, heads, seq_q]

    # Key (Dense)
    # k_shape = [batch, groups, seq_kv, dim // 2]
    k_shape = [batch, groups, seq_kv, dim]
    # k_e_shape = [batch, groups, seq_kv, dim // e_factor]
    # k_shared_shape = [block_N, dim // 2]
    k_shared_shape = [block_N, dim]
    # k_e_shared_shape = [block_N, dim // e_factor]

    # Value (Sparse)
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
    print(f"{k_shape=} {k_shared_shape=}")
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
        for i in T.Parallel(block_M):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)

        for i, j in T.Parallel(block_M, block_N):
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
    def flashattn_sp_v_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        # K_E: T.Tensor(k_e_shape, e_dtype),
        V: T.Tensor(v_shape, dtype),
        V_E: T.Tensor(v_e_shape, e_dtype),
        O: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            cur_kv_head = by // kv_groups_per_head
            Q_shared = T.alloc_shared(q_shared_shape, dtype)
            K_shared = T.alloc_shared(k_shared_shape, dtype)
            # K_E_shared = T.alloc_shared(k_e_shared_shape, e_dtype)
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
                # T.copy(K[bz, cur_kv_head, 0:block_N, :], K_shared)
                # T.copy(K_E[bz, cur_kv_head, 0:block_N, :], K_E_shared)
                T.copy(K[bz, cur_kv_head, 0:block_N, :], K_shared)
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

                T.gemm(K_shared, Q_shared, acc_s_T, transpose_B=True, policy=mma_policy, clear_accum=True)

                T.ptx_wait_group(0)
                if k < loop_range - 1:
                    with T.attr("default", "async_scope", 1):
                        T.copy(K[bz, cur_kv_head, (k + 1) * block_N : (k + 2) * block_N, :], K_shared)
                        # T.copy(K_E[bz, cur_kv_head, (k + 1) * block_N : (k + 2) * block_N, :], K_E_shared)
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

    return flashattn_sp_v_kernel


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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
    from hierasparse.prune_method import prune_topk

    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_q = args.seq_q
    seq_kv = args.seq_kv
    dim = args.dim
    is_causal = args.is_causal
    tune = args.tune

    dev_name = torch.cuda.get_device_name(0)

    Q = torch.randn([batch, heads, seq_q, dim], dtype=torch.float16, device="cuda")
    # Dense K
    K = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")

    # Compress V only
    V = prune_topk(torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"), prune_dim=-2)
    V_sp, V_E = torch_compress_value(V)

    global tune_inputs
    tune_inputs = (Q, K, V_sp, V_E)
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        kernel = flashattn_sp_v(
            batch,
            heads,
            groups,
            dim,
            is_causal,
            **BEST_CONFIGS[dev_name][(batch, heads, groups, is_causal)],
        )
    else:
        kernel = flashattn_sp_v(batch, heads, groups, dim, is_causal)
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
    O_tl, lse_tl = kernel(Q, K, V_sp, V_E)

    if args.run_naive:
        latency = triton.testing.do_bench(
            lambda: ref_program_processed(Q, K, V),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"Unfused: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
    latency_tl = triton.testing.do_bench(
        lambda: kernel(Q, K, V_sp, V_E),
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
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--disable_cache", action="store_true", help="disable compiler cache")
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--rep", type=int, default=2_000)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--run_naive", action="store_true")
    args = parser.parse_args()
    if args.disable_cache:
        tilelang.disable_cache()
    main(args)

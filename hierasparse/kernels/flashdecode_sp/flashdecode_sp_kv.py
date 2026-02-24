import argparse
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

from hierasparse.kernels.compress_torch import torch_compress_key, torch_compress_value
from hierasparse.kernels.configs import (
    acc_t_B_mapping_inv,
    acc_t_C_mapping,
    flashdecode_sp_tune_configs,
    pass_configs,
)

torch.random.manual_seed(0)

BEST_CONFIGS = {
    "NVIDIA L40S": {
        # TileLang: 0.9041793634226969 ms, 4.750127540781016 tflops
        # CUDNN: 1.57 ms 2.73 TFlops 1.74x
        # FLASH_ATTENTION: 1.54 ms 2.78 TFlops 1.71x
        (8, 32, 8): {
            "block_N": 128,
            "block_H": 8,
            "num_split": 1,
            "num_stages": 1,
            "threads": 32,
            "use_movmatrix": True,
        },
        # TileLang: 0.1477799200564623 ms, 3.632908393744412 tflops
        # CUDNN: 0.26 ms 2.03 TFlops 1.79x
        # FLASH_ATTENTION: 0.26 ms 2.04 TFlops 1.78x
        (1, 32, 8): {
            "block_N": 128,
            "block_H": 8,
            "num_split": 16,
            "num_stages": 0,
            "threads": 32,
            "use_movmatrix": False,
        },
    },
}


tune_inputs: Optional[Tuple[torch.Tensor]] = None


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    workspace = []
    for param in params[len(tune_inputs) :]:
        workspace.append(torch.empty(tuple(param.shape), dtype=tune_inputs[0].dtype, device="cuda"))
    return tune_inputs + tuple(workspace)


@autotune(configs=flashdecode_sp_tune_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
@tilelang.jit(out_idx=[-2, -1], pass_configs=pass_configs())
def flashdecode_sp_kv(batch, heads, groups, dim, block_N, block_H, num_split, num_stages, threads, use_movmatrix):
    seq_kv = T.dynamic("seq_kv")
    dtype = T.float16
    accum_dtype = T.float
    e_dtype = T.int16
    e_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[dtype][e_dtype]
    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e  # log2(e)
    # query
    shape_q = [batch, heads, dim]
    # key
    shape_k = [batch, groups, seq_kv, dim // 2]
    shape_k_shared = [block_N, dim // 2]
    shape_k_e = [batch, groups, seq_kv, dim // e_factor]
    shape_k_e_shared = [block_N, dim // e_factor]
    # value
    shape_v = [batch, groups, seq_kv // 2, dim]
    shape_v_shared = [block_N // 2, dim]
    shape_v_e = [batch, groups, seq_kv // e_factor, dim]
    shape_v_e_shared = [block_N // e_factor, dim]
    shape_o = [batch, heads, dim]
    lse_shape = [batch, heads, num_split]

    lse_combined_shape = [batch, heads]

    shape_o = shape_q

    kv_group_num = heads // groups

    part_shape = [batch, heads, num_split, dim]
    valid_block_H = min(block_H, kv_group_num)
    assert kv_group_num <= valid_block_H

    mma_policy = T.GemmWarpPolicy.FullCol

    assert mma_policy in [T.GemmWarpPolicy.FullCol, T.GemmWarpPolicy.FullRow]

    warps = threads // 32
    warp_N = block_N if mma_policy == T.GemmWarpPolicy.FullCol else block_N // warps
    warp_M = block_H if mma_policy == T.GemmWarpPolicy.FullRow else block_H // warps
    atom_row = warp_N // 8
    atom_col = warp_M // 8

    assert atom_row > 0 and atom_col > 0, f"{block_N=} {block_H=} {warps=} {warp_N=} {warp_M=} {atom_row=} {atom_col=}"

    print(f"{atom_row=} {atom_col=}")

    print(f"{part_shape=}, {valid_block_H=}")
    print(
        f"{shape_q=}, {shape_k=}, {shape_k_e=}, {shape_k_shared=}, {shape_k_e_shared=}, {shape_v=}, {shape_v_e=}, {shape_v_shared=}, {shape_v_e_shared=}"
    )

    num_N_tiles = T.ceildiv(seq_kv, block_N)
    tiles_per_split = num_N_tiles // num_split
    kvlen_per_split = tiles_per_split * block_N

    @T.macro
    def ReLayout(
        acc_s_T: T.FragmentBuffer([block_N, block_H], accum_dtype),
        acc_s_cast_T: Union["T.FragmentBuffer([block_N, block_H], dtype)", "T.SharedBuffer([block_N, block_H], dtype)"],
    ):
        if use_movmatrix:
            if accum_dtype == dtype:
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
            else:
                acc_s_T_ = T.alloc_fragment([block_N, block_H], dtype)
                T.copy(acc_s_T, acc_s_T_)
                for src_atom_idx in range(atom_row * atom_col):
                    i, j = acc_t_C_mapping(src_atom_idx, atom_col)
                    dst_atom_idx = acc_t_B_mapping_inv(i, j, atom_col)
                    T.ptx_movmatrix(acc_s_T_.data, src_atom_idx * 2, acc_s_cast_T.data, dst_atom_idx * 2)
        else:
            T.copy(acc_s_T, acc_s_cast_T)

    @T.macro
    def flash_attn(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        K_E: T.Tensor(shape_k_e, e_dtype),
        V: T.Tensor(shape_v, dtype),
        V_E: T.Tensor(shape_v_e, e_dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, threads=threads) as (bx, by):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared(shape_k_shared, dtype)
            K_E_shared = T.alloc_shared(shape_k_e_shared, e_dtype)
            V_shared = T.alloc_shared(shape_v_shared, dtype)
            V_E_shared = T.alloc_shared(shape_v_e_shared, e_dtype)
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
            bid = bx
            hid = by
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seq_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bid, cur_kv_head, k * block_N : (k + 1) * block_N, :], K_shared)
                T.copy(K_E[bid, cur_kv_head, k * block_N : (k + 1) * block_N, :], K_E_shared)
                for i, j in T.Parallel(block_H, block_N):
                    k_idx = k * block_N + j
                    acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, 0, -T.infinity(acc_s_T.dtype))
                T.gemm_sp_v2(K_shared, K_E_shared, Q_shared, acc_s_T, transpose_B=True, policy=mma_policy)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s_T, scores_sum, dim=0)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                ReLayout(acc_s_T, acc_s_cast_T)
                for i, j in T.Parallel(block_H, dim):
                    acc_o_T[j, i] *= scores_scale[i]
                T.copy(V[bid, cur_kv_head, k * block_N // 2 : (k + 1) * block_N // 2, :], V_shared)
                T.copy(V_E[bid, cur_kv_head, k * block_N // e_factor : (k + 1) * block_N // e_factor, :], V_E_shared)
                T.gemm_sp_v2(
                    V_shared,
                    V_E_shared,
                    acc_s_cast_T,
                    acc_o_T,
                    transpose_A=True,
                    transpose_E=True,
                    policy=mma_policy,
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
        K: T.Tensor(shape_k, dtype),
        K_E: T.Tensor(shape_k_e, e_dtype),
        V: T.Tensor(shape_v, dtype),
        V_E: T.Tensor(shape_v_e, e_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
    ):
        with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):

            Q_shared = T.alloc_shared([block_H, dim], dtype)
            K_shared = T.alloc_shared(shape_k_shared, dtype)
            K_E_shared = T.alloc_shared(shape_k_e_shared, e_dtype)
            V_shared = T.alloc_shared(shape_v_shared, dtype)
            V_E_shared = T.alloc_shared(shape_v_e_shared, e_dtype)
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

            bid = bx
            hid = by
            sid = bz
            cur_kv_head = hid // (kv_group_num // valid_block_H)

            T.copy(Q[bid, hid * valid_block_H : hid * valid_block_H + block_H, :], Q_shared)
            T.fill(acc_o_T, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            tiles_this_split = T.if_then_else(
                sid == num_split - 1,
                num_N_tiles - tiles_per_split * (num_split - 1),
                tiles_per_split,
            )  # last split might handle more tiles than average
            for k in T.Pipelined(tiles_this_split, num_stages=num_stages):
                T.copy(
                    K[
                        bid,
                        cur_kv_head,
                        kvlen_per_split * sid + k * block_N : kvlen_per_split * sid + (k + 1) * block_N,
                        :,
                    ],
                    K_shared,
                )
                T.copy(
                    K_E[
                        bid,
                        cur_kv_head,
                        kvlen_per_split * sid + k * block_N : kvlen_per_split * sid + (k + 1) * block_N,
                        :,
                    ],
                    K_E_shared,
                )
                for i, j in T.Parallel(block_H, block_N):
                    k_idx = kvlen_per_split * sid + k * block_N + j
                    acc_s_T[j, i] = T.if_then_else(k_idx < seq_kv, 0, -T.infinity(acc_s_T.dtype))
                T.gemm_sp_v2(K_shared, K_E_shared, Q_shared, acc_s_T, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s_T, scores_max, dim=0, clear=True)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s_T[j, i] = T.exp2(acc_s_T[j, i] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s_T, scores_sum, dim=0)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                ReLayout(acc_s_T, acc_s_cast_T)
                for i, j in T.Parallel(block_H, dim):
                    acc_o_T[j, i] *= scores_scale[i]
                T.copy(
                    V[
                        bid,
                        cur_kv_head,
                        (kvlen_per_split * sid + k * block_N) // 2 : (kvlen_per_split * sid + (k + 1) * block_N) // 2,
                        :,
                    ],
                    V_shared,
                )
                T.copy(
                    V_E[
                        bid,
                        cur_kv_head,
                        (kvlen_per_split * sid + k * block_N)
                        // e_factor : (kvlen_per_split * sid + (k + 1) * block_N)
                        // e_factor,
                        :,
                    ],
                    V_E_shared,
                )
                T.gemm_sp_v2(
                    V_shared,
                    V_E_shared,
                    acc_s_cast_T,
                    acc_o_T,
                    transpose_A=True,
                    transpose_E=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
            for i, j in T.Parallel(block_H, dim):
                acc_o_T[j, i] /= logsum[i]
            for i in T.Parallel(block_H):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            for i in T.Parallel(block_H):
                if i < valid_block_H:
                    glse[bid, hid * valid_block_H + i, sid] = logsum[
                        i
                    ]  # NOTE: don't scale log2e here, scale after combination
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
    def flashattn_gqa_decode_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        K_E: T.Tensor(shape_k_e, e_dtype),
        V: T.Tensor(shape_v, dtype),
        V_E: T.Tensor(shape_v_e, e_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        flash_attn_split(Q, K, K_E, V, V_E, glse, Output_partial)
        combine(glse, Output_partial, Output, lse_combined)

    @T.prim_func
    def flashattn_gqa_decode_no_split(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_k, dtype),
        K_E: T.Tensor(shape_k_e, e_dtype),
        V: T.Tensor(shape_v, dtype),
        V_E: T.Tensor(shape_v_e, e_dtype),
        glse: T.Tensor(lse_shape, dtype),
        Output_partial: T.Tensor(part_shape, dtype),
        Output: T.Tensor(shape_o, dtype),
        lse_combined: T.Tensor(lse_combined_shape, dtype),
    ):
        flash_attn(Q, K, K_E, V, V_E, Output, lse_combined)

    if num_split > 1:
        return flashattn_gqa_decode_split
    else:
        return flashattn_gqa_decode_no_split


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
    from hierasparse.prune_method import prune_topk

    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_kv = args.seq_kv
    dim = args.dim
    tune = args.tune
    qk_flops = 2 * batch * heads * seq_kv * dim
    pv_flops = 2 * batch * heads * seq_kv * dim
    total_flops = qk_flops + pv_flops

    device_name = torch.cuda.get_device_name(0)

    Q = torch.randn([batch, heads, dim], dtype=torch.float16, device="cuda")
    K = prune_topk(torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"), prune_dim=-1)
    V = prune_topk(torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda"), prune_dim=-2)

    K_sp, K_E = torch_compress_key(K)
    V_sp, V_E = torch_compress_value(V)

    global tune_inputs
    tune_inputs = (Q, K_sp, K_E, V_sp, V_E)

    if not tune:
        config = BEST_CONFIGS[device_name][(batch, heads, groups)]
        kernel = flashdecode_sp_kv(batch, heads, groups, dim, **config)
    else:
        kernel = flashdecode_sp_kv(batch, heads, groups, dim)
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
    O_tl, lse_tl = kernel(Q, K_sp, K_E, V_sp, V_E, glse, Output_partial)

    tl_latency = triton.testing.do_bench(
        lambda: kernel(Q, K_sp, K_E, V_sp, V_E, glse, Output_partial),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"TileLang: {tl_latency} ms, {total_flops / tl_latency * 1e-9} tflops")
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
        print("Precision check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--heads", type=int, default=32, help="heads")
    parser.add_argument("--groups", type=int, default=8, help="groups")
    parser.add_argument("--seq_kv", type=int, default=32768, help="kv sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    parser.add_argument("--disable_cache", action="store_true", help="disable compiler cache")
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--rep", type=int, default=1_000)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.disable_cache:
        tilelang.disable_cache()
    main(args)

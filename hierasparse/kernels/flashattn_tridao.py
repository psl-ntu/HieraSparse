import argparse
from typing import List, Optional, Tuple

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F
import triton
from tilelang.autotuner import *
from tilelang.engine.param import KernelParam
from tilelang.utils.tensor import torch_assert_close
from torch.nn.attention import SDPBackend, sdpa_kernel

from hierasparse.kernels.configs import flashattn_tune_configs

tune_inputs: Optional[Tuple[torch.Tensor]] = None


BEST_CONFIGS = {
    "NVIDIA L40S": {
        (8, 32, 8, True): {"block_M": 128, "block_N": 64, "threads": 128},
        (1, 32, 8, True): {"block_M": 128, "block_N": 64, "threads": 128},
    },
}


def supply_prog(params: List[KernelParam]):
    global tune_inputs
    assert tune_inputs is not None, "Tune inputs are not set"
    return tune_inputs


@autotune(configs=flashattn_tune_configs(), warmup=250, rep=1000, supply_prog=supply_prog)
@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flashattn(batch, heads, groups, dim, is_causal, block_M, block_N, threads):
    dtype = T.float16
    accum_dtype = T.float32
    seq_q = T.dynamic("seq_q")
    seq_kv = T.dynamic("seq_kv")

    log2e = 1.44269504
    scale = (1.0 / dim) ** 0.5 * log2e
    q_shape = [batch, heads, seq_q, dim]
    lse_shape = [batch, heads, seq_q]
    kv_shape = [batch, groups, seq_kv, dim]
    num_groups_per_head = heads // groups
    masked_blocks = max(1, (block_M + block_N - 1) // block_N + 1)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.gemm_v1(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def Softmax(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        scores_max: T.FragmentBuffer([block_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
        scores_sum: T.FragmentBuffer([block_M], accum_dtype),
        logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
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
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
        acc_o: T.FragmentBuffer([block_M, dim], accum_dtype),
        scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, dim):
            acc_o[i, j] *= scores_scale[i]

    @T.prim_func
    def flashattn_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
        lse: T.Tensor(lse_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)

            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            with T.attr("default", "async_scope", 1):
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.copy(K[bz, by // num_groups_per_head, 0:block_N, :], K_shared)  # 0
            T.ptx_commit_group()

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )
            for k in T.serial(loop_range):
                T.ptx_wait_group(0)

                with T.attr("default", "async_scope", 1):
                    T.copy(V[bz, by // num_groups_per_head, k * block_N : (k + 1) * block_N, :], V_shared)
                T.ptx_commit_group()

                T.gemm_v1(
                    Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True
                )

                T.ptx_wait_group(0)
                if k < loop_range - 1:
                    with T.attr("default", "async_scope", 1):
                        T.copy(K[bz, by // num_groups_per_head, (k + 1) * block_N : (k + 2) * block_N, :], K_shared)
                    T.ptx_commit_group()

                if loop_range - masked_blocks <= k:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i
                        k_idx = k * block_N + j
                        if is_causal:
                            acc_s[i, j] = T.if_then_else(
                                (k_idx <= q_idx) & (k_idx < seq_kv), acc_s[i, j], -T.infinity(acc_s.dtype)
                            )
                        else:
                            acc_s[i, j] = T.if_then_else(k_idx < seq_kv, acc_s[i, j], -T.infinity(acc_s.dtype))

                Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                Rescale(acc_o, scores_scale)
                MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]

            for i in T.Parallel(block_M):
                logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) * (1 / log2e)

            T.copy(acc_o, Q_shared)
            T.copy(Q_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])
            T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

    return flashattn_kernel


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


def main(args):
    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_q = args.seq_q
    seq_kv = args.seq_kv
    dim = args.dim
    is_causal = args.is_causal
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul

    dev = torch.cuda.get_device_name()

    Q = torch.randn([batch, heads, seq_q, dim], dtype=torch.float16, device="cuda")
    K = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")
    V = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")

    global tune_inputs
    tune_inputs = (Q, K, V)

    if is_causal:
        total_flops /= 2

    if not args.tune:
        kernel = flashattn(batch, heads, groups, dim, is_causal, **BEST_CONFIGS[dev][(batch, heads, groups, is_causal)])
    else:
        kernel = flashattn(batch, heads, groups, dim, is_causal)
        print(f"Best config: {kernel.config}")

    O_tl, lse_tl = kernel(Q, K, V)
    latency = triton.testing.do_bench(
        lambda: kernel(Q, K, V),
        warmup=250,
        rep=1_000,
    )
    print(f"tilelang: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")

    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"CUDNN: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
        if args.check:
            O_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)
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
        print(f"FLASH_ATTENTION: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
        if args.check:
            O_fa2 = F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, enable_gqa=True)
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
    parser.add_argument("--groups", type=int, default=8, help="groups")
    parser.add_argument("--seq_q", type=int, default=32768, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=32768, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--disable_cache", action="store_true", help="disable tilelang cache")
    parser.add_argument("--check", action="store_true", help="check correctness against naive implementation")
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--rep", type=int, default=2_000)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    if args.disable_cache:
        tilelang.disable_cache()
    main(args)

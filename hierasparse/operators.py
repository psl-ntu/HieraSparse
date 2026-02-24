from typing import Tuple

import torch

from hierasparse.kernels import (
    blockattn_sp_mk_mv,
    blockdecode_sp_mk_mv,
    flashattn,
    flashattn_sp_kv,
    flashattn_sp_v,
    flashdecode_sp_kv,
    flashdecode_sp_v,
)
from hierasparse.utils import to_tl_stride

DEV_NAME = torch.cuda.get_device_name(torch.cuda.current_device())

# @torch.compile(mode="max-autotune", dynamic=True)
# def naive_attn_op(Q, K, V, is_decode):
#     Q_len = Q.size(-2)
#     KV_len = K.size(-2)
#     K = K.repeat_interleave(Q.size(1) // K.size(1), dim=1)
#     V = V.repeat_interleave(Q.size(1) // V.size(1), dim=1)
#     attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)
#     if not is_decode:
#         mask = torch.tril(torch.ones((Q_len, KV_len), device=Q.device)).unsqueeze(0).unsqueeze(0)
#         attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
#     lse = torch.logsumexp(attn_scores, dim=-1)
#     attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)
#     output = torch.matmul(attn_probs, V)
#     return output, lse


def naive_attn_op(Q, K, V, is_decode):
    if is_decode:
        from flash_attn import flash_attn_func

        O, lse, _ = flash_attn_func(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            softmax_scale=None,
            causal=False,
            return_attn_probs=True,
        )
        return O.transpose(1, 2), lse.half()
    else:
        from hierasparse.kernels.flashattn_tridao import BEST_CONFIGS

        if not Q.is_contiguous():
            Q = Q.contiguous()

        if not K.is_contiguous():
            K = K.contiguous()

        if not V.is_contiguous():
            V = V.contiguous()

        Q = to_tl_stride(Q)
        K = to_tl_stride(K)
        V = to_tl_stride(V)

        b, head, seq_q, dim = Q.shape
        b, group, seq_kv, dim = K.shape
        kernel = flashattn(
            batch=b,
            heads=head,
            groups=group,
            dim=dim,
            is_causal=not is_decode,
            **BEST_CONFIGS[DEV_NAME][(b, head, group, not is_decode)],
        )
        O, lse = kernel(Q, K, V)
        return O, lse


def flashattn_sp_kv_op(
    Q: torch.Tensor,
    K_SP: torch.Tensor,
    K_E: torch.Tensor,
    V_SP: torch.Tensor,
    V_E: torch.Tensor,
    is_causal: bool,
    chunked_prefill: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, heads, _, dim = Q.shape
    _, groups, _, _ = K_SP.shape
    from hierasparse.kernels.flashattn_sp.flashattn_sp_kv import BEST_CONFIGS

    kernel = flashattn_sp_kv(
        batch=b,
        heads=heads,
        groups=groups,
        dim=dim,
        is_causal=is_causal,
        **BEST_CONFIGS[DEV_NAME][(b, heads, groups, is_causal)],
        chunked_prefill=chunked_prefill,
    )
    return kernel(Q, K_SP, K_E, V_SP, V_E)


def flashdecode_sp_kv_op(
    Q: torch.Tensor,
    K_SP: torch.Tensor,
    K_E: torch.Tensor,
    V_SP: torch.Tensor,
    V_E: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, heads, dim = Q.shape
    _, groups, _, _ = K_SP.shape
    from hierasparse.kernels.flashdecode_sp.flashdecode_sp_kv import BEST_CONFIGS

    config = BEST_CONFIGS[DEV_NAME][(b, heads, groups)].copy()
    block_N = config["block_N"]
    # NOTE: might get NaN if too many splits for short sequences
    num_split = min((K_SP.shape[2] + block_N - 1) // block_N, 32)
    config["num_split"] = num_split
    glse = torch.empty(b, heads, config["num_split"], device=Q.device, dtype=Q.dtype)
    Output_partial = torch.empty(b, heads, config["num_split"], dim, device=Q.device, dtype=Q.dtype)

    kernel = flashdecode_sp_kv(
        batch=b,
        heads=heads,
        groups=groups,
        dim=dim,
        **config,
    )
    return kernel(Q, K_SP, K_E, V_SP, V_E, glse, Output_partial)


def blockattn_sp_kv_op(
    Q: torch.Tensor,
    K_dense_blocks: torch.Tensor,
    K_sparse_blocks: torch.Tensor,
    K_meta_blocks: torch.Tensor,
    K_page_idx: torch.Tensor,
    V_dense_blocks: torch.Tensor,
    V_sparse_blocks: torch.Tensor,
    V_meta_blocks: torch.Tensor,
    V_page_idx: torch.Tensor,
    is_causal: bool,
    chunked_prefill: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from hierasparse.kernels.blockattn_sp.blockattn_sp_mk_mv import BEST_CONFIGS

    b, heads, seq_q, dim = Q.shape
    _, groups, _, _, _ = K_dense_blocks.shape
    block_N = K_dense_blocks.shape[3]
    seq_kv = K_page_idx.shape[2] * block_N

    kernel = blockattn_sp_mk_mv(
        batch=b,
        heads=heads,
        groups=groups,
        dim=dim,
        block_N=block_N,
        is_causal=is_causal,
        **BEST_CONFIGS[DEV_NAME][(b, heads, groups, is_causal)],
        chunked_prefill=chunked_prefill,
    )
    return kernel(
        Q,
        seq_kv,
        K_page_idx,
        V_page_idx,
        K_dense_blocks,
        K_sparse_blocks,
        K_meta_blocks,
        V_dense_blocks,
        V_sparse_blocks,
        V_meta_blocks,
    )


def blockdecode_sp_mk_mv_op(
    Q: torch.Tensor,
    K_dense_blocks: torch.Tensor,
    K_sparse_blocks: torch.Tensor,
    K_meta_blocks: torch.Tensor,
    K_page_idx: torch.Tensor,
    V_dense_blocks: torch.Tensor,
    V_sparse_blocks: torch.Tensor,
    V_meta_blocks: torch.Tensor,
    V_page_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from hierasparse.kernels.blockdecode_sp.blockdecode_sp_mk_mv import BEST_CONFIGS

    b, heads, dim = Q.shape
    _, groups, _, _, _ = K_dense_blocks.shape

    config = BEST_CONFIGS[DEV_NAME][(b, heads, groups)].copy()

    block_N = K_dense_blocks.shape[3]
    seq_kv = K_page_idx.shape[2] * block_N

    # NOTE: might get NaN if too many splits for short sequences
    num_split = min((seq_kv + block_N - 1) // block_N, 32)
    config["num_split"] = num_split
    glse = torch.empty(b, heads, config["num_split"], device=Q.device, dtype=Q.dtype)
    Output_partial = torch.empty(b, heads, config["num_split"], dim, device=Q.device, dtype=Q.dtype)

    # workspace
    glse = torch.empty(b, heads, num_split, device=Q.device, dtype=Q.dtype)
    Output_partial = torch.empty(b, heads, num_split, dim, device=Q.device, dtype=Q.dtype)

    kernel = blockdecode_sp_mk_mv(
        batch=b,
        heads=heads,
        groups=groups,
        block_N=block_N,
        dim=dim,
        **config,
    )

    return kernel(
        Q,
        seq_kv,
        K_dense_blocks,
        K_sparse_blocks,
        K_meta_blocks,
        K_page_idx,
        V_dense_blocks,
        V_sparse_blocks,
        V_meta_blocks,
        V_page_idx,
        glse,
        Output_partial,
    )


def flashattn_sp_v_op(
    Q: torch.Tensor,
    K: torch.Tensor,
    V_SP: torch.Tensor,
    V_E: torch.Tensor,
    chunked_prefill: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Q = to_tl_stride(Q.contiguous())
    K = to_tl_stride(K)
    V_SP = to_tl_stride(V_SP)
    V_E = to_tl_stride(V_E)

    b, heads, seq_q, dim = Q.shape
    b, groups, seq_kv, dim = K.shape
    from hierasparse.kernels.flashattn_sp.flashattn_sp_v import BEST_CONFIGS

    kernel = flashattn_sp_v(
        b,
        heads,
        groups,
        dim,
        is_causal=True,
        **BEST_CONFIGS[DEV_NAME][(b, heads, groups, True)],
        chunked_prefill=chunked_prefill,
    )
    return kernel(Q, K, V_SP, V_E)


def flashdecode_sp_v_op(
    Q: torch.Tensor,
    K: torch.Tensor,
    V_SP: torch.Tensor,
    V_E: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Q = to_tl_stride(Q.contiguous())
    K = to_tl_stride(K)
    V_SP = to_tl_stride(V_SP)
    V_E = to_tl_stride(V_E)

    b, heads, dim = Q.shape
    b, groups, seq_kv, dim = K.shape
    from hierasparse.kernels.flashdecode_sp.flashdecode_sp_v import BEST_CONFIGS

    config = BEST_CONFIGS[DEV_NAME][(b, heads, groups)].copy()
    block_N = config["block_N"]
    # NOTE: might get NaN if too many splits for short sequences
    num_split = min((seq_kv + block_N - 1) // block_N, 32)
    config["num_split"] = num_split
    glse = torch.empty(b, heads, config["num_split"], device=Q.device, dtype=Q.dtype)
    Output_partial = torch.empty(b, heads, config["num_split"], dim, device=Q.device, dtype=Q.dtype)

    kernel = flashdecode_sp_v(
        batch=b,
        heads=heads,
        groups=groups,
        dim=dim,
        **config,
    )
    return kernel(Q, K, V_SP, V_E, glse, Output_partial)

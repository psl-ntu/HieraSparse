import tilelang
import torch

from hierasparse.utils import record_func


@record_func
def torch_compress(dense):
    """
    A naive compression function, where each 4-bit meta matches 4 elements in original matrix in row major layout.
    """
    if dense.dim() != 2:
        raise RuntimeError(f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor")

    m, k = dense.shape

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    # if meta_dtype == torch.int32:
    #     if m % 16 != 0:
    #         raise RuntimeError(f"Number of rows of dense matrix {m} must be divisible by 16")
    # else:
    #     if m % 32 != 0:
    #         raise RuntimeError(f"Number of rows of dense matrix {m} must be divisible by 32")
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}")

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, _m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, _m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = meta_n[:, :, 0] | (meta_n[:, :, 1] << 4) | (meta_n[:, :, 2] << 8) | (meta_n[:, :, 3] << 12)
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    return (sparse, meta)


def torch_compress_key(K: torch.Tensor):
    if not K.is_contiguous():
        K = K.contiguous()
    assert K.dim() == 4, f"Expected K to be a 4D tensor, got {K.dim()}D."
    b, hn, seq, dim = K.shape
    K_SP, K_E = torch_compress(K.view(-1, dim))
    K_SP = K_SP.view(b, hn, seq, dim // 2)
    K_E = K_E.view(b, hn, seq, -1)
    return K_SP, K_E


def torch_compress_value(V: torch.Tensor):
    if not V.is_contiguous():
        V = V.contiguous()
    assert V.dim() == 4, f"Expected V to be a 4D tensor, got {V.dim()}D."
    b, hn, seq, hd = V.shape
    V_SP, V_E = torch_compress(V.permute(0, 1, 3, 2).contiguous().view(-1, seq))
    V_SP = V_SP.view(b, hn, hd, seq // 2).permute(0, 1, 3, 2).contiguous()
    V_E = V_E.view(b, hn, hd, -1).permute(0, 1, 3, 2).contiguous()
    return V_SP, V_E


# def torch_block_compress_key_original(K: torch.Tensor, prune_ratio: float, block_s: int):
#     assert K.is_contiguous()
#     b, hn, s, hd = K.shape
#     assert s % block_s == 0, f"{s=} {hd=} {block_s=}"
#     block_size = block_s * hd
#     block_mask, sparse_blocks_per_head = prune_block_key_mask(K, block_s, hd, prune_ratio=prune_ratio)
#     blocks_per_head = s // block_s
#     K = K.view(b, hn, s // block_s, block_s, hd)
#     idx_map = torch.empty_like(block_mask, dtype=torch.int32)
#     sparse_nonzero = torch.empty((b, hn, sparse_blocks_per_head, block_size // 2), dtype=torch.float16, device="cuda")
#     sparse_meta = torch.empty(
#         (b, hn, sparse_blocks_per_head, block_size // 16), dtype=torch.int16, device="cuda"
#     )  # NOTE: hard code meta size
#     dense = torch.empty(
#         (b, hn, blocks_per_head - sparse_blocks_per_head, block_size), dtype=torch.float16, device="cuda"
#     )
#     # NOTE: to parallel
#     for i in range(b):
#         for j in range(hn):
#             sparse_seq_idx = 0
#             dense_seq_idx = 0
#             for k in range(s // block_s):
#                 if block_mask[i, j, k].item() is True:
#                     dense[i, j, dense_seq_idx, :] = K[i, j, k, :, :].contiguous().view(-1)
#                     idx_map[i, j, k] = dense_seq_idx + 1
#                     dense_seq_idx += 1
#                 else:
#                     pruned = prune_topk(K[i, j, k, :, :], prune_dim=-1)
#                     nonzero, meta = torch_compress(pruned)
#                     sparse_nonzero[i, j, sparse_seq_idx, :] = nonzero.view(-1)
#                     sparse_meta[i, j, sparse_seq_idx, :] = meta.view(-1)
#                     idx_map[i, j, k] = -(sparse_seq_idx + 1)
#                     sparse_seq_idx += 1
#     return idx_map, dense, sparse_nonzero, sparse_meta


import tilelang
import tilelang.language as T


@tilelang.jit
def block_compress_idx_map_kernel(b, hn, threads=128):
    dtype = T.bool
    idx_dtype = T.int16
    blocks_per_head = T.dynamic("blocks_per_head")

    @T.prim_func
    def kernel(
        block_mask: T.Tensor([b, hn, blocks_per_head], dtype),
        idx_map: T.Tensor([b, hn, blocks_per_head], idx_dtype),
    ):
        with T.Kernel(b, hn, threads=threads) as (bx, hx):
            # Sequential counters for this (batch, head) pair
            dense_count = T.alloc_local([1], idx_dtype)
            sparse_count = T.alloc_local([1], idx_dtype)

            dense_count[0] = 0
            sparse_count[0] = 0

            # Process blocks sequentially for this (batch, head) pair
            for k in T.serial(blocks_per_head):
                mask_val = block_mask[bx, hx, k]

                if mask_val:
                    dense_count[0] += 1
                    idx_map[bx, hx, k] = dense_count[0]
                else:
                    sparse_count[0] += 1
                    idx_map[bx, hx, k] = -sparse_count[0]

    return kernel

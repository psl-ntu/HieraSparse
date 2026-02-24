import torch

from hierasparse.kernels.compress_tilelang import (
    block_compress_dense,
    block_compress_sparse_key,
    block_compress_sparse_value,
    prune_and_compress_key_kernel,
    prune_and_compress_value_kernel,
)
from hierasparse.kernels.compress_torch import (
    block_compress_idx_map_kernel,
    torch_compress,
    torch_compress_key,
    torch_compress_value,
)
from hierasparse.kernels.topk import topk_multipass
from hierasparse.prune_method import (
    prune_topk,
    tl_prune_block_key_mask,
    tl_prune_block_value_mask,
    torch_prune_block_key_mask,
    torch_prune_block_value_mask,
)
from hierasparse.utils import E_FACTOR, record_func, to_tl_stride


# @torch.compile(mode="max-autotune")
def _torch_prune_and_compress_key(
    K: torch.Tensor,
):
    K = K.contiguous()
    b, h, s, d = K.shape
    indices = topk_multipass(K.abs().view(-1, 4))
    mask = torch.zeros_like(K.view(-1, 4), dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    pruned = K.view(-1, 4) * mask
    pruned = pruned.view(b, h, s, d)
    nonzero, meta = torch_compress_key(pruned)
    return nonzero, meta


@record_func
def torch_prune_and_compress_key(
    K: torch.Tensor,
):
    if K.size(2) == 0:
        return torch.empty((K.size(0), K.size(1), 0, K.size(3) // 2), device=K.device, dtype=K.dtype), torch.empty(
            (K.size(0), K.size(1), 0, K.size(3) // E_FACTOR), device=K.device, dtype=torch.int16
        )
    nonzero, meta = _torch_prune_and_compress_key(K)
    return nonzero.clone(), meta.clone()


# @torch.compile(mode="max-autotune")
def _torch_prune_and_compress_value(
    V: torch.Tensor,
):
    b, h, s, d = V.shape
    assert s % E_FACTOR == 0, "s must be divisible by E_FACTOR"
    V = V.permute(0, 1, 3, 2).contiguous()
    indices = topk_multipass(V.abs().view(-1, 4))
    mask = torch.zeros_like(V.view(-1, 4), dtype=torch.bool)
    mask.scatter_(dim=-1, index=indices, value=True)
    pruned = V.view(-1, 4) * mask
    pruned = pruned.view(b, h, d, s)
    pruned = pruned.permute(0, 1, 3, 2).contiguous()
    nonzero, meta = torch_compress_value(pruned)
    return nonzero, meta


@record_func
def torch_prune_and_compress_value(
    V: torch.Tensor,
):
    if V.size(2) == 0:
        return torch.empty_like(V), torch.empty_like(V, dtype=torch.int16)
    nonzero, meta = _torch_prune_and_compress_value(V)
    return nonzero.clone(), meta.clone()


@record_func
def tilelang_prune_and_compress_key(K):
    K = K.contiguous()
    if K.size(2) == 0:
        return torch.empty((K.size(0), K.size(1), 0, K.size(3) // 2), device=K.device, dtype=K.dtype), torch.empty(
            (K.size(0), K.size(1), 0, K.size(3) // E_FACTOR), device=K.device, dtype=torch.int16
        )
    B, H, S, D = K.shape

    sparse = torch.empty((B, H, S, D // 2), dtype=K.dtype, device=K.device)
    meta = torch.empty((B, H, S, D // 16), dtype=torch.int16, device=K.device)

    kernel = prune_and_compress_key_kernel(B, H, D)
    kernel(K, sparse, meta)

    return sparse, meta


@record_func
def tilelang_prune_and_compress_value(V):
    V = V.contiguous()
    if V.size(2) == 0:
        return torch.empty_like(V), torch.empty_like(V, dtype=torch.int16)
    B, H, S, D = V.shape
    assert S % E_FACTOR == 0, "S must be divisible by E_FACTOR"

    sparse = torch.empty((B, H, S // 2, D), dtype=V.dtype, device=V.device)
    meta = torch.empty((B, H, S // 16, D), dtype=torch.int16, device=V.device)

    kernel = prune_and_compress_value_kernel(B, H, D)
    kernel(V, sparse, meta)

    return sparse, meta


@record_func
def torch_block_compress_key(
    K: torch.Tensor, prune_ratio: float, block_s: int, num_sink_blocks: int = 0, num_local_blocks: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, hn, s, hd = K.shape

    num_sink_tokens = num_sink_blocks * block_s
    num_local_tokens = num_local_blocks * block_s

    num_blockable_tokens = s - num_sink_tokens - num_local_tokens
    if num_blockable_tokens % block_s != 0:
        pad_tokens = block_s - (num_blockable_tokens % block_s)
        K = torch.nn.functional.pad(K, (0, 0, 0, pad_tokens))

    block_size = block_s * hd
    total_tokens = K.shape[2]
    blocks_per_head_full = total_tokens // block_s

    block_mask, sparse_blocks_per_head = torch_prune_block_key_mask(
        K, block_s, prune_ratio=prune_ratio, num_sink_blocks=num_sink_blocks, num_local_blocks=num_local_blocks
    )
    block_mask_seq = block_mask

    dense_blocks_per_head = blocks_per_head_full - sparse_blocks_per_head

    K_blocks = K.view(b, hn, blocks_per_head_full, block_s, hd)
    K_flat = K_blocks.reshape(b, hn, blocks_per_head_full, block_size)

    block_mask_expanded = block_mask_seq.unsqueeze(-1).expand(-1, -1, -1, block_size)
    dense_blocks = K_flat[block_mask_expanded].view(b, hn, dense_blocks_per_head, block_s, hd)

    sparse_mask_expanded = (~block_mask_seq).unsqueeze(-1).expand(-1, -1, -1, block_size)
    sparse_blocks_flat = K_flat[sparse_mask_expanded].view(b, hn, sparse_blocks_per_head, block_size)

    sparse_blocks_reshaped = sparse_blocks_flat.view(b * hn * sparse_blocks_per_head, block_s, hd)
    pruned = prune_topk(sparse_blocks_reshaped, prune_dim=-1)
    pruned_flat = pruned.view(-1, block_size)

    sparse_blocks, meta_blocks = torch_compress(pruned_flat)
    sparse_blocks = sparse_blocks.view(b, hn, sparse_blocks_per_head, block_s, hd // 2)
    meta_blocks = meta_blocks.view(b, hn, sparse_blocks_per_head, block_s, hd // E_FACTOR)

    idx_map = torch.zeros_like(block_mask_seq, dtype=torch.int16)
    kernel = block_compress_idx_map_kernel(b, hn)
    kernel(block_mask_seq, idx_map)

    return (
        idx_map,
        to_tl_stride(dense_blocks),
        to_tl_stride(sparse_blocks),
        to_tl_stride(meta_blocks),
    )


@record_func
def tilelang_block_compress_key(K: torch.Tensor, prune_ratio: float, block_s: int):
    K = K.contiguous()
    b, hn, s, hd = K.shape
    assert s % block_s == 0, f"{s=} {hd=} {block_s=}"
    assert block_s % E_FACTOR == 0, f"{block_s=} must be divisible by {E_FACTOR=}"
    blocks_per_head = s // block_s

    block_mask, sparse_blocks_per_head = tl_prune_block_key_mask(K, block_s, prune_ratio=prune_ratio)
    dense_blocks_per_head = blocks_per_head - sparse_blocks_per_head

    idx_map = torch.empty((b, hn, blocks_per_head), dtype=torch.int16, device=K.device)
    dense_flat = torch.empty((b, hn, dense_blocks_per_head, block_s, hd), dtype=torch.float16, device=K.device)
    sparse_nonzero = torch.empty(
        (b, hn, sparse_blocks_per_head, block_s, hd // 2), dtype=torch.float16, device=K.device
    )
    sparse_meta = torch.empty(
        (b, hn, sparse_blocks_per_head, block_s, hd // E_FACTOR), dtype=torch.int16, device=K.device
    )

    kernel_dense = block_compress_dense(b, hn, block_s, hd)
    if dense_blocks_per_head > 0:
        kernel_dense(K, block_mask, idx_map, dense_flat)

    kernel_sparse = block_compress_sparse_key(b, hn, block_s, hd)
    if sparse_blocks_per_head > 0:
        kernel_sparse(K, block_mask, idx_map, sparse_nonzero, sparse_meta)

    return (
        idx_map,
        to_tl_stride(dense_flat),
        to_tl_stride(sparse_nonzero),
        to_tl_stride(sparse_meta),
    )


@record_func
def torch_block_compress_value(
    V: torch.Tensor, prune_ratio: float, block_s: int, num_sink_blocks: int = 0, num_local_blocks: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, hn, s, hd = V.shape

    num_sink_tokens = num_sink_blocks * block_s
    num_local_tokens = num_local_blocks * block_s

    num_blockable_tokens = s - num_sink_tokens - num_local_tokens
    if num_blockable_tokens % block_s != 0:
        pad_tokens = block_s - (num_blockable_tokens % block_s)
        V = torch.nn.functional.pad(V, (0, 0, 0, pad_tokens))

    block_size = block_s * hd
    total_tokens = V.shape[2]
    blocks_per_head_full = total_tokens // block_s

    block_mask, sparse_blocks_per_head = torch_prune_block_value_mask(
        V, block_s, prune_ratio=prune_ratio, num_sink_blocks=num_sink_blocks, num_local_blocks=num_local_blocks
    )
    block_mask_seq = block_mask

    dense_blocks_per_head = blocks_per_head_full - sparse_blocks_per_head

    V_blocks = V.view(b, hn, blocks_per_head_full, block_s, hd)
    V_flat = V_blocks.reshape(b, hn, blocks_per_head_full, block_size)

    block_mask_expanded = block_mask_seq.unsqueeze(-1).expand(-1, -1, -1, block_size)
    dense_blocks = V_flat[block_mask_expanded].view(b, hn, dense_blocks_per_head, block_s, hd)
    sparse_mask_expanded = (~block_mask_seq).unsqueeze(-1).expand(-1, -1, -1, block_size)
    sparse_blocks_flat = V_flat[sparse_mask_expanded].view(b, hn, sparse_blocks_per_head, block_size)

    sparse_blocks_reshaped = sparse_blocks_flat.view(b * hn * sparse_blocks_per_head, block_s, hd)
    pruned = prune_topk(sparse_blocks_reshaped, prune_dim=-2)
    pruned_flat = pruned.permute(0, 2, 1).contiguous().view(-1, block_size)

    sparse_blocks, meta_blocks = torch_compress(pruned_flat)
    sparse_blocks = (
        sparse_blocks.view(b, hn, sparse_blocks_per_head, hd, block_s // 2).permute(0, 1, 2, 4, 3).contiguous()
    )
    meta_blocks = (
        meta_blocks.view(b, hn, sparse_blocks_per_head, hd, block_s // E_FACTOR).permute(0, 1, 2, 4, 3).contiguous()
    )

    idx_map = torch.zeros_like(block_mask_seq, dtype=torch.int16)
    kernel = block_compress_idx_map_kernel(b, hn)
    kernel(block_mask_seq, idx_map)

    return (
        idx_map,
        to_tl_stride(dense_blocks),
        to_tl_stride(sparse_blocks),
        to_tl_stride(meta_blocks),
    )


@record_func
def tilelang_block_compress_value(V: torch.Tensor, prune_ratio: float, block_s: int):
    V = V.contiguous()
    b, hn, s, hd = V.shape
    assert s % block_s == 0, f"{s=} {hd=} {block_s=}"
    assert block_s % E_FACTOR == 0, f"{block_s=} must be divisible by {E_FACTOR=}"
    blocks_per_head = s // block_s

    block_mask, sparse_blocks_per_head = tl_prune_block_value_mask(V, block_s, prune_ratio=prune_ratio)
    dense_blocks_per_head = blocks_per_head - sparse_blocks_per_head

    idx_map = torch.empty((b, hn, blocks_per_head), dtype=torch.int16, device=V.device)
    dense_flat = torch.empty((b, hn, dense_blocks_per_head, block_s, hd), dtype=torch.float16, device=V.device)
    sparse_nonzero = torch.empty(
        (b, hn, sparse_blocks_per_head, block_s // 2, hd), dtype=torch.float16, device=V.device
    )
    sparse_meta = torch.empty(
        (b, hn, sparse_blocks_per_head, block_s // E_FACTOR, hd), dtype=torch.int16, device=V.device
    )

    kernel_dense = block_compress_dense(b, hn, block_s, hd)
    if dense_blocks_per_head > 0:
        kernel_dense(V, block_mask, idx_map, dense_flat)

    kernel_sparse = block_compress_sparse_value(b, hn, block_s, hd)
    if sparse_blocks_per_head > 0:
        kernel_sparse(V, block_mask, idx_map, sparse_nonzero, sparse_meta)

    return (
        idx_map,
        to_tl_stride(dense_flat),
        to_tl_stride(sparse_nonzero),
        to_tl_stride(sparse_meta),
    )

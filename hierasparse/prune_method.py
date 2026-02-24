from math import ceil

import torch

from hierasparse.kernels.compress_tilelang import (
    prune_block_key_mask,
    prune_block_value_mask,
)
from hierasparse.utils import ELEM, GROUP, record_func


@record_func
def prune_topk(
    X: torch.Tensor,
    prune_dim: int = -1,
    topk: int = ELEM,
    group: int = GROUP,
):
    if prune_dim < 0:
        prune_dim = X.dim() + prune_dim
    assert 0 <= prune_dim < X.dim(), f"prune_dim {prune_dim} is out of bounds for tensor with {X.dim()} dimensions."

    shape = X.shape
    group_shape = shape[prune_dim]
    if group_shape % group != 0:
        raise ValueError(f"Dimension {prune_dim} must be divisible by {group}, got {group_shape}.")

    view_shape = shape[:prune_dim] + (group_shape // group, group) + shape[prune_dim + 1 :]
    X = X.view(view_shape)
    _, topk_indices = X.abs().topk(k=topk, dim=prune_dim + 1, sorted=False, largest=True)

    mask = torch.zeros_like(X, dtype=torch.bool)
    mask.scatter_(dim=prune_dim + 1, index=topk_indices, value=True)

    return (X * mask).view(shape)


def topk_mask(
    X: torch.Tensor,
    prune_dim: int = -1,
    topk: int = ELEM,
    group: int = GROUP,
):
    if prune_dim < 0:
        prune_dim = X.dim() + prune_dim
    assert 0 <= prune_dim < X.dim(), f"prune_dim {prune_dim} is out of bounds for tensor with {X.dim()} dimensions."

    shape = X.shape
    group_shape = shape[prune_dim]
    if group_shape % group != 0:
        raise ValueError(f"Dimension {prune_dim} must be divisible by {group}, got {group_shape}.")

    view_shape = shape[:prune_dim] + (group_shape // group, group) + shape[prune_dim + 1 :]
    X = X.view(view_shape)
    _, topk_indices = X.abs().topk(k=topk, dim=prune_dim + 1, sorted=False, largest=True)

    mask = torch.zeros_like(X, dtype=torch.bool)
    mask.scatter_(dim=prune_dim + 1, index=topk_indices, value=True)

    return mask.view(shape)


def prune_block_key(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
):
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."
    num_blocks_per_head = seq_len // block_seq_size
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)
    if sparse_blocks_per_head == 0:
        return X
    X_reshaped = X.view(
        b,
        hn,
        seq_len // block_seq_size,
        block_seq_size,
        dim // group,
        group,
    )
    _, topk_indices = X_reshaped.abs().topk(k=topk, dim=-1, sorted=False, largest=True)
    elem_mask = torch.zeros_like(X_reshaped, dtype=torch.bool)
    elem_mask.scatter_(dim=-1, index=topk_indices, value=True)

    prune_loss = (X_reshaped.abs() * ~elem_mask).sum(dim=(-1, -2, -3)).view(b, hn, -1)
    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask = block_mask.view(b, hn, seq_len // block_seq_size, 1, 1, 1)

    # find common zero values in two masks and prune them, then inverse to get non-zero value mask
    final_mask = block_mask | elem_mask  #  ~(~block_mask & ~elem_mask)

    return (X_reshaped * final_mask).view(b, hn, seq_len, dim)


@record_func
def torch_prune_block_key_mask(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
    num_sink_blocks: int = 0,
    num_local_blocks: int = 0,
):
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."

    total_blocks_seq = seq_len // block_seq_size
    num_sink_tokens = num_sink_blocks * block_seq_size
    num_local_tokens = num_local_blocks * block_seq_size

    blockable_seq_len = seq_len - num_sink_tokens - num_local_tokens
    X_blockable = X[:, :, num_sink_tokens : num_sink_tokens + blockable_seq_len, :]

    blockable_blocks_seq = blockable_seq_len // block_seq_size
    num_blocks_per_head = blockable_blocks_seq
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)

    if sparse_blocks_per_head == 0 or blockable_blocks_seq == 0:
        block_mask = torch.ones((b, hn, total_blocks_seq), dtype=torch.bool, device=X.device)
        return block_mask, 0

    X_reshaped = X_blockable.view(
        b,
        hn,
        blockable_blocks_seq,
        block_seq_size,
        dim // group,
        group,
    )
    _, topk_indices = X_reshaped.abs().topk(k=topk, dim=-1, sorted=False, largest=True)
    elem_mask = torch.zeros_like(X_reshaped, dtype=torch.bool)
    elem_mask.scatter_(dim=-1, index=topk_indices, value=True)

    prune_loss = (X_reshaped.abs() * ~elem_mask).sum(dim=(-1, -2, -3)).view(b, hn, -1)
    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask_blockable = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask_blockable.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask_blockable = block_mask_blockable.view(b, hn, blockable_blocks_seq)

    block_mask_parts = []
    if num_sink_blocks > 0:
        sink_mask = torch.ones((b, hn, num_sink_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(sink_mask)

    block_mask_parts.append(block_mask_blockable)

    if num_local_blocks > 0:
        local_mask = torch.ones((b, hn, num_local_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(local_mask)

    block_mask = torch.cat(block_mask_parts, dim=2)

    return block_mask, sparse_blocks_per_head


@record_func
def tl_prune_block_key_mask(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
    num_sink_blocks: int = 0,
    num_local_blocks: int = 0,
):
    assert topk == ELEM and group == GROUP, "tl_prune_block_key_mask only supports default topk and group values."
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."

    total_blocks_seq = seq_len // block_seq_size
    num_sink_tokens = num_sink_blocks * block_seq_size
    num_local_tokens = num_local_blocks * block_seq_size

    blockable_seq_len = seq_len - num_sink_tokens - num_local_tokens
    blockable_blocks_seq = blockable_seq_len // block_seq_size

    X_blockable = X[:, :, num_sink_tokens : num_sink_tokens + blockable_seq_len, :].view(
        b, hn, blockable_blocks_seq, block_seq_size, dim
    )

    num_blocks_per_head = blockable_blocks_seq
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)

    if sparse_blocks_per_head == 0 or blockable_blocks_seq == 0:
        block_mask = torch.ones((b, hn, total_blocks_seq), dtype=torch.bool, device=X.device)
        return block_mask, 0

    loss_kernel = prune_block_key_mask(b, hn, block_seq_size, dim)
    prune_loss = torch.empty(X_blockable.shape[:3], dtype=X_blockable.dtype, device=X_blockable.device)
    loss_kernel(X_blockable, prune_loss)
    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask_blockable = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask_blockable.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask_blockable = block_mask_blockable.view(b, hn, blockable_blocks_seq)

    block_mask_parts = []
    if num_sink_blocks > 0:
        sink_mask = torch.ones((b, hn, num_sink_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(sink_mask)

    block_mask_parts.append(block_mask_blockable)

    if num_local_blocks > 0:
        local_mask = torch.ones((b, hn, num_local_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(local_mask)

    block_mask = torch.cat(block_mask_parts, dim=2)

    return block_mask, sparse_blocks_per_head


def prune_block_value(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
):
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."
    num_blocks_per_head = seq_len // block_seq_size
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)
    if sparse_blocks_per_head == 0:
        return X
    X_reshaped = X.view(b, hn, seq_len // block_seq_size, block_seq_size // group, group, dim)
    _, topk_indices = X_reshaped.abs().topk(k=topk, dim=-2, sorted=False, largest=True)
    elem_mask = torch.zeros_like(X_reshaped, dtype=torch.bool)
    elem_mask.scatter_(dim=-3, index=topk_indices, value=True)

    prune_loss = (X_reshaped.abs() * ~elem_mask).sum(dim=(-1, -2, -3)).view(b, hn, -1)
    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask = block_mask.view(b, hn, seq_len // block_seq_size, 1, 1, 1)

    # find common zero values in two masks and prune them, then inverse to get non-zero value mask
    final_mask = block_mask | elem_mask  #  ~(~block_mask & ~elem_mask)

    return (X_reshaped * final_mask).view(b, hn, seq_len, dim)


@record_func
def torch_prune_block_value_mask(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
    num_sink_blocks: int = 0,
    num_local_blocks: int = 0,
):
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."

    total_blocks_seq = seq_len // block_seq_size
    num_sink_tokens = num_sink_blocks * block_seq_size
    num_local_tokens = num_local_blocks * block_seq_size

    blockable_seq_len = seq_len - num_sink_tokens - num_local_tokens
    X_blockable = X[:, :, num_sink_tokens : num_sink_tokens + blockable_seq_len, :]

    blockable_blocks_seq = blockable_seq_len // block_seq_size
    num_blocks_per_head = blockable_blocks_seq
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)

    if sparse_blocks_per_head == 0 or blockable_blocks_seq == 0:
        block_mask = torch.ones((b, hn, total_blocks_seq), dtype=torch.bool, device=X.device)
        return block_mask, 0

    X_reshaped = X_blockable.view(
        b,
        hn,
        blockable_blocks_seq,
        block_seq_size // group,
        group,
        dim,
    )
    _, topk_indices = X_reshaped.abs().topk(k=topk, dim=-2, sorted=False, largest=True)
    elem_mask = torch.zeros_like(X_reshaped, dtype=torch.bool)
    elem_mask.scatter_(dim=-2, index=topk_indices, value=True)

    prune_loss = (X_reshaped.abs() * ~elem_mask).sum(dim=(-1, -2, -3)).view(b, hn, -1)

    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask_blockable = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask_blockable.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask_blockable = block_mask_blockable.view(b, hn, blockable_blocks_seq)

    block_mask_parts = []
    if num_sink_blocks > 0:
        sink_mask = torch.ones((b, hn, num_sink_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(sink_mask)

    block_mask_parts.append(block_mask_blockable)

    if num_local_blocks > 0:
        local_mask = torch.ones((b, hn, num_local_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(local_mask)

    block_mask = torch.cat(block_mask_parts, dim=2)

    return block_mask, sparse_blocks_per_head


def tl_prune_block_value_mask(
    X: torch.Tensor,
    block_seq_size: int,
    prune_ratio: float,
    topk: int = ELEM,
    group: int = GROUP,
    num_sink_blocks: int = 0,
    num_local_blocks: int = 0,
):
    assert topk == ELEM and group == GROUP, "tl_prune_block_value_mask only supports default topk and group values."
    assert X.dim() == 4, f"Expected K to be a 4D tensor, got {X.dim()}D."
    b, hn, seq_len, dim = X.shape
    assert seq_len % block_seq_size == 0, "Sequence length in K must be divisible by block_seq_size."

    total_blocks_seq = seq_len // block_seq_size
    num_sink_tokens = num_sink_blocks * block_seq_size
    num_local_tokens = num_local_blocks * block_seq_size

    blockable_seq_len = seq_len - num_sink_tokens - num_local_tokens
    blockable_blocks_seq = blockable_seq_len // block_seq_size

    X_blockable = X[:, :, num_sink_tokens : num_sink_tokens + blockable_seq_len, :].view(
        b, hn, blockable_blocks_seq, block_seq_size, dim
    )

    num_blocks_per_head = blockable_blocks_seq
    sparse_blocks_per_head = ceil(num_blocks_per_head * prune_ratio)

    if sparse_blocks_per_head == 0 or blockable_blocks_seq == 0:
        block_mask = torch.ones((b, hn, total_blocks_seq), dtype=torch.bool, device=X.device)
        return block_mask, 0

    loss_kernel = prune_block_value_mask(b, hn, block_seq_size, dim)
    prune_loss = torch.empty(X_blockable.shape[:3], dtype=X_blockable.dtype, device=X_blockable.device)
    loss_kernel(X_blockable, prune_loss)

    _, prune_block_indices = prune_loss.topk(k=sparse_blocks_per_head, dim=-1, sorted=False, largest=False)
    block_mask_blockable = torch.ones_like(prune_loss, dtype=torch.bool)
    block_mask_blockable.scatter_(dim=-1, index=prune_block_indices, value=False)
    block_mask_blockable = block_mask_blockable.view(b, hn, blockable_blocks_seq)

    block_mask_parts = []
    if num_sink_blocks > 0:
        sink_mask = torch.ones((b, hn, num_sink_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(sink_mask)

    block_mask_parts.append(block_mask_blockable)

    if num_local_blocks > 0:
        local_mask = torch.ones((b, hn, num_local_blocks), dtype=torch.bool, device=X.device)
        block_mask_parts.append(local_mask)

    block_mask = torch.cat(block_mask_parts, dim=2)

    return block_mask, sparse_blocks_per_head

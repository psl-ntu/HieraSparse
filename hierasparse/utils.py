import functools
from typing import Tuple

import torch
import torch.nn.functional as F
from tilelang.language.proxy import TensorProxy

ELEM, GROUP = 2, 4  # 2-of-4 sparsity

E_FACTOR = 16  # for 16b mma and 16b meta


def check_sparsity_2of4(
    x: torch.Tensor,
    sparse_dim: int = -1,
):
    if sparse_dim < 0:
        sparse_dim = x.dim() + sparse_dim
    assert 0 <= sparse_dim < x.dim(), f"sparse_dim {sparse_dim} is out of bounds for tensor with {x.dim()} dimensions."
    shape = x.shape
    group_shape = shape[sparse_dim]
    if group_shape % GROUP != 0:
        raise ValueError(f"Sparse dimension {sparse_dim} must be divisible by {GROUP}, got {group_shape}.")
    view_shape = shape[:sparse_dim] + (group_shape // GROUP, GROUP) + shape[sparse_dim + 1 :]
    x = x.view(view_shape)
    x = (x != 0).sum(dim=sparse_dim + 1)
    return not (x > ELEM).any().item()


def generate_sparse_tensor(shape: Tuple[int, ...], sparse_dim: int, dtype=torch.float16, device="cuda"):
    if sparse_dim < 0:
        sparse_dim = len(shape) + sparse_dim
    assert (
        0 <= sparse_dim < len(shape)
    ), f"sparse_dim {sparse_dim} is out of bounds for tensor with {len(shape)} dimensions."

    sparse_shape = shape[sparse_dim]

    view_shape = shape[:sparse_dim] + (sparse_shape // GROUP, GROUP) + shape[sparse_dim + 1 :]

    if sparse_shape % GROUP != 0:
        raise ValueError(f"Sparse dimension {sparse_dim} must be divisible by {GROUP}, got {sparse_shape}.")

    full_tensor = torch.randn(view_shape, dtype=dtype, device=device)
    _, topk_indices = full_tensor.abs().topk(k=ELEM, dim=sparse_dim + 1)

    mask = torch.zeros_like(full_tensor, dtype=torch.bool)
    mask.scatter_(dim=sparse_dim + 1, index=topk_indices, value=True)

    return (full_tensor * mask).view(shape)


def round(seqlen: int, group: int = GROUP) -> int:
    if seqlen < 0:
        return 0
    return group * (seqlen // group)


RECORDED_FUNC = {}


def record_func(func):
    RECORDED_FUNC[func.__name__] = func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.profiler.record_function(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out = out - F.sigmoid(block_lse - lse).unsqueeze(-1) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


@record_func
def update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out.clone(), lse.clone()


@torch.compile(mode="max-autotune")
def _update_out_and_lse_nan(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse_valid = ~(torch.isnan(lse) | torch.isinf(lse))
    block_lse_valid = ~(torch.isnan(block_lse) | torch.isinf(block_lse))

    merged_out = out - F.sigmoid(block_lse - lse).unsqueeze(-1) * (out - block_out)
    merged_lse = lse - F.logsigmoid(lse - block_lse)

    both_valid = lse_valid & block_lse_valid
    only_lse_valid = lse_valid & ~block_lse_valid

    out = torch.where(both_valid.unsqueeze(-1), merged_out, torch.where(only_lse_valid.unsqueeze(-1), out, block_out))
    lse = torch.where(both_valid, merged_lse, torch.where(only_lse_valid, lse, block_lse))

    return out, lse


@record_func
def update_out_and_lse_nan(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = _update_out_and_lse_nan(out, lse, block_out, block_lse)
    return out.clone(), lse.clone()


def to_tl_stride(X: torch.Tensor):
    # NOTE: workaround for inconsistency between torch and tilelang stride calculation
    assert X.is_contiguous(), f"Input tensor must be contiguous, but got {X.stride()}"
    return X.as_strided(X.shape, TensorProxy._construct_strides(X.shape))

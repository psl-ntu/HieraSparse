# NOTE: static cache are for benchmarking purpose only, not for accuracy benchmarking

# The contiguous() is required by both fla2 (wo page) and customized kernel, which will
# also be included in benchmark.
from typing import Any, Optional, Tuple

import torch

from hierasparse.caches.base_cache import BaseCache
from hierasparse.compress_method import (
    tilelang_prune_and_compress_key,
    tilelang_prune_and_compress_value,
)
from hierasparse.utils import E_FACTOR


class DenseStaticCache(BaseCache):

    is_compileable = True

    def __init__(
        self,
        bsz: int,
        max_cache_len: int,
        num_kv_heads: int,
        head_dim: int,
        layers: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.bsz = bsz
        self.max_cache_len = max_cache_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self._seen_tokens = 0
        self.layers = layers

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        key_shape = (self.bsz, self.num_kv_heads, self.max_cache_len, self.head_dim)
        value_shape = (self.bsz, self.num_kv_heads, self.max_cache_len, self.head_dim)
        for _ in range(layers):
            new_layer_key_cache = torch.zeros(key_shape, dtype=torch.float16, device="cuda")
            new_layer_value_cache = torch.zeros(value_shape, dtype=torch.float16, device="cuda")

            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)

            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += new_tokens

        if new_tokens == self.chunk_size:
            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = key_states
            self.value_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = value_states
        else:
            assert new_tokens == 1, f"need to be decode, found new_tokens={new_tokens}"

            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = key_states
            self.value_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = value_states

        return (
            self.key_cache[layer_idx][:, :, : self._seen_tokens, :].contiguous(),
            self.value_cache[layer_idx][:, :, : self._seen_tokens, :].contiguous(),
        )

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = self._seen_tokens * self.bsz * self.num_kv_heads * self.head_dim * 2  # float16
        value_bytes = self._seen_tokens * self.bsz * self.num_kv_heads * self.head_dim * 2  # float16
        return key_bytes * self.layers, value_bytes * self.layers


class KeyDenseValueSparseStaticCache(BaseCache):

    ATTN_IMPLEMENTATION = "hf_chunked_k_dense_v_sp"

    is_compileable = True

    def __init__(
        self,
        bsz: int,
        max_cache_len: int,
        num_kv_heads: int,
        head_dim: int,
        layers: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.bsz = bsz
        self.max_cache_len = max_cache_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self._seen_tokens = 0
        self.layers = layers

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.value_meta_cache: list[torch.Tensor] = []
        self.value_remnant_cache: list[torch.Tensor] = []

        key_shape = (self.bsz, self.num_kv_heads, self.max_cache_len, self.head_dim)
        value_shape = (self.bsz, self.num_kv_heads, self.max_cache_len // 2, self.head_dim)
        value_meta_shape = (self.bsz, self.num_kv_heads, self.max_cache_len // E_FACTOR, self.head_dim)
        remnant_shape = (self.bsz, self.num_kv_heads, E_FACTOR, self.head_dim)
        for _ in range(layers):
            new_layer_key_cache = torch.zeros(key_shape, dtype=torch.float16, device="cuda")
            new_layer_value_cache = torch.zeros(value_shape, dtype=torch.float16, device="cuda")
            new_layer_value_meta_cache = torch.zeros(value_meta_shape, dtype=torch.int16, device="cuda")
            new_layer_value_remnant_cache = torch.zeros(remnant_shape, dtype=torch.float16, device="cuda")

            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            self.value_meta_cache.append(new_layer_value_meta_cache)
            self.value_remnant_cache.append(new_layer_value_remnant_cache)

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += new_tokens

        prune_remnant = self._seen_tokens % E_FACTOR
        prune_len = self._seen_tokens - prune_remnant
        if new_tokens == self.chunk_size:
            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = key_states
            v_sp, v_e = tilelang_prune_and_compress_value(value_states)
            self.value_cache[layer_idx][:, :, (self._seen_tokens - new_tokens) // 2 : self._seen_tokens // 2, :] = v_sp
            self.value_meta_cache[layer_idx][
                :, :, (self._seen_tokens - new_tokens) // E_FACTOR : self._seen_tokens // E_FACTOR, :
            ] = v_e
        else:
            assert new_tokens == 1, f"need to be decode, found new_tokens={new_tokens}"

            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = key_states

            if prune_remnant == 0:
                self.value_remnant_cache[layer_idx][:, :, E_FACTOR - new_tokens :, :] = value_states
                v_sp, v_e = tilelang_prune_and_compress_value(self.value_remnant_cache[layer_idx])
                self.value_cache[layer_idx][:, :, (prune_len - E_FACTOR) // 2 : prune_len // 2, :] = v_sp
                self.value_meta_cache[layer_idx][
                    :, :, (prune_len - E_FACTOR) // E_FACTOR : prune_len // E_FACTOR, :
                ] = v_e
            else:
                self.value_remnant_cache[layer_idx][:, :, prune_remnant : prune_remnant + 1, :] = value_states

        # for simplicity of benchmarking, only return the pruned length
        return self.key_cache[layer_idx][:, :, :prune_len, :].contiguous(), (
            self.value_cache[layer_idx][:, :, : prune_len // 2, :].contiguous(),
            self.value_meta_cache[layer_idx][:, :, : prune_len // E_FACTOR, :].contiguous(),
        )

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = self._seen_tokens * self.bsz * self.num_kv_heads * self.head_dim * 2  # float16
        value_bytes = (self._seen_tokens) // 2 * self.bsz * self.num_kv_heads * self.head_dim * 2 + (  # float16
            self._seen_tokens
        ) // E_FACTOR * self.bsz * self.num_kv_heads * self.head_dim * 2  # int16
        return key_bytes * self.layers, value_bytes * self.layers


class KeySparseValueSparseStaticCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_chunked_k_sp_v_sp"
    is_compileable = True

    def __init__(
        self,
        bsz: int,
        max_cache_len: int,
        num_kv_heads: int,
        head_dim: int,
        layers: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.bsz = bsz
        self.max_cache_len = max_cache_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self._seen_tokens = 0
        self.layers = layers

        self.key_cache: list[torch.Tensor] = []
        self.key_meta_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        self.value_meta_cache: list[torch.Tensor] = []
        self.value_remnant_cache: list[torch.Tensor] = []

        key_shape = (self.bsz, self.num_kv_heads, self.max_cache_len, self.head_dim // 2)
        key_meta_shape = (self.bsz, self.num_kv_heads, self.max_cache_len, self.head_dim // E_FACTOR)
        value_shape = (self.bsz, self.num_kv_heads, self.max_cache_len // 2, self.head_dim)
        value_meta_shape = (self.bsz, self.num_kv_heads, self.max_cache_len // E_FACTOR, self.head_dim)
        remnant_shape = (self.bsz, self.num_kv_heads, E_FACTOR, self.head_dim)
        for _ in range(layers):
            new_layer_key_cache = torch.zeros(key_shape, dtype=torch.float16, device="cuda")
            new_layer_key_meta_cache = torch.zeros(key_meta_shape, dtype=torch.int16, device="cuda")
            new_layer_value_cache = torch.zeros(value_shape, dtype=torch.float16, device="cuda")
            new_layer_value_meta_cache = torch.zeros(value_meta_shape, dtype=torch.int16, device="cuda")
            new_layer_value_remnant_cache = torch.zeros(remnant_shape, dtype=torch.float16, device="cuda")

            self.key_cache.append(new_layer_key_cache)
            self.key_meta_cache.append(new_layer_key_meta_cache)
            self.value_cache.append(new_layer_value_cache)
            self.value_meta_cache.append(new_layer_value_meta_cache)
            self.value_remnant_cache.append(new_layer_value_remnant_cache)

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            self._seen_tokens += new_tokens

        prune_remnant = self._seen_tokens % E_FACTOR
        prune_len = self._seen_tokens - prune_remnant
        if new_tokens == self.chunk_size:
            k_sp, k_e = tilelang_prune_and_compress_key(key_states)
            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = k_sp
            self.key_meta_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = k_e
            v_sp, v_e = tilelang_prune_and_compress_value(value_states)
            self.value_cache[layer_idx][:, :, (self._seen_tokens - new_tokens) // 2 : self._seen_tokens // 2, :] = v_sp
            self.value_meta_cache[layer_idx][
                :, :, (self._seen_tokens - new_tokens) // E_FACTOR : self._seen_tokens // E_FACTOR, :
            ] = v_e
        else:
            assert new_tokens == 1, f"need to be decode, found new_tokens={new_tokens}"

            k_sp, k_e = tilelang_prune_and_compress_key(key_states)

            self.key_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = k_sp
            self.key_meta_cache[layer_idx][:, :, self._seen_tokens - new_tokens : self._seen_tokens, :] = k_e

            if prune_remnant == 0:
                self.value_remnant_cache[layer_idx][:, :, E_FACTOR - new_tokens :, :] = value_states
                v_sp, v_e = tilelang_prune_and_compress_value(self.value_remnant_cache[layer_idx])
                self.value_cache[layer_idx][:, :, (prune_len - E_FACTOR) // 2 : prune_len // 2, :] = v_sp
                self.value_meta_cache[layer_idx][
                    :, :, (prune_len - E_FACTOR) // E_FACTOR : prune_len // E_FACTOR, :
                ] = v_e
            else:
                self.value_remnant_cache[layer_idx][:, :, prune_remnant : prune_remnant + 1, :] = value_states

        # for simplicity of benchmarking, only return the pruned length
        return (
            self.key_cache[layer_idx][:, :, :prune_len, :].contiguous(),
            self.key_meta_cache[layer_idx][:, :, :prune_len, :].contiguous(),
        ), (
            self.value_cache[layer_idx][:, :, : prune_len // 2, :].contiguous(),
            self.value_meta_cache[layer_idx][:, :, : prune_len // E_FACTOR, :].contiguous(),
        )

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = (self._seen_tokens) // 2 * self.bsz * self.num_kv_heads * self.head_dim * 2 + (  # float16
            self._seen_tokens
        ) // E_FACTOR * self.bsz * self.num_kv_heads * self.head_dim * 2  # int16
        value_bytes = (self._seen_tokens) // 2 * self.bsz * self.num_kv_heads * self.head_dim * 2 + (  # float16
            self._seen_tokens
        ) // E_FACTOR * self.bsz * self.num_kv_heads * self.head_dim * 2  # int16
        return key_bytes * self.layers, value_bytes * self.layers

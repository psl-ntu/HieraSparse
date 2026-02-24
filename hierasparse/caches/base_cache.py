from typing import Any, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache


class BaseCache(DynamicCache):
    ATTN_IMPLEMENTATION = "flash_attention_2"

    def update_after_attn(
        self,
        query_states: torch.Tensor,
        layer_idx: int,
    ):
        pass

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.element_size() * key_layer_cache.numel()

        value_bytes = 0
        for value_layer_cache in self.value_cache:
            value_bytes += value_layer_cache.element_size() * value_layer_cache.numel()

        return key_bytes, value_bytes

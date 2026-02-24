from typing import Any, Optional

import torch

from hierasparse.caches.base_cache import BaseCache
from hierasparse.prune_method import prune_block_key, prune_block_value, prune_topk
from hierasparse.utils import ELEM, GROUP, round


class DenseCache(BaseCache):
    def __init__(self, *args, cache_query: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_query = cache_query
        if self.cache_query:
            self.query_cache: list[torch.Tensor] = []

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    if self.cache_query:
                        self.query_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                if self.cache_query:
                    self.query_cache.append(query_states)
            elif not self.key_cache[
                layer_idx
            ].numel():  # prefers not t.numel() to len(t) == 0 to export the model  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                if self.cache_query:
                    self.query_cache[layer_idx] = query_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                if self.cache_query:
                    self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class FlashAttnSPSimulationCache(BaseCache):
    def __init__(
        self,
        prune_key_prefill: bool,
        prune_key_decode: bool,
        prune_value_prefill: bool,
        prune_value_decode: bool,
        sink: int,
        local_window: int,
        start_layer: int,
        end_layer: int,
    ):
        self.prune_key_prefill = prune_key_prefill
        self.prune_key_decode = prune_key_decode
        self.prune_value_prefill = prune_value_prefill
        self.prune_value_decode = prune_value_decode
        self.local_window = local_window
        self.sink = sink
        self.start_layer = start_layer
        self.end_layer = end_layer
        self._prefill_key_pruned: list[bool] = []
        self._prefill_value_pruned: list[bool] = []
        super().__init__()

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self._prefill_key_pruned.append(False)
                self._prefill_value_pruned.append(False)

                # shape (b, hc, seq, hd)
                if self.prune_key_prefill and self.start_layer <= layer_idx < self.end_layer:
                    # NOTE: for headdim, we can always assume it's divisible by 4
                    self.key_cache[layer_idx][:, :, self.sink : self._seen_tokens - self.local_window, :] = prune_topk(
                        self.key_cache[layer_idx][:, :, self.sink : self._seen_tokens - self.local_window, :],
                        prune_dim=-1,
                        topk=ELEM,
                        group=GROUP,
                    )
                    self._prefill_key_pruned[layer_idx] = True

                if self.prune_value_prefill and self.start_layer <= layer_idx < self.end_layer:
                    prune_len = round(self._seen_tokens - self.local_window - self.sink)
                    self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :] = prune_topk(
                        self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :],
                        prune_dim=-2,
                        topk=ELEM,
                        group=GROUP,
                    )
                    self._prefill_value_pruned[layer_idx] = True

            elif not self.key_cache[
                layer_idx
            ].numel():  # prefers not t.numel() to len(t) == 0 to export the model  # fills previously skipped layers; checking for tensor causes errors
                raise NotImplementedError("Skipping layers is not supported")
            else:
                if self.prune_key_decode and self.start_layer <= layer_idx < self.end_layer:
                    if not self._prefill_key_pruned[layer_idx]:
                        # prune the existing cache if it was not pruned during prefill
                        self.key_cache[layer_idx][:, :, self.sink : self._seen_tokens - self.local_window, :] = (
                            prune_topk(
                                self.key_cache[layer_idx][:, :, self.sink : self._seen_tokens - self.local_window, :],
                                prune_dim=-1,
                                topk=ELEM,
                                group=GROUP,
                            )
                        )
                        self._prefill_key_pruned[layer_idx] = True
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    # always prune the new coming keys except for sink and local window
                    if self._seen_tokens > self.local_window + self.sink:
                        self.key_cache[layer_idx][:, :, self._seen_tokens - self.local_window - 1, :] = prune_topk(
                            self.key_cache[layer_idx][:, :, self._seen_tokens - self.local_window - 1, :],
                            prune_dim=-1,
                            topk=ELEM,
                            group=GROUP,
                        )
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)

                if self.prune_value_decode and self.start_layer <= layer_idx < self.end_layer:
                    if not self._prefill_value_pruned[layer_idx]:
                        # prune the existing cache if it was not pruned during prefill
                        # exclude current decoding token
                        prune_len = round(self._seen_tokens - self.local_window - self.sink)
                        self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :] = prune_topk(
                            self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :],
                            prune_dim=-2,
                            topk=ELEM,
                            group=GROUP,
                        )
                        self._prefill_value_pruned[layer_idx] = True

                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                    prune_len = round(self._seen_tokens - self.local_window - self.sink)
                    if prune_len % GROUP == 0:
                        # take last group and prune
                        self.value_cache[layer_idx][:, :, self.sink + prune_len - GROUP : self.sink + prune_len, :] = (
                            prune_topk(
                                self.value_cache[layer_idx][
                                    :, :, self.sink + prune_len - GROUP : self.sink + prune_len, :
                                ],
                                prune_dim=-2,
                                topk=ELEM,
                                group=GROUP,
                            )
                        )
                else:
                    # directly concat if not pruning
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class HieraSparseSimulationCache(BaseCache):
    def __init__(
        self,
        prune_key_prefill: bool,
        prune_key_decode: bool,
        prune_value_prefill: bool,
        prune_value_decode: bool,
        prune_key_prefill_ratio: float,
        prune_value_prefill_ratio: float,
        block_seq_size: int,
        sink: int,
        local_window: int,
        start_layer: int,
    ):
        self.prune_key_prefill = prune_key_prefill
        self.prune_key_decode = prune_key_decode
        self.prune_value_prefill = prune_value_prefill
        self.prune_value_decode = prune_value_decode
        self.block_seq_size = block_seq_size
        self.sink = sink
        self.local_window = local_window
        self.prune_key_prefill_ratio = prune_key_prefill_ratio
        self.prune_value_prefill_ratio = prune_value_prefill_ratio
        self._prefill_key_pruned: list[bool] = []
        self._prefill_value_pruned: list[bool] = []
        self.start_layer = start_layer
        super().__init__()

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                self._prefill_key_pruned.append(False)
                self._prefill_value_pruned.append(False)

                # shape (b, hc, seq, hd)
                if self.prune_key_prefill and layer_idx >= self.start_layer:
                    # NOTE: for headdim, we can always assume it's divisible by 4
                    # NOTE: for seqdim, we need to make it divisible by block_seq_size
                    prune_len = round(self._seen_tokens - self.sink - self.local_window, self.block_seq_size)
                    self.key_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :] = prune_block_key(
                        self.key_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :],
                        block_seq_size=self.block_seq_size,
                        prune_ratio=self.prune_key_prefill_ratio,
                    )
                    self._prefill_key_pruned[layer_idx] = True

                if self.prune_value_prefill and layer_idx >= self.start_layer:
                    # NOTE: for seqdim, we need to make it divisible by 4
                    prune_len = round(self._seen_tokens - self.sink - self.local_window, self.block_seq_size)
                    self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :] = prune_block_value(
                        self.value_cache[layer_idx][:, :, self.sink : self.sink + prune_len, :],
                        block_seq_size=self.block_seq_size,
                        prune_ratio=self.prune_value_prefill_ratio,
                    )
                    self._prefill_value_pruned[layer_idx] = True

            elif not self.key_cache[
                layer_idx
            ].numel():  # prefers not t.numel() to len(t) == 0 to export the model  # fills previously skipped layers; checking for tensor causes errors
                raise NotImplementedError("Skipping layers is not supported")
            else:
                if self.prune_key_decode and layer_idx >= self.start_layer:
                    if not self._prefill_key_pruned[layer_idx]:
                        raise ValueError("Always block prune prefill")
                    # always prune the new coming keys
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    # always prune the new coming keys except for sink and local window
                    if self._seen_tokens > self.local_window + self.sink:
                        self.key_cache[layer_idx][:, :, self._seen_tokens - self.local_window - 1, :] = prune_topk(
                            self.key_cache[layer_idx][:, :, self._seen_tokens - self.local_window - 1, :],
                            prune_dim=-1,
                            topk=ELEM,
                            group=GROUP,
                        )
                else:
                    # directly concat if not pruning
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)

                if self.prune_value_decode and layer_idx >= self.start_layer:
                    if not self._prefill_value_pruned[layer_idx]:
                        raise ValueError("Always block prune prefill")
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                    prune_len = round(self._seen_tokens - self.local_window - self.sink)
                    if prune_len % GROUP == 0:
                        # take last group and prune
                        self.value_cache[layer_idx][:, :, self.sink + prune_len - GROUP : self.sink + prune_len, :] = (
                            prune_topk(
                                self.value_cache[layer_idx][
                                    :, :, self.sink + prune_len - GROUP : self.sink + prune_len, :
                                ],
                                prune_dim=-2,
                                topk=ELEM,
                                group=GROUP,
                            )
                        )
                else:
                    # directly concat if not pruning
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


# NOTE: to eliminate cache concat overhead
class DenseCacheNoUpdate(BaseCache):
    def __init__(self, *args, cache_query: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_query = cache_query
        if self.cache_query:
            self.query_cache: list[torch.Tensor] = []

    def update(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                    if self.cache_query:
                        self.query_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
                if self.cache_query:
                    self.query_cache.append(query_states)
            elif not self.key_cache[
                layer_idx
            ].numel():  # prefers not t.numel() to len(t) == 0 to export the model  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
                if self.cache_query:
                    self.query_cache[layer_idx] = query_states
            else:
                pass
                # self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                # self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                # if self.cache_query:
                #     self.query_cache[layer_idx] = torch.cat([self.query_cache[layer_idx], query_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

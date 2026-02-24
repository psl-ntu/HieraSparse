from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple, Union

import torch

from hierasparse.caches.base_cache import BaseCache
from hierasparse.compress_method import (
    tilelang_block_compress_key,
    tilelang_block_compress_value,
    tilelang_prune_and_compress_key,
    tilelang_prune_and_compress_value,
)
from hierasparse.utils import E_FACTOR, round, to_tl_stride


# enum
class RangeType(Enum):
    DENSE = "dense"
    COMPRESSED = "compressed"
    BLOCK_COMPRESSED = "block_compressed"


@dataclass
class BaseRange:
    # [start_tok, end_tok)
    start_tok: int
    end_tok: int

    def __post_init__(self):
        assert self.start_tok <= self.end_tok, "start_tok must be less than end_tok"

    def is_empty(self) -> bool:
        return self.start_tok == self.end_tok

    def get(self) -> Any:
        raise NotImplementedError("BaseClass")

    def type(self) -> str:
        raise NotImplementedError("BaseClass")

    def is_dense(self) -> bool:
        return self.type() == RangeType.DENSE.value

    def is_compressed(self) -> bool:
        return self.type() == RangeType.COMPRESSED.value

    def is_block_compressed(self) -> bool:
        return self.type() == RangeType.BLOCK_COMPRESSED.value

    def is_mergeable(self, other: "BaseRange") -> bool:
        return self.type() == other.type() and self.end_tok == other.start_tok

    def merge(self, other: "BaseRange") -> "BaseRange":
        raise NotImplementedError("BaseClass")

    @classmethod
    def from_raw(cls, start_tok: int, end_tok: int, *args, **kwargs) -> "BaseRange":
        raise NotImplementedError("BaseClass")

    @classmethod
    def from_empty(cls, pos: int) -> "BaseRange":
        raise NotImplementedError("BaseClass")


@dataclass
class DenseRange(BaseRange):
    data: torch.Tensor  # shape (b, hc, seq, hd)

    def get(self) -> torch.Tensor:
        return self.data

    def type(self) -> str:
        return RangeType.DENSE.value

    def merge(self, other: "DenseRange") -> "DenseRange":
        assert self.is_mergeable(other), "Ranges are not mergeable"
        merged_data = torch.cat([self.data, other.data], dim=-2)
        return DenseRange(start_tok=self.start_tok, end_tok=other.end_tok, data=merged_data)

    @classmethod
    def from_raw(cls, start_tok: int, end_tok: int, data: torch.Tensor) -> "DenseRange":
        assert (
            data.shape[-2] == end_tok - start_tok
        ), f"Data shape does not match the token range, found {data.shape[-2]}, should be {end_tok - start_tok} {start_tok} {end_tok}"
        return cls(start_tok=start_tok, end_tok=end_tok, data=data)

    @classmethod
    def from_empty(cls, pos: int) -> "DenseRange":
        return cls(start_tok=pos, end_tok=pos, data=torch.empty(0, device="cuda", dtype=torch.float16))


@dataclass
class CompressedRange(BaseRange):
    data: tuple[torch.Tensor, torch.Tensor]  # (non-zero, meta)

    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data

    def type(self) -> str:
        return RangeType.COMPRESSED.value

    def merge(self, other: "CompressedRange") -> "CompressedRange":
        assert self.is_mergeable(other), "Ranges are not mergeable"
        non_zero_1, meta_1 = self.data
        non_zero_2, meta_2 = other.data
        merged_non_zero = torch.cat([non_zero_1, non_zero_2], dim=-2)
        merged_meta = torch.cat([meta_1, meta_2], dim=-2)
        return CompressedRange(start_tok=self.start_tok, end_tok=other.end_tok, data=(merged_non_zero, merged_meta))

    @classmethod
    def from_raw(cls, start_tok: int, end_tok: int, data: tuple[torch.Tensor, torch.Tensor]) -> "CompressedRange":
        assert (
            data[0].shape[-2] == end_tok - start_tok
            or data[0].shape[-2] == (end_tok - start_tok) // 2  # key stays the same, value should be halved
        ), f"Data shape does not match the token range, found {data[0].shape[-2]}, should be {end_tok - start_tok} or {(end_tok - start_tok) // 2}"
        return cls(start_tok=start_tok, end_tok=end_tok, data=data)

    @classmethod
    def from_empty(cls, pos: int) -> "CompressedRange":
        return cls(
            start_tok=pos,
            end_tok=pos,
            data=(
                torch.empty(0, device="cuda", dtype=torch.float16),
                torch.empty(0, device="cuda", dtype=torch.float16),
            ),
        )


@dataclass
class BlockCompressedRange(BaseRange):
    dense_blocks: torch.Tensor
    sparse_blocks: torch.Tensor
    meta_blocks: torch.Tensor
    page_idx: torch.Tensor

    def get(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dense_blocks, self.sparse_blocks, self.meta_blocks, self.page_idx

    def type(self):
        return RangeType.BLOCK_COMPRESSED.value

    def merge(self, other: "BlockCompressedRange") -> "BlockCompressedRange":
        assert self.is_mergeable(
            other
        ), f"Ranges are not mergeable, self range: {self.start_tok}-{self.end_tok}, other range: {other.start_tok}-{other.end_tok}"
        merged_dense_blocks = to_tl_stride(torch.cat([self.dense_blocks, other.dense_blocks], dim=2))
        merged_sparse_blocks = to_tl_stride(torch.cat([self.sparse_blocks, other.sparse_blocks], dim=2))
        merged_meta_blocks = to_tl_stride(torch.cat([self.meta_blocks, other.meta_blocks], dim=2))
        merged_page_idx = torch.cat([self.page_idx, other.page_idx], dim=2)
        return BlockCompressedRange(
            start_tok=self.start_tok,
            end_tok=other.end_tok,
            dense_blocks=merged_dense_blocks,
            sparse_blocks=merged_sparse_blocks,
            meta_blocks=merged_meta_blocks,
            page_idx=merged_page_idx,
        )

    @classmethod
    def from_raw(
        cls,
        start_tok: int,
        end_tok: int,
        dense_blocks: torch.Tensor,
        sparse_blocks: torch.Tensor,
        meta_blocks: torch.Tensor,
        page_idx: torch.Tensor,
    ) -> "BlockCompressedRange":
        return cls(
            start_tok=start_tok,
            end_tok=end_tok,
            dense_blocks=dense_blocks,
            sparse_blocks=sparse_blocks,
            meta_blocks=meta_blocks,
            page_idx=page_idx,
        )

    @classmethod
    def from_empty(cls, pos: int) -> "BlockCompressedRange":
        return cls(
            start_tok=pos,
            end_tok=pos,
            dense_blocks=torch.empty(0, device="cuda", dtype=torch.float16),
            sparse_blocks=torch.empty(0, device="cuda", dtype=torch.float16),
            meta_blocks=torch.empty(0, device="cuda", dtype=torch.float16),
            page_idx=torch.empty(0, device="cuda", dtype=torch.int16),
        )


@dataclass
class CompressedCacheProto:
    sink: DenseRange
    compressed: CompressedRange
    local_window: DenseRange

    def __post_init__(self):
        # check range is continuous
        assert self.sink.start_tok == 0, "Sink start_tok must be 0"
        assert self.sink.end_tok == self.compressed.start_tok, "Sink end_tok must be equal to compressed start_tok"
        assert (
            self.compressed.end_tok == self.local_window.start_tok
        ), "Compressed end_tok must be equal to local_window start_tok"


class PrefillKVDecodeKVCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_flashattn_sp_prefill_kv_decode_kv"
    MIN_BLK_SIZE = E_FACTOR * 2

    def __init__(
        self,
        sink: int,
        local_window: int,
    ):
        super().__init__()
        self.sink = sink
        self.local_window = local_window
        self.key_cache: list[CompressedCacheProto] = []
        self.value_cache: list[CompressedCacheProto] = []

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
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += new_tokens

        prune_len = round(self._seen_tokens - self.sink - self.local_window, self.MIN_BLK_SIZE)
        prune_remnant = self._seen_tokens - self.sink - self.local_window - prune_len
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # there should be no skip layers
            key_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
            )
            key_compressed = CompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                data=tilelang_prune_and_compress_key(key_states[:, :, self.sink : self.sink + prune_len, :]),
            )
            # key_compressed = DenseRange.from_raw(
            #     start_tok=self.sink,
            #     end_tok=self.sink + prune_len,
            #     data=key_states[:, :, self.sink : self.sink + prune_len, :],
            # )
            key_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            value_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=value_states[:, :, : self.sink, :].contiguous()
            )
            value_compressed = CompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                data=tilelang_prune_and_compress_value(value_states[:, :, self.sink : self.sink + prune_len, :]),
            )
            # value_compressed = DenseRange.from_raw(
            #     start_tok=self.sink,
            #     end_tok=self.sink + prune_len,
            #     data=value_states[:, :, self.sink : self.sink + prune_len, :]
            # )
            value_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=value_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            self.key_cache.append(
                CompressedCacheProto(
                    sink=key_sink,
                    compressed=key_compressed,
                    local_window=key_local_window,
                )
            )
            self.value_cache.append(
                CompressedCacheProto(
                    sink=value_sink,
                    compressed=value_compressed,
                    local_window=value_local_window,
                )
            )
            # key_states.storage().resize_(0)
            # value_states.storage().resize_(0)
        else:

            if prune_remnant == 0:
                key_local_window = self.key_cache[layer_idx].local_window
                self.key_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                key_local_window = key_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )
                self.key_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=key_local_window.data[:, :, key_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.key_cache[layer_idx].compressed = self.key_cache[layer_idx].compressed.merge(
                    CompressedRange.from_raw(
                        start_tok=key_local_window.start_tok,
                        end_tok=key_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=tilelang_prune_and_compress_key(key_local_window.data[:, :, : self.MIN_BLK_SIZE, :]),
                    )
                )
            else:
                self.key_cache[layer_idx].local_window = self.key_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )

            if prune_remnant == 0:
                value_local_window = self.value_cache[layer_idx].local_window
                self.value_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                value_local_window = value_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )
                self.value_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=value_local_window.data[:, :, value_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.value_cache[layer_idx].compressed = self.value_cache[layer_idx].compressed.merge(
                    CompressedRange.from_raw(
                        start_tok=value_local_window.start_tok,
                        end_tok=value_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=tilelang_prune_and_compress_value(value_local_window.data[:, :, : self.MIN_BLK_SIZE, :]),
                    )
                )
            else:
                self.value_cache[layer_idx].local_window = self.value_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.sink.data.element_size() * key_layer_cache.sink.data.numel()

            non_zero, meta = key_layer_cache.compressed.data
            key_bytes += non_zero.element_size() * non_zero.numel()
            key_bytes += meta.element_size() * meta.numel()

            key_bytes += key_layer_cache.local_window.data.element_size() * key_layer_cache.local_window.data.numel()

        value_bytes = 0
        for value_layer_cache in self.value_cache:
            value_bytes += value_layer_cache.sink.data.element_size() * value_layer_cache.sink.data.numel()

            non_zero, meta = value_layer_cache.compressed.data
            value_bytes += non_zero.element_size() * non_zero.numel()
            value_bytes += meta.element_size() * meta.numel()

            value_bytes += (
                value_layer_cache.local_window.data.element_size() * value_layer_cache.local_window.data.numel()
            )

        return key_bytes, value_bytes


@dataclass
class BlockCompressedCacheProto:
    sink: DenseRange
    compressed: BlockCompressedRange
    local_window: DenseRange

    def __post_init__(self):
        # check range is continuous
        assert self.sink.start_tok == 0, "Sink start_tok must be 0"
        assert self.sink.end_tok == self.compressed.start_tok, "Sink end_tok must be equal to compressed start_tok"
        assert (
            self.compressed.end_tok == self.local_window.start_tok
        ), "Compressed end_tok must be equal to local_window start_tok"


class HieraSparseCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_blockattn_sp_forward"

    def __init__(
        self,
        block_size: int,
        key_prune_ratio: float,
        value_prune_ratio: float,
        sink: int,
        local_window: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.key_prune_ratio = key_prune_ratio
        self.value_prune_ratio = value_prune_ratio
        self.sink = sink
        self.local_window = local_window
        self.key_cache: list[BlockCompressedCacheProto] = []
        self.value_cache: list[BlockCompressedCacheProto] = []

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
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += new_tokens

        prune_len = round(self._seen_tokens - self.sink - self.local_window, self.block_size)
        prune_remnant = max(self._seen_tokens - self.sink - self.local_window - prune_len, 0)

        # Update the cache
        if len(self.key_cache) <= layer_idx:
            key_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
            )

            idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_key(
                K=key_states[:, :, self.sink : self.sink + prune_len, :],
                prune_ratio=self.key_prune_ratio,
                block_s=self.block_size,
                # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
            )

            key_compressed = BlockCompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                dense_blocks=dense_blocks,
                sparse_blocks=sparse_blocks,
                meta_blocks=meta_blocks,
                page_idx=idx_map,
            )

            key_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            value_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=value_states[:, :, : self.sink, :].contiguous()
            )
            idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_value(
                V=value_states[:, :, self.sink : self.sink + prune_len, :],
                prune_ratio=self.value_prune_ratio,
                block_s=self.block_size,
                # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
            )

            value_compressed = BlockCompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                dense_blocks=dense_blocks,
                sparse_blocks=sparse_blocks,
                meta_blocks=meta_blocks,
                page_idx=idx_map,
            )

            value_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=value_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            self.key_cache.append(
                BlockCompressedCacheProto(
                    sink=key_sink,
                    compressed=key_compressed,
                    local_window=key_local_window,
                )
            )
            self.value_cache.append(
                BlockCompressedCacheProto(
                    sink=value_sink,
                    compressed=value_compressed,
                    local_window=value_local_window,
                )
            )

        else:
            # key
            if prune_remnant == 0:
                key_local_window = self.key_cache[layer_idx].local_window
                self.key_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                key_local_window = key_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )
                self.key_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=key_local_window.data[:, :, key_local_window.data.size(2) - self.local_window :, :],
                )

                idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_key(
                    K=key_local_window.data[:, :, : self.block_size, :],
                    prune_ratio=1,
                    block_s=self.block_size,
                    # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                    # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                )

                # merge dense part into compressed part
                self.key_cache[layer_idx].compressed = self.key_cache[layer_idx].compressed.merge(
                    BlockCompressedRange.from_raw(
                        start_tok=key_local_window.start_tok,
                        end_tok=key_local_window.start_tok + self.block_size,
                        dense_blocks=dense_blocks,
                        sparse_blocks=sparse_blocks,
                        meta_blocks=meta_blocks,
                        page_idx=idx_map,
                    )
                )
            else:
                self.key_cache[layer_idx].local_window = self.key_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )

            # value
            if prune_remnant == 0:
                value_local_window = self.value_cache[layer_idx].local_window
                self.value_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                value_local_window = value_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )
                self.value_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=value_local_window.data[:, :, value_local_window.data.size(2) - self.local_window :, :],
                )

                idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_value(
                    V=value_local_window.data[:, :, : self.block_size, :],
                    prune_ratio=1,
                    block_s=self.block_size,
                    # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                    # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                )

                self.value_cache[layer_idx].compressed = self.value_cache[layer_idx].compressed.merge(
                    BlockCompressedRange.from_raw(
                        start_tok=value_local_window.start_tok,
                        end_tok=value_local_window.start_tok + self.block_size,
                        dense_blocks=dense_blocks,
                        sparse_blocks=sparse_blocks,
                        meta_blocks=meta_blocks,
                        page_idx=idx_map,
                    )
                )
            else:
                self.value_cache[layer_idx].local_window = self.value_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.sink.data.element_size() * key_layer_cache.sink.data.numel()

            dense_blocks, sparse_blocks, meta_blocks, page_idx = key_layer_cache.compressed.get()
            key_bytes += dense_blocks.element_size() * dense_blocks.numel()
            key_bytes += sparse_blocks.element_size() * sparse_blocks.numel()
            key_bytes += meta_blocks.element_size() * meta_blocks.numel()
            key_bytes += page_idx.element_size() * page_idx.numel()

            key_bytes += key_layer_cache.local_window.data.element_size() * key_layer_cache.local_window.data.numel()
        value_bytes = 0
        for value_layer_cache in self.value_cache:
            key_bytes += value_layer_cache.sink.data.element_size() * value_layer_cache.sink.data.numel()
            dense_blocks, sparse_blocks, meta_blocks, page_idx = value_layer_cache.compressed.get()
            value_bytes += dense_blocks.element_size() * dense_blocks.numel()
            value_bytes += sparse_blocks.element_size() * sparse_blocks.numel()
            value_bytes += meta_blocks.element_size() * meta_blocks.numel()
            value_bytes += page_idx.element_size() * page_idx.numel()

            value_bytes += (
                value_layer_cache.local_window.data.element_size() * value_layer_cache.local_window.data.numel()
            )
        return key_bytes, value_bytes


class HieraSparseDecodeCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_blockattn_sp_forward"

    def __init__(
        self,
        block_size: int,
        key_prune_ratio: float,
        value_prune_ratio: float,
        sink: int,
        local_window: int,
    ):
        super().__init__()
        self.block_size = block_size
        self.key_prune_ratio = key_prune_ratio
        self.value_prune_ratio = value_prune_ratio
        self.sink = sink
        self.local_window = local_window
        self.key_cache: list[BlockCompressedCacheProto] = []
        self.value_cache: list[BlockCompressedCacheProto] = []

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

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(
                BlockCompressedCacheProto(
                    sink=DenseRange.from_raw(start_tok=0, end_tok=self._seen_tokens, data=key_states),
                    compressed=BlockCompressedRange.from_empty(self._seen_tokens),
                    local_window=DenseRange.from_empty(self._seen_tokens),
                )
            )
            self.value_cache.append(
                BlockCompressedCacheProto(
                    sink=DenseRange.from_raw(start_tok=0, end_tok=self._seen_tokens, data=value_states),
                    compressed=BlockCompressedRange.from_empty(self._seen_tokens),
                    local_window=DenseRange.from_empty(self._seen_tokens),
                )
            )
        else:
            self.key_cache[layer_idx].local_window = self.key_cache[layer_idx].local_window.merge(
                DenseRange.from_raw(
                    start_tok=self._seen_tokens - new_tokens,
                    end_tok=self._seen_tokens,
                    data=key_states,
                )
            )
            self.value_cache[layer_idx].local_window = self.value_cache[layer_idx].local_window.merge(
                DenseRange.from_raw(
                    start_tok=self._seen_tokens - new_tokens,
                    end_tok=self._seen_tokens,
                    data=value_states,
                )
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_after_attn(self, query_states, layer_idx):
        new_tokens = query_states.shape[-2]
        is_decode = new_tokens == 1
        prune_len = round(self._seen_tokens - self.sink - self.local_window, self.block_size)
        prune_remnant = max(self._seen_tokens - self.sink - self.local_window - prune_len, 0)

        if not is_decode:
            key_states = self.key_cache[layer_idx].sink.data
            self.key_cache[layer_idx] = None  # free ref
            key_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
            )

            idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_key(
                K=key_states[:, :, self.sink : self.sink + prune_len, :],
                prune_ratio=self.key_prune_ratio,
                block_s=self.block_size,
                # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
            )

            key_compressed = BlockCompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                dense_blocks=dense_blocks,
                sparse_blocks=sparse_blocks,
                meta_blocks=meta_blocks,
                page_idx=idx_map,
            )

            key_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            value_states = self.value_cache[layer_idx].sink.data
            self.value_cache[layer_idx] = None  # free ref

            value_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=value_states[:, :, : self.sink, :].contiguous()
            )
            idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_value(
                V=value_states[:, :, self.sink : self.sink + prune_len, :],
                prune_ratio=self.value_prune_ratio,
                block_s=self.block_size,
                # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
            )

            value_compressed = BlockCompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                dense_blocks=dense_blocks,
                sparse_blocks=sparse_blocks,
                meta_blocks=meta_blocks,
                page_idx=idx_map,
            )

            value_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=value_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            self.key_cache[layer_idx] = BlockCompressedCacheProto(
                sink=key_sink,
                compressed=key_compressed,
                local_window=key_local_window,
            )
            self.value_cache[layer_idx] = BlockCompressedCacheProto(
                sink=value_sink,
                compressed=value_compressed,
                local_window=value_local_window,
            )
        else:
            if prune_remnant == 0:
                key_local_window = self.key_cache[layer_idx].local_window
                self.key_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=key_local_window.data[:, :, key_local_window.data.size(2) - self.local_window :, :],
                )

                idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_key(
                    K=key_local_window.data[:, :, : self.block_size, :],
                    prune_ratio=1,
                    block_s=self.block_size,
                    # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                    # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                )

                # merge dense part into compressed part
                self.key_cache[layer_idx].compressed = self.key_cache[layer_idx].compressed.merge(
                    BlockCompressedRange.from_raw(
                        start_tok=key_local_window.start_tok,
                        end_tok=key_local_window.start_tok + self.block_size,
                        dense_blocks=dense_blocks,
                        sparse_blocks=sparse_blocks,
                        meta_blocks=meta_blocks,
                        page_idx=idx_map,
                    )
                )

            # value
            if prune_remnant == 0:
                value_local_window = self.value_cache[layer_idx].local_window
                self.value_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=value_local_window.data[:, :, value_local_window.data.size(2) - self.local_window :, :],
                )

                idx_map, dense_blocks, sparse_blocks, meta_blocks = tilelang_block_compress_value(
                    V=value_local_window.data[:, :, : self.block_size, :],
                    prune_ratio=1,
                    block_s=self.block_size,
                    # num_sink_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                    # num_local_blocks=0,  # NOTE: due to fully compressed value, we calculate this part use dense attn
                )

                self.value_cache[layer_idx].compressed = self.value_cache[layer_idx].compressed.merge(
                    BlockCompressedRange.from_raw(
                        start_tok=value_local_window.start_tok,
                        end_tok=value_local_window.start_tok + self.block_size,
                        dense_blocks=dense_blocks,
                        sparse_blocks=sparse_blocks,
                        meta_blocks=meta_blocks,
                        page_idx=idx_map,
                    )
                )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.sink.data.element_size() * key_layer_cache.sink.data.numel()

            dense_blocks, sparse_blocks, meta_blocks, page_idx = key_layer_cache.compressed.get()
            key_bytes += dense_blocks.element_size() * dense_blocks.numel()
            key_bytes += sparse_blocks.element_size() * sparse_blocks.numel()
            key_bytes += meta_blocks.element_size() * meta_blocks.numel()
            key_bytes += page_idx.element_size() * page_idx.numel()

            key_bytes += key_layer_cache.local_window.data.element_size() * key_layer_cache.local_window.data.numel()
        value_bytes = 0
        for value_layer_cache in self.value_cache:
            key_bytes += value_layer_cache.sink.data.element_size() * value_layer_cache.sink.data.numel()
            dense_blocks, sparse_blocks, meta_blocks, page_idx = value_layer_cache.compressed.get()
            value_bytes += dense_blocks.element_size() * dense_blocks.numel()
            value_bytes += sparse_blocks.element_size() * sparse_blocks.numel()
            value_bytes += meta_blocks.element_size() * meta_blocks.numel()
            value_bytes += page_idx.element_size() * page_idx.numel()

            value_bytes += (
                value_layer_cache.local_window.data.element_size() * value_layer_cache.local_window.data.numel()
            )
        return key_bytes, value_bytes


@dataclass
class PrefillVDecodeKVProto:
    sink: DenseRange
    compressed: Union[CompressedRange, DenseRange]
    local_window: DenseRange

    def __post_init__(self):
        # check range is continuous
        assert self.sink.start_tok == 0, "Sink start_tok must be 0"
        assert self.sink.end_tok == self.compressed.start_tok, "Sink end_tok must be equal to compressed start_tok"
        assert (
            self.compressed.end_tok == self.local_window.start_tok
        ), "Compressed end_tok must be equal to local_window start_tok"


class PrefillVDecodeKVCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_flashattn_sp_prefill_v_decode_kv"

    MIN_BLK_SIZE = E_FACTOR * 2

    def __init__(
        self,
        sink: int,
        local_window: int,
    ):
        super().__init__()
        self.sink = sink
        self.local_window = local_window
        self.key_cache: list[Union[torch.Tensor, PrefillVDecodeKVProto]] = []
        self.value_cache: list[PrefillVDecodeKVProto] = []

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
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += new_tokens

        prune_len = round(self._seen_tokens - self.sink - self.local_window, self.MIN_BLK_SIZE)
        prune_remnant = self._seen_tokens - self.sink - self.local_window - prune_len
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # key_sink = DenseRange.from_raw(
            #     start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
            # )
            # key_compressed = CompressedRange.from_raw(
            #     start_tok=self.sink,
            #     end_tok=self.sink + prune_len,
            #     data=tilelang_prune_and_compress_key(key_states[:, :, self.sink : self.sink + prune_len, :]),
            # )
            # key_local_window = DenseRange.from_raw(
            #     start_tok=self.sink + prune_len,
            #     end_tok=self._seen_tokens,
            #     data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
            # )

            value_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=value_states[:, :, : self.sink, :].contiguous()
            )
            value_compressed = CompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                data=tilelang_prune_and_compress_value(value_states[:, :, self.sink : self.sink + prune_len, :]),
            )
            value_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=value_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            self.key_cache.append(
                PrefillVDecodeKVProto(
                    sink=DenseRange.from_raw(
                        start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
                    ),
                    compressed=DenseRange.from_raw(
                        start_tok=self.sink,
                        end_tok=self.sink + prune_len,
                        data=key_states[:, :, self.sink : self.sink + prune_len, :].contiguous(),
                    ),
                    local_window=DenseRange.from_raw(
                        start_tok=self.sink + prune_len,
                        end_tok=self._seen_tokens,
                        data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
                    ),
                )
            )
            self.value_cache.append(
                PrefillVDecodeKVProto(
                    sink=value_sink,
                    compressed=value_compressed,
                    local_window=value_local_window,
                )
            )

        else:

            if prune_remnant == 0:
                key_local_window = self.key_cache[layer_idx].local_window
                self.key_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                key_local_window = key_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )
                self.key_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=key_local_window.data[:, :, key_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.key_cache[layer_idx].compressed = self.key_cache[layer_idx].compressed.merge(
                    CompressedRange.from_raw(
                        start_tok=key_local_window.start_tok,
                        end_tok=key_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=tilelang_prune_and_compress_key(key_local_window.data[:, :, : self.MIN_BLK_SIZE, :]),
                    )
                )
            else:
                self.key_cache[layer_idx].local_window = self.key_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )

            if prune_remnant == 0:
                value_local_window = self.value_cache[layer_idx].local_window
                self.value_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                value_local_window = value_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )
                self.value_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=value_local_window.data[:, :, value_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.value_cache[layer_idx].compressed = self.value_cache[layer_idx].compressed.merge(
                    CompressedRange.from_raw(
                        start_tok=value_local_window.start_tok,
                        end_tok=value_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=tilelang_prune_and_compress_value(value_local_window.data[:, :, : self.MIN_BLK_SIZE, :]),
                    )
                )
            else:
                self.value_cache[layer_idx].local_window = self.value_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_after_attn(self, query_states, layer_idx):
        new_tokens = query_states.shape[-2]
        is_decode = new_tokens == 1
        if not is_decode:
            self.key_cache[layer_idx] = PrefillVDecodeKVProto(
                sink=self.key_cache[layer_idx].sink,
                compressed=CompressedRange.from_raw(
                    start_tok=self.key_cache[layer_idx].compressed.start_tok,
                    end_tok=self.key_cache[layer_idx].compressed.end_tok,
                    data=tilelang_prune_and_compress_key(self.key_cache[layer_idx].compressed.data),
                ),
                local_window=self.key_cache[layer_idx].local_window,
            )

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def memory_usage_bytes(self) -> Tuple[int, int]:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.sink.data.element_size() * key_layer_cache.sink.data.numel()

            non_zero, meta = key_layer_cache.compressed.data
            key_bytes += non_zero.element_size() * non_zero.numel()
            key_bytes += meta.element_size() * meta.numel()

            key_bytes += key_layer_cache.local_window.data.element_size() * key_layer_cache.local_window.data.numel()

        value_bytes = 0
        for value_layer_cache in self.value_cache:
            value_bytes += value_layer_cache.sink.data.element_size() * value_layer_cache.sink.data.numel()

            non_zero, meta = value_layer_cache.compressed.data
            value_bytes += non_zero.element_size() * non_zero.numel()
            value_bytes += meta.element_size() * meta.numel()

            value_bytes += (
                value_layer_cache.local_window.data.element_size() * value_layer_cache.local_window.data.numel()
            )

        return key_bytes, value_bytes


@dataclass
class PrefillVDecodeVProto:
    sink: DenseRange
    compressed: Union[CompressedRange, DenseRange]
    local_window: DenseRange

    def __post_init__(self):
        # check range is continuous
        assert self.sink.start_tok == 0, "Sink start_tok must be 0"
        assert self.sink.end_tok == self.compressed.start_tok, "Sink end_tok must be equal to compressed start_tok"
        assert (
            self.compressed.end_tok == self.local_window.start_tok
        ), "Compressed end_tok must be equal to local_window start_tok"


class PrefillVDecodeVCache(BaseCache):
    ATTN_IMPLEMENTATION = "hf_flashattn_sp_prefill_v_decode_v"

    MIN_BLK_SIZE = E_FACTOR * 2

    def __init__(
        self,
        sink: int,
        local_window: int,
    ):
        super().__init__()
        self.sink = sink
        self.local_window = local_window
        self.key_cache: list[PrefillVDecodeVProto] = []
        self.value_cache: list[PrefillVDecodeVProto] = []

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
        new_tokens = key_states.shape[-2]
        if layer_idx == 0:
            # NOTE: we don't prune key along seq dim, so it's safe to use key_states here
            # as layer 0 is unlikely to be skipped
            self._seen_tokens += new_tokens

        prune_len = round(self._seen_tokens - self.sink - self.local_window, self.MIN_BLK_SIZE)
        prune_remnant = self._seen_tokens - self.sink - self.local_window - prune_len
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            # key_sink = DenseRange.from_raw(
            #     start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
            # )
            # key_compressed = CompressedRange.from_raw(
            #     start_tok=self.sink,
            #     end_tok=self.sink + prune_len,
            #     data=tilelang_prune_and_compress_key(key_states[:, :, self.sink : self.sink + prune_len, :]),
            # )
            # key_local_window = DenseRange.from_raw(
            #     start_tok=self.sink + prune_len,
            #     end_tok=self._seen_tokens,
            #     data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
            # )

            value_sink = DenseRange.from_raw(
                start_tok=0, end_tok=self.sink, data=value_states[:, :, : self.sink, :].contiguous()
            )
            value_compressed = CompressedRange.from_raw(
                start_tok=self.sink,
                end_tok=self.sink + prune_len,
                data=tilelang_prune_and_compress_value(value_states[:, :, self.sink : self.sink + prune_len, :]),
            )
            value_local_window = DenseRange.from_raw(
                start_tok=self.sink + prune_len,
                end_tok=self._seen_tokens,
                data=value_states[:, :, self.sink + prune_len :, :].contiguous(),
            )

            self.key_cache.append(
                PrefillVDecodeVProto(
                    sink=DenseRange.from_raw(
                        start_tok=0, end_tok=self.sink, data=key_states[:, :, : self.sink, :].contiguous()
                    ),
                    compressed=DenseRange.from_raw(
                        start_tok=self.sink,
                        end_tok=self.sink + prune_len,
                        data=key_states[:, :, self.sink : self.sink + prune_len, :].contiguous(),
                    ),
                    local_window=DenseRange.from_raw(
                        start_tok=self.sink + prune_len,
                        end_tok=self._seen_tokens,
                        data=key_states[:, :, self.sink + prune_len :, :].contiguous(),
                    ),
                )
            )
            self.value_cache.append(
                PrefillVDecodeVProto(
                    sink=value_sink,
                    compressed=value_compressed,
                    local_window=value_local_window,
                )
            )

        else:

            if prune_remnant == 0:
                key_local_window = self.key_cache[layer_idx].local_window
                self.key_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                key_local_window = key_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )
                self.key_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=key_local_window.data[:, :, key_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.key_cache[layer_idx].compressed = self.key_cache[layer_idx].compressed.merge(
                    DenseRange.from_raw(
                        start_tok=key_local_window.start_tok,
                        end_tok=key_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=key_local_window.data[:, :, : self.MIN_BLK_SIZE, :],
                    )
                )
            else:
                self.key_cache[layer_idx].local_window = self.key_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=key_states,
                    )
                )

            if prune_remnant == 0:
                value_local_window = self.value_cache[layer_idx].local_window
                self.value_cache[layer_idx].local_window = DenseRange.from_empty(self._seen_tokens)  # free ref
                value_local_window = value_local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )
                self.value_cache[layer_idx].local_window = DenseRange.from_raw(
                    start_tok=self._seen_tokens - self.local_window,
                    end_tok=self._seen_tokens,
                    data=value_local_window.data[:, :, value_local_window.data.size(2) - self.local_window :, :],
                )

                # merge dense part into compressed part
                self.value_cache[layer_idx].compressed = self.value_cache[layer_idx].compressed.merge(
                    CompressedRange.from_raw(
                        start_tok=value_local_window.start_tok,
                        end_tok=value_local_window.start_tok + self.MIN_BLK_SIZE,
                        data=tilelang_prune_and_compress_value(value_local_window.data[:, :, : self.MIN_BLK_SIZE, :]),
                    )
                )
            else:
                self.value_cache[layer_idx].local_window = self.value_cache[layer_idx].local_window.merge(
                    DenseRange.from_raw(
                        start_tok=self._seen_tokens - new_tokens,
                        end_tok=self._seen_tokens,
                        data=value_states,
                    )
                )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_after_attn(self, query_states, layer_idx):
        pass

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens

    def memory_usage_bytes(self) -> int:
        key_bytes = 0
        for key_layer_cache in self.key_cache:
            key_bytes += key_layer_cache.sink.data.element_size() * key_layer_cache.sink.data.numel()
            key_bytes += key_layer_cache.compressed.data.element_size() * key_layer_cache.compressed.data.numel()
            key_bytes += key_layer_cache.local_window.data.element_size() * key_layer_cache.local_window.data.numel()

        value_bytes = 0
        for value_layer_cache in self.value_cache:
            value_bytes += value_layer_cache.sink.data.element_size() * value_layer_cache.sink.data.numel()

            non_zero, meta = value_layer_cache.compressed.data
            value_bytes += non_zero.element_size() * non_zero.numel()
            value_bytes += meta.element_size() * meta.numel()

            value_bytes += (
                value_layer_cache.local_window.data.element_size() * value_layer_cache.local_window.data.numel()
            )

        return key_bytes, value_bytes

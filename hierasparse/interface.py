from typing import Optional, Tuple

import torch
from torch.profiler import record_function

from hierasparse.caches.compressed_cache import (
    BlockCompressedCacheProto,
    CompressedCacheProto,
    PrefillVDecodeKVProto,
    PrefillVDecodeVProto,
)
from hierasparse.operators import (
    blockattn_sp_kv_op,
    blockdecode_sp_mk_mv_op,
    flashattn_sp_kv_op,
    flashattn_sp_v_op,
    flashdecode_sp_kv_op,
    flashdecode_sp_v_op,
    naive_attn_op,
)
from hierasparse.utils import update_out_and_lse


def hf_flashattn_sp_prefill_kv_decode_kv(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: CompressedCacheProto,
    value: CompressedCacheProto,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, CompressedCacheProto) and isinstance(value, CompressedCacheProto)

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output = None
            lse = None
            for key_cache, value_cache in zip(
                [key.sink, key.compressed, key.local_window],
                [value.sink, value.compressed, value.local_window],
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                assert key_cache.type() == value_cache.type(), f"Found {key_cache.type()} and {value_cache.type()}"
                if key_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query,
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=True,
                    )
                else:
                    assert key_cache.is_compressed()
                    K_SP, K_E = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    output_partial, lse_partial = flashdecode_sp_kv_op(query.squeeze(2), K_SP, K_E, V_SP, V_E)
                    output_partial = output_partial.unsqueeze(2)
                    lse_partial = lse_partial.unsqueeze(2)

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output, lse = update_out_and_lse(
                        out=output,
                        lse=lse,
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )
    else:
        with record_function("prefill"):
            output = None
            lse = None
            for i, (key_cache, value_cache) in enumerate(
                zip(
                    [key.sink, key.compressed, key.local_window],
                    [value.sink, value.compressed, value.local_window],
                )
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                assert key_cache.type() == value_cache.type(), f"Found {key_cache.type()} and {value_cache.type()}"
                assert (
                    key_cache.start_tok == value_cache.start_tok and key_cache.end_tok == value_cache.end_tok
                ), f"Found {key_cache.start_tok}-{key_cache.end_tok} and {value_cache.start_tok}-{value_cache.end_tok}"

                if i == 0:  # sink
                    q_start = 0
                    q_end = seq_q
                elif i == 1:  # compressed or dense simulation
                    q_start = key_cache.start_tok
                    q_end = seq_q
                else:  # local
                    q_start = key_cache.start_tok
                    q_end = key_cache.end_tok

                if key_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query[:, :, q_start:q_end, :].contiguous(),
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=False,
                    )
                else:
                    K_SP, K_E = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    # TODO: the contiguous here need to be removed
                    output_partial, lse_partial = flashattn_sp_kv_op(
                        query[:, :, q_start:q_end, :].contiguous(), K_SP, K_E, V_SP, V_E, True
                    )
                    lse_partial = lse_partial

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output[:, :, q_start:q_end, :], lse[:, :, q_start:q_end] = update_out_and_lse(
                        out=output[:, :, q_start:q_end, :],
                        lse=lse[:, :, q_start:q_end],
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )

    # output = output.half()
    # key_all = torch.concat([key.dense_sink.data, key.compressed.data, key.local_window.data], dim=2)
    # value_all = torch.concat([value.dense_sink.data, value.compressed.data, value.local_window.data], dim=2)
    # o_ref, _ = naive_attn(
    #     Q=query,
    #     K=key_all,
    #     V=value_all,
    #     is_decode=is_decode,
    # )
    # o_ref = flash_attn_func(
    #     query.transpose(1, 2), key_all.transpose(1, 2), value_all.transpose(1, 2), softmax_scale=None, causal=True
    # )
    # assert torch.allclose(output, o_ref), f"{output.shape=} diff {torch.abs(output - o_ref).max()=}"
    # assert not output.isnan().any(), f"Output has NaN values."
    return output.transpose(1, 2), None  # attn weight is none


def hf_blockattn_sp_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: BlockCompressedCacheProto,
    value: BlockCompressedCacheProto,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, BlockCompressedCacheProto) and isinstance(
        value, BlockCompressedCacheProto
    ), f"Found {type(key)} {type(value)}"

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output = None
            lse = None
            for key_cache, value_cache in zip(
                [key.sink, key.compressed, key.local_window],
                [value.sink, value.compressed, value.local_window],
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                if key_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query,
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=True,
                    )
                else:
                    assert key_cache.is_block_compressed()
                    K_dense_blocks, K_sparse_blocks, K_meta_blocks, K_page_idx = key_cache.get()
                    V_dense_blocks, V_sparse_blocks, V_meta_blocks, V_page_idx = value_cache.get()
                    block_N = K_dense_blocks.shape[3]
                    kv_len = K_page_idx.shape[2] * block_N
                    min((kv_len + 31) // 32, 8)
                    output_partial, lse_partial = blockdecode_sp_mk_mv_op(
                        query.squeeze(2),
                        K_dense_blocks,
                        K_sparse_blocks,
                        K_meta_blocks,
                        K_page_idx,
                        V_dense_blocks,
                        V_sparse_blocks,
                        V_meta_blocks,
                        V_page_idx,
                    )
                    output_partial = output_partial.unsqueeze(2)
                    lse_partial = lse_partial.unsqueeze(2)

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output, lse = update_out_and_lse(
                        out=output,
                        lse=lse,
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )
    else:
        with record_function("prefill"):
            output = None
            lse = None
            for i, (key_cache, value_cache) in enumerate(
                zip(
                    [key.sink, key.compressed, key.local_window],
                    [value.sink, value.compressed, value.local_window],
                )
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                if i == 0:  # sink
                    q_start = 0
                    q_end = seq_q
                elif i == 1:  # compressed or dense simulation
                    q_start = key_cache.start_tok
                    q_end = seq_q
                else:  # local
                    q_start = key_cache.start_tok
                    q_end = key_cache.end_tok

                if key_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query[:, :, q_start:q_end, :].contiguous(),
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=False,
                    )
                else:
                    K_dense_blocks, K_sparse_blocks, K_meta_blocks, K_page_idx = key_cache.get()
                    V_dense_blocks, V_sparse_blocks, V_meta_blocks, V_page_idx = value_cache.get()
                    output_partial, lse_partial = blockattn_sp_kv_op(
                        query[:, :, q_start:q_end, :].contiguous(),
                        K_dense_blocks,
                        K_sparse_blocks,
                        K_meta_blocks,
                        K_page_idx,
                        V_dense_blocks,
                        V_sparse_blocks,
                        V_meta_blocks,
                        V_page_idx,
                        True,
                    )

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output[:, :, q_start:q_end, :], lse[:, :, q_start:q_end] = update_out_and_lse(
                        out=output[:, :, q_start:q_end, :],
                        lse=lse[:, :, q_start:q_end],
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )

    return output.transpose(1, 2), None


def hf_chunked_k_dense_v_sp(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, torch.Tensor) and isinstance(value, tuple)

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output, _ = flashdecode_sp_v_op(
                query.squeeze(2),
                key,
                value[0],
                value[1],
            )
    else:
        with record_function("prefill"):
            output, _ = flashattn_sp_v_op(
                query,
                key,
                value[0],
                value[1],
                True,
            )

    return output.transpose(1, 2), None


def hf_chunked_k_sp_v_sp(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: Tuple[torch.Tensor, torch.Tensor],
    value: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, tuple) and isinstance(value, tuple)

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output, _ = flashdecode_sp_kv_op(
                query.squeeze(2),
                key[0],
                key[1],
                value[0],
                value[1],
            )
    else:
        with record_function("prefill"):
            output, _ = flashattn_sp_kv_op(
                query.contiguous(),
                key[0],
                key[1],
                value[0],
                value[1],
                is_causal=True,
                chunked_prefill=True,
            )
    return output.transpose(1, 2), None


def hf_flashattn_sp_prefill_v_decode_kv(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, PrefillVDecodeKVProto) and isinstance(value, PrefillVDecodeKVProto)

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output = None
            lse = None
            for key_cache, value_cache in zip(
                [key.sink, key.compressed, key.local_window],
                [value.sink, value.compressed, value.local_window],
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                assert key_cache.type() == value_cache.type(), f"Found {key_cache.type()} and {value_cache.type()}"
                if key_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query,
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=True,
                    )
                else:
                    assert key_cache.is_compressed()
                    K_SP, K_E = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    output_partial, lse_partial = flashdecode_sp_kv_op(query.squeeze(2), K_SP, K_E, V_SP, V_E)
                    output_partial = output_partial.unsqueeze(2)
                    lse_partial = lse_partial.unsqueeze(2)

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output, lse = update_out_and_lse(
                        out=output,
                        lse=lse,
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )
    else:
        with record_function("prefill"):
            output = None
            lse = None
            for i, (key_cache, value_cache) in enumerate(
                zip(
                    [key.sink, key.compressed, key.local_window],
                    [value.sink, value.compressed, value.local_window],
                )
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                assert (
                    key_cache.start_tok == value_cache.start_tok and key_cache.end_tok == value_cache.end_tok
                ), f"Found {key_cache.start_tok}-{key_cache.end_tok} and {value_cache.start_tok}-{value_cache.end_tok}"

                if i == 0:  # sink
                    q_start = 0
                    q_end = seq_q
                elif i == 1:  # compressed or dense simulation
                    q_start = key_cache.start_tok
                    q_end = seq_q
                else:  # local
                    q_start = key_cache.start_tok
                    q_end = key_cache.end_tok

                if key_cache.is_dense() and value_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query[:, :, q_start:q_end, :].contiguous(),
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=False,
                    )
                elif key_cache.is_dense() and value_cache.is_compressed():
                    K = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    # TODO: the contiguous here need to be removed
                    output_partial, lse_partial = flashattn_sp_v_op(
                        query[:, :, q_start:q_end, :].contiguous(), K, V_SP, V_E
                    )
                    lse_partial = lse_partial
                else:
                    raise ValueError(f"Unsupported cache type combination: {key_cache.type()} and {value_cache.type()}")

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output[:, :, q_start:q_end, :], lse[:, :, q_start:q_end] = update_out_and_lse(
                        out=output[:, :, q_start:q_end, :],
                        lse=lse[:, :, q_start:q_end],
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )

    # output = output.half()
    # key_all = torch.concat([key.dense_sink.data, key.compressed.data, key.local_window.data], dim=2)
    # value_all = torch.concat([value.dense_sink.data, value.compressed.data, value.local_window.data], dim=2)
    # o_ref, _ = naive_attn(
    #     Q=query,
    #     K=key_all,
    #     V=value_all,
    #     is_decode=is_decode,
    # )
    # o_ref = flash_attn_func(
    #     query.transpose(1, 2), key_all.transpose(1, 2), value_all.transpose(1, 2), softmax_scale=None, causal=True
    # )
    # assert torch.allclose(output, o_ref), f"{output.shape=} diff {torch.abs(output - o_ref).max()=}"
    # assert not output.isnan().any(), f"Output has NaN values."
    return output.transpose(1, 2), None  # attn weight is none


def hf_flashattn_sp_prefill_v_decode_v(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    assert attention_mask is None, f"Attention mask must be None, found {attention_mask}"
    assert dropout == 0.0, f"Dropout must be 0.0, found {dropout}"
    assert sliding_window is None, f"Sliding window must be None, found {sliding_window}"
    assert softcap is None, f"Softcap must be None, found {softcap}"
    assert isinstance(key, PrefillVDecodeVProto) and isinstance(
        value, PrefillVDecodeVProto
    ), f"Found {type(key)} {type(value)}"

    assert kwargs.get("causal", True) is True, f"Found {kwargs.get('causal')}"
    assert kwargs.get("return_attn_probs", True) is True, f"Found {kwargs.get('return_attn_probs')}"
    b, hn, seq_q, hd = query.shape
    assert -0.001 <= (scaling - 1 / (hd**0.5)) <= 0.001, f"Scaling must be {1 / (hd ** 0.5)}, found {scaling}"

    is_decode = seq_q == 1

    if is_decode:
        with record_function("decode"):
            output = None
            lse = None
            for key_cache, value_cache in zip(
                [key.sink, key.compressed, key.local_window],
                [value.sink, value.compressed, value.local_window],
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                if key_cache.is_dense() and value_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query,
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=True,
                    )
                else:
                    assert key_cache.is_dense() and value_cache.is_compressed()
                    K = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    output_partial, lse_partial = flashdecode_sp_v_op(query.squeeze(2), K, V_SP, V_E)
                    output_partial = output_partial.unsqueeze(2)
                    lse_partial = lse_partial.unsqueeze(2)

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output, lse = update_out_and_lse(
                        out=output,
                        lse=lse,
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )
    else:
        with record_function("prefill"):
            output = None
            lse = None
            for i, (key_cache, value_cache) in enumerate(
                zip(
                    [key.sink, key.compressed, key.local_window],
                    [value.sink, value.compressed, value.local_window],
                )
            ):
                if key_cache.is_empty() and value_cache.is_empty():
                    continue

                assert (
                    key_cache.start_tok == value_cache.start_tok and key_cache.end_tok == value_cache.end_tok
                ), f"Found {key_cache.start_tok}-{key_cache.end_tok} and {value_cache.start_tok}-{value_cache.end_tok}"

                if i == 0:  # sink
                    q_start = 0
                    q_end = seq_q
                elif i == 1:  # compressed or dense simulation
                    q_start = key_cache.start_tok
                    q_end = seq_q
                else:  # local
                    q_start = key_cache.start_tok
                    q_end = key_cache.end_tok

                if key_cache.is_dense() and value_cache.is_dense():
                    output_partial, lse_partial = naive_attn_op(
                        Q=query[:, :, q_start:q_end, :].contiguous(),
                        K=key_cache.get(),
                        V=value_cache.get(),
                        is_decode=False,
                    )
                elif key_cache.is_dense() and value_cache.is_compressed():
                    K = key_cache.get()
                    V_SP, V_E = value_cache.get()
                    # TODO: the contiguous here need to be removed
                    output_partial, lse_partial = flashattn_sp_v_op(
                        query[:, :, q_start:q_end, :].contiguous(), K, V_SP, V_E
                    )
                    lse_partial = lse_partial
                else:
                    raise ValueError(f"Unsupported cache type combination: {key_cache.type()} and {value_cache.type()}")

                if output is None:
                    output = output_partial
                    lse = lse_partial
                else:
                    output[:, :, q_start:q_end, :], lse[:, :, q_start:q_end] = update_out_and_lse(
                        out=output[:, :, q_start:q_end, :],
                        lse=lse[:, :, q_start:q_end],
                        block_out=output_partial,
                        block_lse=lse_partial,
                    )

    # output = output.half()
    # key_all = torch.concat([key.dense_sink.data, key.compressed.data, key.local_window.data], dim=2)
    # value_all = torch.concat([value.dense_sink.data, value.compressed.data, value.local_window.data], dim=2)
    # o_ref, _ = naive_attn(
    #     Q=query,
    #     K=key_all,
    #     V=value_all,
    #     is_decode=is_decode,
    # )
    # o_ref = flash_attn_func(
    #     query.transpose(1, 2), key_all.transpose(1, 2), value_all.transpose(1, 2), softmax_scale=None, causal=True
    # )
    # assert torch.allclose(output, o_ref), f"{output.shape=} diff {torch.abs(output - o_ref).max()=}"
    # assert not output.isnan().any(), f"Output has NaN values."
    return output.transpose(1, 2), None  # attn weight is none

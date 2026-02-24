from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from hierasparse.interface import (
    hf_blockattn_sp_forward,
    hf_chunked_k_dense_v_sp,
    hf_chunked_k_sp_v_sp,
    hf_flashattn_sp_prefill_kv_decode_kv,
    hf_flashattn_sp_prefill_v_decode_kv,
    hf_flashattn_sp_prefill_v_decode_v,
)

ALL_ATTENTION_FUNCTIONS.register("hf_flashattn_sp_prefill_kv_decode_kv", hf_flashattn_sp_prefill_kv_decode_kv)
ALL_ATTENTION_FUNCTIONS.register("hf_flashattn_sp_prefill_v_decode_kv", hf_flashattn_sp_prefill_v_decode_kv)
ALL_ATTENTION_FUNCTIONS.register("hf_blockattn_sp_forward", hf_blockattn_sp_forward)
ALL_ATTENTION_FUNCTIONS.register("hf_chunked_k_dense_v_sp", hf_chunked_k_dense_v_sp)
ALL_ATTENTION_FUNCTIONS.register("hf_chunked_k_sp_v_sp", hf_chunked_k_sp_v_sp)
ALL_ATTENTION_FUNCTIONS.register("hf_flashattn_sp_prefill_v_decode_v", hf_flashattn_sp_prefill_v_decode_v)

from hierasparse.models.modeling_llama import *
from hierasparse.models.modeling_mistral import *
from hierasparse.models.modeling_qwen3 import *


def get_model_cls(model_name: str):
    if "llama" in model_name.lower():
        from hierasparse.models.modeling_llama import LlamaForCausalLM

        return LlamaForCausalLM

    elif "qwen3" in model_name.lower():
        from hierasparse.models.modeling_qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM

    elif "mistral" in model_name.lower():
        from hierasparse.models.modeling_mistral import MistralForCausalLM

        return MistralForCausalLM
    else:
        raise ValueError(
            f"Unsupported model name: {model_name}, it should be easily implemented by passsing query states into kv cache update or otherwise remove it from cache and use with hf models directly."
        )

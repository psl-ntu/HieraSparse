import json
import os
import random
import warnings
from dataclasses import asdict

import numpy as np
import torch
from datasets import load_dataset
from eval import eval
from summarize import summarize
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer
from utils.process_args import ModelArguments, PruneArguments, process_args

from hierasparse.caches.compressed_cache import (
    HieraSparseCache,
    HieraSparseDecodeCache,
    PrefillKVDecodeKVCache,
    PrefillVDecodeKVCache,
)
from hierasparse.caches.simulator_cache import (
    DenseCache,
    FlashAttnSPSimulationCache,
    HieraSparseSimulationCache,
)
from hierasparse.models import get_model_cls


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    messages = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return prompt


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    prune_args: PruneArguments,
    model_args: ModelArguments,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True
            )
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_args.model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        context_length = input.input_ids.shape[-1]

        if prune_args.cache == "dense":
            past_key_values = DenseCache()
        elif prune_args.cache == "simu_flashattn_sp":
            past_key_values = FlashAttnSPSimulationCache(
                prune_key_prefill=prune_args.prune_key_prefill,
                prune_key_decode=prune_args.prune_key_decode,
                prune_value_prefill=prune_args.prune_value_prefill,
                prune_value_decode=prune_args.prune_value_decode,
                sink=prune_args.sink,
                local_window=prune_args.local_window,
                start_layer=prune_args.start_layer,
                end_layer=prune_args.end_layer,
            )
        elif prune_args.cache == "simu_blockattn_sp":
            past_key_values = HieraSparseSimulationCache(
                prune_key_prefill=prune_args.prune_key_prefill,
                prune_key_decode=prune_args.prune_key_decode,
                prune_value_prefill=prune_args.prune_value_prefill,
                prune_value_decode=prune_args.prune_value_decode,
                prune_key_prefill_ratio=prune_args.prune_key_prefill_ratio,
                prune_value_prefill_ratio=prune_args.prune_value_prefill_ratio,
                block_seq_size=prune_args.block_seq_size,
                sink=prune_args.sink,
                local_window=prune_args.local_window,
                start_layer=prune_args.start_layer,
            )
        elif prune_args.cache == "best_perf":
            past_key_values = PrefillKVDecodeKVCache(
                sink=prune_args.sink,
                local_window=prune_args.local_window,
            )
        elif prune_args.cache == "balanced":
            past_key_values = PrefillVDecodeKVCache(
                sink=prune_args.sink,
                local_window=prune_args.local_window,
            )
        elif prune_args.cache == "hierasparse":
            past_key_values = HieraSparseCache(
                block_size=prune_args.block_seq_size,
                key_prune_ratio=prune_args.prune_key_prefill_ratio,
                value_prune_ratio=prune_args.prune_value_prefill_ratio,
                sink=prune_args.sink,
                local_window=prune_args.local_window,
            )
        elif prune_args.cache == "hierasparse_decode":
            past_key_values = HieraSparseDecodeCache(
                block_size=prune_args.block_seq_size,
                key_prune_ratio=prune_args.prune_key_prefill_ratio,
                value_prune_ratio=prune_args.prune_value_prefill_ratio,
                sink=prune_args.sink,
                local_window=prune_args.local_window,
            )

        # NOTE: for sink + local >= seq_len, directly use dense cache
        # This is rare in the dataset and should not affect the overall results
        if prune_args.cache != "dense" and prune_args.sink + prune_args.local_window >= context_length:
            warnings.warn(
                f"Using DenseCache since sink + local_window >= context_length ({prune_args.sink} + {prune_args.local_window} >= {context_length})"
            )
            past_key_values = DenseCache()
            model.config._attn_implementation = "sdpa"

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                past_key_values=past_key_values,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            output = model.generate(
                **input,
                past_key_values=past_key_values,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        if prune_args.cache != "dense" and prune_args.sink + prune_args.local_window >= context_length:
            model.config._attn_implementation = get_attn_implementation(prune_args.cache)

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_attn_implementation(cache_type: str):
    if cache_type == "best_perf":
        return PrefillKVDecodeKVCache.ATTN_IMPLEMENTATION
    elif cache_type == "balanced":
        return PrefillVDecodeKVCache.ATTN_IMPLEMENTATION
    elif cache_type == "hierasparse":
        return HieraSparseCache.ATTN_IMPLEMENTATION
    elif cache_type == "hierasparse_decode":
        return HieraSparseCache.ATTN_IMPLEMENTATION
    else:
        return "flash_attention_2"


if __name__ == "__main__":
    seed_everything(42)

    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda")

    model_args, data_args, prune_args = process_args()
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=False)
    MODEL_CLS = get_model_cls(model_args.model_name)
    model = MODEL_CLS.from_pretrained(
        model_args.model_name,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation=get_attn_implementation(prune_args.cache),
    )

    max_length = model2maxlen[model_args.model_name.split("/")[-1]]
    if data_args.e:
        print("Evaluating on Extended Benchmark Set!")
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
        output_dir_prefix = "pred_e"
    else:
        if data_args.fast_eval:
            print("Using fast evaluation datasets!")
            datasets = [
                # "triviaqa",
                # "passage_retrieval_en",
                # "hotpotqa",
                # "multifieldqa_en",
                "2wikimqa",
                # "trec",
                # "multi_news",
            ]
            output_dir_prefix = "pred_fast"
        else:
            print("Using full benchmark datasets!")
            # time measured with fa2
            datasets = [
                "narrativeqa",
                "qasper",
                "multifieldqa_en",
                "hotpotqa",
                "2wikimqa",
                "musique",
                "gov_report",
                "qmsum",
                "multi_news",
                "trec",
                "triviaqa",
                "samsum",
                "passage_count",
                "passage_retrieval_en",
                "lcc",
                "repobench-p",
            ]
            output_dir_prefix = "pred"
    output_dir_prefix += f"/{model_args.model_name}"
    output_dir = output_dir_prefix + "/" + data_args.output_postfix
    os.makedirs(output_dir, exist_ok=True)
    print(f"Writing results to {output_dir}")

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    with open(f"{output_dir}/args.json", "w") as f:
        json.dump(
            {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                "prune_args": asdict(prune_args),
            },
            f,
            indent=4,
            sort_keys=True,
        )

    for dataset in datasets:
        if os.path.exists(f"{output_dir}/{dataset}.jsonl"):
            print(f"Skipping {dataset} since already exists.")
            continue
        print(f"Processing dataset: {dataset}")
        if data_args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            out_path = f"{output_dir}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")
            out_path = f"{output_dir}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        with torch.no_grad():
            preds = get_pred(
                model, tokenizer, data, max_length, max_gen, prompt_format, dataset, prune_args, model_args
            )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

    eval(output_dir, data_args.e)
    summarize(f"{output_dir}/result.json")

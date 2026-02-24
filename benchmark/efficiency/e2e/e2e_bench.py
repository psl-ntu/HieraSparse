import argparse
import json
import os
from contextlib import nullcontext

import numpy as np
import torch
from kernels import Mode, kernelize
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from hierasparse.caches.static_cache import (
    DenseStaticCache,
    KeyDenseValueSparseStaticCache,
    KeySparseValueSparseStaticCache,
)
from hierasparse.models import get_model_cls


def print_metrics(title, metrics):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    for key, value in metrics.items():
        print(f"{key:<40} | {value}")
    print(f"{'-'*60}\n")


def get_profiler():
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=False,
    )


def replicate_batch(input_ids, batch):
    return {
        "input_ids": input_ids["input_ids"].repeat(batch, 1),
        "attention_mask": input_ids["attention_mask"].repeat(batch, 1),
    }


def profile_prefill(model, input_ids_batched, cache_factory, args):
    if args.prefill_iterations <= 0:
        return (float("nan"), float("nan")), (float("nan"), float("nan"))

    model = model.model  # no need to profile the LM head
    input_ids = input_ids_batched["input_ids"]
    seq_len = input_ids.shape[1]

    def run_prefill(cache):
        for i in tqdm(range(0, seq_len, args.chunk_size), desc="Chunked Prefill"):
            input_ids_chunk = input_ids[:, i : min(i + args.chunk_size, seq_len)]
            with record_function(f"Chunk {i}"):
                model(
                    input_ids=input_ids_chunk,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )

    print("Prefill warming up...")
    for _ in range(args.warmup):
        cache = None  # free first
        cache = cache_factory()
        with torch.no_grad():
            run_prefill(cache)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.prefill_iterations)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.prefill_iterations)]
    print("Prefill eval...")
    torch.cuda.synchronize()
    for i in range(args.prefill_iterations):
        cache = None
        cache = cache_factory()
        start_events[i].record()
        with torch.no_grad(), record_function("prefill"):
            run_prefill(cache)
        end_events[i].record()

    end_events[-1].synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) / 1000.0 for i in range(args.prefill_iterations)]
    key_bytes, value_bytes = cache.memory_usage_bytes()

    return (key_bytes, value_bytes), (np.mean(times), np.std(times))


def profile_decode(model, input_ids_batched, cache_factory, args):

    if args.max_new_tokens <= 0:
        return float("nan"), float("nan")

    cache = cache_factory()
    input_ids = input_ids_batched["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        for i in range(0, seq_len, args.chunk_size):
            input_ids_chunk = input_ids[:, i : min(i + args.chunk_size, seq_len)]
            outputs = model(
                input_ids=input_ids_chunk,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    static_input_ids = pred_token_idx.clone()

    if args.cudagraph:
        graph = torch.cuda.CUDAGraph()
        model(
            input_ids=static_input_ids,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        with torch.cuda.graph(graph):
            model(
                input_ids=static_input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

        def run_step():
            graph.replay()

    else:

        def run_step():
            model(
                input_ids=static_input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

    print(f"Decode warming up...")
    for _ in range(args.warmup):
        with torch.no_grad():
            run_step()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.max_new_tokens)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(args.max_new_tokens)]
    print(f"Decode eval...")
    torch.cuda.synchronize()
    for i in tqdm(range(args.max_new_tokens), desc="Decode rep"):
        start_events[i].record()
        with torch.no_grad(), record_function("decode"):
            run_step()
        end_events[i].record()

    end_events[-1].synchronize()
    times = [start_events[i].elapsed_time(end_events[i]) / 1000.0 for i in range(args.max_new_tokens)]
    return np.mean(times), np.std(times)


def main(args):
    torch.set_default_dtype(torch.float16)

    MODEL_CLS = get_model_cls(args.model_name)
    if args.prompt is not None:
        prompt = args.prompt
    else:
        args.prompt_length = (args.prompt_length + args.chunk_size - 1) // args.chunk_size * args.chunk_size
        print(f"Using prompt length: {args.prompt_length}")
        prompt = "apple bear" * (args.prompt_length // 2 + 10)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.prompt_length).to("cuda")
    input_length = input_ids.input_ids.shape[-1]
    input_ids_batched = replicate_batch(input_ids, args.batch)

    if args.cache_type == "k_dense_v_sp":
        cache_cls = KeyDenseValueSparseStaticCache
        cache_kwargs = {}
    elif args.cache_type == "k_sp_v_sp":
        cache_cls = KeySparseValueSparseStaticCache
        cache_kwargs = {}
    else:
        raise ValueError(f"Unknown cache type: {args.cache_type}")

    attn_implementation = cache_cls.ATTN_IMPLEMENTATION

    # torch.cuda.memory._record_memory_history()
    # try:
    with torch.device("cuda"):
        model_compressed = MODEL_CLS(
            AutoConfig.from_pretrained(
                args.model_name,
                attn_implementation=attn_implementation,
            )
        )
        model_compressed.eval()
        model_compressed = kernelize(model_compressed, mode=Mode.INFERENCE)

    config = model_compressed.config
    static_cache_kwargs = {
        "bsz": args.batch,
        "chunk_size": args.chunk_size,
        "max_cache_len": args.prompt_length + args.max_new_tokens + 10,
        "num_kv_heads": (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        ),
        "head_dim": getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        "layers": config.num_hidden_layers,
    }
    cache_kwargs.update(static_cache_kwargs)

    cache_factory = lambda: cache_cls(**cache_kwargs)
    with get_profiler() if args.profile else nullcontext() as prof:
        (key_bytes, value_bytes), (t_prefill_mean, t_prefill_std) = profile_prefill(
            model_compressed, input_ids_batched, cache_factory, args
        )
        t_decode_mean, t_decode_std = profile_decode(model_compressed, input_ids_batched, cache_factory, args)

    if args.profile:
        prof.export_chrome_trace("e2e_bench_compressed_prof.json")

    del model_compressed
    torch.cuda.empty_cache()
    # finally:
    #     torch.cuda.memory._dump_snapshot("static_cache.pickle")
    metrics = {
        f"TTFT({args.cache_type})": f"{t_prefill_mean:.1f} ± {t_prefill_std:.2f}s",
        f"TPOT({args.cache_type})": f"{t_decode_mean * 1e3:.1f} ± {t_decode_std * 1e3:.2f} ms",
        "Prefill throughput": f"{args.batch * input_length / t_prefill_mean:.1f} tok/s",
        "Decode throughput": f"{args.batch / t_decode_mean:.1f} tok/s",
        "Key size": f"{key_bytes / (1024**3):.2f} GB",
        "Value size": f"{value_bytes / (1024**3):.2f} GB",
        "Max memory allocated": f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB",
    }
    print_metrics("HieraSparse Attention Performance", metrics)

    if args.run_baseline:

        with torch.device("cuda"):
            model_baseline = MODEL_CLS(
                AutoConfig.from_pretrained(
                    args.model_name,
                    attn_implementation=args.attn_implementation,
                    # num_hidden_layers=2,
                )
            )
            model_baseline.eval()
            model_baseline = kernelize(model_baseline, mode=Mode.INFERENCE)

        with get_profiler() if args.profile_baseline else nullcontext() as prof:
            (key_bytes_base, value_bytes_base), (t_prefill_mean_base, t_prefill_std_base) = profile_prefill(
                model_baseline, input_ids_batched, lambda: DenseStaticCache(**static_cache_kwargs), args
            )
            t_decode_mean_base, t_decode_std_base = profile_decode(
                model_baseline, input_ids_batched, lambda: DenseStaticCache(**static_cache_kwargs), args
            )

        if args.profile_baseline:
            prof.export_chrome_trace("e2e_bench_baseline_prof.json")

        metrics_baseline = {
            "TTFT(baseline)": f"{t_prefill_mean_base:.1f} ± {t_prefill_std_base:.2f}s",
            "TPOT(baseline)": f"{t_decode_mean_base * 1e3:.1f} ± {t_decode_std_base * 1e3:.2f} ms",
            "Prefill throughput": f"{args.batch * input_length / t_prefill_mean_base:.1f} tok/s",
            "Decode throughput": f"{args.batch / t_decode_mean_base:.1f} tok/s",
            "Key size": f"{key_bytes_base / (1024**3):.2f} GB",
            "Value size": f"{value_bytes_base / (1024**3):.2f} GB",
            "Max memory allocated": f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB",
        }
        print_metrics("Baseline Performance", metrics_baseline)

        del model_baseline
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.empty_cache()

        comparison = {
            "Speedup Prefill": f"{t_prefill_mean_base / t_prefill_mean:.2f}x",
            "Speedup Decode": f"{t_decode_mean_base / t_decode_mean:.2f}x",
            "Key compression rate": f"{key_bytes / key_bytes_base * 100:.2f}%",
            "Value compression rate": f"{value_bytes / value_bytes_base * 100:.2f}%",
        }
        print_metrics("Comparison", comparison)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    json_filename = os.path.join(
        output_dir, f"e2e_bench_b{args.batch}_p{args.prompt_length}_c{args.chunk_size}_{args.cache_type}.json"
    )
    final_results = {
        "config": vars(args),
        "sparse_attention_performance": metrics,
    }
    if args.run_baseline:
        final_results["baseline_performance"] = metrics_baseline
        final_results["comparison"] = comparison

    with open(json_filename, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Results saved to {json_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--prefill_iterations", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--prompt_length", type=int, default=32768)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--cache_type",
        type=str,
        choices=["k_dense_v_sp", "k_sp_v_sp"],
        default="k_sp_v_sp",
    )
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--profile_baseline", action="store_true", help="Enable profiling with torch.profiler")
    parser.add_argument("--profile", action="store_true", help="Enable profiling with torch.profiler")
    parser.add_argument("--cudagraph", action="store_true")
    args = parser.parse_args()
    main(args)

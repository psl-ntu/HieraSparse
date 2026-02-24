import argparse
import functools

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile, record_function

from hierasparse.caches.compressed_cache import (
    HieraSparseCache,
    PrefillKVDecodeKVCache,
    PrefillVDecodeVCache,
)
from hierasparse.caches.simulator_cache import DenseCache
from hierasparse.models.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)
from hierasparse.utils import RECORDED_FUNC


def record_func_with_name(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with record_function(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--kv_heads", type=int, default=8, help="Number of KV heads (default: same as heads)")
    parser.add_argument("--intermediate_size", type=int, default=14336, help="MLP intermediate size")

    parser.add_argument("--cache_type", type=str, choices=["dense", "best_perf", "best_acc"], default="dense")
    parser.add_argument("--sink", type=int, default=0, help="Sink size for compressed caches")
    parser.add_argument("--local_window", type=int, default=0, help="Local window size for compressed caches")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for hierasparse cache")
    parser.add_argument("--key_prune_ratio", type=float, default=0.5, help="Pruning ratio for hierasparse cache")
    parser.add_argument("--value_prune_ratio", type=float, default=0.5, help="Pruning ratio for hierasparse cache")

    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=10, help="Measurement iterations")
    parser.add_argument("--export_trace", action="store_true", help="Export chrome trace to trace.json")
    parser.add_argument(
        "--memory_snapshot", action="store_true", help="Enable memory snapshot (dump to memory_snapshot.pickle)"
    )
    return parser.parse_args()


def benchmark_layer(args):
    device = "cuda"
    dtype = torch.float16

    print(f"Benchmarking Prefilling with Profiler:")
    print(f"  Batch Size: {args.batch}")
    print(f"  Seq Len:    {args.seq_len}")
    print(f"  Hidden:     {args.hidden_size}")
    print(f"  Heads:      {args.heads}")
    print(f"  KV Heads:   {args.kv_heads}")
    print(f"  Cache Type: {args.cache_type}")

    cache_cls = DenseCache
    cache_kwargs = {}
    forward_kwargs = {}

    if args.cache_type == "best_perf":
        cache_cls = PrefillKVDecodeKVCache
        cache_kwargs = {
            "sink": args.sink,
            "local_window": args.local_window,
        }
    elif args.cache_type == "best_acc":
        cache_cls = PrefillVDecodeVCache
        cache_kwargs = {
            "sink": args.sink,
            "local_window": args.local_window,
        }
    elif args.cache_type == "hierasparse":
        cache_cls = HieraSparseCache
        cache_kwargs = {
            "sink": args.sink,
            "local_window": args.local_window,
            "block_size": args.block_size,
            "key_prune_ratio": args.key_prune_ratio,
            "value_prune_ratio": args.value_prune_ratio,
        }
        forward_kwargs["block_N"] = args.block_size

    if getattr(args, "memory_snapshot", False):
        print("Enabling memory history...")
        torch.cuda.memory._record_memory_history()

    config = LlamaConfig(
        hidden_size=args.hidden_size,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.seq_len * 2, 312000),
        rms_norm_eps=1e-5,
    )
    config._attn_implementation = cache_cls.ATTN_IMPLEMENTATION

    layer = LlamaDecoderLayer(config, layer_idx=0).to(device).to(dtype)
    layer.eval()

    layer.forward = record_func_with_name("Layer Total")(layer.forward)

    layer.self_attn.forward = record_func_with_name("Self Attention")(layer.self_attn.forward)
    layer.self_attn.q_proj.forward = record_func_with_name("Q Proj")(layer.self_attn.q_proj.forward)
    layer.self_attn.k_proj.forward = record_func_with_name("K Proj")(layer.self_attn.k_proj.forward)
    layer.self_attn.v_proj.forward = record_func_with_name("V Proj")(layer.self_attn.v_proj.forward)
    layer.self_attn.o_proj.forward = record_func_with_name("O Proj")(layer.self_attn.o_proj.forward)

    layer.mlp.gate_proj.forward = record_func_with_name("Gate Proj")(layer.mlp.gate_proj.forward)
    layer.mlp.up_proj.forward = record_func_with_name("Up Proj")(layer.mlp.up_proj.forward)
    layer.mlp.down_proj.forward = record_func_with_name("Down Proj")(layer.mlp.down_proj.forward)

    inputs = torch.randn(args.batch, args.seq_len, args.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(args.batch, -1)

    rotary = LlamaRotaryEmbedding(config=config).to(device)
    cos, sin = rotary(inputs, position_ids)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        try:
            for _ in range(args.warmup):
                cache = cache_cls(**cache_kwargs)
                layer(
                    inputs,
                    position_ids=position_ids,
                    past_key_value=cache,
                    position_embeddings=(cos, sin),
                    **forward_kwargs,
                )
        except torch.cuda.OutOfMemoryError:
            print("OOM detected during warmup!")
            if getattr(args, "memory_snapshot", False):
                print("Dumping memory snapshot...")
                try:
                    torch.cuda.memory._dump_snapshot("memory_snapshot_warmup.pickle")
                except Exception as e:
                    print(f"Failed to dump snapshot: {e}")
            raise

    torch.cuda.synchronize()

    # Profile
    print("Running benchmark with Profiler...")

    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()

    results = {}

    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as prof:
            with torch.no_grad():
                for _ in range(args.rep):
                    cache = cache_cls(**cache_kwargs)
                    layer(
                        inputs,
                        position_ids=position_ids,
                        past_key_value=cache,
                        position_embeddings=(cos, sin),
                        **forward_kwargs,
                    )
                    # Add cache memory usage
                    if hasattr(cache, "memory_usage_bytes"):
                        k_bytes, v_bytes = cache.memory_usage_bytes()
                        results["k_cache_memory_mb"] = k_bytes / (1024 * 1024)
                        results["v_cache_memory_mb"] = v_bytes / (1024 * 1024)
                        results["kv_cache_memory_mb"] = (k_bytes + v_bytes) / (1024 * 1024)
                    else:
                        # Fallback for standard dynamic cache (though our wrapped ones support it, HF unmodified might not)
                        # Estimate based on shapes if needed, or just 0
                        results["k_cache_memory_mb"] = 0.0
                        results["v_cache_memory_mb"] = 0.0
                        results["kv_cache_memory_mb"] = 0.0

    except torch.cuda.OutOfMemoryError:
        print("OOM detected! Dumping memory snapshot...")
        if getattr(args, "memory_snapshot", False):
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        raise
    finally:
        if getattr(args, "memory_snapshot", False):
            torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)

    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    results["peak_memory_mb"] = peak_mem / (1024 * 1024)
    results["memory_growth_mb"] = (end_mem - start_mem) / (1024 * 1024)

    # Process results
    events_avg = prof.key_averages()

    custom_labels = [
        "Layer Total",
        "LayerNorm Input",
        "Self Attention",
        "Q Proj",
        "K Proj",
        "V Proj",
        "O Proj",
        "LayerNorm Post",
        "MLP",
        "Gate Proj",
        "Up Proj",
        "Down Proj",
        *list(RECORDED_FUNC.keys()),
    ]

    for label in custom_labels:
        for e in events_avg:
            if e.key == label and e.device_type == DeviceType.CUDA:
                results[label] = e.cuda_time / 1e3
                break

    results["Attention Core"] = results["Self Attention"] - (
        results["Q Proj"] + results["K Proj"] + results["V Proj"] + results["O Proj"]
    )
    results["Linear Projections"] = (
        results["Q Proj"]
        + results["K Proj"]
        + results["V Proj"]
        + results["O Proj"]
        + results["Gate Proj"]
        + results["Up Proj"]
        + results["Down Proj"]
    )

    if args.export_trace:
        trace_path = f"prefill_{args.cache_type}_{args.seq_len}.json"
        prof.export_chrome_trace(trace_path)
        print(f"Trace exported to {trace_path}")

    return results


def main():
    args = get_args()
    results = benchmark_layer(args)

    print("\n" + "=" * 60)
    print(f"{'Component':<30} | {'Time (ms)':<10}")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:<30} | {v:<10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

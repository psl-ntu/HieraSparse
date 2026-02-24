import argparse
import functools

import torch
from torch.profiler import DeviceType, ProfilerActivity, profile, record_function

from hierasparse.caches.compressed_cache import (
    HieraSparseCache,
    PrefillKVDecodeKVCache,
    PrefillVDecodeVCache,
)
from hierasparse.caches.simulator_cache import DenseCacheNoUpdate
from hierasparse.models.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaRotaryEmbedding,
)


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
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length (prefill length)")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--kv_heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--intermediate_size", type=int, default=14336, help="MLP intermediate size")

    parser.add_argument("--cache_type", type=str, choices=["dense", "best_perf", "best_acc"], default="dense")
    parser.add_argument("--sink", type=int, default=0, help="Sink size for compressed caches")
    parser.add_argument("--local_window", type=int, default=0, help="Local window size for compressed caches")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for hierasparse cache")
    parser.add_argument("--key_prune_ratio", type=float, default=0, help="Pruning ratio for hierasparse cache")
    parser.add_argument("--value_prune_ratio", type=float, default=1.0, help="Pruning ratio for hierasparse cache")

    parser.add_argument("--rep", type=int, default=256)
    parser.add_argument("--export_trace", action="store_true", help="Export chrome trace")
    parser.add_argument(
        "--memory_snapshot", action="store_true", help="Enable memory snapshot (dump to memory_snapshot.pickle)"
    )
    return parser.parse_args()


def benchmark_decode(args):
    device = "cuda"
    dtype = torch.float16

    print(f"Benchmarking Decoding Performance:")
    print(f"  Batch Size: {args.batch}")
    print(f"  Seq Len:    {args.seq_len}")
    print(f"  Hidden:     {args.hidden_size}")
    print(f"  Heads:      {args.heads}")
    print(f"  KV Heads:   {args.kv_heads}")
    print(f"  Cache Type: {args.cache_type}")

    cache_cls = DenseCacheNoUpdate
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
        max_position_embeddings=args.seq_len,
        rms_norm_eps=1e-5,
    )
    config._attn_implementation = cache_cls.ATTN_IMPLEMENTATION

    layer = LlamaDecoderLayer(config, layer_idx=0).to(device).to(dtype)
    layer.eval()

    prefill_inputs = torch.randn(args.batch, args.seq_len, args.hidden_size, device=device, dtype=dtype)
    prefill_pos = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(args.batch, -1)

    rotary = LlamaRotaryEmbedding(config=config).to(device)
    cos_pre, sin_pre = rotary(prefill_inputs, prefill_pos)

    decode_inputs = torch.randn(args.batch, 1, args.hidden_size, device=device, dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    results = {}
    cache = cache_cls(**cache_kwargs)
    layer(
        prefill_inputs,
        position_ids=prefill_pos,
        past_key_value=cache,
        position_embeddings=(cos_pre, sin_pre),
        **forward_kwargs,
    )

    torch.cuda.synchronize()

    del prefill_inputs, cos_pre, sin_pre
    torch.cuda.empty_cache()

    print("Capturing CUDA Graph...")
    static_pos_ids = torch.tensor([[args.seq_len]], device=device).expand(args.batch, -1)
    static_cos_dec, static_sin_dec = rotary(decode_inputs, static_pos_ids)

    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            layer(
                decode_inputs,
                position_ids=static_pos_ids,
                past_key_value=cache,
                position_embeddings=(static_cos_dec, static_sin_dec),
                **forward_kwargs,
            )
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        layer(
            decode_inputs,
            position_ids=static_pos_ids,
            past_key_value=cache,
            position_embeddings=(static_cos_dec, static_sin_dec),
            **forward_kwargs,
        )

    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(args.rep):
            g.replay()

    events_avg = prof.key_averages()
    results["Attention Core"] = 0.0
    results["Linear Projections"] = 0.0
    results["Layer Total"] = 0.0
    for e in events_avg:
        if e.device_type != DeviceType.CUDA:
            continue
        name = e.key
        if (
            "flash" in name.lower()
            or "attention" in name.lower()
            or "decode" in name.lower()
            or "sigmoid" in name.lower()
        ):
            # print(f"attention: {e.cuda_time} {name}")
            results["Attention Core"] += e.cuda_time / 1e3
        elif "gemv" in name.lower() or "gemm" in name.lower() or "cutlass" in name.lower() or "cublas" in name.lower():
            # print(f"linear: {e.cuda_time} {name}")
            results["Linear Projections"] += e.cuda_time / 1e3
        else:
            # print(f"other: {e.cuda_time} {name}")
            pass
        results["Layer Total"] += e.cuda_time / 1e3

    if args.export_trace:
        trace_path = f"decode_{args.cache_type}_{args.seq_len}.json"
        prof.export_chrome_trace(trace_path)
        print(f"Trace exported to {trace_path}")

    end_mem = torch.cuda.memory_allocated()
    peak_mem = torch.cuda.max_memory_allocated()

    results["peak_memory_mb"] = peak_mem / (1024 * 1024)
    results["memory_growth_mb"] = (end_mem - start_mem) / (1024 * 1024)

    return results


def main():
    args = get_args()
    results = benchmark_decode(args)

    print("\n" + "=" * 60)
    print(f"Results for Cache: {args.cache_type}, Prefill: {args.seq_len}")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:<30} | {v:<10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

import argparse
from functools import partial

import torch

from hierasparse.compress_method import (
    tilelang_block_compress_key,
    tilelang_block_compress_value,
    torch_block_compress_key,
    torch_block_compress_value,
)
from hierasparse.prune_method import torch_prune_block_key_mask

PRUNE_RATIO = 0.5
NUM_SINK_BLOCKS = 0
NUM_LOCAL_BLOCKS = 0


def test_accuracy_key():
    print("\n" + "=" * 100)
    print("KEY ACCURACY TEST")
    print("=" * 100)

    if NUM_SINK_BLOCKS > 0 or NUM_LOCAL_BLOCKS > 0:
        print(f"\nSkipping accuracy test (sink_blocks={NUM_SINK_BLOCKS}, local_blocks={NUM_LOCAL_BLOCKS})")
        print("Accuracy test requires default configuration to compare implementations.")
        return True

    batch, heads, seq, dim = 8, 8, 32768, 128
    block_s = 64

    print(f"\nConfig: batch={batch}, heads={heads}, seq={seq}, dim={dim}, block_s={block_s}, prune_ratio={PRUNE_RATIO}")

    K = torch.randn([batch, heads, seq, dim], dtype=torch.float16, device="cuda")

    print(f"\nTesting Torch Key Compression")
    idx_map_v1, dense_v1, sparse_v1, meta_v1 = torch_block_compress_key(
        K, PRUNE_RATIO, block_s, NUM_SINK_BLOCKS, NUM_LOCAL_BLOCKS
    )

    print(f"Testing TileLang Key Compression")
    idx_map_v3, dense_v3, sparse_v3, meta_v3 = tilelang_block_compress_key(K, PRUNE_RATIO, block_s)

    print(f"\nTorch vs TileLang:")
    idx_match_ratio = (idx_map_v1 == idx_map_v3).float().mean().item()
    idx_match = idx_match_ratio > 0.9

    dense_match_ratio = torch.isclose(dense_v1, dense_v3, rtol=1e-3, atol=1e-3).float().mean().item()
    dense_match = dense_match_ratio > 0.9

    sparse_match_ratio = torch.isclose(sparse_v1, sparse_v3, rtol=1e-3, atol=1e-3).float().mean().item()
    sparse_match = sparse_match_ratio > 0.9

    meta_match_ratio = (meta_v1 == meta_v3).float().mean().item()
    meta_match = meta_match_ratio > 0.9

    print(f"  idx_map match:  {idx_match} (ratio: {idx_match_ratio:.6f})")
    print(f"  dense match:    {dense_match} (ratio: {dense_match_ratio:.6f})")
    print(f"  sparse match:   {sparse_match} (ratio: {sparse_match_ratio:.6f})")
    print(f"  meta match:     {meta_match} (ratio: {meta_match_ratio:.6f})")

    return idx_match and dense_match and sparse_match and meta_match


def test_accuracy_value():
    print("\n" + "=" * 100)
    print("VALUE ACCURACY TEST")
    print("=" * 100)

    if NUM_SINK_BLOCKS > 0 or NUM_LOCAL_BLOCKS > 0:
        print(f"\nSkipping accuracy test (sink_blocks={NUM_SINK_BLOCKS}, local_blocks={NUM_LOCAL_BLOCKS})")
        print("Accuracy test requires default configuration to compare implementations.")
        return True

    batch, heads, seq, dim = 8, 8, 32768, 128
    block_s = 64

    print(f"\nConfig: batch={batch}, heads={heads}, seq={seq}, dim={dim}, block_s={block_s}, prune_ratio={PRUNE_RATIO}")

    V = torch.ones([batch, heads, seq, dim], dtype=torch.float16, device="cuda")

    print(f"\nTesting Torch Value Compression")
    idx_map_v1, dense_v1, sparse_v1, meta_v1 = torch_block_compress_value(
        V, PRUNE_RATIO, block_s, NUM_SINK_BLOCKS, NUM_LOCAL_BLOCKS
    )

    print(f"Testing TileLang Value Compression")
    idx_map_v3, dense_v3, sparse_v3, meta_v3 = tilelang_block_compress_value(V, PRUNE_RATIO, block_s)

    print(f"\nTorch vs TileLang:")
    idx_match_ratio = (idx_map_v1 == idx_map_v3).float().mean().item()
    idx_match = idx_match_ratio > 0.9

    dense_match_ratio = torch.isclose(dense_v1, dense_v3, rtol=1e-3, atol=1e-3).float().mean().item()
    dense_match = dense_match_ratio > 0.9

    sparse_match_ratio = torch.isclose(sparse_v1, sparse_v3, rtol=1e-3, atol=1e-3).float().mean().item()
    sparse_match = sparse_match_ratio > 0.9

    meta_match_ratio = (meta_v1 == meta_v3).float().mean().item()
    meta_match = meta_match_ratio > 0.9

    print(f"  idx_map match:  {idx_match} (ratio: {idx_match_ratio:.6f})")
    print(f"  dense match:    {dense_match} (ratio: {dense_match_ratio:.6f})")
    print(f"  sparse match:   {sparse_match} (ratio: {sparse_match_ratio:.6f})")
    print(f"  meta match:     {meta_match} (ratio: {meta_match_ratio:.6f})")
    return idx_match and dense_match and sparse_match and meta_match


def elementwise_baseline(X: torch.Tensor) -> torch.Tensor:
    torch_prune_block_key_mask(X, 64, prune_ratio=PRUNE_RATIO)
    return X * 2.0


def benchmark_with_cuda_events(func, K, prune_ratio=None, block_s=None, iters=50):
    for _ in range(10):
        if prune_ratio is None:
            _ = func(K)
        else:
            _ = func(K, prune_ratio, block_s)

    torch.cuda.synchronize()

    start_events = []
    end_events = []

    for _ in range(iters):
        start_events.append(torch.cuda.Event(enable_timing=True))
        end_events.append(torch.cuda.Event(enable_timing=True))

    for i in range(iters):
        start_events[i].record()
        if prune_ratio is None:
            _ = func(K)
        else:
            _ = func(K, prune_ratio, block_s)
        end_events[i].record()

    torch.cuda.synchronize()

    total_time = sum(start.elapsed_time(end) for start, end in zip(start_events, end_events))
    avg_time = total_time / iters

    return avg_time  # in milliseconds


def benchmark_large_configs(check_key=True, check_value=True):
    print("\n" + "=" * 100)
    print("SPEED BENCHMARK (Large Configs with CUDA Events)")
    print("=" * 100)

    batch_size = 8
    heads = 8
    block_s = 64
    dim = 128

    seq_lengths = [8192, 16384, 32768, 65536]

    print(f"\nConfiguration:")
    print(f"  Batch size:        {batch_size}")
    print(f"  Heads:             {heads}")
    print(f"  Dim:               {dim}")
    print(f"  Block size:        {block_s}")
    print(f"  Prune ratio:       {PRUNE_RATIO}")

    results_key = []
    results_value = []

    def run_benchmark(is_key: bool):
        label = "Key" if is_key else "Value"
        print(f"\nBenchmarking {label} Compression...")

        print(
            f"\n{'Seq Len':<12} {'Input (GB)':<15} {'Baseline (ms)':<15} {'Torch (ms)':<12} {'TileLang (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 110)

        results = []
        for seq_len in seq_lengths:
            K = torch.randn([batch_size, heads, seq_len, dim], dtype=torch.float16, device="cuda")

            input_bytes = batch_size * heads * seq_len * dim * 2  # float16 = 2 bytes
            input_gb = input_bytes / (1024**3)

            time_baseline = benchmark_with_cuda_events(elementwise_baseline, K, None, None, iters=20)

            if is_key:
                func_torch = partial(
                    torch_block_compress_key, num_sink_blocks=NUM_SINK_BLOCKS, num_local_blocks=NUM_LOCAL_BLOCKS
                )
                func_tl = tilelang_block_compress_key
            else:
                func_torch = partial(
                    torch_block_compress_value, num_sink_blocks=NUM_SINK_BLOCKS, num_local_blocks=NUM_LOCAL_BLOCKS
                )
                func_tl = tilelang_block_compress_value

            time_torch = benchmark_with_cuda_events(func_torch, K, PRUNE_RATIO, block_s, iters=20)

            time_tl = benchmark_with_cuda_events(func_tl, K, PRUNE_RATIO, block_s, iters=20)

            speedup = time_torch / time_tl

            results.append((seq_len, input_gb, time_baseline, time_torch, time_tl, speedup))

            print(
                f"{seq_len:<12} {input_gb:<15.4f} {time_baseline:<15.3f} {time_torch:<12.3f} {time_tl:<12.3f} {speedup:<10.2f}x"
            )
        return results

    if check_key:
        results_key = run_benchmark(is_key=True)
    if check_value:
        results_value = run_benchmark(is_key=False)

    return results_key, results_value


def main():
    parser = argparse.ArgumentParser(description="Block Compression Benchmark")
    parser.add_argument("--profile", action="store_true", help="Run profiler and export trace")
    args = parser.parse_args()

    print("\n" + "=" * 100)
    print("BLOCK COMPRESSION BENCHMARK SUITE")
    print("=" * 100)
    print("\nTesting Torch vs TileLang implementations")

    accuracy_pass_key = test_accuracy_key()
    accuracy_pass_value = test_accuracy_value()

    if not (accuracy_pass_key and accuracy_pass_value):
        print("\n⚠️  Accuracy test FAILED! Skipping speed benchmarks.")
        return

    results_key, results_value = benchmark_large_configs()

    print("\n" + "=" * 100)
    print("✓ BENCHMARK COMPLETE")
    print("=" * 100)

    if results_key:
        avg_speedup_key = sum(r[5] for r in results_key) / len(results_key)
        print(f"\nKey Compression Results:")
        print(f"  Average speedup (TileLang vs Torch): {avg_speedup_key:.2f}x")

    if results_value:
        avg_speedup_value = sum(r[5] for r in results_value) / len(results_value)
        print(f"\nValue Compression Results:")
        print(f"  Average speedup (TileLang vs Torch): {avg_speedup_value:.2f}x")

    if args.profile:
        run_profile()


def run_profile():
    print("\n" + "=" * 100)
    print("PROFILING")
    print("=" * 100)

    B, H, S, D = 8, 8, 4096, 128
    block_s = 32
    K = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    print("Warming up...")
    for _ in range(5):
        torch_block_compress_key(K, PRUNE_RATIO, block_s)
        tilelang_block_compress_key(K, PRUNE_RATIO, block_s)
        torch_block_compress_value(K, PRUNE_RATIO, block_s)
        tilelang_block_compress_value(K, PRUNE_RATIO, block_s)

    print("Profiling all versions...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:

        print(f"Profiling Key Compression...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"key_torch_iter_{i}")
            torch_block_compress_key(K, PRUNE_RATIO, block_s)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"key_tl_iter_{i}")
            tilelang_block_compress_key(K, PRUNE_RATIO, block_s)
            torch.cuda.nvtx.range_pop()

        print(f"Profiling Value Compression...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"value_torch_iter_{i}")
            torch_block_compress_value(K, PRUNE_RATIO, block_s)
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push(f"value_tl_iter_{i}")
            tilelang_block_compress_value(K, PRUNE_RATIO, block_s)
            torch.cuda.nvtx.range_pop()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    prof.export_chrome_trace("profile_block_compress.json")
    print("Exported profile_block_compress.json")


if __name__ == "__main__":
    main()

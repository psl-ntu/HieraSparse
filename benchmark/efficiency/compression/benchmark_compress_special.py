import argparse

import torch
import triton

from hierasparse.compress_method import (
    tilelang_prune_and_compress_key,
    tilelang_prune_and_compress_value,
    torch_prune_and_compress_key,
    torch_prune_and_compress_value,
)


def check_correctness(B, H, S, D):
    print(f"Checking correctness with B={B}, H={H}, S={S}, D={D}")
    K = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")
    V = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")

    print("Checking Key compression correctness...")
    k_sp_ref, k_e_ref = torch_prune_and_compress_key(K)
    k_sp_tl, k_e_tl = tilelang_prune_and_compress_key(K)

    try:
        torch.testing.assert_close(k_e_ref, k_e_tl, rtol=0, atol=0)
        torch.testing.assert_close(k_sp_ref, k_sp_tl, rtol=1e-3, atol=1e-3)
        print("Key Sparse: PASS")
    except AssertionError as e:
        print(f"Key Sparse: FAIL - {e}")

    print("Checking Value compression correctness...")
    v_sp_ref, v_e_ref = torch_prune_and_compress_value(V)
    v_sp_tl, v_e_tl = tilelang_prune_and_compress_value(V)

    try:
        torch.testing.assert_close(v_e_ref, v_e_tl, rtol=0, atol=0)
        torch.testing.assert_close(v_sp_ref, v_sp_tl, rtol=1e-3, atol=1e-3)
        print("Value Sparse: PASS")
    except AssertionError as e:
        print(f"Value Sparse: FAIL - {e}")


def benchmark_speed(B, H, S, D):
    print(f"\nBenchmarking with B={B}, H={H}, S={S}, D={D}")
    K = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")
    V = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")

    print("\nBenchmarking...")

    ms = triton.testing.do_bench(lambda: torch_prune_and_compress_key(K))
    print(f"Torch Key: {ms:.3f} ms")

    ms = triton.testing.do_bench(lambda: tilelang_prune_and_compress_key(K))
    print(f"TileLang Key: {ms:.3f} ms")

    ms = triton.testing.do_bench(lambda: torch_prune_and_compress_value(V))
    print(f"Torch Value: {ms:.3f} ms")

    ms = triton.testing.do_bench(lambda: tilelang_prune_and_compress_value(V))
    print(f"TileLang Value: {ms:.3f} ms")


def run_profile(B, H, S, D):
    import torch.profiler

    print("\n" + "=" * 100)
    print("PROFILING")
    print("=" * 100)

    K = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")
    V = torch.randn((B, H, S, D), dtype=torch.float16, device="cuda")

    # Warmup
    for _ in range(5):
        torch_prune_and_compress_key(K)
        tilelang_prune_and_compress_key(K)
        torch_prune_and_compress_value(V)
        tilelang_prune_and_compress_value(V)

    print("Running profiler...")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        print("Profiling Torch Key...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"torch_key_iter_{i}")
            torch_prune_and_compress_key(K)
            torch.cuda.nvtx.range_pop()

        print("Profiling TileLang Key...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"tilelang_key_iter_{i}")
            tilelang_prune_and_compress_key(K)
            torch.cuda.nvtx.range_pop()

        print("Profiling Torch Value...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"torch_value_iter_{i}")
            torch_prune_and_compress_value(V)
            torch.cuda.nvtx.range_pop()

        print("Profiling TileLang Value...")
        for i in range(10):
            torch.cuda.nvtx.range_push(f"tilelang_value_iter_{i}")
            tilelang_prune_and_compress_value(V)
            torch.cuda.nvtx.range_pop()

    prof.export_chrome_trace("profile_compress.json")
    print("Profile saved to profile_compress.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true", help="Run profiling")
    args = parser.parse_args()

    B, H, S, D = 8, 8, 32768, 128

    check_correctness(B, H, S, D)
    benchmark_speed(B, H, S, D)

    if args.profile:
        run_profile(B, H, S, D)

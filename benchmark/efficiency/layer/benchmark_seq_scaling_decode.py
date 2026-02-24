import argparse
import copy
import json
import os

import torch
from benchmark_layer_breakdown_decode import benchmark_decode


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=[i * 8192 for i in range(1, 33)],
        help="Sequence lengths (prefill length) to benchmark decoding on",
    )
    parser.add_argument(
        "--cache_types",
        type=str,
        nargs="+",
        default=["dense", "best_perf", "best_acc"],
        help="Cache types to benchmark",
    )
    parser.add_argument(
        "--output_file", type=str, default="output/decode_seq_scaling_results.json", help="Output JSON file"
    )

    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--kv_heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--intermediate_size", type=int, default=14336, help="MLP intermediate size")

    parser.add_argument("--sink", type=int, default=0)
    parser.add_argument("--local_window", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--key_prune_ratio", type=float, default=0.5)
    parser.add_argument("--value_prune_ratio", type=float, default=0.5, help="Pruning ratio for hierasparse cache")

    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--export_trace", action="store_true", help="Export chrome trace (disabled during scaling)")

    args = parser.parse_args()

    results_data = []

    for cache_type in args.cache_types:
        for seq_len in args.seq_lens:
            print(f"\nRunning Decode Scaling: Cache={cache_type}, SeqLen={seq_len}", flush=True)

            current_args = copy.deepcopy(args)
            current_args.cache_type = cache_type
            current_args.seq_len = seq_len
            current_args.export_trace = False

            try:
                metrics = benchmark_decode(current_args)

                entry = {"cache_type": cache_type, "seq_len": seq_len, "metrics": metrics}
                results_data.append(entry)

            except Exception as e:
                print(f"Failed to run {cache_type} with seq_len {seq_len}: {e}")
                # Optionally print stack trace for debugging
                import traceback

                traceback.print_exc()
                torch.cuda.empty_cache()

    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(results_data, f, indent=4)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    print("Starting benchmark_seq_scaling_decode.py...", flush=True)
    main()

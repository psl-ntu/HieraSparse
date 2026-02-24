#!/bin/bash

SEQLENS=(65536 98304 131072 163840 196608 229376 262144)

python benchmark/efficiency/layer/benchmark_seq_scaling_prefill.py --seq_lens "${SEQLENS[@]}"
python benchmark/efficiency/layer/visualize.py --input_file output/prefill_seq_scaling_results.json

python benchmark/efficiency/layer/benchmark_seq_scaling_decode.py --seq_lens "${SEQLENS[@]}"
python benchmark/efficiency/layer/visualize.py --input_file output/decode_seq_scaling_results.json --is_decode

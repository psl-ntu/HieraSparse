#!/bin/bash

export FREQ=1410 # from nvidia-smi -q -d SUPPORTED_CLOCKS
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 -lgc $FREQ,$FREQ

export TQDM_DISABLE=1

for k in 32 64 96 128 160 192 224 256; do
    seq_len=$((k * 1024))
    echo "======= Running with prompt_length=${k}k ======="
    python benchmark/efficiency/e2e/e2e_bench.py \
        --cache_type k_sp_v_sp \
        --batch 1 \
        --prompt_length ${seq_len} \
        --chunk_size 32768 \
        --run_baseline
done

for k in 32 64 96 128 160 192 224 256; do
    seq_len=$((k * 1024))
    echo "======= Running with prompt_length=${k}k ======="
    python benchmark/efficiency/e2e/e2e_bench.py \
        --cache_type k_dense_v_sp \
        --batch 1 \
        --prompt_length ${seq_len} \
        --chunk_size 32768 \
        --run_baseline
done

sudo nvidia-smi -i 0 -rgc

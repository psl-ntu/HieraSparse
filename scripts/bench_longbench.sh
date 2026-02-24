#!/bin/bash

cd benchmark/quality

TIMESTAMP=$(date +%Y%m%d_%H%M)
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
FAST_EVAL=1

python pred.py \
    --model_name $MODEL_NAME \
    --cache dense \
    --sink 0 \
    --local_window 0 \
    --output_postfix dense_$TIMESTAMP \
    --fast_eval $FAST_EVAL

python pred.py \
    --model_name $MODEL_NAME \
    --cache hierasparse_decode \
    --block_seq_size 64 \
    --prune_key_prefill_ratio 1.0 \
    --prune_value_prefill_ratio 1.0 \
    --sink 64 \
    --local_window 256 \
    --output_postfix hierasparse_decode_k1.0_v1.0_$TIMESTAMP \
    --fast_eval $FAST_EVAL

python pred.py \
    --model_name $MODEL_NAME \
    --cache hierasparse_decode \
    --block_seq_size 64 \
    --prune_key_prefill_ratio 0.0 \
    --prune_value_prefill_ratio 1.0 \
    --sink 64 \
    --local_window 256 \
    --output_postfix hierasparse_decode_k0.0_v1.0_$TIMESTAMP \
    --fast_eval $FAST_EVAL

python pred.py \
    --model_name $MODEL_NAME \
    --cache hierasparse_decode \
    --block_seq_size 64 \
    --prune_key_prefill_ratio 1.0 \
    --prune_value_prefill_ratio 0.0 \
    --sink 64 \
    --local_window 256 \
    --output_postfix hierasparse_decode_k1.0_v0.0_$TIMESTAMP \
    --fast_eval $FAST_EVAL


python pred.py \
    --model_name $MODEL_NAME \
    --cache balanced \
    --sink 64 \
    --local_window 256 \
    --output_postfix balanced_$TIMESTAMP \
    --fast_eval $FAST_EVAL

python pred.py \
  --model_name $MODEL_NAME \
  --cache hierasparse \
  --prune_key_prefill_ratio 0.0 \
  --prune_value_prefill_ratio 1.0 \
  --block_seq_size 64 \
  --sink 64 \
  --local_window 256 \
  --output_postfix hierasparse_k0.0_v1.0_$TIMESTAMP \
  --fast_eval $FAST_EVAL

if [ -d "pred" ]; then
    mv pred ../../output/pred_$TIMESTAMP
fi

if [ -d "pred_fast" ]; then
    mv pred_fast ../../output/pred_fast_$TIMESTAMP
fi

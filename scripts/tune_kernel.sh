#!/bin/bash

export TILELANG_AUTO_TUNING_CPU_COUNTS=16

batch=1

echo "====== flashattn_sp_kv ======"
python hierasparse/kernels/flashattn_sp/flashattn_sp_kv.py --is_causal --check --tune --batch $batch

echo "====== flashdecode_sp_kv ======"
python hierasparse/kernels/flashdecode_sp/flashdecode_sp_kv.py --check --tune --batch $batch

echo "====== flashattn_sp_v ======"
python hierasparse/kernels/flashattn_sp/flashattn_sp_v.py --is_causal --check --tune --batch $batch

echo "====== flashdecode_sp_v ======"
python hierasparse/kernels/flashdecode_sp/flashdecode_sp_v.py --check --tune --batch $batch

echo "====== blockattn_sp_mk_sv ======"
python hierasparse/kernels/blockattn_sp/blockattn_sp_mk_sv.py --is_causal --block_N 64 --check --key_prune_ratio 1.0 --tune --batch $batch

echo "====== blockdecode_sp_mk_mv ======"
python hierasparse/kernels/blockdecode_sp/blockdecode_sp_mk_mv.py --check --block_N 64 --key_prune_ratio 1.0 --value_prune_ratio 1.0 --tune --batch $batch

echo "====== blockattn_sp_mk_mv ======"
python hierasparse/kernels/blockattn_sp/blockattn_sp_mk_mv.py --is_causal --block_N 64 --check --key_prune_ratio 1.0 --value_prune_ratio 1.0 --tune --batch $batch

echo "====== blockattn_sp_dk_mv ======"
python hierasparse/kernels/blockattn_sp/blockattn_sp_dk_mv.py --is_causal --block_N 64 --check --value_prune_ratio 1.0 --tune --batch $batch

echo "====== flashattn_tridao ======"
python hierasparse/kernels/flashattn_tridao.py --is_causal --check --tune --batch $batch

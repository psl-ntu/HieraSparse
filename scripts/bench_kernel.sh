#!/bin/bash

for x in 0.0 0.2 0.4 0.6 0.8 1.0; do
    echo "====== blockattn_sp_mk_mv ${x} ======"
    python hierasparse/kernels/blockattn_sp/blockattn_sp_mk_mv.py --is_causal --block_N 64 --check --key_prune_ratio "$x" --value_prune_ratio "$x"
    echo "====== blockdecode_sp_mk_mv ${x} ======"
    python hierasparse/kernels/blockdecode_sp/blockdecode_sp_mk_mv.py --check --block_N 64 --key_prune_ratio "$x" --value_prune_ratio "$x"
done

echo "====== flashattn_sp_kv ======"
python hierasparse/kernels/flashattn_sp/flashattn_sp_kv.py --is_causal --check
echo "====== flashdecode_sp_kv ======"
python hierasparse/kernels/flashdecode_sp/flashdecode_sp_kv.py --check


echo "====== dense key ======"
for v in 0.0 0.2 0.4 0.6 0.8 1.0; do
    echo "====== blockattn_sp_dk_mv ${v} ======"
    python hierasparse/kernels/blockattn_sp/blockattn_sp_dk_mv.py --is_causal --block_N 64 --check --value_prune_ratio "$v"
    python hierasparse/kernels/blockdecode_sp/blockdecode_sp_mk_mv.py --check --block_N 64 --key_prune_ratio 0 --value_prune_ratio "$v"
done
echo "====== flashattn_sp_v ======"
python hierasparse/kernels/flashattn_sp/flashattn_sp_v.py --is_causal --check
echo "====== flashdecode_sp_v ======"
python hierasparse/kernels/flashdecode_sp/flashdecode_sp_v.py --check


echo "====== sparse value mixed key ======"
for k in 0.0 0.2 0.4 0.6 0.8 1.0; do
    echo "====== blockattn_sp_mk_sv ${k} ======"
    python hierasparse/kernels/blockattn_sp/blockattn_sp_mk_sv.py --is_causal --block_N 64 --check --key_prune_ratio "$k"
    echo "====== blockdecode_sp_mk_mv ${k} ======"
    python hierasparse/kernels/blockdecode_sp/blockdecode_sp_mk_mv.py --check --block_N 64 --key_prune_ratio "$k" --value_prune_ratio 1
done

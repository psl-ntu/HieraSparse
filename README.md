# HieraSparse

**HieraSparse** is a sparse KV cache system for LLM inference that reduces memory and computation cost while preserving generation quality. It combines a hierarchical block-based memory layout with N:M structured sparse attention kernels and near-zero-overhead online compression.

---

## Requirements

- NVIDIA GPU (pre-tuned kernels: **L40S** only; tuning scripts provided for other GPUs in the same architecture generation)
- CUDA 12.8 or above

---

## Installation

```bash
bash scripts/install_hierasparse.sh
```

This creates a `hierasparse` conda environment with Python 3.10, PyTorch 2.10, flash-attn, and TileLang.

---

## Quick Start

```bash
conda activate hierasparse

python example/generation.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --cache hierasparse \
  --block_seq_size 64 \
  --prune_key_prefill_ratio 0.5 \
  --prune_value_prefill_ratio 0.5
```

---

## Benchmarks

Run each task after installation:

```bash
# Quality evaluation on LongBench (~300 min full, ~30 min fast subset)
bash scripts/bench_longbench.sh

# Compression kernel latency
bash scripts/bench_compression.sh

# Attention kernel latency (prefill + decode)
bash scripts/bench_kernel.sh

# Optimization ablation
bash scripts/bench_optimization.sh

# Baseline comparison (requires installing MUSTAFAR separately)
bash scripts/bench_mustafar.sh

# Layer-wise breakdown vs. sequence length
bash scripts/bench_layer.sh

# End-to-end generation latency and memory usage
bash scripts/bench_e2e.sh
```

---

## Code Structure

```
hierasparse/
  caches/         # KV cache implementations (dense, compressed, hierarchical)
  kernels/        # Sparse prefill/decode attention and compression kernels
  models/         # Patched model classes (Llama, Mistral, Qwen3)
  interface.py    # HuggingFace attention interface wiring
  operators.py    # Kernel dispatch logic
  prune_method.py # Pruning/sparsification methods
  compress_method.py
archived_kernels/ # Pre-compiled kernel sources for L40S
scripts/          # Installation and benchmark scripts
benchmark/        # Benchmark scripts (quality + efficiency)
example/          # generation.py end-to-end example
```

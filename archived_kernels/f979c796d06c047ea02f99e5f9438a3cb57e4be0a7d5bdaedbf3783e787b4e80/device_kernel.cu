#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const signed char* __restrict__ block_mask, short* __restrict__ idx_map, int blocks_per_head);
extern "C" __global__ void __launch_bounds__(128, 1) kernel_kernel(const signed char* __restrict__ block_mask, short* __restrict__ idx_map, int blocks_per_head) {
  short dense_count[1];
  short sparse_count[1];
  dense_count[0] = (short)0;
  sparse_count[0] = (short)0;
  for (int k = 0; k < blocks_per_head; ++k) {
    bool mask_val = ((bool)block_mask[((((((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head))) + ((int64_t)k))]);
    if (mask_val) {
      dense_count[0] = ((short)(((int)dense_count[0]) + 1));
      idx_map[((((((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head))) + ((int64_t)k))] = dense_count[0];
    } else {
      sparse_count[0] = ((short)(((int)sparse_count[0]) + 1));
      idx_map[((((((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head))) + ((int64_t)k))] = (sparse_count[0] * (short)-1);
    }
  }
}

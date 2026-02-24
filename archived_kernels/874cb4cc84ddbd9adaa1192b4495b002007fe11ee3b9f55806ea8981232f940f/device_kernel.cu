#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void prune_block_key_mask_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ prune_loss, int num_blocks);
extern "C" __global__ void __launch_bounds__(256, 1) prune_block_key_mask_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ prune_loss, int num_blocks) {
  float loss_local = 0x0p+0f/*0.000000e+00*/;
  extern __shared__ __align__(1024) float loss_shared[];
  half_t dense_local[32];
  half_t min1_val = half_t(0x0p+0f/*0.000000e+00*/);
  half_t min2_val = half_t(0x0p+0f/*0.000000e+00*/);
  loss_local = 0x0p+0f/*0.000000e+00*/;
  if ((((int)threadIdx.y) == 0) && (((int)threadIdx.x) == 0)) {
    loss_shared[0] = 0x0p+0f/*0.000000e+00*/;
  }
  __syncthreads();
  for (int i = 0; i < 4; ++i) {
    *(uint4*)(dense_local + (i * 8)) = *(uint4*)(K + (((((((((int64_t)((int)blockIdx.x)) * ((int64_t)num_blocks)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)num_blocks)) * (int64_t)8192)) + (((int64_t)((int)blockIdx.z)) * (int64_t)8192)) + (((int64_t)((int)threadIdx.x)) * (int64_t)128)) + (((int64_t)((int)threadIdx.y)) * (int64_t)32)) + (((int64_t)i) * (int64_t)8)));
  }
  for (int g_i = 0; g_i < 8; ++g_i) {
    min1_val = __habs(dense_local[(g_i * 4)]);
    min2_val = __habs(dense_local[((g_i * 4) + 1)]);
    if (min2_val < min1_val) {
      half_t tmp_val = min1_val;
      min1_val = min2_val;
      min2_val = tmp_val;
    }
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      half_t val = __habs(dense_local[(((g_i * 4) + i_1) + 2)]);
      if (val < min1_val) {
        min2_val = min1_val;
        min1_val = val;
      } else {
        if (val < min2_val) {
          min2_val = val;
        }
      }
    }
    loss_local = (loss_local + ((float)(min1_val + min2_val)));
  }
  AtomicAdd((&(loss_shared[0])), loss_local);
  __syncthreads();
  if ((((int)threadIdx.x) == 0) && (((int)threadIdx.y) == 0)) {
    prune_loss[((((((int64_t)((int)blockIdx.x)) * ((int64_t)num_blocks)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)num_blocks))) + ((int64_t)((int)blockIdx.z)))] = ((half_t)loss_shared[0]);
  }
}

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void block_compress_dense_kernel_kernel(const half_t* __restrict__ K, const signed char* __restrict__ block_mask, half_t* __restrict__ dense, short* __restrict__ idx_map, int blocks_per_head, int dense_blocks_per_head, int s);
extern "C" __global__ void __launch_bounds__(128, 1) block_compress_dense_kernel_kernel(const half_t* __restrict__ K, const signed char* __restrict__ block_mask, half_t* __restrict__ dense, short* __restrict__ idx_map, int blocks_per_head, int dense_blocks_per_head, int s) {
  short dense_count = (short)0;
  extern __shared__ __align__(1024) half_t block_shared[];
  dense_count = (short)0;
  for (int k = 0; k < blocks_per_head; ++k) {
    bool mask_val = ((bool)block_mask[((((((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head))) + ((int64_t)k))]);
    __syncthreads();
    if (mask_val) {
      #pragma unroll
      for (int i = 0; i < 8; ++i) {
        uint4 condval;
        if (((((k * 64) + (i * 8)) + (((int)threadIdx.x) >> 4)) < s)) {
          condval = *(uint4*)(K + (((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)s)) * (int64_t)1024)) + (((int64_t)i) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)s)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)));
        } else {
          condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
        }
        *(uint4*)(block_shared + ((i * 1024) + (((int)threadIdx.x) * 8))) = condval;
      }
      short dense_idx = dense_count;
      dense_count = ((short)(((int)dense_count) + 1));
      idx_map[((((((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head))) + ((int64_t)k))] = dense_count;
      __syncthreads();
      #pragma unroll
      for (int i_1 = 0; i_1 < 8; ++i_1) {
        if ((short)0 <= dense_idx) {
          if (((int)dense_idx) < dense_blocks_per_head) {
            *(uint4*)(dense + ((((((((int64_t)((int)blockIdx.x)) * ((int64_t)dense_blocks_per_head)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)dense_blocks_per_head)) * (int64_t)8192)) + (((int64_t)dense_idx) * (int64_t)8192)) + (((int64_t)i_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(block_shared + ((i_1 * 1024) + (((int)threadIdx.x) * 8)));
          }
        }
      }
    }
  }
}

#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void block_compress_sparse_value_kernel_kernel(const half_t* __restrict__ K, const signed char* __restrict__ block_mask, short* __restrict__ idx_map, short* __restrict__ sparse_meta, half_t* __restrict__ sparse_nonzero, int blocks_per_head, int s, int sparse_blocks_per_head);
extern "C" __global__ void __launch_bounds__(256, 1) block_compress_sparse_value_kernel_kernel(const half_t* __restrict__ K, const signed char* __restrict__ block_mask, short* __restrict__ idx_map, short* __restrict__ sparse_meta, half_t* __restrict__ sparse_nonzero, int blocks_per_head, int s, int sparse_blocks_per_head) {
  short sparse_count[1];
  half_t sparse_local[16];
  short meta_local[2];
  half_t dense_local[32];
  uchar non_zero_elt_log_idx[2];
  uchar max1_idx = (uchar)0;
  half_t max1_val = half_t(0x0p+0f/*0.000000e+00*/);
  uchar max2_idx = (uchar)0;
  half_t max2_val = half_t(0x0p+0f/*0.000000e+00*/);
  sparse_count[0] = (short)0;
  for (int k = 0; k < blocks_per_head; ++k) {
    bool mask_val = ((bool)block_mask[((((((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head))) + ((int64_t)k))]);
    if (!mask_val) {
      short sparse_idx = sparse_count[0];
      sparse_count[0] = ((short)(((int)sparse_count[0]) + 1));
      idx_map[((((((int64_t)((int)blockIdx.y)) * ((int64_t)blocks_per_head)) * (int64_t)8) + (((int64_t)((int)blockIdx.x)) * ((int64_t)blocks_per_head))) + ((int64_t)k))] = (sparse_count[0] * (short)-1);
      for (int i = 0; i < 2; ++i) {
        *(uint4*)(sparse_local + (i * 8)) = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
      }
      *(short2*)(meta_local + 0) = make_short2((short)0, (short)0);
      for (int i_1 = 0; i_1 < 32; ++i_1) {
        half_t condval;
        if (((((k * 64) + (((int)threadIdx.x) * 32)) + i_1) < s)) {
          condval = K[((((((((int64_t)k) * (int64_t)8192) + (((int64_t)((int)threadIdx.x)) * (int64_t)4096)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)s)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)s)) * (int64_t)128)) + (((int64_t)i_1) * (int64_t)128)) + ((int64_t)((int)threadIdx.y)))];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        dense_local[i_1] = condval;
      }
      for (int g_i = 0; g_i < 8; ++g_i) {
        *(uchar2*)(non_zero_elt_log_idx + 0) = make_uchar2((uchar)0, (uchar)0);
        max1_idx = (uchar)0;
        max1_val = __habs(dense_local[(g_i * 4)]);
        max2_idx = (uchar)1;
        max2_val = __habs(dense_local[((g_i * 4) + 1)]);
        if (max1_val < max2_val) {
          uchar tmp_idx = max1_idx;
          half_t tmp_val = max1_val;
          max1_idx = max2_idx;
          max1_val = max2_val;
          max2_idx = tmp_idx;
          max2_val = tmp_val;
        }
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          half_t val = __habs(dense_local[(((g_i * 4) + i_2) + 2)]);
          if (max1_val < val) {
            max2_idx = max1_idx;
            max2_val = max1_val;
            max1_idx = ((uchar)(i_2 + 2));
            max1_val = val;
          } else {
            if (max2_val < val) {
              max2_idx = ((uchar)(i_2 + 2));
              max2_val = val;
            }
          }
        }
        non_zero_elt_log_idx[0] = min(max1_idx, max2_idx);
        non_zero_elt_log_idx[1] = max(max1_idx, max2_idx);
        sparse_local[(g_i * 2)] = dense_local[((g_i * 4) + ((int)non_zero_elt_log_idx[0]))];
        sparse_local[((g_i * 2) + 1)] = dense_local[((g_i * 4) + ((int)non_zero_elt_log_idx[1]))];
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          uchar val_1 = non_zero_elt_log_idx[i_3];
          meta_local[(g_i >> 2)] = (meta_local[(g_i >> 2)] | (((short)val_1) << ((short)(((g_i & 3) * 4) + (i_3 * 2)))));
        }
      }
      for (int i_4 = 0; i_4 < 16; ++i_4) {
        if ((short)0 <= sparse_idx) {
          if (((int)sparse_idx) < sparse_blocks_per_head) {
            sparse_nonzero[(((((((((int64_t)((int)blockIdx.y)) * ((int64_t)sparse_blocks_per_head)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)sparse_blocks_per_head)) * (int64_t)4096)) + (((int64_t)sparse_idx) * (int64_t)4096)) + (((int64_t)((int)threadIdx.x)) * (int64_t)2048)) + (((int64_t)i_4) * (int64_t)128)) + ((int64_t)((int)threadIdx.y)))] = sparse_local[i_4];
          }
        }
      }
      for (int i_5 = 0; i_5 < 2; ++i_5) {
        if ((short)0 <= sparse_idx) {
          if (((int)sparse_idx) < sparse_blocks_per_head) {
            sparse_meta[(((((((((int64_t)((int)blockIdx.y)) * ((int64_t)sparse_blocks_per_head)) * (int64_t)4096) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)sparse_blocks_per_head)) * (int64_t)512)) + (((int64_t)sparse_idx) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)256)) + (((int64_t)i_5) * (int64_t)128)) + ((int64_t)((int)threadIdx.y)))] = meta_local[i_5];
          }
        }
      }
    }
  }
}

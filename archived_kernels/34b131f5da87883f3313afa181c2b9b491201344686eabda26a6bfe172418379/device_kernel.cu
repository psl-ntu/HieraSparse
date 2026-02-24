#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void prune_and_compress_key_kernel_kernel(const half_t* __restrict__ Dense, short* __restrict__ Meta, half_t* __restrict__ Sparse, int S);
extern "C" __global__ void __launch_bounds__(64, 1) prune_and_compress_key_kernel_kernel(const half_t* __restrict__ Dense, short* __restrict__ Meta, half_t* __restrict__ Sparse, int S) {
  half_t sparse_local[16];
  short meta_local[2];
  half_t dense_local[32];
  uchar non_zero_elt_log_idx[2];
  uchar max1_idx = (uchar)0;
  half_t max1_val = half_t(0x0p+0f/*0.000000e+00*/);
  uchar max2_idx = (uchar)0;
  half_t max2_val = half_t(0x0p+0f/*0.000000e+00*/);
  for (int i = 0; i < 2; ++i) {
    *(uint4*)(sparse_local + (i * 8)) = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
  }
  *(short2*)(meta_local + 0) = make_short2((short)0, (short)0);
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    uint4 condval;
    if ((((((int)blockIdx.x) * 16) + ((int)threadIdx.x)) < S)) {
      condval = *(uint4*)(Dense + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)2048) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)S)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)S)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)128)) + (((int64_t)((int)threadIdx.y)) * (int64_t)32)) + (((int64_t)i_1) * (int64_t)8)));
    } else {
      condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(dense_local + (i_1 * 8)) = condval;
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
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    *(uint4*)(Sparse + ((((((((int64_t)((int)blockIdx.x)) * (int64_t)1024) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)S)) * (int64_t)512)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)S)) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)64)) + (((int64_t)((int)threadIdx.y)) * (int64_t)16)) + (((int64_t)i_4) * (int64_t)8))) = *(uint4*)(sparse_local + (i_4 * 8));
  }
  *(short2*)(Meta + (((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)S)) * (int64_t)64)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)S)) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (((int64_t)((int)threadIdx.y)) * (int64_t)2))) = *(short2*)(meta_local + 0);
}

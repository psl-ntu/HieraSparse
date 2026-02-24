#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void flashattn_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, half_t* __restrict__ lse, int seq_kv, int seq_q);
extern "C" __global__ void __launch_bounds__(128, 1) flashattn_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, half_t* __restrict__ lse, int seq_kv, int seq_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[128];
  float logsum[4];
  float scores_max[4];
  float acc_s[64];
  float scores_max_prev[4];
  float scores_scale[4];
  float scores_sum[4];
  half_t acc_s_cast[64];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+(((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((((int)blockIdx.x) * 128) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_q));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)1024) + (((int64_t)i_1) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < seq_kv));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    *(float2*)(acc_o + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    logsum[i_3] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    scores_max[i_4] = -CUDART_INF_F;
  }
  for (int k = 0; k < min(((seq_kv + 63) >> 6), ((((int)blockIdx.x) * 2) + 2)); ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), V+(((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)1024)) + (((int64_t)i_5) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((k * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) < seq_kv));
    }
    tl::cp_async_commit();
    __syncthreads();
    tl::gemm_ss<128, 64, 128, 4, 1, 0, 1, 1, 128, 128, 0, 0>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[16384])), (&(acc_s[0])));
    tl::cp_async_wait<0>();
    __syncthreads();
    if (((k + 1) < ((seq_kv + 63) >> 6)) && (k <= (((int)blockIdx.x) * 2))) {
      #pragma unroll
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+((((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)1024)) + (((int64_t)i_6) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)8192), (((((k * 64) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) + 64) < seq_kv));
      }
      tl::cp_async_commit();
    }
    if ((((seq_kv + 63) >> 6) <= (k + 3)) || ((((int)blockIdx.x) * 2) <= (k + 1))) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 64; ++i_7) {
        float condval;
        if (((((((k * 64) + ((i_7 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) <= (((((((int)blockIdx.x) * 128) + (((i_7 & 7) >> 2) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + (((i_7 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))) & (((((k * 64) + ((i_7 >> 3) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_7 & 1)) < seq_kv))) {
          condval = acc_s[i_7];
        } else {
          condval = -CUDART_INF_F;
        }
        acc_s[i_7] = condval;
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      scores_max_prev[i_8] = scores_max[i_8];
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      scores_max[i_9] = -CUDART_INF_F;
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_10] = max(scores_max[i_10], acc_s[((((rv & 7) * 8) + (i_10 * 2)) + (rv >> 3))]);
      }
      scores_max[i_10] = tl::AllReduce<tl::MaxOp, 4, 1, 0>::run(scores_max[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      scores_max[i_11] = max(scores_max[i_11], scores_max_prev[i_11]);
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      scores_scale[i_12] = exp2f(((scores_max_prev[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_12] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 64; ++i_13) {
      acc_s[i_13] = exp2f(((acc_s[i_13] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[((i_13 & 7) >> 1)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      scores_sum[i_14] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_14] = (scores_sum[i_14] + acc_s[((((rv_1 & 7) * 8) + (i_14 * 2)) + (rv_1 >> 3))]);
      }
      scores_sum[i_14] = tl::AllReduce<tl::SumOp, 4, 1, 0>::run(scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      logsum[i_15] = ((logsum[i_15] * scores_scale[i_15]) + scores_sum[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 32; ++i_16) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s + (((((i_16 >> 3) * 16) + (((i_16 & 3) >> 1) * 8)) + (((i_16 & 7) >> 2) * 4)) + ((i_16 & 1) * 2)));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_cast + (i_16 * 2)) = __1;
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 128; ++i_17) {
      acc_o[i_17] = (acc_o[i_17] * scores_scale[((i_17 & 7) >> 1)]);
    }
    __syncthreads();
    tl::gemm_rs<128, 128, 64, 4, 1, 0, 0, 0, 64, 128, 0, 0>((&(acc_s_cast[0])), (&(((half_t*)buf_dyn_shmem)[24576])), (&(acc_o[0])));
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 128; ++i_18) {
    acc_o[i_18] = (acc_o[i_18] / logsum[((i_18 & 7) >> 1)]);
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 4; ++i_19) {
    logsum[i_19] = ((log2f(logsum[i_19]) + (scores_max[i_19] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i_20 = 0; i_20 < 64; ++i_20) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_o + (i_20 * 2));
    ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
    *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((((i_20 >> 5) * 8192) + (((i_20 & 3) >> 1) * 4096)) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_20 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((i_20 & 31) >> 4) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((i_20 & 15) >> 3) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((i_20 & 7) >> 2) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = __2;
  }
  __syncthreads();
  #pragma unroll
  for (int i_21 = 0; i_21 < 16; ++i_21) {
    if ((((((int)blockIdx.x) * 128) + (i_21 * 8)) + (((int)threadIdx.x) >> 4)) < seq_q) {
      *(uint4*)(Output + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i_21) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_21 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
  if ((((int)threadIdx.x) % 4) == 0) {
    #pragma unroll
    for (int i_22 = 0; i_22 < 4; ++i_22) {
      if ((((((((int)blockIdx.x) * 128) + ((i_22 >> 1) * 64)) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_22 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_q) {
        lse[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)i_22) >> (int64_t)1) * (int64_t)64)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_22) & (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) >> (int64_t)2)) + (((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)))] = ((half_t)logsum[i_22]);
      }
    }
  }
}

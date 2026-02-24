#include <tl_templates/cuda/instruction/mma.h>
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

extern "C" __global__ void flashattn_gqa_decode_no_split_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse_combined, int seq_kv);
extern "C" __global__ void __launch_bounds__(32, 1) flashattn_gqa_decode_no_split_kernel(const half_t* __restrict__ K, half_t* __restrict__ Output, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse_combined, int seq_kv) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o_T[32];
  float logsum[2];
  float scores_max[2];
  float acc_s_T[32];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t acc_s_T_[32];
  half_t acc_s_cast_T[32];
  half_t A_local[128];
  half_t B_local[8];
  half_t A_local_1[64];
  short E_local[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint4 condval;
    if ((((i >> 1) + ((int)blockIdx.y)) < 8)) {
      condval = *(uint4*)(Q + ((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 512)) + (i * 256)) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)), __pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
    }
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 512) + (i * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (i >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval;
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    *(float2*)(acc_o_T + (i_1 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  *(float2*)(logsum + 0) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  *(float2*)(scores_max + 0) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_2 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_2 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 2048), K+(((((((int64_t)((int)blockIdx.x)) * ((int64_t)seq_kv)) * (int64_t)1024) + (((int64_t)i_2) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_2 * 2) + (((int)threadIdx.x) >> 4)) < seq_kv));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 32; ++i_3) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 34816), V+(((((((int64_t)((int)blockIdx.x)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)1024) + (((int64_t)i_3) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_3 * 2) + (((int)threadIdx.x) >> 4)) < (seq_kv >> 1)));
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_4 * 512) + (((int)threadIdx.x) * 16)) + 51200), V_E+(((((((int64_t)((int)blockIdx.x)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)1024) + (((int64_t)i_4) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_4 * 2) + (((int)threadIdx.x) >> 4)) < (seq_kv >> 4)));
  }
  tl::cp_async_commit();
  for (int k = 0; k < (((seq_kv + 127) >> 7) - 1); ++k) {
    #pragma unroll
    for (int i_5 = 0; i_5 < 32; ++i_5) {
      float condval_1;
      if (((((k * 128) + ((i_5 & 15) * 8)) + (((int)threadIdx.x) >> 2)) < seq_kv)) {
        condval_1 = 0x0p+0f/*0.000000e+00*/;
      } else {
        condval_1 = -CUDART_INF_F;
      }
      acc_s_T[(((i_5 & 15) * 2) + (i_5 >> 4))] = condval_1;
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_6 = 0; i_6 < 8; ++i_6) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 8192) + (i_6 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])) + 0, A_local + (i_6 * 16));
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 8192) + (i_6 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])) + 0, A_local + ((i_6 * 16) + 8));
      }
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_7 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_7 * 16)), reinterpret_cast<const unsigned*>(B_local + 0));
        tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_7 * 4)), reinterpret_cast<const unsigned*>(A_local + ((i_7 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + 4));
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 64; ++i_8) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_8 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_8 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_8 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 2048), K+((((((((int64_t)k) * (int64_t)16384) + ((((int64_t)((int)blockIdx.x)) * ((int64_t)seq_kv)) * (int64_t)1024)) + (((int64_t)i_8) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)16384), (((((k * 128) + (i_8 * 2)) + (((int)threadIdx.x) >> 4)) + 128) < seq_kv));
    }
    tl::cp_async_commit();
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_9 = 0; i_9 < 2; ++i_9) {
      scores_max[i_9] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_9] = max(scores_max[i_9], acc_s_T[((rv * 2) + i_9)]);
      }
      scores_max[i_9] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_9]);
    }
    #pragma unroll
    for (int i_10 = 0; i_10 < 2; ++i_10) {
      scores_max[i_10] = max(scores_max[i_10], scores_max_prev[i_10]);
    }
    #pragma unroll
    for (int i_11 = 0; i_11 < 2; ++i_11) {
      scores_scale[i_11] = exp2f(((scores_max_prev[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_11] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 32; ++i_12) {
      acc_s_T[(((i_12 & 15) * 2) + (i_12 >> 4))] = exp2f(((acc_s_T[(((i_12 & 15) * 2) + (i_12 >> 4))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_12 >> 4)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 2; ++i_13) {
      scores_sum[i_13] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_13] = (scores_sum[i_13] + acc_s_T[((rv_1 * 2) + i_13)]);
      }
      scores_sum[i_13] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 2; ++i_14) {
      logsum[i_14] = ((logsum[i_14] * scores_scale[i_14]) + scores_sum[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 16; ++i_15) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_15 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_T_ + (i_15 * 2)) = __1;
    }
    for (int src_atom_idx = 0; src_atom_idx < 16; ++src_atom_idx) {
      tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (src_atom_idx * 2)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 32; ++i_16) {
      acc_o_T[(((i_16 & 15) * 2) + (i_16 >> 4))] = (acc_o_T[(((i_16 & 15) * 2) + (i_16 >> 4))] * scores_scale[(i_16 >> 4)]);
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_17 = 0; i_17 < 8; ++i_17) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_17 >> 2) * 4096) + (ki_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_17 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 17408)])) + 0, A_local_1 + (i_17 * 8));
      }
      for (int i_18 = 0; i_18 < 8; ++i_18) {
        for (int j = 0; j < 2; ++j) {
          E_local[((i_18 * 2) + j)] = ((short*)buf_dyn_shmem)[((((((ki_1 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_18 * 16)) + (j * 8)) + (((int)threadIdx.x) >> 2)) + 25600)];
        }
      }
      for (int i_19 = 0; i_19 < 8; ++i_19) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + (i_19 * 4)))[0]), "=f"(((float *)(acc_o_T + (i_19 * 4)))[1]), "=f"(((float *)(acc_o_T + (i_19 * 4)))[2]), "=f"(((float *)(acc_o_T + (i_19 * 4)))[3])
      : "r"(((unsigned *)(A_local_1 + (i_19 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_19 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_19 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_19 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + (ki_1 * 8)))[0]), "r"(((unsigned *)(acc_s_cast_T + (ki_1 * 8)))[1]), "r"(((unsigned *)(acc_s_cast_T + (ki_1 * 8)))[2]), "r"(((unsigned *)(acc_s_cast_T + (ki_1 * 8)))[3]), "f"(((float *)(acc_o_T + (i_19 * 4)))[0]), "f"(((float *)(acc_o_T + (i_19 * 4)))[1]), "f"(((float *)(acc_o_T + (i_19 * 4)))[2]), "f"(((float *)(acc_o_T + (i_19 * 4)))[3]), "r"(((unsigned *)(E_local + (i_19 * 2)))[0]));
  }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_20 = 0; i_20 < 32; ++i_20) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_20 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_20 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_20 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 34816), V+((((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.x)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)i_20) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)8192), (((((k * 64) + (i_20 * 2)) + (((int)threadIdx.x) >> 4)) + 64) < (seq_kv >> 1)));
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 4; ++i_21) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_21 * 512) + (((int)threadIdx.x) * 16)) + 51200), V_E+(((((((((int64_t)((int)blockIdx.x)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)1024) + (((int64_t)k) * (int64_t)1024)) + (((int64_t)i_21) * (int64_t)256)) + ((((int64_t)((int)blockIdx.y)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)1024), (((((k * 8) + (i_21 * 2)) + (((int)threadIdx.x) >> 4)) + 8) < (seq_kv >> 4)));
    }
    tl::cp_async_commit();
  }
  #pragma unroll
  for (int i_22 = 0; i_22 < 32; ++i_22) {
    float condval_2;
    if (((((((seq_kv + 127) >> 7) * 128) + ((i_22 & 15) * 8)) + (((int)threadIdx.x) >> 2)) < (seq_kv + 128))) {
      condval_2 = 0x0p+0f/*0.000000e+00*/;
    } else {
      condval_2 = -CUDART_INF_F;
    }
    acc_s_T[(((i_22 & 15) * 2) + (i_22 >> 4))] = condval_2;
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
    for (int i_23 = 0; i_23 < 8; ++i_23) {
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki_2 >> 1) * 8192) + (i_23 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])) + 0, A_local + (i_23 * 16));
      tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki_2 >> 1) * 8192) + (i_23 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1024)])) + 0, A_local + ((i_23 * 16) + 8));
    }
    tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki_2 >> 1) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_24 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_24 * 16)), reinterpret_cast<const unsigned*>(B_local + 0));
      tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_24 * 4)), reinterpret_cast<const unsigned*>(A_local + ((i_24 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + 4));
    }
  }
  *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
  #pragma unroll
  for (int i_25 = 0; i_25 < 2; ++i_25) {
    scores_max[i_25] = -CUDART_INF_F;
    #pragma unroll
    for (int rv_2 = 0; rv_2 < 16; ++rv_2) {
      scores_max[i_25] = max(scores_max[i_25], acc_s_T[((rv_2 * 2) + i_25)]);
    }
    scores_max[i_25] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_25]);
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 2; ++i_26) {
    scores_max[i_26] = max(scores_max[i_26], scores_max_prev[i_26]);
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 2; ++i_27) {
    scores_scale[i_27] = exp2f(((scores_max_prev[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_27] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 32; ++i_28) {
    acc_s_T[(((i_28 & 15) * 2) + (i_28 >> 4))] = exp2f(((acc_s_T[(((i_28 & 15) * 2) + (i_28 >> 4))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_28 >> 4)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
  }
  #pragma unroll
  for (int i_29 = 0; i_29 < 2; ++i_29) {
    scores_sum[i_29] = 0x0p+0f/*0.000000e+00*/;
    #pragma unroll
    for (int rv_3 = 0; rv_3 < 16; ++rv_3) {
      scores_sum[i_29] = (scores_sum[i_29] + acc_s_T[((rv_3 * 2) + i_29)]);
    }
    scores_sum[i_29] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_29]);
  }
  #pragma unroll
  for (int i_30 = 0; i_30 < 2; ++i_30) {
    logsum[i_30] = ((logsum[i_30] * scores_scale[i_30]) + scores_sum[i_30]);
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    uint1 __2;
    float2 v__1 = *(float2*)(acc_s_T + (i_31 * 2));
    ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&v__1))[0]);
    *(uint1*)(acc_s_T_ + (i_31 * 2)) = __2;
  }
  for (int src_atom_idx_1 = 0; src_atom_idx_1 < 16; ++src_atom_idx_1) {
    tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx_1 * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (src_atom_idx_1 * 2)));
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 32; ++i_32) {
    acc_o_T[(((i_32 & 15) * 2) + (i_32 >> 4))] = (acc_o_T[(((i_32 & 15) * 2) + (i_32 >> 4))] * scores_scale[(i_32 >> 4)]);
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
    for (int i_33 = 0; i_33 < 8; ++i_33) {
      tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_33 >> 2) * 4096) + (ki_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_33 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_33 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 17408)])) + 0, A_local_1 + (i_33 * 8));
    }
    for (int i_34 = 0; i_34 < 8; ++i_34) {
      for (int j_1 = 0; j_1 < 2; ++j_1) {
        E_local[((i_34 * 2) + j_1)] = ((short*)buf_dyn_shmem)[((((((ki_3 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_34 * 16)) + (j_1 * 8)) + (((int)threadIdx.x) >> 2)) + 25600)];
      }
    }
    for (int i_35 = 0; i_35 < 8; ++i_35) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + (i_35 * 4)))[0]), "=f"(((float *)(acc_o_T + (i_35 * 4)))[1]), "=f"(((float *)(acc_o_T + (i_35 * 4)))[2]), "=f"(((float *)(acc_o_T + (i_35 * 4)))[3])
      : "r"(((unsigned *)(A_local_1 + (i_35 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_35 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_35 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_35 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + (ki_3 * 8)))[0]), "r"(((unsigned *)(acc_s_cast_T + (ki_3 * 8)))[1]), "r"(((unsigned *)(acc_s_cast_T + (ki_3 * 8)))[2]), "r"(((unsigned *)(acc_s_cast_T + (ki_3 * 8)))[3]), "f"(((float *)(acc_o_T + (i_35 * 4)))[0]), "f"(((float *)(acc_o_T + (i_35 * 4)))[1]), "f"(((float *)(acc_o_T + (i_35 * 4)))[2]), "f"(((float *)(acc_o_T + (i_35 * 4)))[3]), "r"(((unsigned *)(E_local + (i_35 * 2)))[0]));
  }
    }
  }
  #pragma unroll
  for (int i_36 = 0; i_36 < 32; ++i_36) {
    acc_o_T[(((i_36 & 15) * 2) + (i_36 >> 4))] = (acc_o_T[(((i_36 & 15) * 2) + (i_36 >> 4))] / logsum[(i_36 >> 4)]);
  }
  #pragma unroll
  for (int i_37 = 0; i_37 < 2; ++i_37) {
    logsum[i_37] = ((log2f(logsum[i_37]) + (scores_max[i_37] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  if ((((int)threadIdx.x) >> 2) == 0) {
    if ((((int)threadIdx.x) & 3) < 2) {
      uint1 __3;
      float2 v__2 = *(float2*)(logsum + 0);
      ((half2*)(&__3))[0] = __float22half2_rn(((float2*)(&v__2))[0]);
      *(uint1*)(lse_combined + (((((int)blockIdx.x) * 32) + (((int)blockIdx.y) * 4)) + ((((int)threadIdx.x) & 3) * 2))) = __3;
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_38 = 0; i_38 < 32; ++i_38) {
    if ((((int)threadIdx.x) & 3) < 2) {
      ((half_t*)buf_dyn_shmem)[(((((((((i_38 & 15) >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + ((i_38 >> 4) * 64)) + (((i_38 & 7) >> 2) * 32)) + (((((i_38 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_38 >> 4) + (i_38 & 1)) & 1) * 8)) + (((int)threadIdx.x) >> 2))] = ((half_t)acc_o_T[(((i_38 & 15) * 2) + (i_38 >> 4))]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_39 = 0; i_39 < 2; ++i_39) {
    *(uint4*)(Output + ((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 512)) + (i_39 * 256)) + (((int)threadIdx.x) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 512) + (i_39 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((int)threadIdx.x) & 7) >> 2) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_39) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
}

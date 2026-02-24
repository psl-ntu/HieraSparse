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

extern "C" __global__ void flashattn_sp_v_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse, int seq_kv, int seq_q);
extern "C" __global__ void __launch_bounds__(256, 1) flashattn_sp_v_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse, int seq_kv, int seq_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o_T[64];
  float logsum[4];
  float scores_max[4];
  float acc_s_T[64];
  half_t A_local[128];
  half_t B_local[16];
  float scores_max_prev[4];
  float scores_scale[4];
  float scores_sum[4];
  half_t acc_s_T_[64];
  half_t acc_s_cast_T[64];
  half_t A_local_1[64];
  short E_local[16];
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+(((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((((int)blockIdx.x) * 128) + (i * 16)) + (((int)threadIdx.x) >> 4)) < seq_q));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_1 * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+((((((int64_t)i_1) * (int64_t)2048) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_1 * 16) + (((int)threadIdx.x) >> 4)) < seq_kv));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 32; ++i_2) {
    *(float2*)(acc_o_T + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 2; ++i_3) {
    *(float2*)(logsum + (i_3 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 2; ++i_4) {
    *(float2*)(scores_max + (i_4 * 2)) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  }
  for (int k = 0; k < min(((seq_kv + 127) >> 7), (((int)blockIdx.x) + 1)); ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 65536), V+(((((((int64_t)k) * (int64_t)8192) + (((int64_t)i_5) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.z)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((k * 64) + (i_5 * 16)) + (((int)threadIdx.x) >> 4)) < (seq_kv >> 1)));
    }
    tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((int)threadIdx.x) * 8) + 81920), V_E+(((((((int64_t)((int)blockIdx.z)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)1024) + (((int64_t)k) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), (((k * 8) + (((int)threadIdx.x) >> 5)) < (seq_kv >> 4)));
    tl::cp_async_commit();
    #pragma unroll
    for (int i_6 = 0; i_6 < 32; ++i_6) {
      *(float2*)(acc_s_T + (i_6 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 8192) + (i_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, A_local + (i_7 * 16));
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 8192) + (i_7 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, A_local + ((i_7 * 16) + 8));
      }
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((ki >> 1) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + (i_8 * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + (i_8 * 8));
      }
      for (int i_9 = 0; i_9 < 8; ++i_9) {
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_9 * 8) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + (i_9 * 16)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_9 * 8) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + ((i_9 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    if (((k + 1) < ((seq_kv + 127) >> 7)) && (k < ((int)blockIdx.x))) {
      #pragma unroll
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i_10 * 2048)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+((((((((int64_t)k) * (int64_t)16384) + (((int64_t)i_10) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)16384), (((((k * 128) + (i_10 * 16)) + (((int)threadIdx.x) >> 4)) + 128) < seq_kv));
      }
      tl::cp_async_commit();
    }
    if ((((seq_kv + 127) >> 7) <= (k + 2)) || (((int)blockIdx.x) <= (k + 1))) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 64; ++i_11) {
        float condval;
        if ((((((k * 128) + ((i_11 & 15) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_11 >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_11 & 31) >> 4))) & ((((k * 128) + ((i_11 & 15) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_kv))) {
          condval = acc_s_T[((((((i_11 & 15) >> 1) * 8) + ((i_11 >> 5) * 4)) + ((i_11 & 1) * 2)) + ((i_11 & 31) >> 4))];
        } else {
          condval = -CUDART_INF_F;
        }
        acc_s_T[((((((i_11 & 15) >> 1) * 8) + ((i_11 >> 5) * 4)) + ((i_11 & 1) * 2)) + ((i_11 & 31) >> 4))] = condval;
      }
    }
    #pragma unroll
    for (int i_12 = 0; i_12 < 2; ++i_12) {
      *(float2*)(scores_max_prev + (i_12 * 2)) = *(float2*)(scores_max + (i_12 * 2));
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      scores_max[i_13] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_13] = max(scores_max[i_13], acc_s_T[(((((rv & 7) * 8) + ((i_13 >> 1) * 4)) + ((rv >> 3) * 2)) + (i_13 & 1))]);
      }
      scores_max[i_13] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_13]);
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 4; ++i_14) {
      scores_max[i_14] = max(scores_max[i_14], scores_max_prev[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      scores_scale[i_15] = exp2f(((scores_max_prev[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_15] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 64; ++i_16) {
      acc_s_T[((((((i_16 & 15) >> 1) * 8) + ((i_16 >> 5) * 4)) + ((i_16 & 1) * 2)) + ((i_16 & 31) >> 4))] = exp2f(((acc_s_T[((((((i_16 & 15) >> 1) * 8) + ((i_16 >> 5) * 4)) + ((i_16 & 1) * 2)) + ((i_16 & 31) >> 4))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_16 >> 4)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 4; ++i_17) {
      scores_sum[i_17] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_17] = (scores_sum[i_17] + acc_s_T[(((((rv_1 & 7) * 8) + ((i_17 >> 1) * 4)) + ((rv_1 >> 3) * 2)) + (i_17 & 1))]);
      }
      scores_sum[i_17] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_17]);
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 32; ++i_18) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_18 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_T_ + (i_18 * 2)) = __1;
    }
    for (int src_atom_idx = 0; src_atom_idx < 32; ++src_atom_idx) {
      tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (((((src_atom_idx >> 3) * 16) + (((src_atom_idx & 3) >> 1) * 8)) + (((src_atom_idx & 7) >> 2) * 4)) + ((src_atom_idx & 1) * 2))));
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 4; ++i_19) {
      logsum[i_19] = ((logsum[i_19] * scores_scale[i_19]) + scores_sum[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 64; ++i_20) {
      acc_o_T[((((((i_20 & 15) >> 1) * 8) + ((i_20 >> 5) * 4)) + ((i_20 & 1) * 2)) + ((i_20 & 31) >> 4))] = (acc_o_T[((((((i_20 & 15) >> 1) * 8) + ((i_20 >> 5) * 4)) + ((i_20 & 1) * 2)) + ((i_20 & 31) >> 4))] * scores_scale[(i_20 >> 4)]);
    }
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_21 = 0; i_21 < 8; ++i_21) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_21 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_21 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_21 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 32768)])) + 0, A_local_1 + (i_21 * 8));
      }
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        for (int j_1 = 0; j_1 < 2; ++j_1) {
          E_local[((i_22 * 2) + j_1)] = ((short*)buf_dyn_shmem)[((((((ki_1 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_22 * 16)) + (j_1 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 40960)];
        }
      }
      for (int i_23 = 0; i_23 < 8; ++i_23) {
        for (int j_2 = 0; j_2 < 2; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[0]), "=f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[1]), "=f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[2]), "=f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[3])
      : "r"(((unsigned *)(A_local_1 + (i_23 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_23 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_23 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_23 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 16) + (j_2 * 8))))[0]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 16) + (j_2 * 8))))[1]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 16) + (j_2 * 8))))[2]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 16) + (j_2 * 8))))[3]), "f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[0]), "f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[1]), "f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[2]), "f"(((float *)(acc_o_T + ((i_23 * 8) + (j_2 * 4))))[3]), "r"(((unsigned *)(E_local + (i_23 * 2)))[0]));
  }
        }
      }
    }
  }
  #pragma unroll
  for (int i_24 = 0; i_24 < 64; ++i_24) {
    acc_o_T[((((((i_24 & 15) >> 1) * 8) + ((i_24 >> 5) * 4)) + ((i_24 & 1) * 2)) + ((i_24 & 31) >> 4))] = (acc_o_T[((((((i_24 & 15) >> 1) * 8) + ((i_24 >> 5) * 4)) + ((i_24 & 1) * 2)) + ((i_24 & 31) >> 4))] / logsum[(i_24 >> 4)]);
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 4; ++i_25) {
    logsum[i_25] = ((log2f(logsum[i_25]) + (scores_max[i_25] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i_26 = 0; i_26 < 64; ++i_26) {
    ((half_t*)buf_dyn_shmem)[(((((((((((i_26 & 15) >> 3) * 8192) + ((((int)threadIdx.x) >> 5) * 1024)) + ((i_26 >> 5) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_26 & 31) >> 4) * 64)) + (((((i_26 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_26 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_26 & 31) >> 4) + (i_26 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((half_t)acc_o_T[((((((i_26 & 15) >> 1) * 8) + ((i_26 >> 5) * 4)) + ((i_26 & 1) * 2)) + ((i_26 & 31) >> 4))]);
  }
  if (((((int)threadIdx.x) & 31) >> 2) == 0) {
    #pragma unroll
    for (int i_27 = 0; i_27 < 4; ++i_27) {
      if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_27 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_27 & 1)) < seq_q) {
        lse[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_27) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q))) + (((int64_t)i_27) & (int64_t)1))] = ((half_t)logsum[i_27]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_28 = 0; i_28 < 8; ++i_28) {
    if ((((((int)blockIdx.x) * 128) + (i_28 * 16)) + (((int)threadIdx.x) >> 4)) < seq_q) {
      *(uint4*)(O + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i_28) * (int64_t)2048)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_28 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}

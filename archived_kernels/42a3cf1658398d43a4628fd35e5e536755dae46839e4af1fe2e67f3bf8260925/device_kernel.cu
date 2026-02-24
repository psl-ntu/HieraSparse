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

extern "C" __global__ void flashattn_sp_kv_kernel_kernel(const half_t* __restrict__ K, const short* __restrict__ K_E, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse, int seq_kv, int seq_q);
extern "C" __global__ void __launch_bounds__(128, 1) flashattn_sp_kv_kernel_kernel(const half_t* __restrict__ K, const short* __restrict__ K_E, half_t* __restrict__ O, const half_t* __restrict__ Q, const half_t* __restrict__ V, const short* __restrict__ V_E, half_t* __restrict__ lse, int seq_kv, int seq_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o_T[96];
  float logsum[6];
  float scores_max[6];
  float acc_s_T[96];
  half_t A_local[64];
  short E_local[16];
  half_t B_local[24];
  float scores_max_prev[6];
  float scores_scale[6];
  float scores_sum[6];
  half_t acc_s_T_[96];
  half_t acc_s_cast_T[96];
  half_t A_local_1[64];
  short E_local_1[16];
  #pragma unroll
  for (int i = 0; i < 12; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 12288) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+(((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((((int)blockIdx.x) * 96) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_q));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K+((((((int64_t)i_1) * (int64_t)1024) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)512)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_1 * 16) + (((int)threadIdx.x) >> 3)) < seq_kv));
  }
  tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((int)threadIdx.x) * 16) + 40960), K_E+((((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)64) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((int)threadIdx.x) < seq_kv));
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 48; ++i_2) {
    *(float2*)(acc_o_T + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 3; ++i_3) {
    *(float2*)(logsum + (i_3 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 3; ++i_4) {
    *(float2*)(scores_max + (i_4 * 2)) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  }
  for (int k = 0; k < (min((seq_kv + 127), ((((int)blockIdx.x) * 96) + 223)) >> 7); ++k) {
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 43008), V+(((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.z)) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)1024)) + (((int64_t)i_5) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (((int64_t)seq_kv) >> (int64_t)1)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((k * 64) + (i_5 * 8)) + (((int)threadIdx.x) >> 4)) < (seq_kv >> 1)));
    }
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((int)threadIdx.x) * 16) + 59392), V_E+(((((((int64_t)((int)blockIdx.z)) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)1024) + (((int64_t)k) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * (((int64_t)seq_kv) >> (int64_t)4)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((k * 8) + (((int)threadIdx.x) >> 4)) < (seq_kv >> 4)));
    tl::cp_async_commit();
    #pragma unroll
    for (int i_6 = 0; i_6 < 48; ++i_6) {
      *(float2*)(acc_s_T + (i_6 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_7 = 0; i_7 < 8; ++i_7) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((i_7 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local + (i_7 * 8));
      }
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        for (int j = 0; j < 2; ++j) {
          E_local[((i_8 * 2) + j)] = ((short*)buf_dyn_shmem)[((((((i_8 * 128) + (j * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + (ki * 2)) + (((int)threadIdx.x) & 1)) + 20480)];
        }
      }
      for (int i_9 = 0; i_9 < 3; ++i_9) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((ki >> 1) * 6144) + ((((int)threadIdx.x) >> 5) * 1536)) + (i_9 * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + (i_9 * 8));
      }
      for (int i_10 = 0; i_10 < 8; ++i_10) {
        for (int j_1 = 0; j_1 < 3; ++j_1) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[0]), "=f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[1]), "=f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[2]), "=f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[3])
      : "r"(((unsigned *)(A_local + (i_10 * 8)))[0]), "r"(((unsigned *)(A_local + (i_10 * 8)))[1]), "r"(((unsigned *)(A_local + (i_10 * 8)))[2]), "r"(((unsigned *)(A_local + (i_10 * 8)))[3]), "r"(((unsigned *)(B_local + (j_1 * 8)))[0]), "r"(((unsigned *)(B_local + (j_1 * 8)))[1]), "r"(((unsigned *)(B_local + (j_1 * 8)))[2]), "r"(((unsigned *)(B_local + (j_1 * 8)))[3]), "f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[0]), "f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[1]), "f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[2]), "f"(((float *)(acc_s_T + ((i_10 * 12) + (j_1 * 4))))[3]), "r"(((unsigned *)(E_local + (i_10 * 2)))[0]));
  }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((k + 1) < (min((seq_kv + 127), ((((int)blockIdx.x) * 96) + 223)) >> 7)) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_11 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K+((((((((int64_t)k) * (int64_t)8192) + (((int64_t)i_11) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)512)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)64)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)8192), (((((k * 128) + (i_11 * 16)) + (((int)threadIdx.x) >> 3)) + 128) < seq_kv));
      }
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((int)threadIdx.x) * 16) + 40960), K_E+(((((((int64_t)k) * (int64_t)1024) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_kv)) * (int64_t)64)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)seq_kv)) * (int64_t)8)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)1024), ((((k * 128) + ((int)threadIdx.x)) + 128) < seq_kv));
      tl::cp_async_commit();
    }
    if ((min((seq_kv + 127), ((((int)blockIdx.x) * 96) + 223)) >> 7) <= (k + 2)) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 96; ++i_12) {
        float condval;
        if ((((((k * 128) + ((i_12 & 15) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((((int)blockIdx.x) * 96) + ((((int)threadIdx.x) >> 5) * 24)) + ((i_12 >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_12 & 31) >> 4))) & ((((k * 128) + ((i_12 & 15) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_kv))) {
          condval = acc_s_T[((((((i_12 & 15) >> 1) * 12) + ((i_12 >> 5) * 4)) + ((i_12 & 1) * 2)) + ((i_12 & 31) >> 4))];
        } else {
          condval = -CUDART_INF_F;
        }
        acc_s_T[((((((i_12 & 15) >> 1) * 12) + ((i_12 >> 5) * 4)) + ((i_12 & 1) * 2)) + ((i_12 & 31) >> 4))] = condval;
      }
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 3; ++i_13) {
      *(float2*)(scores_max_prev + (i_13 * 2)) = *(float2*)(scores_max + (i_13 * 2));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 6; ++i_14) {
      scores_max[i_14] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 16; ++rv) {
        scores_max[i_14] = max(scores_max[i_14], acc_s_T[(((((rv & 7) * 12) + ((i_14 >> 1) * 4)) + ((rv >> 3) * 2)) + (i_14 & 1))]);
      }
      scores_max[i_14] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 6; ++i_15) {
      scores_max[i_15] = max(scores_max[i_15], scores_max_prev[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 6; ++i_16) {
      scores_scale[i_16] = exp2f(((scores_max_prev[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 96; ++i_17) {
      acc_s_T[((((((i_17 & 15) >> 1) * 12) + ((i_17 >> 5) * 4)) + ((i_17 & 1) * 2)) + ((i_17 & 31) >> 4))] = exp2f(((acc_s_T[((((((i_17 & 15) >> 1) * 12) + ((i_17 >> 5) * 4)) + ((i_17 & 1) * 2)) + ((i_17 & 31) >> 4))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_17 >> 4)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 6; ++i_18) {
      scores_sum[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 16; ++rv_1) {
        scores_sum[i_18] = (scores_sum[i_18] + acc_s_T[(((((rv_1 & 7) * 12) + ((i_18 >> 1) * 4)) + ((rv_1 >> 3) * 2)) + (i_18 & 1))]);
      }
      scores_sum[i_18] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 48; ++i_19) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_19 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_T_ + (i_19 * 2)) = __1;
    }
    for (int src_atom_idx = 0; src_atom_idx < 48; ++src_atom_idx) {
      tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (((((src_atom_idx / 12) * 24) + (((src_atom_idx % 6) >> 1) * 8)) + (((src_atom_idx % 12) / 6) * 4)) + ((src_atom_idx & 1) * 2))));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 6; ++i_20) {
      logsum[i_20] = ((logsum[i_20] * scores_scale[i_20]) + scores_sum[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 96; ++i_21) {
      acc_o_T[((((((i_21 & 15) >> 1) * 12) + ((i_21 >> 5) * 4)) + ((i_21 & 1) * 2)) + ((i_21 & 31) >> 4))] = (acc_o_T[((((((i_21 & 15) >> 1) * 12) + ((i_21 >> 5) * 4)) + ((i_21 & 1) * 2)) + ((i_21 & 31) >> 4))] * scores_scale[(i_21 >> 4)]);
    }
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      for (int i_22 = 0; i_22 < 8; ++i_22) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_22 >> 2) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 21504)])) + 0, A_local_1 + (i_22 * 8));
      }
      for (int i_23 = 0; i_23 < 8; ++i_23) {
        for (int j_2 = 0; j_2 < 2; ++j_2) {
          E_local_1[((i_23 * 2) + j_2)] = ((short*)buf_dyn_shmem)[((((((ki_1 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_23 * 16)) + (j_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 29696)];
        }
      }
      for (int i_24 = 0; i_24 < 8; ++i_24) {
        for (int j_3 = 0; j_3 < 3; ++j_3) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[0]), "=f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[1]), "=f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[2]), "=f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[3])
      : "r"(((unsigned *)(A_local_1 + (i_24 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_24 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_24 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_24 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 24) + (j_3 * 8))))[0]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 24) + (j_3 * 8))))[1]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 24) + (j_3 * 8))))[2]), "r"(((unsigned *)(acc_s_cast_T + ((ki_1 * 24) + (j_3 * 8))))[3]), "f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[0]), "f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[1]), "f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[2]), "f"(((float *)(acc_o_T + ((i_24 * 12) + (j_3 * 4))))[3]), "r"(((unsigned *)(E_local_1 + (i_24 * 2)))[0]));
  }
        }
      }
    }
  }
  #pragma unroll
  for (int i_25 = 0; i_25 < 96; ++i_25) {
    acc_o_T[((((((i_25 & 15) >> 1) * 12) + ((i_25 >> 5) * 4)) + ((i_25 & 1) * 2)) + ((i_25 & 31) >> 4))] = (acc_o_T[((((((i_25 & 15) >> 1) * 12) + ((i_25 >> 5) * 4)) + ((i_25 & 1) * 2)) + ((i_25 & 31) >> 4))] / logsum[(i_25 >> 4)]);
  }
  #pragma unroll
  for (int i_26 = 0; i_26 < 6; ++i_26) {
    logsum[i_26] = ((log2f(logsum[i_26]) + (scores_max[i_26] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i_27 = 0; i_27 < 96; ++i_27) {
    ((half_t*)buf_dyn_shmem)[(((((((((((i_27 & 15) >> 3) * 6144) + ((((int)threadIdx.x) >> 5) * 1536)) + ((i_27 >> 5) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_27 & 31) >> 4) * 64)) + (((((i_27 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_27 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_27 & 31) >> 4) + (i_27 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((half_t)acc_o_T[((((((i_27 & 15) >> 1) * 12) + ((i_27 >> 5) * 4)) + ((i_27 & 1) * 2)) + ((i_27 & 31) >> 4))]);
  }
  if (((((int)threadIdx.x) & 31) >> 2) == 0) {
    #pragma unroll
    for (int i_28 = 0; i_28 < 6; ++i_28) {
      if ((((((((int)blockIdx.x) * 96) + ((((int)threadIdx.x) >> 5) * 24)) + ((i_28 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_28 & 1)) < seq_q) {
        lse[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)96) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)24)) + ((((int64_t)i_28) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q))) + (((int64_t)i_28) & (int64_t)1))] = ((half_t)logsum[i_28]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 12; ++i_29) {
    if ((((((int)blockIdx.x) * 96) + (i_29 * 8)) + (((int)threadIdx.x) >> 4)) < seq_q) {
      *(uint4*)(O + (((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i_29) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 6144) + (i_29 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}

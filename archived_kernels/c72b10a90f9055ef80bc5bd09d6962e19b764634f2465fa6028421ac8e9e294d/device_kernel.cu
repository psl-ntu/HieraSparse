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

extern "C" __global__ void blockattn_sp_dk_mv_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const short* __restrict__ V_E_blocks, const half_t* __restrict__ V_dense_blocks, const short* __restrict__ V_page_idx, const half_t* __restrict__ V_sparse_blocks, half_t* __restrict__ lse, int dummy0, int dummy1, int seq_kv, int seq_q, int v_dense_blocks, int v_sparse_blocks);
extern "C" __global__ void __launch_bounds__(128, 1) blockattn_sp_dk_mv_kernel_kernel(const half_t* __restrict__ K, half_t* __restrict__ O, const half_t* __restrict__ Q, const short* __restrict__ V_E_blocks, const half_t* __restrict__ V_dense_blocks, const short* __restrict__ V_page_idx, const half_t* __restrict__ V_sparse_blocks, half_t* __restrict__ lse, int dummy0, int dummy1, int seq_kv, int seq_q, int v_dense_blocks, int v_sparse_blocks) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o_T[128];
  float logsum[8];
  float scores_max[8];
  short v_page_idx = (short)0;
  float acc_s_T[64];
  half_t A_local[64];
  half_t B_local[32];
  float scores_max_prev[8];
  float scores_scale[8];
  float scores_sum[8];
  half_t acc_s_T_[64];
  half_t acc_s_cast_T[64];
  half_t A_local_1[128];
  half_t A_local_2[64];
  short E_local[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 16384) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+(((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((((int)blockIdx.x) * 128) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_q));
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 8; ++i_1) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy0)) * (int64_t)1024) + (((int64_t)i_1) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy0)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < dummy0));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 64; ++i_2) {
    *(float2*)(acc_o_T + (i_2 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_3 = 0; i_3 < 4; ++i_3) {
    *(float2*)(logsum + (i_3 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 4; ++i_4) {
    *(float2*)(scores_max + (i_4 * 2)) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  }
  for (int k = 0; k < min(((seq_kv + 63) >> 6), ((((int)blockIdx.x) * 2) + 2)); ++k) {
    short condval;
    if ((k < dummy1)) {
      condval = V_page_idx[((((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy1)) * (int64_t)8) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy1))) + ((int64_t)k))];
    } else {
      condval = (short)0;
    }
    v_page_idx = condval;
    tl::cp_async_wait<0>();
    __syncthreads();
    if (0 < ((int)v_page_idx)) {
      int k_page_idx = (((int)v_page_idx) - 1);
      #pragma unroll
      for (int i_5 = 0; i_5 < 8; ++i_5) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), V_dense_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)v_dense_blocks)) * (int64_t)65536) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)v_dense_blocks)) * (int64_t)8192)) + (((int64_t)k_page_idx) * (int64_t)8192)) + (((int64_t)i_5) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx) && (k_page_idx < v_dense_blocks)));
      }
    } else {
      int k_page_idx_1 = (((int)(v_page_idx * (short)-1)) - 1);
      #pragma unroll
      for (int i_6 = 0; i_6 < 4; ++i_6) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 49152), V_sparse_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)v_sparse_blocks)) * (int64_t)32768) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)v_sparse_blocks)) * (int64_t)4096)) + (((int64_t)k_page_idx_1) * (int64_t)4096)) + (((int64_t)i_6) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_1) && (k_page_idx_1 < v_sparse_blocks)));
      }
      tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((int)threadIdx.x) * 8) + 57344), V_E_blocks+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)v_sparse_blocks)) * (int64_t)4096) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)v_sparse_blocks)) * (int64_t)512)) + (((int64_t)k_page_idx_1) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), ((0 <= k_page_idx_1) && (k_page_idx_1 < v_sparse_blocks)));
    }
    tl::cp_async_commit();
    #pragma unroll
    for (int i_7 = 0; i_7 < 32; ++i_7) {
      *(float2*)(acc_s_T + (i_7 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
    }
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int i_8 = 0; i_8 < 4; ++i_8) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, A_local + (i_8 * 16));
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 16384)])) + 0, A_local + ((i_8 * 16) + 8));
      }
      for (int i_9 = 0; i_9 < 4; ++i_9) {
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((ki >> 1) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + (i_9 * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + (i_9 * 8));
      }
      for (int i_10 = 0; i_10 < 4; ++i_10) {
        for (int j = 0; j < 4; ++j) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_10 * 16) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 16)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_10 * 16) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + ((i_10 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    if (((k + 1) < ((seq_kv + 63) >> 6)) && (k <= (((int)blockIdx.x) * 2))) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_11 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 32768), K+((((((((int64_t)k) * (int64_t)8192) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy0)) * (int64_t)1024)) + (((int64_t)i_11) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy0)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)) + (int64_t)8192), (((((k * 64) + (i_11 * 8)) + (((int)threadIdx.x) >> 4)) + 64) < dummy0));
      }
      tl::cp_async_commit();
    }
    if ((((seq_kv + 63) >> 6) <= (k + 3)) || ((((int)blockIdx.x) * 2) <= (k + 1))) {
      #pragma unroll
      for (int i_12 = 0; i_12 < 64; ++i_12) {
        float condval_1;
        if ((((((k * 64) + ((i_12 & 7) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_12 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_12 & 15) >> 3))) & ((((k * 64) + ((i_12 & 7) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_kv))) {
          condval_1 = acc_s_T[((((((i_12 & 7) >> 1) * 16) + ((i_12 >> 4) * 4)) + ((i_12 & 1) * 2)) + ((i_12 & 15) >> 3))];
        } else {
          condval_1 = -CUDART_INF_F;
        }
        acc_s_T[((((((i_12 & 7) >> 1) * 16) + ((i_12 >> 4) * 4)) + ((i_12 & 1) * 2)) + ((i_12 & 15) >> 3))] = condval_1;
      }
    }
    #pragma unroll
    for (int i_13 = 0; i_13 < 4; ++i_13) {
      *(float2*)(scores_max_prev + (i_13 * 2)) = *(float2*)(scores_max + (i_13 * 2));
    }
    #pragma unroll
    for (int i_14 = 0; i_14 < 8; ++i_14) {
      scores_max[i_14] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_14] = max(scores_max[i_14], acc_s_T[(((((rv & 3) * 16) + ((i_14 >> 1) * 4)) + ((rv >> 2) * 2)) + (i_14 & 1))]);
      }
      scores_max[i_14] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_14]);
    }
    #pragma unroll
    for (int i_15 = 0; i_15 < 8; ++i_15) {
      scores_max[i_15] = max(scores_max[i_15], scores_max_prev[i_15]);
    }
    #pragma unroll
    for (int i_16 = 0; i_16 < 8; ++i_16) {
      scores_scale[i_16] = exp2f(((scores_max_prev[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_16] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_17 = 0; i_17 < 64; ++i_17) {
      acc_s_T[((((((i_17 & 7) >> 1) * 16) + ((i_17 >> 4) * 4)) + ((i_17 & 1) * 2)) + ((i_17 & 15) >> 3))] = exp2f(((acc_s_T[((((((i_17 & 7) >> 1) * 16) + ((i_17 >> 4) * 4)) + ((i_17 & 1) * 2)) + ((i_17 & 15) >> 3))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_17 >> 3)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 8; ++i_18) {
      scores_sum[i_18] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_18] = (scores_sum[i_18] + acc_s_T[(((((rv_1 & 3) * 16) + ((i_18 >> 1) * 4)) + ((rv_1 >> 2) * 2)) + (i_18 & 1))]);
      }
      scores_sum[i_18] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_18]);
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 32; ++i_19) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_19 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_T_ + (i_19 * 2)) = __1;
    }
    for (int src_atom_idx = 0; src_atom_idx < 32; ++src_atom_idx) {
      tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (((((src_atom_idx >> 4) * 32) + (((src_atom_idx & 7) >> 1) * 8)) + (((src_atom_idx & 15) >> 3) * 4)) + ((src_atom_idx & 1) * 2))));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 8; ++i_20) {
      logsum[i_20] = ((logsum[i_20] * scores_scale[i_20]) + scores_sum[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 128; ++i_21) {
      acc_o_T[((((((i_21 & 15) >> 1) * 16) + ((i_21 >> 5) * 4)) + ((i_21 & 1) * 2)) + ((i_21 & 31) >> 4))] = (acc_o_T[((((((i_21 & 15) >> 1) * 16) + ((i_21 >> 5) * 4)) + ((i_21 & 1) * 2)) + ((i_21 & 31) >> 4))] * scores_scale[(i_21 >> 4)]);
    }
    __syncthreads();
    if (0 < ((int)v_page_idx)) {
      for (int ki_1 = 0; ki_1 < 2; ++ki_1) {
        for (int i_22 = 0; i_22 < 8; ++i_22) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_22 >> 2) * 4096) + (ki_1 * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])) + 0, A_local_1 + (i_22 * 16));
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_22 >> 2) * 4096) + (ki_1 * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_22 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_22 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 25600)])) + 0, A_local_1 + ((i_22 * 16) + 8));
        }
        for (int i_23 = 0; i_23 < 8; ++i_23) {
          for (int j_1 = 0; j_1 < 4; ++j_1) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o_T + ((i_23 * 16) + (j_1 * 4))), reinterpret_cast<const unsigned*>(A_local_1 + (i_23 * 16)), reinterpret_cast<const unsigned*>(acc_s_cast_T + ((ki_1 * 32) + (j_1 * 8))));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o_T + ((i_23 * 16) + (j_1 * 4))), reinterpret_cast<const unsigned*>(A_local_1 + ((i_23 * 16) + 8)), reinterpret_cast<const unsigned*>(acc_s_cast_T + (((ki_1 * 32) + (j_1 * 8)) + 4)));
          }
        }
      }
    } else {
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        for (int i_24 = 0; i_24 < 8; ++i_24) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_24 >> 2) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_24 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_24 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 24576)])) + 0, A_local_2 + (i_24 * 8));
        }
        for (int i_25 = 0; i_25 < 8; ++i_25) {
          for (int j_2 = 0; j_2 < 2; ++j_2) {
            E_local[((i_25 * 2) + j_2)] = ((short*)buf_dyn_shmem)[((((((ki_2 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_25 * 16)) + (j_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 28672)];
          }
        }
        for (int i_26 = 0; i_26 < 8; ++i_26) {
          for (int j_3 = 0; j_3 < 4; ++j_3) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[0]), "=f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[1]), "=f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[2]), "=f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[3])
      : "r"(((unsigned *)(A_local_2 + (i_26 * 8)))[0]), "r"(((unsigned *)(A_local_2 + (i_26 * 8)))[1]), "r"(((unsigned *)(A_local_2 + (i_26 * 8)))[2]), "r"(((unsigned *)(A_local_2 + (i_26 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 32) + (j_3 * 8))))[0]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 32) + (j_3 * 8))))[1]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 32) + (j_3 * 8))))[2]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 32) + (j_3 * 8))))[3]), "f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[0]), "f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[1]), "f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[2]), "f"(((float *)(acc_o_T + ((i_26 * 16) + (j_3 * 4))))[3]), "r"(((unsigned *)(E_local + (i_26 * 2)))[0]));
  }
          }
        }
      }
    }
  }
  #pragma unroll
  for (int i_27 = 0; i_27 < 128; ++i_27) {
    acc_o_T[((((((i_27 & 15) >> 1) * 16) + ((i_27 >> 5) * 4)) + ((i_27 & 1) * 2)) + ((i_27 & 31) >> 4))] = (acc_o_T[((((((i_27 & 15) >> 1) * 16) + ((i_27 >> 5) * 4)) + ((i_27 & 1) * 2)) + ((i_27 & 31) >> 4))] / logsum[(i_27 >> 4)]);
  }
  #pragma unroll
  for (int i_28 = 0; i_28 < 8; ++i_28) {
    logsum[i_28] = ((log2f(logsum[i_28]) + (scores_max[i_28] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i_29 = 0; i_29 < 128; ++i_29) {
    ((half_t*)buf_dyn_shmem)[(((((((((((i_29 & 15) >> 3) * 8192) + ((((int)threadIdx.x) >> 5) * 2048)) + ((i_29 >> 5) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_29 & 31) >> 4) * 64)) + (((((i_29 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_29 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_29 & 31) >> 4) + (i_29 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((half_t)acc_o_T[((((((i_29 & 15) >> 1) * 16) + ((i_29 >> 5) * 4)) + ((i_29 & 1) * 2)) + ((i_29 & 31) >> 4))]);
  }
  if (((((int)threadIdx.x) & 31) >> 2) == 0) {
    #pragma unroll
    for (int i_30 = 0; i_30 < 8; ++i_30) {
      if ((((((((int)blockIdx.x) * 128) + ((((int)threadIdx.x) >> 5) * 32)) + ((i_30 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_30 & 1)) < seq_q) {
        lse[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)32)) + ((((int64_t)i_30) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q))) + (((int64_t)i_30) & (int64_t)1))] = ((half_t)logsum[i_30]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_31 = 0; i_31 < 16; ++i_31) {
    if ((((((int)blockIdx.x) * 128) + (i_31 * 8)) + (((int)threadIdx.x) >> 4)) < seq_q) {
      *(uint4*)(O + (((((((int64_t)((int)blockIdx.x)) * (int64_t)16384) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i_31) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_31 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}

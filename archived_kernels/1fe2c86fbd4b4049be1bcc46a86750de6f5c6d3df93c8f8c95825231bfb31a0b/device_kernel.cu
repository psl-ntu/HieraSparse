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

extern "C" __global__ void blockattn_sp_mk_sv_kernel_kernel(const short* __restrict__ K_E_blocks, const half_t* __restrict__ K_dense_blocks, const short* __restrict__ K_page_idx, const half_t* __restrict__ K_sparse_blocks, half_t* __restrict__ O, const half_t* __restrict__ Q, const short* __restrict__ V_E, const half_t* __restrict__ V_SP, half_t* __restrict__ lse, int dummy0, int dummy1, int dummy2, int k_dense_blocks, int k_sparse_blocks, int seq_kv, int seq_q);
extern "C" __global__ void __launch_bounds__(128, 1) blockattn_sp_mk_sv_kernel_kernel(const short* __restrict__ K_E_blocks, const half_t* __restrict__ K_dense_blocks, const short* __restrict__ K_page_idx, const half_t* __restrict__ K_sparse_blocks, half_t* __restrict__ O, const half_t* __restrict__ Q, const short* __restrict__ V_E, const half_t* __restrict__ V_SP, half_t* __restrict__ lse, int dummy0, int dummy1, int dummy2, int k_dense_blocks, int k_sparse_blocks, int seq_kv, int seq_q) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  short next_k_page_idx = (short)0;
  float acc_o_T[96];
  float logsum[6];
  float scores_max[6];
  short k_page_idx = (short)0;
  float acc_s_T[48];
  half_t A_local[64];
  half_t B_local[24];
  half_t A_local_1[32];
  short E_local[8];
  half_t B_local_1[24];
  float scores_max_prev[6];
  float scores_scale[6];
  float scores_sum[6];
  half_t acc_s_T_[48];
  half_t acc_s_cast_T[48];
  half_t A_local_2[64];
  short E_local_1[16];
  next_k_page_idx = K_page_idx[(((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy0)) * (int64_t)8) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy0)))];
  #pragma unroll
  for (int i = 0; i < 12; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 12288) + (i * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+(((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((((int)blockIdx.x) * 96) + (i * 8)) + (((int)threadIdx.x) >> 4)) < seq_q));
  }
  if (0 < ((int)next_k_page_idx)) {
    int k_page_idx_1 = (((int)next_k_page_idx) - 1);
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K_dense_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_dense_blocks)) * (int64_t)65536) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_dense_blocks)) * (int64_t)8192)) + (((int64_t)k_page_idx_1) * (int64_t)8192)) + (((int64_t)i_1) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_1) && (k_page_idx_1 < k_dense_blocks)));
    }
  } else {
    int k_page_idx_2 = (((int)(next_k_page_idx * (short)-1)) - 1);
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_2 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K_sparse_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_sparse_blocks)) * (int64_t)32768) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_sparse_blocks)) * (int64_t)4096)) + (((int64_t)k_page_idx_2) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_2) && (k_page_idx_2 < k_sparse_blocks)));
    }
    tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((int)threadIdx.x) * 8) + 32768), K_E_blocks+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_sparse_blocks)) * (int64_t)4096) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_sparse_blocks)) * (int64_t)512)) + (((int64_t)k_page_idx_2) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), ((0 <= k_page_idx_2) && (k_page_idx_2 < k_sparse_blocks)));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_3 = 0; i_3 < 48; ++i_3) {
    *(float2*)(acc_o_T + (i_3 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_4 = 0; i_4 < 3; ++i_4) {
    *(float2*)(logsum + (i_4 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 3; ++i_5) {
    *(float2*)(scores_max + (i_5 * 2)) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  }
  for (int k = 0; k < (min((seq_kv + 63), ((((int)blockIdx.x) * 96) + 159)) >> 6); ++k) {
    short condval;
    if ((k < dummy0)) {
      condval = K_page_idx[((((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy0)) * (int64_t)8) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy0))) + ((int64_t)k))];
    } else {
      condval = (short)0;
    }
    k_page_idx = condval;
    tl::cp_async_wait<0>();
    __syncthreads();
    #pragma unroll
    for (int i_6 = 0; i_6 < 4; ++i_6) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 40960), V_SP+(((((((int64_t)k) * (int64_t)4096) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy1)) * (int64_t)1024)) + (((int64_t)i_6) * (int64_t)1024)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy1)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((((k * 32) + (i_6 * 8)) + (((int)threadIdx.x) >> 4)) < dummy1));
    }
    tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((int)threadIdx.x) * 8) + 49152), V_E+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy2)) * (int64_t)1024) + (((int64_t)k) * (int64_t)512)) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy2)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), (((k * 4) + (((int)threadIdx.x) >> 5)) < dummy2));
    tl::cp_async_commit();
    __syncthreads();
    if (0 < ((int)k_page_idx)) {
      #pragma unroll
      for (int i_7 = 0; i_7 < 24; ++i_7) {
        *(float2*)(acc_s_T + (i_7 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_8 = 0; i_8 < 4; ++i_8) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local + (i_8 * 16));
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_8 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local + ((i_8 * 16) + 8));
        }
        for (int i_9 = 0; i_9 < 3; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((ki >> 1) * 6144) + ((((int)threadIdx.x) >> 5) * 1536)) + (i_9 * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + (i_9 * 8));
        }
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          for (int j = 0; j < 3; ++j) {
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_10 * 12) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + (i_10 * 16)), reinterpret_cast<const unsigned*>(B_local + (j * 8)));
            tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + ((i_10 * 12) + (j * 4))), reinterpret_cast<const unsigned*>(A_local + ((i_10 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + ((j * 8) + 4)));
          }
        }
      }
    } else {
      #pragma unroll
      for (int i_11 = 0; i_11 < 24; ++i_11) {
        *(float2*)(acc_s_T + (i_11 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((i_12 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 12288)])) + 0, A_local_1 + (i_12 * 8));
        }
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            E_local[((i_13 * 2) + j_1)] = ((short*)buf_dyn_shmem)[((((((i_13 * 128) + (j_1 * 64)) + (((((int)threadIdx.x) & 31) >> 2) * 8)) + (ki_1 * 2)) + (((int)threadIdx.x) & 1)) + 16384)];
          }
        }
        for (int i_14 = 0; i_14 < 3; ++i_14) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((((ki_1 >> 1) * 6144) + ((((int)threadIdx.x) >> 5) * 1536)) + (i_14 * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local_1 + (i_14 * 8));
        }
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          for (int j_2 = 0; j_2 < 3; ++j_2) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[0]), "=f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[1]), "=f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[2]), "=f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[3])
      : "r"(((unsigned *)(A_local_1 + (i_15 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_15 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_15 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_15 * 8)))[3]), "r"(((unsigned *)(B_local_1 + (j_2 * 8)))[0]), "r"(((unsigned *)(B_local_1 + (j_2 * 8)))[1]), "r"(((unsigned *)(B_local_1 + (j_2 * 8)))[2]), "r"(((unsigned *)(B_local_1 + (j_2 * 8)))[3]), "f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[0]), "f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[1]), "f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[2]), "f"(((float *)(acc_s_T + ((i_15 * 12) + (j_2 * 4))))[3]), "r"(((unsigned *)(E_local + (i_15 * 2)))[0]));
  }
          }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    if ((k + 1) < (min((seq_kv + 63), ((((int)blockIdx.x) * 96) + 159)) >> 6)) {
      short condval_1;
      if (((k + 1) < dummy0)) {
        condval_1 = K_page_idx[(((((((int64_t)((int)blockIdx.z)) * ((int64_t)dummy0)) * (int64_t)8) + ((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)dummy0))) + ((int64_t)k)) + (int64_t)1)];
      } else {
        condval_1 = (short)0;
      }
      next_k_page_idx = condval_1;
      if (0 < ((int)next_k_page_idx)) {
        int k_page_idx_3 = (((int)next_k_page_idx) - 1);
        #pragma unroll
        for (int i_16 = 0; i_16 < 8; ++i_16) {
          tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 4) * 128)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K_dense_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_dense_blocks)) * (int64_t)65536) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_dense_blocks)) * (int64_t)8192)) + (((int64_t)k_page_idx_3) * (int64_t)8192)) + (((int64_t)i_16) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_3) && (k_page_idx_3 < k_dense_blocks)));
        }
      } else {
        int k_page_idx_4 = (((int)(next_k_page_idx * (short)-1)) - 1);
        #pragma unroll
        for (int i_17 = 0; i_17 < 4; ++i_17) {
          tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_17 * 2048) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 24576), K_sparse_blocks+((((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_sparse_blocks)) * (int64_t)32768) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_sparse_blocks)) * (int64_t)4096)) + (((int64_t)k_page_idx_4) * (int64_t)4096)) + (((int64_t)i_17) * (int64_t)1024)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_4) && (k_page_idx_4 < k_sparse_blocks)));
        }
        tl::cp_async_gs_conditional<8>(buf_dyn_shmem+((((int)threadIdx.x) * 8) + 32768), K_E_blocks+(((((((int64_t)((int)blockIdx.z)) * ((int64_t)k_sparse_blocks)) * (int64_t)4096) + (((((int64_t)((int)blockIdx.y)) >> (int64_t)2) * ((int64_t)k_sparse_blocks)) * (int64_t)512)) + (((int64_t)k_page_idx_4) * (int64_t)512)) + (((int64_t)((int)threadIdx.x)) * (int64_t)4)), ((0 <= k_page_idx_4) && (k_page_idx_4 < k_sparse_blocks)));
      }
      tl::cp_async_commit();
    }
    if ((min((seq_kv + 63), ((((int)blockIdx.x) * 96) + 159)) >> 6) <= (k + 3)) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 48; ++i_18) {
        float condval_2;
        if ((((((k * 64) + ((i_18 & 7) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) <= (((((((int)blockIdx.x) * 96) + ((((int)threadIdx.x) >> 5) * 24)) + ((i_18 >> 4) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + ((i_18 & 15) >> 3))) & ((((k * 64) + ((i_18 & 7) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < seq_kv))) {
          condval_2 = acc_s_T[((((((i_18 & 7) >> 1) * 12) + ((i_18 >> 4) * 4)) + ((i_18 & 1) * 2)) + ((i_18 & 15) >> 3))];
        } else {
          condval_2 = -CUDART_INF_F;
        }
        acc_s_T[((((((i_18 & 7) >> 1) * 12) + ((i_18 >> 4) * 4)) + ((i_18 & 1) * 2)) + ((i_18 & 15) >> 3))] = condval_2;
      }
    }
    #pragma unroll
    for (int i_19 = 0; i_19 < 3; ++i_19) {
      *(float2*)(scores_max_prev + (i_19 * 2)) = *(float2*)(scores_max + (i_19 * 2));
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 6; ++i_20) {
      scores_max[i_20] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_20] = max(scores_max[i_20], acc_s_T[(((((rv & 3) * 12) + ((i_20 >> 1) * 4)) + ((rv >> 2) * 2)) + (i_20 & 1))]);
      }
      scores_max[i_20] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 6; ++i_21) {
      scores_max[i_21] = max(scores_max[i_21], scores_max_prev[i_21]);
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 6; ++i_22) {
      scores_scale[i_22] = exp2f(((scores_max_prev[i_22] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_22] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 48; ++i_23) {
      acc_s_T[((((((i_23 & 7) >> 1) * 12) + ((i_23 >> 4) * 4)) + ((i_23 & 1) * 2)) + ((i_23 & 15) >> 3))] = exp2f(((acc_s_T[((((((i_23 & 7) >> 1) * 12) + ((i_23 >> 4) * 4)) + ((i_23 & 1) * 2)) + ((i_23 & 15) >> 3))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_23 >> 3)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 6; ++i_24) {
      scores_sum[i_24] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_24] = (scores_sum[i_24] + acc_s_T[(((((rv_1 & 3) * 12) + ((i_24 >> 1) * 4)) + ((rv_1 >> 2) * 2)) + (i_24 & 1))]);
      }
      scores_sum[i_24] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_24]);
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 24; ++i_25) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_25 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(acc_s_T_ + (i_25 * 2)) = __1;
    }
    for (int src_atom_idx = 0; src_atom_idx < 24; ++src_atom_idx) {
      tl::ptx_movmatrix(reinterpret_cast<const int32_t *>(acc_s_T_ + (src_atom_idx * 2)), reinterpret_cast<int32_t*>(acc_s_cast_T + (((((src_atom_idx / 12) * 24) + (((src_atom_idx % 6) >> 1) * 8)) + (((src_atom_idx % 12) / 6) * 4)) + ((src_atom_idx & 1) * 2))));
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 6; ++i_26) {
      logsum[i_26] = ((logsum[i_26] * scores_scale[i_26]) + scores_sum[i_26]);
    }
    #pragma unroll
    for (int i_27 = 0; i_27 < 96; ++i_27) {
      acc_o_T[((((((i_27 & 15) >> 1) * 12) + ((i_27 >> 5) * 4)) + ((i_27 & 1) * 2)) + ((i_27 & 31) >> 4))] = (acc_o_T[((((((i_27 & 15) >> 1) * 12) + ((i_27 >> 5) * 4)) + ((i_27 & 1) * 2)) + ((i_27 & 31) >> 4))] * scores_scale[(i_27 >> 4)]);
    }
    __syncthreads();
    for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
      for (int i_28 = 0; i_28 < 8; ++i_28) {
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_28 >> 2) * 2048) + (ki_2 * 1024)) + (((((int)threadIdx.x) & 31) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_28 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_28 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 20480)])) + 0, A_local_2 + (i_28 * 8));
      }
      for (int i_29 = 0; i_29 < 8; ++i_29) {
        for (int j_3 = 0; j_3 < 2; ++j_3) {
          E_local_1[((i_29 * 2) + j_3)] = ((short*)buf_dyn_shmem)[((((((ki_2 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_29 * 16)) + (j_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2)) + 24576)];
        }
      }
      for (int i_30 = 0; i_30 < 8; ++i_30) {
        for (int j_4 = 0; j_4 < 3; ++j_4) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[0]), "=f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[1]), "=f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[2]), "=f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[3])
      : "r"(((unsigned *)(A_local_2 + (i_30 * 8)))[0]), "r"(((unsigned *)(A_local_2 + (i_30 * 8)))[1]), "r"(((unsigned *)(A_local_2 + (i_30 * 8)))[2]), "r"(((unsigned *)(A_local_2 + (i_30 * 8)))[3]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 24) + (j_4 * 8))))[0]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 24) + (j_4 * 8))))[1]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 24) + (j_4 * 8))))[2]), "r"(((unsigned *)(acc_s_cast_T + ((ki_2 * 24) + (j_4 * 8))))[3]), "f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[0]), "f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[1]), "f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[2]), "f"(((float *)(acc_o_T + ((i_30 * 12) + (j_4 * 4))))[3]), "r"(((unsigned *)(E_local_1 + (i_30 * 2)))[0]));
  }
        }
      }
    }
  }
  #pragma unroll
  for (int i_31 = 0; i_31 < 96; ++i_31) {
    acc_o_T[((((((i_31 & 15) >> 1) * 12) + ((i_31 >> 5) * 4)) + ((i_31 & 1) * 2)) + ((i_31 & 31) >> 4))] = (acc_o_T[((((((i_31 & 15) >> 1) * 12) + ((i_31 >> 5) * 4)) + ((i_31 & 1) * 2)) + ((i_31 & 31) >> 4))] / logsum[(i_31 >> 4)]);
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 6; ++i_32) {
    logsum[i_32] = ((log2f(logsum[i_32]) + (scores_max[i_32] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)) * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/);
  }
  __syncthreads();
  #pragma unroll
  for (int i_33 = 0; i_33 < 96; ++i_33) {
    ((half_t*)buf_dyn_shmem)[(((((((((((i_33 & 15) >> 3) * 6144) + ((((int)threadIdx.x) >> 5) * 1536)) + ((i_33 >> 5) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((i_33 & 31) >> 4) * 64)) + (((((i_33 & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((i_33 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + (((((i_33 & 31) >> 4) + (i_33 & 1)) & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))] = ((half_t)acc_o_T[((((((i_33 & 15) >> 1) * 12) + ((i_33 >> 5) * 4)) + ((i_33 & 1) * 2)) + ((i_33 & 31) >> 4))]);
  }
  if (((((int)threadIdx.x) & 31) >> 2) == 0) {
    #pragma unroll
    for (int i_34 = 0; i_34 < 6; ++i_34) {
      if ((((((((int)blockIdx.x) * 96) + ((((int)threadIdx.x) >> 5) * 24)) + ((i_34 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + (i_34 & 1)) < seq_q) {
        lse[(((((((((int64_t)((int)blockIdx.x)) * (int64_t)96) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)32)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)24)) + ((((int64_t)i_34) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q))) + (((int64_t)i_34) & (int64_t)1))] = ((half_t)logsum[i_34]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 12; ++i_35) {
    if ((((((int)blockIdx.x) * 96) + (i_35 * 8)) + (((int)threadIdx.x) >> 4)) < seq_q) {
      *(uint4*)(O + (((((((int64_t)((int)blockIdx.x)) * (int64_t)12288) + ((((int64_t)((int)blockIdx.z)) * ((int64_t)seq_q)) * (int64_t)4096)) + (((int64_t)i_35) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)seq_q)) * (int64_t)128)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 6144) + (i_35 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
    }
  }
}

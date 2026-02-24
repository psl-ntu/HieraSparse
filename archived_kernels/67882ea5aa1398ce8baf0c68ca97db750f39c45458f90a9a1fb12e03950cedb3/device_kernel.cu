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

extern "C" __global__ void blockdecode_sp_mk_mv_split_kernel_kernel(const short* __restrict__ K_E_blocks, const half_t* __restrict__ K_dense_blocks, const short* __restrict__ K_page_idx, const half_t* __restrict__ K_sparse_blocks, half_t* __restrict__ Output_partial, const half_t* __restrict__ Q, const short* __restrict__ V_E_blocks, const half_t* __restrict__ V_dense_blocks, const short* __restrict__ V_page_idx, const half_t* __restrict__ V_sparse_blocks, half_t* __restrict__ glse, int dummy0, int dummy1, int k_dense_blocks_len, int k_sparse_blocks_len, int seq_kv, int v_dense_blocks_len, int v_sparse_blocks_len);
extern "C" __global__ void blockdecode_sp_mk_mv_split_kernel_kernel_1(half_t* __restrict__ Output, const half_t* __restrict__ Output_partial, const half_t* __restrict__ glse, half_t* __restrict__ lse_combined);
extern "C" __global__ void __launch_bounds__(32, 1) blockdecode_sp_mk_mv_split_kernel_kernel(const short* __restrict__ K_E_blocks, const half_t* __restrict__ K_dense_blocks, const short* __restrict__ K_page_idx, const half_t* __restrict__ K_sparse_blocks, half_t* __restrict__ Output_partial, const half_t* __restrict__ Q, const short* __restrict__ V_E_blocks, const half_t* __restrict__ V_dense_blocks, const short* __restrict__ V_page_idx, const half_t* __restrict__ V_sparse_blocks, half_t* __restrict__ glse, int dummy0, int dummy1, int k_dense_blocks_len, int k_sparse_blocks_len, int seq_kv, int v_dense_blocks_len, int v_sparse_blocks_len) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  short next_k_page_idx = (short)0;
  float acc_o_T[32];
  float logsum[2];
  float scores_max[2];
  short v_page_idx = (short)0;
  short k_page_idx = (short)0;
  float acc_s_T[16];
  half_t A_local[64];
  half_t B_local[8];
  half_t A_local_1[32];
  short E_local[8];
  half_t B_local_1[8];
  float scores_max_prev[2];
  float scores_scale[2];
  float scores_sum[2];
  half_t A_local_2[128];
  half_t B_local_2[8];
  half_t A_local_3[64];
  short E_local_1[16];
  half_t B_local_3[8];
  short condval;
  if (((0 <= (((seq_kv + 63) >> 7) * ((int)blockIdx.z))) && ((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) < dummy0))) {
    condval = K_page_idx[((((((int64_t)((int)blockIdx.x)) * ((int64_t)dummy0)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)dummy0))) + (((((int64_t)seq_kv) + (int64_t)63) >> (int64_t)7) * ((int64_t)((int)blockIdx.z))))];
  } else {
    condval = (short)0;
  }
  next_k_page_idx = condval;
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((((((int)threadIdx.x) & 15) >> 3) * 1024) + (i * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)), Q+((((((int)blockIdx.x) * 4096) + (((int)blockIdx.y) * 512)) + (i * 256)) + (((int)threadIdx.x) * 8)), (((i >> 1) + ((int)blockIdx.y)) < 8));
  }
  if (0 < ((int)next_k_page_idx)) {
    int k_page_idx_1 = (((int)next_k_page_idx) - 1);
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_1 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 3072), K_dense_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_dense_blocks_len)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_dense_blocks_len)) * (int64_t)8192)) + (((int64_t)k_page_idx_1) * (int64_t)8192)) + (((int64_t)i_1) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_1) && (k_page_idx_1 < k_dense_blocks_len)));
    }
  } else {
    int k_page_idx_2 = (((int)(next_k_page_idx * (short)-1)) - 1);
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_2 * 512) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_2 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 3072), K_sparse_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)4096)) + (((int64_t)k_page_idx_2) * (int64_t)4096)) + (((int64_t)i_2) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_2) && (k_page_idx_2 < k_sparse_blocks_len)));
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 2; ++i_3) {
      tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_3 * 512) + (((int)threadIdx.x) * 16)) + 11264), K_E_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)4096) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)512)) + (((int64_t)k_page_idx_2) * (int64_t)512)) + (((int64_t)i_3) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_2) && (k_page_idx_2 < k_sparse_blocks_len)));
    }
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_4 = 0; i_4 < 16; ++i_4) {
    *(float2*)(acc_o_T + (i_4 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  }
  *(float2*)(logsum + 0) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
  *(float2*)(scores_max + 0) = make_float2(-CUDART_INF_F, -CUDART_INF_F);
  int condval_1;
  if ((((int)blockIdx.z) == 1)) {
    condval_1 = (((seq_kv + 63) >> 6) - ((seq_kv + 63) >> 7));
  } else {
    condval_1 = ((seq_kv + 63) >> 7);
  }
  for (int k_local = 0; k_local < condval_1; ++k_local) {
    short condval_2;
    if (((0 <= ((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local)) && (((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local) < dummy0))) {
      condval_2 = K_page_idx[(((((((int64_t)((int)blockIdx.x)) * ((int64_t)dummy0)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)dummy0))) + (((((int64_t)seq_kv) + (int64_t)63) >> (int64_t)7) * ((int64_t)((int)blockIdx.z)))) + ((int64_t)k_local))];
    } else {
      condval_2 = (short)0;
    }
    k_page_idx = condval_2;
    short condval_3;
    if (((0 <= ((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local)) && (((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local) < dummy1))) {
      condval_3 = V_page_idx[(((((((int64_t)((int)blockIdx.x)) * ((int64_t)dummy1)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)dummy1))) + (((((int64_t)seq_kv) + (int64_t)63) >> (int64_t)7) * ((int64_t)((int)blockIdx.z)))) + ((int64_t)k_local))];
    } else {
      condval_3 = (short)0;
    }
    v_page_idx = condval_3;
    tl::cp_async_wait<0>();
    __syncthreads();
    if (0 < ((int)v_page_idx)) {
      int k_page_idx_3 = (((int)v_page_idx) - 1);
      #pragma unroll
      for (int i_5 = 0; i_5 < 32; ++i_5) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_5 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_5 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 19456), V_dense_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)v_dense_blocks_len)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)v_dense_blocks_len)) * (int64_t)8192)) + (((int64_t)k_page_idx_3) * (int64_t)8192)) + (((int64_t)i_5) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_3) && (k_page_idx_3 < v_dense_blocks_len)));
      }
    } else {
      int k_page_idx_4 = (((int)(v_page_idx * (short)-1)) - 1);
      #pragma unroll
      for (int i_6 = 0; i_6 < 16; ++i_6) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_6 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_6 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_6 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 19456), V_sparse_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)v_sparse_blocks_len)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)v_sparse_blocks_len)) * (int64_t)4096)) + (((int64_t)k_page_idx_4) * (int64_t)4096)) + (((int64_t)i_6) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_4) && (k_page_idx_4 < v_sparse_blocks_len)));
      }
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_7 * 512) + (((int)threadIdx.x) * 16)) + 27648), V_E_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)v_sparse_blocks_len)) * (int64_t)4096) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)v_sparse_blocks_len)) * (int64_t)512)) + (((int64_t)k_page_idx_4) * (int64_t)512)) + (((int64_t)i_7) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_4) && (k_page_idx_4 < v_sparse_blocks_len)));
      }
    }
    tl::cp_async_commit();
    __syncthreads();
    if (0 < ((int)k_page_idx)) {
      #pragma unroll
      for (int i_8 = 0; i_8 < 8; ++i_8) {
        *(float2*)(acc_s_T + (i_8 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      for (int ki = 0; ki < 4; ++ki) {
        for (int i_9 = 0; i_9 < 4; ++i_9) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((int)threadIdx.x) & 3) >> 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1536)])) + 0, A_local + (i_9 * 16));
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 4096) + (i_9 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + 1) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1536)])) + 0, A_local + ((i_9 * 16) + 8));
        }
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki >> 1) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local + 0);
        for (int i_10 = 0; i_10 < 4; ++i_10) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_10 * 4)), reinterpret_cast<const unsigned*>(A_local + (i_10 * 16)), reinterpret_cast<const unsigned*>(B_local + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_s_T + (i_10 * 4)), reinterpret_cast<const unsigned*>(A_local + ((i_10 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local + 4));
        }
      }
    } else {
      #pragma unroll
      for (int i_11 = 0; i_11 < 8; ++i_11) {
        *(float2*)(acc_s_T + (i_11 * 2)) = make_float2(0x0p+0f/*0.000000e+00*/, 0x0p+0f/*0.000000e+00*/);
      }
      for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
        for (int i_12 = 0; i_12 < 4; ++i_12) {
          tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((i_12 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 1536)])) + 0, A_local_1 + (i_12 * 8));
        }
        for (int i_13 = 0; i_13 < 4; ++i_13) {
          for (int j = 0; j < 2; ++j) {
            E_local[((i_13 * 2) + j)] = ((short*)buf_dyn_shmem)[((((((i_13 * 128) + (j * 64)) + ((((int)threadIdx.x) >> 2) * 8)) + (ki_1 * 2)) + (((int)threadIdx.x) & 1)) + 5632)];
          }
        }
        tl::ptx_ldmatrix_x4((&(((half_t*)buf_dyn_shmem)[((((((ki_1 >> 1) * 512) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))])) + 0, B_local_1 + 0);
        for (int i_14 = 0; i_14 < 4; ++i_14) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_s_T + (i_14 * 4)))[0]), "=f"(((float *)(acc_s_T + (i_14 * 4)))[1]), "=f"(((float *)(acc_s_T + (i_14 * 4)))[2]), "=f"(((float *)(acc_s_T + (i_14 * 4)))[3])
      : "r"(((unsigned *)(A_local_1 + (i_14 * 8)))[0]), "r"(((unsigned *)(A_local_1 + (i_14 * 8)))[1]), "r"(((unsigned *)(A_local_1 + (i_14 * 8)))[2]), "r"(((unsigned *)(A_local_1 + (i_14 * 8)))[3]), "r"(((unsigned *)(B_local_1 + 0))[0]), "r"(((unsigned *)(B_local_1 + 0))[1]), "r"(((unsigned *)(B_local_1 + 0))[2]), "r"(((unsigned *)(B_local_1 + 0))[3]), "f"(((float *)(acc_s_T + (i_14 * 4)))[0]), "f"(((float *)(acc_s_T + (i_14 * 4)))[1]), "f"(((float *)(acc_s_T + (i_14 * 4)))[2]), "f"(((float *)(acc_s_T + (i_14 * 4)))[3]), "r"(((unsigned *)(E_local + (i_14 * 2)))[0]));
  }
        }
      }
    }
    tl::cp_async_wait<0>();
    __syncthreads();
    int condval_4;
    if ((((int)blockIdx.z) == 1)) {
      condval_4 = (((seq_kv + 63) >> 6) - ((seq_kv + 63) >> 7));
    } else {
      condval_4 = ((seq_kv + 63) >> 7);
    }
    if ((k_local + 1) < condval_4) {
      short condval_5;
      if (((-1 <= ((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local)) && ((((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local) + 1) < dummy0))) {
        condval_5 = K_page_idx[((((((((int64_t)((int)blockIdx.x)) * ((int64_t)dummy0)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)dummy0))) + (((((int64_t)seq_kv) + (int64_t)63) >> (int64_t)7) * ((int64_t)((int)blockIdx.z)))) + ((int64_t)k_local)) + (int64_t)1)];
      } else {
        condval_5 = (short)0;
      }
      next_k_page_idx = condval_5;
      if (0 < ((int)next_k_page_idx)) {
        int k_page_idx_5 = (((int)next_k_page_idx) - 1);
        #pragma unroll
        for (int i_15 = 0; i_15 < 32; ++i_15) {
          tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_15 * 256)) + ((((int)threadIdx.x) >> 4) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_15 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_15 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 3072), K_dense_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_dense_blocks_len)) * (int64_t)65536) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_dense_blocks_len)) * (int64_t)8192)) + (((int64_t)k_page_idx_5) * (int64_t)8192)) + (((int64_t)i_15) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_5) && (k_page_idx_5 < k_dense_blocks_len)));
        }
      } else {
        int k_page_idx_6 = (((int)(next_k_page_idx * (short)-1)) - 1);
        #pragma unroll
        for (int i_16 = 0; i_16 < 16; ++i_16) {
          tl::cp_async_gs_conditional<16>(buf_dyn_shmem+((((((i_16 * 512) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_16 & 1)) & 1) * 64)) + ((((((int)threadIdx.x) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 3072), K_sparse_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)32768) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)4096)) + (((int64_t)k_page_idx_6) * (int64_t)4096)) + (((int64_t)i_16) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_6) && (k_page_idx_6 < k_sparse_blocks_len)));
        }
        #pragma unroll
        for (int i_17 = 0; i_17 < 2; ++i_17) {
          tl::cp_async_gs_conditional<16>(buf_dyn_shmem+(((i_17 * 512) + (((int)threadIdx.x) * 16)) + 11264), K_E_blocks+((((((((int64_t)((int)blockIdx.x)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)4096) + ((((int64_t)((int)blockIdx.y)) * ((int64_t)k_sparse_blocks_len)) * (int64_t)512)) + (((int64_t)k_page_idx_6) * (int64_t)512)) + (((int64_t)i_17) * (int64_t)256)) + (((int64_t)((int)threadIdx.x)) * (int64_t)8)), ((0 <= k_page_idx_6) && (k_page_idx_6 < k_sparse_blocks_len)));
        }
      }
      tl::cp_async_commit();
    }
    int condval_6;
    if ((((int)blockIdx.z) == 1)) {
      condval_6 = (((seq_kv + 63) >> 6) - ((seq_kv + 63) >> 7));
    } else {
      condval_6 = ((seq_kv + 63) >> 7);
    }
    if (condval_6 <= (k_local + 2)) {
      #pragma unroll
      for (int i_18 = 0; i_18 < 16; ++i_18) {
        float condval_7;
        if (((((((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) * 64) + (k_local * 64)) + ((i_18 & 7) * 8)) + (((int)threadIdx.x) >> 2)) < seq_kv)) {
          condval_7 = acc_s_T[(((i_18 & 7) * 2) + (i_18 >> 3))];
        } else {
          condval_7 = -CUDART_INF_F;
        }
        acc_s_T[(((i_18 & 7) * 2) + (i_18 >> 3))] = condval_7;
      }
    }
    *(float2*)(scores_max_prev + 0) = *(float2*)(scores_max + 0);
    #pragma unroll
    for (int i_19 = 0; i_19 < 2; ++i_19) {
      scores_max[i_19] = -CUDART_INF_F;
      #pragma unroll
      for (int rv = 0; rv < 8; ++rv) {
        scores_max[i_19] = max(scores_max[i_19], acc_s_T[((rv * 2) + i_19)]);
      }
      scores_max[i_19] = tl::AllReduce<tl::MaxOp, 32, 4, 0>::run(scores_max[i_19]);
    }
    #pragma unroll
    for (int i_20 = 0; i_20 < 2; ++i_20) {
      scores_max[i_20] = max(scores_max[i_20], scores_max_prev[i_20]);
    }
    #pragma unroll
    for (int i_21 = 0; i_21 < 2; ++i_21) {
      scores_scale[i_21] = exp2f(((scores_max_prev[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[i_21] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_22 = 0; i_22 < 16; ++i_22) {
      acc_s_T[(((i_22 & 7) * 2) + (i_22 >> 3))] = exp2f(((acc_s_T[(((i_22 & 7) * 2) + (i_22 >> 3))] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/) - (scores_max[(i_22 >> 3)] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/)));
    }
    #pragma unroll
    for (int i_23 = 0; i_23 < 2; ++i_23) {
      scores_sum[i_23] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
        scores_sum[i_23] = (scores_sum[i_23] + acc_s_T[((rv_1 * 2) + i_23)]);
      }
      scores_sum[i_23] = tl::AllReduce<tl::SumOp, 32, 4, 0>::run(scores_sum[i_23]);
    }
    #pragma unroll
    for (int i_24 = 0; i_24 < 8; ++i_24) {
      uint1 __1;
      float2 v_ = *(float2*)(acc_s_T + (i_24 * 2));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      *(uint1*)(((half_t*)buf_dyn_shmem) + (((i_24 * 64) + (((int)threadIdx.x) * 2)) + 1024)) = __1;
    }
    #pragma unroll
    for (int i_25 = 0; i_25 < 2; ++i_25) {
      logsum[i_25] = ((logsum[i_25] * scores_scale[i_25]) + scores_sum[i_25]);
    }
    #pragma unroll
    for (int i_26 = 0; i_26 < 32; ++i_26) {
      acc_o_T[(((i_26 & 15) * 2) + (i_26 >> 4))] = (acc_o_T[(((i_26 & 15) * 2) + (i_26 >> 4))] * scores_scale[(i_26 >> 4)]);
    }
    short condval_8;
    if (((0 <= ((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local)) && (((((seq_kv + 63) >> 7) * ((int)blockIdx.z)) + k_local) < dummy1))) {
      condval_8 = V_page_idx[(((((((int64_t)((int)blockIdx.x)) * ((int64_t)dummy1)) * (int64_t)8) + (((int64_t)((int)blockIdx.y)) * ((int64_t)dummy1))) + (((((int64_t)seq_kv) + (int64_t)63) >> (int64_t)7) * ((int64_t)((int)blockIdx.z)))) + ((int64_t)k_local))];
    } else {
      condval_8 = (short)0;
    }
    v_page_idx = condval_8;
    __syncthreads();
    if (0 < ((int)v_page_idx)) {
      for (int ki_2 = 0; ki_2 < 2; ++ki_2) {
        for (int i_27 = 0; i_27 < 8; ++i_27) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_27 >> 2) * 4096) + (ki_2 * 2048)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_27 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 9728)])) + 0, A_local_2 + (i_27 * 16));
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_27 >> 2) * 4096) + (ki_2 * 2048)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_27 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_27 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 10752)])) + 0, A_local_2 + ((i_27 * 16) + 8));
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((ki_2 * 256) + (((int)threadIdx.x) * 8)) + 1024)])) + 0, B_local_2 + 0);
        for (int i_28 = 0; i_28 < 8; ++i_28) {
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o_T + (i_28 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_28 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
          tl::mma_sync<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(acc_o_T + (i_28 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + ((i_28 * 16) + 8)), reinterpret_cast<const unsigned*>(B_local_2 + 4));
        }
      }
    } else {
      for (int ki_3 = 0; ki_3 < 2; ++ki_3) {
        for (int i_29 = 0; i_29 < 8; ++i_29) {
          tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((((((((i_29 >> 2) * 2048) + (ki_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((i_29 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_29 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 9728)])) + 0, A_local_3 + (i_29 * 8));
        }
        for (int i_30 = 0; i_30 < 8; ++i_30) {
          for (int j_1 = 0; j_1 < 2; ++j_1) {
            E_local_1[((i_30 * 2) + j_1)] = ((short*)buf_dyn_shmem)[((((((ki_3 * 256) + ((((int)threadIdx.x) & 1) * 128)) + (i_30 * 16)) + (j_1 * 8)) + (((int)threadIdx.x) >> 2)) + 13824)];
          }
        }
        tl::ptx_ldmatrix_x4_trans((&(((half_t*)buf_dyn_shmem)[(((ki_3 * 256) + (((int)threadIdx.x) * 8)) + 1024)])) + 0, B_local_3 + 0);
        for (int i_31 = 0; i_31 < 8; ++i_31) {

  {
    __asm__ __volatile__(
      "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32"
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9, %10, %11}, {%12, %13, %14, %15}, %16, 0;\n"
      :  "=f"(((float *)(acc_o_T + (i_31 * 4)))[0]), "=f"(((float *)(acc_o_T + (i_31 * 4)))[1]), "=f"(((float *)(acc_o_T + (i_31 * 4)))[2]), "=f"(((float *)(acc_o_T + (i_31 * 4)))[3])
      : "r"(((unsigned *)(A_local_3 + (i_31 * 8)))[0]), "r"(((unsigned *)(A_local_3 + (i_31 * 8)))[1]), "r"(((unsigned *)(A_local_3 + (i_31 * 8)))[2]), "r"(((unsigned *)(A_local_3 + (i_31 * 8)))[3]), "r"(((unsigned *)(B_local_3 + 0))[0]), "r"(((unsigned *)(B_local_3 + 0))[1]), "r"(((unsigned *)(B_local_3 + 0))[2]), "r"(((unsigned *)(B_local_3 + 0))[3]), "f"(((float *)(acc_o_T + (i_31 * 4)))[0]), "f"(((float *)(acc_o_T + (i_31 * 4)))[1]), "f"(((float *)(acc_o_T + (i_31 * 4)))[2]), "f"(((float *)(acc_o_T + (i_31 * 4)))[3]), "r"(((unsigned *)(E_local_1 + (i_31 * 2)))[0]));
  }
        }
      }
    }
  }
  #pragma unroll
  for (int i_32 = 0; i_32 < 32; ++i_32) {
    acc_o_T[(((i_32 & 15) * 2) + (i_32 >> 4))] = (acc_o_T[(((i_32 & 15) * 2) + (i_32 >> 4))] / logsum[(i_32 >> 4)]);
  }
  #pragma unroll
  for (int i_33 = 0; i_33 < 2; ++i_33) {
    logsum[i_33] = (log2f(logsum[i_33]) + (scores_max[i_33] * 0x1.0527dbd5cafffp-3f/*1.275174e-01*/));
  }
  if ((((int)threadIdx.x) >> 2) == 0) {
    #pragma unroll
    for (int i_34 = 0; i_34 < 2; ++i_34) {
      if ((((int)threadIdx.x) & 3) < 2) {
        glse[(((((((int)blockIdx.x) * 64) + (((int)blockIdx.y) * 8)) + ((((int)threadIdx.x) & 3) * 4)) + (i_34 * 2)) + ((int)blockIdx.z))] = ((half_t)logsum[i_34]);
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_35 = 0; i_35 < 32; ++i_35) {
    if ((((int)threadIdx.x) & 3) < 2) {
      ((half_t*)buf_dyn_shmem)[(((((((((i_35 & 15) >> 3) * 512) + ((((int)threadIdx.x) & 3) * 128)) + ((i_35 >> 4) * 64)) + (((i_35 & 7) >> 2) * 32)) + (((((i_35 & 3) >> 1) + (((int)threadIdx.x) & 1)) & 1) * 16)) + ((((i_35 >> 4) + (i_35 & 1)) & 1) * 8)) + (((int)threadIdx.x) >> 2))] = ((half_t)acc_o_T[(((i_35 & 15) * 2) + (i_35 >> 4))]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 2; ++i_36) {
    *(uint4*)(Output_partial + ((((((((int)blockIdx.x) * 8192) + (((int)blockIdx.y) * 1024)) + (i_36 * 512)) + ((((int)threadIdx.x) >> 4) * 256)) + (((int)blockIdx.z) * 128)) + ((((int)threadIdx.x) & 15) * 8))) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 512) + (i_36 * 128)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((int)threadIdx.x) & 7) >> 2) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_36) & 1) * 16)) + ((((((int)threadIdx.x) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)));
  }
}

extern "C" __global__ void __launch_bounds__(128, 1) blockdecode_sp_mk_mv_split_kernel_kernel_1(half_t* __restrict__ Output, const half_t* __restrict__ Output_partial, const half_t* __restrict__ glse, half_t* __restrict__ lse_combined) {
  float lse_logsum_local[1];
  float o_accum_local[1];
  float lse_max_local[1];
  float lse_local_split[1];
  half_t po_local[1];
  float scale_local[1];
  lse_logsum_local[0] = 0x0p+0f/*0.000000e+00*/;
  o_accum_local[0] = 0x0p+0f/*0.000000e+00*/;
  lse_max_local[0] = -CUDART_INF_F;
  for (int k = 0; k < 2; ++k) {
    lse_max_local[0] = max(lse_max_local[0], ((float)glse[(((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + k)]));
  }
  lse_local_split[0] = ((float)glse[((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2))]);
  lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
  lse_local_split[0] = ((float)glse[(((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + 1)]);
  lse_logsum_local[0] = (lse_logsum_local[0] + exp2f((lse_local_split[0] - lse_max_local[0])));
  lse_logsum_local[0] = (log2f(lse_logsum_local[0]) + lse_max_local[0]);
  for (int k_1 = 0; k_1 < 2; ++k_1) {
    po_local[0] = Output_partial[((((((int)blockIdx.y) * 8192) + (((int)blockIdx.x) * 256)) + (k_1 * 128)) + ((int)threadIdx.x))];
    lse_local_split[0] = ((float)glse[(((((int)blockIdx.y) * 64) + (((int)blockIdx.x) * 2)) + k_1)]);
    scale_local[0] = exp2f((lse_local_split[0] - lse_logsum_local[0]));
    o_accum_local[0] = (o_accum_local[0] + (((float)po_local[0]) * scale_local[0]));
  }
  Output[(((((int)blockIdx.y) * 4096) + (((int)blockIdx.x) * 128)) + ((int)threadIdx.x))] = ((half_t)o_accum_local[0]);
  lse_combined[((((int)blockIdx.y) * 32) + ((int)blockIdx.x))] = ((half_t)(lse_logsum_local[0] * 0x1.62e42ff34ed5cp-1f/*6.931472e-01*/));
}

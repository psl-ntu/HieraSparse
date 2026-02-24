#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tl_topk_multipass_kernel_kernel(const half_t* __restrict__ logits, int* __restrict__ topk_indices, int M);
extern "C" __global__ void __launch_bounds__(128, 1) tl_topk_multipass_kernel_kernel(const half_t* __restrict__ logits, int* __restrict__ topk_indices, int M) {
  half_t logits_frag[2];
  int expand_max_idx[2];
  half_t max_val[1];
  int max_idx[1];
  uint1 condval;
  if ((((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < M)) {
    condval = *(uint1*)(logits + ((((int64_t)((int)blockIdx.x)) * (int64_t)256) + (((int64_t)((int)threadIdx.x)) * (int64_t)2)));
  } else {
    condval = make_uint1(__pack_half2(half_t(0x0p+0f/*0.000000e+00*/), half_t(0x0p+0f/*0.000000e+00*/)));
  }
  *(uint1*)(logits_frag + 0) = condval;
  for (int k = 0; k < 2; ++k) {
    *(int2*)(expand_max_idx + 0) = make_int2(2147483647, 2147483647);
    max_val[0] = -std::numeric_limits<half_t>::infinity();
    #pragma unroll
    for (int rv = 0; rv < 2; ++rv) {
      max_val[0] = cutlass::fast_max(max_val[0], logits_frag[rv]);
    }
    max_val[0] = tl::AllReduce<tl::MaxOp, 2, 1, 0>::run(max_val[0]);
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      int condval_1;
      if ((max_val[0] == logits_frag[i])) {
        condval_1 = (((((int)threadIdx.x) & 1) * 2) + i);
      } else {
        condval_1 = expand_max_idx[i];
      }
      expand_max_idx[i] = condval_1;
    }
    max_idx[0] = 2147483647;
    #pragma unroll
    for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
      max_idx[0] = min(max_idx[0], expand_max_idx[rv_1]);
    }
    max_idx[0] = tl::AllReduce<tl::MinOp, 2, 1, 0>::run(max_idx[0]);
    #pragma unroll
    for (int i_1 = 0; i_1 < 2; ++i_1) {
      half_t condval_2;
      if ((max_idx[0] == (((((int)threadIdx.x) & 1) * 2) + i_1))) {
        condval_2 = (std::numeric_limits<half_t>::infinity() * half_t(-0x1p+0f/*-1.000000e+00*/));
      } else {
        condval_2 = logits_frag[i_1];
      }
      logits_frag[i_1] = condval_2;
    }
    if ((((int)threadIdx.x) % 2) == 0) {
      if (((((int)blockIdx.x) * 64) + (((int)threadIdx.x) >> 1)) < M) {
        topk_indices[(((((int64_t)((int)blockIdx.x)) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)1) * (int64_t)2)) + ((int64_t)k))] = max_idx[0];
      }
    }
  }
}

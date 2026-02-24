// tilelang target: c -keys=cpu
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_error_ndim_mismatch_packed = NULL;
static void* __tvm_error_dtype_mismatch_packed = NULL;
static void* __tvm_error_expect_eq_packed = NULL;
static void* __tvm_error_byte_offset_mismatch_packed = NULL;
static void* __tvm_error_device_type_mismatch_packed = NULL;
static void* __tvm_error_null_ptr_packed = NULL;
static void* __tvm_set_device_packed = NULL;
static void* prune_block_value_mask_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t prune_block_value_mask_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t prune_block_value_mask_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TVMFFIAny stack[11];
  void* stack_ffi_any = stack;
  if (!((num_args == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "prune_block_value_mask_kernel: num_args should be 2", (long long)(num_args), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "prune_block_value_mask_kernel: args pointer is NULL");
    return -1;
  }
  int32_t V_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((V_handle_type_index == 0) || (V_handle_type_index == 4)) || (V_handle_type_index == 7)) || (64 <= V_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel prune_block_value_mask_kernel input V expected pointer or tensor handle");
    return -1;
  }
  int32_t prune_loss_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((prune_loss_handle_type_index == 0) || (prune_loss_handle_type_index == 4)) || (prune_loss_handle_type_index == 7)) || (64 <= prune_loss_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel prune_block_value_mask_kernel input prune_loss expected pointer or tensor handle");
    return -1;
  }
  void* V_handle = ((V_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* prune_loss_handle = ((prune_loss_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  bool prune_block_value_mask_kernel_V_is_null = (V_handle == NULL);
  if (!(!prune_block_value_mask_kernel_V_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "prune_block_value_mask_kernel.V is expected to have non-NULL pointer");
    return -1;
  }
  bool prune_block_value_mask_kernel_prune_loss_is_null = (prune_loss_handle == NULL);
  if (!(!prune_block_value_mask_kernel_prune_loss_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "prune_block_value_mask_kernel.prune_loss is expected to have non-NULL pointer");
    return -1;
  }
  void* prune_block_value_mask_kernel_V_shape = (((DLTensor*)V_handle)[0].shape);
  void* prune_block_value_mask_kernel_prune_loss_shape = (((DLTensor*)prune_loss_handle)[0].shape);
  if ((((DLTensor*)V_handle)[0].ndim) != 5) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)5;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_1;
    result_1.type_index = kTVMFFINone;
    result_1.zero_padding = 0;
    result_1.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_1) != 0) {
      return -1;
    }
  }
  if (!((bool)1)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable num_blocks requires at least one non-null buffer among: prune_block_value_mask_kernel.V, prune_block_value_mask_kernel.prune_loss");
    return -1;
  }
  int32_t num_blocks = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[2]);
  void* prune_block_value_mask_kernel_V_strides = (((DLTensor*)V_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)V_handle)[0].device.device_id);
  void* V = (((DLTensor*)V_handle)[0].data);
  if ((((DLTensor*)prune_loss_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_2;
    result_2.type_index = kTVMFFINone;
    result_2.zero_padding = 0;
    result_2.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_2) != 0) {
      return -1;
    }
  }
  void* prune_block_value_mask_kernel_prune_loss_strides = (((DLTensor*)prune_loss_handle)[0].strides);
  void* prune_loss = (((DLTensor*)prune_loss_handle)[0].data);
  if ((((((DLTensor*)V_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)V_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)V_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_3;
    result_3.type_index = kTVMFFINone;
    result_3.zero_padding = 0;
    result_3.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_3) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[0]) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_4;
    result_4.type_index = kTVMFFINone;
    result_4.zero_padding = 0;
    result_4.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_4) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[1]) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_5;
    result_5.type_index = kTVMFFINone;
    result_5.zero_padding = 0;
    result_5.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_5) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[3]) != 64) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[3]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)64;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[3]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_6;
    result_6.type_index = kTVMFFINone;
    result_6.zero_padding = 0;
    result_6.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_6) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[4]) != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[4]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[4]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_7;
    result_7.type_index = kTVMFFINone;
    result_7.zero_padding = 0;
    result_7.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_7) != 0) {
      return -1;
    }
  }
  int32_t condval;
  if ((prune_block_value_mask_kernel_V_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[4]);
  }
  if (condval != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[4]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_1;
    if ((prune_block_value_mask_kernel_V_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[4]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_1);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_8;
    result_8.type_index = kTVMFFINone;
    result_8.zero_padding = 0;
    result_8.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_8) != 0) {
      return -1;
    }
  }
  int32_t condval_2;
  if ((prune_block_value_mask_kernel_V_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[3]);
  }
  if (condval_2 != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[3]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_3;
    if ((prune_block_value_mask_kernel_V_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[3]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_3);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_9;
    result_9.type_index = kTVMFFINone;
    result_9.zero_padding = 0;
    result_9.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_9) != 0) {
      return -1;
    }
  }
  int32_t condval_4;
  if ((prune_block_value_mask_kernel_V_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[2]);
  }
  if (condval_4 != 8192) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8192;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_5;
    if ((prune_block_value_mask_kernel_V_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_5);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_10;
    result_10.type_index = kTVMFFINone;
    result_10.zero_padding = 0;
    result_10.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_10) != 0) {
      return -1;
    }
  }
  int32_t condval_6;
  if ((prune_block_value_mask_kernel_V_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[1]);
  }
  if (condval_6 != (num_blocks * 8192)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (((int64_t)num_blocks) * (int64_t)8192);
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_7;
    if ((prune_block_value_mask_kernel_V_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_7);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_11;
    result_11.type_index = kTVMFFINone;
    result_11.zero_padding = 0;
    result_11.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_11) != 0) {
      return -1;
    }
  }
  int32_t condval_8;
  if ((prune_block_value_mask_kernel_V_strides == NULL)) {
    condval_8 = 1;
  } else {
    condval_8 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[0]);
  }
  if (condval_8 != (num_blocks * 65536)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (((int64_t)num_blocks) * (int64_t)65536);
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_9;
    if ((prune_block_value_mask_kernel_V_strides == NULL)) {
      condval_9 = 1;
    } else {
      condval_9 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_9);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_12;
    result_12.type_index = kTVMFFINone;
    result_12.zero_padding = 0;
    result_12.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_12) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)V_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_13;
    result_13.type_index = kTVMFFINone;
    result_13.zero_padding = 0;
    result_13.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_13) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)V_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_14;
    result_14.type_index = kTVMFFINone;
    result_14.zero_padding = 0;
    result_14.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_14) != 0) {
      return -1;
    }
  }
  if (num_blocks != 0) {
    if (V == NULL) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
      if (__tvm_error_null_ptr_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_15;
      result_15.type_index = kTVMFFINone;
      result_15.zero_padding = 0;
      result_15.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_15) != 0) {
        return -1;
      }
    }
  } else {
  }
  if ((((((DLTensor*)prune_loss_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)prune_loss_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)prune_loss_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_16;
    result_16.type_index = kTVMFFINone;
    result_16.zero_padding = 0;
    result_16.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_16) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[0]) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_17;
    result_17.type_index = kTVMFFINone;
    result_17.zero_padding = 0;
    result_17.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_17) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[1]) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_18;
    result_18.type_index = kTVMFFINone;
    result_18.zero_padding = 0;
    result_18.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_18) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[2]) != ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[2])) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_19;
    result_19.type_index = kTVMFFINone;
    result_19.zero_padding = 0;
    result_19.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_19) != 0) {
      return -1;
    }
  }
  int32_t condval_10;
  if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
    condval_10 = 1;
  } else {
    condval_10 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[2]);
  }
  if (condval_10 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_11;
    if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
      condval_11 = 1;
    } else {
      condval_11 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_11);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_20;
    result_20.type_index = kTVMFFINone;
    result_20.zero_padding = 0;
    result_20.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_20) != 0) {
      return -1;
    }
  }
  int32_t condval_12;
  if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
    condval_12 = 1;
  } else {
    condval_12 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[1]);
  }
  if (condval_12 != ((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[2])) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)((int32_t)((int64_t*)prune_block_value_mask_kernel_V_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_13;
    if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
      condval_13 = 1;
    } else {
      condval_13 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_13);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_21;
    result_21.type_index = kTVMFFINone;
    result_21.zero_padding = 0;
    result_21.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_21) != 0) {
      return -1;
    }
  }
  int32_t condval_14;
  if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
    condval_14 = 1;
  } else {
    condval_14 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[0]);
  }
  if (condval_14 != (num_blocks * 8)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (((int64_t)num_blocks) * (int64_t)8);
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_15;
    if ((prune_block_value_mask_kernel_prune_loss_strides == NULL)) {
      condval_15 = 1;
    } else {
      condval_15 = ((int32_t)((int64_t*)prune_block_value_mask_kernel_prune_loss_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_15);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_22;
    result_22.type_index = kTVMFFINone;
    result_22.zero_padding = 0;
    result_22.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_22) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)prune_loss_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_23;
    result_23.type_index = kTVMFFINone;
    result_23.zero_padding = 0;
    result_23.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_23) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)prune_loss_handle)[0].device.device_id) != (((DLTensor*)V_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_24;
    result_24.type_index = kTVMFFINone;
    result_24.zero_padding = 0;
    result_24.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_24) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)prune_loss_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)prune_loss_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_25;
    result_25.type_index = kTVMFFINone;
    result_25.zero_padding = 0;
    result_25.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_25) != 0) {
      return -1;
    }
  }
  if (num_blocks != 0) {
    if (prune_loss == NULL) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"prune_block_value_mask_kernel";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"prune_loss";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
      if (__tvm_error_null_ptr_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_26;
      result_26.type_index = kTVMFFINone;
      result_26.zero_padding = 0;
      result_26.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_26) != 0) {
        return -1;
      }
    }
  } else {
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_27;
  result_27.type_index = kTVMFFINone;
  result_27.zero_padding = 0;
  result_27.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_27) != 0) {
    return -1;
  }
  if (V == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = V;
  if (prune_loss == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = prune_loss;
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)num_blocks);
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)8);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)8);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)num_blocks);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = ((int64_t)4);
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = (int64_t)0;
  if (prune_block_value_mask_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "prune_block_value_mask_kernel_kernel", &prune_block_value_mask_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_28;
  result_28.type_index = kTVMFFINone;
  result_28.zero_padding = 0;
  result_28.v_int64 = 0;
  if (TVMFFIFunctionCall(prune_block_value_mask_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 10, &result_28) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return prune_block_value_mask_kernel(self, args, num_args, result);
}

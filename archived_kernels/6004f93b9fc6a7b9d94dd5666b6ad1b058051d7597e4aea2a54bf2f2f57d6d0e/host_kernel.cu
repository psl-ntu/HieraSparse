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
static void* tl_topk_multipass_kernel_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t tl_topk_multipass_kernel(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t tl_topk_multipass_kernel(void* self_handle, void* args, int32_t num_args, void* result) {
  TVMFFIAny stack[9];
  void* stack_ffi_any = stack;
  if (!((num_args == 2))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "tl_topk_multipass_kernel: num_args should be 2", (long long)(num_args), (long long)(2));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tl_topk_multipass_kernel: args pointer is NULL");
    return -1;
  }
  int32_t logits_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((logits_handle_type_index == 0) || (logits_handle_type_index == 4)) || (logits_handle_type_index == 7)) || (64 <= logits_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tl_topk_multipass_kernel input logits expected pointer or tensor handle");
    return -1;
  }
  int32_t topk_indices_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((topk_indices_handle_type_index == 0) || (topk_indices_handle_type_index == 4)) || (topk_indices_handle_type_index == 7)) || (64 <= topk_indices_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel tl_topk_multipass_kernel input topk_indices expected pointer or tensor handle");
    return -1;
  }
  void* logits_handle = ((logits_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* topk_indices_handle = ((topk_indices_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  bool tl_topk_multipass_kernel_logits_is_null = (logits_handle == NULL);
  if (!(!tl_topk_multipass_kernel_logits_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tl_topk_multipass_kernel.logits is expected to have non-NULL pointer");
    return -1;
  }
  bool tl_topk_multipass_kernel_topk_indices_is_null = (topk_indices_handle == NULL);
  if (!(!tl_topk_multipass_kernel_topk_indices_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "tl_topk_multipass_kernel.topk_indices is expected to have non-NULL pointer");
    return -1;
  }
  void* tl_topk_multipass_kernel_logits_shape = (((DLTensor*)logits_handle)[0].shape);
  void* tl_topk_multipass_kernel_topk_indices_shape = (((DLTensor*)topk_indices_handle)[0].shape);
  if ((((DLTensor*)logits_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].ndim));
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
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "Symbolic shape variable M requires at least one non-null buffer among: tl_topk_multipass_kernel.logits, tl_topk_multipass_kernel.topk_indices");
    return -1;
  }
  int32_t M = ((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_shape)[0]);
  void* tl_topk_multipass_kernel_logits_strides = (((DLTensor*)logits_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)logits_handle)[0].device.device_id);
  void* logits = (((DLTensor*)logits_handle)[0].data);
  if ((((DLTensor*)topk_indices_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].ndim));
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
  void* tl_topk_multipass_kernel_topk_indices_strides = (((DLTensor*)topk_indices_handle)[0].strides);
  void* topk_indices = (((DLTensor*)topk_indices_handle)[0].data);
  if ((((((DLTensor*)logits_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)logits_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)logits_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].dtype.lanes));
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
  if (((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_shape)[1]) != 4) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_shape)[1]));
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
  int32_t condval;
  if ((tl_topk_multipass_kernel_logits_strides == NULL)) {
    condval = 1;
  } else {
    condval = ((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_strides)[1]);
  }
  if (condval != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_1;
    if ((tl_topk_multipass_kernel_logits_strides == NULL)) {
      condval_1 = 1;
    } else {
      condval_1 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_strides)[1]);
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
    TVMFFIAny result_5;
    result_5.type_index = kTVMFFINone;
    result_5.zero_padding = 0;
    result_5.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_5) != 0) {
      return -1;
    }
  }
  int32_t condval_2;
  if ((tl_topk_multipass_kernel_logits_strides == NULL)) {
    condval_2 = 1;
  } else {
    condval_2 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_strides)[0]);
  }
  if (condval_2 != 4) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_3;
    if ((tl_topk_multipass_kernel_logits_strides == NULL)) {
      condval_3 = 1;
    } else {
      condval_3 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_strides)[0]);
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
    TVMFFIAny result_6;
    result_6.type_index = kTVMFFINone;
    result_6.zero_padding = 0;
    result_6.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_6) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)logits_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_7;
    result_7.type_index = kTVMFFINone;
    result_7.zero_padding = 0;
    result_7.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_7) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)logits_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_8;
    result_8.type_index = kTVMFFINone;
    result_8.zero_padding = 0;
    result_8.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_8) != 0) {
      return -1;
    }
  }
  if (M != 0) {
    if (logits == NULL) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"logits";
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
      TVMFFIAny result_9;
      result_9.type_index = kTVMFFINone;
      result_9.zero_padding = 0;
      result_9.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_9) != 0) {
        return -1;
      }
    }
  } else {
  }
  if ((((((DLTensor*)topk_indices_handle)[0].dtype.code) != (uint8_t)0) || ((((DLTensor*)topk_indices_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)topk_indices_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
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
    TVMFFIAny result_10;
    result_10.type_index = kTVMFFINone;
    result_10.zero_padding = 0;
    result_10.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_10) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_shape)[0]) != ((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_shape)[0])) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)tl_topk_multipass_kernel_logits_shape)[0]));
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
  if (((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_shape)[1]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_shape)[1]));
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
  int32_t condval_4;
  if ((tl_topk_multipass_kernel_topk_indices_strides == NULL)) {
    condval_4 = 1;
  } else {
    condval_4 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_strides)[1]);
  }
  if (condval_4 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_5;
    if ((tl_topk_multipass_kernel_topk_indices_strides == NULL)) {
      condval_5 = 1;
    } else {
      condval_5 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_strides)[1]);
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
    TVMFFIAny result_13;
    result_13.type_index = kTVMFFINone;
    result_13.zero_padding = 0;
    result_13.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_13) != 0) {
      return -1;
    }
  }
  int32_t condval_6;
  if ((tl_topk_multipass_kernel_topk_indices_strides == NULL)) {
    condval_6 = 1;
  } else {
    condval_6 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_strides)[0]);
  }
  if (condval_6 != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_7;
    if ((tl_topk_multipass_kernel_topk_indices_strides == NULL)) {
      condval_7 = 1;
    } else {
      condval_7 = ((int32_t)((int64_t*)tl_topk_multipass_kernel_topk_indices_strides)[0]);
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
    TVMFFIAny result_14;
    result_14.type_index = kTVMFFINone;
    result_14.zero_padding = 0;
    result_14.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_14) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)topk_indices_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_15;
    result_15.type_index = kTVMFFINone;
    result_15.zero_padding = 0;
    result_15.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_15) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)topk_indices_handle)[0].device.device_id) != (((DLTensor*)logits_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)logits_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_16;
    result_16.type_index = kTVMFFINone;
    result_16.zero_padding = 0;
    result_16.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_16) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)topk_indices_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)topk_indices_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_17;
    result_17.type_index = kTVMFFINone;
    result_17.zero_padding = 0;
    result_17.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_17) != 0) {
      return -1;
    }
  }
  if (M != 0) {
    if (topk_indices == NULL) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"tl_topk_multipass_kernel";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"topk_indices";
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
      TVMFFIAny result_18;
      result_18.type_index = kTVMFFINone;
      result_18.zero_padding = 0;
      result_18.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_18) != 0) {
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
  TVMFFIAny result_19;
  result_19.type_index = kTVMFFINone;
  result_19.zero_padding = 0;
  result_19.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_19) != 0) {
    return -1;
  }
  if (logits == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = logits;
  if (topk_indices == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = topk_indices;
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)M);
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)((M + 63) >> 6));
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)128);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)0;
  if (tl_topk_multipass_kernel_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "tl_topk_multipass_kernel_kernel", &tl_topk_multipass_kernel_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_20;
  result_20.type_index = kTVMFFINone;
  result_20.zero_padding = 0;
  result_20.v_int64 = 0;
  if (TVMFFIFunctionCall(tl_topk_multipass_kernel_kernel_packed, (TVMFFIAny*) stack_ffi_any, 7, &result_20) != 0) {
    return -1;
  }
  return 0;
}

// CodegenC: NOTE: Auto-generated entry function
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self, void* args,int num_args, void* result) {
  return tl_topk_multipass_kernel(self, args, num_args, result);
}

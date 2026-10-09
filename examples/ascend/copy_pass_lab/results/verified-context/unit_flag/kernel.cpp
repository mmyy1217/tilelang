#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/gemm.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* a, __gm__ bfloat16_t* b, __gm__ float* dst) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *acc = (__cc__ float *)0;
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(a[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 32, 64, 0, 0);
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), (&(b[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 32, 64, 0, 0);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  ascend_gemm_l1<32, 64, 32, 64, true, bfloat16_t>(((__cc__ float*)acc), ((__cbuf__ bfloat16_t*)buf_dyn_l1), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), (bool)1, 0, 3);
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_copy_l0c2gm((&(dst[0])), ((__cc__ float*)acc), 32, 32, 32, 32, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(3), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
}


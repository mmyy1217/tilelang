#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ fp8_e4_t* a, __gm__ fp8_e4_t* b, __gm__ float* dst, __gm__ uint16_t* sa, __gm__ uint16_t* sb) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __ca__ fp8_e4_t *a0 = (__ca__ fp8_e4_t *)0;
  __cb__ fp8_e4_t *b0 = (__cb__ fp8_e4_t *)0;
  __cc__ float *acc = (__cc__ float *)0;
  asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
  asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
  asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(16), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ fp8_e4_t*)buf_dyn_l1), (&(a[0])), 256, static_cast<asc_load_l2_cache_mode>(0), 16, 256, 0, 0);
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(16), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ fp8_e4_t*)buf_dyn_l1 + 4096), (&(b[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 16, 128, 0, 0);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(2), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ uint16_t*)buf_dyn_l1 + 3072), (&(sa[0])), 32, static_cast<asc_load_l2_cache_mode>(0), 2, 16, 0, 0);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(2), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ uint16_t*)buf_dyn_l1 + 3328), (&(sb[0])), 32, static_cast<asc_load_l2_cache_mode>(0), 2, 16, 0, 0);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(2));
  asc_copy_l12l0b(((__cb__ fp8_e4_t*)b0), ((__cbuf__ fp8_e4_t*)buf_dyn_l1 + 4096), 0, 0, 1, 4, 1, 1);
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
  asc_copy_l12l0a_mx((uint64_t)(uintptr_t)(((__ca__ fp8_e4_t*)a0)) / 16, (__cbuf__ fp8_e8m0_t*)(((__cbuf__ uint16_t*)buf_dyn_l1 + 3072)), 0, 0, 1, 2, 2, 2);
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_copy_l12l0b_mx((uint64_t)(uintptr_t)(((__cb__ fp8_e4_t*)b0)) / 16, (__cbuf__ fp8_e8m0_t*)(((__cbuf__ uint16_t*)buf_dyn_l1 + 3328)), 0, 0, 1, 2, 2, 2);
  for (int32_t i = 0; i < 2; ++i) {
    asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_copy_l12l0a(((__ca__ fp8_e4_t*)a0), ((__cbuf__ fp8_e4_t*)buf_dyn_l1), 0, (i * 4), 1, 4, 1, 1);
    asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
    asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(i));
    asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
    asc_mmad_mx(((__cc__ float*)acc + (i * 256)), ((__ca__ fp8_e4_t*)a0), ((__cb__ fp8_e4_t*)b0), 16, 128, 16, 0, 1, 0, (bool)1);
    asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(i));
    asc_sync_notify(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(i));
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2gm((&(dst[(i * 256)])), ((__cc__ float*)acc + (i * 256)), 16, 16, 16, 16, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_sync_notify(PIPE_FIX, PIPE_M, static_cast<event_t>(i));
  }
  asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(0));
  asc_sync_wait(PIPE_FIX, PIPE_M, static_cast<event_t>(1));
  asc_sync_wait(PIPE_M, PIPE_MTE1, static_cast<event_t>(0));
}


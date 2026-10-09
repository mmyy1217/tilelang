#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* src) {
  asc_init();
  __ubuf__ float *ub = (__ubuf__ float *)0;
  __cbuf__ float *l1 = (__cbuf__ float *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(ub[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, 8192, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 8192, 8192);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ float*)l1)), (__ubuf__ void*)((&(ub[0]))), 256, 1, 0, 0);
}


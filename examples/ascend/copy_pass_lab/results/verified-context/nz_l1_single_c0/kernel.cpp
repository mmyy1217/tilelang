#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* src) {
  asc_init();
  __ubuf__ bfloat16_t *ub = (__ubuf__ bfloat16_t *)0;
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(ub[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, 1024, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 1024, 1024);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)l1)), (__ubuf__ void*)((&(ub[0]))), 1, 32, 1, 0);
}


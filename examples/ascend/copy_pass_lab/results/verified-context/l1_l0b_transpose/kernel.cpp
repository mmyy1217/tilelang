#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* src) {
  asc_init();
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  __cb__ bfloat16_t *l0 = (__cb__ bfloat16_t *)0;
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(64), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)l1), (&(src[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 64, 64, 0, 0);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_copy_l12l0b_transpose(((__cb__ bfloat16_t*)l0), ((__cbuf__ bfloat16_t*)l1), 0, 0, 2, 2, 4, 2);
}


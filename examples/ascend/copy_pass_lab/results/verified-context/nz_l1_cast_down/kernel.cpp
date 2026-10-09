#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  ascend_nd2nz_scatter<32, 128, float, bfloat16_t>((__ubuf__ float*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[8192]))));
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)l1)), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[8192]))), 8, 32, 1, 0);
}


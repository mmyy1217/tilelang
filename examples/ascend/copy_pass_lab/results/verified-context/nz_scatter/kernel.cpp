#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* dst, __gm__ bfloat16_t* src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[2112]))), (__gm__ uint8_t*)((&(src[0]))), 1, 4096, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 4096, 4096);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  ascend_nd2nz_scatter<32, 64, bfloat16_t, bfloat16_t>((__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[2112]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[0]))));
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[2112]))), 1, 4096, static_cast<asc_store_l2_cache_mode>(4), 4096, 4096);
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2l1((__cbuf__ void*)(((__cbuf__ bfloat16_t*)l1)), (__ubuf__ void*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[0]))), 4, 32, 1, 0);
}


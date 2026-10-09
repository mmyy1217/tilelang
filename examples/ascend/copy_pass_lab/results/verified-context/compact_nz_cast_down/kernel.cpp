#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/nd2nz_copy.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* dst, __gm__ float* src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, 16384, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 16384, 16384);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  ascend_nd2nz_scatter<32, 128, float, bfloat16_t>((__ubuf__ float*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__ubuf__ bfloat16_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[8192]))));
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[8192]))), 1, 8448, static_cast<asc_store_l2_cache_mode>(4), 8448, 8448);
}


#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* dst, __gm__ float* src, int32_t n) {
  asc_init();
  __ubuf__ float *ub = (__ubuf__ float *)0;
  if (0 < n) {
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(ub[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, (n * 4), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (n * 4), (n * 4));
  }
  asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  if (0 < n) {
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(ub[0]))), 1, (n * 4), static_cast<asc_store_l2_cache_mode>(4), (n * 4), (n * 4));
  }
}


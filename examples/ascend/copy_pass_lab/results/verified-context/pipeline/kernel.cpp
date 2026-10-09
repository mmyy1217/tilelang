#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* dst, __gm__ float* src) {
  asc_init();
  __ubuf__ float *ub = (__ubuf__ float *)0;
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(1));
  for (int32_t step = 0; step < 4; ++step) {
    asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>((step & 1)));
    asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(ub[((step & 1) * 256)]))), (__gm__ uint8_t*)((&(src[(step * 256)]))), 1, 1024, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 1024, 1024);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>((step & 1)));
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[(step * 256)]))), (__ubuf__ uint8_t*)((&(ub[((step & 1) * 256)]))), 1, 1024, static_cast<asc_store_l2_cache_mode>(4), 1024, 1024);
    asc_sync_notify(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>((step & 1)));
  }
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(1));
}


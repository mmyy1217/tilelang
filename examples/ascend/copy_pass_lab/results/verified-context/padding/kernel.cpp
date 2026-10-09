#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* dst, __gm__ float* src) {
  asc_init();
  __ubuf__ float *ub = (__ubuf__ float *)0;
  float __asc_pad_val = float(-0x1p+0f/*-1.000000e+00*/);
  asc_set_copy_pad_val(*reinterpret_cast<uint32_t*>(&__asc_pad_val));
  asc_copy_gm2ub_align((__ubuf__ uint32_t*)((&(ub[0]))), (__gm__ uint32_t*)((&(src[0]))), 4, 120, 0, 2, 1, static_cast<asc_load_l2_cache_mode>(0), 120, 128);
  asc_sync_notify(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(ub[0]))), 1, 512, static_cast<asc_store_l2_cache_mode>(4), 512, 512);
}


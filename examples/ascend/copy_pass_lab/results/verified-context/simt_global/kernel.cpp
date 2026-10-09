#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void main_kernel_simt_vf_0(__ubuf__ float* ub, __gm__ float* src) {
  *(__ubuf__ float2*)(ub + (((int32_t)threadIdx.x) * 2)) = *(__gm__ float2*)(src + (((int32_t)threadIdx.x) * 2));
}

extern "C" __global__ __vector__ void main_kernel(__gm__ float* dst, __gm__ float* src) {
  asc_init();
  __ubuf__ float *ub = (__ubuf__ float *)0;
  asc_vf_call<main_kernel_simt_vf_0>(cce::dim3(128), ub, src);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(ub[0]))), 1, 1024, static_cast<asc_store_l2_cache_mode>(4), 1024, 1024);
}


#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void main_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  vector_bool mask = asc_create_mask_b16(PAT_ALL);
  for (int32_t vchunk = 0; vchunk < 4; ++vchunk) {
    auto vreg_0 = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(vchunk * 64)])), 0);
    auto vreg_1 = simd_inst::vcvt<bfloat16_t>(vreg_0, mask, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
    simd_inst::vsts_norm(vreg_1, (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((vchunk * 64) + 512)])), 0, mask);
  }
}

extern "C" __global__ __vector__ void main_kernel(__gm__ bfloat16_t* dst, __gm__ float* src) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[0]))), (__gm__ uint8_t*)((&(src[0]))), 1, 1024, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 1024, 1024);
  asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(0));
  main_kernel_simd_vf_0(buf_dyn_shmem);
  asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
  asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[512]))), 1, 512, static_cast<asc_store_l2_cache_mode>(4), 512, 512);
}


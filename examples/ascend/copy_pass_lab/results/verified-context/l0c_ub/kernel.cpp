#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/gemm.h>
extern "C" __global__ __mix__(1, 2) void main_kernel(__gm__ bfloat16_t* a, __gm__ bfloat16_t* b, __gm__ float* dst) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __cc__ float *acc = (__cc__ float *)0;
  __ubuf__ float *ub = (__ubuf__ float *)0;
  if ASC_IS_AIC {
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(a[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 32, 64, 0, 0);
    asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
    asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), (&(b[0])), 128, static_cast<asc_load_l2_cache_mode>(0), 32, 64, 0, 0);
    asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
    asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
    ascend_gemm_l1<32, 64, 32, 64, true, bfloat16_t>(((__cc__ float*)acc), ((__cbuf__ bfloat16_t*)buf_dyn_l1), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), (bool)1, 0, 0);
    asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
    asc_set_l0c_copy_nz_para(1, 0, 0);
    asc_copy_l0c2ub((&(ub[0])), ((__cc__ float*)acc), 32, 32, 32, 32, 0, static_cast<asc_dual_dst_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
    asc_sync_intra_arrive(PIPE_FIX, 0);
    asc_sync_intra_arrive(PIPE_FIX, 16);
  }
  if ASC_IS_AIV {
    int32_t sid = asc_get_sub_block_id();
    asc_sync_intra_wait(PIPE_MTE3, 0);
    asc_copy_ub2gm_align((__gm__ uint8_t*)((&(dst[0]))), (__ubuf__ uint8_t*)((&(ub[0]))), 1, 4096, static_cast<asc_store_l2_cache_mode>(4), 4096, 4096);
  }
}


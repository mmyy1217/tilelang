#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* a, __gm__ bfloat16_t* b, __gm__ float* c) {
  asc_init();
  __cbuf__ uint8_t *buf_dyn_l1 = (__cbuf__ uint8_t *)0;
  __ca__ bfloat16_t *a_l0 = (__ca__ bfloat16_t *)0;
  __cb__ bfloat16_t *b_l0 = (__cb__ bfloat16_t *)0;
  __cc__ float *acc = (__cc__ float *)0;
  asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048) + 3072), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(1), .blk_num = static_cast<uint64_t>(32), .dst_gap = static_cast<uint64_t>(0)});
  asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1) + 3072), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(1), .blk_num = static_cast<uint64_t>(32), .dst_gap = static_cast<uint64_t>(0)});
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1), (&(a[0])), 94, static_cast<asc_load_l2_cache_mode>(0), 31, 47, 0, 0);
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
  asc_copy_gm2l1_nd2nz(((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), (&(b[0])), 94, static_cast<asc_load_l2_cache_mode>(0), 19, 47, 0, 0);
  asc_sync_pipe(PIPE_MTE2);
  asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048) + 608), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(4), .blk_num = static_cast<uint64_t>(13), .dst_gap = static_cast<uint64_t>(19)});
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
  asc_sync_pipe(PIPE_MTE2);
  asc_fill_l1((__cbuf__ uint16_t*)((__cbuf__ uint8_t*)((__cbuf__ bfloat16_t*)buf_dyn_l1) + 992), (uint32_t)((uint32_t)0), { .repeat = static_cast<uint64_t>(4), .blk_num = static_cast<uint64_t>(1), .dst_gap = static_cast<uint64_t>(31)});
  asc_sync_notify(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(1));
  asc_copy_l12l0b(((__cb__ bfloat16_t*)b_l0), ((__cbuf__ bfloat16_t*)buf_dyn_l1 + 2048), 0, 0, 2, 4, 2, 2);
  asc_sync_wait(PIPE_MTE2, PIPE_MTE1, static_cast<event_t>(0));
  asc_copy_l12l0a(((__ca__ bfloat16_t*)a_l0), ((__cbuf__ bfloat16_t*)buf_dyn_l1), 0, 0, 2, 4, 2, 2);
  asc_sync_notify(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE1, PIPE_M, static_cast<event_t>(0));
  asc_mmad(((__cc__ float*)acc), ((__ca__ bfloat16_t*)a_l0), ((__cb__ bfloat16_t*)b_l0), 32, 64, 32, 0, 1, 0, (bool)1);
  asc_sync_notify(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  asc_sync_wait(PIPE_M, PIPE_FIX, static_cast<event_t>(0));
  asc_set_l0c_copy_nz_para(1, 0, 0);
  asc_copy_l0c2gm((&(c[0])), ((__cc__ float*)acc), 19, 31, 19, 32, static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);
}


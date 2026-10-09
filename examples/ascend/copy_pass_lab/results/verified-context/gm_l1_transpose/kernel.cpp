#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
extern "C" __global__ __cube__ void main_kernel(__gm__ bfloat16_t* src) {
  asc_init();
  __cbuf__ bfloat16_t *l1 = (__cbuf__ bfloat16_t *)0;
  asc_set_gm2l1_nz_para(1, 1, static_cast<uint16_t>(32), 0);
  asc_copy_gm2l1_dn2nz(((__cbuf__ bfloat16_t*)l1), (&(src[0])), 64, static_cast<asc_load_l2_cache_mode>(0), 32, 64, 0, 0);
}


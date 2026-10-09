#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/dcache_bypass.h>
extern "C" __global__ __vector__ void main_kernel(__gm__ float* dst, __gm__ float* src) {
  asc_init();
  tl::write_gm_bypass_dcache(((__gm__ float*)dst + 0), src[0]);
}


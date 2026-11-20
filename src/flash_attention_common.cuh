#pragma once

#include <cfloat>

// Reference-compatible exp decomposition used by all three versions.
__device__ __forceinline__ float ex2_approx_ftz(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float expf_parts(float x, float &scale) {
  const float range = __saturatef(
      fmaf(x, __int_as_float(0x3BBB989D), 0.5f));
  const float biased = __fmaf_rd(
      range, 252.0f, __int_as_float(0x4B400001));
  const float exponent = biased + __int_as_float(0xCB40007F);
  const float high = fmaf(x, __int_as_float(0x3FB8AA3B), -exponent);
  const float low = fmaf(x, __int_as_float(0x32A57060), high);
  scale = __int_as_float(__float_as_int(biased) << 23);
  return ex2_approx_ftz(low);
}

template <typename T>
struct FlashDeviceCache {
  T *q = nullptr;
  T *k = nullptr;
  T *v = nullptr;
  T *o = nullptr;
  size_t q_cap = 0;
  size_t k_cap = 0;
  size_t v_cap = 0;
  size_t o_cap = 0;
};

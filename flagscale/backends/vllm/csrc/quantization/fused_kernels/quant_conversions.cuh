// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

/**
 * __device__ helper functions to deal with float -> quant datatype conversion
 */

#include "quantization/vectorization.cuh"
// TODO(luka/varun):refactor common.cuh to use this file instead
#include "quantization/fp8/common.cuh"
#include "attention/dtype_float16.cuh"

namespace vllm {

// TODO(luka/varun): combine into common utilities for int8
//  (with int8_quant_kernels.cu)
static __device__ __forceinline__ int8_t float_to_int8_rn(float const x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  // uint32_t dst;
  // asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "r"(x));
  // return reinterpret_cast<const int8_t&>(dst);

  // int32_t dst;
  // dst = (int32_t)(x > 0? x + 0.5: x - 0.5);
  // dst = (char)(dst > 127 ? 127 : (dst < -128 ? -128 : dst));
  // return reinterpret_cast<const int8_t&>(dst);

  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -127);
  return reinterpret_cast<const int8_t&>(dst);

  // return static_cast<int8_t>(fmaxf(-127.0f, fminf(roundf(x), 127.0f)));
#endif
}

template <typename Tin>
__inline__ __device__ int8_t scaled_quant_to_int8(
    const Tin& x, const float *scale);

// half -> int8
template <>
__inline__ __device__ int8_t scaled_quant_to_int8<uint16_t>(
    const uint16_t& x, const float *scale) {
  return float_to_int8_rn(half_to_float(x) / *scale);
}

// bf16 -> int8
template <>
__inline__ __device__ int8_t scaled_quant_to_int8<__nv_bfloat16>(
    const __nv_bfloat16& x, const float *scale) {
  return float_to_int8_rn(__bfloat162float(x) / *scale);
}

static __device__ __forceinline__ FP8_TYPE float_to_fp8(float const x) {
  float const r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<FP8_TYPE>(r);
}

template <typename quant_type_t, bool is_scale_inverted, typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, FP8_TYPE>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_fp8(x * scale);
    } else {
      return float_to_fp8(x / scale);
    }
  }
};

template <typename scalar_t, typename quant_type_t, bool is_scale_inverted>
__device__ void scaled_quant_conversion(quant_type_t* __restrict__ output,
                                        scalar_t const* __restrict__ input,
                                        float const scale, int const tid,
                                        int const num_elements,
                                        int const step) {
  for (int i = tid; i < num_elements; i += step) {
    output[i] = ScaledQuant<quant_type_t, is_scale_inverted>(input[i], scale);
  }
}

}  // namespace vllm

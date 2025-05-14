// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  //uint32_t dst;
  //asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  //return reinterpret_cast<const int8_t&>(dst);
  constexpr float c = 0.5;
  int32_t dst;
  dst = (int32_t)(x > 0 ? x + c: x - c);
  dst = (char)(dst > 127 ? 127 : (dst < -128 ? -128 : dst));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

static inline __device__ int32_t float_to_int32_rn(float x) {
#ifdef USE_ROCM
  // int32_max is not exactly representable as float.
  // Therefore, we need to be careful and manually return int32_max on overflow.
  // For symmetry, we also do the same for int32_min, even though it is exactly
  // representable as float and the conversion should be exact.
  static constexpr auto i32_min = std::numeric_limits<int32_t>::min();
  static constexpr auto i32_min_f = static_cast<float>(i32_min);
  static constexpr auto i32_max = std::numeric_limits<int32_t>::max();
  static constexpr auto i32_max_f = static_cast<float>(i32_max);

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate on the higher end.
  if (dst >= i32_max_f) {
    return i32_max;
  }
  // saturate on the lower end.
  if (dst <= i32_min_f) {
    return i32_min;
  }

  return static_cast<int32_t>(dst);
#else
  // CUDA path
  uint32_t dst;
#ifdef MX_MACA
  asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(dst) : "f"(x));
#endif
  return reinterpret_cast<const int32_t&>(dst);
#endif
}

static inline __device__ int8_t int32_to_int8(int32_t x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<int32_t>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<int32_t>(std::numeric_limits<int8_t>::max());

  // saturate
  int32_t dst = std::clamp(x, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
#ifdef MX_MACA
  asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(dst) : "r"(x));
#endif
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) / scale);
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void static_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type const* scale_ptr, azp_type const* azp_ptr,
    const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  scale_type const scale = *scale_ptr;
  azp_type const azp = *azp_ptr;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[token_idx * hidden_size + i]);
    auto const quant_val = int32_to_int8(float_to_int32_rn(val / scale) + azp);
    out[token_idx * hidden_size + i] = quant_val;
  }
}

template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[token_idx * hidden_size + i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) * tmp_scale);
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_kernel_sreg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  scalar_t absmax_val = static_cast<scalar_t>(0.0f);
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  scalar_t reg_src0[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = blockDim_x * N;
  int length = min(hidden_size, reg_length);
  int index = tid * N;
  if(index < length) {
    *(VT*)reg_src0 = *(VT*)(ptr_input + index);
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      scalar_t val = abs(reg_src0[i]);
      absmax_val = max(absmax_val, val);
    }
  }

  using BlockReduce = cub::BlockReduce<scalar_t, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
  __shared__ scale_type block_absmax_val;
  if (tid == 0) {
    block_absmax_val = static_cast<scale_type>(block_absmax_val_maybe);
    scale[token_idx] = static_cast<scale_type>(block_absmax_val / 127.0f);
  }
  __syncthreads();
  float const tmp_scale = 127.0f / block_absmax_val;
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
            static_cast<float>(reg_src0[i]) * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_kernel_reg_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  scalar_t absmax_val = static_cast<scalar_t>(0.0f);
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  scalar_t reg_src0[N];
  scalar_t reg_src1[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = 2 * blockDim_x * N;
  int length = min(hidden_size, reg_length);
  int index = 2 * tid * N;
  if(index < length) {
    *(VT*)reg_src0 = *(VT*)(ptr_input + index);
    *(VT*)reg_src1 = *(VT*)(ptr_input + index + N);
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      scalar_t val = abs(reg_src0[i]);
      absmax_val =  max(val, absmax_val);
      val = abs(reg_src1[i]);
      absmax_val = max(val, absmax_val);
    }
  }

  using BlockReduce = cub::BlockReduce<scalar_t, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  scalar_t const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
  __shared__ scale_type block_absmax_val;
  if (tid == 0) {
    block_absmax_val = static_cast<scale_type>(block_absmax_val_maybe);
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();
  float const tmp_scale = 127.0f / block_absmax_val;
  int8_t* ptr_output = out + token_idx * hidden_size;
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    constexpr int ON = 2 * N;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
             static_cast<float>(reg_src0[i]) * tmp_scale);
    }
    ptr_reg = ptr_reg + N;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
            static_cast<float>(reg_src1[i]) * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__global__ void dynamic_scaled_int8_quant_kernel_sm_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, int blockDim_x) {
  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int stride = blockDim_x * N;
  __shared__ float sm_buffer[8064];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        ptr_sm_buffer[j] = val;
        val = val > zero ? val : -val;
        absmax_val = val > absmax_val ? val : absmax_val;
    }
  }
  using BlockReduce = cub::BlockReduce<float, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }

  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  int8_t* ptr_output = out + token_idx * hidden_size;
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_reg[j] = float_to_int8_rn(
            ptr_sm_buffer[j] * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1>
__launch_bounds__(1024) __global__ void dynamic_scaled_int8_quant_kernel_opt(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, const int hidden_size, const int blockDim_x) {
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int const tid = threadIdx.x * N;
  int const token_idx = blockIdx.x;
  float absmax_val = 0.0f;
  int stride = blockDim_x * N;
  const scalar_t * ptr_input = input + token_idx * hidden_size;

  for (int i = tid ; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        val = val > 0 ? val : -val;
        absmax_val = val > absmax_val ? val : absmax_val;
    }
  }

    using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[token_idx] = block_absmax_val / 127.0f;
  }
  __syncthreads();

  float const tmp_scale = 127.0f / block_absmax_val;
  int8_t* ptr_output = out + token_idx * hidden_size;
  for (int i = tid; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    VT1 vdst;
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    int8_t* ptr_dst = (int8_t*)&vdst;
    #pragma unroll N
    for(int j = 0; j < N; ++j) {
        ptr_dst[j] = float_to_int8_rn(
        static_cast<float>(ptr_src[j]) * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename azp_type>
__global__ void dynamic_scaled_int8_azp_quant_kernel(
    scalar_t const* __restrict__ input, int8_t* __restrict__ out,
    scale_type* scale, azp_type* azp, const int hidden_size) {
  int const token_idx = blockIdx.x;

  // Scan for the min and max value for this token
  float max_val = std::numeric_limits<float>::min();
  float min_val = std::numeric_limits<float>::max();
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto val = static_cast<float>(input[token_idx * hidden_size + i]);
    max_val = std::max(max_val, val);
    min_val = std::min(min_val, val);
  }

  // Reduce the max and min values across the block
  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  max_val = BlockReduce(reduceStorage).Reduce(max_val, cub::Max{}, blockDim.x);
  __syncthreads();  // Make sure min doesn't mess with max shared memory
  min_val = BlockReduce(reduceStorage).Reduce(min_val, cub::Min{}, blockDim.x);

  __shared__ scale_type scale_sh;
  __shared__ azp_type azp_sh;

  // Compute the scale and zero point and store them, only on the first thread
  if (threadIdx.x == 0) {
    float const scale_val = (max_val - min_val) / 255.0f;
    // Use rounding to even (same as torch.round)
    auto const azp_float = std::nearbyint(-128.0f - min_val / scale_val);
    auto const azp_val = static_cast<azp_type>(azp_float);

    // Store the scale and azp into shared and global
    scale[token_idx] = scale_sh = scale_val;
    azp[token_idx] = azp_sh = azp_val;
  }

  // Wait for the scale and azp to be computed
  __syncthreads();

  float const scale_val = scale_sh;
  azp_type const azp_val = azp_sh;

  // Quantize the values
  for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    auto const val = static_cast<float>(input[token_idx * hidden_size + i]);
    auto const quant_val =
        int32_to_int8(float_to_int32_rn(val / scale_val) + azp_val);
    out[token_idx * hidden_size + i] = quant_val;
  }
}

}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,          // [..., hidden_size]
                              torch::Tensor const& input,  // [..., hidden_size]
                              torch::Tensor const& scale,
                              c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scale.numel() == 1);
  TORCH_CHECK(!azp || azp->numel() == 1);

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          vllm::static_scaled_int8_quant_kernel<scalar_t, float>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), hidden_size);
        } else {
          vllm::static_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scale.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}

void dynamic_scaled_int8_quant(
    torch::Tensor& out,          // [..., hidden_size]
    torch::Tensor const& input,  // [..., hidden_size]
    torch::Tensor& scales, c10::optional<torch::Tensor> const& azp) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(scales.is_contiguous());
  TORCH_CHECK(!azp || azp->is_contiguous());

  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
        if (!azp) {
          int n = 16 / sizeof(scalar_t);
          if(hidden_size <= 4096 && ((hidden_size & (n - 1)) == 0) && n == 8) {
            int gridsize = num_tokens;
            int blocksize = 512;
            vllm::dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2><<<gridsize, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize);
          } else if(hidden_size > 4096 &&hidden_size <= 8192 && ((hidden_size & (2*n - 1)) == 0) && n == 8) {
            int gridsize = num_tokens; int blocksize = 512;
            vllm::dynamic_scaled_int8_quant_kernel_reg_opt<scalar_t, float, float4, float4><<<gridsize, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize);
          } else if(hidden_size <= 8064 && (hidden_size & (n - 1)) == 0 && n == 8) {
            int gridsize = num_tokens;
            int blocksize = 512;
            vllm::dynamic_scaled_int8_quant_kernel_sm_opt<scalar_t, float, float4, float2><<<gridsize, blocksize, 0, stream>>>(
              input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize);
          } else if (hidden_size > 8064 && ((hidden_size & (n - 1)) == 0 && n == 8)) {
            int blocksize = 1024;
            vllm::dynamic_scaled_int8_quant_kernel_opt<scalar_t, float,float4,float2>
                    <<<grid, blocksize, 0, stream>>>(
                        input.data_ptr<scalar_t>(),out.data_ptr<int8_t>(),scales.data_ptr<float>(),hidden_size,blocksize);
          } else {
              vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float>
                  <<<grid, block, 0, stream>>>(
                      input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                      scales.data_ptr<float>(), hidden_size);
          }
        } else {
          vllm::dynamic_scaled_int8_azp_quant_kernel<scalar_t, float, int32_t>
              <<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(),
                  scales.data_ptr<float>(), azp->data_ptr<int32_t>(),
                  hidden_size);
        }
      });
}

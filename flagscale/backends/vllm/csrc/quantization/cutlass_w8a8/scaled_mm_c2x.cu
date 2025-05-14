// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <stddef.h>
#include <torch/all.h>
#include "mctlass/mctlass.h"

#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm75_dispatch.cuh"

using namespace vllm;

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).
*/

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm75_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
#if 0
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
#endif
}

void cutlass_scaled_mm_sm75(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
    int32_t m = a.size(0);
    int32_t n = b.size(1);
    int32_t k = a.size(1);

    using ArchTag = mctlass::arch::Sm80;
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = mctlass::half_t;
    using ElementCompute = float;
    using LayoutA = mctlass::layout::RowMajor;
    //using LayoutB = mctlass::layout::RowMajor;
    using LayoutB = mctlass::layout::ColumnMajor;
    using LayoutC = mctlass::layout::RowMajor;

    if (out.dtype() == torch::kBFloat16)
    {
    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<maca_bfloat16*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    if (bias) {
        mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
        mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
        using mctlassGemmScaleOp = mctlassGemmScale<
          ElementA,
          LayoutA,
          ElementB,
          LayoutB,
          maca_bfloat16,
          LayoutC,
          ElementCompute,
          ArchTag,
          scale_type
        >;
        maca_bfloat16 *bias_t;
        bias_t = static_cast<maca_bfloat16 *>(bias.value().data_ptr());
        mctlassGemmScaleOp mctlass_op;
        mctlass::gemm::GemmCoord problem_size(m, n, k);
        typename mctlassGemmScaleOp::Arguments arguments{
            mctlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,//batch_count
            {scale_a, scale_b, bias_t},
            a_ptr,
            b_ptr,
            c_ptr,
            c_ptr,
            m * k,
            n * k,
            m * n,
            m * n,
            k,
            n,
            n,
            n
        };
        mctlass_op(arguments, NULL, stream);
    }
    else{
        mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
        mctlass::epilogue::thread::ScaleType::ScaleAvBv;
        using mctlassGemmScaleOp = mctlassGemmScale<
          ElementA,
          LayoutA,
          ElementB,
          LayoutB,
          maca_bfloat16,
          LayoutC,
          ElementCompute,
          ArchTag,
          scale_type
        >;
        mctlassGemmScaleOp mctlass_op;
        mctlass::gemm::GemmCoord problem_size(m, n, k);
        typename mctlassGemmScaleOp::Arguments arguments{
            mctlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,//batch_count
            {scale_a, scale_b, nullptr},
            a_ptr,
            b_ptr,
            c_ptr,
            c_ptr,
            m * k,
            n * k,
            m * n,
            m * n,
            k,
            n,
            n,
            n
        };
        mctlass_op(arguments, NULL, stream);
    }
    }
    else{
    auto a_ptr = static_cast<ElementA const*>(a.data_ptr());
    auto b_ptr = static_cast<ElementB const*>(b.data_ptr());
    auto c_ptr = static_cast<ElementC*>(out.data_ptr());
    auto scale_a = a_scales.data_ptr<float>();
    auto scale_b = b_scales.data_ptr<float>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    if (bias) {
        mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
        mctlass::epilogue::thread::ScaleType::ScaleAvBvBias;
        using mctlassGemmScaleOp = mctlassGemmScale<
          ElementA,
          LayoutA,
          ElementB,
          LayoutB,
          ElementC,
          LayoutC,
          ElementCompute,
          ArchTag,
          scale_type
        >;
        ElementC *bias_t;
        bias_t = static_cast<ElementC *>(bias.value().data_ptr());
        mctlassGemmScaleOp mctlass_op;
        mctlass::gemm::GemmCoord problem_size(m, n, k);
        typename mctlassGemmScaleOp::Arguments arguments{
            mctlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,//batch_count
            {scale_a, scale_b, bias_t},
            a_ptr,
            b_ptr,
            c_ptr,
            c_ptr,
            m * k,
            n * k,
            m * n,
            m * n,
            k,
            n,
            n,
            n
        };
        mctlass_op(arguments, NULL, stream);
    }
    else{
        mctlass::epilogue::thread::ScaleType::ScaleBiasKind const scale_type =
        mctlass::epilogue::thread::ScaleType::ScaleAvBv;
        using mctlassGemmScaleOp = mctlassGemmScale<
          ElementA,
          LayoutA,
          ElementB,
          LayoutB,
          ElementC,
          LayoutC,
          ElementCompute,
          ArchTag,
          scale_type
        >;
        mctlassGemmScaleOp mctlass_op;
        mctlass::gemm::GemmCoord problem_size(m, n, k);
        typename mctlassGemmScaleOp::Arguments arguments{
            mctlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,//batch_count
            {scale_a, scale_b, nullptr},
            a_ptr,
            b_ptr,
            c_ptr,
            c_ptr,
            m * k,
            n * k,
            m * n,
            m * n,
            k,
            n,
            n,
            n
        };
        mctlass_op(arguments, NULL, stream);
    }
    }
}

#if 0
void cutlass_scaled_mm_azp_sm75(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm80_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  TORCH_CHECK(a.dtype() == torch::kInt8);
  TORCH_CHECK(b.dtype() == torch::kInt8);

  if (out.dtype() == torch::kBFloat16) {
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    TORCH_CHECK(out.dtype() == torch::kFloat16);
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm80(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm80(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(torch::Tensor& out, torch::Tensor const& a,
                                     torch::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == torch::kInt8) {
    TORCH_CHECK(b.dtype() == torch::kInt8);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      assert(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    TORCH_CHECK(a.dtype() == torch::kFloat8_e4m3fn);
    TORCH_CHECK(b.dtype() == torch::kFloat8_e4m3fn);

    if (out.dtype() == torch::kBFloat16) {
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      TORCH_CHECK(out.dtype() == torch::kFloat16);
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_mm_sm89(torch::Tensor& out, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            c10::optional<torch::Tensor> const& bias) {
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);
  if (bias) {
    TORCH_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}
#endif
void cutlass_scaled_mm_azp_sm89(torch::Tensor& out, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                c10::optional<torch::Tensor> const& azp,
                                c10::optional<torch::Tensor> const& bias) {
#if 0
  TORCH_CHECK(a_scales.dtype() == torch::kFloat32);
  TORCH_CHECK(b_scales.dtype() == torch::kFloat32);

  if (azp) {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
#endif
}

/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Tests for device-wide Implicit GEMM interface
*/

#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"


#include "hytlass/conv/kernel/default_conv2d_fprop.h"
#include "hytlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"


////////////////////////////////////////////////////////////////////////////////

std::vector<hytlass::conv::Conv2dProblemSize> Conv2dFewChannelProblemSizes(int channels) {

  std::vector<hytlass::conv::Conv2dProblemSize> problems;

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 8, 8, channels},   // input size  (NHWC)
    {16, 3, 3, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 16, 16, channels},   // input size  (NHWC)
    {16, 3, 3, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 16, 16, channels},   // input size  (NHWC)
    {16, 7, 7, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {32, 7, 7, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 7, 7, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 5, 5, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {1, 1},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  problems.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 224, 224, channels},   // input size  (NHWC)
    {64, 5, 5, channels},   // filter size (KRSC)
    {1, 1, 1, 1},                      // padding (pad_h, _, pad_w, _)
    {2, 2},                            // stride (stride_h, stride_w)
    {1, 1}                             // dilation (dilation_h, dilation_w)
  ));

  return problems;
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_8,
  128x128_64x1_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 8;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_4,
  128x128_64x1_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 4;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_2,
  128x128_64x1_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 2;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(2 * kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_1,
  128x128_64x1_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 1;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(3 * kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_8,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 8;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_4,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 4;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_2,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 2;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(2 * kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_1,
  128x128_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  int const kChannelCount = 1;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kFewChannels,
    hytlass::conv::StrideSupport::kStrided,
    kChannelCount,
    kChannelCount
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificConv2d<Conv2dFprop>(
    Conv2dFewChannelProblemSizes(3 * kChannelCount)));
}

////////////////////////////////////////////////////////////////////////////////

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

#include "hytlass/conv/kernel/default_conv2d_dgrad.h"
#include "hytlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_testbed.h"


////////////////////////////////////////////////////////////////////////////////
//                           Strided Dgrad (Analytic)
////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;


// run specific problem size in the unit test first
    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 4, 4, 8},     // input size (NHWC)
      {8, 3, 3, 8},     // filter size (KRSC)
      {0, 0, 0, 0},         // padding (pad_h, _, pad_w, _)
      {3, 3},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x256_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x256_64x1_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_align4,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
      4,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  test::conv::device::Conv2dProblemVector problem_size_list;

  // run specific problem size in the unit test first
  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 4, 4, 12},     // input size (NHWC)
    {8, 3, 3, 12},     // filter size (KRSC)
    {0, 0, 0, 0},      // padding (pad_h, _, pad_w, _)
    {3, 3},            // stride (stride_h, stride_w)
    {1, 1}             // dilation (dilation_h, dilation_w)
  ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}


////////////////////////////////////////////////////////////////////////////////
//                           Strided Dgrad (Optimized)
////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;

 // run specific problem size in the unit test first
    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 56, 56, 8},   // input size (NHWC)
      {8, 1, 1, 8},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 55, 55, 8},   // input size (NHWC)
      {8, 1, 1, 8},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_align4,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
      4,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;

  // run specific problem size in the unit test first
  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 56, 56, 12},   // input size (NHWC)
    {8, 1, 1, 12},     // filter size (KRSC)
    {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
    {2, 2},           // stride (stride_h, stride_w)
    {1, 1}            // dilation (dilation_h, dilation_w)
  ));

  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 55, 55, 12},   // input size (NHWC)
    {8, 1, 1, 12},     // filter size (KRSC)
    {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
    {2, 2},           // stride (stride_h, stride_w)
    {1, 1}            // dilation (dilation_h, dilation_w)
  ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;


// run specific problem size in the unit test first
    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 4, 4, 8},     // input size (NHWC)
      {8, 3, 3, 8},     // filter size (KRSC)
      {0, 0, 0, 0},         // padding (pad_h, _, pad_w, _)
      {3, 3},               // stride (stride_h, stride_w)
      {1, 1}                // dilation (dilation_h, dilation_w)
    ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x256_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x256_64x2_64x64x64) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_align4,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
      4,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  test::conv::device::Conv2dProblemVector problem_size_list;

  // run specific problem size in the unit test first
  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 4, 4, 12},     // input size (NHWC)
    {8, 3, 3, 12},     // filter size (KRSC)
    {0, 0, 0, 0},      // padding (pad_h, _, pad_w, _)
    {3, 3},            // stride (stride_h, stride_w)
    {1, 1}             // dilation (dilation_h, dilation_w)
  ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//                           Strided Dgrad (Optimized)
////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kStrided
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;

 // run specific problem size in the unit test first
    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 56, 56, 8},   // input size (NHWC)
      {8, 1, 1, 8},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

    problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
      {1, 55, 55, 8},   // input size (NHWC)
      {8, 1, 1, 8},   // filter size (KRSC)
      {0, 0, 0, 0},       // padding (pad_h, _, pad_w, _)
      {2, 2},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Strided_Dgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_align4,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  /// Device-level Conv2d instance
  using Conv2dDgradKernel = typename hytlass::conv::kernel::DefaultConv2dDgrad<
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
      4,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kStrided,
    4,
    4
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  test::conv::device::Conv2dProblemVector problem_size_list;

  // run specific problem size in the unit test first
  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 56, 56, 12},   // input size (NHWC)
    {8, 1, 1, 12},     // filter size (KRSC)
    {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
    {2, 2},           // stride (stride_h, stride_w)
    {1, 1}            // dilation (dilation_h, dilation_w)
  ));

  problem_size_list.push_back(hytlass::conv::Conv2dProblemSize(
    {1, 55, 55, 12},   // input size (NHWC)
    {8, 1, 1, 12},     // filter size (KRSC)
    {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
    {2, 2},           // stride (stride_h, stride_w)
    {1, 1}            // dilation (dilation_h, dilation_w)
  ));

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>(problem_size_list));
}

////////////////////////////////////////////////////////////////////////////////
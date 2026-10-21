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
TEST(GFX928_Device_Conv2d_Dgrad_Analytic_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::tfloat32_t;
  using ElementB           = hytlass::tfloat32_t;
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
    hytlass::gemm::GemmShape<128, 128, 16>,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Dgrad_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::tfloat32_t;
  using ElementB           = hytlass::tfloat32_t;
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
    hytlass::gemm::GemmShape<128, 128, 16>,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Conv2d_Dgrad_Analytic_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::tfloat32_t;
  using ElementB           = hytlass::tfloat32_t;
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
    hytlass::gemm::GemmShape<128, 128, 16>,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic,
    hytlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;


  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_Device_Conv2d_Dgrad_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::tfloat32_t;
  using ElementB           = hytlass::tfloat32_t;
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
    hytlass::gemm::GemmShape<128, 128, 16>,
    hytlass::gemm::GemmShape<64, 64, 16>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
      ElementC,
      128 / hytlass::sizeof_bits<ElementC>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kOptimized,
    hytlass::conv::StrideSupport::kUnity
  >::Kernel;

  using Conv2dDgrad = hytlass::conv::device::ImplicitGemmConvolution<Conv2dDgradKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dDgrad>());
}
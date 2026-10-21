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
#include "hytlass/array.h"
#include "hytlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "hytlass/epilogue/thread/linear_combination_residual_block.h"
#include "hytlass/epilogue/thread/activation.h"
#include "hytlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "hytlass/conv/device/implicit_gemm_convolution.h"

#include "conv2d_with_broadcast_testbed.h"

TEST(GFX928_Device_Conv2d_Fprop_With_Broadcast_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationBiasElementwise<
    hytlass::half_t,
    float,
    float,
    hytlass::half_t,
    hytlass::half_t,
    8,
    hytlass::epilogue::thread::ReLu<float>
  >;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    EpilogueOutputOp,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2dWithBroadcast<Conv2dFprop>());
}

TEST(GFX928_Device_Conv2d_Fprop_With_Broadcast_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_splice,
  128x128_32x1_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationBiasElementwise<
    hytlass::half_t,
    float,
    float,
    hytlass::half_t,
    hytlass::half_t,
    8,
    hytlass::epilogue::thread::ReLu<float>
  >;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    EpilogueOutputOp,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    1,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2dWithBroadcast<Conv2dFprop>());
}

TEST(GFX928_Device_Conv2d_Fprop_With_Broadcast_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
  128x128_32x2_64x64x32) {

  /// Conv operation element types for the Gemm equivalent (ImplicitGemm)
  using ElementA           = hytlass::half_t;
  using ElementB           = hytlass::half_t;
  using ElementC           = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationBiasElementwise<
    hytlass::half_t,
    float,
    float,
    hytlass::half_t,
    hytlass::half_t,
    8,
    hytlass::epilogue::thread::ReLu<float>
  >;

  /// Device-level Conv2d instance
  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    EpilogueOutputOp,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestAllConv2dWithBroadcast<Conv2dFprop>());
}

// Test residual block fusion: UnaryOp(BinaryOp(ActivationOp(Conv2d(X) + bias), residual))
// LinearCombinationResidualBlock does not support the split-k mode unless ActivationOp is Identity.
// This is because the activation needs to be applied to the fully accumulated output of the Conv2d op,
// which only the last thread block would have an access to, before applying BinaryOp.
// The epilogue functor in the last thread block would have to be given three inputs, namely
// partial outputs, bias, and residual, but this is not supported in the current interface.
// Set TestSplitK = false to skip split-k tests with non-trivial ActivationOp.
template <
 typename ElementAccumulator,
 template<typename T> class ActivationOp,
 template<typename T> class BinaryOp,
 template<typename T> class UnaryOp,
 bool TestSplitK = true,
 bool EnableSplice = false,
 int Stage = 1
>
void TestResidaulBlock() {
  using ElementA = hytlass::half_t;
  using ElementB = hytlass::half_t;
  using ElementC = hytlass::half_t;
  using ElementD = ElementC;
  using ElementCompute = ElementAccumulator;

  constexpr int INST_K = EnableSplice ? 32 : 16;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationResidualBlock<
    ElementD,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    8,
    ActivationOp,
    BinaryOp,
    UnaryOp
  >;

  using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
    ElementA, hytlass::layout::TensorNHWC,
    ElementB, hytlass::layout::TensorNHWC,
    ElementC, hytlass::layout::TensorNHWC,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, INST_K>,
    EpilogueOutputOp,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Stage,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::IteratorAlgorithm::kAnalytic
  >::Kernel;

  using Conv2dFprop = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  struct ReferenceOp {
    using OutputOp = typename Conv2dFprop::EpilogueOutputOp;
    using ElementZ = typename OutputOp::ElementZ;

    ActivationOp<ElementCompute> activation;
    BinaryOp<ElementCompute> binary_op;
    UnaryOp<ElementCompute> unary_op;

    void operator()(ElementZ &Z, ElementZ&, ElementCompute conv2d, ElementCompute residual) {
      Z = ElementZ(unary_op(binary_op(activation(conv2d), residual)));
    }
  };

  bool passed = test::conv::device::TestAllConv2dWithBroadcast<Conv2dFprop, ReferenceOp, true, TestSplitK>();
  EXPECT_TRUE(passed);
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Plus_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
     128x128_32x1_64x64x32) {
  // Resnet
  TestResidaulBlock<float, hytlass::epilogue::thread::Identity, hytlass::plus, hytlass::epilogue::thread::ReLu>();
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Plus_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_splice,
     128x128_32x1_64x64x32) {
  // Resnet
  TestResidaulBlock<float, hytlass::epilogue::thread::Identity, hytlass::plus, hytlass::epilogue::thread::ReLu, true, true>();
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Multiply_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
     128x128_32x1_64x64x32) {
  // EfficientNet V2
  // Do not run split-K tests since the activation op is not Identity.
  TestResidaulBlock<float, hytlass::epilogue::thread::Sigmoid, hytlass::multiplies, hytlass::epilogue::thread::Identity, false>();
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Multiply_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_splice,
     128x128_32x1_64x64x32) {
  // EfficientNet V2
  // Do not run split-K tests since the activation op is not Identity.
  TestResidaulBlock<float, hytlass::epilogue::thread::Sigmoid, hytlass::multiplies, hytlass::epilogue::thread::Identity, false, true>();
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Plus_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
     128x128_32x2_64x64x32) {
  // Resnet
  TestResidaulBlock<float, hytlass::epilogue::thread::Identity, hytlass::plus, hytlass::epilogue::thread::ReLu, true, false, 2>();
}

TEST(GFX928_Device_Conv2d_Fprop_With_Residual_Block_Multiply_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32,
     128x128_32x2_64x64x32) {
  // EfficientNet V2
  // Do not run split-K tests since the activation op is not Identity.
  TestResidaulBlock<float, hytlass::epilogue::thread::Sigmoid, hytlass::multiplies, hytlass::epilogue::thread::Identity, false, false, 2>();
}

////////////////////////////////////////////////////////////////////////////////

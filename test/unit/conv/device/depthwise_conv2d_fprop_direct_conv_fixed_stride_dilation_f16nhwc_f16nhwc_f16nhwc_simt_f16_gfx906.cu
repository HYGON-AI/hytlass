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
    \brief Tests for device-wide Depthwise Direct Conv interface
*/

#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"


#include "hytlass/conv/kernel/default_depthwise_fprop.h"
#include "hytlass/conv/device/direct_convolution.h"

#include "conv2d_testbed.h"
#include "depthwise_conv2d_direct_conv_testbed.h"

std::vector<hytlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter3x3_stride1x1_dilation1x1() {
  std::vector<hytlass::conv::Conv2dProblemSize> problems;

  for (int channels = 16; channels <= 512; channels += 16) {
    problems.push_back(hytlass::conv::Conv2dProblemSize(
        {1, 8, 8, channels},                     // input size  (NHWC)
        {channels, 3, 3, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        hytlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));
    }
    return problems;
}

std::vector<hytlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter3x3_stride2x2_dilation2x2() {
    std::vector<hytlass::conv::Conv2dProblemSize> problems;
    for (int channels = 16; channels <= 512; channels += 16) {
        problems.push_back(hytlass::conv::Conv2dProblemSize(
            {1, 16, 16, channels},                   // input size  (NHWC)
            {channels, 3, 3, 1},                     // filter size (KRSC)
            {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
            {2, 2},                                  // stride (stride_h, stride_w)
            {2, 2},                                  // dilation (dilation_h, dilation_w)
            hytlass::conv::Mode::kCrossCorrelation,  // Convolution mode
            16,                                      // split_k_slices
            channels                                 // groups
            ));
    }

  return problems;
}

std::vector<hytlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter5x5_stride1x1_dilation1x1() {
  std::vector<hytlass::conv::Conv2dProblemSize> problems;

  for (int channels = 16; channels < 256; channels += 16) {
    problems.push_back(hytlass::conv::Conv2dProblemSize(
        {1, 16, 16, channels},                   // input size  (NHWC)
        {channels, 5, 5, 1},                     // filter size (KRSC)
        {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
        {1, 1},                                  // stride (stride_h, stride_w)
        {1, 1},                                  // dilation (dilation_h, dilation_w)
        hytlass::conv::Mode::kCrossCorrelation,  // Convolution mode
        16,                                      // split_k_slices
        channels                                 // groups
        ));
    }

    return problems;

}

std::vector<hytlass::conv::Conv2dProblemSize> DepthwiseFpropProblemSizes_filter5x5_stride2x2_dilation2x2() {
    std::vector<hytlass::conv::Conv2dProblemSize> problems;
    for (int channels = 16; channels < 256; channels += 16) {
        problems.push_back(hytlass::conv::Conv2dProblemSize(
            {1, 112, 112, channels},                 // input size  (NHWC)
            {channels, 5, 5, 1},                     // filter size (KRSC)
            {1, 1, 1, 1},                            // padding (pad_h, _, pad_w, _)
            {2, 2},                                  // stride (stride_h, stride_w)
            {2, 2},                                  // dilation (dilation_h, dilation_w)
            hytlass::conv::Mode::kCrossCorrelation,  // Convolution mode
            16,                                      // split_k_slices
            channels                                 // groups
            ));
    }

  return problems;
}


////////////////////////////////////////////////////////////////////////////////
TEST(
    GFX906_Device_Depthwise_conv2d_Fprop_Direct_Conv_FixedStrideDilation_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x32_2_8x32_Filter3x3_Stride1x1_Dilation1x1) {

  using ElementInputA = hytlass::half_t;
  using ElementInputB = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = hytlass::half_t;
  using ElementComputeEpilogue = hytlass::half_t;

  using LayoutInputA = hytlass::layout::TensorNHWC;
  using LayoutInputB = hytlass::layout::TensorNHWC;
  using LayoutOutput = hytlass::layout::TensorNHWC;

  // Operator class tag
  using MMAOp = hytlass::arch::OpClassSimt;

  // This code section describes GFX architecture number
  using SmArch = hytlass::arch::Gfx906;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 32;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = hytlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = hytlass::MatrixShape<3, 3>;

  // Threadblock tile shape
  using ThreadblockShape =
      hytlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = hytlass::gemm::GemmShape<8, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      hytlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static hytlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      hytlass::conv::IteratorAlgorithm::kFixedStrideDilation;
  using StrideShape = hytlass::MatrixShape<1, 1>;
  using DilationShape = hytlass::MatrixShape<1, 1>;

  constexpr int kEpilogueElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      hytlass::epilogue::thread::ScaleType::Default>;     

  using DepthwiseDirect2dConv = typename hytlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      hytlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      hytlass::conv::StrideSupport::kFixed,
      StrideShape,
      DilationShape>::Kernel;

  using Direct2dConv = hytlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
    DepthwiseFpropProblemSizes_filter3x3_stride1x1_dilation1x1()));
}

////////////////////////////////////////////////////////////////////////////////
TEST(
    GFX906_Device_Depthwise_conv2d_Fprop_Direct_Conv_FixedStrideDilation_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x32_4_8x32_Filter3x3_Stride2x2_Dilation2x2) {

  using ElementInputA = hytlass::half_t;
  using ElementInputB = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = hytlass::half_t;
  using ElementComputeEpilogue = hytlass::half_t;

  using LayoutInputA = hytlass::layout::TensorNHWC;
  using LayoutInputB = hytlass::layout::TensorNHWC;
  using LayoutOutput = hytlass::layout::TensorNHWC;

  // Operator class tag
  using MMAOp = hytlass::arch::OpClassSimt;

  // This code section describes GFX architecture number
  using SmArch = hytlass::arch::Gfx906;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 32;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = hytlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape =  hytlass::MatrixShape<3, 3>;

  // Threadblock tile shape
  using ThreadblockShape =
      hytlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = hytlass::gemm::GemmShape<8, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      hytlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static hytlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      hytlass::conv::IteratorAlgorithm::kFixedStrideDilation;
  using StrideShape = hytlass::MatrixShape<2, 2>;
  using DilationShape = hytlass::MatrixShape<2, 2>;

  constexpr int kEpilogueElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      hytlass::epilogue::thread::ScaleType::Default>;     

  using DepthwiseDirect2dConv = typename hytlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      hytlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      hytlass::conv::StrideSupport::kFixed,
      StrideShape,
      DilationShape>::Kernel;

  using Direct2dConv = hytlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
    DepthwiseFpropProblemSizes_filter3x3_stride2x2_dilation2x2()));
}

////////////////////////////////////////////////////////////////////////////////
TEST(
    GFX906_Device_Depthwise_conv2d_Fprop_Direct_Conv_FixedStrideDilation_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x64_2_16x64_Filter5x5_Stride1x1_Dilation1x1) {

  using ElementInputA = hytlass::half_t;
  using ElementInputB = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = hytlass::half_t;
  using ElementComputeEpilogue = hytlass::half_t;

  using LayoutInputA = hytlass::layout::TensorNHWC;
  using LayoutInputB = hytlass::layout::TensorNHWC;
  using LayoutOutput = hytlass::layout::TensorNHWC;

  // Operator class tag
  using MMAOp = hytlass::arch::OpClassSimt;

  // This code section describes GFX architecture number
  using SmArch = hytlass::arch::Gfx906;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 64;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = hytlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = hytlass::MatrixShape<5, 5>;

  // Threadblock tile shape
  using ThreadblockShape =
      hytlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = hytlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      hytlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static hytlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      hytlass::conv::IteratorAlgorithm::kFixedStrideDilation;
  using StrideShape = hytlass::MatrixShape<1, 1>;
  using DilationShape = hytlass::MatrixShape<1, 1>;

  constexpr int kEpilogueElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      hytlass::epilogue::thread::ScaleType::Default>;  

  using DepthwiseDirect2dConv = typename hytlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      hytlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      hytlass::conv::StrideSupport::kFixed,
      StrideShape,
      DilationShape>::Kernel;

  using Direct2dConv = hytlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
    DepthwiseFpropProblemSizes_filter5x5_stride1x1_dilation1x1()));
}

////////////////////////////////////////////////////////////////////////////////
TEST(
    GFX906_Device_Depthwise_conv2d_Fprop_Direct_Conv_FixedStrideDilation_f16nhwc_f16nhwc_f16nhwc_simt_f16,
    64x64_2_16x64_Filter5x5_Stride2x2_Dilation2x2) {

  using ElementInputA = hytlass::half_t;
  using ElementInputB = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = hytlass::half_t;
  using ElementComputeEpilogue = hytlass::half_t;

  using LayoutInputA = hytlass::layout::TensorNHWC;
  using LayoutInputB = hytlass::layout::TensorNHWC;
  using LayoutOutput = hytlass::layout::TensorNHWC;

  // Operator class tag
  using MMAOp = hytlass::arch::OpClassSimt;

  // This code section describes GFX architecture number
  using SmArch = hytlass::arch::Gfx906;

  // This code section describes the groups a thread block will compute
  constexpr int groups_per_cta = 32;

  // This code section describes the output tile <N, P, Q, C> a thread block will compute
  using ThreadBlockOutputShape = hytlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;

  // This code section describes the filter shape <R, S>
  using FilterShape = hytlass::MatrixShape<5, 5>;

  // Threadblock tile shape
  using ThreadblockShape =
      hytlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;

  // This code section describes tile size a warp will computes
  using WarpShape = hytlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;

  // This code section describes the size of MMA op
  using InstructionShape = hytlass::gemm::GemmShape<1, 1, 1>;

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock =
      hytlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
          1,
          ThreadBlockOutputShape::kN,
          ThreadBlockOutputShape::kH,
          ThreadBlockOutputShape::kW>;

  // Number of pipelines you want to use
  constexpr int NumStages = 2;

  // This code section describe iterator algorithm selected is Analytic or Optimized
  static hytlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      hytlass::conv::IteratorAlgorithm::kFixedStrideDilation;
  using StrideShape = hytlass::MatrixShape<2, 2>;
  using DilationShape = hytlass::MatrixShape<2, 2>;

  constexpr int kEpilogueElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
      ElementOutput,               // Data type of output matrix.
      kEpilogueElementsPerAccess,  // The number of elements per vectorized.
                                   // memory access. This becomes the vector width of
                                   // math instructions in the epilogue too.
      ElementAccumulator,          // Data type of accumulator
      ElementComputeEpilogue,      // Data type for alpha/beta in linear combination
      hytlass::epilogue::thread::ScaleType::Default>;  

  using DepthwiseDirect2dConv = typename hytlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      ThreadBlockOutputShape,
      FilterShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      hytlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      hytlass::conv::StrideSupport::kFixed,
      StrideShape,
      DilationShape>::Kernel;

  using Direct2dConv = hytlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;

  /// Run all unit test sizes with device-level Conv2d instance
  EXPECT_TRUE(test::conv::device::TestSpecificDepthwiseDirectConv2d<Direct2dConv>(
    DepthwiseFpropProblemSizes_filter5x5_stride2x2_dilation2x2()));
}

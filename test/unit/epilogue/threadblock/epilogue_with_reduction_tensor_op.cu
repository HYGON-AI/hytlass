/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

  \brief Unit tests for thread-level GEMM
*/

#include "../../common/hytlass_unit_test.h"

#include "hytlass/aligned_buffer.h"
#include "hytlass/half.h"

#include "hytlass/epilogue/thread/linear_combination_drelu.h"
#include "hytlass/gemm/warp/default_mma_tensor_op.h"
#include "hytlass/epilogue/threadblock/default_epilogue_with_reduction.h"
#include "hytlass/epilogue/threadblock/epilogue_with_reduction.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/host/tensor_fill.h"

#include "epilogue_with_reduction_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_64x64_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_64x64_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_128x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_128x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_128x64_64x32x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

#if ENABLE_BLOCKED_TESTS

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_128x64_64x32x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_64x128_32x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_64x128_32x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_128x256_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_128x256_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<128, 256, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f32_tensor_op_256x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_with_reduction_threadblock, f16_tensor_op_256x128_64x64x16) {

  //
  // Define the warp-level matrix multiply
  //

  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  int const kElementsPerAccess = 128 / hytlass::sizeof_bits<ElementOutput>::value;
  int const kPartitionsK = 1;
  
  using Shape = hytlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = ElementAccumulator;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutC = hytlass::layout::RowMajor;

  using WarpMmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      WarpShape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      LayoutC>::Type;

  //
  // Output operator
  //

  using OutputOp = hytlass::epilogue::thread::LinearCombinationDRelu<
    ElementAccumulator,
    ElementAccumulator,
    ElementOutput,
    ElementOutput,
    kElementsPerAccess
  >;

  using ReductionOp = hytlass::plus<ElementAccumulator>;

  //
  // Define the epilogue
  //

  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueWithReductionTensorOp<
    Shape,
    WarpMmaTensorOp,
    kPartitionsK,
    ElementOutput,
    OutputOp,
    ReductionOp,
    kElementsPerAccess
  >::Epilogue;

  //
  // Instantiate epilogue
  //

  EpilogueWithReductionTestbed<Epilogue> testbed;

  bool passed = testbed.run_all();

  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

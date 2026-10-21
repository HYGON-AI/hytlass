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

    \brief Unit tests for thread-level GEMM
*/

#include "../../common/hytlass_unit_test.h"

#include "hytlass/aligned_buffer.h"
#include "hytlass/half.h"

#include "hytlass/gemm/warp/default_mma_tensor_op.h"

#include "hytlass/core_io.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"

#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/gemm.h"

#include "testbed.h"

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_f16, 128x128x16_32x128x16_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<32, 128, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_f16, 128x128x32_64x64x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_f16, 128x128x32_32x32x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_f16, 32x32x32_32x32x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<32, 32, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x32_64x64x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x32_64x32x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x32_32x32x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x32_32x16x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x32_16x16x32_16x16x16) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x64_64x64x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x64_64x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x64_32x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x64_32x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_f16, 128x128x64_16x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////
TEST(GFX928_warp_gemm_tensor_op_crosswise_i8, 128x128x64_64x64x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_i8, 128x128x64_64x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_i8, 128x128x64_32x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_i8, 128x128x64_32x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_i8, 128x128x64_16x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 64>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_i8, 128x128x64_64x64x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_i8, 128x128x64_64x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_i8, 128x128x64_32x32x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_i8, 128x128x64_32x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_i8, 128x128x64_16x16x64_16x16x32) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor, hytlass::arch::OpMultiplyAdd>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 64> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x16_64x64x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x16_64x32x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x16_32x32x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x16_32x16x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x16_16x16x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 16>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x32_64x64x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x32_64x32x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x32_32x32x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x32_32x16x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 16, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_crosswise_tf32, 128x128x32_16x16x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<16, 16, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_tf32, 128x128x16_64x64x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_tf32, 128x128x16_32x32x16_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 16> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_tf32, 128x128x32_64x64x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_warp_gemm_tensor_op_congruous_tf32, 128x128x32_32x32x32_16x16x8) {
  using Shape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;
  using Element = hytlass::tfloat32_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
      Shape, InstructionShape, Element, LayoutA, Element, LayoutB, ElementC,
      hytlass::layout::RowMajor>::Type;

  test::gemm::warp::Testbed<MmaTensorOp,
                            hytlass::gemm::GemmShape<128, 128, 32> >()
      .run();
}

////////////////////////////////////////////////////////////////////////////////

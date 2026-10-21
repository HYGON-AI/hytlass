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
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"

#include "../../common/hytlass_unit_test.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/gemm.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x256x64_64x64x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 256x128x64_64x64x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<256, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x128x64_64x64x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x128x64_32x64x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 128, 64>,
    hytlass::gemm::GemmShape<32, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x64x64_64x32x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 64>,
    hytlass::gemm::GemmShape<64, 32, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x64x64_32x32x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<32, 32, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

TEST(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x128x64_64x32x64) {

  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 128, 64>,
    hytlass::gemm::GemmShape<64, 32, 64>,
    hytlass::gemm::GemmShape<16, 16, 32>,
    hytlass::epilogue::thread::LinearCombinationClamp<
      ElementOutput,
      128 / hytlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementAccumulator
    >,
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2
  >;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
}

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x256x128_64x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<128, 256, 128>,
      hytlass::gemm::GemmShape<64, 64, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 256x128x128_64x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<256, 128, 128>,
      hytlass::gemm::GemmShape<64, 64, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x128x128_64x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<128, 128, 128>,
      hytlass::gemm::GemmShape<64, 64, 128>,
      hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 256x64x128_64x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<256, 64, 128>,
      hytlass::gemm::GemmShape<64, 64, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x256x128_64x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<64, 256, 128>,
      hytlass::gemm::GemmShape<64, 64, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x128x128_32x64x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<64, 128, 128>,
      hytlass::gemm::GemmShape<32, 64, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 128x64x128_64x32x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<128, 64, 128>,
      hytlass::gemm::GemmShape<64, 32, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x64x128_32x32x128, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<64, 64, 128>,
      hytlass::gemm::GemmShape<32, 32, 128>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 256x64x64_64x64x64, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<256, 64, 64>,
      hytlass::gemm::GemmShape<64, 64, 64>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

HYTLASS_TEST_L1(GFX928_Device_Gemm_s8t_s8n_s32t_tensor_op_s32, 64x256x64_64x64x64, {
  using ElementOutput = int32_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = int32_t;

  using Gemm = hytlass::gemm::device::Gemm<
      int8_t, hytlass::layout::RowMajor, int8_t,
      hytlass::layout::ColumnMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
      hytlass::gemm::GemmShape<64, 256, 64>,
      hytlass::gemm::GemmShape<64, 64, 64>, hytlass::gemm::GemmShape<16, 16, 32>,
      hytlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator, ElementCompute>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllGemm<Gemm>());
} )

/////////////////////////////////////////////////////////////////////////////////////////////////

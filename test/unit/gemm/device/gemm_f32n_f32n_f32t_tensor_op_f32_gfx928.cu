/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-2-Clause
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
 * 2. Neither the name of the copyright holder nor the names of its
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 256x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<256, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 128x256x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 256, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 256x64x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<256, 64, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 64x256x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 256, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 32>,
    hytlass::gemm::GemmShape<64, 32, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 64x128x32_32x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 128, 32>,
    hytlass::gemm::GemmShape<32, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 64x64x32_32x32x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<32, 32, 32>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 128x128x64_64x64x64) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 256x64x64_64x64x64) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<256, 64, 64>,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 128x64x64_64x32x64) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 64>,
    hytlass::gemm::GemmShape<64, 32, 64>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

TEST(GFX928_Device_Gemm_f32n_f32n_f32t_tensor_op_f32, 64x64x64_32x32x64) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::Gemm<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<64, 64, 64>,
    hytlass::gemm::GemmShape<32, 32, 64>,
    hytlass::gemm::GemmShape<16, 16, 8>,
    hytlass::epilogue::thread::LinearCombination<
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

/////////////////////////////////////////////////////////////////////////////////////////////////
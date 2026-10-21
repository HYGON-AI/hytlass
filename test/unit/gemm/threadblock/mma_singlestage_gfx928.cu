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

#include "mma_pipelined_testbed.h"

/// All tests use single staged (kStages=1) mma pipeline for the gemm mainloop

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f16, 64x64x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 32);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f16, 128x128x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(128, 128, 64);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f16, multicta_256x256x96_128x128x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(256, 256, 96);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f32, 64x64x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f32, 128x128x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1;   

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_f32, multicta_256x256x96_128x128x32_64x64x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1;   

  hytlass::gemm::GemmCoord problem_size(256, 256, 96);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_s8, 64x64x32_64x64x32_16x16x32) {
 
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_s8, 64x64x64_64x64x64_16x16x32) {
 
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(GFX928_gemm_threadblock_singlestage_tensor_op_col_row_row_s8, 64x64x32_64x64x32_16x16x32) {
 
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_col_row_row_s8, 64x64x64_64x64x64_16x16x32) {
 
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_row_u8, 64x64x128_64x64x128_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::RowMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadBlockShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(GFX928_gemm_threadblock_singlestage_tensor_op_row_col_col_u8, 64x64x64_64x64x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;
  static const int kStages = 1; 

  hytlass::gemm::GemmCoord problem_size(64, 64, 64);

  using ThreadBlockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, 
      ElementA, LayoutA,
      ElementB, LayoutB, 
      ElementC, LayoutC, 
      hytlass::arch::OpClassTensorOp, kStages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore, kStages>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}
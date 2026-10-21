/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Unit tests for thread-level GEMM
*/

#include "mma_pipelined_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

// fp16

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x64x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     tensor_op_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x64x32_32x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x64x32_64x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x64x64_64x32x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using OperatorShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, OperatorShape, ElementA, LayoutA, ElementB,
      LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x128x32_32x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     tensor_op_256x128x32_32x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_256x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x32_32x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x64x32_64x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x64x64_64x32x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using OperatorShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, OperatorShape, ElementA, LayoutA, ElementB,
      LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x128x32_32x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_32x128x64_32x32x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(32, 256, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<32, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x32x64_32x32x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 32, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 32, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_32x128x32_32x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 256, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<32, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 2, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x32x32_32x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 32, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 2, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x64x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x128x64_32x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     tensor_op_128x128x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}


TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x64_32x32x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x128x64_32x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_128x128x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_256x256x384_128x128x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x64_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_256x256x384_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_512x256x768_256x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x384_128x128x64_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x32_64x64x64_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x384_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_256x256x96_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

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
      hytlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x96_128x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

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
      hytlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x384_256x128x32_64x64x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC,
      hytlass::arch::OpClassTensorOp>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

// tf32

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x64x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x64x32_64x32x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x128x32_32x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_256x256x192_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_512x256x384_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(4, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x64x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x64x16_64x32x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_64x128x16_32x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous, tensor_op_128x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_256x256x192_128x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_congruous,
     multicta_512x256x384_256x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 384);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x32_32x32x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x64x32_64x32x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x128x32_32x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x192_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x192_128x128x32_64x64x32_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(4, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x16_32x32x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x64x16_64x32x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x128x16_32x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_128x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 128);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x192_128x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x192_256x128x16_64x64x16_16x16x8) {
  using ElementA = hytlass::tfloat32_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::tfloat32_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 192);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 16>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 8>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

// int8

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_64x64x128_64x64x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_64x64x128_32x32x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_128x64x128_64x32x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 128>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_64x128x128_32x64x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 128>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_128x128x128_64x64x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x768_128x128x128_64x64x128_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 128>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x128_64x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x64_64x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x64x64_32x32x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_uint8_threadblock_crosswise, tensor_op_128x64x64_64x32x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 64, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x128x64_32x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     tensor_op_128x128x64_64x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 512);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x768_128x128x64_64x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_512x256x768_256x128x64_64x64x64_16x16x32) {
  using ElementA = int8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadblockShape = hytlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.0f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

// uint8

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_32x32x64_32x32x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(32, 32, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_64x32x64_32x32x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 32, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<64, 32, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 2, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_crosswise, tensor_op_32x64x64_32x32x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(32, 64, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 2, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_uint8_threadblock_crosswise, tensor_op_64x64x64_32x32x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_uint8_threadblock_crosswise, tensor_op_64x128x64_32x64x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 128, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_uint8_threadblock_crosswise, tensor_op_128x128x64_64x64x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(128, 128, 256);

  using ThreadBlockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_crosswise,
     multicta_256x256x192_128x128x64_64x64x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(256, 256, 192);

  using ThreadBlockShape = hytlass::gemm::GemmShape<128, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_uint8_threadblock_crosswise,
     multicta_512x256x768_256x128x64_64x64x64_16x16x32) {
  using ElementA = uint8_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = uint8_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = int32_t;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(512, 256, 768);

  using ThreadBlockShape = hytlass::gemm::GemmShape<256, 128, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;

  float alpha = 1.f;
  float beta = 0.f;
  int const Stages = 2;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadBlockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp,
      Stages, hytlass::arch::OpMultiplyAdd>;

  dim3 grid(2, 2);
  dim3 block(WARP_SIZE_GPU, 8, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

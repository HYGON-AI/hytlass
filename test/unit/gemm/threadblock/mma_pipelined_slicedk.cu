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
    \brief Unit tests for CTA-level GEMM specifically for sliced-k kernels (SM_61 and SM_75)
*/

#include "mma_pipelined_testbed_slicedk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NT DP4A
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_igemm_sliced_k, igemm_int8_nt_32x32x128_32x32x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 32, 128>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 32, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    hytlass::gemm::GemmCoord problem_size(32, 32, 128);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm_sliced_k_big, igemm_int8_nt_32x32x128_32x32x4_bigk) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 32, 128>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 32, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    hytlass::gemm::GemmCoord problem_size(32, 32, 1024);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}


TEST(GFX906_igemm_sliced_k, igemm_int8_nt_32x64x128_32x32x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 64, 128>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 32, 64>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2>;                                     // Stages,

    hytlass::gemm::GemmCoord problem_size(32, 64, 256);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Tensor Op GEMM for GFX928
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_gemm_threadblock_congruous_sliced, tensor_op_64x64x256_tb64x64x64_warp64x32x32_16x16x16) {

  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::ColumnMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp, 2,
      hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

TEST(GFX928_gemm_threadblock_crosswise_sliced, tensor_op_64x64x256_tb64x64x64_warp64x32x32_16x16x16) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::ColumnMajor;
  using ElementC = float;
  using LayoutC = hytlass::layout::ColumnMajor;

  hytlass::gemm::GemmCoord problem_size(64, 64, 256);

  using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = hytlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  float alpha = 1.f;
  float beta = 0.0f;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementC, LayoutC, hytlass::arch::OpClassTensorOp, 2,
      hytlass::arch::OpMultiplyAdd>;

  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 4, 1);

  test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                            problem_size.k(), alpha, beta)
      .run(grid, block);
}

////////////////////////////////////////////////////////////////////////////////

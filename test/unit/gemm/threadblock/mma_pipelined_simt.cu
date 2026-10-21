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

#include "mma_pipelined_testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// sgemm_NT
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_sgemm, sgemm_nt_32x64x8_32x64x1) {
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      hytlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
      hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
      hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
      float,                                  // ElementA,
      hytlass::layout::ColumnMajor,           // LayoutA,
      float,                                  // ElementB,
      hytlass::layout::RowMajor,              // LayoutB,
      float,                                  // ElementC,
      hytlass::layout::RowMajor,              // LayoutC,
      hytlass::arch::OpClassSimt,             // OpClass,
      2,                                      // Stages,
      hytlass::arch::OpMultiplyAdd            // Operator,
      >;                                     

  hytlass::gemm::GemmCoord problem_size(32, 64, 48);
  float alpha = 1.f;
  float beta = 0.0f;
  dim3 grid(1, 1);
  dim3 block(WARP_SIZE_GPU, 1, 1);
  test::gemm::threadblock::Testbed<MmaCore>(
      problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
      .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_sgemm, sgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_sgemm, sgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_sgemm, sgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        float,                                  // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        float,                                  // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_sgemm, sgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        float,                                    // ElementA,
        hytlass::layout::ColumnMajor,             // LayoutA,
        float,                                    // ElementB,
        hytlass::layout::RowMajor,                // LayoutB,
        float,                                    // ElementC,
        hytlass::layout::RowMajor,                // LayoutC,
        hytlass::arch::OpClassSimt,               // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd              // Operator,
        >;                                       

    hytlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// dgemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_dgemm, dgemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_dgemm, dgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_dgemm, dgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_dgemm, dgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        double,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        double,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        double,                                 // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_dgemm, dgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        double,                                   // ElementA,
        hytlass::layout::ColumnMajor,             // LayoutA,
        double,                                   // ElementB,
        hytlass::layout::RowMajor,                // LayoutB,
        double,                                   // ElementC,
        hytlass::layout::RowMajor,                // LayoutC,
        hytlass::arch::OpClassSimt,               // OpClass
        2,                                        // Stages,
        hytlass::arch::OpMultiplyAdd              // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_igemm, igemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        int,                                    // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int,                                    // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        int,                                      // ElementA,
        hytlass::layout::ColumnMajor,             // LayoutA,
        int,                                      // ElementB,
        hytlass::layout::RowMajor,                // LayoutB,
        int,                                      // ElementC,
        hytlass::layout::RowMajor,                // LayoutC,
        hytlass::arch::OpClassSimt,               // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd              // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// hgemm_NN
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_hgemm, hgemm_nt_32x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        hytlass::half_t,                        // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm, hgemm_nt_64x64x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        hytlass::half_t,                        // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm, hgemm_nt_32x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<32, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        hytlass::half_t,                        // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(32, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm, hgemm_nt_64x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 128, 8>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        hytlass::half_t,                        // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 128, 16);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm, hgemm_nt_128x128x8_32x64x1) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 8>,    // ThreadblockShape,
        hytlass::gemm::GemmShape<32, 64, 8>,      // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 1>,        // InstructionShape,
        hytlass::half_t,                          // ElementA,
        hytlass::layout::ColumnMajor,             // LayoutA,
        hytlass::half_t,                          // ElementB,
        hytlass::layout::RowMajor,                // LayoutB,
        hytlass::half_t,                          // ElementC,
        hytlass::layout::RowMajor,                // LayoutC,
        hytlass::arch::OpClassSimt,               // OpClass
        2,                                        // Stages,
        hytlass::arch::OpMultiplyAdd              // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 48);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(),
        problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// igemm_NT DP4A
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_igemm, igemm_int8_nt_64x64x16_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_64x64x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 4096);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_128x64x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 64, 32>,  // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_128x128x16_64x64x8) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 32>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_256x128x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 256, 32>, // ThreadblockShape,
        hytlass::gemm::GemmShape<128, 64, 32>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_128x256x64_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 256, 64>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 256, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nt_256x128x64_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 128, 64>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 128, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_64x64x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 32>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_128x64x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 64, 32>,  // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_128x128x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 32>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 32>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_256x128x32_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 256, 32>, // ThreadblockShape,
        hytlass::gemm::GemmShape<128, 64, 32>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_128x256x64_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 256, 64>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 256, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_tn_256x128x64_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 128, 64>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 64>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 128, 64);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_igemm, igemm_int8_nn_64x64x16_64x64x4) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 4>,      // InstructionShape,
        int8_t,                                 // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        int8_t,                                 // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        int,                                    // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_nt_64x64x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_nt_128x64x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 64, 16>,  // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_nt_128x128x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 16>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_nt_256x128x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 256, 16>, // ThreadblockShape,
        hytlass::gemm::GemmShape<128, 64, 16>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::RowMajor,              // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_tn_64x64x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_tn_128x64x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 64, 16>,  // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 2, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_tn_128x128x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<128, 128, 16>, // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(128, 128, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 4, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_tn_256x128x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<256, 256, 16>, // ThreadblockShape,
        hytlass::gemm::GemmShape<128, 64, 16>,   // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::RowMajor,              // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(256, 256, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 8, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

TEST(GFX906_hgemm_dot, hgemm_half_nn_64x64x16_64x64x2) {
    using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
        hytlass::gemm::GemmShape<64, 64, 16>,   // ThreadblockShape,
        hytlass::gemm::GemmShape<64, 64, 16>,    // WarpShape,
        hytlass::gemm::GemmShape<1, 1, 2>,      // InstructionShape,
        hytlass::half_t,                        // ElementA,
        hytlass::layout::ColumnMajor,           // LayoutA,
        hytlass::half_t,                        // ElementB,
        hytlass::layout::ColumnMajor,           // LayoutB,
        float,                                  // ElementC,
        hytlass::layout::RowMajor,              // LayoutC,
        hytlass::arch::OpClassSimt,             // OpClass
        2,                                      // Stages,
        hytlass::arch::OpMultiplyAdd            // Operator,
        >;

    hytlass::gemm::GemmCoord problem_size(64, 64, 32);
    float alpha = 1.f;
    float beta = 0.0f;
    dim3 grid(1, 1);
    dim3 block(WARP_SIZE_GPU, 1, 1);
    test::gemm::threadblock::Testbed<MmaCore>(
        problem_size.m(), problem_size.n(), problem_size.k(), alpha, beta)
        .run(grid, block, hytlass::Distribution::Uniform, hytlass::Distribution::Uniform);
}

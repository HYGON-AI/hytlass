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

#include "hytlass/gemm/thread/mma.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Sgemm_thread, col_row_3x4x2) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<3, 4, 2>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, col_row_4x4x2) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<4, 4, 2>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, row_col_4x4x2) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<4, 4, 2>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, col_row_4x5x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<4, 5, 3>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, col_row) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, row_col) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, col_col) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Sgemm_thread, row_row) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Dgemm_thread, col_row) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Dgemm_thread, row_col) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 1>,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_col_1x1x16) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<1, 1, 16>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_row_1x1x16) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<1, 1, 16>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_col_1x3x8) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<1, 3, 8>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_row_row_7x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_row_7x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_row_7x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_row_7x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_row_row_7x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_row_7x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_row_7x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_row_7x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<7, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_row_col_16x3x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_col_16x3x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_col_16x3x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_col_16x3x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, row_row_col_16x3x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, row_col_col_16x3x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, col_row_col_16x3x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, col_col_col_16x3x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 3, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, row_row_row_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, row_row_col_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}    

TEST(GFX906_Hgemm_thread, row_col_row_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}TEST(GFX906_Hgemm_thread, row_col_col_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_row_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_col_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_row_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_col_16x8x3) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 3>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_row_row_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_row_col_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_row_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, row_col_col_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_row_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_row_col_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_row_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor
  >().run();
}

TEST(GFX906_Hgemm_thread, col_col_col_16x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<16, 8, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Igemm_thread, col_row_1x1x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<1, 1, 4>,
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    int32_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Igemm_thread, col_row_2x3x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<2, 3, 4>,
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    int32_t,
    hytlass::layout::ColumnMajor
  >().run();
}

TEST(GFX906_Igemm_thread, col_row_8x8x4) {

  test::gemm::thread::Testbed<
    hytlass::gemm::GemmShape<8, 8, 4>,
    int8_t,
    hytlass::layout::RowMajor,
    int8_t,
    hytlass::layout::ColumnMajor,
    int32_t,
    hytlass::layout::ColumnMajor
  >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

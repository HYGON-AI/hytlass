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

#include "hytlass/complex.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/warp/mma_simt.h"

#include "testbed.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// NT SMEM layout
TEST(GFX906_warp_gemm_f32_col_row_col, 32x16x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

// TN SMEM layout
TEST(GFX906_warp_gemm_f32_row_col_col, 32x16x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

// TT SMEM layout
TEST(GFX906_warp_gemm_f32_row_row_col, 32x16x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

// NN SMEM layout
TEST(GFX906_warp_gemm_f32_col_col_col, 32x16x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// NT SMEM layout
TEST(GFX906_warp_gemm_f32_col_row_row, 16x32x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<4, 8>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 32, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

// TN SMEM layout
TEST(GFX906_warp_gemm_f32_row_col_row, 16x32x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<4, 8>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 32, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// NT SMEM layout
TEST(GFX906_warp_gemm_f32_col_row_col, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f32_col_row_row, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

// TN SMEM layout
TEST(GFX906_warp_gemm_f32_row_col_col, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f32_row_col_row, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
// NT SMEM layout
TEST(GFX906_warp_gemm_f32_col_row_col, 32x64x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f32_col_row_row, 32x64x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<4, 8>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 64, 8>,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

// TN SMEM layout
TEST(GFX906_warp_gemm_f32_row_col_col, 32x64x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f32_row_col_row, 32x64x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<4, 8>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 64, 8>,
    float,
    hytlass::layout::RowMajor,
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}
/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_complex_f32_col_row_col, 64x32x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using complex_f32_t = hytlass::complex<float>;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    complex_f32_t,
    hytlass::layout::ColumnMajor,
    complex_f32_t,
    hytlass::layout::RowMajor,
    complex_f32_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_complex_f32_col_row_row, 64x32x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using complex_f32_t = hytlass::complex<float>;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    complex_f32_t,
    hytlass::layout::ColumnMajor,
    complex_f32_t,
    hytlass::layout::RowMajor,
    complex_f32_t,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_f64_col_row_col, 8x4x1_1x1x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f64_col_row_row, 8x4x1_1x1x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_f64_col_row_col, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f64_col_row_row, 32x16x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_f64_col_row_col, 64x32x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_f64_col_row_row, 64x32x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    double,
    hytlass::layout::ColumnMajor,
    double,
    hytlass::layout::RowMajor,
    double,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_complex_f64_col_row_col, 32x16x1_1x1x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  using complex_f64_t = hytlass::complex<double>;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 16, 8>,
    complex_f64_t,
    hytlass::layout::ColumnMajor,
    complex_f64_t,
    hytlass::layout::RowMajor,
    complex_f64_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

TEST(GFX906_warp_gemm_complex_f64_col_row_row, 32x16x1_1x1x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::RowMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  using complex_f64_t = hytlass::complex<double>;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 16, 8>,
    complex_f64_t,
    hytlass::layout::ColumnMajor,
    complex_f64_t,
    hytlass::layout::RowMajor,
    complex_f64_t,
    hytlass::layout::RowMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8>>().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_f16_col_row, 8x4x1_1x1x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

TEST(GFX906_warp_gemm_f16_col_row, 16x8x1_2x2x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 8, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_f16_col_row, 32x16x1_4x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

TEST(GFX906_warp_gemm_f16_col_row, 64x16x1_8x4x1) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<8, 8, 1>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 32, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 128, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_warp_gemm_int8_col_row, col_row_8x4x8_1x1x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<8, 4, 8> >().run();
}

TEST(GFX906_warp_gemm_int8_col_row, col_row_8x4x4_1x1x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 8> >().run();
}

TEST(GFX906_warp_gemm_int8_col_row, col_row_16x4x4_2x1x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 1, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 4, 4>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<16, 4, 4> >().run();
}

TEST(GFX906_warp_gemm_int8_col_row, col_row_16x4x4_2x2x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 8, 4>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<16, 8, 4> >().run();
}

TEST(GFX906_warp_gemm_int8_col_row, col_row_32x16x4_4x4x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 16>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 16> >().run();
}


TEST(GFX906_warp_gemm_int8_col_row, col_row_128x64x4_16x16x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<16, 16, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<128, 64, 4>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 4> >().run();
}

TEST(GFX906_warp_gemm_int8_col_row, col_row_64x64x4_4x4x4) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 4>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 64, 8>,
    int8_t,
    hytlass::layout::ColumnMajorInterleaved<4>,
    int8_t,
    hytlass::layout::RowMajorInterleaved<4>,
    int,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<64, 64, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////
TEST(GFX906_warp_gemm_half_col_row, col_row_8x4x8_1x1x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 2>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<8, 4, 8> >().run();
}

TEST(GFX906_warp_gemm_half_col_row, col_row_8x4x4_1x1x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<1, 1, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<8, 4, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 8> >().run();
}

TEST(GFX906_warp_gemm_half_col_row, col_row_16x4x4_2x1x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 1, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 4, 2>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<16, 4, 4> >().run();
}

TEST(GFX906_warp_gemm_half_col_row, col_row_16x4x4_2x2x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<2, 2, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<16, 8, 2>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<16, 8, 4> >().run();
}

TEST(GFX906_warp_gemm_half_col_row, col_row_32x16x4_4x4x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<32, 16, 16>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 16> >().run();
}


TEST(GFX906_warp_gemm_half_col_row, col_row_128x64x4_16x16x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<16, 16, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<128, 64, 4>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<128, 64, 4> >().run();
}

TEST(GFX906_warp_gemm_half_col_row, col_row_64x64x4_4x4x2) {

  using Policy = hytlass::gemm::warp::MmaSimtPolicy<
    hytlass::MatrixShape<8, 4>,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::gemm::GemmShape<4, 4, 2>
  >;

  using Mma = hytlass::gemm::warp::MmaSimt<
    hytlass::gemm::GemmShape<64, 64, 8>,
    hytlass::half_t,
    hytlass::layout::ColumnMajorInterleaved<2>,
    hytlass::half_t,
    hytlass::layout::RowMajorInterleaved<2>,
    float,
    hytlass::layout::ColumnMajor,
    Policy
  >;

  test::gemm::warp::Testbed<Mma, hytlass::gemm::GemmShape<64, 64, 8> >().run();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

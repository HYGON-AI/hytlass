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
    \brief Tests for device-wide ELL block sparse GEMM interface
*/

#include <iostream>
#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/ell_gemm.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/reference/host/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/tensor_view_io.h"
#include "testbed_ell_block_sparse_gemm.h"

////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_Ell_Block_Sparse_Gemm_f16t_f16t_f16t_simt_f32, 128x256x8_64x64x8) {
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::EllGemm<
      hytlass::half_t, hytlass::layout::RowMajor, hytlass::half_t,
      hytlass::layout::RowMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
      hytlass::gemm::GemmShape<128, 256, 8>,
      hytlass::gemm::GemmShape<64, 64, 8>, hytlass::gemm::GemmShape<1, 1, 1>,
      hytlass::epilogue::thread::LinearCombination<
          ElementOutput, 1,
          ElementAccumulator, ElementAccumulator>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllEllBlockSparseGemm<Gemm>());
}

TEST(GFX906_Device_Ell_Block_Sparse_Gemm_f16t_f16t_f16t_simt_f32, 64x128x8_32x64x8) {
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::EllGemm<
      hytlass::half_t, hytlass::layout::RowMajor, hytlass::half_t,
      hytlass::layout::RowMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
      hytlass::gemm::GemmShape<64, 128, 8>,
      hytlass::gemm::GemmShape<32, 64, 8>, hytlass::gemm::GemmShape<1, 1, 1>,
      hytlass::epilogue::thread::LinearCombination<
          ElementOutput, 1,
          ElementAccumulator, ElementAccumulator>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllEllBlockSparseGemm<Gemm>());
}

TEST(GFX906_Device_Ell_Block_Sparse_Gemm_f16t_f16t_f16t_simt_f32, 64x64x8_32x32x8) {
  using ElementOutput = hytlass::half_t;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::EllGemm<
      hytlass::half_t, hytlass::layout::RowMajor, hytlass::half_t,
      hytlass::layout::RowMajor, ElementOutput, hytlass::layout::RowMajor,
      ElementAccumulator, hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
      hytlass::gemm::GemmShape<64, 64, 8>,
      hytlass::gemm::GemmShape<32, 32, 8>, hytlass::gemm::GemmShape<1, 1, 1>,
      hytlass::epilogue::thread::LinearCombination<
          ElementOutput, 1,
          ElementAccumulator, ElementAccumulator>,
      hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(test::gemm::device::TestAllEllBlockSparseGemm<Gemm>());
}

////////////////////////////////////////////////////////////////////////////////

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
#include "hytlass/gemm/device/gemm_splitk_parallel.h"

#include "../../common/hytlass_unit_test.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/gemm.h"

#include "testbed_splitk.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_GemmSplitKParallel_f32n_f32t_f32t_simt_f32, 128x128x8) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::GemmSplitKParallel<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::RowMajor,
    ElementOutput,
    hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 128, 8>,
    hytlass::gemm::GemmShape<32, 64, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(GFX906_Device_GemmSplitKParallel_f32n_f32n_f32n_simt_f32, 128x128x8) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using Gemm = hytlass::gemm::device::GemmSplitKParallel<
    float,
    hytlass::layout::ColumnMajor,
    float,
    hytlass::layout::ColumnMajor,
    ElementOutput,
    hytlass::layout::ColumnMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 128, 8>,
    hytlass::gemm::GemmShape<32, 64, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_GemmSplitKParallel_f64n_f64n_f64t_simt_f64, 64x128x8) {

  using Element = double;

  using Gemm = hytlass::gemm::device::GemmSplitKParallel<
    Element,
    hytlass::layout::ColumnMajor,
    Element,
    hytlass::layout::ColumnMajor,
    Element,
    hytlass::layout::RowMajor,
    Element,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<64, 128, 8>,
    hytlass::gemm::GemmShape<32, 64, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

TEST(GFX906_Device_GemmSplitKParallel_f64t_f64t_f64n_simt_f64, 64x64x8) {

  using Element = double;

  using Gemm = hytlass::gemm::device::GemmSplitKParallel<
    Element,
    hytlass::layout::RowMajor,
    Element,
    hytlass::layout::RowMajor,
    Element,
    hytlass::layout::ColumnMajor,
    Element,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<64, 64, 8>,
    hytlass::gemm::GemmShape<32, 64, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>
  >;

  test::gemm::device::TestAllGemmSplitK<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////


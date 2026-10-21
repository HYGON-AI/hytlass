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
#include "hute/tensor.hpp"
#include "hute/atom/mma_atom.hpp"

#include "hytlass/numeric_types.h"

#include "hytlass/gemm/device/gemm_universal_adapter.h"
#include "default_gemm_configuration.hpp"

#include "../../common/hytlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace hute;

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_Gemm_s8n_s8n_s32n_simt_s32, 128x128x64_64x64x64) {
  using Config = hytlass::gemm::device::DefaultGemmConfigurationToHytlass3Types<
    hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
    int8_t, hytlass::layout::ColumnMajor,
    int8_t, hytlass::layout::ColumnMajor,
    int32_t, hytlass::layout::ColumnMajor,
    int32_t>;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_Gemm_s8n_s8t_s32n_simt_s32, 128x128x64_64x64x64) {
  using Config = hytlass::gemm::device::DefaultGemmConfigurationToHytlass3Types<
    hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
    int8_t, hytlass::layout::ColumnMajor,
    int8_t, hytlass::layout::RowMajor,
    int32_t, hytlass::layout::ColumnMajor,
    int32_t>;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_Gemm_s8t_s8n_s32n_simt_s32, 128x128x64_64x64x64) {
  using Config = hytlass::gemm::device::DefaultGemmConfigurationToHytlass3Types<
    hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
    int8_t, hytlass::layout::RowMajor,
    int8_t, hytlass::layout::ColumnMajor,
    int32_t, hytlass::layout::ColumnMajor,
    int32_t>;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX906_Device_Gemm_s8t_s8t_s32n_simt_s32, 128x128x64_64x64x64) {
  using Config = hytlass::gemm::device::DefaultGemmConfigurationToHytlass3Types<
    hytlass::arch::OpClassSimt, hytlass::arch::Gfx906,
    int8_t, hytlass::layout::RowMajor,
    int8_t, hytlass::layout::RowMajor,
    int32_t, hytlass::layout::ColumnMajor,
    int32_t>;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      Config::CollectiveMainloop,
      Config::CollectiveEpilogue
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>());
}

/////////////////////////////////////////////////////////////////////////////////////////////////

//#endif // #if defined(HYTLASS_ARCH_MMA_GFX906_SUPPORTED)

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
    \brief Tests for device-wide GEMM interface with stream-K scheduling
*/

#include <iostream>

#include "hytlass/hytlass.h"
#include "hute/tensor.hpp"
#include "hute/atom/mma_atom.hpp"

#include "hytlass/numeric_types.h"

#include "hytlass/gemm/device/gemm_universal_adapter.h"
#include "hytlass/gemm/kernel/gemm_universal.hpp"
#include "hytlass/gemm/kernel/tile_scheduler.hpp"
#include "hytlass/gemm/collective/collective_builder.hpp"
#include "hytlass/epilogue/collective/collective_builder.hpp"
#include "hytlass/epilogue/collective/gfx928_epilogue_vectorized.hpp"
#include "hytlass/epilogue/collective/default_epilogue.hpp"
#include "hytlass/epilogue/thread/linear_combination.h"

#include "../../common/hytlass_unit_test.h"

#include "gemm_testbed_3x.hpp"

using namespace hute;

TEST(GFX928_Device_Gemm_f16t_f16t_f16n_tensor_op_f32_stream_k, 128x128x64) {
  using ElementA = hytlass::half_t;
  using LayoutA = hytlass::layout::RowMajor;
  using ElementB = hytlass::half_t;
  using LayoutB = hytlass::layout::RowMajor;
  using ElementAccumulator = float;
  using LayoutC = hytlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using WarpShape_MNK = Shape<_2, _2, _1>;
  using InstructionShape_MNK = Shape<_32, _32, _16>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      ElementA, LayoutA, 8,
      ElementB, LayoutB, 8,
      ElementAccumulator,
      TileShape_MNK, WarpShape_MNK, 
      InstructionShape_MNK, ClusterShape_MNK,
      hytlass::gemm::collective::StageCount<2>,
      hytlass::gemm::KernelStreamKSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      hytlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      hytlass::half_t, LayoutC, 8,
      hytlass::half_t, LayoutC, 8,
      hytlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      hytlass::gemm::StreamKScheduler
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Gemm_f16t_f16n_f16n_tensor_op_f32_stream_k, 256x128x64) {
  using LayoutA = hytlass::layout::RowMajor;
  using LayoutB = hytlass::layout::ColumnMajor;
  using LayoutC = hytlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using WarpShape_MNK = Shape<_2, _2, _1>;
  using InstructionShape_MNK = Shape<_32, _32, _16>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      hytlass::half_t, LayoutA, 8,
      hytlass::half_t, LayoutB, 8,
      float,
      TileShape_MNK, WarpShape_MNK, 
      InstructionShape_MNK, ClusterShape_MNK,
      hytlass::gemm::collective::StageCount<2>,
      hytlass::gemm::KernelStreamKSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      hytlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      hytlass::half_t, LayoutC, 8,
      hytlass::half_t, LayoutC, 8,
      hytlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      hytlass::gemm::StreamKScheduler
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Gemm_f16n_f16t_f16n_tensor_op_f32_stream_k, 128x128x64) {
  using LayoutA = hytlass::layout::ColumnMajor;
  using LayoutB = hytlass::layout::RowMajor;
  using LayoutC = hytlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_128,_128,_64>;
  using WarpShape_MNK = Shape<_2, _2, _1>;
  using InstructionShape_MNK = Shape<_32, _32, _16>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      hytlass::half_t, LayoutA, 8,
      hytlass::half_t, LayoutB, 8,
      float,
      TileShape_MNK, WarpShape_MNK, 
      InstructionShape_MNK, ClusterShape_MNK,
      hytlass::gemm::collective::StageCount<2>,
      hytlass::gemm::KernelStreamKSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      hytlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      hytlass::half_t, LayoutC, 8,
      hytlass::half_t, LayoutC, 8,
      hytlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      hytlass::gemm::StreamKScheduler
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 1.0));
}

///////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_Gemm_f16n_f16n_f16n_tensor_op_f32_stream_k, 256x128x64) {
  using LayoutA = hytlass::layout::ColumnMajor;
  using LayoutB = hytlass::layout::ColumnMajor;
  using LayoutC = hytlass::layout::ColumnMajor;
  using TileShape_MNK = Shape<_256,_128,_64>;
  using WarpShape_MNK = Shape<_2, _2, _1>;
  using InstructionShape_MNK = Shape<_32, _32, _16>;
  using ClusterShape_MNK = Shape<_1,_1,_1>;

  using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      hytlass::half_t, LayoutA, 8,
      hytlass::half_t, LayoutB, 8,
      float,
      TileShape_MNK, WarpShape_MNK, 
      InstructionShape_MNK, ClusterShape_MNK,
      hytlass::gemm::collective::StageCount<2>,
      hytlass::gemm::KernelStreamKSpecialized
    >::CollectiveOp;

  using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
      hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
      TileShape_MNK, ClusterShape_MNK,
      hytlass::epilogue::collective::EpilogueTileAuto,
      float, float,
      hytlass::half_t, LayoutC, 8,
      hytlass::half_t, LayoutC, 8,
      hytlass::epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

  using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
      Shape<int,int,int,int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      hytlass::gemm::StreamKScheduler
  >;

  using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 0.0));
  EXPECT_TRUE(test::gemm::device::TestAll<Gemm>(1.0, 1.0));
}


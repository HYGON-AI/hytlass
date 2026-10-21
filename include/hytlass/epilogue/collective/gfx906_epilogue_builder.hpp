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
#pragma once

#include "hytlass/detail/dependent_false.hpp"
#include "hytlass/detail/layout.hpp"
#include "hytlass/epilogue/dispatch_policy.hpp"
#include "hytlass/epilogue/collective/collective_epilogue.hpp"
#include "hytlass/epilogue/thread/linear_combination.h"
#include "hytlass/epilogue/thread/linear_combination_generic.h"
#include "hytlass/epilogue/thread/linear_combination_bias_elementwise.h"

#include <type_traits>

///////////////////////////////////////////////////////////////////////////////

namespace hytlass::epilogue::collective {

// Auto builder
template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveBuilder<
    arch::Gfx906,
    arch::OpClassSimt,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    void> {
  using CollectiveOp = hytlass::epilogue::collective::DefaultEpilogue<
    hytlass::detail::TagToStrideC_t<GmemLayoutTagC>,
    hytlass::detail::TagToStrideC_t<GmemLayoutTagD>,
    thread::LinearCombination<ElementC, AlignmentC, ElementAccumulator, ElementCompute>,
    hytlass::gemm::EpilogueDefault>;
};
///////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::epilogue::collective

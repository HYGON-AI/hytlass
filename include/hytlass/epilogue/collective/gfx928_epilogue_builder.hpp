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

///////////////////////////////////////////////////////////////////////////////

namespace detail {




} // namespace detail

///////////////////////////////////////////////////////////////////////////////
// No-smem builder
template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueSchedule,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Gfx928,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    hute::enable_if_t<hute::is_same_v<EpilogueSchedule, NoSmemWarpSpecialized> ||
                      hute::is_same_v<EpilogueSchedule, PtrArrayNoSmemWarpSpecialized> >> {

    // Passing void C disables source load
    using ElementC = hute::conditional_t<hute::is_void_v<ElementC_>,
        ElementD, ElementC_>; // prevents hute breakages
    using GmemLayoutTagC = hute::conditional_t<hute::is_void_v<ElementC_>,
        GmemLayoutTagD, GmemLayoutTagC_>;
    static constexpr thread::ScaleType::Kind ScaleType = hute::is_void_v<ElementC_> ?
        thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

    static constexpr int FragmentSize = 1;
    using ThreadOp = thread::LinearCombination<
      ElementD, FragmentSize, ElementAccumulator, ElementCompute,
      ScaleType, RoundStyle, ElementC>;

    using CollectiveOp = hute::conditional_t<
      hute::is_same_v<EpilogueSchedule, NoSmemWarpSpecialized>,
      hytlass::epilogue::collective::DefaultEpilogue<
                              hytlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                              hytlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                              ThreadOp,
                              hytlass::gemm::EpilogueDefault>,
      hytlass::epilogue::collective::DefaultEpilogueArray<
                              hytlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                              hytlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                              ThreadOp,
                              EpilogueSchedule>>;

};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class EpilogueSchedule,
  FloatRoundStyle RoundStyle
>
struct CollectiveBuilder<
    arch::Gfx928,
    arch::OpClassTensorOp,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    hute::enable_if_t<hute::is_same_v<EpilogueSchedule, VectorSpecialized> ||
                      hute::is_same_v<EpilogueSchedule, PtrArrayVectorSpecialized> >> {
  // Passing void C disables source load
  using ElementC = hute::conditional_t<hute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents hute breakages
  using GmemLayoutTagC = hute::conditional_t<hute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = hute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  // For vectorized situation, the definitive alignment is determined by min(R->G, S->R)
  static constexpr int AdaptedAlignmentD = hute::min(AlignmentD, 128 / sizeof_bits_v<ElementAccumulator>);

  using AlignmentTypeD = hute::uint_byte_t<static_cast<int>(sizeof(ElementD)) * AdaptedAlignmentD>;
  static_assert(sizeof_bits_v<AlignmentTypeD> <= 128);

  using ThreadOp = thread::LinearCombination<
        ElementD, AdaptedAlignmentD, ElementAccumulator, ElementCompute,
        ScaleType, RoundStyle, ElementC>;

  // This interface has no MMA parameters; defer these types until the pipeline
  // can use the GEMM collective builder's MMA.
  
  using SmemLayoutC = void;

  using TiledCopyS2R = void;

  using CollectiveOp = hute::conditional_t<
      hute::is_same_v<EpilogueSchedule, VectorSpecialized>,
      hytlass::epilogue::collective::Epilogue<
                            hytlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                            hytlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                            ThreadOp,
                            SmemLayoutC,   
                            Copy_Atom<DefaultCopy,ElementAccumulator>,                                        // R2S with tiled_mma layout
                            TiledCopyS2R,                     // Value layout
                            Copy_Atom<UniversalCopy<AlignmentTypeD>, ElementD>                                    // R2G with S2R_dst layout
                            >,
      hytlass::epilogue::collective::EpilogueArray<
                            hytlass::detail::TagToStrideC_t<GmemLayoutTagC>,
                            hytlass::detail::TagToStrideC_t<GmemLayoutTagD>,
                            ThreadOp,
                            SmemLayoutC,   
                            Copy_Atom<DefaultCopy,ElementAccumulator>,                                        // R2S with tiled_mma layout
                            TiledCopyS2R,                     // Value layout
                            Copy_Atom<UniversalCopy<AlignmentTypeD>, ElementD>
                            >>;
};
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
    arch::Gfx928,
    arch::OpClassTensorOp,
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
private:
  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance) 
  // since TMA epilogues are not compatible with non-TMA non-WS mainloops
  using EpilogueSchedule = VectorSpecialized;
  using _CollectiveBuilder = CollectiveBuilder<
    arch::Gfx928,
    arch::OpClassTensorOp,
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
    EpilogueSchedule,
    FusionOperation
  >;

public:
  using CollectiveOp = typename _CollectiveBuilder::CollectiveOp;
};
///////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::epilogue::collective

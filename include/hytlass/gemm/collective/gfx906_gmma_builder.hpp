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

#include "hytlass/arch/mma.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/gemm/collective/mma_twostage.hpp"

#include "hute/atom/mma_atom.hpp"
#include "hute/atom/copy_atom.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
namespace detail_gfx906 {

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  auto... Args                       
>
HUTE_HOST_DEVICE constexpr
auto
dot_op_selector() {
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {
    static_assert(is_same_v<ElementA, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(is_same_v<ElementB, half_t>, "Element types for AB must be half if ElementC is half.");
    return UniversalFMA<half_t>{};
  }

  // FP32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {
    // FP16 inputs
    if constexpr (is_same_v<ElementA, half_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      return GFX906_DP2A{};
    }
    // FP32 inputs
    else if constexpr (is_same_v<ElementA, float>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      return UniversalFMA<float>{};
    } else{
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }
  // int32_t accumulators
  else if constexpr (is_same_v<ElementC, int32_t>) {
    static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
    if constexpr (is_same_v<ElementA, int8_t>) {
        return GFX906_DP4A{};
      } else {
        static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
      }
    }  
  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}

// Generates the most efficient possible TiledCopy with cp  atom given a set of parameters.
template<int ThreadCount, class Element, int Alignment, class StrideType, class TileMN, class TileK>
constexpr auto
make_cp_gmem_tiled_copy() {
  constexpr int TileSizeMN = hute::size(TileMN{});
  constexpr int TileSizeK = hute::size(TileK{});

  constexpr int MaxElementsPerThread = TileSizeMN * TileSizeK / ThreadCount;

  // Maximize the number of threads along the gmem major mode to promote coalesced reads
  // While making sure our thread layout tiles the threadblock tile evenly

  if constexpr (hytlass::gemm::detail::is_k_major<StrideType>()) {
    constexpr int Alignment_ = hute::min(MaxElementsPerThread,Alignment);
    using AlignmentType = hute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment_>;

    // K major thread layout for K major gmem
    constexpr int threads_major = TileSizeK / Alignment_;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeMN % threads_minor == 0));
    return make_tiled_copy(
      Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
      Layout<Shape<Int<threads_minor>, Int<threads_major>>,
             Stride<Int<threads_major>, _1>>{},
      Layout<Shape<_1, Int<Alignment_>>>{});
  } else if constexpr (hytlass::gemm::detail::is_mn_major<StrideType>()) {
    // MN major thread layout for MN major gmem
    static_assert(TileSizeMN * TileSizeK / ThreadCount > 0);
    constexpr int Alignment_ = hute::min(MaxElementsPerThread, Alignment);
    using AlignmentType = hute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment_>;

    constexpr int threads_major = TileSizeMN / Alignment_;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeK % threads_minor == 0));
    return make_tiled_copy(
      Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
      Layout<Shape<Int<threads_major>, Int<threads_minor>>,
             Stride<_1, Int<threads_major>>>{},
      Layout<Shape<Int<Alignment_>, _1>>{});
  } else {
    static_assert(hute::is_void_v<Element>, "Unsupported gmem layout for automatic gmem tiled copy builder.");
  }
}

template <class StrideType, class ElementType, class BLK_MN, class BLK_K>
HUTE_HOST_DEVICE constexpr
auto
tiled_smem_selector()
{
  constexpr int BLK_MN0 = size<0>(BLK_MN{});  
  constexpr int BLK_K0  = size<0>(BLK_K{});

  using BLK_MN_C = hute::C<BLK_MN0>;  
  using BLK_K_C  = hute::C<BLK_K0>;

  if constexpr (is_same_v<ElementType, int8_t>) {
    constexpr int inner_k_v = BLK_K0 / 4;  
    using InnerK_C = hute::C<inner_k_v>;
    using Stride_C = hute::C<BLK_MN0 * 4>;

    return Layout<Shape<BLK_MN_C, Shape<_4, InnerK_C>>, 
                  Stride<_4, Stride<_1, Stride_C>>>{};
  } else if constexpr (is_same_v<ElementType, half_t>) {
    constexpr int inner_k_v = BLK_K0 / 2;
    using InnerK_C = hute::C<inner_k_v>;
    using Stride_C = hute::C<BLK_MN0 * 2>;

    return Layout<Shape<BLK_MN_C, Shape<_2, InnerK_C>>, 
                  Stride<_2, Stride<_1, Stride_C>>>{};
  } else if constexpr (is_same_v<ElementType, float>) {
    return Layout<Shape<BLK_MN_C, BLK_K_C>, Stride<BLK_K_C, _1>>{};
  } else {
    static_assert(sizeof(ElementType) == 0, "Unsupported type to infer SmemLayout.");
  }
}
}
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class WarpShape_MNK,
  class InstructionShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Gfx906,
    arch::OpClassSimt,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    WarpShape_MNK,
    InstructionShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    hute::enable_if_t<
      hute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using TileShape = TileShape_MNK;

  using DispatchPolicy = MainloopDispatch<2, arch::Gfx906, ClusterShape_MNK, KernelScheduleType>;

  using AtomLayoutMNK = Layout<
      Shape<
          hute::C<hute::get<0>(WarpShape_MNK{}) * 8>, 
          hute::C<hute::get<1>(WarpShape_MNK{}) * 8>, 
          decltype(hute::get<2>(WarpShape_MNK{}))
      >>;

  using TiledMma = decltype(hute::make_tiled_mma(detail_gfx906::dot_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK>(), AtomLayoutMNK{}));

  static constexpr uint32_t blockSize = hute::size(TiledMma{});

  // A
  using GmemTiledCopyA = decltype(detail_gfx906::make_cp_gmem_tiled_copy<
      blockSize, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutA>,decltype(hute::get<0>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());
  
  // B
  using GmemTiledCopyB = decltype(detail_gfx906::make_cp_gmem_tiled_copy<
      blockSize, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutB>,decltype(hute::get<1>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail_gfx906::tiled_smem_selector<
      TagToStrideA_t<GmemLayoutA>, ElementA, decltype(hute::get<0>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomB = decltype(detail_gfx906::tiled_smem_selector<
      TagToStrideB_t<GmemLayoutB>, ElementB, decltype(hute::get<1>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());
  
  using SmemCopyAtomA = Copy_Atom<DefaultCopy, ElementA>;

  using SmemCopyAtomB = Copy_Atom<DefaultCopy, ElementB>;
  
  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
      DispatchPolicy, TileShape,
      ElementA,
      TagToStrideA_t<GmemLayoutA>,
      ElementB,
      TagToStrideB_t<GmemLayoutB>,
      TiledMma,
      GmemTiledCopyA, 
      SmemLayoutAtomA, 
      SmemCopyAtomA, 
      hute::identity,  // A
      GmemTiledCopyB, 
      SmemLayoutAtomB, 
      SmemCopyAtomB, 
      hute::identity   // B
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// Auto kernel schedule
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class WarpShape_MNK,
  class InstructionShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Gfx906,
    arch::OpClassSimt,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    WarpShape_MNK,
    InstructionShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    hute::enable_if_t<hute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using CollectiveOp = typename CollectiveBuilder<
      arch::Gfx906,
      arch::OpClassSimt,
      ElementA,
      GmemLayoutA,
      AlignmentA,
      ElementB,
      GmemLayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      WarpShape_MNK,
      InstructionShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelMultistage
    >::CollectiveOp;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////

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
#include "hytlass/detail/layout.hpp"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/gemm/collective/mma_twostage.hpp"

#include "hute/atom/mma_atom.hpp"
#include "hute/atom/copy_atom.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
namespace detail {
using hytlass::detail::GemmLayout;

template <class TileShape_MNK, class InstructionShape_MNK>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_f16_no_alt() {
  constexpr auto Instruction_M = hute::get<0>(InstructionShape_MNK{});
  constexpr auto Instruction_N = hute::get<1>(InstructionShape_MNK{});
  constexpr auto Instruction_K = hute::get<2>(InstructionShape_MNK{});

  if constexpr (Instruction_M == 32 && Instruction_N == 32) {
    if constexpr (Instruction_K == 16) {
      return GFX928_32x32x16_F32F16F16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_32x32x32_F32F16F16F32_NT{};
    }
  } else if constexpr (Instruction_M == 32 && Instruction_N == 16) {
    if constexpr (Instruction_K == 16) {
      return GFX928_32x16x16_F32F16F16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_32x16x32_F32F16F16F32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 32) {
    if constexpr (Instruction_K == 16) {
      return GFX928_16x32x16_F32F16F16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_16x32x32_F32F16F16F32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 16) {
    if constexpr (Instruction_K == 16) {
      return GFX928_16x16x16_F32F16F16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_16x16x32_F32F16F16F32_NT{};
    }
  } else {
    static_assert(Instruction_M < 0, "Unreachable mmac selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <class TileShape_MNK, class InstructionShape_MNK>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_bf16_no_alt() {
  constexpr auto Instruction_M = hute::get<0>(InstructionShape_MNK{});
  constexpr auto Instruction_N = hute::get<1>(InstructionShape_MNK{});
  constexpr auto Instruction_K = hute::get<2>(InstructionShape_MNK{});

  if constexpr (Instruction_M == 32 && Instruction_N == 32) {
    if constexpr (Instruction_K == 16) {
      return GFX928_32x32x16_F32BF16BF16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_32x32x32_F32BF16BF16F32_NT{};
    }
  } else if constexpr (Instruction_M == 32 && Instruction_N == 16) {
    if constexpr (Instruction_K == 16) {
      return GFX928_32x16x16_F32BF16BF16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_32x16x32_F32BF16BF16F32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 32) {
    if constexpr (Instruction_K == 16) {
      return GFX928_16x32x16_F32BF16BF16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_16x32x32_F32BF16BF16F32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 16) {
    if constexpr (Instruction_K == 16) {
      return GFX928_16x16x16_F32BF16BF16F32_NT{};
    } else if constexpr (Instruction_K == 32) {
      return GFX928_16x16x32_F32BF16BF16F32_NT{};
    }
  } else {
    static_assert(Instruction_M < 0, "Unreachable mmac selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <class TileShape_MNK, class InstructionShape_MNK, GemmLayout Layout>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_f16() {
  if constexpr (Layout == detail::GemmLayout::NT) {
    return GFX928_32x32x16_F32F16F16F32_NT_ALT{};
  } else {
    return mmac_selector_f16_no_alt<TileShape_MNK, InstructionShape_MNK>();
  }
  HUTE_GCC_UNREACHABLE;
}

template <class TileShape_MNK, class InstructionShape_MNK, GemmLayout Layout>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_bf16() {
  if constexpr (Layout == detail::GemmLayout::NT) {
    return GFX928_32x32x16_F32BF16BF16F32_NT_ALT{};
  } else {
    return mmac_selector_bf16_no_alt<TileShape_MNK, InstructionShape_MNK>();
  }
  HUTE_GCC_UNREACHABLE;
}

template <class TileShape_MNK, class InstructionShape_MNK, GemmLayout Layout>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_u8() {
  constexpr auto Instruction_M = hute::get<0>(InstructionShape_MNK{});
  constexpr auto Instruction_N = hute::get<1>(InstructionShape_MNK{});
  constexpr auto Instruction_K = hute::get<2>(InstructionShape_MNK{});

  if constexpr (Layout == detail::GemmLayout::NT) {
    return GFX928_32x32x32_I32U8U8I32_NT{};
  }

  if constexpr (Instruction_M == 32 && Instruction_N == 32) {
    if constexpr (Instruction_K == 32) {
      return GFX928_32x32x32_I32U8U8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_32x32x64_I32U8U8I32_NT{};
    }
  } else if constexpr (Instruction_M == 32 && Instruction_N == 16) {
    if constexpr (Instruction_K == 32) {
      return GFX928_32x16x32_I32U8U8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_32x16x64_I32U8U8I32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 32) {
    if constexpr (Instruction_K == 32) {
      return GFX928_16x32x32_I32U8U8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_16x32x64_I32U8U8I32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 16) {
    if constexpr (Instruction_K == 32) {
      return GFX928_16x16x32_I32U8U8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_16x16x64_I32U8U8I32_NT{};
    }
  } else {
    static_assert(Instruction_M < 0, "Unreachable mmac selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <class TileShape_MNK, class InstructionShape_MNK, GemmLayout Layout>
HUTE_HOST_DEVICE constexpr
auto mmac_selector_i8() {
  constexpr auto Instruction_M = hute::get<0>(InstructionShape_MNK{});
  constexpr auto Instruction_N = hute::get<1>(InstructionShape_MNK{});
  constexpr auto Instruction_K = hute::get<2>(InstructionShape_MNK{});

  if constexpr (Layout == detail::GemmLayout::NT) {
    return GFX928_32x32x32_I32I8I8I32_NT{};
  }

  if constexpr (Instruction_M == 32 && Instruction_N == 32) {
    if constexpr (Instruction_K == 32) {
      return GFX928_32x32x32_I32I8I8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_32x32x64_I32I8I8I32_NT{};
    }
  } else if constexpr (Instruction_M == 32 && Instruction_N == 16) {
    if constexpr (Instruction_K == 32) {
      return GFX928_32x16x32_I32I8I8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_32x16x64_I32I8I8I32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 32) {
    if constexpr (Instruction_K == 32) {
      return GFX928_16x32x32_I32I8I8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_16x32x64_I32I8I8I32_NT{};
    }
  } else if constexpr (Instruction_M == 16 && Instruction_N == 16) {
    if constexpr (Instruction_K == 32) {
      return GFX928_16x16x32_I32I8I8I32_NT{};
    } else if constexpr (Instruction_K == 64) {
      return GFX928_16x16x64_I32I8I8I32_NT{};
    }
  } else {
    static_assert(Instruction_M < 0, "Unreachable mmac selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <class InstructionShape_MN, class InstructionShape_K, bool kMNMajor>
HUTE_HOST_DEVICE constexpr
auto ds_read_selector_f16_no_alt() {
  constexpr auto Instruction_MN = InstructionShape_MN{};
  constexpr auto Instruction_K = InstructionShape_K{};

  if constexpr (kMNMajor) {
    static_assert(Instruction_MN % 32 == 0, "MN-major Instruction M must be multiple of ds_read_m shape(32).");
    static_assert(Instruction_K == 16, "MN-major Instruction K must be 16.");
    return GFX928_DS_READ_DS_M32x16_B16{};
  } else if constexpr (Instruction_K == 32) {
    return UniversalCopy<uint128_t>{};
  } else if constexpr (Instruction_K == 16) {
    return UniversalCopy<uint64_t>{};
  } else {
    static_assert(Instruction_MN < 0, "Unreachable ds_read selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <class InstructionShape_MN, class InstructionShape_K, GemmLayout Layout, bool kMNMajor>
HUTE_HOST_DEVICE constexpr
auto ds_read_selector_f16() {
  if constexpr (Layout == detail::GemmLayout::NT) {
    return GFX928_DS_READ_DS_M32x16_B16_ALT{};
  } else {
    return ds_read_selector_f16_no_alt<InstructionShape_MN, InstructionShape_K, kMNMajor>();
  }
  HUTE_GCC_UNREACHABLE;
}

template <class InstructionShape_MN, class InstructionShape_K, GemmLayout Layout, bool kMNMajor>
HUTE_HOST_DEVICE constexpr
auto ds_read_selector_i8() {
  constexpr auto Instruction_MN = InstructionShape_MN{};
  constexpr auto Instruction_K = InstructionShape_K{};

  if constexpr (kMNMajor) {
    static_assert(Instruction_MN % 32 == 0, "MN-major Instruction M must be multiple of ds_read_m shape(32).");
    static_assert(Instruction_K == 32, "MN-major Instruction K must be 32.");
    return GFX928_DS_READ_DS_M32x32_B8{};
  } else if constexpr (Instruction_K == 64) {
    return UniversalCopy<uint128_t>{};
  } else if constexpr (Instruction_K == 32) {
    return UniversalCopy<uint64_t>{};
  } else {
    static_assert(Instruction_MN < 0, "Unreachable ds_read selector.");
  }
  HUTE_GCC_UNREACHABLE;
}

template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  class InstructionShape_MNK,
  GemmLayout Layout,
  auto... Args
>
HUTE_HOST_DEVICE constexpr
auto
mmac_op_selector() {
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  static_assert(size<0>(TileShape_MNK{}) % 32 == 0, "Tile_M must be a multiple of 32.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {
    static_assert(is_same_v<ElementA, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(is_same_v<ElementB, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
  }

  // FP32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {
    // FP16 inputs
    if constexpr (is_same_v<ElementA, half_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
      return mmac_selector_f16<TileShape_MNK, InstructionShape_MNK, Layout>();
    }

    // BF16 inputs
    else if constexpr (is_same_v<ElementA, bfloat16_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
      return mmac_selector_bf16<TileShape_MNK, InstructionShape_MNK, Layout>();
    }

    // TF32 inputs
    else if constexpr (is_same_v<ElementA, tfloat32_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");
      if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x8_F32TF32TF32F32_NT{};
      }
    }
    else if constexpr (is_same_v<ElementA, float>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 8 == 0, "Tile_K must be a multiple of 8.");
      if constexpr (Tile_N % 16 == 0) {
        return GFX928_16x16x8_F32F32F32F32_NT{};
      }
    } else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }
  // int32_t accumulators
  else if constexpr (is_same_v<ElementC, int32_t>) {
    static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
    static_assert(size<2>(TileShape_MNK{}) % 32 == 0, "Tile_K must be a multiple of 32.");
    if constexpr (is_same_v<ElementA, int8_t>) {
      return mmac_selector_i8<TileShape_MNK, InstructionShape_MNK, Layout>();
    } else if constexpr (is_same_v<ElementA, uint8_t>) {
      return mmac_selector_u8<TileShape_MNK, InstructionShape_MNK, Layout>();
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

template<class BLK_MN, class BLK_K, class InstructionShape_MN, class InstructionShape_K, bool MN_MAJOR>
HUTE_HOST_DEVICE constexpr
auto
smem_layout_selector_f16() {
  if constexpr (MN_MAJOR) {
    if constexpr (BLK_MN{} % 64 == 0) {
      return composition(Swizzle<3, 3, 3>{}, Layout<Shape<_64, _16>, Stride<_1, _64>>{});
    } else if constexpr (BLK_MN{} % 32 == 0) {
      return composition(Swizzle<2, 3, 3>{}, Layout<Shape<_32, _16>, Stride<_1, _32>>{});
    }
  } else {
    static_assert(BLK_K{} <= 64, "Hute swizzle params require tile_K <= 64.");
    if constexpr (InstructionShape_K{} == 32) {
      static_assert(BLK_K{} <= 32, "Hute swizzle params require tile_K <= 32 while instruction_K == 32.");
      constexpr uint32_t k_row = 64 / BLK_K{};
      constexpr uint32_t k_col = BLK_K{} / 4;
      constexpr uint32_t k_sw = bit_width(k_col) - 1;
      return composition(Swizzle<k_sw, 3, 3>{}, Layout<Shape<Shape<Int<k_row>, Int<k_col>>, BLK_K>, Stride<Stride<BLK_K, _64>, _1>>{});
    } else if constexpr (InstructionShape_K{} == 16) {
      constexpr uint32_t k_row = 64 / BLK_K{};
      constexpr uint32_t k_col = BLK_K{} / 4;
      constexpr uint32_t k_sw = bit_width(k_col) - 1;
      return composition(Swizzle<k_sw, 2, 4>{}, Layout<Shape<Shape<Int<k_row>, Int<k_col>>, BLK_K>, Stride<Stride<BLK_K, _64>, _1>>{});
    }
  }
}

template<class BLK_MN, class BLK_K, class InstructionShape_MN, class InstructionShape_K, bool MN_MAJOR>
HUTE_HOST_DEVICE constexpr
auto
smem_layout_selector_i8() {
  if constexpr (MN_MAJOR) {
    if constexpr (BLK_MN{} % 64 == 0) {
      return composition(Swizzle<3, 4, 3>{}, Layout<Shape<_64, _32>, Stride<_1, _64>>{});
    } else if constexpr (BLK_MN{} % 32 == 0) {
      return composition(Swizzle<2, 4, 3>{}, Layout<Shape<_32, _32>, Stride<_1, _32>>{});
    }
  } else {
    static_assert(BLK_K{} <= 128, "Hute swizzle params require tile_K <= 64.");
    if constexpr (InstructionShape_K{} == 64) {
      static_assert(BLK_K{} <= 64, "Hute swizzle params require tile_K <= 64 while instruction_K == 64.");
      constexpr uint32_t k_row = 128 / BLK_K{};
      constexpr uint32_t k_col = BLK_K{} / 8;
      constexpr uint32_t k_sw = bit_width(k_col) - 1;
      return composition(Swizzle<k_sw, 4, 3>{}, Layout<Shape<Shape<Int<k_row>, Int<k_col>>, BLK_K>, Stride<Stride<BLK_K, _128>, _1>>{});
    } else if constexpr (InstructionShape_K{} == 32) {
      constexpr uint32_t k_row = 128 / BLK_K{};
      constexpr uint32_t k_col = BLK_K{} / 8;
      constexpr uint32_t k_sw = bit_width(k_col) - 1;
      return composition(Swizzle<k_sw, 3, 4>{}, Layout<Shape<Shape<Int<k_row>, Int<k_col>>, BLK_K>, Stride<Stride<BLK_K, _128>, _1>>{});
    }
  }
}

template <
  class ElementType,
  class BLK_MN,
  class BLK_K,
  class InstructionShape_MN,
  class InstructionShape_K,
  bool MN_MAJOR>
HUTE_HOST_DEVICE constexpr
auto
tiled_smem_selector()
{
  constexpr auto BLK_MN0 = size<0>(BLK_MN{});
  constexpr auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 32 == 0, "BLK_MN0 must be a multiple of 32.");
  if constexpr (is_same_v<ElementType, float> || is_same_v<ElementType, tfloat32_t>) {
    if constexpr (BLK_MN0 % 32 == 0) {
      return composition(Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{});
    }
  }
  else if constexpr (is_same_v<ElementType, half_t> || is_same_v<ElementType, bfloat16_t>) {
    return smem_layout_selector_f16<BLK_MN, BLK_K, InstructionShape_MN, InstructionShape_K, MN_MAJOR>();
  }
  else if constexpr (is_same_v<ElementType, int8_t> || is_same_v<ElementType, uint8_t>) {
    return smem_layout_selector_i8<BLK_MN, BLK_K, InstructionShape_MN, InstructionShape_K, MN_MAJOR>();
  } else {
    static_assert(sizeof(ElementType) == 0, "Unsupported type to inference SmemLayout.");
  }
}

template <class ElementType, class InstructionShape_MN, class InstructionShape_K, GemmLayout Layout, bool kMNMajor>
HUTE_HOST_DEVICE constexpr
auto
ds_read_selector() {
  if constexpr (is_same_v<ElementType, half_t> || is_same_v<ElementType, bfloat16_t>) {
    return ds_read_selector_f16<InstructionShape_MN, InstructionShape_K, Layout, kMNMajor>();
  } else if constexpr (is_same_v<ElementType, float> || is_same_v<ElementType, tfloat32_t>) {
    return UniversalCopy<float>{};
  } else if constexpr (is_same_v<ElementType, int8_t> || is_same_v<ElementType, uint8_t>) {
    return ds_read_selector_i8<InstructionShape_MN, InstructionShape_K, Layout, kMNMajor>();
  } else {
    static_assert(sizeof(ElementType) == 0, "Unsupported ds_read_matrix data type.");
  }
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

// MainloopDispatch
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
    arch::Gfx928,
    arch::OpClassTensorOp,
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
      hute::is_same_v<KernelScheduleType, KernelMultistage> ||
      hute::is_same_v<KernelScheduleType, KernelStreamKSpecialized> ||
      hute::is_same_v<KernelScheduleType, KernelSplitkParallelSpecialized> ||
      hute::is_same_v<KernelScheduleType, KernelPtrArraySpecialized>>> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using TileShape = TileShape_MNK;

  static constexpr bool IsArrayOfPointersGemm = (hute::is_same_v<KernelScheduleType, KernelPtrArraySpecialized>);

  using DispatchPolicy = hute::conditional_t<IsArrayOfPointersGemm,
    MainloopDispatchPtrArray<StageCountType::value, arch::Gfx928, ClusterShape_MNK, KernelScheduleType>,
    MainloopDispatch<StageCountType::value, arch::Gfx928, ClusterShape_MNK, KernelScheduleType>>;

  using AtomLayoutMNK = Layout<Shape<decltype(hute::get<0>(WarpShape_MNK{})),decltype(hute::get<1>(WarpShape_MNK{})),decltype(hute::get<2>(WarpShape_MNK{}))>>;

  static constexpr auto Instruction_M = hute::get<0>(InstructionShape_MNK{});
  static constexpr auto Instruction_N = hute::get<1>(InstructionShape_MNK{});
  static constexpr auto Instruction_K = hute::get<2>(InstructionShape_MNK{});
  static_assert(hute::get<0>(TileShape_MNK{}) % Instruction_M == 0, "Tile M must be a multiple of Instruction M.");
  static_assert(hute::get<1>(TileShape_MNK{}) % Instruction_N == 0, "Tile N must be a multiple of Instruction N.");
  static_assert(hute::get<2>(TileShape_MNK{}) % Instruction_K == 0, "Tile K must be a multiple of Instruction K.");

  static constexpr bool MN_MajorA = hytlass::gemm::detail::is_mn_major_A<GmemLayoutA>();
  static constexpr bool MN_MajorB = hytlass::gemm::detail::is_mn_major_B<GmemLayoutB>();

  using UnderlyingStrideA = hute::remove_pointer_t<TagToStrideA_t<GmemLayoutA>>;
  using UnderlyingStrideB = hute::remove_pointer_t<TagToStrideB_t<GmemLayoutB>>;
  static constexpr bool IsGroupedGemmKernel = !hute::is_same_v<UnderlyingStrideA, TagToStrideA_t<GmemLayoutA>>;  
  static constexpr auto GemmLayoutType =
      hytlass::detail::get_gemm_layout<TagToStrideA_t<GmemLayoutA>, TagToStrideB_t<GmemLayoutB>, IsGroupedGemmKernel>();

  using TiledMma = decltype(hute::make_tiled_mma(detail::mmac_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK, InstructionShape_MNK, GemmLayoutType>(), AtomLayoutMNK{}));

  static constexpr uint32_t blockSize = hute::size(TiledMma{});

  // A
  using GmemTiledCopyA = decltype(detail::make_cp_gmem_tiled_copy<
      blockSize, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutA>,decltype(hute::get<0>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());
  // B
  using GmemTiledCopyB = decltype(detail::make_cp_gmem_tiled_copy<
      blockSize, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutB>,decltype(hute::get<1>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::tiled_smem_selector<
    ElementA, decltype(hute::get<0>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{})), decltype(Instruction_M),
    decltype(Instruction_K), MN_MajorA>());
  using SmemLayoutAtomB = decltype(detail::tiled_smem_selector<
    ElementB, decltype(hute::get<1>(TileShape_MNK{})), decltype(hute::get<2>(TileShape_MNK{})), decltype(Instruction_N),
    decltype(Instruction_K), MN_MajorB>());

  // GFX928_DS_READ_DS_M32x16_B16_ALT only support M/N major
  using SmemCopyAtomA = Copy_Atom<decltype(detail::ds_read_selector<
                                    ElementA, decltype(Instruction_M), decltype(Instruction_K), GemmLayoutType,
                                    MN_MajorA>()), ElementA>;
  using SmemCopyAtomB = Copy_Atom<decltype(detail::ds_read_selector<
                                    ElementB, decltype(Instruction_N), decltype(Instruction_K), GemmLayoutType,
                                    MN_MajorB>()), ElementB>;

  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
      DispatchPolicy,
      TileShape,
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
    arch::Gfx928,
    arch::OpClassTensorOp,
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
      arch::Gfx928,
      arch::OpClassTensorOp,
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

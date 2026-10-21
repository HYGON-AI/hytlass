/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "hytlass/layout/matrix.h"
#include "hytlass/layout/tensor.h"
#include "hytlass/numeric_types.h"

#include "hute/layout.hpp"
#include "hute/util/type_traits.hpp"
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////
// For each hytlass::layout, provides its corresponding hute stride types, 64b by default

template <class L>
struct TagToStrideA {
  using type = L;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::RowMajor> {
  using type = hute::Stride<int64_t, hute::Int<1>, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::ColumnMajor> {
  using type = hute::Stride<hute::Int<1>, int64_t, int64_t>;
  using tag = layout::ColumnMajor;
};

template <class L>
struct TagToStrideB {
  using type = L;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::RowMajor> {
  using type = hute::Stride<hute::Int<1>, int64_t, int64_t>;
  using tag = layout::RowMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::ColumnMajor> {
  using type = hute::Stride<int64_t, hute::Int<1>, int64_t>;
  using tag = layout::ColumnMajor;
};

// For each hytlass::layout *, provides its corresponding hute stride types, 64b by default
// Used by pointer array and grouped gemm
// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::RowMajor *> {
  using UnderlyingType = hute::Stride<int64_t, hute::Int<1>, hute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::RowMajor;
};

// Maps to modes [M, K, L]
template <>
struct TagToStrideA<layout::ColumnMajor *> {
  using UnderlyingType = hute::Stride<hute::Int<1>, int64_t, hute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::ColumnMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::RowMajor *> {
  using UnderlyingType = hute::Stride<hute::Int<1>, int64_t, hute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::RowMajor;
};

// Maps to modes [N, K, L]
template <>
struct TagToStrideB<layout::ColumnMajor *> {
  using UnderlyingType = hute::Stride<int64_t, hute::Int<1>, hute::Int<0>>;
  using type = UnderlyingType*;
  using tag = layout::ColumnMajor;
};

// Maps to modes [M, N, L]
template <class LayoutTag>
struct TagToStrideC : TagToStrideA<LayoutTag> { };

// Conv: Maps to modes ((P,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorNWC> {
  using type = hute::Stride<hute::Stride<int64_t, int64_t>, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes (PN, C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorLinearizedNWC> {
  using type = hute::Stride<int64_t, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes ((P,Q,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorNHWC> {
  using type = hute::Stride<hute::Stride<int64_t, int64_t, int64_t>, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes (PQN, C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorLinearizedNHWC> {
  using type = hute::Stride<int64_t, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes ((P,Q,Z,N), C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorNDHWC> {
  using type = hute::Stride<hute::Stride<int64_t, int64_t, int64_t, int64_t>, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes (PQZN, C, _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorLinearizedNDHWC> {
  using type = hute::Stride<int64_t, hute::Int<1>, hute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorKCS> {
  using type = hute::Stride<int64_t, hute::Stride<hute::Int<1>, int64_t>, hute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S,R), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorKCSR> {
  using type = hute::Stride<int64_t, hute::Stride<hute::Int<1>, int64_t, int64_t>, hute::Int<0>>;
};

// Conv: Maps to modes (K, (C,S,R,T), _0) for compatiblity with GEMM epilogues expecting a batch mode stride
template <>
struct TagToStrideC<hytlass::layout::TensorKCSRT> {
  using type = hute::Stride<int64_t, hute::Stride<hute::Int<1>, int64_t, int64_t, int64_t>, hute::Int<0>>;
};

// Convenience aliases
template<class LayoutTag>
using TagToStrideA_t = typename TagToStrideA<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideB_t = typename TagToStrideB<LayoutTag>::type;

template<class LayoutTag>
using TagToStrideC_t = typename TagToStrideC<LayoutTag>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////
// For 2.x compatibility APIs, provide stride->layout tag mappers

template<int ModeIndex, class Stride>
constexpr bool
is_major(Stride = {}) {
  // Account for stride types with and without batch mode and batch modes with static zero stride
  return hute::is_constant<1, decltype(hute::front(hute::get<ModeIndex>(hute::remove_pointer_t<Stride>{})))>::value;
}

// Note : This method can be used for deducing the Layout Tag of A, C, D Matrices
template<class StrideA>
constexpr
auto
stride_to_layout_tag_A() {
  if constexpr (is_major<0, StrideA>()) { // M major
    return layout::ColumnMajor{};
  }
  else { // K major
    return layout::RowMajor{};
  }

  HUTE_GCC_UNREACHABLE;
}

template<class StrideB>
constexpr
auto
stride_to_layout_tag_B() {
  if constexpr (is_major<0, StrideB>()) { // N major
    return layout::RowMajor{};
  }
  else { // K major
    return layout::ColumnMajor{};
  }

  HUTE_GCC_UNREACHABLE;
}

template<class StrideC>
constexpr
auto
stride_to_layout_tag_C() {
  if constexpr (is_major<0, StrideC>()) { // M major
    return layout::ColumnMajor{};
  }
  else { // N major
    return layout::RowMajor{};
  }

  HUTE_GCC_UNREACHABLE;
}

// Utilities to map Stride back on to their corresponding layout tags
template <class S>
struct StrideToLayoutTagA {
  using type = decltype(detail::stride_to_layout_tag_A<S>());
};

template <class S>
struct StrideToLayoutTagB {
  using type = decltype(detail::stride_to_layout_tag_B<S>());
};

template <class S>
struct StrideToLayoutTagC {
  using type = decltype(detail::stride_to_layout_tag_C<S>());
};

// Convenience aliases
template<class S>
using StrideToLayoutTagA_t = typename StrideToLayoutTagA<S>::type;

template<class S>
using StrideToLayoutTagB_t = typename StrideToLayoutTagB<S>::type;

template<class S>
using StrideToLayoutTagC_t = typename StrideToLayoutTagC<S>::type;

enum class GemmLayout { TN, NT, NN, TT };

template <typename StrideA, typename StrideB, bool IsGrouped = false>
constexpr GemmLayout get_gemm_layout() {
  using RowMajorA = typename TagToStrideA<
      std::conditional_t<IsGrouped, layout::RowMajor*, layout::RowMajor>>::type;
  using ColMajorA = typename TagToStrideA<
      std::conditional_t<IsGrouped, layout::ColumnMajor*, layout::ColumnMajor>>::type;
  using RowMajorB = typename TagToStrideB<
      std::conditional_t<IsGrouped, layout::RowMajor*, layout::RowMajor>>::type;
  using ColMajorB = typename TagToStrideB<
      std::conditional_t<IsGrouped, layout::ColumnMajor*, layout::ColumnMajor>>::type;

  if constexpr ((hute::is_same_v<StrideA, RowMajorA>) && (hute::is_same_v<StrideB, ColMajorB>)) {
    return GemmLayout::TN;
  } else if constexpr ((hute::is_same_v<StrideA, ColMajorA>) && (hute::is_same_v<StrideB, RowMajorB>)) {
    return GemmLayout::NT;
  } else if constexpr ((hute::is_same_v<StrideA, ColMajorA>) && (hute::is_same_v<StrideB, ColMajorB>)) {
    return GemmLayout::NN;
  } else if constexpr ((hute::is_same_v<StrideA, RowMajorA>) && (hute::is_same_v<StrideB, RowMajorB>)) {
    return GemmLayout::TT;
  } else {
    static_assert(hute::dependent_false<StrideA>, "Unsupported matrix layout");
  }
  HYTLASS_GCC_UNREACHABLE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <class X, class = void>
struct RawDtype { using type = X; };

template <class X>
struct RawDtype<X,hute::void_t<typename X::raw_type>> { using type = typename X::raw_type; };


// Inspects a TiledCopy and returns its alignment in terms of element count
template <class GmemTiledCopy, class Element, class ElementMma = Element>
constexpr int
get_alignment_count_from_gmem_tiled_copy() {

  if constexpr (hute::is_void_v<GmemTiledCopy>) {
    return 1;
  }

  // Account for ElementC = void kernels
  else if constexpr (hute::is_void_v<Element>) {
    return 0;
  }

  else {
      // TiledCopy holds the alignment count directly in its TiledShape_MN
    return GmemTiledCopy::NumValSrc;
  }
}

// Return the shape that is associated with stride-1 mode, or 1 if not found
template<typename Shape, typename Stride>
HYTLASS_HOST_DEVICE constexpr
auto
get_contiguous_shape(Shape const & shape, Stride const & stride) {
  using namespace hute;
  auto idx = find_if(append(flatten(stride), _1{}), [](auto s){ return is_constant<1,decltype(s)>{}; });
  return get<decltype(idx)::value>(append(flatten(shape), _1{}));
}

// Check if tensor shape satisfies a given major alignment
template<int Alignment, class Shape, class Stride>
HYTLASS_HOST_DEVICE constexpr
bool
check_alignment(Shape const & shape, Stride const & stride) {
  return is_major<0>(stride)
    ? get_contiguous_shape(hute::get<0>(shape), hute::get<0>(stride)) % Alignment == 0
    : get_contiguous_shape(hute::get<1>(shape), hute::get<1>(stride)) % Alignment == 0;
}

// Check if tensor shape satisfies a given major alignment

template<int B, int M, int S>
HYTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(hute::Swizzle<B, M, S>) {
  static_assert(B >= 0 and M >= 0);
  return size_t(1) << size_t(B + M + hute::abs(S));
}

template<class Layout>
HYTLASS_HOST_DEVICE constexpr
size_t
alignment_for_swizzle(Layout layout) {
  return alignment_for_swizzle(hute::detail::get_swizzle_portion(layout));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::detail

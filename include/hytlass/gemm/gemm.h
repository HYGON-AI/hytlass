/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Defines common types used for all GEMM-like operators.
*/
#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/coord.h"
#include "hytlass/gemm_coord.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/gemm/gemm_enumerated_types.h"
#include "hute/layout.hpp"
#include "hytlass/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {


////////////////////////////////////////////////////////////////////////////////////////////////////

using hytlass::detail::TagToStrideA;
using hytlass::detail::TagToStrideB;
using hytlass::detail::TagToStrideC;
using hytlass::detail::TagToStrideA_t;
using hytlass::detail::TagToStrideB_t;
using hytlass::detail::TagToStrideC_t;

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

using hytlass::detail::StrideToLayoutTagA;
using hytlass::detail::StrideToLayoutTagB;
using hytlass::detail::StrideToLayoutTagC;
using hytlass::detail::StrideToLayoutTagA_t;
using hytlass::detail::StrideToLayoutTagB_t;
using hytlass::detail::StrideToLayoutTagC_t;

template<int ModeIndex, class Stride>
constexpr bool
is_major(Stride = {}) {
  return ::hytlass::detail::is_major<ModeIndex, Stride>();
}

template<class Stride>
constexpr bool
is_mn_major() {
  return is_major<0,Stride>();
}

template<class Stride>
constexpr
bool
is_k_major() {
  return is_major<1,Stride>();
}

template<class LayoutA>
constexpr bool
is_mn_major_A() {
  return is_mn_major<TagToStrideA_t<LayoutA>>();
}

template<class LayoutB>
constexpr bool
is_mn_major_B() {
  return is_mn_major<TagToStrideB_t<LayoutB>>();
}

template<class LayoutA>
constexpr bool
is_k_major_A() {
  return is_k_major<TagToStrideA_t<LayoutA>>();
}

template<class LayoutB>
constexpr bool
is_k_major_B() {
  return is_k_major<TagToStrideB_t<LayoutB>>();
}

///////////////////////////////////////////////////////////////////////////////

// The following two metafunctions are used to detect whether a `kernel::Gemm` or `kernel::GemmUniversal`
// is implementing the HYTLASS 3.x API or not, by checking if the problem shape type is aliased within or not.
template <class GemmKernel, class = void>
struct IsHytlass3GemmKernel : hute::false_type { };

template <typename GemmKernel>
struct IsHytlass3GemmKernel<GemmKernel, hute::void_t<typename GemmKernel::ProblemShape>>
    : hute::true_type { };

///////////////////////////////////////////////////////////////////////////////

} // namespace detail

///////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace hytlass

////////////////////////////////////////////////////////////////////////////////////////////////////

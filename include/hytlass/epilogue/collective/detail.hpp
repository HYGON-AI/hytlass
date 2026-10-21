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

#include "hytlass/hytlass.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/epilogue/dispatch_policy.hpp"

#include "hute/tensor.hpp"
#include "hute/numeric/numeric_types.hpp"
#include "hute/util/type_traits.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace epilogue {
namespace collective {

namespace detail {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Stride>
constexpr bool
is_m_major() {
  return hytlass::gemm::detail::is_major<0,Stride>();
}

template <class Stride>
constexpr bool
is_n_major() {
  return hytlass::gemm::detail::is_major<1,Stride>();
}

template <class Stride>
constexpr bool
is_im2col() {
  return hute::is_same_v<Stride, hytlass::detail::TagToStrideC_t<hytlass::layout::TensorNWC>>
      || hute::is_same_v<Stride, hytlass::detail::TagToStrideC_t<hytlass::layout::TensorNHWC>>
      || hute::is_same_v<Stride, hytlass::detail::TagToStrideC_t<hytlass::layout::TensorNDHWC>>;
}

using hytlass::atomic_maximum;

template <class T>
static constexpr int elements_per_access_v = hytlass::sizeof_bits<uint32_t>::value / hytlass::sizeof_bits<T>::value;

template <class GmemLayoutTag>
static constexpr bool is_im2col_mode =
  hute::is_same_v<GmemLayoutTag, hytlass::layout::TensorNWC> ||
  hute::is_same_v<GmemLayoutTag, hytlass::layout::TensorNHWC> ||
  hute::is_same_v<GmemLayoutTag, hytlass::layout::TensorNDHWC>;

template <class T>
struct EmptyStorage {
  HYTLASS_HOST_DEVICE
  T* data() { return nullptr; }
};

template<class EpilogueSchedule, class Stride>
HYTLASS_HOST_DEVICE
auto get_epilogue_stride(Stride stride){
  if constexpr (hute::is_base_of_v<hytlass::gemm::EpilogueTransposed, EpilogueSchedule>) {
    return hute::make_stride(hute::get<1>(stride), hute::get<0>(stride), hute::get<2>(stride));
  }
  else {
    return stride;
  }
}

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithBias { 
  static constexpr bool value = false; 
  using type = typename ThreadEpilogueOp::ElementCompute; 
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithBias <ThreadEpilogueOp, hute::void_t<typename ThreadEpilogueOp::ElementBias>> { 
  static constexpr bool value = true; 
  using type = typename ThreadEpilogueOp::ElementBias; 
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithPerChannelScaling {
  static constexpr bool value = false;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithPerChannelScaling <ThreadEpilogueOp, hute::enable_if_t<ThreadEpilogueOp::IsPerChannelScalingSupported>> {
  static constexpr bool value = true;
};

template <typename ThreadEpilogueOp, typename = void>
struct IsThreadEpilogueOpWithActivation {
  static constexpr bool value = false;
  using type = void;
};

template <typename ThreadEpilogueOp>
struct IsThreadEpilogueOpWithActivation <ThreadEpilogueOp, hute::enable_if_t<ThreadEpilogueOp::IsEltActSupported>> {
  static constexpr bool value = true;
  using type = typename ThreadEpilogueOp::ActivationFn;
};

} // namespace detail
} // namespace collective
} // namespace epilogue
} // namespace hytlass

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

#include "hytlass/arch/arch.h"
#include "hytlass/gemm/gemm.h"

#include "hute/layout.hpp"
#include "hute/numeric/integral_constant.hpp"
//////////////////////////////////////////////////////////////////////////////

namespace hytlass::detail {

template <class T, template <int...> class U>
struct is_kernel_tag_of : hute::false_type {};

template <template <int...> class U, int... Args>
struct is_kernel_tag_of<U<Args...>, U> : hute::true_type {};

template <class T, template <int...> class U>
constexpr bool is_kernel_tag_of_v = is_kernel_tag_of<T, U>::value;

}

//////////////////////////////////////////////////////////////////////////////

namespace hytlass::gemm {
using namespace hute;

//////////////////////////////////////////////////////////////////////////////

namespace detail {

enum class KernelInputTransformType {
    FastF32,
    InterleavedComplexTF32
};

} // namespace detail

//////////////////////////////////////////////////////////////////////////////

//
// Kernel schedule policies (the base class tags, one for each kernel layer file)
//
struct KernelMultistage { };
struct KernelStreamKSpecialized { };
struct KernelSplitkParallelSpecialized { };
struct KernelPtrArraySpecialized { };

//////////////////////////////////////////////////////////////////////////////

// Policies for dispatch of epilogue
struct EpilogueDefault { };
struct EpilogueTransposed { };

//////////////////////////////////////////////////////////////////////////////

//
// Collective Mainloop Policies
//

// 2 stage pipeline through 1 stage in smem, 1 in rmem, WITHOUT predicated gmem loads
struct MainloopGfx928TwoStageUnpredicated {
  constexpr static int Stages = 2;
  using ArchTag = arch::Gfx928;
  using Schedule = KernelMultistage;
  using ClusterShape = Shape<_1,_1,_1>;
};

// 2 stage pipeline, stream-K and split-K dispatch mainloop through 1 stage in smem, 1 in rmem, with predicated gmem loads
template<
  int Stages_ = 2,
  class ArchTag_ = void,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule_ = KernelMultistage
>
struct MainloopDispatch {
  constexpr static int Stages = Stages_;
  using ArchTag = ArchTag_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  static_assert(
    hute::is_same_v<Schedule, KernelMultistage> ||
    hute::is_same_v<Schedule, KernelStreamKSpecialized> ||
    hute::is_same_v<Schedule, KernelSplitkParallelSpecialized> ||
    hute::is_same_v<Schedule, KernelPtrArraySpecialized>,
    "KernelSchedule must be one of the dispatch");
};

template<
  int Stages_ = 2,
  class ArchTag_ = void,
  class ClusterShape_ = Shape<_1,_1,_1>,
  class KernelSchedule_ = KernelPtrArraySpecialized
>
struct MainloopDispatchPtrArray {
  constexpr static int Stages = Stages_;
  using ArchTag = ArchTag_;
  using ClusterShape = ClusterShape_;
  using Schedule = KernelSchedule_;
  static_assert(
    hute::is_same_v<Schedule, KernelPtrArraySpecialized>,
    "KernelSchedule must be one of the dispatchPtrArray");
};

//////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::gemm

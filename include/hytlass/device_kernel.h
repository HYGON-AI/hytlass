#include "hip/hip_runtime.h"
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
    \brief Template for generic HYTLASS kernel.
*/

#pragma once

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC__)
#  define HYTLASS_GRID_CONSTANT_ENABLED
#endif

#if ! defined(HYTLASS_GRID_CONSTANT)
#  if defined(HYTLASS_GRID_CONSTANT_ENABLED)
#    define HYTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define HYTLASS_GRID_CONSTANT
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

////////////////////////////////////////////////////////////////////////////////


template<typename Operator, typename=void>
struct GetThreadCount {
    static constexpr unsigned value = 1024;
};
template<typename Operator>
struct GetThreadCount<Operator, std::void_t<decltype(Operator::kThreadCount)>> {
    static constexpr unsigned value = Operator::kThreadCount;
};

/// Generic HYTLASS kernel template.
template <typename Operator>
HYTLASS_GLOBAL
#ifdef __HIPCC__
__launch_bounds__(GetThreadCount<Operator>::value)
#endif
void Kernel(typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
}


/// Generic HYTLASS kernel template.
template <typename Operator>
HYTLASS_GLOBAL
#ifdef __HIPCC__
__launch_bounds__(GetThreadCount<Operator>::value)
#endif
void Kernel2(typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator::invoke(params, *shared_storage);
}


////////////////////////////////////////////////////////////////////////////////
//
// 3.0 specific launch
//
////////////////////////////////////////////////////////////////////////////////

/// Generic HYTLASS kernel template.
template <typename Operator>
HYTLASS_GLOBAL
#ifdef __HIPCC__
// Enclosing this in __HIPCC__ suppresses MSVC warnings.
__launch_bounds__(Operator::MaxThreadsPerBlock, Operator::MinBlocksPerMultiprocessor)
#endif // __HIPCC__
void device_kernel(HYTLASS_GRID_CONSTANT typename Operator::Params const params)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
}

////////////////////////////////////////////////////////////////////////////////
} /// namespace hytlass

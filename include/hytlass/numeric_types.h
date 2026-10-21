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
/*! 
    \file
    \brief Top-level include for all HYTLASS numeric types.
*/
#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/platform/platform.h"
#include "hytlass/numeric_size.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <size_t... Seq>
struct index_sequence;

template <size_t N, size_t... Next>
struct index_sequence_helper : index_sequence_helper<N - 1, N - 1, Next...> {};

template <size_t... Next>
struct index_sequence_helper<0, 0, Next...> {
  using type = index_sequence<0, Next...>;
};

template <size_t N>
using make_index_sequence = typename index_sequence_helper<N>::type;


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Get the register type used in kernel
//
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template<typename T>
struct get_unpacked_element_type {
  using type = T;
};

} // namespace detail

}  // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "hytlass/integer_subbyte.h"
#include "hytlass/half.h"
#include "hytlass/bfloat16.h"
#include "hytlass/tfloat32.h"
#include "hytlass/float8.h"
#include "hytlass/uint128.h"
/////////////////////////////////////////////////////////////////////////////////////////////////


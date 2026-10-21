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

#include <hip/hip_vector_types.h>
#include <hytlass/numeric_types.h>
#include <hytlass/numeric_size.h>

#include <hute/numeric/int.hpp>
#include <hute/numeric/real.hpp>

namespace hute {

template <typename T>
struct sizeof_bits : public hytlass::sizeof_bits<T> {};

// DO NOT change auto to int, sizeof_bits<sparse_elem> use integral_ratio instead of int 
template <class T>
static constexpr auto sizeof_bits_v = sizeof_bits<T>::value;

using hytlass::bits_to_bytes;

using hytlass::is_subbyte;

template <class T>
static constexpr auto is_subbyte_v = is_subbyte<T>::value;

using hytlass::half_t;
using hytlass::bfloat16_t;

using hytlass::tfloat32_t;

// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
// This umbrella datatype can be enabled when a user provides a specific
// datatype in runtime argument list.
using hytlass::type_erased_dynamic_float8_t;
using hytlass::float_e4m3_t;
using hytlass::float_e5m2_t;

using hytlass::uint1b_t;
using hytlass::int2b_t;
using hytlass::uint2b_t;
using hytlass::int4b_t;
using hytlass::uint4b_t;
using hytlass::bin1_t;

} // end namespace hute

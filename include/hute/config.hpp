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
 
#if defined(__HIPCC__)
#  define HUTE_HOST_DEVICE __forceinline__ __host__ __device__
#  define HUTE_DEVICE      __forceinline__          __device__
#  define HUTE_HOST        __forceinline__ __host__
#else
#  define HUTE_HOST_DEVICE inline
#  define HUTE_DEVICE      inline
#  define HUTE_HOST        inline
#endif // HUTE_HOST_DEVICE, HUTE_DEVICE

# define HUTE_HOST_RTC HUTE_HOST

#ifdef __HIPCC__
#  define HUTE_UNROLL
#  define HUTE_NO_UNROLL
#elif !defined(__clang__) && defined(__HIP_DEVICE_COMPILE__)
#  define HUTE_UNROLL    #pragma unroll
#  define HUTE_NO_UNROLL #pragma unroll 1
#else
#  define HUTE_UNROLL
#  define HUTE_NO_UNROLL
#endif // HUTE_UNROLL

#if defined(__HIP_DEVICE_COMPILE__)
#  define HUTE_INLINE_CONSTANT                 static const __device__
#else
#  define HUTE_INLINE_CONSTANT                 static constexpr
#endif

#if (defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 700))
#  define HUTE_GRID_CONSTANT_ENABLED
#endif

#if ! defined(HUTE_GRID_CONSTANT)
#  if defined(HUTE_GRID_CONSTANT_SUPPORTED) && defined(HUTE_GRID_CONSTANT_ENABLED)
#    define HUTE_GRID_CONSTANT __grid_constant__
#  else
#    define HUTE_GRID_CONSTANT
#  endif
#endif

// Some versions of GCC < 11 have trouble deducing that a
// function with "auto" return type and all of its returns in an "if
// constexpr ... else" statement must actually return.  Thus, GCC
// emits spurious "missing return statement" build warnings.
// Developers can suppress these warnings by using the
// HUTE_GCC_UNREACHABLE macro, which must be followed by a semicolon.
// It's harmless to use the macro for other GCC versions or other
// compilers, but it has no effect.
#if ! defined(HUTE_GCC_UNREACHABLE)
#  if defined(__GNUC__)
#    define HUTE_GCC_UNREACHABLE __builtin_unreachable()
#  else
#    define HUTE_GCC_UNREACHABLE
#  endif
#endif

#if defined(_MSC_VER)
// Provides support for alternative operators 'and', 'or', and 'not'
#  include <iso646.h>
#endif // _MSC_VER

# define HUTE_STL_NAMESPACE std

//
// Assertion helpers
//

# include <cassert>

#define HUTE_STATIC_V(x)            decltype(x)::value

#define HUTE_STATIC_ASSERT          static_assert
#define HUTE_STATIC_ASSERT_V(x,...) static_assert(decltype(x)::value, ##__VA_ARGS__)

#if ( defined(__HIP_DEVICE_COMPILE__) and not defined(__HIPCC__) )
#  define HUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x); __brkpt()
#else
#  define HUTE_INVALID_CONTROL_PATH(x) assert(0 && x); printf(x)
#endif

//
// IO
//

#  include <cstdio>
#  include <iostream>
#  include <iomanip>

//
// Support
//

#include <hute/util/type_traits.hpp>

//
// Basic types
//

#include <hute/numeric/numeric_types.hpp>

//
// Debugging utilities
//

#include <hute/util/print.hpp>
#include <hute/util/debug.hpp>

namespace hute{
    typedef float v4f __attribute__((ext_vector_type(4)));
    typedef int v4i __attribute__((ext_vector_type(4)));
    typedef float v2f __attribute__((ext_vector_type(2)));
    typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
    typedef int intx4_t __attribute__((ext_vector_type(4)));
    typedef int intx2_t __attribute__((ext_vector_type(2)));
    typedef  __fp16  __fp16x4_t __attribute__((ext_vector_type(4)));
    typedef  __fp16  __fp16x8_t __attribute__((ext_vector_type(8)));
    typedef short __bf16x4_t __attribute__((ext_vector_type(4)));
    typedef int8_t __i8x8_t __attribute__((ext_vector_type(8)));
    typedef int8_t __i8x16_t __attribute__((ext_vector_type(16)));
};


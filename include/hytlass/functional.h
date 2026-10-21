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
    \brief Define basic numeric operators

    This is inspired by the Standard Library's <functional> header.
*/
#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"

#include <hip/hip_runtime.h>

#if defined(HYTLASS_ARCH_WMMA_ENABLED)
#include <mma.h>
#endif // defined(HYTLASS_ARCH_WMMA_ENABLED)

#ifdef _MSC_VER
// Provides support for alternate operators such as 'and', 'or', ...
#include <iso646.h>
#endif // _MSC_VER

namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct absolute_value_op {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return abs(lhs);
  }
};

template <>
struct absolute_value_op<float> {
  HYTLASS_HOST_DEVICE
  float operator()(float lhs) const { return fabs(lhs); }
};

template <typename T>
struct plus {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct minus {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs -= rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

template <typename T>
struct scale {
  T const scaling_factor_;

  HYTLASS_HOST_DEVICE
  scale(float scaling_factor) : scaling_factor_(scaling_factor) {
  }

  T operator()(T const &rhs) const {
    T result = rhs * scaling_factor_;
    return result;
  }
};

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ >= 530
/// Partial specializations needed when __HIP_NO_HALF2_OPERATORS__ is set
template<>
struct plus<__half2> {
  HYTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hadd2(lhs, rhs);
  }
};

template<>
struct minus<__half2> {
  HYTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hsub2(lhs, rhs);
  }
};

template<>
struct multiplies<__half2> {
  HYTLASS_HOST_DEVICE
  __half2 operator()(__half2 lhs, __half2 const &rhs) const {
    return __hmul2(lhs, rhs);
  }
};

/// Partial specializations needed when __HIP_NO_HALF_OPERATORS__ is set
template<>
struct plus<__half> {
  HYTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hadd(lhs, rhs);
  }
};

template<>
struct minus<__half> {
  HYTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hsub(lhs, rhs);
  }
};

template<>
struct multiplies<__half> {
  HYTLASS_HOST_DEVICE
  __half operator()(__half lhs, __half const &rhs) const {
    return __hmul(lhs, rhs);
  }
};
#endif // defined(__HIP_DEVICE_COMPILE__)


/// Squares with optional conversion
template <typename T, typename Output = T>
struct square {
  HYTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Returns the magnitude squared of an element.
template <typename T, typename Output = T>
struct magnitude_squared {
  HYTLASS_HOST_DEVICE
  Output operator()(T lhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct square_difference {
  HYTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

/// Computes the square of a difference with optional conversion
template <typename T, typename Output = T>
struct magnitude_squared_difference {
  HYTLASS_HOST_DEVICE
  Output operator()(T lhs, T rhs) const {
    multiplies<Output> mul_op;

    Output y = Output(lhs) - Output(rhs);
    return mul_op(y, y);
  }
};

// Computes the reciprocal square root
template <typename T>
struct inverse_square_root;

template <>
struct inverse_square_root<float> {
  HYTLASS_HOST_DEVICE
  float operator()(float const &lhs) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return rsqrtf(lhs);
#else
    return 1.f / std::sqrt(lhs);
#endif
  }
};

template <>
struct inverse_square_root<half_t> {
  HYTLASS_HOST_DEVICE
  half_t operator()(half_t const &lhs) const {
#if defined(__HIP_DEVICE_COMPILE__)
    auto result = hrsqrt(reinterpret_cast<__half const &>(lhs));
    return reinterpret_cast<half_t const &>(result);
#else
    return half_t(1.f / std::sqrt(half_t::convert(lhs)));
#endif
  }
};

/// Divides
template <typename T>
struct divides {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs /= rhs;
    return lhs;
  }
};

/// reciprocal_approximate
template <typename T>
struct reciprocal_approximate {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return divides<T>{}(T(1), lhs);
  }
};

template <>
struct reciprocal_approximate <float> {
  HYTLASS_HOST_DEVICE
  float operator()(float lhs) const {
    float ret;
      ret = 1.0f / lhs;
    return ret;
  }
};

/// Negate
template <typename T>
struct negate {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return -lhs;
  }
};

/// Greater equal
template <typename T>
struct greater_equal {
  HYTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs >= rhs);
  }
};

/// Greater
template <typename T>
struct greater {
  HYTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs > rhs);
  }
};

/// Less equal
template <typename T>
struct less_equal {
  HYTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs <= rhs);
  }
};

/// Less
template <typename T>
struct less {
  HYTLASS_HOST_DEVICE
  bool operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs);
  }
};

template <typename T, bool PropagateNaN = false>
struct maximum {
  HYTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (lhs < rhs ? rhs : lhs);
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template<typename T>
struct maximum_with_default_nan_propagation : public maximum<T>
{};

// Maximum with nan propagation
// To propagate NANs, the "max" of a two element that contains NaNs should also return a NaN
template <typename T>
struct maximum<T, true> {
  HYTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return lhs > rhs or isnan(lhs) ? lhs : rhs;
#else
    return lhs > rhs or std::isnan(lhs) ? lhs : rhs;
#endif
  }
};

template <>
struct maximum<float, false> {
  HYTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fmaxf(lhs, rhs);
  }
};

template <>
struct maximum<float, true> {
  HYTLASS_HOST_DEVICE
  float operator()(float const lhs, float const rhs) const {
    float res;
#if defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 800)
    asm volatile("max.NaN.f32 %0, %1, %2;\n" : "=f"(res) : "f"(lhs), "f"(rhs));
#elif defined(__HIP_DEVICE_COMPILE__)
    res = lhs > rhs or isnan(lhs) ? lhs : rhs;
#else
    res = lhs > rhs or std::isnan(lhs) ? lhs : rhs;
#endif
    return res;
  }
};

// This is a subclass and not an alias
// in order to work around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename T>
struct maximum_with_nan_propagation : maximum<T, true>
{};

// This alias exists for backwards compatibility only.
// Please use the correctly spelled class template above.
template <typename T>
using maximum_with_nan_propogation = maximum_with_nan_propagation<T>;

template <typename T, bool PropagateNaN = false>
struct minimum{
  HYTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    return (rhs < lhs ? rhs : lhs);
  }
};

template <typename T>
struct minimum<T, true> {
  HYTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return lhs < rhs or isnan(lhs) ? lhs : rhs;
#else
    return lhs < rhs or std::isnan(lhs) ? lhs : rhs;
#endif
  }
};

template <>
struct minimum<float, false> {
  HYTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const {
    return fminf(lhs, rhs);
  }
};

template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value {
  HYTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(abs_op(lhs), abs_op(rhs));
  }
};

// assumes the left operand is already an absolute value
template <typename T, bool PropagateNaN = false>
struct maximum_absolute_value_reduction {
  HYTLASS_HOST_DEVICE
  float operator()(T const &lhs, T const &rhs) const {
    absolute_value_op<T> abs_op;
    maximum<T, PropagateNaN> max_op;

    return max_op(lhs, abs_op(rhs));
  }
};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add {
  HYTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    return C(a) * C(b) + c;
  }
};

template <typename T>
struct square_and_plus {
  HYTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    multiply_add<T> multiply_add_op;
    return multiply_add_op(rhs, rhs, lhs);
  }
};

// Fused multiply-add that takes exactly one template parameter.
// This is useful for working around a known Clang issue,
// where a template template parameter with one template parameter
// does not match classes that take multiple template parameters
// but have defaults for all but the first.
template <typename A>
struct homogeneous_multiply_add : public multiply_add<A, A, A>
{};

/// Fused multiply-add
template <typename A, typename B = A, typename C = A>
struct multiply_add_relu0 {
  HYTLASS_HOST_DEVICE
  C operator()(A const &a, B const &b, C const &c) const {
    maximum<C> mx;
    return mx(C(a) * C(b) + c, C(0));
  }
};

/// Fused multiply-add
template <typename T>
struct and_add {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a & b) + c);
  }
};


/// Fused multiply-add
template <typename T>
struct xor_add {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b, T const &c) const {
    return ((a ^ b) + c);
  }
};

template <typename T>
struct conjugate {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return a;
  }
};

template <typename T>
struct first {
  HYTLASS_HOST_DEVICE
  T operator()(T const & first, T const &...) const {
    return first;
  }
  HYTLASS_HOST_DEVICE
  T operator()(T const & first) const {
    return first;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct logical_and {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) && static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_or {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return ((static_cast<bool>(a) || static_cast<bool>(b)) ? T(1) : T());
  }
};

template <typename T>
struct logical_not {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return T(!(a));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct bit_and {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a & b;
  }
};

template <typename T>
struct bit_or {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a | b;
  }
};

template <typename T>
struct bit_not {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a) const {
    return ~a;
  }
};

template <typename T>
struct bit_xor {
  HYTLASS_HOST_DEVICE
  T operator()(T const &a, T const &b) const {
    return a ^ b;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Atomic reductions

template <typename T>
struct atomic_add
{
  HYTLASS_DEVICE
  void operator()(T *ptr, const T &data)
  {
#if defined(__HIP_DEVICE_COMPILE__)
    atomicAdd(ptr, data);
#endif
  }
};

template<>
struct atomic_add<double>
{
  HYTLASS_DEVICE
  void operator()(double *ptr, const double &data)
  {
#if !defined(__HIP_DEVICE_COMPILE__)
      HYTLASS_UNUSED(ptr);
      HYTLASS_UNUSED(data);
#elif (__HIP_DEVICE_COMPILE__ >= 600)
    atomicAdd(ptr, data);
#else
    // Use CAS loop
    unsigned long long int* ptr_int = reinterpret_cast<unsigned long long int*>(ptr);
    unsigned long long int old_int = *ptr_int;
    unsigned long long int assumed_int;

    do {
      double update = data + __longlong_as_double(old_int);
      assumed_int = old_int;
      old_int = atomicCAS(ptr_int, assumed_int, __double_as_longlong(update));
    } while (assumed_int != old_int);
#endif // (__HIP_DEVICE_COMPILE__ >= 600)
  }
};

template<>
struct atomic_add<half2>
{
  HYTLASS_DEVICE
  void operator()(half2 *ptr, const half2 &data)
  {
#if !defined(__HIP_DEVICE_COMPILE__) || (defined(__HIP_DEVICE_COMPILE__)  && (__HIP_DEVICE_COMPILE__ < 600))
      HYTLASS_UNUSED(ptr);
      HYTLASS_UNUSED(data);
#else
    // Vector-2 atomic reduction requires .target sm_60 or higher
    uint32_t word = reinterpret_cast<const uint32_t&>(data);
    asm volatile ("red.gpu.global.add.noftz.f16x2 [%0], %1;\n" : : "l"(ptr), "r"(word));
#endif // (__HIP_DEVICE_COMPILE__ >= 600)
  }
};

template <typename T>
using red [[deprecated("use atomic_add instead")]] = atomic_add<T>;

template <typename T>
struct atomic_maximum {
  HYTLASS_DEVICE
  T operator()(T *ptr, T value) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return atomicMax(ptr, value);
#else
    HYTLASS_UNUSED(ptr);
    HYTLASS_UNUSED(value);
    HYTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

template <>
struct atomic_maximum<float> {
  HYTLASS_DEVICE
  float operator()(float *ptr, float value) const {
#if defined(__HIP_DEVICE_COMPILE__)
    return !signbit(value) ?
      __int_as_float(atomicMax((int*)ptr, __float_as_int(value))) :
      __uint_as_float(atomicMin((unsigned int*)ptr, __float_as_uint(value)));
#else
    HYTLASS_UNUSED(ptr);
    HYTLASS_UNUSED(value);
    HYTLASS_NOT_IMPLEMENTED();
    return 0;
#endif
  }
};

// is_atomic
template <class Fn>
struct is_atomic : platform::false_type {};
template <class T>
struct is_atomic<atomic_add<T>> : platform::true_type {};
template <class T>
struct is_atomic<atomic_maximum<T>> : platform::true_type {};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

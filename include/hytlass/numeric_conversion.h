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
    \brief Boost-like numeric conversion operator for HYTLASS numeric types
*/

#pragma once

#if !defined(__HIPCC_RTC__)
#include <cfenv>
#endif

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/transform/thread/unary_op.h"

#include "hytlass/array.h"
#include "hytlass/half.h"

namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Floating-point rounding style similare to Standard Library's formats but supporting
/// additional rounding options.
enum class FloatRoundStyle {
  round_indeterminate,          ///< rounding mode unknown
  round_toward_zero,            ///< round toward zero
  round_to_nearest,             ///< round to nearest even
  round_to_nearest_satfinite,   ///< round to nearest even, capping value to min and max of destination type
  round_toward_infinity,        ///< round toward infinity
  round_toward_neg_infinity,    ///< round toward negative infinity
  round_half_ulp_truncate,      ///< add 0.5ulp to integer representation then round toward zero
  round_half_ulp_trunc_dntz     ///< like round_half_ulp_truncate, except denorms are rounded *toward* zero
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
struct NumericConverter {

  using result_type = T;
  using source_type = S;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<result_type>(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__HIP_DEVICE_COMPILE__)
template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rn(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return __float2int_rz(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#elif !defined(__HIPCC_RTC__)

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int32_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    return (result_type)std::nearbyint(s);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float => int8_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__HIP_DEVICE_COMPILE__) 
template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    int32_t intermediate;
    intermediate = (int32_t)s;

    return static_cast<result_type>(intermediate);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    int32_t intermediate;
    intermediate = __float2int_rz(s);

    return static_cast<result_type>(intermediate);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#elif !defined(__HIPCC_RTC__)

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TONEAREST);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }
  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<int8_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = int8_t;
  using source_type = float;
  static FloatRoundStyle const round_style =  FloatRoundStyle::round_toward_zero;

  static result_type convert(source_type const & s) {
    std::fesetround(FE_TOWARDZERO);
    int32_t intermediate = (int32_t)std::nearbyint(s);

    // Low-end saturation
    intermediate = std::max(intermediate, (int32_t)std::numeric_limits<int8_t>::lowest());

    // High-end saturation
    intermediate = std::min(intermediate, (int32_t)std::numeric_limits<int8_t>::max());

    return static_cast<result_type>(intermediate);
  }

  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= hytlass::half_t
template <typename T, FloatRoundStyle Round>
struct NumericConverter<T, T, Round> {

  using result_type = T;
  using source_type = T;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return s;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> hytlass::half_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= hytlass::half_t
template <FloatRoundStyle Round>
struct NumericConverter<float, hytlass::half_t, Round> {

  using result_type = float;
  using source_type = hytlass::half_t;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<float>(s);

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-to-nearest
template <>
struct NumericConverter<hytlass::half_t, float, FloatRoundStyle::round_to_nearest> {

  using result_type = hytlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result = static_cast<hytlass::half_t>(s);

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Specialization for round-toward-zero
template <>
struct NumericConverter<hytlass::half_t, float, FloatRoundStyle::round_toward_zero> {

  using result_type = hytlass::half_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  /// Round toward zero
  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & flt) {

  #if defined(__HIP_DEVICE_COMPILE__)
    return hytlass::half_t(__float2half_rz(flt));
  #else
    // software implementation rounds toward nearest even
    unsigned const& s = reinterpret_cast<unsigned const &>(flt);
    uint16_t sign = uint16_t((s >> 16) & 0x8000);
    int32_t exp = int32_t((s >> 23) & 0xff) - 127;
    int mantissa = s & 0x7fffff;
    uint16_t u = 0;

    if ((s & 0x7fffffff) == 0) {
      // sign-preserving zero
      return hytlass::half_t::bitcast(sign);
    }

    if (exp > 15) {
      if (exp == 128 && mantissa) {
        // not a number
        u = 0x7fff;
      } else {
        // overflow to infinity
        u = sign | 0x7c00;
      }
      return hytlass::half_t::bitcast(u);
    }

    if (exp >= -14) {
      // normal fp32 to normal fp16
      u = uint16_t((uint32_t(exp + 15) & 0x1f) << 10);
      u = uint16_t(u | (mantissa >> 13));
    } else {
      // normal single-precision to subnormal hytlass::half_t-precision representation
      int rshift = (-14 - exp);
      if (rshift < 32) {
        mantissa |= (1 << 23);
        mantissa = (mantissa >> rshift);
        u = (uint16_t(mantissa >> 13) & 0x3ff);
      } else {
        mantissa = 0;
        u = 0;
      }
    }

    u |= sign;

    return hytlass::half_t::bitcast(u);

  #endif // defined(__HIP_DEVICE_COMPILE__)
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> hytlass::bfloat16_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= hytlass::bfloat16_t
template <FloatRoundStyle Round>
struct NumericConverter<float, hytlass::bfloat16_t, Round> {

  using result_type = float;
  using source_type = hytlass::bfloat16_t;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = hytlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return static_cast<hytlass::bfloat16_t>(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::bfloat16_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = hytlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);

    #if defined(__HIP_DEVICE_COMPILE__)
    if (::isfinite(s)) {
      x32 += 0x8000;
    }
    #else
    if (std::isfinite(s)) {
      x32 += 0x8000;
    }
    #endif

    uint16_t x16 = uint16_t((x32 >> 16) & 0xffff);
    return hytlass::bfloat16_t::bitcast(x16);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::bfloat16_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = hytlass::bfloat16_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);
    uint16_t x16 = uint16_t(x32 >> 16);

    return hytlass::bfloat16_t::bitcast(x16);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for float <=> hytlass::tfloat32_t
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for float <= hytlass::tfloat32_t
template <FloatRoundStyle Round>
struct NumericConverter<float, hytlass::tfloat32_t, Round> {

  using result_type = float;
  using source_type = hytlass::tfloat32_t;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    return static_cast<float>(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::tfloat32_t, float, FloatRoundStyle::round_to_nearest> {
  using result_type = hytlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned storage = reinterpret_cast<unsigned const &>(s);

    if ((storage & 0x7f800000) != 0x7f800000) {

      bool mantissa_bit = ((storage & (1 << 13)) != 0);
      bool round_bit = ((storage & (1 << 12)) != 0);
      bool sticky_bit = ((storage & ((1 << 12) - 1)) != 0);

      if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
        storage += uint32_t(1 << 13);
      }

      // Note, the following is intentionally commented out. TF32
      // does not define the low order bits, so they may be left in
      // an undefined state.
      //
      // By not truncating these bit explicitly, we avoid an extra logical
      // operation.
      //
      // TF32 may be implicitly converted to float by performing this
      // operation as needed.
      //
      // storage = (storage & ~0x1fff);
    }
    else if (storage & ~0xff800000) {
      storage = 0x7fffffff;
    }

    return hytlass::tfloat32_t::bitcast(storage);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_truncate> {
  using result_type = hytlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_truncate;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    return hytlass::tfloat32_t::round_half_ulp_truncate(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// This rounding operation is similar to half_ulp_truncate except it rounds denorms toward zero.
/// It avoids predicated code, though it requires a temporary register.
template <>
struct NumericConverter<hytlass::tfloat32_t, float, FloatRoundStyle::round_half_ulp_trunc_dntz> {
  using result_type = hytlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_half_ulp_trunc_dntz;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    unsigned y = reinterpret_cast<unsigned const &>(s);
    y = y & 0xff800000;
    float d = reinterpret_cast<float const &>(y);
    float z = d / float(1 << 11) + s;

    return reinterpret_cast<result_type const &>(z);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <>
struct NumericConverter<hytlass::tfloat32_t, float, FloatRoundStyle::round_toward_zero> {
  using result_type = hytlass::tfloat32_t;
  using source_type = float;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_toward_zero;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {
    uint32_t x = reinterpret_cast<uint32_t const &>(s);
    return hytlass::tfloat32_t::bitcast(x & 0xffffe000);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for float to hytlass::tfloat32_t big and small values
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  FloatRoundStyle RoundBig = FloatRoundStyle::round_toward_zero,
  FloatRoundStyle RoundSmall = FloatRoundStyle::round_half_ulp_truncate
>
struct NumericConverterFastF32 {

  // result_type holds big hytlass::tfloat32_t at idx(0) and small hytlass::tfloat32_t at idx(1)
  using result_type = Array<hytlass::tfloat32_t, 2>;

  // source data type
  using source_type = float;

  // rounding styles for big and small part
  static FloatRoundStyle const kRoundBig = RoundBig;
  static FloatRoundStyle const kRoundSmall = RoundSmall;

  HYTLASS_HOST_DEVICE
    static result_type convert(source_type const & source) {

    result_type result;
    NumericConverter<hytlass::tfloat32_t, float, kRoundBig> convert_big_;
    NumericConverter<hytlass::tfloat32_t, float, kRoundSmall> convert_small_;

    // convert and fill hytlass::tfloat32_t big at idx 0
    result[0] = convert_big_(source);

    // convert and fill hytlass::tfloat32_t small at idx 1
    result[1] = convert_small_(source - static_cast<float>(result[0]));

    return result;
  }

  HYTLASS_HOST_DEVICE
    result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion and Clamp operator for Integers
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S
>
struct NumericConverterClamp {

  using result_type = T;
  using source_type = S;

  HYTLASS_HOST_DEVICE
    static result_type convert(source_type const & s) {
    NumericConverter<result_type, source_type> convert_op;
    result_type const kClamp_max = platform::numeric_limits<result_type>::max();
    result_type const kClamp_min = platform::numeric_limits<result_type>::lowest();
    if (s < (source_type)kClamp_min)
      return kClamp_min;
    if (s > (source_type)kClamp_max)
      return kClamp_max;
    return convert_op(s);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// This converter is needed to enable hytlass::half_t output types when using int32_t accumulators.
// Since floating-point types do not require a clamp, this converter simply casts from
// the source type to hytlass::half_t.
template <
  typename S
>
struct NumericConverterClamp<hytlass::half_t, S> {

  using result_type = hytlass::half_t;
  using source_type = S;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    return static_cast<hytlass::half_t>(source);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conversion operator for Array
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Conversion operator for Array
template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = hytlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverter {

  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      if (platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value) {
        result[i] = convert_(s[i]);
      } else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  typename T,
  int N,
  FloatRoundStyle Round,
  typename Transform
>
struct NumericArrayConverter<T, T, N, Round, Transform> {

  using result_type = Array<T, N>;
  using source_type = Array<T, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const &source) {
    if (platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value) {
      return source;
    } else {
      result_type result;
      HYTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
          result[i] = conj(source[i]);
      }
      return result;
    }
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half, 2> <= Array<float, 2>, round to nearest
template <>
struct NumericArrayConverter<hytlass::half_t, float, 2, FloatRoundStyle::round_to_nearest> {

  using result_type = Array<hytlass::half_t, 2>;
  using source_type = Array<float, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    Array<hytlass::half_t, 2> result;
    #ifdef MIX_FP16_EPILOGUE
      // TODO:  rtn
      __half2 rst;
      float s1 = source[0];
      float s2 = source[1];
      asm volatile(
        "v_cvt_pkrtz_f16_f32 %0 %1 %2"
        :"=v"(rst),"+v"(s1),"+v"(s2)
      );
      
      result = reinterpret_cast<Array<hytlass::half_t, 2> &>(rst);
    #else
    #if defined(__HIP_DEVICE_COMPILE__)
      float s1 = source[0];
      float s2 = source[1];
      __half2 rst;
      asm volatile(
        "v_cvt_pkrtz_f16_f32 %0 %1 %2\n\t"
        :"+v"(rst),"+v"(s1),"+v"(s2)
      );

      result = reinterpret_cast<Array<hytlass::half_t, 2> &>(rst);

    #else
      NumericConverter<hytlass::half_t, float, round_style> convert_;
      // NOTE: hytlass::Array<half, N> is NOT an aggregate type and
      //  below `{}` does NOT conduct zero initialization. Below `{}` will 
      //  conduct default initialization (calling default ctr). We use this syntax
      //  to resolve compiler warning on uninitialized member variable.
      result[0] = convert_(source[0]);
      result[1] = convert_(source[1]);
    #endif
    #endif
    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<hytlass::half_t, 2>, round to nearest
template <FloatRoundStyle Round>
struct NumericArrayConverter<float, hytlass::half_t, 2, Round> {

  using result_type = Array<float, 2>;
  using source_type = Array<hytlass::half_t, 2>;
  static FloatRoundStyle const round_style = FloatRoundStyle::round_to_nearest;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    Array<float, 2> result;

    #if (defined(__HIP_DEVICE_COMPILE__))
      float2 result2 = reinterpret_cast<float2 &>(result) = __half22float2(reinterpret_cast<__half2 const &>(source));
      return {
        float{result2.x},
        float{result2.y}
      };
    #else
      NumericConverter<float, hytlass::half_t, round_style> convert_;
      return {
        convert_(source[0]),
        convert_(source[1])
      };
    #endif
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<hytlass::half_t, float, N, Round> {

  using result_type = Array<hytlass::half_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<hytlass::half_t, float, 2, Round> convert_vector_;
    NumericConverter<hytlass::half_t, float, Round> convert_element_;

    result_type result;

    Array<hytlass::half_t, 2> *result_ptr = reinterpret_cast<Array<hytlass::half_t, 2> *>(&result);
    Array<float, 2> const *source_ptr = reinterpret_cast<Array<float, 2> const *>(&source);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/// Partial specialization for Array<half> <= Array<float>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, hytlass::half_t, N, Round> {

  using result_type = Array<float, N>;
  using source_type = Array<hytlass::half_t, N>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {

    NumericArrayConverter<float, hytlass::half_t, 2, Round> convert_vector_;
    NumericConverter<float, hytlass::half_t, Round> convert_element_;

    result_type result;

    Array<float, 2> *result_ptr = reinterpret_cast<Array<float, 2> *>(&result);
    Array<hytlass::half_t, 2> const *source_ptr = reinterpret_cast<Array<hytlass::half_t, 2> const *>(&source);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 2; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    if (N % 2) {
      result[N - 1] = convert_element_(source[N - 1]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, N> <=> Array<float_e4m3_t, N>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: add pk inst for fp8
/// Partial specialization for Array<float, 2> <= Array<float_e4m3_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, hytlass::float_e4m3_t, 2, Round> {
  using result_element = float;
  using source_element = hytlass::float_e4m3_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 2> <= Array<float, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, float, 2, Round> {
  using result_element = hytlass::float_e4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 2> <= Array<float_e5m2_t, 2>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<float, hytlass::float_e5m2_t, 2, Round> {
  using result_element = float;
  using source_element = hytlass::float_e5m2_t;

  using result_type = Array<result_element, 2>;
  using source_type = Array<source_element, 2>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 2; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};
namespace detail {

/// Special converters that can be used with 4 8-bit elements packed in a register.
/// Common use is for fast FP8 converters.
template <
  typename T,
  typename S,
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
  typename Transform = hytlass::transform::thread::UnaryTransform::Identity
>
struct NumericArrayConverterPacked4Element {
  using result_type = Array<T, 4>;
  using source_type = Array<S, 4>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value ||
                platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Conjugate>::value,
                  "Unary Operator not supported.");

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & s) {

    result_type result;
    NumericConverter<T, S, Round> convert_;
    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      if (platform::is_same<Transform, hytlass::transform::thread::UnaryTransform::Identity>::value) {
        result[i] = convert_(s[i]);
      }
      else { // conjugate
        result[i] = conj(convert_(s[i]));
      }
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, hytlass::float_e4m3_t, Round> {
  using result_element = float;
  using source_element = hytlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, float, Round> {
  using result_element = hytlass::float_e4m3_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float, hytlass::float_e5m2_t, Round> {
  using result_element = float;
  using source_element = hytlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<float, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, float, Round> {
  using result_element = hytlass::float_e5m2_t;
  using source_element = float;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<hytlass::half_t, 4> <=> Array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<hytlass::half_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<hytlass::half_t, hytlass::float_e4m3_t, Round> {
  using result_element = hytlass::half_t;
  using source_element = hytlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<hytlass::half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, hytlass::half_t, Round> {
  using result_element = hytlass::float_e4m3_t;
  using source_element = hytlass::half_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<hytlass::half_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<hytlass::half_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<hytlass::half_t, hytlass::float_e5m2_t, Round> {
  using result_element = hytlass::half_t;
  using source_element = hytlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<hytlass::half_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, hytlass::half_t, Round> {
  using result_element = hytlass::float_e5m2_t;
  using source_element = hytlass::half_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<hytlass::bfloat16_t, 4> <=> Array<float_e4m3_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<hytlass::bfloat16_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<hytlass::bfloat16_t, hytlass::float_e4m3_t, Round> {
  using result_element = hytlass::bfloat16_t;
  using source_element = hytlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<hytlass::bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, hytlass::bfloat16_t, Round> {
  using result_element = hytlass::float_e4m3_t;
  using source_element = hytlass::bfloat16_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<hytlass::bfloat16_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<hytlass::bfloat16_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<hytlass::bfloat16_t, hytlass::float_e5m2_t, Round> {
  using result_element = hytlass::bfloat16_t;
  using source_element = hytlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<hytlass::bfloat16_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, hytlass::bfloat16_t, Round> {
  using result_element = hytlass::float_e5m2_t;
  using source_element = hytlass::bfloat16_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for Array<float_e4m3_t, 4> <=> Array<float_e5m2_t, 4>
//
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<float_e4m3_t, 4> <= Array<float_e5m2_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e4m3_t, hytlass::float_e5m2_t, Round> {
  using result_element = hytlass::float_e4m3_t;
  using source_element = hytlass::float_e5m2_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/// Partial specialization for Array<float_e5m2_t, 4> <= Array<float_e4m3_t, 4>
template <
  FloatRoundStyle Round
>
struct NumericArrayConverterPacked4Element<float_e5m2_t, hytlass::float_e4m3_t, Round> {
  using result_element = hytlass::float_e5m2_t;
  using source_element = hytlass::float_e4m3_t;

  using result_type = Array<result_element, 4>;
  using source_type = Array<source_element, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    NumericConverter<result_element, source_element, Round> converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      result[i] = converter(source[i]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

}
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Partial specializations for:
//       Array<T, N> <=> Array<float_e4m3_t, N>
//       Array<T, N> <=> Array<float_e5m2_t, N>
// using packed converter under the hood
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename T,
  typename S,
  int N,
  FloatRoundStyle Round
>
struct PackedNumericArrayConverter {
  using result_element = T;
  using source_element = S;

  using result_type = Array<result_element, N>;
  using source_type = Array<source_element, N>;

  static FloatRoundStyle const round_style = Round;

private:
  using packed_result_type = Array<result_element, 4>;
  using packed_source_type = Array<source_element, 4>;

public:
  HYTLASS_DEVICE
  static result_type convert(source_type const & source) {
    result_type result;
    packed_result_type* packed_result = reinterpret_cast<packed_result_type*>(&result);
    const packed_source_type* packed_source = reinterpret_cast<const packed_source_type*>(&source);

    detail::NumericArrayConverterPacked4Element<result_element, source_element, Round> packed_converter;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      packed_result[i] = packed_converter(packed_source[i]);
    }

    // Handle leftovers
    NumericConverter<result_element, source_element, Round> converter;
    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N % 4; ++i) {
      int idx = ((N / 4) * 4) + i;
      result[idx] = converter(source[idx]);
    }

    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const{
    return convert(s);
  }
};

/// Partial specialization for Array<T, N> <= Array<float_e4m3_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, hytlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<T, hytlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<T, N> <= Array<float_e5m2_t, N>
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<T, hytlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<T, hytlass::float_e5m2_t, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, S, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<S, N>
template <
  typename S,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, S, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, S, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, hytlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, hytlass::float_e5m2_t, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, hytlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, hytlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<float_e4m3_t, N> <= Array<float_e4m3_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e4m3_t, hytlass::float_e4m3_t, N, Round> :
  public PackedNumericArrayConverter<float_e4m3_t, hytlass::float_e4m3_t, N, Round> {};

/// Partial specialization for Array<float_e5m2_t, N> <= Array<float_e5m2_t, N>
template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<float_e5m2_t, hytlass::float_e5m2_t, N, Round> :
  public PackedNumericArrayConverter<float_e5m2_t, hytlass::float_e5m2_t, N, Round> {};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Array<int8_t> <= Array<float>
/// Conversion is performed with saturation regardless of setting of
/// the `Round` template parameter.
template <
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, 1, Round> {

  using result_type = Array<int8_t, 1>;
  using source_type = Array<float, 1>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    // Convert to int to int8_t
    NumericConverter<int8_t, float, Round> destination_converter;
    result_type result;
    result[0] = destination_converter(source[0]);
    return result;
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

// To convert a FP32 to Int that has less than 32 bits, we need to convert it to int32 first.
template <
  typename T,
  int N,
  FloatRoundStyle Round
>
struct NumericArrayFP32ToIntConverter {

  using result_type = Array<T, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  static_assert(platform::numeric_limits<T>::is_integer, "the dest type has to be int.");

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    // Convert float to int
    Array<int32_t, N> temporary;

    NumericArrayConverter<int32_t, float, N, Round> compute_converter;
    temporary = compute_converter(source);

    // Convert to int to int8_t
    NumericArrayConverter<T, int32_t, N, Round> destination_converter;
    return destination_converter(temporary);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};


template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int8_t, float, N, Round> {

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int8_t, N, Round> converter;
    return converter(source);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint8_t, float, N, Round> {

  using result_type = Array<uint8_t, N>;
  using source_type = Array<float, N>;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint8_t, N, Round> converter;
    return converter(source);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<int4b_t, float, N, Round> {

  using result_type = Array<int4b_t, N>;
  using source_type = Array<float, N>;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<int4b_t, N, Round> converter;
    return converter(source);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

template <
  int N,
  FloatRoundStyle Round
>
struct NumericArrayConverter<uint4b_t, float, N, Round> {

  using result_type = Array<uint4b_t, N>;
  using source_type = Array<float, N>;

  HYTLASS_HOST_DEVICE
  static result_type convert(source_type const & source) {
    NumericArrayFP32ToIntConverter<uint4b_t, N, Round> converter;
    return converter(source);
  }

  HYTLASS_HOST_DEVICE
  result_type operator()(source_type const &s) const {
    return convert(s);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

  /*
      A helper class that can vectorize a numeric converter with implementation for several vector widths.

      The vector widths must be giving in decreasing order or width, and must be a power of 2.

      The vector converters must produce identical results to the scalar converters for consistency.
    */
  class VectorizedConverter {
  private:
    // Base case to handle remainder elements as scalars.
    template <int Offset, size_t ParentWidth, typename ArrayConverter>
    HYTLASS_DEVICE
    static void convert_helper(
      typename ArrayConverter::result_type& result,
      typename ArrayConverter::source_type const& source) {

      using ElementRes = typename ArrayConverter::result_type::Element;
      using ElementSrc = typename ArrayConverter::source_type::Element;
      // If no more converters, handle the remaining elements as scalars.
      constexpr int total_elements = ArrayConverter::result_type::kElements;
      constexpr int remainder = total_elements - Offset;
      static_assert(remainder == (total_elements % ParentWidth), "Unexpected remainder.");

      typename ArrayConverter::ScalarConverter scalar_converter;
      HYTLASS_PRAGMA_UNROLL
      for (int i = Offset; i < ArrayConverter::result_type::kElements; ++i) {
        result[i] = scalar_converter(ElementSrc(source[i]));
      }
    }

    template <int Offset, size_t ParentWidth, typename ArrayConverter, typename ResultVectorArray, typename SourceVectorArray, typename... OtherVectorArrays>
    HYTLASS_DEVICE
    static void convert_helper(typename ArrayConverter::result_type& result, typename ArrayConverter::source_type const& source) {
      static_assert(sizeof...(OtherVectorArrays) % 2 == 0, "Vector converters must come in {dst, src} pairs");
      static_assert(ResultVectorArray::kElements == SourceVectorArray::kElements, "Vector converters must have the same vector width");
      static_assert(hytlass::platform::is_same<typename ArrayConverter::result_type::Element, typename ResultVectorArray::Element>::value,
        "ResultVectorArray must have the same type ArrayConverter::result_type");
      static_assert(hytlass::platform::is_same<typename ArrayConverter::source_type::Element, typename SourceVectorArray::Element>::value,
        "SourceVectorArray must have the same type ArrayConverter::result_type");
      static_assert(Offset >= 0 && Offset <= ArrayConverter::result_type::kElements, "Offset must be between 0 and N");

      static_assert(ParentWidth == 0 || ParentWidth > ResultVectorArray::kElements, "Vector arrays must be given in decreasing order of width");

      constexpr int vector_width = ResultVectorArray::kElements;
      static_assert(ispow2(vector_width), "Vector width must be a power of 2");

      using ElementRes = typename ArrayConverter::result_type::Element;
      using ElementSrc = typename ArrayConverter::source_type::Element;

      constexpr int vector_bits_res = vector_width * hytlass::sizeof_bits<ElementRes>::value;
      constexpr int vector_bits_src = vector_width * hytlass::sizeof_bits<ElementSrc>::value;

      static_assert(vector_bits_res % 8 == 0, "Result vector type must be byte addressed.");
      static_assert(vector_bits_src % 8 == 0, "Source vector type must be byte addressed.");

      constexpr int vector_offset = Offset / vector_width;
      ResultVectorArray* packed_result_vec = reinterpret_cast<ResultVectorArray*>(&result) + vector_offset;
      SourceVectorArray const* packed_source_vec = reinterpret_cast<SourceVectorArray const*>(&source) + vector_offset;

      // Convert the remaining elements as vectors.
      constexpr int total_elements = ArrayConverter::result_type::kElements;
      constexpr int groups_of_vec = (total_elements - Offset) / vector_width;
      HYTLASS_PRAGMA_UNROLL
      for (int i = 0; i < groups_of_vec; ++i) {
        packed_result_vec[i] = ArrayConverter::template packed_convert<ResultVectorArray, SourceVectorArray>(packed_source_vec[i]);
      }

      constexpr int new_offset = Offset + vector_width * groups_of_vec;
      // Recurse to handle other vector converters, or the scalar base case.
      convert_helper<new_offset, ResultVectorArray::kElements, ArrayConverter, OtherVectorArrays...>(result, source);
    }

  public:
    /*
        A method to convert vectors of elements using the packed_convert method of the converter.

        Converters using this class must implement packed convert and support 1 or more vector conversions.
      */
    template <typename ArrayConverter, typename ResultVectorArray, typename SourceVectorArray, typename... OtherVectorArrays>
    HYTLASS_DEVICE
    static void convert(typename ArrayConverter::result_type& result, typename ArrayConverter::source_type const& source) {
      convert_helper<0, 0, ArrayConverter, ResultVectorArray, SourceVectorArray, OtherVectorArrays...>(result, source);
    }
  };
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// FastNumericArrayConverter only works when the source is within center range.
/// Conversion operator for Array.  See the comments before
/// FastLinearCombinationClamp.
template <typename T, typename S, int N,
          FloatRoundStyle Round = FloatRoundStyle::round_to_nearest,
          typename Enable = void>
struct FastNumericArrayConverter {
  using result_type = Array<T, N>;
  using source_type = Array<S, N>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const &s) {
    NumericArrayConverter<T, S, N, Round> convert_;

    return convert_(s);
  }

  HYTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<float> <= Array<int>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<float, int, N, Round> {
  using result_type = Array<float, N>;
  using source_type = Array<int, N>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const &source) {
    result_type result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int tmp = source[i] + 1262485504 /*0x4B400000*/;
      result[i] = reinterpret_cast<float const &>(tmp) - 12582912.0f;
    }

    return result;
  }

  HYTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<int8_t, 4> <= Array<float, 4>
template <FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, 4, Round> {
  using result_type = Array<int8_t, 4>;
  using source_type = Array<float, 4>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const &source) {
    Array<int32_t, 4> result;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
      float tmp = source[i] + 12582912.0f;
      result[i] = reinterpret_cast<int32_t const &>(tmp);
    }

    result[0] = __byte_perm(result[0], result[1], 0x40);
    result[2] = __byte_perm(result[2], result[3], 0x40);
    result[0] = __byte_perm(result[0], result[2], 0x5410);

    return reinterpret_cast<result_type const &>(result[0]);
  }

  HYTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/// Partial specialization for Array<int8_t> <= Array<float>
template <int N, FloatRoundStyle Round>
struct FastNumericArrayConverter<int8_t, float, N, Round> {
  static_assert(!(N % 4), "N must be multiple of 4.");

  using result_type = Array<int8_t, N>;
  using source_type = Array<float, N>;
  static FloatRoundStyle const round_style = Round;

  HYTLASS_DEVICE
  static result_type convert(source_type const &source) {
    FastNumericArrayConverter<int8_t, float, 4, Round> convert_vector_;

    result_type result;

    Array<int8_t, 4> *result_ptr =
        reinterpret_cast<Array<int8_t, 4> *>(&result);
    Array<float, 4> const *source_ptr =
        reinterpret_cast<Array<float, 4> const *>(&source);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      result_ptr[i] = convert_vector_(source_ptr[i]);
    }

    return result;
  }

  HYTLASS_DEVICE
  result_type operator()(source_type const &s) const { return convert(s); }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines preferred rounding mode for a pair of types
template <typename T, typename S>
struct PreferredRoundingMode {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_to_nearest;
};

#if defined(__HIP_DEVICE_COMPILE__)
/// Defines preferred rounding mode for a pair of types
template <>
struct PreferredRoundingMode<hytlass::tfloat32_t, float> {
  static FloatRoundStyle const kRound = FloatRoundStyle::round_half_ulp_truncate;
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Packs predicates into an array.
template <int N>
struct PackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must pack predicates in a count that is a multiple of 4");

  HYTLASS_HOST_DEVICE
  result_type operator()(bool const predicates[]) {

    result_type packed;
    packed.clear();

    int const kWordSize = 8;
    uint8_t *bytes = reinterpret_cast<uint8_t *>(packed.data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      uint8_t mask = static_cast<uint8_t>((predicates[i] ? 1u : 0u) << bit_idx);
      bytes[word_idx] = (bytes[word_idx] | mask);
    }
    return packed;
  }
};

/// Packs predicates into an array
template <int N>
struct UnpackPredicates {
  using result_type = Array<uint1b_t, N>;

  static_assert(!(N % 4), "Must unpack predicates in a count that is a multiple of 4");

  HYTLASS_HOST_DEVICE
  void operator()(bool predicates[], result_type const &packed) {

    int const kWordSize = 8;
    uint8_t const *bytes = reinterpret_cast<uint8_t const *>(packed.data());

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int word_idx = (i / kWordSize);
      int bit_idx = (i % kWordSize);

      predicates[i] = bool((bytes[word_idx] >> bit_idx) & 0x1);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

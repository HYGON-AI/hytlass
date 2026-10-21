/**************************************************************************************************
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
    \brief Defines a class for using IEEE half-precision floating-point types in host or
      device code.
*/

#pragma once

// TODO: Add support for DTK runtime's FP8 types
#undef HIP_FP8_ENABLED 

#ifdef __GNUC__
// Ignore checks on reinterpret-casts that are being used for bitcasts.
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif

#if (defined(__gfx938__)) && __HIP_DEVICE_COMPILE__
  #define HIP_FP8_CVT_ENABLE 1
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__HIPCC_RTC__)

#include "hytlass/floating_point_hiprtc.h"

#else
//
// Standard Library headers belong here to avoid conflicts with HIPRTC.
//
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstring>
#endif

#ifdef HIP_FP8_ENABLED
#include <hip/hip_fp8.h>
#endif

#include <hip/hip_fp16.h>
#include <hip/amd_detail/amd_hip_fp16.h>


#include "hytlass/hytlass.h"
#include "hytlass/numeric_size.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//  FP8 Has 2 encodings possible : E4M3 and E5M2
//
//  E4M3 : 7  |  6 5 4 3  |  2 1 0
//  E5M2 : 7  |  6 5 4 3 2  |  1 0
//
///////////////////////////////////////////////////////////////////////////////////////////////////


template<typename T>
struct data_type_traits;

template<>
struct data_type_traits<float> {
	static constexpr int NUM_BITS = 32;
	static constexpr int NUM_EXPONENT_BITS = 8;
	static constexpr int NUM_MANTISSA_BITS = 23;
	static constexpr int NUM_EXPECT_SIGN = 31;
	static constexpr uint32_t MAX_NAN = 0x7fffffff;           // max nan
	static constexpr uint32_t INFINITY_MASK = 0x7f800000;     // inf mask
	static constexpr int MAX_EXPONENT  =  127;
	static constexpr int MIN_EXPONENT  = -126;
	static constexpr int EXPONENT_BIAS =  127;

	using storage_type = uint32_t;

    HYTLASS_HOST_DEVICE 
    static bool isfinite(float flt) {
		uint32_t s;
#ifdef __HIPCC__
		s = reinterpret_cast<uint32_t const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
    // Exponent all-ones pattern reserved for Inf/NaN; otherwise valid
	return (s & INFINITY_MASK) < INFINITY_MASK;
	}

    HYTLASS_HOST_DEVICE
    static bool isnan(float flt) {
		uint32_t s;
#ifdef __HIPCC__
		s = reinterpret_cast<uint32_t const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
    return (s & 0x7fffffff) > INFINITY_MASK;
	}

    HYTLASS_HOST_DEVICE 
    static bool isinf(float flt) {
		uint32_t s;
#ifdef __HIPCC__
		s = reinterpret_cast<uint32_t const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
    return (s == INFINITY_MASK) || (s == (INFINITY_MASK | (1 << NUM_EXPECT_SIGN)));
	}
};

template<>
struct data_type_traits<__half> {
	static constexpr int NUM_BITS = 16;
	static constexpr int NUM_EXPONENT_BITS = 5;
	static constexpr int NUM_MANTISSA_BITS = 10;
	static constexpr int NUM_EXPECT_SIGN = 15;
	static constexpr uint32_t MAX_NAN = 0x7fff;
	static constexpr uint32_t INFINITY_MASK = 0x7C00;
	static constexpr int MAX_EXPONENT  =  15;
	static constexpr int MIN_EXPONENT  = -14;
	static constexpr int EXPONENT_BIAS =  15;
	
	using storage_type = uint16_t;

    HYTLASS_HOST_DEVICE
    static bool isfinite(__half	flt) {
		storage_type s;
#ifdef __HIPCC__
		s = reinterpret_cast<storage_type const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
	return (s & INFINITY_MASK) < INFINITY_MASK;
	}

    HYTLASS_HOST_DEVICE
    static bool isnan(__half flt) {
		storage_type s;
#ifdef __HIPCC__
		s = reinterpret_cast<storage_type const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
    return (s & 0x7fff) > INFINITY_MASK;
	}

    HYTLASS_HOST_DEVICE
    static bool isinf(__half flt) {
		storage_type s;
#ifdef __HIPCC__
		s = reinterpret_cast<storage_type const&>(flt);
#else
		std::memcpy(&s, &flt, sizeof(s));
#endif
    return (s == INFINITY_MASK) || (s == (INFINITY_MASK | (1 << NUM_EXPECT_SIGN)));
	}
};


enum class FloatEncoding {
    E4M3,
    E5M2
};

template<FloatEncoding T>
struct alignas(1) float8_base {

    static constexpr bool IS_E4M3 = (T == FloatEncoding::E4M3);
    static constexpr bool IS_E5M2 = (T == FloatEncoding::E5M2);

    // Number of Bits representing mantissa and exponents
    static constexpr int FP32_NUM_BITS = 32;
    static constexpr int FP32_NUM_EXPONENT_BITS = 8;
    static constexpr int FP32_NUM_MANTISSA_BITS = 23;
    static constexpr uint32_t FP32_NAN = 0x7fffffff;
    static constexpr uint32_t FP32_INFINITY_MASK = 0x7f800000;
    static constexpr int FP32_MAX_EXPONENT  =  127;
    static constexpr int FP32_MIN_EXPONENT  = -126;
    static constexpr int FP32_EXPONENT_BIAS =  127;

    static constexpr int FP16_NUM_BITS = 16;
    static constexpr int FP16_NUM_EXPONENT_BITS = 5;
    static constexpr int FP16_NUM_MANTISSA_BITS = 10;
    static constexpr uint16_t FP16_NAN = 0x7fff;
    static constexpr uint16_t FP16_INFINITY_MASK = 0x7c00;
    static constexpr int FP16_MAX_EXPONENT  = 15;
    static constexpr int FP16_MIN_EXPONENT  = -14;
    static constexpr int FP16_EXPONENT_BIAS = 15;

    static constexpr int FP8_NUM_BITS = 8;
    static constexpr int FP8_NUM_EXPONENT_BITS = IS_E4M3 ? 4 : 5;
    static constexpr int FP8_NUM_MANTISSA_BITS = IS_E4M3 ? 3 : 2;
    static constexpr uint8_t  FP8_NAN = 0x7f; // Also F8_INF
    static constexpr uint8_t  FP8_INFINITY_MASK = IS_E4M3 ? 0x78 : 0x7c;
    static constexpr int FP8_MAX_EXPONENT  = IS_E4M3 ?  7 :  15;
    static constexpr int FP8_MIN_EXPONENT  = IS_E4M3 ? -6 : -14;
    static constexpr int FP8_EXPONENT_BIAS = IS_E4M3 ?  7 :  15;

    static constexpr uint8_t  FP8_EXPONENT_MASK = (1 << FP8_NUM_EXPONENT_BITS) - 1;
    static constexpr uint8_t  FP8_MANTISSA_MASK = (1 << FP8_NUM_MANTISSA_BITS) - 1;

    static constexpr uint8_t FP8_MAX_FLT = (IS_E4M3 ? 0x7e : 0x7b);

    // 256 in float
    static constexpr uint32_t FP8_SAT_VAL_FP32 = 0x43800000;

    //
    // Data members
    //

    /// Data container
    uint8_t storage;

    /// Ctors.
    HYTLASS_HOST_DEVICE
    float8_base() : storage(0) { }

    /// Is finite implementation
    HYTLASS_HOST_DEVICE
    static bool isfinite(float flt) {
        uint32_t s;

        #if defined(__HIP_DEVICE_COMPILE__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7f800000) < 0x7f800000;
    }

    /// Is NaN implementation
    HYTLASS_HOST_DEVICE
    static bool isnan(float flt) {
        uint32_t s;

        #if defined(__HIP_DEVICE_COMPILE__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        return (s & 0x7fffffff) > 0x7f800000;
    }

    /// Is infinite implementation
    HYTLASS_HOST_DEVICE
    static bool isinf(float flt) {
        uint32_t s;

        #if defined(__HIP_DEVICE_COMPILE__)
        s = reinterpret_cast<uint32_t const &>(flt);
        #else
        std::memcpy(&s, &flt, sizeof(s));
        #endif

        // Sign = 0 for +inf, 1 for -inf
        // Exponent = all ones
        // Mantissa = all zeros
        return (s == 0x7f800000) || (s == 0xff800000);
    }

#if defined(HIP_FP8_CVT_ENABLE)
    template<bool SAT = true>
    HYTLASS_DEVICE
    static uint8_t convert_float_to_fp8_device_impl(float const& flt) {
        union {
            float fval;
            uint32_t uval;
        } val;

	val.fval = flt;

        if constexpr (SAT) {
            if ((val.uval & 0x7F800000) != 0x7F800000) {
                if constexpr (IS_E4M3) {
                    val.fval = (val.fval < -448.0f)   ? -448.0f
                             : (val.fval >  448.0f)   ?  448.0f
                                                      :  val.fval;
                } else {
                    val.fval = (val.fval < -57344.0f) ? -57344.0f
                             : (val.fval >  57344.0f) ?  57344.0f
                                                      :  val.fval;
                }
            }
        }

        uint32_t dst;

 #pragma clang diagnostic push
 #pragma clang diagnostic ignored "-Wuninitialized"
        // __builtin_amdgcn_cvt_pk_fp8_f32() this builtin require the old value, and
        // will generate a v_mov_b32 vxxx [old] before cvt, which result in unwanted ISA
        // so we prepare an uninitialized variable purposely, and turn off the warning
        int dummy_old;
        float tmp = 0.0f;
        if constexpr (IS_E4M3) {
            // float32 -> fp8
            dst = __builtin_hcu_cvt_pk_fp8_f32(val.fval, val.fval, dummy_old, false);
        } else {
            dst = __builtin_hcu_cvt_pk_bf8_f32(val.fval, val.fval, dummy_old, false);
        }
 #pragma clang diagnostic pop

        return (dst & 0xFF);
    }

    HYTLASS_DEVICE
    static float convert_fp8_to_float_device_impl(uint8_t const& x) {
        float result = float();
        uint32_t val = static_cast<uint32_t>(x);
        if constexpr (IS_E4M3) {
            asm volatile("v_cvt_f32_fp8 %0 %1\n\t": "=v"(result): "v"(val));
        } else {
            asm volatile("v_cvt_f32_bf8 %0 %1\n\t": "=v"(result): "v"(val));
        }
        return result;
    }
#endif

	// software implementation rounds toward nearest even
	template<typename SRC_TYPE, bool SAT = true>
    HYTLASS_HOST_DEVICE
	static uint8_t convert_to_fp8(SRC_TYPE const& flt) {
        using src_desc = data_type_traits<SRC_TYPE>;
		using src_stg_type = typename src_desc::storage_type;
		src_stg_type s;

        static_assert(sizeof_bits<SRC_TYPE>::value == 16 || sizeof_bits<SRC_TYPE>::value == 32);

		#if defined(__HIPCC__)
		s = reinterpret_cast<src_stg_type const &>(flt);
		#else
		std::memcpy(&s, &flt, sizeof(s));
		#endif

		// Extract the bits in the SRC type
		uint8_t sign = uint8_t((s >> (src_desc::NUM_BITS - 8) & 0x80));

		int exp = int(
			(s >> src_desc::NUM_MANTISSA_BITS) & ((1 << src_desc::NUM_EXPONENT_BITS) - 1)) - 
		  src_desc::EXPONENT_BIAS;

		int mantissa = s & ((1 << src_desc::NUM_MANTISSA_BITS) - 1);

		uint8_t u = 0;
        // Align with ISA behavior.
		uint8_t const kF8_NaN = IS_E4M3 ? 0x7f : 0x7e;
        // Since OCP E4M3 has no Inf, any operation resulting in infinity must yield NaN.
		uint8_t const kF8_INF = IS_E4M3 ? kF8_NaN : FP8_INFINITY_MASK;

		// NaN => NaN
		if (src_desc::isnan(flt)) {
            return kF8_NaN;
		}

		// Inf => MAX_FLT (satfinite)
		if (src_desc::isinf(flt)) {
            if constexpr (SAT) {
				return sign | FP8_MAX_FLT;
            } else {
                return sign | kF8_INF;
            }
		}

		int sticky_bit = 0;
		bool skip_sign = false;

		if ( (exp >= FP8_MIN_EXPONENT) && (exp <= FP8_MAX_EXPONENT) ) {
            // normal fp32 to normal fp8
            exp = exp + FP8_EXPONENT_BIAS;
            u = uint8_t((uint32_t(exp) & FP8_EXPONENT_MASK) << FP8_NUM_MANTISSA_BITS);
            u = uint8_t(u | (mantissa >> (src_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)));
		} else if(exp < FP8_MIN_EXPONENT) {
            // for snorm
            int rshift = (FP8_MIN_EXPONENT - exp);
            if ((s & src_desc::INFINITY_MASK) == 0) {
                // If the exp bits of src are all 0, which means it is an snorm.
                u = (uint8_t(mantissa >> (src_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            } else if (rshift < src_desc::NUM_MANTISSA_BITS) {
                // source is norm
                mantissa |= (1 << src_desc::NUM_MANTISSA_BITS);
                sticky_bit = ((mantissa & ((1 << rshift) - 1)) != 0);
                mantissa = (mantissa >> rshift);
                u = (uint8_t(mantissa >> (src_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS)) & FP8_MANTISSA_MASK);
            } else {
                mantissa = 0;
                u = 0;
            }
		// Exponent > FP8_MAX_EXPONENT - this is a special case done to match HW
		// 0x4380_0000 to 0x43e0_0000 - maps from 256 to 448, and does not saturate / inf.
		} else {
            if( exp == (FP8_MAX_EXPONENT + 1) ) {
                uint8_t mantissa_tmp = uint8_t(mantissa >> (src_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                if( mantissa_tmp < FP8_MANTISSA_MASK) {
                    exp = exp + FP8_EXPONENT_BIAS;
                    u = uint8_t(uint32_t(exp) << FP8_NUM_MANTISSA_BITS) | mantissa_tmp;
                    // Prevent overflow after rounding.
                    // may_be_nan =  (mantissa_tmp == (FP8_MANTISSA_MASK-1));
                } else {
                    if constexpr (SAT) {
                        return (sign | FP8_MAX_FLT);
                    } else {
                        return (sign | kF8_INF);
                    }
                }
            } else{
                if constexpr (SAT) {
                    return (sign | FP8_MAX_FLT);
                } else {
                    return (sign | kF8_INF);
                }
            }
		}
		
		// round to nearest even
		int NUM_BITS_SHIFT = src_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS - 1;
		int round_bit = ((mantissa >> NUM_BITS_SHIFT) & 1);
		sticky_bit |= ((mantissa & ((1 << NUM_BITS_SHIFT) - 1)) != 0);

		if ((round_bit && sticky_bit) || (round_bit && (u & 1))) {
            u = uint8_t(u + 1);
		}

		if (u > FP8_MAX_FLT) {
            if constexpr (SAT) {
                u = (sign | FP8_MAX_FLT);
            } else {
                u = (sign | kF8_INF);
            }
            return u;
		}

        u |= sign;

		return u;
	}

    /// FP32 -> FP8 conversion - rounds to nearest even
    template<bool SAT = true>
    HYTLASS_HOST_DEVICE
    static uint8_t convert_float_to_fp8(float const& flt) {
#if defined(HIP_FP8_CVT_ENABLE)
      return convert_float_to_fp8_device_impl<SAT>(flt);
#else
      return convert_to_fp8<float, SAT>(flt);
#endif
    }

	template<typename DST_TYPE>
    HYTLASS_HOST_DEVICE
	static auto convert_from_fp8(uint8_t x)-> DST_TYPE {
		using dst_desc = data_type_traits<DST_TYPE>;
		using dst_stg_type = typename dst_desc::storage_type;

        static_assert(sizeof_bits<DST_TYPE>::value == 16 || 
                      sizeof_bits<DST_TYPE>::value == 32);

		dst_stg_type constexpr src_NaN = dst_desc::MAX_NAN;

		uint8_t const& fp8 = x;
		uint32_t sign = (fp8 >> (FP8_NUM_BITS - 1)) & 1;

		uint32_t exp = (fp8 >> FP8_NUM_MANTISSA_BITS) & FP8_EXPONENT_MASK;
		uint32_t mantissa = fp8 & FP8_MANTISSA_MASK;

		dst_stg_type f = (sign << (dst_desc::NUM_BITS - 1));

		if (IS_E4M3 && exp == 15 && mantissa == 0x7) {
			f = src_NaN;
		} else if (exp > 0 && (IS_E4M3 || exp < (FP8_MAX_EXPONENT + FP8_EXPONENT_BIAS + 1))) {
			exp += (dst_desc::EXPONENT_BIAS - FP8_EXPONENT_BIAS);

			f = f | 
			  (exp << dst_desc::NUM_MANTISSA_BITS) | 
				(mantissa << (dst_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
		} else if (exp == 0) {
			// snorm
			if (mantissa) {
                if constexpr (dst_desc::NUM_EXPONENT_BITS == FP8_NUM_EXPONENT_BITS) {
                    f = f | (mantissa << (dst_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                } else {
                    exp += (dst_desc::EXPONENT_BIAS - FP8_EXPONENT_BIAS) + 1;
                    while ((mantissa & (1 << FP8_NUM_MANTISSA_BITS)) == 0) {
                        mantissa <<= 1;
                        exp--;
                    }
                    mantissa &= FP8_MANTISSA_MASK;

                    f = f | 
                        (exp << dst_desc::NUM_MANTISSA_BITS) | 
                            (mantissa << (dst_desc::NUM_MANTISSA_BITS - FP8_NUM_MANTISSA_BITS));
                }
            }
		} else {
			if (mantissa == 0) {
				f = (f | dst_desc::INFINITY_MASK);
			} else {
				f = src_NaN;
			}
		}

		#ifdef __HIPCC__
		return reinterpret_cast<DST_TYPE const&>(f);
		#else
		DST_TYPE flt;
		std::memcpy(&flt, &f, sizeof(flt));
		return flt;
		#endif
	}

    /// Converts a fp8 value stored as a uint8_t to a float
    HYTLASS_HOST_DEVICE
    static float convert_fp8_to_float(uint8_t const& x) {
#if defined(HIP_FP8_CVT_ENABLE)
      return convert_fp8_to_float_device_impl(x);
#else
      return convert_from_fp8<float>(x);
#endif
    }
};


// Forward declaration of float_e5m2_t to define float_e4m3_t <=> float_e5m2_t
// conversions in class float_e4m3_t
struct float_e5m2_t;


///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E4M3
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e4m3_t : float8_base<FloatEncoding::E4M3> {

    using Base = float8_base<FloatEncoding::E4M3>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t
    HYTLASS_HOST_DEVICE
    static float_e4m3_t bitcast(uint8_t x) {
        float_e4m3_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    template<bool SAT = true>
    HYTLASS_HOST_DEVICE
    static float_e4m3_t from_float(float const& flt) {
        return bitcast(Base::convert_float_to_fp8<SAT>(flt));
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    template<bool SAT = true>
    HYTLASS_HOST_DEVICE
    static float_e4m3_t from_half(__half const& flt) {
#if defined(HIP_FP8_CVT_ENABLE)
        return bitcast(Base::convert_float_to_fp8<SAT>(__half2float(flt)));
#else
        return bitcast(Base::convert_to_fp8<__half, SAT>(flt));
#endif
    }

    // E4M3 -> half
    HYTLASS_HOST_DEVICE
    static __half to_half(float_e4m3_t const& x) {
#if defined(HIP_FP8_CVT_ENABLE)
        return __float2half(Base::convert_fp8_to_float(x.storage));
#else
        return Base::convert_from_fp8<__half>(x.storage);
#endif
    }

    // E4M3 -> Float
    HYTLASS_HOST_DEVICE
    static float to_float(float_e4m3_t const& x) {
        return Base::convert_fp8_to_float(x.storage);
    }

    //
    // Methods
    //

    /// Constructor inheritance
    using Base::Base;

    /// Default constructor
    float_e4m3_t() = default;

#ifdef HIP_FP8_ENABLED
    /// Conversion from HIP's FP8 type
    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(__nv_fp8_e4m3 x) {
        storage = x.__x;
    }
#endif

    /// Floating point conversion
    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(float x) {
        storage = from_float(x).storage;
    }

    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(__half x) {
        storage = from_half(x).storage;
    }

    /// Floating point conversion
    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(double x): float_e4m3_t(float(x)) {
    }

    /// Integer conversion
    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(int x): float_e4m3_t(float(x)) {
    }

    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(unsigned x): float_e4m3_t(float(x)) {
    }

    /// E5M2 conversion. Defined after float_e5m2_t is defined.
    HYTLASS_HOST_DEVICE
    explicit float_e4m3_t(float_e5m2_t x);

#ifdef HIP_FP8_ENABLED
    /// Assignment from HIP's FP8 type
    HYTLASS_HOST_DEVICE
    float_e4m3_t & operator=(__nv_fp8_e4m3 x) {
        storage = x.__x;
        return *this;
    }
#endif

    /// Converts to float
    HYTLASS_HOST_DEVICE
    operator float() const {
        return to_float(*this);
    }

    /// Converts to half
    HYTLASS_HOST_DEVICE
    operator __half() const {
        return to_half(*this);
    }

    /// Converts to float
    HYTLASS_HOST_DEVICE
    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int
    HYTLASS_HOST_DEVICE
    explicit operator int() const {
    #if defined(__HIP_DEVICE_COMPILE__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool
    HYTLASS_HOST_DEVICE
    explicit operator bool() const {
    #if defined(__HIP_DEVICE_COMPILE__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state
    HYTLASS_HOST_DEVICE
    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state
    HYTLASS_HOST_DEVICE
    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit
    HYTLASS_HOST_DEVICE
    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    HYTLASS_HOST_DEVICE
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    HYTLASS_HOST_DEVICE
    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa
    HYTLASS_HOST_DEVICE
    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }
};
///////////////////////////////////////////////////////////////
///
/// floating-point 8 type : E5M2
///
///////////////////////////////////////////////////////////////
struct alignas(1) float_e5m2_t : float8_base<FloatEncoding::E5M2> {

    using Base = float8_base<FloatEncoding::E5M2>;

    static constexpr int MAX_EXPONENT = Base::FP8_MAX_EXPONENT;

    //
    // Static conversion operators
    //

    /// Constructs from an uint8_t
    HYTLASS_HOST_DEVICE
    static float_e5m2_t bitcast(uint8_t x) {
        float_e5m2_t f;
        f.storage = x;
        return f;
    }

    /// FP32 -> FP8 conversion - rounds to nearest even
    template<bool SAT = true>
    HYTLASS_HOST_DEVICE
    static float_e5m2_t from_float(float const& flt) {
        return bitcast(Base::convert_float_to_fp8<SAT>(flt));
    }

    /// FP16 -> E5M2 conversion - rounds to nearest even
    template<bool SAT = true>
    HYTLASS_HOST_DEVICE
    static float_e5m2_t from_half(__half const& flt) {
#if defined(HIP_FP8_CVT_ENABLE)
    return bitcast(Base::convert_float_to_fp8<SAT>(__half2float(flt)));
#else
        uint16_t s;
        float_e5m2_t res;
        #ifdef __HIPCC__
        s = reinterpret_cast<uint16_t const&>(flt);
        #else
        memcpy(&s, &flt, sizeof(uint16_t));
        #endif

        uint8_t sign = (s & 0x8000) >> 8;

        constexpr uint8_t kFP8_INF = SAT ? 0x7B : 0x7C;

        if ((s & 0x7C00) == 0x7C00) {
            // Determine whether to return NaN or max FP8 based on the mantissa bits.
            uint8_t exp_and_mantissa = (s & 0x03FF) ? 0x7E : kFP8_INF;
            res = float_e5m2_t::bitcast(((s & 0x8000) >> 8) | exp_and_mantissa);
            return res;
        } else if (~s & 0x7C00) {
            // Assume the FP16 value is split into: S (kept bits), L (least kept bit), and R (discarded bits).
            // R determines whether to round up L:
            //   - If R < 0x80: no rounding up.
            //   - If R > 0x80: always round up.
            //   - If R == 0x80: apply round-to-nearest-even:
            //        Round up if L == 1 (result would be odd).
            //        Do not round up if L == 0 (result would be even).
            uint16_t round_bit = 0x7F + ((s >> 8) & 1);
            s += round_bit;
        }

        // Extra check to prevent overflow.
        if ((s & 0x7C00) == 0x7C00) {
            res = float_e5m2_t::bitcast(sign | kFP8_INF);
            return res;
        }

        res = float_e5m2_t::bitcast(s >> 8);
        return res;
#endif
    }

    // E5M2 -> half
    HYTLASS_HOST_DEVICE
    static half to_half(float_e5m2_t const& x) {
        uint8_t s = x.storage;
        uint16_t data = s << 8;
#if defined(__HIPCC__)
        return reinterpret_cast<__half const&>(data);
#else
        __half res;
        std::memcpy(&res, &data, sizeof(uint16_t));
        return res;
#endif
    }

    // E5M2 -> Float
    HYTLASS_HOST_DEVICE
    static float to_float(float_e5m2_t const& x) {
        return Base::convert_fp8_to_float(x.storage);
    }

    //
    // Methods
    //

    /// Constructor inheritance
    using Base::Base;

    /// Default constructor
    float_e5m2_t() = default;

#ifdef HIP_FP8_ENABLED
    /// Conversion from HIP's FP8 type
    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(__nv_fp8_e5m2 x) {
        storage = x.__x;
    }
#endif

    /// Floating point conversion
    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(float x) {
        storage = from_float(x).storage;
    }

    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(__half x) {
      storage = from_half(x).storage;
    }

    /// Floating point conversion
    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(double x): float_e5m2_t(float(x)) {
    }

    /// Integer conversion
    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(int x): float_e5m2_t(float(x)) {
    }

    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(unsigned x): float_e5m2_t(float(x)) {
    }

    /// E4M3 conversion
    HYTLASS_HOST_DEVICE
    explicit float_e5m2_t(float_e4m3_t x);

#ifdef HIP_FP8_ENABLED
    /// Assignment from HIP's FP8 type
    HYTLASS_HOST_DEVICE
    float_e5m2_t & operator=(__nv_fp8_e5m2 x) {
        storage = x.__x;
        return *this;
    }
#endif

    /// Converts to float
    HYTLASS_HOST_DEVICE
    operator float() const {
        return to_float(*this);
    }

    /// Converts to half
    HYTLASS_HOST_DEVICE
    operator __half() const {
      return to_half(*this);
    }

    /// Converts to float
    HYTLASS_HOST_DEVICE
    explicit operator double() const {
        return double(to_float(*this));
    }

    /// Converts to int
    HYTLASS_HOST_DEVICE
    explicit operator int() const {
    #if defined(__HIP_DEVICE_COMPILE__)
        return __half2int_rn(to_half(*this));
    #else
        return int(to_float(*this));
    #endif
    }

    /// Casts to bool
    HYTLASS_HOST_DEVICE
    explicit operator bool() const {
    #if defined(__HIP_DEVICE_COMPILE__)
        return bool(__half2int_rn(to_half(*this)));
    #else
        return bool(int(to_float(*this)));
    #endif
    }

    /// Accesses raw internal state
    HYTLASS_HOST_DEVICE
    uint8_t& raw() {
        return storage;
    }

    /// Accesses raw internal state
    HYTLASS_HOST_DEVICE
    uint8_t raw() const {
        return storage;
    }

    /// Returns the sign bit
    HYTLASS_HOST_DEVICE
    bool signbit() const {
        return ((storage & (1 << (Base::FP8_NUM_BITS - 1))) != 0);
    }

    /// Returns the biased exponent
    HYTLASS_HOST_DEVICE
    int exponent_biased() const {
        return int((storage >> FP8_NUM_MANTISSA_BITS) & Base::FP8_EXPONENT_MASK);
    }

    /// Returns the unbiased exponent
    HYTLASS_HOST_DEVICE
    int exponent() const {
        return exponent_biased() - 15;
    }

    /// Returns the mantissa
    HYTLASS_HOST_DEVICE
    int mantissa() const {
        return int(storage & Base::FP8_MANTISSA_MASK);
    }
};
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Arithmetic operators
//
///////////////////////////////////////////////////////////////////////////////////////////////////

HYTLASS_HOST_DEVICE
bool operator==(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) == float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator!=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) != float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator<(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) < float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator<=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) <= float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator>(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) > float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator>=(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float(lhs) >= float(rhs);
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator+(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) + float(rhs));
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator-(float_e4m3_t const& lhs) {
    return float_e4m3_t(-float(lhs));
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator-(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) - float(rhs));
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator*(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) * float(rhs));
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator/(float_e4m3_t const& lhs, float_e4m3_t const& rhs) {
    return float_e4m3_t(float(lhs) / float(rhs));
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator+=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) + float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator-=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) - float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator*=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) * float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator/=(float_e4m3_t & lhs, float_e4m3_t const& rhs) {
    lhs = float_e4m3_t(float(lhs) / float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator++(float_e4m3_t & lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_e4m3_t(tmp);
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t& operator--(float_e4m3_t & lhs) {
    float tmp(lhs);
    --tmp;
    lhs = float_e4m3_t(tmp);
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator++(float_e4m3_t & lhs, int) {
    float_e4m3_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_e4m3_t(tmp);
    return ret;
}

HYTLASS_HOST_DEVICE
float_e4m3_t operator--(float_e4m3_t & lhs, int) {
    float_e4m3_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = float_e4m3_t(tmp);
    return ret;
}

HYTLASS_HOST_DEVICE
bool operator==(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) == float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator!=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) != float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator<(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) < float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator<=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) <= float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator>(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) > float(rhs);
}

HYTLASS_HOST_DEVICE
bool operator>=(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float(lhs) >= float(rhs);
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator+(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) + float(rhs));
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator-(float_e5m2_t const& lhs) {
    return float_e5m2_t(-float(lhs));
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator-(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) - float(rhs));
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator*(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) * float(rhs));
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator/(float_e5m2_t const& lhs, float_e5m2_t const& rhs) {
    return float_e5m2_t(float(lhs) / float(rhs));
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator+=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) + float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator-=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) - float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator*=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) * float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator/=(float_e5m2_t & lhs, float_e5m2_t const& rhs) {
    lhs = float_e5m2_t(float(lhs) / float(rhs));
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator++(float_e5m2_t & lhs) {
    float tmp(lhs);
    ++tmp;
    lhs = float_e5m2_t(tmp);
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t& operator--(float_e5m2_t & lhs) {
    float tmp(lhs);
    --tmp;
    lhs = float_e5m2_t(tmp);
    return lhs;
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator++(float_e5m2_t & lhs, int) {
    float_e5m2_t ret(lhs);
    float tmp(lhs);
    tmp++;
    lhs = float_e5m2_t(tmp);
    return ret;
}

HYTLASS_HOST_DEVICE
float_e5m2_t operator--(float_e5m2_t & lhs, int) {
    float_e5m2_t ret(lhs);
    float tmp(lhs);
    tmp--;
    lhs = float_e5m2_t(tmp);
    return ret;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// float_e4m3_t <=> float_e5m2_t conversions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// float_e4m3_t <= float_e5m2_t
HYTLASS_HOST_DEVICE
float_e4m3_t::float_e4m3_t(float_e5m2_t x) {
    storage = from_float(float_e5m2_t::to_float(x)).storage;
}

/// float_e5m2_t <= float_e4m3_t
HYTLASS_HOST_DEVICE
float_e5m2_t::float_e5m2_t(float_e4m3_t x) {
    storage = from_float(float_e4m3_t::to_float(x)).storage;
}

///////////////////////////////////////////////////////////////
///
/// Umbrella floating-point 8-bit data type : type_erased_dynamic_float8_t
/// This umbrella datatype can be enabled when a user provides a specific
/// datatype in runtime argument list.
///
/// Currently supported runtime datatypes compatible with type_erased_dynamic_float8_t:
///   QMMAFormat::E5M2
///   QMMAFormat::E4M3
///
///////////////////////////////////////////////////////////////

union type_erased_dynamic_float8_t {
  uint8_t data;
  hytlass::float_e5m2_t e5m2;
  hytlass::float_e4m3_t e4m3;
  HYTLASS_HOST_DEVICE
  explicit operator hytlass::float_e5m2_t() const {
    return e5m2;
  }

  HYTLASS_HOST_DEVICE
  explicit operator hytlass::float_e4m3_t() const {
    return e4m3;
  }

};

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Standard Library operations and definitions
//
///////////////////////////////////////////////////////////////////////////////////////////////////

#if !defined(__HIPCC_RTC__)
namespace std {

/// Numeric limits common to all float8 types
template <typename T>
struct float8_base_numeric_limits {
private:
  using F8Type = T;
public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
  static std::float_denorm_style const has_denorm = std::denorm_present;
  static bool const has_denorm_loss = true;
  static std::float_round_style const round_style = std::round_to_nearest;
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = F8Type::FP8_NUM_MANTISSA_BITS;

  /// Least positive value
  HYTLASS_HOST_DEVICE
  static F8Type min() { return F8Type::bitcast(0x01); }

  /// Maximum finite value
  HYTLASS_HOST_DEVICE
  static F8Type max() { return F8Type::bitcast(F8Type::FP8_MAX_FLT); }

  /// Returns maximum rounding error
  HYTLASS_HOST_DEVICE
  static F8Type round_error() { return F8Type(0.5f); }

  /// Returns positive infinity value
  HYTLASS_HOST_DEVICE
  static F8Type infinity() { return F8Type::bitcast(F8Type::FP8_INFINITY_MASK); }

  /// Returns quiet NaN value
  HYTLASS_HOST_DEVICE
  static F8Type quiet_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns signaling NaN value
  HYTLASS_HOST_DEVICE
  static F8Type signaling_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns smallest positive subnormal value
  HYTLASS_HOST_DEVICE
  static F8Type denorm_min() { return F8Type::bitcast(0x01); }
};

/// Numeric limits for float_e4m3_t
template <>
struct numeric_limits<hytlass::float_e4m3_t> :
    public float8_base_numeric_limits<hytlass::float_e4m3_t> {
  static bool const has_infinity = false;

  /// Minimum finite value
  static hytlass::float_e4m3_t lowest() { return hytlass::float_e4m3_t::bitcast(0xfe); }

  /// Machine epsilon, that is, the difference between 1.0 and the next representable value
  static hytlass::float_e4m3_t epsilon() { return hytlass::float_e4m3_t::bitcast(0x20); }
};

/// Numeric limits for float_e5m2_t
template <>
struct numeric_limits<hytlass::float_e5m2_t>  :
    public float8_base_numeric_limits<hytlass::float_e5m2_t> {
  static bool const has_infinity = true;

  /// Minimum finite value
  static hytlass::float_e5m2_t lowest() { return hytlass::float_e5m2_t::bitcast(0xfb); }

  /// Machine epsilon, that is, the difference between 1.0 and the next representable value
  static hytlass::float_e5m2_t epsilon() { return hytlass::float_e5m2_t::bitcast(0x34); }
};

}  // namespace std
#endif

namespace platform {

/// Numeric limits common to all float8 types
template <typename T>
struct float8_base_numeric_limits {
private:
  using F8Type = T;
public:
  static bool const is_specialized = true;
  static bool const is_signed = true;
  static bool const is_integer = false;
  static bool const is_exact = false;
  static bool const has_quiet_NaN = true;
  static bool const has_signaling_NaN = false;
#if !defined(__HIPCC_RTC__)
  static std::float_denorm_style const has_denorm = std::denorm_present;
#endif
  static bool const has_denorm_loss = true;
#if !defined(__HIPCC_RTC__)
  static std::float_round_style const round_style = std::round_to_nearest;
#endif
  static bool const is_iec559 = false;
  static bool const is_bounded = true;
  static bool const is_modulo = false;
  static int const digits = F8Type::FP8_NUM_MANTISSA_BITS;

  /// Least positive value
  HYTLASS_HOST_DEVICE
  static F8Type min() { return F8Type::bitcast(0x01); }

  /// Maximum finite value
  HYTLASS_HOST_DEVICE
  static F8Type max() { return F8Type::bitcast(F8Type::FP8_MAX_FLT); }

  /// Returns maximum rounding error
  HYTLASS_HOST_DEVICE
  static F8Type round_error() { return F8Type(0.5f); }

  /// Returns positive infinity value
  HYTLASS_HOST_DEVICE
  static F8Type infinity() { return F8Type::bitcast(F8Type::FP8_INFINITY_MASK); }

  /// Returns quiet NaN value
  HYTLASS_HOST_DEVICE
  static F8Type quiet_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns signaling NaN value
  HYTLASS_HOST_DEVICE
  static F8Type signaling_NaN() { return F8Type::bitcast(F8Type::FP8_NAN); }

  /// Returns smallest positive subnormal value
  HYTLASS_HOST_DEVICE
  static F8Type denorm_min() { return F8Type::bitcast(0x01); }
};

/// std::numeric_limits
template <class T>
struct numeric_limits;

/// Numeric limits for float_e4m3_t
template <>
struct numeric_limits<hytlass::float_e4m3_t> :
    public float8_base_numeric_limits<hytlass::float_e4m3_t> {
  static bool const has_infinity = false;

  /// Minimum finite value
  static hytlass::float_e4m3_t lowest() { return hytlass::float_e4m3_t::bitcast(0xfe); }

  /// Machine epsilon, that is, the difference between 1.0 and the next representable value
  static hytlass::float_e4m3_t epsilon() { return hytlass::float_e4m3_t::bitcast(0x20); }
};

/// Numeric limits for float_e5m2_t
template <>
struct numeric_limits<hytlass::float_e5m2_t>  :
    public float8_base_numeric_limits<hytlass::float_e5m2_t> {
  static bool const has_infinity = true;

  /// Minimum finite value
  static hytlass::float_e5m2_t lowest() { return hytlass::float_e5m2_t::bitcast(0xfb); }

  /// Machine epsilon, that is, the difference between 1.0 and the next representable value
  static hytlass::float_e5m2_t epsilon() { return hytlass::float_e5m2_t::bitcast(0x34); }
};

}  // namespace platform

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// User-defined literals
//

HYTLASS_HOST_DEVICE
hytlass::float_e4m3_t operator "" _fe4m3(long double x) {
  return hytlass::float_e4m3_t(float(x));
}

HYTLASS_HOST_DEVICE
hytlass::float_e4m3_t operator "" _fe4m3(unsigned long long int x) {
  return hytlass::float_e4m3_t(int(x));
}

HYTLASS_HOST_DEVICE
hytlass::float_e5m2_t operator "" _fe5m2(long double x) {
  return hytlass::float_e5m2_t(float(x));
}

HYTLASS_HOST_DEVICE
hytlass::float_e5m2_t operator "" _fe5m2(unsigned long long int x) {
  return hytlass::float_e5m2_t(int(x));
}

/////////////////////////////////////////////////////////////////////////////////////////////////


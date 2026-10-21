/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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
    \brief Matrix multiply for Gfx928
*/

#pragma once

#include "hytlass/arch/mmac_gfx9.h"

#include <assert.h>

#include "hytlass/arch/mma.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/numeric_types.h"


////////////////////////////////////////////////////////////////////////////////
namespace hytlass::arch {
  typedef float v4f __attribute__((ext_vector_type(4)));
  typedef float v2f __attribute__((ext_vector_type(2)));
  typedef int intx4_t __attribute__((ext_vector_type(4)));
  typedef __fp16 __fp16x4_t __attribute__((ext_vector_type(4)));
  typedef short __bf16x4_t __attribute__((ext_vector_type(4)));
  typedef __fp16 __fp16x8_t __attribute__((ext_vector_type(8)));
  typedef int8_t __i8x16_t __attribute__((ext_vector_type(16)));  
}

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace arch {

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16168 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = F32 * F32 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 8>,
  64,
  float,
  layout::RowMajor,
  float,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 8>;
  using ElementA = float;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<float, 2>;
  using ElementB = float;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<float, 2>;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<float, 4>;
  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
          FragmentC &d,
          FragmentA const &a,
          FragmentB const &b,
          FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d;
    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];
    v2f A;
    v2f B;

    A = *(reinterpret_cast<v2f *>(&const_cast<FragmentA &>(a)));
    B = *(reinterpret_cast<v2f *>(&const_cast<FragmentB &>(b)));

    _d = hytlass::arch::detail::mmac_f32_16x16x8_f32(B, A, _c);

    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;

#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 16168 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = TF32 * TF32 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 8>,
  64,
  tfloat32_t,
  layout::RowMajor,
  tfloat32_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 8>;
  using ElementA = tfloat32_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 2>;
  using ElementB = tfloat32_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 2>;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;
  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d;
    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    v2f A = *(reinterpret_cast<v2f *>(&const_cast<FragmentA &>(a)));
    v2f B = *(reinterpret_cast<v2f *>(&const_cast<FragmentB &>(b)));
    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d = hytlass::arch::detail::mmac_f32_16x16x8_tf32(B, A, _c);

    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161616 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = f16 * f16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 4>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    __fp16x4_t A, B;
    A = *(reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentA &>(a)));
    B = *(reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentB &>(b)));
    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d = hytlass::arch::detail::mmac_f32_16x16x16_f16(B, A, _c);

    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161616 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 4>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 4>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d;
    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];
    __bf16x4_t A, B;
    A = *(reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentA &>(a)));
    B = *(reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentB &>(b)));

    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension      
    _d = hytlass::arch::detail::mmac_f32_16x16x16_bf16(B, A, _c);
    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161632 - int32_t accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - int32_t = int8_t * int8_t + int32_t
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  64,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 8>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {

#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    intx4_t _c;
    intx4_t _d;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    const long _a = *(reinterpret_cast<const long *>(&a));
    const long _b = *(reinterpret_cast<const long *>(&b));
    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d = hytlass::arch::detail::mmac_i32_16x16x32_i8(_b, _a, _c);

    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161616 - int32_t accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - int32_t = uint8_t * uint8_t + int32_t
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  64,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<ElementA, 8>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<ElementB, 8>;

  using ElementC = int;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    intx4_t _c;
    intx4_t _d;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    const long _a = *(reinterpret_cast<const long *>(&a));
    const long _b = *(reinterpret_cast<const long *>(&b));
    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d = hytlass::arch::detail::mmac_i32_16x16x32_u8(_b, _a, _c);

    d[0] = _d.x;
    d[1] = _d.y;
    d[2] = _d.z;
    d[3] = _d.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix Multiply 161632 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = f16 * f16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  64,
  half_t,
  layout::RowMajor,
  half_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = half_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<half_t, 8>;

  using ElementB = half_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<half_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d0;
    v4f _d1;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    __fp16x4_t A0, A1, B0, B1;
    A0 = reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentA &>(a))[0];
    B0 = reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentB &>(b))[0];
    A1 = reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentA &>(a))[1];
    B1 = reinterpret_cast<__fp16x4_t *>(&const_cast<FragmentB &>(b))[1];

    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d0 = hytlass::arch::detail::mmac_f32_16x16x16_f16(B0, A0, _c);
    _d1 = hytlass::arch::detail::mmac_f32_16x16x16_f16(B1, A1, _d0);

    d[0] = _d1.x;
    d[1] = _d1.y;
    d[2] = _d1.z;
    d[3] = _d1.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

//
// Matrix Multiply 161632 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - F32 = bf16 * bf16 + F32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 32>,
  64,
  bfloat16_t,
  layout::RowMajor,
  bfloat16_t,
  layout::ColumnMajor,
  float,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 32>;

  using ElementA = bfloat16_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<bfloat16_t, 8>;

  using ElementB = bfloat16_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<bfloat16_t, 8>;

  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _c;
    v4f _d0;
    v4f _d1;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    __bf16x4_t A0, A1, B0, B1;
    A0 = reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentA &>(a))[0];
    B0 = reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentB &>(b))[0];
    A1 = reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentA &>(a))[1];
    B1 = reinterpret_cast<__bf16x4_t *>(&const_cast<FragmentB &>(b))[1];

    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d0 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(B0, A0, _c);
    _d1 = hytlass::arch::detail::mmac_f32_16x16x16_bf16(B1, A1, _d0);

    d[0] = _d1.x;
    d[1] = _d1.y;
    d[2] = _d1.z;
    d[3] = _d1.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

//
// Matrix Multiply 161632 - int32_t accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - int32 = int8 * int8 + int32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 64>,
  64,
  int8_t,
  layout::RowMajor,
  int8_t,
  layout::ColumnMajor,
  int32_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 64>;

  using ElementA = int8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<int8_t, 16>;

  using ElementB = int8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<int8_t, 16>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    intx4_t _c;
    intx4_t _d0;
    intx4_t _d1;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    uint64_t A0, A1, B0, B1;
    A0 = reinterpret_cast<uint64_t *>(&const_cast<FragmentA &>(a))[0];
    B0 = reinterpret_cast<uint64_t *>(&const_cast<FragmentB &>(b))[0];
    A1 = reinterpret_cast<uint64_t *>(&const_cast<FragmentA &>(a))[1];
    B1 = reinterpret_cast<uint64_t *>(&const_cast<FragmentB &>(b))[1];

    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d0 = hytlass::arch::detail::mmac_i32_16x16x32_i8(B0, A0, _c);
    _d1 = hytlass::arch::detail::mmac_i32_16x16x32_i8(B1, A1, _d0);

    d[0] = _d1.x;
    d[1] = _d1.y;
    d[2] = _d1.z;
    d[3] = _d1.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

//
// Matrix Multiply 161664 - int32 accumulation
//
////////////////////////////////////////////////////////////////////////////////
/// Matrix multiply-add operation - int32 = uint8 * uint8 + int32
template <>
struct Mma<
  gemm::GemmShape<16, 16, 64>,
  64,
  uint8_t,
  layout::RowMajor,
  uint8_t,
  layout::ColumnMajor,
  int32_t,
  layout::RowMajor,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 64>;

  using ElementA = uint8_t;
  using LayoutA = layout::RowMajor;
  using FragmentA = Array<uint8_t, 16>;

  using ElementB = uint8_t;
  using LayoutB = layout::ColumnMajor;
  using FragmentB = Array<uint8_t, 16>;

  using ElementC = int32_t;
  using LayoutC = layout::RowMajor;
  using FragmentC = Array<ElementC, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Gfx928;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c) const {
#if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    intx4_t _c;
    intx4_t _d0;
    intx4_t _d1;

    _c.x = c[0];
    _c.y = c[1];
    _c.z = c[2];
    _c.w = c[3];

    uint64_t A0, A1, B0, B1;
    A0 = reinterpret_cast<uint64_t *>(&const_cast<FragmentA &>(a))[0];
    B0 = reinterpret_cast<uint64_t *>(&const_cast<FragmentB &>(b))[0];
    A1 = reinterpret_cast<uint64_t *>(&const_cast<FragmentA &>(a))[1];
    B1 = reinterpret_cast<uint64_t *>(&const_cast<FragmentB &>(b))[1];

    // Swap the order of A and B so that the resulting C threads are contiguous along the row dimension
    _d0 = hytlass::arch::detail::mmac_i32_16x16x32_u8(B0, A0, _c);
    _d1 = hytlass::arch::detail::mmac_i32_16x16x32_u8(B1, A1, _d0);

    d[0] = _d1.x;
    d[1] = _d1.y;
    d[2] = _d1.z;
    d[3] = _d1.w;
#else
    HYTLASS_UNUSED(a);
    HYTLASS_UNUSED(b);
    HYTLASS_UNUSED(c);
    HYTLASS_UNUSED(d);
    HYTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace hytlass

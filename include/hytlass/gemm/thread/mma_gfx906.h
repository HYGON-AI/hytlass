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
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/arch/mma.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/thread/mma.h"
#include "hytlass/reduction/thread/reduce.h"
#include "hytlass/functional.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles all packed matrix layouts
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_,
  /// Operator used to compute GEMM
  typename Operator_
>
struct MmaGeneric {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = Operator_;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Instruction
  using MmaOp = arch::Mma<
    gemm::GemmShape<1, 1, 1>,
    1,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    Operator>;

  static bool const kMultipleOf2 = ((Shape::kM % 2 == 0) && (Shape::kN % 2 == 0));

  static bool const kAllFp32 = platform::is_same<ElementA, float>::value &&
      platform::is_same<ElementB, float>::value &&
      platform::is_same<ElementC, float>::value;
  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementA const, LayoutA> a_ref(
      reinterpret_cast<ElementA const *>(&A), LayoutA::packed({Shape::kM, Shape::kK}));

    TensorRef<ElementB const, LayoutB> b_ref(
      reinterpret_cast<ElementB const *>(&B), LayoutB::packed({Shape::kK, Shape::kN}));

    TensorRef<ElementC, LayoutC> d_ref(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed(make_Coord(Shape::kM, Shape::kN)));

    MmaOp mma_op;

    // Copy accumulators
    D = C;

    // Compute matrix product
    HYTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {
      #if defined(__HIP_DEVICE_COMPILE__) && (__HIP_DEVICE_COMPILE__ >= 860)

      if (kMultipleOf2 && kAllFp32) {
        //2x2 zigzag - m and n loops to increment by 2. Inner loop to process 4 multiply-adds in a 2x2 tile.
        HYTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; n += 2) {
  
          HYTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; m += 2) {
  
            int m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;

            //top-left element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine, n);
              MatrixCoord mk(m_serpentine, k);
              MatrixCoord kn(k, n);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //bottom-left element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine+1, n);
              MatrixCoord mk(m_serpentine+1, k);
              MatrixCoord kn(k, n);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //bottom-right element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine+1, n+1);
              MatrixCoord mk(m_serpentine+1, k);
              MatrixCoord kn(k, n+1);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
  
            //top-right element in 2x2 tile
            {
              MatrixCoord mn(m_serpentine, n+1);
              MatrixCoord mk(m_serpentine, k);
              MatrixCoord kn(k, n+1);
              Array<ElementC, 1> d;
              Array<ElementA, 1> a;
              Array<ElementB, 1> b;
              d[0] = d_ref.at(mn);
              a[0] = a_ref.at(mk);
              b[0] = b_ref.at(kn);
              mma_op(d, a, b, d);
              d_ref.at(mn) = d[0];
            }
          }
        }
      } else 
      #endif
      {
        HYTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; ++n) {
  
          HYTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; ++m) {
  
            int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;
  
            MatrixCoord mn(m_serpentine, n);
            MatrixCoord mk(m_serpentine, k);
            MatrixCoord kn(k, n);
  
            Array<ElementC, 1> d;
            Array<ElementA, 1> a;
            Array<ElementB, 1> b;
  
            d[0] = d_ref.at(mn);
            a[0] = a_ref.at(mk);
            b[0] = b_ref.at(kn);
  
            mma_op(d, a, b, d);
  
            d_ref.at(mn) = d[0];
          }
        }
      }
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Matrix multiply-add operation - assumes operand B is not changing
struct MmaComplexF32_Column {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using ElementC = complex<float>;

  HYTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() =  a[0].real() * b[0].real() + c[0].real();
    d[0].imag() =  a[0].real() * b[0].imag() + d[0].imag();
    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() =  a[0].imag() * b[0].real() + c[0].imag();
  }
};

/// Matrix multiply-add operation - assumes operand A is not changing
struct MmaComplexF32_Corner {

  using Shape = gemm::GemmShape<1, 1, 1>;
  using ElementC = complex<float>;

  HYTLASS_HOST_DEVICE
  void operator()(
    Array<complex<float>, 1> &d,
    Array<complex<float>, 1> const &a,
    Array<complex<float>, 1> const &b,
    Array<complex<float>, 1> const &c
  ) {

    d[0].real() = -a[0].imag() * b[0].imag() + d[0].real();
    d[0].imag() =  a[0].real() * b[0].imag() + d[0].imag();
    d[0].real() =  a[0].real() * b[0].real() + c[0].real();
    d[0].imag() =  a[0].imag() * b[0].real() + c[0].imag();
  }
};

template <
  typename LayoutA,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB>
struct EnableMma_Crow_GFX906 {

  static bool const kIsConventionalLayout =
    (platform::is_same<LayoutA, layout::RowMajor>::value ||
      platform::is_same<LayoutA, layout::ColumnMajor>::value) &&
    (platform::is_same<LayoutB, layout::RowMajor>::value ||
      platform::is_same<LayoutB, layout::ColumnMajor>::value);

  static bool const value = kIsConventionalLayout;
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles all packed matrix layouts
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_
>
struct MmaGeneric<
  Shape_,
  complex<float>,
  LayoutA_,
  complex<float>,
  LayoutB_,
  complex<float>,
  LayoutC_,
  arch::OpMultiplyAdd> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = complex<float>;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = complex<float>;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = complex<float>;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Instruction
  using MmaOp = arch::Mma<
    gemm::GemmShape<1, 1, 1>,
    1,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    Operator>;

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementA const, LayoutA> a_ref(
      reinterpret_cast<ElementA const *>(&A), LayoutA::packed({Shape::kM, Shape::kK}));

    TensorRef<ElementB const, LayoutB> b_ref(
      reinterpret_cast<ElementB const *>(&B), LayoutB::packed({Shape::kK, Shape::kN}));

    TensorRef<ElementC, LayoutC> d_ref(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed(make_Coord(Shape::kM, Shape::kN)));

    detail::MmaComplexF32_Column mma_column;
    detail::MmaComplexF32_Corner mma_corner;

    // Copy accumulators
    D = C;

    // Compute matrix product
    HYTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {

      HYTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        HYTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {

          int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;

          MatrixCoord mn(m_serpentine, n);
          MatrixCoord mk(m_serpentine, k);
          MatrixCoord kn(k, n);

          Array<ElementC, 1> d;
          Array<ElementA, 1> a;
          Array<ElementB, 1> b;

          d[0] = d_ref.at(mn);
          a[0] = a_ref.at(mk);
          b[0] = b_ref.at(kn);

          if ((m == 0 && n) || m == Shape::kM - 1) {
            mma_corner(d, a, b, d);
          }
          else {
            mma_column(d, a, b, d);
          }

          d_ref.at(mn) = d[0];
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles conventional layouts for FFMA and DFMA GEMM
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_
>
struct Mma<
  Shape_,
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  LayoutC_,
  arch::OpMultiplyAdd,
  bool> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename MmaGeneric<
                                    Shape,
                                    ElementA,
                                    LayoutA,
                                    ElementB,
                                    LayoutB,
                                    ElementC,
                                    LayoutC,
                                    Operator>::MmaOp;
  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    MmaGeneric<
      Shape,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      Operator> mma;

    mma(D, A, B, C);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Computes matrix product when C is row-major
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  typename LayoutA_,
  typename LayoutB_
>
struct Mma<
  Shape_,
  half_t,
  LayoutA_,
  half_t,
  LayoutB_,
  half_t,
  layout::RowMajor,
  arch::OpMultiplyAdd,
  typename platform::enable_if<detail::EnableMma_Crow_GFX906<
    LayoutA_,
    LayoutB_
    >::value>::type>{

  using Shape = Shape_;
  using ElementA = half_t;
  using LayoutA = LayoutA_;
  using ElementB = half_t;
  using LayoutB = LayoutB_;
  using ElementC = half_t;
  using LayoutC = layout::RowMajor;
  using Operator = arch::OpMultiplyAdd;

  using TransposeMma = Mma<
    GemmShapeTranspose<Shape>,
    half_t,
    typename layout::LayoutTranspose<LayoutB>::type,
    half_t,
    typename layout::LayoutTranspose<LayoutA>::type,
    half_t,
    layout::ColumnMajor,
    arch::OpMultiplyAdd,
    bool>;

  using FragmentA = Array<ElementA, Shape::kMK>;
  using FragmentB = Array<ElementB, Shape::kKN>;
  using FragmentC = Array<ElementC, Shape::kMN>;

  using ArchMmaOperator = typename TransposeMma::ArchMmaOperator;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TransposeMma mma;
    mma(D, B, A, C);

  }
};

////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MIX_FP16XFP16_SM60
/// Computes matrix product when C is row-major
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  typename LayoutA_,
  typename LayoutB_
>
struct Mma<
  Shape_,
  half_t,
  LayoutA_,
  half_t,
  LayoutB_,
  float,
  layout::RowMajor,
  arch::OpMultiplyAdd,
  typename platform::enable_if<detail::EnableMma_Crow_GFX906<
    LayoutA_,
    LayoutB_
    >::value>::type>{

  using Shape = Shape_;
  using ElementA = half_t;
  using LayoutA = LayoutA_;
  using ElementB = half_t;
  using LayoutB = LayoutB_;
  using ElementC = float;
  using LayoutC = layout::RowMajor;
  using Operator = arch::OpMultiplyAdd;

  using TransposeMma = Mma<
    GemmShapeTranspose<Shape>,
    half_t,
    typename layout::LayoutTranspose<LayoutB>::type,
    half_t,
    typename layout::LayoutTranspose<LayoutA>::type,
    float,
    layout::ColumnMajor,
    arch::OpMultiplyAdd,
    bool>;

  using FragmentA = Array<ElementA, Shape::kMK>;
  using FragmentB = Array<ElementB, Shape::kKN>;
  using FragmentC = Array<ElementC, Shape::kMN>;

  using ArchMmaOperator = typename TransposeMma::ArchMmaOperator;

  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TransposeMma mma;
    mma(D, B, A, C);

  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles conventional layouts for IDP4A
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_
>
struct Mma<
  Shape_,
  int8_t,
  layout::ColumnMajor,
  int8_t,
  layout::RowMajor,
  int32_t,
  LayoutC_,
  arch::OpMultiplyAdd,
  int8_t> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = int8_t;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = layout::ColumnMajor;

  /// Data type of operand B
  using ElementB = int8_t;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = layout::RowMajor;

  /// Element type of operand C
  using ElementC = int32_t;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  /// Use 1x1x4 IDP4A sequence for bulk of computation
  using ArchMmaOperator = arch::Mma<
      gemm::GemmShape<1, 1, 4>,
      1,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      arch::OpMultiplyAdd>; 

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementC, LayoutC> d(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed({ Shape::kM, Shape::kN }));
    
    // Copy accumulators
    D = C;

    /// Underlying matrix multiply operator
    ArchMmaOperator mma;
    
    Array<int8_t, 4> const *ptr_A = reinterpret_cast<Array<int8_t, 4> const *>(&A);
    Array<int8_t, 4> const *ptr_B = reinterpret_cast<Array<int8_t, 4> const *>(&B);

    // Compute matrix product
    HYTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {

      HYTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        HYTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {
          MatrixCoord mn(m, n);

          Array<int32_t, 1> tmp = reinterpret_cast<Array<int32_t, 1> &>(d.at(mn));

          mma(
            tmp,
            ptr_A[m + k * Shape::kM],
            ptr_B[n + k * Shape::kN],
            tmp);

          d.at(mn) = reinterpret_cast<int32_t &>(tmp);
        }
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef MIX_FP16_DOT2
/// Gemplate that handles conventional layouts for dot2_f32_f16
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_
>
struct Mma<
  Shape_,
  half_t,
  layout::ColumnMajor,
  half_t,
  layout::RowMajor,
  float,
  LayoutC_,
  arch::OpMultiplyAdd,
  half_t> {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = half_t;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = layout::ColumnMajor;

  /// Data type of operand B
  using ElementB = half_t;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = layout::RowMajor;

  /// Element type of operand C
  using ElementC = float;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying mathematical operator
  using Operator = arch::OpMultiplyAdd;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  /// Use 1x1x2 dot2_f32_f16 sequence for bulk of computation
  using ArchMmaOperator = arch::Mma<
      gemm::GemmShape<1, 1, 2>,
      1,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      arch::OpMultiplyAdd>; 

  using MathOperator = typename ArchMmaOperator::Operator;

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
  HYTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    TensorRef<ElementC, LayoutC> d(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed({ Shape::kM, Shape::kN }));
    
    // Copy accumulators
    D = C;

    /// Underlying matrix multiply operator
    ArchMmaOperator mma;
    
    Array<half_t, 2> const *ptr_A = reinterpret_cast<Array<half_t, 2> const *>(&A);
    Array<half_t, 2> const *ptr_B = reinterpret_cast<Array<half_t, 2> const *>(&B);

    // Compute matrix product
    HYTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK / ArchMmaOperator::Shape::kK; ++k) {

      HYTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        HYTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {
          MatrixCoord mn(m, n);

          Array<float, 1> tmp = reinterpret_cast<Array<float, 1> &>(d.at(mn));

          mma(
            tmp,
            ptr_A[m + k * Shape::kM],
            ptr_B[n + k * Shape::kN],
            tmp);

          d.at(mn) = reinterpret_cast<float &>(tmp);
        }
      }
    }
  }
};
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

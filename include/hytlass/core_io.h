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
    \brief Helpers for printing hytlass/core objects
*/
#pragma once

#include <iostream>
#include <typeinfo>

#include "hytlass/array.h"
#include "hytlass/coord.h"
#include "hytlass/numeric_types.h"
#include "hytlass/matrix.h"
#include "hytlass/matrix_shape.h"
#include "hytlass/layout/pitch_linear.h"
#include "hytlass/tensor_view.h"
#include "hytlass/gemm/gemm_enumerated_types.h"
#include "hytlass/conv/convolution.h"
#include "hytlass/conv/conv2d_problem_size.h"

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Output operator for HIP built-in dim3 type
inline std::ostream &operator<<(std::ostream &out, dim3 d) {
  return out << d.x << ", " << d.y << ", " << d.z;
}

/// Output operator for HIP built-in error type
inline std::ostream &operator<<(std::ostream &out, hipError_t error) {
  return out << hipGetErrorString(error);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

///////////////////////////////////////////////////////////////////////////////////////////////////
//                    stream operators for hytlass namespace                                     //
///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, int Rank>
inline
std::ostream& operator<<(std::ostream& out, Array<Element, Rank> const& v) {
  for (int i = 0; i < Rank; ++i) {
    out << (i ? ", " : "") << v[i];
  }
  return out;
}

template <int Rank>
inline
std::ostream& operator<<(std::ostream& out, Coord<Rank> const& coord) {
  for (int i = 0; i < Rank; ++i) {
    out << (i ? ", " : "") << coord[i];
  }
  return out;
}

inline
std::istream & operator>>(std::istream &stream, half_t &x) {
  float tmp;
  stream >> tmp;
  x = static_cast<hytlass::half_t>(tmp);
  return stream;
}

inline
std::ostream & operator<<(std::ostream &out, half_t const &x) {
  return out << float(x);
}

inline
std::ostream & operator<<(std::ostream &out, bfloat16_t const &x) {
  return out << float(x);
}

inline
std::ostream & operator<<(std::ostream &out, tfloat32_t const &x) {
  return out << float(x);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to enable formatted printing of HYTLASS scalar types to an ostream
template <typename T>
struct ScalarIO {

  /// Value to print
  T value;

  /// Default ctor
  ScalarIO() { }

  /// Constructs from a value
  ScalarIO(T value): value(value) {}
};

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Default printing to ostream
template <typename T>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<T> const &scalar) {
  return out << scalar.value;
}

/// Printing to ostream of int8_t as integer rather than character
template <>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<int8_t> const &scalar) {
  return out << int(scalar.value);
}

/// Printing to ostream of uint8_t as integer rather than character
template <>
inline std::ostream &operator<<(std::ostream &out, ScalarIO<uint8_t> const &scalar) {
  return out << unsigned(scalar.value);
}


/// Default printing to ostream for MatrixShape
template <int Row, int Column>
inline
std::ostream & operator<<(std::ostream &out, MatrixShape<Row, Column> const &matrix_shape) {
  out << "hytlass::MatrixShape::(kRow, kColumn) {"
    << hytlass::MatrixShape<Row,Column>::kRow <<","
    << hytlass::MatrixShape<Row,Column>::kColumn <<"}";
  return out;
}


/// Prints matrix to ostream
template <typename Element, int Rows, int Columns>
std::ostream & operator<<(std::ostream &out, Matrix<Element, Rows, Columns> const &rhs) {

  for (int i = 0; i < Rows; ++i) {
    for (int j = 0; j < Columns; ++j) {
      ScalarIO<Element> element(rhs.at(i, j));
      out << (j ? ", " : "") << element;
    }
    out << "\\n";
  }

  return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//                         stream operators for hytlass::gemm namespace                          //
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace gemm {

/// Default printing to ostream for GemmShape
template <int M, int N, int K>
inline
std::ostream & operator<<(std::ostream &out, GemmShape<M,N,K> const &gemm_shape) {
  out << "hytlass::gemm::GemmShape::(kM, kN, kK) {"
    << hytlass::gemm::GemmShape<M,N,K>::kM <<","
    << hytlass::gemm::GemmShape<M,N,K>::kN <<","
    << hytlass::gemm::GemmShape<M,N,K>::kK << "}";
  return out;
}

/// Default printing to ostream for GemmCoord
inline
std::ostream & operator<<(std::ostream &out, GemmCoord const &gemm_coord) {
  out << "hytlass::gemm::GemmCoord {"
    << gemm_coord.m() <<","
    << gemm_coord.n() <<","
    << gemm_coord.k() << "}";
  return out;
}

} //namespace gemm
///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
//                       stream operators for hytlass namespace                          //
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Default printing to ostream for PitchLinearShape
template < int Contiguous, int Strided>
inline
std::ostream & operator<<(std::ostream &out, PitchLinearShape<Contiguous, Strided> const &pitch_linear_shape) {
  out << "hytlass::PitchLinearShape:(kContiguous, kStrided) {"
    << hytlass::layout::PitchLinearShape<Contiguous,Strided>::kContiguous <<","
    << hytlass::layout::PitchLinearShape<Contiguous,Strided>::kStrided <<"}";
  return out;
}

///////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////////////////
//                         stream operators for hytlass::conv namespace                          //
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace conv {
/// Default printing to ostream for Conv2dProblemSize
inline
std::ostream& operator<<(std::ostream& out, Conv2dProblemSize const& problem) {
  out << "NHWC: (" << problem.N << ", " << problem.H << ", " << problem.W << ", " << problem.C << ")" << std::endl
      << "KRSC: (" << problem.K << ", " << problem.R << ", " << problem.S << ", " << problem.C / problem.groups << ")" << std::endl
      << "NPQK: (" << problem.N << ", " << problem.P << ", " << problem.Q << ", " << problem.K << ")" << std::endl
      << "groups: (" << problem.groups << ")" << std::endl
      << "Pad_h, Pad_w: (" << problem.pad_h << ", " << problem.pad_w << ")" << std::endl
      << "Stride_h, Stride_w: (" << problem.stride_h << ", " << problem.stride_w << ")" << std::endl
      << "Dilation_h, Dilation_w: (" << problem.dilation_h << ", " << problem.dilation_w << ")" << std::endl
      << "split_k_slices: (" << problem.split_k_slices << ")" << std::endl
      << "mode: (" << ((problem.mode==conv::Mode::kConvolution) ? "conv" : "xcross") << ")";

  return out;
}

} // namespace conv
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass
///////////////////////////////////////////////////////////////////////////////////////////////////

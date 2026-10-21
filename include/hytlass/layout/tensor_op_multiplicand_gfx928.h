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
    \brief 
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/coord.h"
#include "hytlass/matrix_coord.h"
#include "hytlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace layout {

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory.
/// Native version of the tensor core kernel with support for
/// fp32/tf32/fp16/bf16/s8 data types, but without shared-memory
/// reordering or crosswise support.
template <int ElementSize>
struct NaiveTensorOpMultiplicand {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

  static int const kTileShapeStride = 1;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  NaiveTensorOpMultiplicand(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  HYTLASS_HOST_DEVICE
  NaiveTensorOpMultiplicand(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static NaiveTensorOpMultiplicand packed(TensorCoord const &extent) {
    return NaiveTensorOpMultiplicand(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return coord.contiguous() + coord.strided() * stride_[0];
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }

};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory.
template <int ElementSize>
struct NaiveTensorOpMultiplicandCongruous {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = NaiveTensorOpMultiplicand<ElementSize>;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  
  static int const kAccessSize = Base::kAccessSize;
 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  NaiveTensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  HYTLASS_HOST_DEVICE
  NaiveTensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static NaiveTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return NaiveTensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(coord);
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return coord;
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(extent);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// NaiveTensorOpMultiplicand
template <int ElementSize>
struct ColumnMajorNaiveTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = NaiveTensorOpMultiplicandCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  // static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorNaiveTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorNaiveTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static ColumnMajorNaiveTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return ColumnMajorNaiveTensorOpMultiplicandCongruous(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// NaiveTensorOpMultiplicand
template <int ElementSize>
struct RowMajorNaiveTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = NaiveTensorOpMultiplicandCongruous<ElementSize>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;


private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorNaiveTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorNaiveTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static RowMajorNaiveTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return RowMajorNaiveTensorOpMultiplicandCongruous(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};


////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits), defined in terms of pitch-linear
/// memory and Crosswise size (in elements), and primarily intended for
/// m-major or n-major operand layouts.
/// This one is the base class of all gfx928/gfx936 fp32/tf32/fp16/bf16/s8
/// tensor core kernels.
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCongruous128b {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

  // Smallest swizzle unit
  static int const kAccessPerUnit = 32 / kElementsPerAccess;

  static int const kTileShapeStride = (ElementSize == 32) ? 1 :
                                      (ElementSize == 16) ? (Crosswise == 32 ? 1 : 2) :
                                      (ElementSize == 8)  ? (Crosswise == 32 ? 1 : (Crosswise == 64) ? 2 : 4) : 
                                      1;

  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

  /// Partition only to handle s8 with kCrosswise = 64, in units.
  using PartitionShape = PitchLinearShape<2, 2>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous128b(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous128b(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static TensorOpMultiplicandCongruous128b packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCongruous128b(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    // TileShape::kStrided == 1; no bank conflicts, so no need for swizzing.
    if constexpr (TileShape::kStrided == 1) {
      return coord.contiguous() + coord.strided() * stride_[0];
    }

    // Split the tile first
    int tile_contiguous_idx = coord.contiguous() / (kElementsPerAccess * TileShape::kContiguous);
    int tile_strided_idx = coord.strided() / (TileShape::kStrided);

    int in_tile_strided_idx = coord.strided() % TileShape::kStrided;
    
    if constexpr (ElementSize == 8 && TileShape::kStrided == 2) {
      in_tile_strided_idx = (coord.strided() % (TileShape::kStrided * 2)) / 2;
    }

    // Subdivide the tile into units along the contiguous dimension.
    int in_tile_contiguous_idx = coord.contiguous() % (kElementsPerAccess * TileShape::kContiguous);
    int unit_contiguous_idx = in_tile_contiguous_idx / (kElementsPerAccess * kAccessPerUnit);

    int permuted_unit_contiguous_idx = unit_contiguous_idx ^ in_tile_strided_idx;

    int element_contiguous = 
      tile_contiguous_idx * (kElementsPerAccess * TileShape::kContiguous) + 
      permuted_unit_contiguous_idx * (kElementsPerAccess * kAccessPerUnit) + 
      coord.contiguous() % (kElementsPerAccess * kAccessPerUnit);
    
    return element_contiguous + coord.strided() * stride_[0];
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicandCongruous128b
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCongruous128b {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous128b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  // static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous128b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous128b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCongruous128b packed(TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCongruous128b(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCongruous128b
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCongruous128b {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous128b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;


private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous128b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous128b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCongruous128b packed(TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCongruous128b(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

//////////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits), defined in terms of pitch-linear
/// memory and Crosswise size (in elements), and primarily intended for
/// k-major layouts.
/// This one is the base class of tensor core kernels on gfx928/gfx936 arch that
/// splice two instructions along the k-dimension, support fp32/tf32/fp16/bf16/s8.
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise128b {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

  /// Number of kblocks to store PartitionShape::kContiguous Elements
  static int const kFactor =
      kTileShapeContiguous * kElementsPerAccess / kCrosswise;

  static_assert(
      (kFactor > 0),
      "kCrosswise should be no large than one shared memory cache line.");

  
  static int const kTileShapeStride = (kFactor == 1) ? 8 : 4; 
  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  /// For kFactor = 1, TileShape = <8, 8> 
  /// For kFactor > 1, TileShape = <8, 4>
  using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<4, 4>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise128b(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise128b(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static TensorOpMultiplicandCrosswise128b packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCrosswise128b(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    //
    // First, compute c and s of vector within source (in units of vector
    // accesses)
    //

    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
    int vec_strided_idx = coord.strided() / kFactor;

    // Compute the fundamental tile being accessed
    int tile_contiguous_idx =
        vec_contiguous_idx / (TileShape::kContiguous / kFactor);

    int tile_contiguous_residual =
        vec_contiguous_idx % (TileShape::kContiguous / kFactor) +
        ((coord.strided() % kFactor) * (TileShape::kContiguous / kFactor));
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

    // Compute the 'partition' within the fundamental tile
    int partition_contiguous_idx =
        tile_contiguous_residual / PartitionShape::kContiguous;
    int partition_strided_idx =
        tile_strided_residual / PartitionShape::kStrided;

    int partition_contiguous_residual =
        tile_contiguous_residual % PartitionShape::kContiguous;
    int partition_strided_residual =
        tile_strided_residual % PartitionShape::kStrided;

    //
    // Then swizzle
    //

    int permuted_vec_contiguous_within_partition = 
        partition_contiguous_residual ^ (partition_strided_residual % 4);

    int permuted_partition_contiguous_within_tile =   
        partition_contiguous_idx ^ (partition_strided_idx % 2);

    //
    // Compute final element location
    //

    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                              permuted_partition_contiguous_within_tile *
                                  PartitionShape::kContiguous +
                              permuted_vec_contiguous_within_partition) *
                                 kElementsPerAccess +
                             (coord.contiguous() % kElementsPerAccess);

    int element_strided = vec_strided_idx;

    return element_contiguous + element_strided * stride_[0] * kFactor;
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise128b
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise128b {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise128b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  // static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise128b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise128b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCrosswise128b packed(TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCrosswise128b(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    // B (k, n)
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise128b
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise128b {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise128b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;


private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise128b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise128b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCrosswise128b packed(TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCrosswise128b(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits), defined in terms of pitch-linear
/// memory and Crosswise size (in elements), and primarily intended for
/// k-major layouts.
/// This one is the base class of all gfx928/gfx936 fp32/tf32/fp16/bf16/s8
/// tensor core kernels.
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise64b {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 64b accesses
  static int const kAccessSize = 64;

  /// Replace a single 128-byte memory access with two 64-byte accesses.
  static int const kAccessPerDword = 2;

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 16 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);

  /// Number of kblocks to store PartitionShape::kContiguous Elements
  static int const kFactor =
      kTileShapeContiguous * kElementsPerAccess / kCrosswise * 2;

  // For half -> crosswise = 16/32/64
  //                           8/ 4/2
  static_assert(
      (kFactor > 1),
      "kCrosswise should be no large than one shared memory cache line.");

  static int const kTileShapeStride = 16 / kFactor;

  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  /// For kFactor = 1, TileShape = <8, 8> 
  /// For kFactor > 1, TileShape = <8, 4>
  using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<8, 8>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise64b(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  HYTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise64b(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static TensorOpMultiplicandCrosswise64b packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCrosswise64b(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    //
    // First, compute c and s of vector within source (in units of vector
    // accesses)
    //

    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess;
    int vec_strided_idx = coord.strided() / kFactor;
    // A 128-byte memory access is split into two 64-byte accesses; 
    // recording the indices for both.
    int dword_residual = vec_contiguous_idx % kAccessPerDword;

    // Compute the fundamental tile being accessed
    // TileShape::kContiguous / kFactor -> 16 / 2 -> 8
    int tile_contiguous_idx =
        vec_contiguous_idx / kAccessPerDword / (TileShape::kContiguous / kFactor);

    // Offset for 64-byte vectorized memory accesses along the contiguous 
    // dimension within the tile.
    int tile_contiguous_residual =
        (vec_contiguous_idx / kAccessPerDword) % (TileShape::kContiguous / kFactor) +
        ((coord.strided() % kFactor) * (TileShape::kContiguous / kFactor));

    // Offset along the strided dimension within the tile.
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;

    // Compute the 'partition' within the fundamental tile
    int partition_contiguous_idx =
        tile_contiguous_residual / PartitionShape::kContiguous;
    int partition_strided_idx =
        tile_strided_residual / PartitionShape::kStrided;

    int partition_contiguous_residual =
        tile_contiguous_residual % PartitionShape::kContiguous;
    int partition_strided_residual =
        tile_strided_residual % PartitionShape::kStrided;

    //
    // Then swizzle
    //

    int permuted_vec_contiguous_within_partition =
        partition_contiguous_residual ^ (partition_strided_residual % 8);

    int permuted_partition_contiguous_within_tile =
        partition_contiguous_idx ^ (partition_strided_idx % 2);

    //
    // Compute final element location
    //
    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous * kAccessPerDword + 
                              permuted_partition_contiguous_within_tile * 
                              PartitionShape::kContiguous + 
                              permuted_vec_contiguous_within_partition) * kElementsPerAccess + 
                              dword_residual * kElementsPerAccess * TileShape::kContiguous + 
                              (coord.contiguous() % kElementsPerAccess);

    int element_strided = vec_strided_idx;

    return element_contiguous + element_strided * stride_[0] * kFactor;
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicandCongruous128b
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise64b {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise64b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  // static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise64b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise64b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCrosswise64b packed(TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCrosswise64b(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    // B (k, n)
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCongruous128b
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise64b {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise64b<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;


private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise64b(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  HYTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise64b(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  HYTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCrosswise64b packed(TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCrosswise64b(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  HYTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  HYTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  HYTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  HYTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};



} // namespace layout
} // namespace hytlass

////////////////////////////////////////////////////////////////////////////////

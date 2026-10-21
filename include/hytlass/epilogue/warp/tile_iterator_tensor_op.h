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

#include "hytlass/array.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/layout/pitch_linear.h"

#include "hytlass/epilogue/warp/tensor_op_policy.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape,     ///< shape of warp-level GEMM (concept: MatrixShape)
  typename OperatorShape, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element,       ///< data type of element to be written
  typename Layout         ///< target shared memory layout
>
class TileIteratorTensorOp;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_        ///< data type of element to be written
>
class TileIteratorTensorOp<WarpShape_, OperatorShape_, Element_, layout::RowMajor> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;

  using TensorLayout = Layout;
  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
    Policy::kRowsPerIteration,
    WarpShape::kN
  >;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kColumn * Policy::kAccumulatorInRow>;

  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Number of times this iterator can be incremented
  using TileIterations = typename Policy::TileIterations;

  /// Padding quantity
  using Padding = MatrixShape<0, 0>;

private:

  /// Storage type for accessing memory
  // using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;
  using AccessType = Element;  

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  Layout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0] / Policy::kElementsPerAccess) {

    int row_id = (lane_id >> 4);
    int col_id = (lane_id & 15);

    thread_offset_ = {
      row_id, col_id
    };

    pointer_ += layout_({thread_offset_.row(), thread_offset_.column()});
  }

  /// Adds a pointer offset
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow, 
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += layout_({
      coord_offset.row(),
      coord_offset.column() / Policy::kElementsPerAccess
    });

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  HYTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Policy::kAccumulatorInRow; ++i){
      HYTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Policy::OperatorCount::kColumn; ++n){    
        pointer_[n * 16 + i * Policy::kAccumulatorInRow * layout_.stride(0)] = 
          frag_ptr[n + i * Policy::OperatorCount::kColumn];
      }
    }    
  }

  /// Store
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kColumn; ++n) {
      HYTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Policy::kAccumulatorInRow; ++i){
        frag_ptr[n + i * Policy::OperatorCount::kColumn] =
          pointer_[n * 16 + i * Policy::kAccumulatorInRow * layout_.stride(0)];
      }
    }     
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator++() {
    return add_tile_offset({1, 0});
  }
  
  /// Set smem base address
  HYTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory
template <
  typename WarpShape_,     ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_, ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,       ///< data type of element to be written
  int InterleavedK         ///< number of interleaved k
>
class TileIteratorTensorOp<WarpShape_, OperatorShape_, Element_, 
                            layout::ColumnMajorInterleaved<InterleavedK> > {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorInterleaved<InterleavedK>;
  using TensorLayout = Layout;                ///< shared memory tensor ref layout

  using TensorRef = TensorRef<Element, TensorLayout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  /// Shape of the tile in memory
  using Shape = MatrixShape<
//    Policy::kRowsPerIteration,
    WarpShape::kM,
    InterleavedK
  >;

  /// This is the fragment size produced by one tile
  using Fragment = Array<
    Element, 
    Policy::OperatorCount::kRow * Policy::kIterationsPerInstruction 
        * Policy::kElementsPerIteration>;

  /// This is the fragment size produced by one iteration
//  using Fragment = Array<
//    Element, Policy::kElementsPerIteration >;

  /// This is the complete warp-level accumulator tile.
  //using AccumulatorTile = typename Operator::FragmentC;

  /// Number of times this iterator can be incremented
  using TileIterations = typename Policy::TileIterations;

  // Internal constants
  struct Detail {
    static int const kLanesInQuad = 4;
  };

  /// Padding quantity
  using Padding = MatrixShape<
    0,
    Detail::kLanesInQuad * Policy::kElementsPerIteration>;

private:

  /// Storage type for accessing memory
  using AccessType = AlignedArray<Element, Policy::kElementsPerAccess>;

  //
  // Data members
  //

  /// Internal pointer to memory
  AccessType *pointer_;

  /// Internal layout object
  TensorLayout layout_;

  /// Thread offset
  MatrixCoord thread_offset_;

public:

  /// Default constructor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp(): pointer_(nullptr) { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp(
    TensorRef const &ref,
    unsigned lane_id
  ):
    pointer_(reinterpret_cast<AccessType *>(ref.data())),
    layout_(ref.stride()[0]) {

    int quad_id = (lane_id / Detail::kLanesInQuad); 
    int lane_in_quad = (lane_id % Detail::kLanesInQuad);

    thread_offset_ = {
      quad_id, lane_in_quad * Policy::kElementsPerIteration
    };

    pointer_ += (layout_({thread_offset_.row(), thread_offset_.column()}) / Policy::kElementsPerAccess);
  }

  /// Adds a pointer offset
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_pointer_offset(Index pointer_offset) {
    pointer_ += pointer_offset / Policy::kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & add_tile_offset(TensorCoord const &tile_offset) {

    MatrixCoord coord_offset(
      tile_offset.row() * Shape::kRow, 
      tile_offset.column() * Shape::kColumn
    );

    thread_offset_ += coord_offset;

    pointer_ += (layout_({
      coord_offset.row(),
      coord_offset.column()
    }) / Policy::kElementsPerAccess);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  /// Store
  HYTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
      
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kRow * Policy::kIterationsPerInstruction; n++ ) {

      AccessType *ptr = pointer_ + layout_({n * Policy::kRowsPerIteration, 0}) / Policy::kElementsPerAccess;

      HYTLASS_PRAGMA_UNROLL
      for (int a = 0; a < Policy::kAccessPerIteration; ++a) {
        ptr[a + pointer_offset / Policy::kElementsPerAccess] = frag_ptr[n * Policy::kAccessPerIteration + a];
      }
    }
  }

  /// Store
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int n = 0; n < Policy::OperatorCount::kRow * Policy::kIterationsPerInstruction; n++ ) {

      AccessType *ptr = pointer_ + layout_({n * Policy::kRowsPerIteration, 0}) / Policy::kElementsPerAccess;

      HYTLASS_PRAGMA_UNROLL
      for (int a = 0; a < Policy::kAccessPerIteration; ++a) {
        frag_ptr[n * Policy::kAccessPerIteration + a] = ptr[a + pointer_offset / Policy::kElementsPerAccess];
      }
    }
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  HYTLASS_HOST_DEVICE
  TileIteratorTensorOp & operator++() {
    return add_tile_offset({0, 1});
  }

  /// Set smem base address
  HYTLASS_HOST_DEVICE
  void set_smem_base_address(Index address) {
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace warp
} // namespace epilogue
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

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
#include "hytlass/layout/matrix.h"
#include "hytlass/layout/pitch_linear.h"
#include "hytlass/epilogue/warp/tensor_op_policy.h"
#include "hytlass/epilogue/warp/tile_iterator_tensor_op.h"


/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace epilogue {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template for reading and writing tiles of accumulators to shared memory. This is optimized
/// for mixed-precision epilogues in which the accumulators are 32b in width, but the output
/// data type is smaller. 
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,              ///< data type of accumulator element
  int ElementSizeBits,            ///< Size of accumulator element in bits
  int OutputSizeBits,             ///< Size of output element in bits
  int OutputElementCount,         ///< number of elements in output vector
  int ContiguousLanes,            ///< Number of consecutive lanes writing to contiguous memory
  bool EightBitsOutputOrLess = (OutputSizeBits <= 8)
>
class TileIteratorTensorOpMixed;

/////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: Temporarily use the regular shared-load instead of the mixed implementation

/// Template for reading and writing tiles of accumulators to shared memory. 
template <
  typename WarpShape_,            ///< shape of warp-level GEMM (concept: GemmShape)
  typename OperatorShape_,        ///< matrix multiply operation shape (concept: gemm::GemmShape)
  typename Element_,              ///< data type of accumulator element
  int OutputSizeBits,             ///< Size of output element in bits
  int OutputElementCount         ///< number of elements in output vector
>
class TileIteratorTensorOpMixed<WarpShape_, OperatorShape_, Element_, 32,
      OutputSizeBits, OutputElementCount, 16, false> {
public:

  using WarpShape = WarpShape_;
  using OperatorShape = OperatorShape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kOutputElementCount = OutputElementCount;

  using TensorRef = TensorRef<Element, Layout>;         ///< Tensor Reference object
  using TensorCoord = MatrixCoord;                      ///< Logical coordinate in referenced tensor
  using Index = typename TensorRef::Index;
  using LongIndex = typename TensorRef::LongIndex;

  using Policy = TensorOpPolicy<WarpShape, OperatorShape, Layout>;

  using Base = TileIteratorTensorOp<WarpShape, OperatorShape, Element, Layout>;

  /// Shape of the tile in memory
  using Shape = typename Base::Shape;

  /// This is the fragment size produced by one access of the iterator.
  using Fragment = typename Base::Fragment;


  /// Number of times this iterator can be incremented
  static int const kIterations = Policy::kIterations;

  /// Padding quantity
  using Padding = typename Base::Padding;

private:

  /// Storage type for accessing memory
  using AccessType = Element;

  //
  // Data members
  //
  Base iterator_;

public:

  /// Default constructor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed() : iterator_() {}

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed(
    TensorRef const &ref,
    unsigned lane_id
  ) : iterator_(ref, lane_id){ 

  }

  /// Adds a pointer offset
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_pointer_offset(Index pointer_offset) {

    iterator_.add_pointer_offset(pointer_offset);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & add_tile_offset(TensorCoord const &tile_offset) {
    
    iterator_.add_tile_offset(tile_offset);

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_HOST_DEVICE
  TileIteratorTensorOpMixed & operator+=(TensorCoord const &tile_offset) {
    return add_tile_offset(tile_offset);
  }

  /// Store
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {

    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Load
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
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

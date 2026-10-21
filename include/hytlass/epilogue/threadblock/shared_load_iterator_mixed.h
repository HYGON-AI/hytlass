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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops optimized for mixed-precision.

  This assumes the shared memory tile is in a permuted layout which avoids bank conflicts on loading.

  When the fragment is loaded into registers, it matches the row-major thread map assumed by
  the predicated tile iterator writing to global memory.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/array.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/matrix_shape.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/epilogue/threadblock/shared_load_iterator.h"
#include "hytlass/epilogue/threadblock/output_tile_thread_map.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Accumulator data type
  int ElementSizeBits_,      ///< Size of accumulator in bits
  int OutputSizeBits_,       ///< Size of output element in bits
  int ElementsPerAccess,     ///< Vector length of output vector
  int ContiguousLanes,       ///< Number of lanes in the warp writing to contiguous elements
                             ///  in the global memory tensor
  bool EightBitsOutputOrLess = (OutputSizeBits_ <= 8)
>
class SharedLoadIteratorMixed;

/////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: Temporarily use the regular shared-load instead of the mixed implementation

/// Tile iterator used to load output tile from shared memory in epilogue.
///
/// Satisfies: ReadableTileIterator
///
template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_          ///< Accumulator data type
>
class SharedLoadIteratorMixed<ThreadMap_, Element_, 32, 16, 8, 16, false> {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  static int const kAlignment = ThreadMap::kElementsPerAccess * sizeof_bits<Element_>::value / 8;

  static int const kThreads = ThreadMap::kThreads;

  /// Fragment object
  using Fragment = Array<
    Element, 
    ThreadMap::Iterations::kColumn * 
    ThreadMap::Iterations::kRow * 
    ThreadMap::Iterations::kGroup * 
    ThreadMap::Iterations::kCluster * 
    ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<
    Element, 
    ThreadMap::kElementsPerAccess, 
    kAlignment>;

  /// Vector type used for SMEM loads
  using LoadType = AlignedArray<
    Element,
    const_min(128 / sizeof_bits<Element>::value, ThreadMap::kElementsPerAccess),
    const_min(16, kAlignment)
  >;

  static int const kLoadsPerAccess = AccessType::kElements / LoadType::kElements;

  using Base = SharedLoadIterator<ThreadMap, Element>;

private:

  //
  // Data members
  //

  Base iterator_;

public:

  //
  // Methods
  //

  /// Constructor
  HYTLASS_DEVICE
  SharedLoadIteratorMixed(
    TensorRef ref,
    int thread_idx
  ):
    iterator_(ref, thread_idx) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &offset) {
    iterator_.add_tile_offset(offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
    
  }

  /// Set base smem address
  HYTLASS_DEVICE
  void set_smem_base_address(Index address) {}

  /// Loads a fragment
  HYTLASS_DEVICE
  void load(Fragment &frag) const {

    load_with_pointer_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

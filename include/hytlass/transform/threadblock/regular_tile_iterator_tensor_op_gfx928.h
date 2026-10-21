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
    \brief Templates implementing storing of tiles from pitch-linear rank=2 tensors. 
*/

#pragma once

#include "hytlass/transform/threadblock/regular_tile_iterator.h"
#include "hytlass/transform/threadblock/regular_tile_access_iterator_tensor_op_gfx928.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileIterator<
    Shape_, Element_,
    layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = hytlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");
  };

private:

  /// Element type per access
  using AccessType = uint32_t __attribute__((ext_vector_type(4)));

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

private:

  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    address_iterator_.add_tile_offset({0, 1});
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

        frag_ptr[access_idx] = *access_ptr;
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  HYTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {  
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
        AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

        *access_ptr = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileIterator<
    Shape_, Element_,
    layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment>
class RegularTileIterator<
    Shape_, Element_,
    layout::RowMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }
  
  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  // using Layout =
  //     layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;

  using Layout =
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>;


  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = hytlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");
  };

private:

  /// Element type per access
  // using AccessType = Array<Element, Layout::kElementsPerAccess>;
  using AccessType = uint32_t __attribute__((ext_vector_type(4)));

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap, Crosswise>;

private:

  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    address_iterator_.add_tile_offset({0, 1});
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

        frag_ptr[access_idx] = *access_ptr;
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  HYTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {  
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
        AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);
        *access_ptr = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};


////////////////////////////////////////////////////////////////////////////////
/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }
  
  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {

    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {

    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for crosswise arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = hytlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in length.
    static int const kAccessSizeInBits = 128;

    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
      "This iterator requires a policy whose access size is 128bs");
  };

private:

  /// Element type per access
  // using AccessType = Array<Element, Layout::kElementsPerAccess>;
  using AccessType = int __attribute__((ext_vector_type(4)));
public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

private:

  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    // move to next tile along k-dim (the contiguous-dim)
    address_iterator_.add_tile_offset({1, 0});
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();
    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

        frag_ptr[access_idx] = *access_ptr;
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  using v4i = int __attribute__((ext_vector_type(4)));

  HYTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;

        char *byte_ptr = reinterpret_cast<char *>(address_iterator_.get()) + byte_offset;
        AccessType *access_ptr = reinterpret_cast<AccessType *>(byte_ptr);

        *access_ptr = frag_ptr[access_idx];
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};


/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }
  
  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


/// Tile iterator specialized for crosswise arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = hytlass::TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 64 bits in length.
    static int const kAccessSizeInBits = 64;
    static int const kAccessPerVector = 2;
    static_assert(
      sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits * kAccessPerVector,
      "This iterator requires a policy whose access size is 128bs");
  };

private:

  static const int kElementsPerAccess = Detail::kAccessSizeInBits / sizeof_bits<Element>::value;
  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, ThreadMap::Iterations::kCount * Layout::kElementsPerAccess * Detail::kAccessPerVector>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = RegularTileAccessIterator<Shape, Element, Layout,
                                                       kAdvanceRank, ThreadMap>;

private:

  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(TensorRef ref,  ///< Pointer to start of tensor
                      int thread_id   ///< ID of each participating thread
                      )
      : address_iterator_(ref, thread_id) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    // Along the K dimension, i.e., the contiguous direction here
    address_iterator_.add_tile_offset({1, 0});
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    this->operator++();
    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    address_iterator_.add_tile_offset(coord);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    load_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Index byte_offset) {
    address_iterator_.set_iteration_index(0);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;
        AccessType *frag_ptr = reinterpret_cast<AccessType *>(&(frag[access_idx]));
        
        char const *byte_ptr = reinterpret_cast<char const *>(address_iterator_.get()) + byte_offset;
        AccessType const *access_ptr = reinterpret_cast<AccessType const *>(byte_ptr);

        *(frag_ptr + 0) = *access_ptr;
        *(frag_ptr + 1) = *(access_ptr + 16);
        ++address_iterator_;
      }
    }
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    store_with_byte_offset(frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  HYTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Index byte_offset) {
    // TODO: byte_offset-based addressing is not yet supported
    address_iterator_.set_iteration_index(0);
    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        int access_idx = c + s * ThreadMap::Iterations::kContiguous;
        uint128_t const* tmp_ptr = reinterpret_cast<uint128_t const*>((&frag)); 
        AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(tmp_ptr + access_idx);

        AccessType *access_ptr = reinterpret_cast<AccessType *>(address_iterator_.get());
        uint32_t ds_write_base = (uint32_t)(uintptr_t)access_ptr;
        // The fragment is a 128-bit type, implemented by two 64-bit memory accesses
        // whose addresses are always 128 bytes apart. Since the compiler may generates
        // a ds_write2_b64 instruction for this pattern, inline assembly is used here.
        // Synchronization must be handled externally via syncthreads().
        asm volatile(
          "ds_write_b64 %0 %1 offset:0\n\t"
          "ds_write_b64 %0 %2 offset:128\n\t"
          // "s_waitcnt lgkmcnt(0)\n\t"
          :
          : "v"(ds_write_base), "v"(*(frag_ptr)), "v"(*(frag_ptr+1))
        );
        ++address_iterator_;
      }
    }
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for column-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileIterator<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for row-major iterator may along advance along the "
    "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

 public:

  /// Fragment object to be loaded or stored
  using Fragment = Array<Element, UnderlyingIterator::Fragment::kElements>;

private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

public:

  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileIterator(
    TensorRef ref,                              ///< Pointer to start of tensor
    int thread_id                               ///< ID of each participating thread
  ): iterator_({ref.data(), ref.stride()}, thread_id) {

  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }
  
  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileIterator operator++(int) {
    RegularTileIterator prev(*this);
    ++iterator_;

    return prev;
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag, 
    Index pointer_offset) {
    
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};


} // namespace threadblock
} // namespace transform
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

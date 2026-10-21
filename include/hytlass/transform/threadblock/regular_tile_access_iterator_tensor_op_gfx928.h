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
    \brief Templates implementing computing the addresses of storing of tiles
   from pitch-linear rank=2 tensors.
*/

#pragma once

#include "hytlass/array.h"
#include "hytlass/hytlass.h"
#include "hytlass/layout/pitch_linear.h"
#include "hytlass/layout/tensor_op_multiplicand_gfx928.h"
#include "hytlass/matrix_coord.h"
#include "hytlass/matrix_shape.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/transform/threadblock/regular_tile_access_iterator.h"

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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
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
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
                 "This iterator requires a policy whose access size is 128bs");

  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_; 

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : stride_(ref.stride(0) / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    pointer_ = reinterpret_cast<AccessType *>(
        ref.data() + ref.offset(thread_offset_base));    
 
    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Returns a pointer
  HYTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = pointer_;
    int access_offset = iteration_strided_ * ThreadMap::Delta::kStrided * stride_ +
                        iteration_contiguous_ * ThreadMap::Delta::kContiguous /
                            ThreadMap::kElementsPerAccess;

    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);    
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * Shape::kContiguous +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess);
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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
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
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  HYTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::RowMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
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
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  HYTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value * ThreadMap::kElementsPerAccess == kAccessSizeInBits,
                 "This iterator requires a policy whose access size is 128bs");
    
    // For fp32, this is always 1.
    // For fp16, it can be 1 or 2 depending on the BlockMN size.
    // fp32 1/4
    // 64 * 16 -> 1024 -> 4
    // 32 * 16 -> 512 -> 2
    static int constexpr kPointerCount = std::min(
      ThreadMap::Iterations::kStrided, Layout::TileShape::kStrided
    );

    static_assert(kPointerCount >= 1 && (kPointerCount & (kPointerCount - 1)) == 0,
                  "kPointerCount must be positive power of 2");
  };

  /// Element type per access
  // using AccessType = Array<Element, Layout::kElementsPerAccess>;
  using AccessType = uint32_t __attribute__((ext_vector_type(4)));

 private:
  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount]; 

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : stride_(ref.stride(0) / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};
      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(
        ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  HYTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = nullptr;
    // Explicitly access all elements to prevent the compiler from spilling 
    // pointer_ to global memory (GMEM).
    if constexpr (Detail::kPointerCount == 2) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
      }
    } else if (Detail::kPointerCount == 4) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
        case 2 : access_ptr = pointer_[2]; break;
        case 3 : access_ptr = pointer_[3]; break;
      }
    } else {
      access_ptr = pointer_[0];
    }

    int stride_idx = (iteration_strided_ & ~(Detail::kPointerCount - 1));

    int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ +
                        iteration_contiguous_ * ThreadMap::Delta::kContiguous /
                            ThreadMap::kElementsPerAccess;

    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);

    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);


  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter iteration in strided
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;
    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * Shape::kContiguous +
                       coord.strided() * Shape::kStrided * stride_ * Layout::kElementsPerAccess);
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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;
  static int const kCrosswise = Crosswise;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  static_assert(!(ThreadMap::Delta::kContiguous % kCrosswise),
                "kCrosswise is the smallest unit in the contiguous dimension "
                "for shared memory swizzling.");

  static_assert(Shape::kContiguous <= WARP_SIZE_GPU * (128 / sizeof_bits<Element>::value), 
                "This iterator restricts the maximum contiguous dim to 1024 bytes\n");

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");


    /// Records the lane span along the m/n dimension for one warp iteration.
    /// Due to the swizzled layout, lane-to-lane offsets (e.g., lane0 → lane0) may vary during iteration,
    /// requiring multiple pointer instances to be initialized.
    static int const kLanesInStrided = 
      WARP_SIZE_GPU / ((Shape::kContiguous / (128 / sizeof_bits<Element>::value)) / ThreadMap::Iterations::kContiguous);

    /// Number of pointers
    ///
    /// Swizzling along the m/n dimension is performed in units of up to 8 elements.
    static int const kPointerCount = std::min(ThreadMap::Iterations::kStrided,
                                              std::max(8 / kLanesInStrided, 1));

    static_assert(kPointerCount >= 1 && (kPointerCount & (kPointerCount - 1)) == 0,
                 "kPointerCount must be positive power of 2");

  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Sections that a stage has
  int sections_per_stage_;

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : sections_(ref.stride(0) / kCrosswise), 
        sections_per_stage_(Shape::kContiguous / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};
      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    ref.offset(thread_offset_in_threadblock_tile) /
                        Layout::kElementsPerAccess;
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Returns a pointer
  HYTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = nullptr;
    // Explicitly access all elements to prevent the compiler from spilling 
    // pointer_ to global memory (GMEM).
    if constexpr (Detail::kPointerCount == 2) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
      }
    } else if (Detail::kPointerCount == 4) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
        case 2 : access_ptr = pointer_[2]; break;
        case 3 : access_ptr = pointer_[3]; break;
      }
    } else if (Detail::kPointerCount == 8) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
        case 2 : access_ptr = pointer_[2]; break;
        case 3 : access_ptr = pointer_[3]; break;
        case 4 : access_ptr = pointer_[4]; break;
        case 5 : access_ptr = pointer_[5]; break;
        case 6 : access_ptr = pointer_[6]; break;
        case 7 : access_ptr = pointer_[7]; break;
      }
    } else {
      access_ptr = pointer_[0];
    }

    int stride_idx = (iteration_strided_ & ~(Detail::kPointerCount - 1));

    int access_offset =
        stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
        // kCrosswise elements in the contiguous dimension would span to a
        // shared memory cache line.
        iteration_contiguous_ * (ThreadMap::Delta::kContiguous / kCrosswise) *
            Layout::TileShape::kContiguous;
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next section.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * sections_per_stage_ * stride_ *
                           ThreadMap::kElementsPerAccess / sections_ +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess / Layout::kFactor);
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
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;
  static int const kCrosswise = Crosswise;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  static_assert(!(ThreadMap::Delta::kContiguous % kCrosswise),
                "kCrosswise is the smallest unit in the contiguous dimension "
                "for shared memory swizzling.");
  
  static_assert(Shape::kContiguous <= WARP_SIZE_GPU * (128 / sizeof_bits<Element>::value), 
                "This iterator restricts the maximum contiguous dim to 1024 bytes\n");

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// Number of pointers
    ///
    /// Records the lane span along the m/n dimension for one warp iteration.
    /// Due to the swizzled layout, lane-to-lane offsets (e.g., lane0 → lane0) may vary during iteration,
    /// requiring multiple pointer instances to be initialized.
    static int constexpr kLanesInStrided = WARP_SIZE_GPU / 
      ((Shape::kContiguous / (128 / sizeof_bits<Element>::value)) / ThreadMap::Iterations::kContiguous);

    /// Swizzling along the m/n dimension is performed in units of up to 16 elements.
    static int constexpr kPointerCount = std::min(ThreadMap::Iterations::kStrided,
                                                  std::max(16 / kLanesInStrided, 1));

    static_assert(kPointerCount >= 1 && (kPointerCount & (kPointerCount - 1)) == 0,
                  "kPointerCount must be positive power of 2");

  };

  /// Element type per access
  // using AccessType = Array<Element, Layout::kElementsPerAccess>;
  using AccessType = Element;

 private:
  //
  // Data members
  //

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Sections that a stage has
  int sections_per_stage_;

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : sections_(ref.stride(0) / kCrosswise),
        sections_per_stage_(Shape::kContiguous / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};
      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    ref.offset(thread_offset_in_threadblock_tile);
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  HYTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Returns a pointer
  HYTLASS_HOST_DEVICE
  Element *get() const {
    AccessType* access_ptr = nullptr;
    // Explicitly access all elements to prevent the compiler from spilling 
    // pointer_ to global memory (GMEM).
    if constexpr (Detail::kPointerCount == 2) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
      }
    } else if (Detail::kPointerCount == 4) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
        case 2 : access_ptr = pointer_[2]; break;
        case 3 : access_ptr = pointer_[3]; break;
      }
    } else if (Detail::kPointerCount == 8) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0]; break;
        case 1 : access_ptr = pointer_[1]; break;
        case 2 : access_ptr = pointer_[2]; break;
        case 3 : access_ptr = pointer_[3]; break;
        case 4 : access_ptr = pointer_[4]; break;
        case 5 : access_ptr = pointer_[5]; break;
        case 6 : access_ptr = pointer_[6]; break;
        case 7 : access_ptr = pointer_[7]; break;
      }
    } else if (Detail::kPointerCount == 16) {
      switch (iteration_strided_ % Detail::kPointerCount) {
        case 0 : access_ptr = pointer_[0];  break;
        case 1 : access_ptr = pointer_[1];  break;
        case 2 : access_ptr = pointer_[2];  break;
        case 3 : access_ptr = pointer_[3];  break;
        case 4 : access_ptr = pointer_[4];  break;
        case 5 : access_ptr = pointer_[5];  break;
        case 6 : access_ptr = pointer_[6];  break;
        case 7 : access_ptr = pointer_[7];  break;
        case 8 : access_ptr = pointer_[8];  break;
        case 9 : access_ptr = pointer_[9];  break;
        case 10: access_ptr = pointer_[10]; break;
        case 11: access_ptr = pointer_[11]; break;
        case 12: access_ptr = pointer_[12]; break;
        case 13: access_ptr = pointer_[13]; break;
        case 14: access_ptr = pointer_[14]; break;
        case 15: access_ptr = pointer_[15]; break;
      }
    } else {
      access_ptr = pointer_[0];
    }

    int stride_idx = (iteration_strided_ & ~(Detail::kPointerCount - 1));

    int access_offset =
        stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
        // kCrosswise elements in the contiguous dimension would span to a
        // shared memory cache line.
        iteration_contiguous_ * (ThreadMap::Delta::kContiguous / kCrosswise) *
            (Layout::TileShape::kContiguous) * Layout::kElementsPerAccess * 2;
    
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<Element *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next section.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  HYTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  HYTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * sections_per_stage_ * stride_ / sections_ +
                       coord.strided() * Shape::kStrided * stride_ / Layout::kFactor);
  }
};

}  // namespace threadblock
}  // namespace transform
}  // namespace hytlass

////////////////////////////////////////////////////////////////////////////////

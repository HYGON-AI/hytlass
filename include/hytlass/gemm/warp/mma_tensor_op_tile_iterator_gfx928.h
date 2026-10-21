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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "hytlass/hytlass.h"

#include "hytlass/array.h"
#include "hytlass/numeric_types.h"
#include "hytlass/tensor_ref.h"
#include "hytlass/matrix_shape.h"

#include "hytlass/arch/memory_gfx928.h"
#include "hytlass/gemm/gemm.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/layout/tensor.h"
#include "hytlass/layout/pitch_linear.h"
#include "hytlass/layout/tensor_op_multiplicand_gfx928.h"

#include "hytlass/platform/platform.h"
#include "hytlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    /// Decide whether or not to use ds_read_matrix
    bool DsReadEnable>
class MmaGfx928TensorOpMultiplicandTileIteratorBase;

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It uses ds_read_m to load 
/// from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// The shared memory layout must be congruous and non-swizzled, and the warp shape's M and 
/// N dimensions must each be at least 32.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaGfx928TensorOpMultiplicandTileIteratorBase<
    Shape_, Operand_, Element_,
    Layout_, InstructionShape_,
    OpDelta_, PartitionsK_, true> {
public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  static_assert(kThreads == 64, 
    "Make sure WARP_SIZE_GPU equal 64 to use MmaTensorOpMultiplicandTileIterator\n");

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Number of threads along the contiguous dimension.
    // Eg.  float 32 / 4 -> 8
    static int const kThreadsInContiguous = 32 / 
      (128 / hytlass::sizeof_bits<Element>::value);

    // ds_read fetches 32 elements along the m/n dimension.
    static int const kLdsOpOuter = 32;

    // Thread count along k is derived from the data type.
    // Eg.  float: 256 / 32 = 8
    //      fp16:  256 / 16 = 16
    //      s8:    256 / 8  = 32
    static int const kLdsOpInner = 256 / hytlass::sizeof_bits<Element>::value;
    

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");
    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const* pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {

    int access_contiguous = (lane_id & (Policy::kThreadsInContiguous - 1)) * 
        (128 / hytlass::sizeof_bits<Element>::value);
    int access_strided = (lane_id) / Policy::kThreadsInContiguous;

    pointer_ = reinterpret_cast<AccessType const*>(ref.data()) + 
        access_contiguous + access_strided * stride_;
    
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_tile_offset(
      TensorCoord const &tile_offset) {
    
    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 tile_offset.contiguous() * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    __uint128_t *fetch_ptr = reinterpret_cast<__uint128_t *>(&frag);
    
    HYTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c){ 
          int access_idx = c ;
          int access_idx_contiguous = c * Policy::kLdsOpOuter;

          AccessType const *source_ptr = pointer_ + access_idx_contiguous + 
              (byte_offset + byte_offset_) / sizeof(AccessType);
              
          arch::ds_read<Element>(source_ptr, fetch_ptr[access_idx]);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};


////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It uses ds_read_m to load 
/// from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// The shared memory layout must be congruous and swizzled, and the warp shape's M and 
/// N dimensions must each be at least 32.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaGfx928TensorOpMultiplicandTileIteratorBase<
    Shape_, Operand_, Element_,
    layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>, 
    InstructionShape_,
    OpDelta_, PartitionsK_, true> {
public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  static_assert(kThreads == 64, 
    "Make sure WARP_SIZE_GPU equal 64 to use MmaTensorOpMultiplicandTileIterator\n");

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Number of threads along the contiguous dimension.
    // Eg.  float 32 / 4 -> 8
    static int const kThreadsInContiguous = 32 / 
      (128 / hytlass::sizeof_bits<Element>::value);

    // ds_read fetches 32 elements along the m/n dimension.
    static int const kLdsOpOuter = 32;

    // Thread count along k is derived from the data type.
    // Eg.  float: 256 / 32 = 8
    //      fp16:  256 / 16 = 16
    //      s8:    256 / 8  = 32
    static int const kLdsOpInner = 256 / hytlass::sizeof_bits<Element>::value;
    

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = std::min(
    Policy::LdsIterations::kContiguous,
    Layout::TileShape::kStrided);

  /// Vectorized access is not used
  static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {

    int access_contiguous = (lane_id & (Policy::kThreadsInContiguous - 1)) * 
        kElementsPerAccess;
    int access_strided = (lane_id) / Policy::kThreadsInContiguous;

    HYTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      TensorCoord base_coord = TensorCoord {
        access_contiguous + i * Policy::kLdsOpOuter,
        access_strided
      };

      Index offset = ref.offset(base_coord);

      pointer_[i] = reinterpret_cast<AccessType const*>(ref.data() + offset);
    }
    
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_tile_offset(
      TensorCoord const &tile_offset) {
    int k_delta_inst = tile_offset.contiguous() * Policy::LdsIterations::kContiguous;
    int k_delta_mn = k_delta_inst % Layout::TileShape::kStrided;
    int k_whole_tile_mn = (k_delta_inst / Layout::TileShape::kStrided) * Layout::TileShape::kStrided;

    if (k_delta_mn != 0) {
      if constexpr (hytlass::sizeof_bits<Element>::value == 16) {
        uintptr_t ptr_base = reinterpret_cast<uintptr_t>(pointer_[0]);
        pointer_[0] = reinterpret_cast<AccessType const*>(
          ptr_base ^ (k_delta_mn << 6)
        );
      } else if (hytlass::sizeof_bits<Element>::value == 8) {
        // Three cases are handled here:
        // 1. Layout::TileShape::kStrided == 2
        // 2. Layout::TileShape::kStrided == 4 && kPointer == 1
        // 3. Layout::TileShape::kStrided == 4 && kPointer == 2
        HYTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount; ++i) {
          uintptr_t ptr_base = reinterpret_cast<uintptr_t>(pointer_[i]);
          pointer_[i] = reinterpret_cast<AccessType const*>(
            ptr_base ^ (k_delta_mn << 5)
          );
        }
      }
    }

    int offset = (tile_offset.strided() * Policy::kLdsOpInner) * stride_ + 
                  k_whole_tile_mn * Policy::kLdsOpOuter;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    __uint128_t* fetch_ptr = reinterpret_cast<__uint128_t *>(&frag);
    
    HYTLASS_PRAGMA_UNROLL
    for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c){ 
          int access_idx = c ;
          int access_idx_contiguous = c * Policy::kLdsOpOuter;

          AccessType const* source_ptr = nullptr;

          // Explicitly access all elements to prevent the compiler from spilling 
          // pointer_ to global memory (GMEM).
          if constexpr (kPointerCount == 4) {
            switch (c % kPointerCount) {
              case 0 : source_ptr = pointer_[0]; break;
              case 1 : source_ptr = pointer_[1]; break;
              case 2 : source_ptr = pointer_[2]; break;
              case 3 : source_ptr = pointer_[3]; break;
            }
          } else if (kPointerCount == 2) {
            switch (c % kPointerCount) {
              case 0 : source_ptr = pointer_[0]; break;
              case 1 : source_ptr = pointer_[1]; break;
            }
          } else {
            source_ptr = pointer_[0];
          }

          source_ptr += (c / kPointerCount) * kPointerCount * Layout::kAccessPerUnit;

          char const* source_byte_ptr = reinterpret_cast<char const*>(source_ptr) + 
            byte_offset + byte_offset_;

          hytlass::arch::ds_read<Element>(source_byte_ptr, fetch_ptr[access_idx]);
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    // byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It uses ds_read_elem to load 
/// from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// The shared memory layout must be congruous and non-swizzled, and and this specialization 
/// is only applicable when the warp shape's M and N dimensions are both less than 32.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaGfx928TensorOpMultiplicandTileIteratorBase<
    Shape_, Operand_, Element_,
    Layout_, InstructionShape_,
    OpDelta_, PartitionsK_, false> {
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = Layout_;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  static_assert(kThreads == 64, 
    "Make sure WARP_SIZE_GPU equal 64 to use MmaGfx928TensorOpMultiplicandTileIteratorBase\n");

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    static int const kValStridedPerThread = 8 / sizeof(Element);
    static int const kThreadsInStrided = 4;

    // Simulates half of a ds_read, sufficient to issue one basic MMA instruction.
    static int const kLdsOpOuter = 16;
    static int const kLdsOpInner = kThreadsInStrided * kValStridedPerThread;
    

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpOuter;
    static int const LdsShapeStrided = 
        InstructionShape::kStrided / kLdsOpInner;
    
    // Number of ds_read_m operations required to fulfill the instruction.
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const* pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
    int access_strided = ((lane_id >> 4) << 3) / sizeof(Element);
    int access_contiguous = (lane_id & 15);

    pointer_ = reinterpret_cast<AccessType const*>(ref.data()) + 
               access_contiguous + access_strided * stride_;
    
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_tile_offset(
      TensorCoord const &tile_offset) {
    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 tile_offset.contiguous() * Shape::kContiguous;

    add_pointer_offset(offset);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
            int access_idx =
                (((c + s * Policy::LdsIterations::kContiguous) *
                               Policy::LdsShape::kStrided) *
                         Policy::LdsShape::kContiguous) * Policy::kValStridedPerThread;
            int access_idx_contiguous = (c * Policy::LdsShape::kContiguous) * Policy::kLdsOpOuter;
            int access_idx_strided = (s * Policy::LdsShape::kStrided) * Policy::kLdsOpInner;

            for(int i = 0; i < Policy::kValStridedPerThread; i++){

              AccessType const *source_ptr = pointer_ + access_idx_contiguous + (access_idx_strided + i) * stride_;

              char const *source_byte_ptr =
                  reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

              fetch_ptr[access_idx + i] =
                  *reinterpret_cast<Element const *>(source_byte_ptr);
          }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It uses ds_read_128 to load 
/// from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// The shared memory layout must be crosswise and swizzled. Applicable to instructions 
/// that perform two MMAC operations splice along the K dimension
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaGfx928TensorOpMultiplicandTileIteratorBase<
    Shape_, Operand_, Element_,
    hytlass::layout::TensorOpMultiplicandCrosswise128b<hytlass::sizeof_bits<Element_>::value, Crosswise>, 
    InstructionShape_,
    OpDelta_, PartitionsK_, false> {
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::TensorOpMultiplicandCrosswise128b<hytlass::sizeof_bits<Element>::value, Crosswise>; 

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  static_assert((64 / hytlass::sizeof_bits<Element>::value) * 8 == InstructionShape::kContiguous, 
    "The specialization requires mmac insturctions to be spliced once in k dim");
  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  static_assert(kThreads == 64, 
    "Make sure WARP_SIZE_GPU equal 64 to use TensorOpMultiplicandCrosswise128b\n");

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  static int const kCrosswise = Crosswise;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {

    static int const kValContiguousPerThread = 128 / sizeof_bits<Element>::value;
    static int const kThreadsInContiguous = 4;

    // Accesses 16 elements along the m/n dimension, corresponding to the strided dimension.
    static int const kLdsOpOuter = 16;
    // Number of elements accessed along the k dimension, corresponding to the contiguous dimension.
    static int const kLdsOpInner = kValContiguousPerThread * kThreadsInContiguous;

    static_assert(!(Shape::kContiguous % kLdsOpInner),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    static_assert(!(Shape::kStrided % kLdsOpOuter),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    // Number of LDS accesses required in contiguous, based on the size of the MMA instruction.
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpInner;
    // Typically 1, since InstructionShape matches the ds_read_m shape.
    static int const LdsShapeStrided = 
        InstructionShape::kStrided / kLdsOpOuter;
    
    // Number of ds_read_m operations required to complete the instruction.
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        1, Shape::kStrided / LdsShapeStrided / kLdsOpOuter>;

    // When kFactor == 1, one TileShape::kContiguous is composed of two mmac operations.
    // Valid only for kFactor == 1.
    static int const kGroupsPerTile = Layout::TileShape::kContiguous / 
                                      Layout::kFactor / kThreadsInContiguous;

    static int const kIterationsPerStage = Shape::kContiguous / InstructionShape::kContiguous;
    
    static_assert(Layout::kFactor != 8, "TensorOpMultiplicandCrosswise128b don't support kFactor ==8 yet.\n");
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  static int const kElementsPerAccess = 128 / sizeof_bits<Element>::value;

  /// Pointer type used for accesses
  using AccessType = int __attribute__((ext_vector_type(4)));

  /// Internal counter used to jump to next K partition
  // Used only when kFactor == 1 to track the current iteration state.
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kStrided * InstructionShape::kContiguous / kThreads>;

 private:

 int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const* pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  Index byte_offset_next_;

 public:
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase() : stride_(0), byte_offset_(0) {}

  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0) {
    // Record per-thread offsets in byte_offset_.
    int logical_coord_contiguous = lane_id / 16;
    int logical_coord_strided = lane_id & 15;
    TensorCoord logical_coord{
      logical_coord_contiguous * Layout::kElementsPerAccess,
      logical_coord_strided
    };
    byte_offset_ = ref.offset(logical_coord) * sizeof_bits<Element>::value / 8;

    if constexpr(Layout::kFactor == 1) {
      logical_coord = TensorCoord{
          (logical_coord_contiguous + 4) * Layout::kElementsPerAccess,
          logical_coord_strided
      };
      byte_offset_next_ = ref.offset(logical_coord) * sizeof_bits<Element>::value / 8;
    }

  } 

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_tile_offset(
      TensorCoord const &tile_offset) {
    // kFactor == 1 is a special case: a single tile along the contiguous dimension
    // contains two MMA instructions, requiring special handling for k-iteration.
    // When stepping by 1, k-iteration is achieved by jumping the byte_offset.

    // Normally, the strided-dimension offset per warp is fixed—except when
    // kFactor == 8 (crosswise == 8) with a warp shape of 32.
    // Therefore, apply the strided offset first.
    pointer_ += tile_offset.strided() * Shape::kStrided * stride_ / Layout::kFactor;

    // kFactor == 1 is a special case for the contiguous dimension:
    // a single tile spans two MMA instructions, so traversal is split into two steps.
    if constexpr (Layout::kFactor == 1) {
      int divisor = Policy::kGroupsPerTile;

      int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
      int k_group_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

      pointer_ += whole_tiles * Layout::TileShape::kContiguous;

      // Additional offset for transition from state 1 to state 2.
      int etc_move = k_group_idx_ % Policy::kGroupsPerTile;

      if (tile_offset.contiguous() > 0 && etc_move) {
        pointer_ += Layout::TileShape::kContiguous;
      }

      // Finally handle the offset jump.
      if (k_group_delta != 0) {
        hytlass::swap(byte_offset_, byte_offset_next_);
      }
    } else {
      pointer_ += tile_offset.contiguous() * Layout::TileShape::kContiguous;
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator++() {
    // Step for sub-iterations in the mainloop: 
    // advance by one InstructionShape along the contiguous dimension.
    add_tile_offset({1, 0});

    k_group_idx_ += 1;

    // Handle out-of-bounds access by stepping to the next shared memory block.
    if (k_group_idx_ >= Policy::kIterationsPerStage) {
      if constexpr (Layout::kFactor == 1 && Policy::kIterationsPerStage == 1) {
        // k_group_idx_ is 1 here, but iteration should start from 0—apply an offset jump.
        // This only occurs when kIterationsPerWarp == 1.
        hytlass::swap(byte_offset_, byte_offset_next_);
        pointer_ += Layout::TileShape::kContiguous * (kPartitionsK / Policy::kGroupsPerTile);
      } else if (Layout::kFactor == 1) {
        pointer_ += Policy::kIterationsPerStage * Layout::TileShape::kContiguous * (kPartitionsK - 1) / Policy::kGroupsPerTile;
      } else {
        pointer_ += Policy::kIterationsPerStage * Layout::TileShape::kContiguous * (kPartitionsK - 1);
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator--() {
    assert(0);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    AccessType* fetch_ptr = 
        reinterpret_cast<AccessType*>(&frag);
    

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      AccessType const* source_ptr = 
          pointer_ + s * Policy::kLdsOpOuter * stride_ / Layout::kFactor + 
          (byte_offset_ + byte_offset) / 16;
      fetch_ptr[s] = *source_ptr;
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() *
                          InstructionShape::kContiguous /
                          Layout::kElementsPerAccess +
                          tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group;
 }
};


/// This tile iterator is specialized for 64-thread TensorOps. It uses ds_read_64 to load 
/// from shared memory and therefore must be initialized with a TensorRef to shared memory. 
/// The shared memory layout must be crosswise and swizzled. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaGfx928TensorOpMultiplicandTileIteratorBase<
    Shape_, Operand_, Element_,
    hytlass::layout::TensorOpMultiplicandCrosswise64b<hytlass::sizeof_bits<Element_>::value, Crosswise>, 
    InstructionShape_,
    OpDelta_, PartitionsK_, false> {
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::TensorOpMultiplicandCrosswise64b<hytlass::sizeof_bits<Element>::value, Crosswise>; 

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  static_assert(kThreads == 64, 
    "Make sure WARP_SIZE_GPU equal 64 to use TensorOpMultiplicandCrosswise128b\n");

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {

    static int const kValContiguousPerThread = 64 / sizeof_bits<Element>::value;
    static int const kThreadsInContiguous = 4;

    // Accesses 16 elements along the m/n dimension, corresponding to the strided dimension.
    static int const kLdsOpOuter = 16;
    // Number of elements accessed along the k dimension, corresponding to the contiguous dimension.
    static int const kLdsOpInner = kValContiguousPerThread * kThreadsInContiguous;

    static_assert(!(Shape::kContiguous % kLdsOpInner),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    static_assert(!(Shape::kStrided % kLdsOpOuter),
                  "Shape of warp-level mma must satisfy the requirements of the warp-level ds_read.");

    // Number of LDS accesses required in contiguous, based on the size of the MMA instruction.
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpInner;
    // Typically 1, since InstructionShape matches the ds_read_m shape.
    static int const LdsShapeStrided = 
        InstructionShape::kStrided / kLdsOpOuter;
    

    // Number of ds_read_m operations required to complete the instruction.
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        1, Shape::kStrided / LdsShapeStrided / kLdsOpOuter>;

    /// Number of MMA instructions' worth of k-dimension data contained in one tile.
    static int const kGroupsPerTile = Layout::TileShape::kContiguous / (Layout::kFactor * 2);

    static int const kIterationsPerStage = Shape::kContiguous / InstructionShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  static int const kElementsPerAccess = 64 / sizeof_bits<Element>::value;

  /// Pointer type used for accesses
  using AccessType = int __attribute__((ext_vector_type(2)));

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kStrided * InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  int sections_;

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const* pointer_;

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

  // Indicates which stage of k-dimension iteration is currently active.
  int k_group_idx_;

  /// Marks the starting index of the PartitionK split.
  int k_start_by_;

  /// Indicates whether k_start_by_ has been initialized.
  bool k_start_by_init_;

 public:
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase(TensorRef const &ref, int lane_id)
      : pointer_(reinterpret_cast<AccessType const*>(ref.data())),
        sections_(ref.stride(0) / Crosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0),
        k_group_idx_(0),
        k_start_by_(0),
        k_start_by_init_(false) {
    int logical_coord_contiguous = lane_id / 16;
    int logical_coord_strided = lane_id & 15;
    TensorCoord logical_coord {
      logical_coord_contiguous * Layout::kElementsPerAccess,
      logical_coord_strided
    };

    byte_offset_ = ref.offset(logical_coord) * sizeof_bits<Element>::value / 8;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_pointer_offset(LongIndex offset) {
    byte_offset_ = offset * sizeof_bits<Element>::value / 8;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &add_tile_offset(
      TensorCoord const &tile_offset) {
    int current_k_group_idx = k_start_by_ + k_group_idx_;

    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_group_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    uint32_t mask = 0;

    if constexpr (Layout::kFactor == 2) {
      // In this case, 4 MMA instructions reside within a single TileShape::kContiguous;
      // compute the current lane's offset using a mask.
      // A forward step of 3 occurs only when initializing warp offsets.
      mask = (k_group_delta == 2) ? (4 << 3) :
             (k_group_delta == 1) ? ((2 + 4 * (current_k_group_idx & 1)) << 3) : 
             (k_group_delta == 3) ? ((2 + 4 * (~current_k_group_idx & 1)) << 3) : 0;
      
      // Advance to the next TileShape::kContiguous and adjust pointer_ offset.
      if (tile_offset.contiguous() > 0) {
        whole_tiles += ((current_k_group_idx % Policy::kGroupsPerTile) + k_group_delta) / 
          Policy::kGroupsPerTile;
      }
    } else if (Layout::kFactor == 4) {
      // In this case, 2 MMA instructions reside within a single TileShape::kContiguous;
      // compute the current lane's offset using a mask.
      mask = k_group_delta == 0 ? 0 : 2 << 3;

      // Advance to the next TileShape::kContiguous and adjust the pointer_ offset.
      if (tile_offset.contiguous() > 0) {
        whole_tiles += ((current_k_group_idx % Policy::kGroupsPerTile) + k_group_delta) / 
          Policy::kGroupsPerTile;
      }
    }
    
    byte_offset_ ^= mask;

    pointer_ += 
          tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor + 
          whole_tiles * Layout::TileShape::kContiguous * 2;

    // Initialize only on the first call to tile_offset, computing the starting offsets for each PartitionK segment.
    if (!k_start_by_init_) {
      k_start_by_ = tile_offset.contiguous();
      k_start_by_init_ = true;
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator++() {
    add_tile_offset({1, 0});

    k_group_idx_ += 1;

    // Handle out-of-bounds logic in operator++: when exceeding the current warp's maximum iteration count,
    // adjust byte_offset_ and pointer_ to support multi-stage pipelining.
    if (k_group_idx_ >= Policy::kIterationsPerStage) {
      if constexpr (kPartitionsK != 1) {
        int current_k_group_idx = k_start_by_ + k_group_idx_;
        int whole_tiles = Policy::kIterationsPerStage / Policy::kGroupsPerTile;
        int k_group_delta = Policy::kIterationsPerStage % Policy::kGroupsPerTile;
        whole_tiles += ((k_start_by_ % Policy::kGroupsPerTile) + k_group_delta) / Policy::kGroupsPerTile;

        uint32_t mask = 0;

        if constexpr (Layout::kFactor == 2) {
          // Construct a fallback mask to adjust byte_offset_.
          mask = (k_group_delta == 2) ? (4 << 3) :
                 (k_group_delta == 1) ? ((2 + 4 * (~current_k_group_idx & 1)) << 3) : 0;
        } else if constexpr (Layout::kFactor == 4) {
          mask = (k_group_delta == 0) ? 0 : 2 << 3;
        } else if constexpr (Layout::kFactor == 8) {
          mask = 0;
        } else {
          assert(0);
        }

        // go back
        byte_offset_ ^= mask;
        pointer_ -= whole_tiles * Layout::TileShape::kContiguous * 2;

        pointer_ += kPartitionsK * Policy::kIterationsPerStage / Policy::kGroupsPerTile * 
          Layout::TileShape::kContiguous * 2;
      }

      k_group_idx_ = 0;
    }
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  HYTLASS_HOST_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator--() {
    assert(0);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  HYTLASS_DEVICE
  MmaGfx928TensorOpMultiplicandTileIteratorBase &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    AccessType *fetch_ptr = reinterpret_cast<AccessType *>(&frag);
    // Normally there is no iteration along the contiguous dimension
    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      AccessType const* source_ptr = 
        pointer_ + s * Policy::kLdsOpOuter * stride_ / Layout::kFactor + 
        (byte_offset_ + byte_offset) / 8;
      fetch_ptr[s] = *source_ptr;

    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = tile_offset.contiguous() * 
                           InstructionShape::kContiguous / 
                           Layout::kElementsPerAccess + 
                           tile_offset.strided() * Shape::kStrided * stride_;

    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    k_group_idx_ = k_group;
  }
};


///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to congruous shared-memory 
/// layouts. Selects the corresponding Base implementation based on WarpShapeM and WarpShapeN.

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_,     
    64,
    PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  // Check whether the M and N dimensions satisfy the requirements of ds_read_m
  static const bool enable_ds_read = (Shape::kRow >= 32);

  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to congruous shared-memory 
/// layouts. Selects the corresponding Base implementation based on WarpShapeM and WarpShapeN.

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  static const bool enable_ds_read = (Shape::kColumn >= 32);

  /// Underlying tile iterator implementation
  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::NaiveTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to congruous shared-memory 
/// layouts. the warp shape's M and N dimensions must each be at least 32.

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_,     
    64,
    PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::ColumnMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value,
    Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  // Check whether the M and N dimensions satisfy the requirements of ds_read_m
  static_assert((Shape::kRow >= 32), "TensorOpMultiplicandCongruous128b requires m/n ge 32");

  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, PartitionsK_, true>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to congruous shared-memory 
/// layouts. the warp shape's M and N dimensions must each be at least 32.

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::RowMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::RowMajorTensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value,
    Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  static_assert((Shape::kColumn >= 32), "TensorOpMultiplicandCongruous128b requires m/n ge 32");


  /// Underlying tile iterator implementation
  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous128b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, PartitionsK_, true>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to crosswise shared-memory 
/// layouts. Applicable to instructions that perform two MMAC operations splice along the K dimension

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_,     
    64,
    PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  /// For K-major LDS, memory access must use ds_write_b64 only
  static const bool enable_ds_read = false;

  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to crosswise shared-memory 
/// layouts. Applicable to instructions that perform two MMAC operations splice along the K dimension

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  static const bool enable_ds_read = false;

  /// Underlying tile iterator implementation
  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise128b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to crosswise shared-memory 
/// layouts. 

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_,     
    64,
    PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  /// For K-major LDS, memory access must use ds_write_b64 only
  static const bool enable_ds_read = false;

  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps Applicable to crosswise shared-memory 
/// layouts. 

///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_,
    int Crosswise>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 64, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  static const bool enable_ds_read = false;

  /// Underlying tile iterator implementation
  using Base = MmaGfx928TensorOpMultiplicandTileIteratorBase<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise64b<sizeof_bits<Element_>::value, Crosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, PartitionsK_, enable_ds_read>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    --iterator_;
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  HYTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};



////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 64-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, hytlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  static int const kElementsPerAccess = 1;
  // 4 lanes along the n dimension
  static int const kRowsPerTile = 4;
  // 16 lanes along the m dimension
  static int const kColumnsPerTile = 16;

  // Number of registers per tile along the m dimension.
   static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {
    int row_id = lane_id >> 4;
    int col_id = lane_id & 15;

    // MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
    MatrixCoord lane_offset(row_id, col_id);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    HYTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow;

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        HYTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          frag[mma_accum_start + row] = offset_ref.at({accum_m + row * kRowsPerTile, accum_n});
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset
    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles
    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset
    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    HYTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow;

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        HYTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          offset_ref.at({accum_m + row * kRowsPerTile, accum_n}) = frag[mma_accum_start + row];
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset
    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles
    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, hytlass::layout::ColumnMajor, InstructionShape_, OpDelta_> {
 public:

  // TODO: Update the mapping scheme if an instruction-level column-major specialization is added

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = WARP_SIZE_GPU;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:


  static int const kElementsPerAccess = 1;

  // 4 lane along n-dim
  static int const kRowsPerTile = 4;
  // 16 lanes along m-dim
  static int const kColumnsPerTile = 16;

  // Number of registers a tile has in the M dimension
   static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {
    int row_id = lane_id >> 4;
    int col_id = lane_id & 15;

    // MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);
    MatrixCoord lane_offset(row_id, col_id);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    HYTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow;

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        HYTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          frag[mma_accum_start + row] = offset_ref.at({accum_m + row * kRowsPerTile, accum_n});
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset
    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles
    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset
    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    HYTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn;

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow;

        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        HYTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          offset_ref.at({accum_m + row * kRowsPerTile, accum_n}) = frag[mma_accum_start + row];
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset
    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles
    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};


////////////////////////////////////////////////////////////////////////////////
/// This tile iterator is specialized for 64-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element typ
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Interleaved N
    int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, hytlass::layout::TensorNCxHWx<InterleavedN>,
    InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = hytlass::layout::TensorNCxHWx<InterleavedN>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 64;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");
    

    /// Number of elements in strided dimension that each STG writes
    static int const kStridedPerSTG = 4;

    /// Factor to calculate reorder index to pack accumulator.
    static int const kAccumsInCrosswise = InterleavedN / 16 * 4;
    static int const kCrosswisesInColumn = Shape::kColumn / InterleavedN; 

    static_assert(Shape::kRow / kStridedPerSTG > 0 &&
                  Shape::kColumn / InterleavedN > 0,
        "Iteration in row and column must great equal 1\n");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / kStridedPerSTG,
                                      Shape::kColumn / InterleavedN>;
  };

private:

  static int const kElementsPerAccess = 1;

public:

  //
  // Derived quantities
  //

  struct alignas((kElementsPerAccess * sizeof_bits<Element>::value / 8)) AccessType {
      Array<Element, kElementsPerAccess> storage;
  };

  /// Fragment object holding a thread's part of a tile
  // using Fragment = Array<int32_t, Shape::kCount / kThreads>;
  using Fragment = Array<typename std::conditional<
                            std::is_same<Element, int8_t>::value  || 
                            std::is_same<Element, int32_t>::value || 
                            std::is_same<Element, uint8_t>::value,
                            int32_t, float>::type, 
                            Shape::kCount / kThreads>;


private:

  /// Reference to output tensor
  TensorRef ref_;

  /// Row offset index globally
  LongIndex global_offset_row_;

  /// Column offset index globally
  LongIndex global_offset_col_;

  /// Output tensor size
  TensorCoord extent_;

  /// Alpha 
  float alpha_;

  /// Beta
  float beta_;

public:
  
  /// Default ctor constructs null iterator
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int const lane_id,
    TensorCoord extent,
    float alpha = 1.0f,
    float beta = 0.0f
  ):
    ref_(ref),
    extent_(extent),
    alpha_(alpha),
    beta_(beta) {

    int row_id = (lane_id >> 4);
    int col_id = (lane_id & 15);

    global_offset_row_ = row_id;

    global_offset_col_ = col_id * kElementsPerAccess;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(MatrixCoord const &tile_offset) {

    global_offset_row_ += tile_offset.row() * Shape::kRow;

    global_offset_col_ += tile_offset.column() * Shape::kColumn;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  HYTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  HYTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  HYTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType* frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kN; ++mma_n) {
      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kM; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kM;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                 accum_m * offset_ref.stride(0) + accum_n);

        frag_ptr[idx] = access_ptr[0];
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  HYTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  HYTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    Array<float, Shape::kCount / kThreads> output_frag_f;
    Array<Element, Shape::kCount / kThreads> output_frag;

    LongIndex pq = extent_.h() * extent_.w();

    LongIndex extent_row = extent_.n() * pq;
    LongIndex extent_col = extent_.c();

    LongIndex k_major = (global_offset_col_ / InterleavedN) * pq;
    Index k_minor = global_offset_col_ % InterleavedN;
    LongIndex k_offset = k_major * InterleavedN + k_minor;
    LongIndex k_offset_delta = pq * InterleavedN;

    LongIndex stride_n = pq * extent_.c();

    Index n;
    LongIndex pq_rem;

    unsigned int pq_mul, pq_shr;
    find_divisor(pq_mul, pq_shr, pq);

    if (beta_ == 0.0f) {
      HYTLASS_PRAGMA_UNROLL
      for(int i = 0; i < frag.size(); ++i) {
        output_frag_f[i] = frag[i];
      }

      // reorder accum
      HYTLASS_PRAGMA_UNROLL
      for (int i = 0; i < frag.size(); ++i) {

        int tile_idx = i / (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);
        int in_tile_idx = i % (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);

        int remap_in_tile_idx = (in_tile_idx % (Shape::kColumn / 16)) * 4 
          + (in_tile_idx / (Shape::kColumn / 16)) 
          + tile_idx * (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);

        output_frag[i] = (Element)(output_frag_f[remap_in_tile_idx] * alpha_);

      }


      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        HYTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
       
          int accum_n = mma_n * InterleavedN;

          int idx = (mma_n + mma_m * Policy::MmaIterations::kColumn) * (InterleavedN / 16);
  
          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            HYTLASS_PRAGMA_UNROLL
            for (int _k = 0; _k < InterleavedN / 16; _k++) {
              access_ptr[_k * 16] = frag_ptr[idx + _k];

            }
          }
        }
      }
    } 
    else {
     HYTLASS_PRAGMA_UNROLL
      for (int i = 0; i < frag.size(); ++i) {
        int tile_idx = i / (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);
        int in_tile_idx = i % (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);
        int remap_in_tile_idx = (in_tile_idx % (Shape::kColumn / 16)) * 4 
          + (in_tile_idx / (Shape::kColumn / 16)) 
          + tile_idx * (Policy::kCrosswisesInColumn * Policy::kAccumsInCrosswise);

        output_frag_f[i] = frag[remap_in_tile_idx];
      }

      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      Array<Element, kElementsPerAccess> ref_frag;
      AccessType *ref_frag_ptr = reinterpret_cast<AccessType *>(&ref_frag);

      HYTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        HYTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
       
          int accum_n = mma_n * InterleavedN;

          int idx = (mma_n + mma_m * Policy::MmaIterations::kColumn) * (InterleavedN / 16);
         
          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            HYTLASS_PRAGMA_UNROLL
            for (int _k = 0; _k < InterleavedN / 16; _k++) {
              Element c_item = Element(alpha_ * output_frag_f[idx + _k] + beta_ * access_ptr->storage[_k * 16]);
              access_ptr->storage[_k * 16] = c_item;
            }

          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  HYTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  HYTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

} // namespace warp
} // namespace gemm
} // namespace hytlass

////////////////////////////////////////////////////////////////////////////////

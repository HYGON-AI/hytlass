#include "hip/hip_runtime.h"
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
    \brief Templates implementing loading of tiles from pitch-linear rank=2 tensors. 

    This iterator uses masks to guard out-of-bounds accesses and visits the last "residue" tile
    first, with the objective of minimizing predicate mask updates during steady-state operation.

    A precomputed "Params" object minimizes the amount of state that must be stored in registers,
    and integer addition is used to advance the pointer through memory.
*/

#pragma once

#include "hytlass/arch/memory_buffer.h"
#include "hytlass/transform/threadblock/predicated_tile_access_iterator_2dthreadtile.h"
#include "hytlass/transform/thread/transpose.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileIterator2dThreadTile
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
/// Regular tile iterator using a precomputed control structure to minimize register liveness
/// and integer arithmetic.
///
/// Layout is assumed to be invariant at the time the precomputed "Params" object is constructed.
///
/// Base pointer and tensor extents may be specified at the time the iterator is constructed.
/// Subsequently, they are assumed to be immutable.
///
/// Adding a logical coordinate offset may be performed at the time the iterator is constructed.
/// Subsequent additions to logical coordinate offset may be performed but are relatively expensive.
///
/// Vistitation order is intended to first visit a "residual" tile that may be partially full in
/// both the advance dimension and the steady-state dimension. This is assumed to be the last
/// tile in the iteration sequence. Advancing an iterator that has just been constructed moves to
/// the first tile that is full in the advance dimension and recomputes predicates. Subsequent
/// accesses may be performed without updating internal predicates and are efficient in terms of
/// live register state and pointer arithmetic instructions.
///
/// To be efficient, this assumes the iteraor will be dereferenced and advanced at least once
/// outside any looping structure to minimize integer arithmetic. 
///
/// Acceses out of bounds are safe so long as `clear_mask()` is called prior to dereferencing
/// the iterator.
///
///
/// Example:
///
/// An efficient pipeline structure may be constructed as follows:
///
// template <typename Iterator>
// __global__ void kernel(
//   typename Iterator::Params params, 
//   typename Iterator::Element *ptr,
//   TensorCoord extent) {
//
//   typename Iterator::Fragment fragment;
//
//   TensorCoord threadblock_offset(0, 0);
//
//   Iterator iter(params, ptr, extent, threadIdx.x, threadblock_offsets);
//
//
//   fragment = *iter;        // load "residue" tile first
//   ++iter;                  // advance to first "steady state" tile and update internal masks
//
//
//   #pragma unroll
//   for (int i = Remaining - 1; i >= 0; --i) {
//
//     f(fragment);
//
//     if (!i) {
//       iter.clear_mask();   // light-weight operation to clear masks - subsequent loads become NO-OPs.
//     }
//  
//     fragment = *iter;      // load tile during "steady state" phase
//     ++iter;                // advance to next tile - lightweight due to steady-state masks
//   }
// }
//
// void host(TensorView<Element, 2, layout::PitchLinear> view) {
//
//   using Iterator = transform::threadblock::PredicatedTileIterator2dThreadTile;
//
//   typename Iterator::Params params(view.layout());
//
//   kernel<Iterator>(params, view.data());
// }
///
///
template <
  typename Shape,
  typename Element,
  typename Layout,
  int AdvanceRank,
  typename ThreadMap,
  bool Transpose = false,
  int OffsetNoGuard = 0,
  bool BufferAccess = false,
  bool EnStaggerK = false
>
class PredicatedTileIterator2dThreadTile;

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_, 
  typename Element_, 
  int AdvanceRank, 
  typename ThreadMap_, 
  bool Transpose_, 
  int OffsetNoGuard, 
  bool BufferAccess,
  bool EnStaggerK_
>
class PredicatedTileIterator2dThreadTile<
    Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, Transpose_, 
    OffsetNoGuard, BufferAccess, EnStaggerK_> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  /// Type used for internal memory accesses
  /// extra set of parenthesis is needed for VS compiler
  struct alignas((ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value /
                  8)) AccessType {

    Array<Element, ThreadMap::kElementsPerAccess> storage;

    static int const kElements = ThreadMap::kElementsPerAccess;
  };
#ifdef MIX_FP16_DOT2
  using TransposeShape = typename std::conditional<
    hute::is_same_v<Element_, half_t>,
    layout::PitchLinearShape<2,2>,
    typename std::conditional<
      (hute::is_same_v<Element_, int8_t> || hute::is_same_v<Element_, uint8_t>),
      layout::PitchLinearShape<4,4>,
      void
    >::type
  >::type;

  static_assert(!hute::is_same_v<TransposeShape, void>);
  
  /// Optinally this fragment can be 4x4 or 2x2 transposed
  using Transform = thread::Transpose< ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount , TransposeShape, Element>;

#else
  /// Optinally this fragment can be 4x4 transposed
  using Transform = thread::Transpose< ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount , layout::PitchLinearShape<4,4>, Element>;
#endif
  static bool const transpose = Transpose_;

  static bool const DirectInc = BufferAccess;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = PredicatedTileAccessIterator2dThreadTile<
      Shape, Element, Layout, kAdvanceRank, ThreadMap, AccessType,
      OffsetNoGuard, BufferAccess, EnStaggerK, DirectInc>;

  /// Fragment object to be loaded or stored
  using Fragment = hytlass::Array<Element, ThreadMap::Iterations::kCount *
                                               ThreadMap::ThreadAccessShape::kCount>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    using Base = typename TileAccessIterator::Params::Base;

    friend PredicatedTileIterator2dThreadTile;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    HYTLASS_HOST_DEVICE
    Params(Layout const &layout) : params_(layout) { }
    
    HYTLASS_HOST_DEVICE
    Params() { }

    HYTLASS_HOST_DEVICE
    Params(Layout const &layout, Index stagger_k_log2, Index stagger_k_stride_log2):
      params_(layout, stagger_k_log2, stagger_k_stride_log2) { }

    HYTLASS_HOST_DEVICE
    Params(Base const &base) 
        : params_(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
      )
      : address_iterator_(params.params_, pointer, extent, thread_id,
                          threadblock_offset) {}

  /// Construct a PredicatedTileIterator2dThreadTile with zero threadblock offset
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : PredicatedTileIterator2dThreadTile(params, pointer, extent, thread_id,
                               make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile &operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile operator++(int) {
    PredicatedTileIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { address_iterator_.clear_mask(enable); }

  /// Adjust guard_offsets so that the subsequently computed vofft becomes -1,
  /// thereby masking out memory accesses
  HYTLASS_HOST_DEVICE
  void clear_buffer_vofft(bool enable = true) { address_iterator_.clear_buffer_vofft(enable); }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void enable_mask() { address_iterator_.enable_mask(); }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  HYTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { address_iterator_.set_mask(mask); }

  /// Gets the mask
  HYTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { address_iterator_.get_mask(mask); }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        HYTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++){
          HYTLASS_PRAGMA_UNROLL
          for (int tc = 0; tc < (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess); tc++) {

            int access_idx = ts + c * ThreadMap::ThreadAccessShape::kStrided  + \
                s * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided;

            address_iterator_.set_iteration_index(access_idx, tc);

            access_idx = tc + access_idx * (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess);

            if (OffsetNoGuard || address_iterator_.valid()) {

              if constexpr (BufferAccess) {
                hytlass::arch::BufferAccessor buffer_accessor = address_iterator_.get_buffer_accessor();
                // pointer_offset are added to vgpr offset now, but may be it should be added to sgpr offset in some cases
                // in common case, pointer_offset == 0
                hytlass::arch::buffer_load(frag_ptr[access_idx], buffer_accessor, pointer_offset, 0);
                // TODO: Add constant offset support to buffer_load when not 32-bit aligned,
                // i.e., when kElementsPerAccess < ThreadMap::ThreadAccessShape::kContiguous.
                //
                // Rationale: For fp16, two adjacent 16-bit elements can be loaded via
                // buffer_load_ushort with a fixed constant offset between them.
                // Since the intermediate segment cannot cross a 16-bit boundary (it always
                // lies within the valid region of the residue tile), out-of-bounds checks
                // are only needed at 32-bit granularity.
                // Using a constant offset allows both buffer_load_ushort instructions to
                // share the same VGPR offset, halving VGPR usage in the buffer load phase.
              } else {
                frag_ptr[access_idx] =
                    *(address_iterator_.get() + pointer_offset);
              }

            }
          }

          if (!(OffsetNoGuard && BufferAccess && DirectInc)) {
            ++address_iterator_;
          }

        }
      }
    }
    // TODO: Temporarily use this approach to avoid transpose issues. Generalize this without affecting the int8 transpose process.
    if(platform::is_same<Element, half_t>::value) {
      if (transpose && ThreadMap::ThreadAccessShape::kStrided == 2 && ThreadMap::ThreadAccessShape::kContiguous == 2) {
        Transform t;
        t.transform(frag, frag);
      }
    } else {
    if (transpose) {
      Transform t;
      t.transform(frag, frag);
      }
    }
  }
    HYTLASS_DEVICE
  void load_with_pointer_offset_always_check(Fragment &frag, Index pointer_offset) {

    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        HYTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++){
          HYTLASS_PRAGMA_UNROLL
          for (int tc = 0; tc < (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess); tc++) {

            int access_idx = ts + c * ThreadMap::ThreadAccessShape::kStrided  + \
                s * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided;

            address_iterator_.set_iteration_index(access_idx, tc);

            access_idx = tc + access_idx * (ThreadMap::ThreadAccessShape::kContiguous / ThreadMap::kElementsPerAccess);

            if (address_iterator_.valid()) {

              if constexpr (BufferAccess) {
                hytlass::arch::BufferAccessor buffer_accessor = address_iterator_.get_buffer_accessor();
                // pointer_offset are added to vgpr offset now, but may be it should be added to sgpr offset in some cases
                // in common case, pointer_offset == 0
                hytlass::arch::buffer_load(frag_ptr[access_idx], buffer_accessor, pointer_offset, 0);
              } else {
                frag_ptr[access_idx] =
                    *(address_iterator_.get() + pointer_offset);
              }

            }
          }

          if (!(OffsetNoGuard && BufferAccess && DirectInc)) {
            ++address_iterator_;
          }
        }
      }
    }

    if(platform::is_same<Element, half_t>::value)
    {
      if (transpose && ThreadMap::ThreadAccessShape::kStrided == 2 && ThreadMap::ThreadAccessShape::kContiguous == 2) {
        Transform t;
        t.transform(frag, frag);
      }
    }
    else
    {
    if (transpose) {
      Transform t;
      t.transform(frag, frag);
      }
    }
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) { load_with_pointer_offset(frag, 0); }

  /// Loads a fragment from memory, always check access is valid or not
  HYTLASS_DEVICE
  void load_always_check(Fragment &frag) { load_with_pointer_offset_always_check(frag, 0); }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    HYTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      HYTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        HYTLASS_PRAGMA_UNROLL
        for (int ts = 0; ts < ThreadMap::ThreadAccessShape::kStrided; ts++){

          int access_idx = ts + c * ThreadMap::ThreadAccessShape::kStrided  + \
              s * ThreadMap::Iterations::kContiguous * ThreadMap::ThreadAccessShape::kStrided;

          address_iterator_.set_iteration_index(access_idx);
          if (address_iterator_.valid()) {
            *(address_iterator_.get() + pointer_offset) = frag_ptr[access_idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) { store_with_pointer_offset(frag, 0); }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  bool Transpose_,
  int OffsetNoGuard,
  bool BufferAccess,
  bool EnStaggerK_
>
class PredicatedTileIterator2dThreadTile<
    Shape_, Element_, layout::ColumnMajor, AdvanceRank, ThreadMap_, Transpose_,
    OffsetNoGuard, BufferAccess, EnStaggerK_> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static bool const Transpose = Transpose_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 0 : 1),
    ThreadMap,
    Transpose,
    OffsetNoGuard,
    BufferAccess,
    EnStaggerK
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = hytlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:
    
    HYTLASS_HOST_DEVICE
    Params() { }

    /// Construct the Params object given a pitch-linear tensor's layout
    HYTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) {}

    HYTLASS_HOST_DEVICE
    Params(Layout const &layout, Index stagger_k_log2, Index stagger_k_stride_log2): 
      params_(layout::PitchLinear(layout.stride(0)), stagger_k_log2, stagger_k_stride_log2) {}

    HYTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,         ///< Initial offset of threadblock
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.row(), extent.column()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.row(), threadblock_offset.column())
    ) { }

  /// Construct a PredicatedTileIterator2dThreadTile with zero threadblock offset
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator2dThreadTile(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile operator++(int) {
    PredicatedTileIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Adjust guard_offsets so that the subsequently computed vofft equals -1.
  HYTLASS_HOST_DEVICE
  void clear_buffer_vofft(bool enable = true) {
    iterator_.clear_buffer_vofft(enable);
  }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  HYTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  HYTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory, always check access is valid or not
  HYTLASS_DEVICE
  void load_with_pointer_offset_always_check(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset_always_check(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory, always check access is valid or not
  HYTLASS_DEVICE
  void load_always_check(Fragment &frag) {
    load_with_pointer_offset_always_check(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of PredicatedTileIterator2dThreadTile for pitch-linear data.
///
/// Satisfies: ForwardTileIteratorConcept | 
///            ReadableContiguousTileIteratorConcept | 
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
  typename Shape_,
  typename Element_,
  int AdvanceRank,
  typename ThreadMap_,
  bool Transpose_,
  int OffsetNoGuard,
  bool BufferAccess,
  bool EnStaggerK_
>
class PredicatedTileIterator2dThreadTile<
    Shape_, Element_, layout::RowMajor, AdvanceRank, ThreadMap_, Transpose_,
    OffsetNoGuard, BufferAccess, EnStaggerK_> {
public:

  static_assert(AdvanceRank == 0 || AdvanceRank == 1, 
    "Specialization for pitch-linear iterator may along advance along the "
    "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  static bool const Transpose = Transpose_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingIterator = PredicatedTileIterator2dThreadTile<
    layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
    Element,
    layout::PitchLinear,
    (kAdvanceRank == 0 ? 1 : 0),
    ThreadMap,
    Transpose,
    OffsetNoGuard,
    BufferAccess,
    EnStaggerK
  >;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = hytlass::Array<Element, ThreadMap::Iterations::kCount * ThreadMap::ThreadAccessShape::kCount>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
  private:

    friend PredicatedTileIterator2dThreadTile;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

  public:
    
    HYTLASS_HOST_DEVICE
    Params() { } 

    /// Construct the Params object given a pitch-linear tensor's layout
    HYTLASS_HOST_DEVICE
    Params(Layout const &layout): params_(layout::PitchLinear(layout.stride(0))) { }

    HYTLASS_HOST_DEVICE
    Params(Layout const &layout, Index stagger_k_log2, Index stagger_k_stride_log2): 
      params_(layout::PitchLinear(layout.stride(0)), stagger_k_log2, stagger_k_stride_log2) {}

    HYTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const &base) 
        : params_(base) {}
  };


private:

  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

public:

  /// Constructs a TileIterator from its precomputed state, threadblock offset, and thread ID
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
    Params const &params,                         ///< Precomputed parameters object 
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id,                                ///< ID of each participating thread
    TensorCoord const &threadblock_offset,         ///< Initial offset of threadblock
    int const *indices = nullptr     ///< gather/scatter indices, note no support for gather/scatter at this specialization
  ):
    iterator_(
      params.params_,
      pointer,
      layout::PitchLinearCoord(extent.column(), extent.row()),
      thread_id,
      layout::PitchLinearCoord(threadblock_offset.column(), threadblock_offset.row())
    ) { }

  /// Construct a PredicatedTileIterator2dThreadTile with zero threadblock offset
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile(
    Params const &params,                         ///< Precomputed parameters object
    Pointer pointer,                              ///< Pointer to start of tensor
    TensorCoord extent,                           ///< Extent of tensor
    int thread_id                                 ///< ID of each participating thread
  ): PredicatedTileIterator2dThreadTile(params, pointer, extent, thread_id, make_Coord(0, 0)) { }

  /// Adds a pointer offset in units of Element
  HYTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  HYTLASS_HOST_DEVICE
  PredicatedTileIterator2dThreadTile operator++(int) {
    PredicatedTileIterator2dThreadTile self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Adjust guard_offsets so that the subsequently computed vofft becomes -1,
  /// thereby masking out memory accesses
  HYTLASS_HOST_DEVICE
  void clear_buffer_vofft(bool enable = true) {
    iterator_.clear_buffer_vofft(enable);
  }

  /// Clears the predicate set efficiently
  HYTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  HYTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  HYTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory, always check access is valid or not
  HYTLASS_DEVICE
  void load_with_pointer_offset_always_check(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset_always_check(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  HYTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory, always check access is valid or not
  HYTLASS_DEVICE
  void load_always_check(Fragment &frag) {
    load_with_pointer_offset_always_check(frag, 0);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  HYTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace hytlass

////////////////////////////////////////////////////////////////////////////////

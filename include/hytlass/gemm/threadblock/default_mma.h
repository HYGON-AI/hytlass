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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/arch/arch.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/layout/permute.h"
#include "hytlass/transform/threadblock/predicated_tile_iterator.h"
#include "hytlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/threadblock/default_mma_core_simt.h"
#include "hytlass/gemm/threadblock/default_mma_core_gfx928.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    /// Use buffer load
    bool BufferAccess = true,
    /// Use stragger-k method
    bool EnStaggerK = false
    >
struct DefaultMma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass Simt)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout,
    /// Use buffer load
    bool BufferAccess_,
    /// Use stragger-k method
    bool EnStaggerK_
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout, 
                  BufferAccess_, EnStaggerK_> {
  static constexpr bool BufferAccess = BufferAccess_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout, BufferAccess, EnStaggerK>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout, BufferAccess, EnStaggerK>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element, 
                            IteratorB::Fragment::kElements>,
      bool>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for arch::gfx928 row-major output (OperatorClass TensorOp)
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined or singlestage mainloop
    int stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout,
    /// Use buffer load
    bool BufferAccess_,
    /// Use stragger-k method
    bool EnStaggerK_
    >
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, arch::Gfx928, ThreadblockShape, WarpShape,
                  InstructionShape, stages, Operator, false, SharedMemoryClear,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout,
                  BufferAccess_, EnStaggerK_> {
  static constexpr bool BufferAccess = BufferAccess_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout, BufferAccess, EnStaggerK>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout, BufferAccess, EnStaggerK>;

  // Select pipeline or singleStage based on the stage
  using ThreadblockMma = typename platform::conditional<
      stages == 2,
      hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element, 
                            IteratorB::Fragment::kElements>,
      bool>,

      hytlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element,
                            IteratorB::Fragment::kElements>,
      bool>>::type;
   
};

////////////////////////////////////////////////////////////////////////////////

// This specialization should only be used when the instruction shape is GemmShape<16, 16, 16>

/// Specialization for row-major output (OperatorClass TensorOp)
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Number of stages used in the pipelined or singlestage mainloop
    int stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultMma<float, LayoutA, kAlignmentA, float, LayoutB,
                  kAlignmentB, float, layout::RowMajor,
                  arch::OpClassTensorOp, arch::Gfx928, ThreadblockShape, WarpShape,
                  GemmShape<16, 16, 16>, stages, Operator, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout, false, false> {

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, GemmShape<16, 16, 16>, float, LayoutA, float,
      LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
      arch::OpMultiplyAddFastF16>;

  // FastF16 does not currently support memory access via buffer_load
  static const int OffsetNoGuardA = 0;

  static const int OffsetNoGuardB = 0;

  static const bool BufferAccess = false;

  static constexpr bool EnStaggerK = BufferAccess;  


  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::PredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = typename platform::conditional<
      stages == 2,
      hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element, 
                            IteratorB::Fragment::kElements>,
      bool>,

      hytlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element,
                            IteratorB::Fragment::kElements>,
      bool>>::type;      
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column-major-interleaved output
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Number of Interleaved K
    int InterleavedK,
    /// Use buffer load
    bool BufferAccess_,    
    /// Use stragger-k method
    bool EnStaggerK_>
struct DefaultMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape, 2,
                  Operator, true, SharedMemoryClearOption::kNone, false, false,
                  layout::NoPermute, layout::NoPermute, BufferAccess_, EnStaggerK_> {
  static constexpr bool BufferAccess = BufferAccess_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2, Operator,
      true>;

  static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value, 
    "Alignment must match thread data map's vector length");

  static_assert(kAlignmentB ==128 / sizeof_bits<ElementB>::value,
    "Alignment must match thread data map's vector length");

  // Define iterators over tiles from the A operand
  using IteratorA = hytlass::transform::threadblock::PredicatedTileIterator<
      hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA,
      LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB = hytlass::transform::threadblock::PredicatedTileIterator<
      hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB,
      LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>,
      typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Use buffer load
    bool BufferAccess_,       
    /// Use stragger-k method
    bool EnStaggerK_
    >
struct DefaultMma<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                  ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
                  ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, 2,
                  Operator, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute,
                  BufferAccess_, EnStaggerK_> {
  static constexpr bool BufferAccess = BufferAccess_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using OperatorClass =  arch::OpClassSimt;

  static const bool transposeA = platform::is_same< LayoutA, layout::ColumnMajor >::value;
  static const bool transposeB = platform::is_same< LayoutB, layout::RowMajor >::value;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      OperatorClass, 2, Operator,
      DefaultMmaCoreConfig::AccumulatorsInRowMajor,
      DefaultMmaCoreConfig::CacheOpA,
      DefaultMmaCoreConfig::CacheOpB,
      DefaultMmaCoreConfig::TransformA,
      DefaultMmaCoreConfig::TransformB,
      DefaultMmaCoreConfig::IsComplex,
      kAlignmentA, kAlignmentB
      >;
    
  static int constexpr OffsetNoGuardA = BufferAccess ? 3 : 0;
  static int constexpr OffsetNoGuardB = BufferAccess ? 3 : 0;


  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA,
          OffsetNoGuardA, BufferAccess, EnStaggerK>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB,
          OffsetNoGuardB, BufferAccess, EnStaggerK>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element, 
                            IteratorB::Fragment::kElements>,
      bool>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT dot-fp16-fp32 Kernels
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Operation performed by GEMM
    typename Operator,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Use buffer load
    bool BufferAccess_,     
    /// Use stragger-k method
    bool EnStaggerK_>
struct DefaultMma<half_t, LayoutA, kAlignmentA, half_t, LayoutB, kAlignmentB,
                  ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
                  ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 2>, 2,
                  Operator, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute,
                  BufferAccess_, EnStaggerK_> {
  static constexpr bool BufferAccess = BufferAccess_;
  static constexpr bool EnStaggerK = EnStaggerK_;

  using InstructionShape = GemmShape<1, 1, 2>;
  using ElementA = half_t;
  using ElementB = half_t;
  using OperatorClass =  arch::OpClassSimt;

  static const bool transposeA = platform::is_same< LayoutA, layout::ColumnMajor >::value;
  static const bool transposeB = platform::is_same< LayoutB, layout::RowMajor >::value;

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      OperatorClass, 2, Operator, 
      DefaultMmaCoreConfig::AccumulatorsInRowMajor,
      DefaultMmaCoreConfig::CacheOpA,
      DefaultMmaCoreConfig::CacheOpB,
      DefaultMmaCoreConfig::TransformA,
      DefaultMmaCoreConfig::TransformB,
      DefaultMmaCoreConfig::IsComplex,
      kAlignmentA, kAlignmentB>;

  static int constexpr OffsetNoGuardA = BufferAccess ? 3 : 0;
  static int constexpr OffsetNoGuardB = BufferAccess ? 3 : 0;    

  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA,
          OffsetNoGuardA, BufferAccess, EnStaggerK>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB,
          OffsetNoGuardB, BufferAccess, EnStaggerK>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = hytlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy,
      NumericArrayConverter<typename MmaCore::SmemIteratorA::Element,
                            typename IteratorA::Element,
                            IteratorA::Fragment::kElements>,
      NumericArrayConverter<typename MmaCore::SmemIteratorB::Element, 
                            typename IteratorB::Element, 
                            IteratorB::Fragment::kElements>,
      bool>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace hytlass 

////////////////////////////////////////////////////////////////////////////////

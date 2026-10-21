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
    \brief Default template for a Blocked-Ell MMA.
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/arch/arch.h"
#include "hytlass/layout/matrix.h"
#include "hytlass/transform/threadblock/predicated_tile_iterator.h"
#include "hytlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/threadblock/default_mma_core_simt.h"
#include "hytlass/gemm/threadblock/default_mma_core_gfx928.h"
#include "hytlass/gemm/threadblock/ell_mma_pipelined.h"
#include "hytlass/gemm/threadblock/ell_mma_singlestage.h"
#include "hytlass/transform/threadblock/ell_predicated_tile_iterator.h"

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
    bool AccumulatorsInRowMajor = false
    >
struct DefaultEllMma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for arch::gfx928 row-major output (OperatorClass Simt)
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
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator>
struct DefaultEllMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static_assert((Stages == 1 || Stages == 2), "Ell block sparse gemm requires Stages set to value 1 or 2");

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::EllPredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::EllPredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Select pipeline or singleStage based on the number of stages
  using ThreadblockMma = typename platform::conditional<
      Stages == 2,
      hytlass::gemm::threadblock::EllMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>,

      hytlass::gemm::threadblock::EllMmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>
  >::type;
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
    /// Number of Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator
    >
struct DefaultEllMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, arch::Gfx928, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static_assert((Stages == 1 || Stages == 2), "Ell block sparse gemm requires Stages set to value 1 or 2");

  // Define the MmaCore components
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      hytlass::transform::threadblock::EllPredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      hytlass::transform::threadblock::EllPredicatedTileIterator<
          hytlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Select pipeline or singleStage based on the number of stages
  using ThreadblockMma = typename platform::conditional<
      Stages == 2,
      hytlass::gemm::threadblock::EllMmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>,

      hytlass::gemm::threadblock::EllMmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>
  >::type;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace hytlass 

////////////////////////////////////////////////////////////////////////////////

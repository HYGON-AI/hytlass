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
    \brief Default kernel-level Blocked-Ell sparse gemm operators.
      This operator combines threadblock-scoped ELL MMA
      with the appropriate threadblock-scoped epilogue.
*/

#pragma once

#include "hytlass/hytlass.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/numeric_types.h"

#include "hytlass/epilogue/threadblock/epilogue.h"
#include "hytlass/epilogue/thread/linear_combination.h"

#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/kernel/gemm.h"
#include "hytlass/gemm/kernel/gemm_pipelined.h"
#include "hytlass/gemm/threadblock/default_mma.h"
#include "hytlass/gemm/threadblock/default_mma_core_simt.h"
#include "hytlass/gemm/threadblock/threadblock_swizzle.h"
#include "hytlass/gemm/threadblock/default_mma_core_gfx928.h"

#include "hytlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "hytlass/epilogue/threadblock/default_epilogue_simt.h"
#include "hytlass/transform/threadblock/predicated_tile_iterator.h"

#include "hytlass/gemm/kernel/ell_gemm.h"
#include "hytlass/gemm/threadblock/default_ell_mma.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace kernel {

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
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse>
struct DefaultEllGemm;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Gfx928 Architecture
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
  /// Element type for C and D matrix operands
  typename ElementC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of Stages
  int Stages,
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial,
  /// Operation performed by GEMM
  typename Operator,
  /// Sparse matrix is A or not
  bool IsASparse
>
struct DefaultEllGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Gfx928,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  SplitKSerial,
  Operator,
  IsASparse
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename hytlass::gemm::threadblock::DefaultEllMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Gfx928,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    Stages,
    Operator
  >::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for SIMT
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
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of Stages
    int Stages,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator,
    /// Sparse matrix is A or not
    bool IsASparse
  >
struct DefaultEllGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    layout::RowMajor,
    ElementAccumulator,
    arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    SplitKSerial,
    Operator,
    IsASparse> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename hytlass::gemm::threadblock::DefaultEllMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      layout::RowMajor,
      arch::OpClassSimt,
      arch::Gfx906,
      ThreadblockShape,
      WarpShape,
      GemmShape<1, 1, 1>,
      Stages,
      Operator>::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename hytlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::EllGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, IsASparse>;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace hytlass

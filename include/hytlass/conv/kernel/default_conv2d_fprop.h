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
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped 
      matrix multiply-add with the appropriate threadblock-scoped epilogue.  
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/conv/kernel/default_conv2d.h"

#include "hytlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "hytlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "hytlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h"
#include "hytlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h"

#include "hytlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "hytlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "hytlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_fixed_channels.h"
#include "hytlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_few_channels.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kUnity,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / hytlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / hytlass::sizeof_bits<ElementB>::value
> struct DefaultConv2dFprop;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp convolutions 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and two stage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kFixedChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFixedChannels<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFixedChannels<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and single stage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kFixedChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      1, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFixedChannels<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFixedChannels<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for FewChannel IteratorAlgorithm and two stage
/// pipeline
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kFewChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFewChannels<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =

    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFewChannels<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static hytlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? hytlass::arch::CacheOperation::Global
          : hytlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and single stage
/// pipeline
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kFewChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      1, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFewChannels<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =

    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFewChannels<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static hytlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? hytlass::arch::CacheOperation::Global
          : hytlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and two stage
/// pipeline
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and single stage
/// pipeline
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      1, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and two stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      2, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and single stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      1, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm
/// and two stage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA,
        AccessTypeA 
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm
/// and single stage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      1, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = hytlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA,
        AccessTypeA 
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = hytlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and two stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      2, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and single stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      1, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                            OpClassSimt convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm, 
/// 2 stage pipeline, and FFMA-based mainloop for gfx906.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm, 
/// 2 stage pipeline, and FFMA-based mainloop for gfx906.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm, 
/// 1 stage pipeline, and FFMA-based mainloop for gfx906.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm, 
/// 1 stage pipeline, and FFMA-based mainloop for gfx906.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  1,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename hytlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    hytlass::conv::threadblock::TileIterator<
      hytlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        hytlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmSingleStage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = hytlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

} // namespace kernel
} // namespace conv
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////


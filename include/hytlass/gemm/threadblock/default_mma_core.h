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
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp instructions.
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/array.h"

#include "hytlass/numeric_types.h"
#include "hytlass/matrix_shape.h"

#include "hytlass/gemm/warp/mma.h"
#include "hytlass/gemm/threadblock/mma_pipelined.h"
#include "hytlass/gemm/threadblock/mma_singlestage.h"
#include "hytlass/arch/cache_operation.h" 
#include "hytlass/arch/mma.h" 

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

struct DefaultMmaCoreConfig {
    static constexpr bool AccumulatorsInRowMajor = false;
    static constexpr hytlass::arch::CacheOperation::Kind CacheOpA = hytlass::arch::CacheOperation::Global;
    static constexpr hytlass::arch::CacheOperation::Kind CacheOpB = hytlass::arch::CacheOperation::Global;
    static constexpr ComplexTransform TransformA = ComplexTransform::kNone;
    static constexpr ComplexTransform TransformB = ComplexTransform::kNone;
    static constexpr bool IsComplex = false;
    static constexpr int kAlignmentA = -1;
    static constexpr int kAlignmentB = -1;
};

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Number of stages
    int Stages = 2,
    /// Operation performed by MMA
    typename Operator = typename platform::conditional<
        (platform::is_same<OperatorClass,
                           hytlass::arch::OpClassTensorOp>::value) &&
            (platform::is_same<ElementA, int8_t>::value ||
             platform::is_same<ElementA, int4b_t>::value ||
             platform::is_same<ElementA, uint8_t>::value ||
             platform::is_same<ElementA, uint4b_t>::value),
        hytlass::arch::OpMultiplyAddSaturate,
        hytlass::arch::OpMultiplyAdd>::type,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = DefaultMmaCoreConfig::AccumulatorsInRowMajor,
    /// Cache operation of operand A
    hytlass::arch::CacheOperation::Kind CacheOpA =
        DefaultMmaCoreConfig::CacheOpA,
    /// Cache operation of operand B
    hytlass::arch::CacheOperation::Kind CacheOpB =
        DefaultMmaCoreConfig::CacheOpB,
    /// per-element transformation for elements of A
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// per-element transformation for elements of B
    ComplexTransform TransformB = ComplexTransform::kNone,
    bool IsComplex = DefaultMmaCoreConfig::IsComplex, // (is_complex<ElementA>::value || is_complex<ElementB>::value)
    int kAlignmentA = -1,
    int kAlignmentB = -1
>
struct DefaultMmaCore;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace hytlass

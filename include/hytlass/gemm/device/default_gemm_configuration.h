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
    \brief Definitions for GEMM structures
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/numeric_types.h"
#include "hytlass/arch/arch.h"
#include "hytlass/arch/mma.h"

#include "hytlass/gemm/gemm.h"
#include "hytlass/epilogue/thread/linear_combination.h"
#include "hytlass/epilogue/thread/linear_combination_clamp.h"

////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementA, 
  typename ElementB, 
  typename ElementC, 
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt, 
  ArchTag,
  ElementA, 
  ElementB, 
  ElementC, 
  ElementAccumulator> {
  
  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template < 
  typename ArchTag,
  typename ElementC>
struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {
  
  static int const kAlignmentA = 4;
  static int const kAlignmentB = 4;
  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 64, 32>;
  using InstructionShape = GemmShape<1, 1, 4>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
    ElementC,
    1,
    int32_t,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

// A float 
// B float
// C float
// Acc float
template <typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  float, 
  float, 
  ElementC, 
  float> {

  static_assert(std::is_same<ElementC, float>::value || std::is_same<ElementC, hytlass::half_t>::value, 
    "for ElementInputA/B ElementOutput must be float or half_t\n");

  static int const kAlignmentA = 128 / sizeof_bits<float>::value;
  static int const kAlignmentB = 128 / sizeof_bits<float>::value;

  using ThreadblockShape = GemmShape<128, 128, 16>;
  using WarpShape = GemmShape<128, 128, 16>;
  using InstructionShape = GemmShape<16, 16, 8>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    float,
    128 / sizeof_bits<ElementC>::value,
    float,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  hytlass::tfloat32_t, 
  hytlass::tfloat32_t, 
  ElementC, 
  float> {
  
  static_assert(std::is_same<ElementC, float>::value || std::is_same<ElementC, hytlass::half_t>::value, 
    "for ElementInputA/B ElementOutput must be float or half_t\n");

  static int const kAlignmentA = 128 / sizeof_bits<hytlass::tfloat32_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<hytlass::tfloat32_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 16>;
  using WarpShape = GemmShape<64, 64, 16>;
  using InstructionShape = GemmShape<16, 16, 8>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    float,
    128 / sizeof_bits<ElementC>::value,
    float,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

// A half_t
// B half_t
// C float
// Acc float
template <>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  half_t, 
  half_t, 
  float, 
  float> {
  
  static int const kAlignmentA = 128 / sizeof_bits<half_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<half_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    float,
    128 / sizeof_bits<float>::value,
    float,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

// A half_t
// B half_t
// C half_t
// Acc float
template <>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  half_t, 
  half_t, 
  half_t, 
  float> {
  
  static int const kAlignmentA = 128 / sizeof_bits<half_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<half_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    half_t,
    128 / sizeof_bits<half_t>::value,
    float,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  hytlass::bfloat16_t, 
  hytlass::bfloat16_t, 
  ElementC, 
  float> {

  static_assert(std::is_same<ElementC, hytlass::half_t>::value || std::is_same<ElementC, hytlass::bfloat16_t>::value || 
                     std::is_same<ElementC, float>::value, 
    "for ElementInputA/B ElementOutput must be int or int8_t\n");

  static int const kAlignmentA = 128 / sizeof_bits<hytlass::bfloat16_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<hytlass::bfloat16_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    float,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  int8_t, 
  int8_t, 
  ElementC, 
  int32_t> {
  
  static_assert(std::is_same<ElementC, int>::value || std::is_same<ElementC, int8_t>::value || 
                     std::is_same<ElementC, int32_t>::value, 
    "for ElementInputA/B ElementOutput must be int or int8_t\n");

  static int const kAlignmentA = 128 / sizeof_bits<int8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<int8_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 16, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    int32_t,
    int32_t
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp, 
  arch::Gfx928, 
  uint8_t, 
  uint8_t, 
  ElementC, 
  int32_t> {
  
  static_assert(std::is_same<ElementC, int>::value || std::is_same<ElementC, uint8_t>::value || 
                     std::is_same<ElementC, int32_t>::value, 
    "for ElementInputA/B ElementOutput must be int or uint8_t\n");

  static int const kAlignmentA = 128 / sizeof_bits<uint8_t>::value;
  static int const kAlignmentB = 128 / sizeof_bits<uint8_t>::value;

  using ThreadblockShape = GemmShape<128, 128, 64>;
  using WarpShape = GemmShape<64, 64, 64>;
  using InstructionShape = GemmShape<16, 16, 32>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    128 / sizeof_bits<ElementC>::value,
    int32_t,
    int32_t
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace hytlass

////////////////////////////////////////////////////////////////////////////////

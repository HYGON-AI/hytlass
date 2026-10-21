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
    \brief Unit tests for thread-level GEMM
*/

#include "../../common/hytlass_unit_test.h"

#include "hytlass/aligned_buffer.h"
#include "hytlass/half.h"
#include "hytlass/gemm/warp/default_mma_tensor_op.h"

#include "hytlass/epilogue/warp/fragment_iterator_tensor_op.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Epilogue_warp_FragmentIterator, mma_f32_64x64x16) {

  using Shape = hytlass::gemm::GemmShape<64, 64, 16>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;
  using Element = hytlass::half_t;
  using ElementC = float;
  using LayoutA = hytlass::layout::ColumnMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;
  using LayoutB = hytlass::layout::RowMajorNaiveTensorOpMultiplicandCongruous<
      hytlass::sizeof_bits<Element>::value>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
    Shape,
    InstructionShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementC,
    hytlass::layout::RowMajor
  >::Type;

  using FragmentIterator = hytlass::epilogue::warp::FragmentIteratorTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    hytlass::layout::RowMajor
  >;

  // This test just prints things.
  #if 0
  typename MmaTensorOp::FragmentC accum;

  std::cout << "Native accumulators:\n";

  for (int i = 0; i < MmaTensorOp::FragmentC::kElements; ++i) {
    accum[i] = ElementC(i);

    std::cout << accum[i] << " ";
    if (i && !((i + 1) % 4)) { 
      std::cout << "\n";
    }
  }

  std::cout << std::endl;

  std::cout << "FragmentIterator::Policy = { \n"
    << "  kAccessesPerInstruction:  " << FragmentIterator::Policy::kIterationsPerInstruction << "\n"
    << "  kAccumulatorRowStride:    " << FragmentIterator::Policy::kAccumulatorRowStride << "\n"
    << "  kAccumulatorColumnStride: " << FragmentIterator::Policy::kAccumulatorColumnStride << "\n"
    << "  kIterations:              " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  FragmentIterator fragment_iterator(accum);

  for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

    typename FragmentIterator::Fragment frag;

    fragment_iterator.load(frag);

    std::cout << "Iteration " << iter << ":\n";

    for (int i = 0; i < FragmentIterator::Fragment::kElements; ++i) {
      std::cout << frag[i] << " ";
    }

    std::cout << std::endl;

    ++fragment_iterator;
  }
  #endif
}

TEST(GFX928_Epilogue_warp_FragmentIterator, mma_s32_64x64x32) {

  using Shape = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;
  using Element = int8_t;
  using ElementC = int32_t;
  using LayoutA = hytlass::layout::RowMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;
  using LayoutB = hytlass::layout::ColumnMajorTensorOpMultiplicandCrosswise64b<
      hytlass::sizeof_bits<Element>::value, 32>;

  using MmaTensorOp = typename hytlass::gemm::warp::DefaultMmaTensorOp<
    Shape,
    InstructionShape,
    Element,
    LayoutA,
    Element,
    LayoutB,
    ElementC,
    hytlass::layout::RowMajor
  >::Type;

  using FragmentIterator = hytlass::epilogue::warp::FragmentIteratorTensorOp<
    Shape,
    typename MmaTensorOp::Policy::Operator::Shape,
    typename MmaTensorOp::Policy::Operator::ElementC,
    typename MmaTensorOp::Policy::Operator::FragmentC,
    hytlass::layout::RowMajor
  >;

  // This test just prints things.
  #if 0
  typename MmaTensorOp::FragmentC accum;

  std::cout << "Native accumulators:\n";

  for (int i = 0; i < MmaTensorOp::FragmentC::kElements; ++i) {
    accum[i] = ElementC(i);

    std::cout << accum[i] << " ";
    if (i && !((i + 1) % 8)) { 
      std::cout << "\n";
    }
  }

  std::cout << std::endl;

  std::cout << "FragmentIterator::Policy = { \n"
    << "  kAccessesPerInstruction:  " << FragmentIterator::Policy::kIterationsPerInstruction << "\n"
    << "  kAccumulatorRowStride:    " << FragmentIterator::Policy::kAccumulatorRowStride << "\n"
    << "  kAccumulatorColumnStride: " << FragmentIterator::Policy::kAccumulatorColumnStride << "\n"
    << "  kIterations:              " << FragmentIterator::Policy::kIterations << "\n"
    << " }" << std::endl;

  FragmentIterator fragment_iterator(accum);

  for (int iter = 0; iter < FragmentIterator::kIterations; ++iter) {

    typename FragmentIterator::Fragment frag;

    fragment_iterator.load(frag);

    std::cout << "Iteration " << iter << ":\n";

    for (int i = 0; i < FragmentIterator::Fragment::kElements; ++i) {
      std::cout << frag[i] << " ";
    }

    std::cout << std::endl;

    ++fragment_iterator;
  }
  #endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////

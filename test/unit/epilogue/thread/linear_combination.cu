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

#include "hytlass/epilogue/thread/linear_combination.h"
#include "hytlass/epilogue/thread/linear_combination_gelu.h"
#include "hytlass/epilogue/thread/activation.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination, device_side_f16_f32_value) {

  using Element = float;
  using ElementOutput = hytlass::half_t;
  int const kCount = 8;

  using LinearCombination = hytlass::epilogue::thread::LinearCombination<
    ElementOutput, 
    kCount,
    Element, 
    Element>;

  Element alpha = Element(2);
  Element beta = Element(1);

  typename LinearCombination::Params params(alpha, beta);

  LinearCombination linear_combination_op(params);

  hytlass::Array<ElementOutput, kCount> source;
  hytlass::Array<Element, kCount> accum;

  for (int i = 0; i < kCount; ++i) {
    accum[i] = Element(i * 2);
    source[i] = ElementOutput((i * 7 % 9) - 4);
  }

  hytlass::Array<ElementOutput, kCount> destination = linear_combination_op(accum, source);

  for (int i = 0; i < kCount; ++i) {
    
    ElementOutput expected = ElementOutput(
      alpha * accum[i] + 
      beta * Element(ElementOutput(source[i]))
    );

    ElementOutput got = destination[i];
    
    EXPECT_TRUE(expected == got);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination, device_side_f16_f32_ptr) {

  using Element = float;
  using ElementOutput = hytlass::half_t;
  int const kCount = 8;

  using LinearCombination = hytlass::epilogue::thread::LinearCombination<
    ElementOutput, 
    kCount,
    Element, 
    Element>;

  Element alpha = Element(2);
  Element beta = Element(1);

  typename LinearCombination::Params params(&alpha, &beta);

  LinearCombination linear_combination_op(params);

  hytlass::Array<ElementOutput, kCount> source;
  hytlass::Array<Element, kCount> accum;

  for (int i = 0; i < kCount; ++i) {
    accum[i] = Element(i * 2);
    source[i] = ElementOutput((i * 7 % 9) - 4);
  }

  hytlass::Array<ElementOutput, kCount> destination = linear_combination_op(accum, source);

  for (int i = 0; i < kCount; ++i) {
    
    ElementOutput expected = ElementOutput(
      alpha * accum[i] + 
      beta * Element(ElementOutput(source[i]))
    );

    ElementOutput got = destination[i];
    
    EXPECT_TRUE(expected == got);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////


TEST(Epilogue_thread_linear_combination_gelu, device_side_f16_f16_ptr) {

  using Element = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  int const kCount = 8;

  using LinearCombinationGELU = hytlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,
    kCount,
    Element,
    Element>;

  Element alpha = Element(1);
  Element beta = Element(0);

  typename LinearCombinationGELU::Params params(&alpha, &beta);

  LinearCombinationGELU linear_combination_op(params);

  hytlass::Array<Element, kCount> accum;

  for (int i = 0; i < kCount; ++i) {
    accum[i] = Element((float)i * 0.3f);
  }

  hytlass::Array<ElementOutput, kCount> destination = linear_combination_op(accum, accum);
  hytlass::epilogue::thread::GELU<ElementOutput> gelu_func;

  for (int i = 0; i < kCount; ++i) {
    ElementOutput expected = gelu_func(accum[i]);
    ElementOutput got = destination[i];
    EXPECT_TRUE(expected == got);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Epilogue_thread_linear_combination_gelu_taylor, device_side_f16_f16_ptr) {

  using Element = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  int const kCount = 8;

  using LinearCombinationGELU = hytlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,
    kCount,
    Element,
    Element>;

  Element alpha = Element(1);
  Element beta = Element(0);

  typename LinearCombinationGELU::Params params(&alpha, &beta);

  LinearCombinationGELU linear_combination_op(params);

  hytlass::Array<Element, kCount> accum;

  for (int i = 0; i < kCount; ++i) {
    accum[i] = Element((float)i * 0.3f);
  }

  hytlass::Array<ElementOutput, kCount> destination = linear_combination_op(accum, accum);
  hytlass::epilogue::thread::GELU<ElementOutput> gelu_func;

  for (int i = 0; i < kCount; ++i) {
    ElementOutput expected = gelu_func(accum[i]);
    ElementOutput got = destination[i];
    EXPECT_TRUE(expected == got);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////


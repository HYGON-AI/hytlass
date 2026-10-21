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
/* \file
   \brief Defines operations for reduction operation in HYTLASS Library.
*/
  
#include "hytlass/hytlass.h"
#include "hytlass/library/library.h"
#include "hytlass/library/manifest.h"

#include "reduction_operation.h"

namespace hytlass {
namespace library {

// naming convention initialize_reduce_[ReductionOp]_[EpilogueOp]_[ElementWorkspace]_[ElementAccumulator]_[ElementOutput]

//add fp16 * fp16 + fp32
void initialize_reduce_add_linear_combination_f16_f32_f16(Manifest &manifest) {

  using ElementWorkspace = hytlass::half_t;
  using ElementAccumulator = float;
  using ElementOutput = hytlass::half_t;
  
#ifdef MIX_FP16_EPILOGUE
  using ElementCompute = hytlass::half_t;
#else
  using ElementCompute = float;
#endif

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f16_f32_f16 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f16_f32_f16>(
      "reduce_add_linear_combination_f16_f32_f16"
  ));
}


void initialize_reduce_add_linear_combination_f16_f16_f16(Manifest &manifest) {

  using ElementWorkspace = hytlass::half_t;
  using ElementAccumulator = hytlass::half_t;
  using ElementOutput = hytlass::half_t;
  using ElementCompute = hytlass::half_t;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f16_f16_f16 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f16_f16_f16>(
      "reduce_add_linear_combination_f16_f16_f16"
  ));
}

void initialize_reduce_add_linear_combination_f32_f32_f16(Manifest &manifest) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = hytlass::half_t;
  using ElementCompute = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f32_f32_f16 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f32_f32_f16>(
      "reduce_add_linear_combination_f32_f32_f16"
  ));
}

void initialize_reduce_add_linear_combination_f32_f32_bf16(Manifest &manifest) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = hytlass::bfloat16_t;
  using ElementCompute = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f32_f32_bf16 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f32_f32_bf16>(
      "reduce_add_linear_combination_f32_f32_bf16"
  ));
}

void initialize_reduce_add_linear_combination_f32_f32_f32(Manifest &manifest) {

  using ElementWorkspace = float;
  using ElementAccumulator = float;
  using ElementOutput = float;
  using ElementCompute = float;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f32_f32_f32 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f32_f32_f32>(
      "reduce_add_linear_combination_f32_f32_f32"
  ));
}

void initialize_reduce_add_linear_combination_f64_f64_f64(Manifest &manifest) {

  using ElementWorkspace = double;
  using ElementAccumulator = double;
  using ElementOutput = double;
  using ElementCompute = double;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_f64_f64_f64 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_f64_f64_f64>(
      "reduce_add_linear_combination_f64_f64_f64"
  ));
}

void initialize_reduce_add_linear_combination_cf32_cf32_cf32(Manifest &manifest) {

  using ElementWorkspace = hytlass::complex<float>;
  using ElementAccumulator = hytlass::complex<float>;
  using ElementOutput = hytlass::complex<float>;
  using ElementCompute = hytlass::complex<float>;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_cf32_cf32_cf32 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_cf32_cf32_cf32>(
      "reduce_add_linear_combination_cf32_cf32_cf32"
  ));
}

void initialize_reduce_add_linear_combination_ks32_ks32_ks32(Manifest &manifest) {

  using ElementWorkspace = int32_t;
  using ElementAccumulator = int32_t;
  using ElementOutput = int32_t;
  using ElementCompute = int32_t;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_ks32_ks32_ks32 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_ks32_ks32_ks32>(
      "reduce_add_linear_combination_ks32_ks32_ks32"
  ));
}

void initialize_reduce_add_linear_combination_ks32_ks32_ks8(Manifest &manifest) {

  using ElementWorkspace = int32_t;
  using ElementAccumulator = int32_t;
  using ElementOutput = int8_t;
  using ElementCompute = int32_t;

  using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationClamp<
    ElementOutput,
    128 / hytlass::sizeof_bits<ElementWorkspace>::value,
    ElementAccumulator,
    ElementCompute
  >;

  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using Operation_reduce_add_linear_combination_ks32_ks32_ks8 = hytlass::reduction::device::ReduceSplitK<
    hytlass::reduction::kernel::ReduceSplitK<
      hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
      EpilogueOutputOp,
      ReductionOp
    >
  >;

  manifest.append(new ReductionOperation<
    Operation_reduce_add_linear_combination_ks32_ks32_ks8>(
      "reduce_add_linear_combination_ks32_ks32_ks8"
  ));
}

} 
}

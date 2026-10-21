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
    \brief Unit tests for epilogues
*/
#pragma once
#include "hip/hip_runtime.h"
#include <fstream>
#include <cfenv>

#include "../../common/hytlass_unit_test.h"

#include "hytlass/aligned_buffer.h"
#include "hytlass/half.h"
#include "hytlass/complex.h"
#include "hytlass/platform/platform.h"
#include "hytlass/epilogue/thread/linear_combination.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/host/tensor_fill.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace test {
namespace kernel {

template <typename Epilogue>
__launch_bounds__(Epilogue::WarpCount::kCount * WARP_SIZE_GPU)
__global__ void epilogue_threadblock(
  typename Epilogue::OutputTileIterator::Params params_D,
  typename Epilogue::OutputTileIterator::Element *ptr_D,
  typename Epilogue::OutputTileIterator::Params params_C,
  typename Epilogue::OutputTileIterator::Element *ptr_C,
  typename Epilogue::OutputOp::Params params_output_op,
  hytlass::MatrixCoord problem_size,
  hytlass::TensorRef<
    typename Epilogue::WarpMmaOperator::ElementC, 
    typename Epilogue::WarpMmaOperator::LayoutC> accumulator_ref,
    int epilogue_count = 1) {

  __shared__ typename Epilogue::SharedStorage shared_storage;

  int thread_idx = threadIdx.x;
  int warp_idx = threadIdx.x / WARP_SIZE_GPU;
  int lane_idx = threadIdx.x % WARP_SIZE_GPU;

  //
  // Construct the epilogue
  //

  // Tile iterator writing to output tile
  typename Epilogue::OutputTileIterator iterator_D(
    params_D,
    ptr_D,
    problem_size,
    thread_idx
  );

  // Tile iterator writing to output tile
  typename Epilogue::OutputTileIterator iterator_C(
    params_C,
    ptr_C,
    problem_size,
    thread_idx
  );

  // Epilogue operator
  Epilogue epilogue(
    shared_storage, 
    thread_idx, 
    warp_idx, 
    lane_idx);

  //
  // Initialize the accumulators
  //

  int warp_mn = warp_idx % (Epilogue::WarpCount::kM * Epilogue::WarpCount::kN);
  int warp_m = warp_mn % Epilogue::WarpCount::kM;
  int warp_n = warp_mn / Epilogue::WarpCount::kM;

  accumulator_ref.add_coord_offset({
    warp_m * Epilogue::WarpMmaOperator::Shape::kM, 
    warp_n * Epilogue::WarpMmaOperator::Shape::kN});

  typename Epilogue::WarpMmaOperator::IteratorC accumulator_iterator(accumulator_ref, lane_idx);
  
  typename Epilogue::AccumulatorTile accumulators;

  accumulators.clear();
  accumulator_iterator.load(accumulators);

#if 0
  // For debugging, enable this block of code to fill each accumulator element with its
  // source thread ID.
  HYTLASS_PRAGMA_UNROLL
  for (size_t i = 0; i < accumulators.size(); ++i) {
    typename Epilogue::WarpMmaOperator::ElementC x(threadIdx.x);
    accumulators[i] = x;
  }

  __syncthreads();

#endif

  //
  // Perform the epilogue operation
  //

  typename Epilogue::OutputOp output_op(params_output_op);

  // Place the epilogue in a loop
  for (int iter = 0; iter < epilogue_count; ++iter) {
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }
}

} // namespace kernel
} // namespace test


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Epilogue_
>
class EpilogueTestbed {
public:

  using Epilogue = Epilogue_;
  using ElementAccumulator = typename Epilogue::ElementAccumulator;
  using ElementCompute = typename Epilogue::OutputOp::ElementCompute;
  using ElementOutput = typename Epilogue::ElementOutput;
  using OutputOpParams = typename Epilogue::OutputOp::Params;

public:

  //
  // Data members
  //

  hytlass::MatrixCoord quantized_size;
  hytlass::HostTensor<ElementAccumulator, hytlass::layout::RowMajor> accumulator_tensor;
  hytlass::HostTensor<ElementOutput, hytlass::layout::RowMajor> source_tensor;
  hytlass::HostTensor<ElementOutput, hytlass::layout::RowMajor> output_tensor;

public:

  //
  // Methods
  //

  EpilogueTestbed(): 
    quantized_size(Epilogue::Shape::kM, Epilogue::Shape::kN),
    accumulator_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}),
    source_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}),
    output_tensor({Epilogue::Shape::kM, Epilogue::Shape::kN}) {

    //
    // Initialize problem space
    //

    uint64_t seed = 2019;

    hytlass::reference::host::TensorFillRandomUniform(
      accumulator_tensor.host_view(), 
      seed, 
      2,
      -2,
      0);

    hytlass::reference::host::TensorFillRandomUniform(
      source_tensor.host_view(),
      seed + 2018, 
      2,
      -2,
      0);
  }

  bool run_all() {
   
    double alpha_values[] = {1, 0, 2.25};
    double beta_values[] = {0, 1, -1.25};

    // Test runtime explodes if we tried to test every case exhaustively. This tests the full
    // output tile and several smaller sizes to stress predication.
    for (int m_idx = 0; m_idx < 3; ++m_idx) {
      for (int n_idx = 0; n_idx < 3; ++n_idx) {

        int m = quantized_size.row() - m_idx * 3;
        int n = quantized_size.column() - n_idx * Epilogue::kElementsPerAccess;

        for (double const &alpha : alpha_values) {
          for (double const &beta : beta_values) {

            bool passed = run({m, n}, {hytlass::from_real<ElementCompute>(alpha), hytlass::from_real<ElementCompute>(beta)});

            if (!passed) {
              return false;
            }
          }
        }
      }
    }

    return true;
  }

  /// Runs the test
  bool run(
    hytlass::MatrixCoord problem_size,
    OutputOpParams output_params) { 

    //
    // Initialize problem space
    //

    ElementOutput default_output = ElementOutput(-127);
    hytlass::reference::host::TensorFill(output_tensor.host_view(), default_output);

    accumulator_tensor.sync_device();
    output_tensor.sync_device();
    source_tensor.sync_device();

    //
    // Initialize epilogue parameters
    //

    typename Epilogue::OutputTileIterator::Params params_D(output_tensor.device_ref().layout());
    typename Epilogue::OutputTileIterator::Params params_C(source_tensor.device_ref().layout());

    //
    // Launch kernel
    //

    dim3 grid(1, 1);
    dim3 block(Epilogue::WarpCount::kCount * WARP_SIZE_GPU, 1);

    test::kernel::epilogue_threadblock<Epilogue><<< grid, block >>>(
      params_D,
      output_tensor.device_data(),
      params_C,
      source_tensor.device_data(),
      output_params,
      problem_size, 
      accumulator_tensor.device_view());

    hipError_t result = hipDeviceSynchronize();

    if (result != hipSuccess) {
      std::cerr << "Kernel error: " << hipGetErrorString(result) << std::endl;
      return false;
    }

    //
    // Verify results
    //
    output_tensor.sync_host();

    int errors = 0;
    int const kMaxErrors = 5;

    for (int r = 0; errors < kMaxErrors && r < quantized_size.row(); ++r) {
      for (int c = 0; errors < kMaxErrors && c < quantized_size.column(); ++c) {

        hytlass::MatrixCoord coord{r, c};
        ElementOutput got = output_tensor.at(coord);
        
        ElementOutput expected;
        if (coord.row() < problem_size.row() && coord.column() < problem_size.column()) {
          ElementCompute intermediate =
            output_params.alpha * ElementCompute(accumulator_tensor.at(coord)) + 
            output_params.beta * ElementCompute(source_tensor.at(coord));
          
          if ((hytlass::platform::is_same<ElementOutput, hytlass::int4b_t>::value
              || hytlass::platform::is_same<ElementOutput, hytlass::uint4b_t>::value
              || std::numeric_limits<ElementOutput>::is_integer)
              && !std::numeric_limits<ElementCompute>::is_integer) {
            std::fesetround(FE_TONEAREST);
            expected = ElementOutput(std::nearbyint(float(hytlass::real(intermediate))));
          } else {
            expected = ElementOutput(intermediate);
          }
        } else {
          expected = default_output;
        }

        if (expected != got) {

          using OutputIO = hytlass::ScalarIO<ElementOutput>;

          EXPECT_TRUE(false)
            << "-------\n"
            << "Error - output element (" << coord << ") - expected: " 
            << OutputIO(expected) 
            << ",  got: " << OutputIO(got)
            << ",  accum: " << (accumulator_tensor.at(coord))
            << ",  source: " << OutputIO(source_tensor.at(coord))
            << ",  alpha: " << (output_params.alpha)
            << ",  beta: " << (output_params.beta) << "\n";

          ++errors;
        }
      }
    }

    //
    // Report results on error
    //

    if (errors) {
      std::stringstream ss;
      ss 
        << "output_tensor_op_" << Epilogue::Shape::kM << "x" << Epilogue::Shape::kN << "_" 
        << Epilogue::WarpTileIterator::WarpShape::kM << "x" 
        << Epilogue::WarpTileIterator::WarpShape::kN 
        << "_slice_" << Epilogue::WarpCount::kK << ".csv"; 

      std::ofstream output_file(ss.str()); 
      output_file << output_tensor.host_view(); 
    }

    return !errors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

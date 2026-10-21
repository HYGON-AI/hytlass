/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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

// This example fuses gather before GEMM and scatter after GEMM into the same
// GEMM kernel.  Gather and scatter operation is controled by an index vector
// to select rows or columns from A, B, C or D matrices.
//
// Suppose, all matrices are column major.  The pseudo code of the fused kernel
// in this example is essentially
//
//    for (int i = 0; i < problem_size.m(); ++i) {
//      for (int j = 0; j < options.index_size; ++j) {
//        int b_c_d_col = tensor_indices.at({j, 0});
//
//        for (int k = 0; k < options.index_size; ++k) {
//            tensor_d_ref.at({i, b_c_d_col}) +=
//              alpha * tensor_a.at({i, k}) * tensor_b.at({k, b_c_d_col});
//        }
//      }
//
// Note that the index vector contains unique random integers with max to be N - 1
//
// The gather/scatter operation works best when we can still keep the biggest
// alignment. For example, when the matrix is row major, we select rows. When
// the matrix is column major, we select columns.
//
// Not all the combination of gather and scatter are legal. For example, if A is
// row major and C/D is column major, we cannot gather A and scatter C/D at the
// same time.
//
// Also, we don't check the index value is legal and index array point is valid
// for the sake of the performance.
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <numeric>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm_universal.h"
#include "hytlass/epilogue/thread/linear_combination.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/command_line.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/tensor_view_io.h"
#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  hytlass::Status status;
  hipError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    hytlass::Status status = hytlass::Status::kSuccess,
    hipError_t error = hipSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) 
  {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  hytlass::gemm::GemmCoord problem_size;
  int index_size;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({248, 1024, 1024}),
    index_size(240),
    reference_check(true),
    iterations(50) 
  {}

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    hytlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());
    cmd.get_cmd_line_argument("index_size", index_size);
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_gather_scatter_fusion example\n\n"
      << "  This example uses the HYTLASS Library to fuse gather/scatter into GEMM\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --index_size=<int>          size of N dimension index\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_gather_scatter_fusion/gather_scatter_fusion --m=1024 --n=512 --k=1024 \\\n"
      << "     --index_size=128\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds
    int64_t fmas = problem_size.m() * int64_t(index_size) * problem_size.k();
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = hytlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = hytlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices.
// Column Major for Matrix A, B and C.
//
using LayoutInputA = hytlass::layout::ColumnMajor;
using LayoutInputB = hytlass::layout::ColumnMajor;
using LayoutOutput = hytlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes GFX architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    hytlass::gemm::GemmShape<128, 256, 32>;
// This code section describes tile size a warp will compute
using ShapeMMAWarp = hytlass::gemm::GemmShape<64, 128, 32>;
// This code section describes the size of MMA op
using ShapeMMAOp = hytlass::gemm::GemmShape<16, 16, 16>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombination. This is approximately equal to
//
//    d_ij = alpha * sum_k(a_ik * b_kj) + c_ij
//
using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / hytlass::sizeof_bits<ElementOutput>::value, // <- this is the number of elements per
                                                      // vectorized memory access. For half
                                                      // precision, it's 8 elements. This becomes
                                                      // the vector width of math instructions in
                                                      // epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 1;

using Gemm = hytlass::gemm::device::GemmUniversal<ElementInputA,
                                                  LayoutInputA,
                                                  ElementInputB,
                                                  LayoutInputB,
                                                  ElementOutput,
                                                  LayoutOutput,
                                                  ElementAccumulator,
                                                  MMAOp,
                                                  SmArch,
                                                  ShapeMMAThreadBlock,
                                                  ShapeMMAWarp,
                                                  ShapeMMAOp,
                                                  EpilogueOp,
                                                  SwizzleThreadBlock,
                                                  NumStages,
                                                  8, /*alignmentA*/
                                                  8, /*alignmentB*/
                                                  hytlass::arch::OpMultiplyAdd,
                                                  hytlass::ComplexTransform::kNone,
                                                  hytlass::ComplexTransform::kNone,
                                                  false, /*GatherA*/
                                                  true,  /*GatherB*/
                                                  true,  /*ScatterD*/
                                                  hytlass::layout::NoPermute,
                                                  hytlass::layout::NoPermute,
                                                  hytlass::layout::NoPermute,
                                                  true>;

int run(Options &options) {

  // ================================================================================
  // Initialization setup

  // Create a tuple of problem size for matrix multiplication
  hytlass::gemm::GemmCoord problem_size = options.problem_size;

  // Create a tuple of problem size for matrix multiplication
  hytlass::gemm::GemmCoord problem_size_real(problem_size.m(),
                                             options.index_size,
                                             problem_size.k());

  // Initialize tensors using HYTLASS helper functions
  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_scattered(
      problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                          // HYTLASS kernel

  // Fill input and output matrices on host using HYTLASS helper functions
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(3),
      ElementInputA(-3),
      0); // <- Fill matrix A on host with uniform-distribution random data

  hytlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputA(3),
      ElementInputA(-3),
      0); // <- Fill matrix B on host with uniform-distribution random data

  hytlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(7),
      ElementOutput(-8),
      0); // <- Fill matrix C on host with uniform-distribution random data

  hytlass::reference::host::TensorFill(
      tensor_d_scattered.host_view()); // <- fill matrix D on host with zeros

  hytlass::HostTensor<int, LayoutOutput> tensor_indices(
      {options.index_size, 1}); // <- Create scatter indices with dimensions val_len x 1

  // <- Fill tensor_b_indices on host with unique random integers
  std::vector<int> to_fill(problem_size.n());           // vector with ints.
  std::iota(std::begin(to_fill), std::end(to_fill), 0); // Fill with 0, 1, ...., problem_size.n()
  { // std::random_shuffle was deprecated in C++14 and removed in C++17
    std::random_device make_seed;
    std::mt19937 source_of_randomness(make_seed());
    std::shuffle(to_fill.begin(), to_fill.end(), source_of_randomness);
  }
  memcpy(tensor_indices.host_data(), to_fill.data(), options.index_size * sizeof(int));

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_indices.sync_device();
  tensor_c.sync_device();
  tensor_d_scattered.sync_device();

  // Initialize alpha/beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated HYTLASS kernel
  typename Gemm::Arguments arguments{
      hytlass::gemm::GemmUniversalMode::kGemm,
      problem_size_real,                // <- problem size of matrix multiplication
      split_k_slices,                   // <- k-dimension split factor
      {alpha, beta},                    // <- alpha, beta
      tensor_a.device_data(),           // <- reference to matrix A on device
      tensor_b.device_data(),           // <- reference to matrix B on device
      tensor_c.device_data(),           // <- reference to matrix C on device
      tensor_d_scattered.device_data(), // <- reference to matrix D on device
      tensor_a.layout().capacity(problem_size.mk()),
      tensor_b.layout().capacity(hytlass::make_Coord(options.index_size, problem_size.k())),
      tensor_c.layout().capacity(problem_size.mn()),
      tensor_d_scattered.layout().capacity(problem_size.mn()),
      tensor_a.layout().stride(),
      tensor_b.layout().stride(),
      tensor_c.layout().stride(),
      tensor_d_scattered.layout().stride(),
      nullptr,                       // <- pointer to index vector to gather A on device
      tensor_indices.device_data(),  // <- pointer to index vector to gather B on device
      tensor_indices.device_data()}; // <- pointer to index vector to scatter D on device

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate HYTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not
  hytlass::Status status = gemm_op.can_implement(arguments);
  HYTLASS_CHECK(status);

  // Initialize HYTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  HYTLASS_CHECK(status);

  // CPU reference calculation
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d_ref(problem_size.mn());
  hytlass::reference::host::TensorFill(
      tensor_d_ref.host_view()); // <- Fill matrix D on host with zeros

  status = gemm_op();
  (void)hipDeviceSynchronize();
  HYTLASS_CHECK(status);

  if (options.reference_check) {
    for (int i = 0; i < problem_size.m(); ++i) {
      for (int j = 0; j < options.index_size; ++j) {
        int b_c_d_col = tensor_indices.at({j, 0});

        for (int k = 0; k < problem_size.k(); ++k) {
            tensor_d_ref.at({i, b_c_d_col}) +=
              alpha * tensor_a.at({i, k}) * tensor_b.at({k, b_c_d_col});
        }

        tensor_d_ref.at({i, b_c_d_col}) += (beta * tensor_c.at({i, b_c_d_col}));
      }
    }

    // Copy output data from HYTLASS and reference kernel to host for comparison
    tensor_d_scattered.sync_host();
  
    bool passed = hytlass::reference::host::TensorEquals(
                    tensor_d_scattered.host_view(),
                    tensor_d_ref.host_view());

    if (!passed) {
      std::cout << "Failed!\n";

      std::stringstream fname;
      fname << "error_gather_GEMM_scatter_fusion.txt";
      std::cerr << "Dumping results in " << fname.str() << "\n";

      std::ofstream file(fname.str());

      file 
        << "A =\n" << tensor_a.host_view()
        << "\nB =\n" << tensor_b.host_view()
        << "\nindices =\n" << tensor_indices.host_view()
        << "\nC =\n" << tensor_c.host_view()
        << "\n\nReference =\n" << tensor_d_ref.host_view()
        << "\nComputed =\n" << tensor_d_scattered.host_view();
      return -1;
    } else {
      std::cout << "Passed!\n";
    }
  }

  // Result structure
  Result result;

  //
  // Construct events
  //

  hipEvent_t events[2];

  for (auto & event : events) {
    result.error = hipEventCreate(&event);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventCreate() failed: " << hipGetErrorString(result.error) << std::endl;
      return -1;
    }
  }
  
  // warm up
  for (int iter = 0; iter < 10; ++iter) {
    status = gemm_op();
    HYTLASS_CHECK(status);
  }

  // Record an event at the start of a series of GEMMs
  result.error = hipEventRecord(events[0]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

  //
  // Run profiling loop
  //

  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized HYTLASS kernel
    status = gemm_op();
    HYTLASS_CHECK(status);
  }

  //
  // Stop profiling loop
  //

  // Record an event when the GEMMs are complete
  result.error = hipEventRecord(events[1]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Wait for work on the device to complete.
  result.error = hipEventSynchronize(events[1]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventSynchronize() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = hipEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventElapsed() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Compute average runtime and GFLOPs.
  result.runtime_ms = double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)hipEventDestroy(event);
  }

  std::cout << "Runtime: " << result.runtime_ms << " ms\n";
  std::cout << "GFLOPs: " << result.gflops << "\n";

  return 0;
}

int main(int argc, const char ** argv) {
  bool notSupported = false;

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << "\n";
    return 0;
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << "\n";
    return -1;
  }

  return run(options);
}


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

/**
*/

#include <iostream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/epilogue/thread/linear_combination_relu.h"
#include "hytlass/util/command_line.h"
#include "hytlass/util/host_tensor.h"
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

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  hytlass::gemm::GemmCoord problem_size;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  int split_k_slices;

  Options():
    help(false),
    problem_size({8192, 8192, 2048}),
    reference_check(true),
    iterations(10),
    split_k_slices(1),
    alpha(1),
    beta() 
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

    cmd.get_cmd_line_argument("alpha", alpha);

    cmd.get_cmd_line_argument("split_k_slices", split_k_slices);

    cmd.get_cmd_line_argument("iterations", iterations);
    

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "08_tensorop_fused_gemm example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --split_k_slices=<int>      Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/08_tensorop_fused_gemm/gfx928_tensorop_gemm_bias_relu --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 \n\n";

    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = hytlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = hytlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = hytlass::half_t;                        // <- data type of elements in output matrix D

// Note that if the output is column major, the bias has to be per row. i.e. every row has different bias.
// If the output is row major, the bias has to be per column, i.e. every column has different bias.
// Below list some other notices:
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the
//      row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

using LayoutInputA = hytlass::layout::ColumnMajor;
using LayoutInputB = hytlass::layout::ColumnMajor;
using LayoutOutput = hytlass::layout::ColumnMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes GFX architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    hytlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = hytlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// This code section describes the size of MMA op
using ShapeMMAOp = hytlass::gemm::GemmShape<16, 16, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

constexpr int kAlignmentA = 128 / hytlass::sizeof_bits<ElementInputA>::value;
constexpr int kAlignmentB = 128 / hytlass::sizeof_bits<ElementInputB>::value;

constexpr int kAlignmentC = 128 / hytlass::sizeof_bits<ElementOutput>::value;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombinationRelu. This is approximately equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOp = hytlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                        // <- data type of output matrix
    kAlignmentC,     // <- this is the number of elements per
                                                          // vectorized memory access. For half
                                                          // precision, it's 8 elements. This becomes
                                                          // the vector width of math instructions in
                                                          // epilogue too
    ElementAccumulator,                                   // <- data type of accumulator
    ElementComputeEpilogue,                          // <- data type for alpha in linear combination function
    hytlass::epilogue::thread::ScaleType::NoBetaScaling>; // <- alpha x C + bias

// Number of pipelines you want to use
constexpr int NumStages = 1;

using Gemm = hytlass::gemm::device::Gemm<ElementInputA,
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
                                         kAlignmentA,
                                         kAlignmentB>;

int run(Options &options) {

  hytlass::gemm::GemmCoord problem_size = options.problem_size;

  // Initialize tensors using HYTLASS helper functions
  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N

  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias(
      {problem_size.m(), 1});  // <- Create matrix C with dimensions M x 1

  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // HYTLASS kernel
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using HYTLASS helper functions
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_c_bias.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  hytlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  hytlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c_bias.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated HYTLASS kernel
  typename Gemm::Arguments arguments{
    problem_size,                       // <- problem size of matrix multiplication
    tensor_a.device_ref(),              // <- reference to matrix A on device
    tensor_b.device_ref(),              // <- reference to matrix B on device

    {tensor_c_bias.device_data(), 0},   // <- the C matrix is treated as the bias vector. We can enable the GEMM
                                        //    to project away the N dimension by setting the stride to zero.

    tensor_d.device_ref(),              // <- reference to matrix D on device
    {alpha},                              // <- alpha
    split_k_slices};                    // <- k-dimension split factor

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

  // Launch initialized HYTLASS kernel
  status = gemm_op();
  HYTLASS_CHECK(status);

  Result result;

  hipEvent_t events[2];

  for (auto & event : events) {
    result.error = hipEventCreate(&event);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventCreate() failed: " << hipGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  // Record an event at the start of a series of GEMMs
  result.error = hipEventRecord(events[0]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

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

  // Cleanup
  for (auto event : events) {
    (void)hipEventDestroy(event);
  }

  //
  // Create instantiation for device reference gemm kernel
  //
  if (options.reference_check) {

    hytlass::reference::device::Gemm<ElementInputA,
                                    LayoutInputA,
                                    ElementInputB,
                                    LayoutInputB,
                                    ElementOutput,
                                    LayoutOutput,
                                    ElementComputeEpilogue,
                                    ElementComputeEpilogue>
        gemm_device_reference;

    // Launch device reference to compute strictly the product A * B
    gemm_device_reference(
      problem_size,
      alpha,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      0,
      tensor_ref_d.device_ref());

    // Wait for kernels to finish
    (void)hipDeviceSynchronize();

    // Copy output data from HYTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Compute bias + relu in host code
    for (int i = 0; i < problem_size.m(); ++i) {
      for (int j = 0; j < problem_size.n(); ++j) {
        tensor_ref_d.at({i, j}) = std::max(
          ElementOutput(0), 
          ElementOutput(tensor_ref_d.at({i, j}) + tensor_c_bias.at({i, 0}))
        );
      }
    }

    ElementOutput eps(0.05);

    const ElementOutput non_zero_floor(1e-6f);
    result.passed = hytlass::reference::host::TensorRelativelyEquals(tensor_ref_d.host_view(), 
      tensor_d.host_view(), eps, non_zero_floor);
  }

  if (result.passed) {

    std::cout << "verify success " << std::endl;
  }else{
    std::cout << "verify failed " << std::endl;
  }
  return (result.passed ? 0  : -1);
}

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}

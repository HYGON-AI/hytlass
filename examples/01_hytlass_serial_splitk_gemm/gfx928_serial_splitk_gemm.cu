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
This example shows how to use serial split-k version of matrix multiplication using functions and
data structures provided by HYTLASS.

    Examples:

      # Runs a serial split-K GEMM with the given problem size (use split_k_slices to set number of K slices)
      $ ./gfx928_serial_splitk_gemm --m=5120 --n=5120 --k=8192 --alpha=1 --beta=0 --split_k_slices=2 --iterations=10 
*/

#include <fstream>
#include <iostream>

#include "hip/hip_runtime.h"

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/gemm/device/gemm_universal.h"

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

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    hytlass::Status status = hytlass::Status::kSuccess,
    hipError_t error = hipSuccess)
  :
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
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split_k_slices", split_k_slices);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {
    out << "01_hytlass_serial_splitk_gemm\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split_k_slices=<int>      Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/01_hytlass_serial_splitk_gemm/gfx928_serial_splitk_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 --split_k_slices=2 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    int64_t fmas = problem_size.product();
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                  // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator; // <- data type of epilogue operations
using ElementInputA = hytlass::half_t;             // <- data type of elements in input matrix A
using ElementInputB = hytlass::half_t;             // <- data type of elements in input matrix B
using ElementOutput = hytlass::half_t;             // <- data type of elements in output matrix D

constexpr int kAlignmentA = 128 / hytlass::sizeof_bits<ElementInputA>::value;
constexpr int kAlignmentB = 128 / hytlass::sizeof_bits<ElementInputB>::value;

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = hytlass::layout::ColumnMajor;
using LayoutInputB = hytlass::layout::RowMajor;
using LayoutOutput = hytlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes GFX architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    hytlass::gemm::GemmShape<128, 128, 32>; // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = hytlass::gemm::GemmShape<64, 64, 32>; // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = hytlass::gemm::GemmShape<16, 16, 16>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
static bool const kSplitKSerial = true;

// This code section describes the epilogue part of the kernel
using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,                                    // <- data type of output matrix
    128 / hytlass::sizeof_bits<ElementOutput>::value, // <- the number of elements per vectorized
                                                      // memory access. For a byte, it's 16
                                                      // elements. This becomes the vector width of
                                                      // math instructions in the epilogue too
    ElementAccumulator,                               // <- data type of accumulator
    ElementComputeEpilogue>;                          // <- data type for alpha/beta in linear combination function

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
                                         kAlignmentB,
                                         kSplitKSerial>;

int run(Options &options) {
  // Create a tuple of problem size for matrix multiplication
  hytlass::gemm::GemmCoord problem_size = options.problem_size;

  // Initialize tensors using HYTLASS helper functions
  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk()); // <- Create matrix A with dimensions M x K
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn()); // <- Create matrix B with dimensions K x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn()); // <- Create matrix C with dimensions M x N
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                          // HYTLASS kernel
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn()); // <- Create matrix D with dimensions M x N used to store output from
                          // reference kernel

  // Fill input and output matrices on host using HYTLASS helper functions
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0); // <- Fill matrix A on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0); // <- Fill matrix B on host with uniform-distribution random data
  // hytlass::reference::host::TensorFillSequential(
  //     tensor_a.host_view());  // <- Fill matrix A on host with uniform-distribution random data
  // hytlass::reference::host::TensorFillSequential(
  //     tensor_b.host_view());  // <- Fill matrix B on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);                                                     // <- Fill matrix C on host with uniform-distribution random data
  hytlass::reference::host::TensorFill(tensor_d.host_view()); // <- fill matrix D on host with zeros
  hytlass::reference::host::TensorFill(
      tensor_ref_d.host_view()); // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = ElementComputeEpilogue(options.beta);

  // Split K dimension into 1 partitions
  int split_k_slices = options.split_k_slices;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated HYTLASS kernel

  typename Gemm::Arguments arguments{problem_size,          // <- problem size of matrix multiplication
                                     tensor_a.device_ref(), // <- reference to matrix A on device
                                     tensor_b.device_ref(), // <- reference to matrix B on device
                                     tensor_c.device_ref(), // <- reference to matrix C on device
                                     tensor_d.device_ref(), // <- reference to matrix D on device
                                     {alpha, beta},         // <- tuple of alpha and beta
                                     split_k_slices};       // <- k-dimension split factor

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

  // Result structure
  Result result;

  for (int i=0; i<10; i++) {
    status = gemm_op();
  }

  HYTLASS_CHECK(status);

  //
  // Construct events
  //

  hipEvent_t events[2];

  for (auto &event : events) {
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

  if (options.reference_check) {

    // Create instantiation for device reference gemm kernel
    hytlass::reference::device::Gemm<ElementInputA,
                                     LayoutInputA,
                                     ElementInputB,
                                     LayoutInputB,
                                     ElementOutput,
                                     LayoutOutput,
                                     ElementComputeEpilogue,
                                     ElementComputeEpilogue>
        gemm_device;

    // Launch device reference gemm kernel
    gemm_device(problem_size,
                alpha,
                tensor_a.device_ref(),
                tensor_b.device_ref(),
                beta,
                tensor_c.device_ref(),
                tensor_ref_d.device_ref());

    // Wait for kernels to finish
    (void)hipDeviceSynchronize();

    // Copy output data from HYTLASS and reference kernel to host for comparison
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    // Check if output from HYTLASS kernel and reference kernel are equal or not

    ElementOutput eps(0.05);

    const ElementOutput non_zero_floor(1e-6f);
    result.passed = hytlass::reference::host::TensorRelativelyEquals(
        tensor_ref_d.host_view(), tensor_d.host_view(), eps, non_zero_floor);
  }

  if (!result.passed) {
    std::stringstream fname;
    fname << "error_Gemm_device_" << problem_size.m() << "x" << problem_size.n() << "x"
          << problem_size.k() << "_" << ShapeMMAThreadBlock{}.kM << "_" << ShapeMMAThreadBlock{}.kN
          << "_" << ShapeMMAThreadBlock{}.kK << ".csv";

    std::ofstream file(fname.str());
    file << "problem: " << ' ' << problem_size.m() << "x" << problem_size.n() << "x"
         << problem_size.k() << ", alpha: " << float(alpha) << ", beta: " << float(beta) << "\n\n";

    file << "A =\n"
         << tensor_a.host_view() << "\nB =\n"
         << tensor_b.host_view() << "\nC =\n"
         << tensor_c.host_view() << "\n\nReference =\n"
         << tensor_ref_d.host_view() << "\n\nComputed =\n"
         << tensor_d.host_view();
  }

  std::cout << (result.passed ? "Passed" : "Failed") << std::endl;

  if (result.passed) {
    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << "GFLOPs: " << result.gflops << std::endl;
  }

  return (result.passed ? 0 : -1);
}

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d tensor op Matrix Multiply\n",
         options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}

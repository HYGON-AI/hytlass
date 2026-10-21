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
This example shows how to run group convolution kernels using functions and data structures
provided by HYTLASS using tensor cores.

There are 2 group conv mode:
  1. hytlass::conv::GroupMode::kSingleGroup
      This mode is for large K problem size: k_per_group (K/groups) equals or larger than
      threadblock_tile_N. One or multiple threadblocks calculate data of one group.
  2. hytlass::conv::GroupMode::kMultipleGroup
      This mode is for small K problem size: k_per_group (K/groups) is smaller than threadblock_tile_N.
      One threadblock will calculate data from more than one group.

Function profile_convolution_selecter() shows how to choose kernel with different group mode according
to problem size and threadblock_tile size.
*/

#include <iostream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/conv/kernel/default_conv2d_group_fprop.h"
#include "hytlass/conv/device/implicit_gemm_convolution.h"

#include "hytlass/util/command_line.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/convolution.h"
#include "hytlass/util/reference/device/convolution.h"
#include "hytlass/util/tensor_view_io.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements 
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = hytlass::half_t;             // Data type of elements in input tensor
using ElementInputB = hytlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = hytlass::layout::TensorNHWC;
using LayoutInputB = hytlass::layout::TensorNHWC;
using LayoutOutput = hytlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes hip Gfx architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;   // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;          // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 1;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / hytlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// Analytic kernel and operation for single group problem size
using AnalyticSingleGroupKernel = typename hytlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::GroupMode::kSingleGroup,
    hytlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;
using AnalyticSingleGroupOperation = hytlass::conv::device::ImplicitGemmConvolution<AnalyticSingleGroupKernel>;

// Analytic kernel and operation for multiple group problem size
using AnalyticMultipleGroupKernel = typename hytlass::conv::kernel::DefaultConv2dGroupFprop<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    NumStages,
    hytlass::arch::OpMultiplyAdd,
    hytlass::conv::GroupMode::kMultipleGroup,
    hytlass::conv::IteratorAlgorithm::kAnalytic>::Kernel;
using AnalyticMultipleGroupOperation = hytlass::conv::device::ImplicitGemmConvolution<AnalyticMultipleGroupKernel>;

// Optimized kernel and operation for single group problem size
using OptimizedSingleGroupKernel = typename hytlass::conv::kernel::DefaultConv2dGroupFprop<
  ElementInputA, LayoutInputA,
  ElementInputB, LayoutInputB,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  hytlass::arch::OpMultiplyAdd,
  hytlass::conv::GroupMode::kSingleGroup,
  hytlass::conv::IteratorAlgorithm::kOptimized
>::Kernel;
using OptimizedSingleGroupOperation = hytlass::conv::device::ImplicitGemmConvolution<OptimizedSingleGroupKernel>;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  hytlass::Tensor4DCoord input_size;
  hytlass::Tensor4DCoord filter_size;
  hytlass::Tensor4DCoord padding;
  hytlass::MatrixCoord conv_stride;
  hytlass::MatrixCoord dilation;
  int groups;
  bool reference_check;
  bool measure_performance;
  int iterations;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  bool optimized;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    groups(1),
    reference_check(true),
    measure_performance(true),
    iterations(20),
    alpha(1),
    beta(0),
    optimized(false) { }

  // Verify the problem size is compatible with the HYTLASS Convolution implementation.
  bool valid() {

    //
    // HYTLASS attempts to load 128b vectors of hytlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 128 / hytlass::sizeof_bits<ElementInputA>::value;

    if ((input_size.c() % kAlignment) ||
      (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) ||
      (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
    hytlass::Tensor4DCoord input_size,
    hytlass::Tensor4DCoord filter_size) {

    this->input_size = input_size;
    this->filter_size = filter_size;

    padding.n() = filter_size.h() / 2;
    padding.h() = filter_size.h() / 2;
    padding.w() = filter_size.w() / 2;
    padding.c() = filter_size.w() / 2;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    hytlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("ref-check")) {
      reference_check = true;
    }

    if (cmd.check_cmd_line_flag("perf-check")) {
      measure_performance = true;
    }

    if (cmd.check_cmd_line_flag("optimized")) {
      optimized = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());

    cmd.get_cmd_line_argument("g", groups);
    filter_size.c() = input_size.c() / groups;

    cmd.get_cmd_line_argument("u", conv_stride.row());
    cmd.get_cmd_line_argument("v", conv_stride.column());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    if (filter_size.h() == 3 && filter_size.w() == 3) {
      padding = {1, 1, 1, 1};
    }
    else {
      filter_size.h() = 1;
      filter_size.w() = 1;
      padding = {0, 0, 0, 0};
    }
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "11_hytlass_tensorop_group_conv example\n\n"
      << "  This example uses Ampere's Tensor Core operators on F16 data types to compute\n"
      << "  forward grouped convolution on tensors of layout NHWC.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --g=<int>            Conv groups G\n\n"
      << "  --u=<int>            Conv stride_h\n\n"
      << "  --v=<int>            Conv stride_w\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --ref-check          If set (true), reference check is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --optimized          If set (true), use optimized kernel, otherwise use analytic kernel.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/11_hytlass_tensorop_group_conv/gfx928_tensorop_group_conv2dfprop  --n=4 --h=16 --w=16 --c=256 --k=128 --r=3 --s=3 --g=8 --ref-check\n\n"
      << "$ ./examples/11_hytlass_tensorop_group_conv/gfx928_tensorop_group_conv2dfprop  --n=4 --h=16 --w=16 --c=256 --k=128 --r=3 --s=3 --g=2 --ref-check\n\n"
      << "$ ./examples/11_hytlass_tensorop_group_conv/gfx928_tensorop_group_conv2dfprop  --n=4 --h=16 --w=16 --c=256 --k=128 --r=3 --s=3 --g=2 --ref-check --optimized\n\n";

    return out;
  }
  
  /// Computes the output tensor size (NPQK)
  hytlass::Tensor4DCoord output_size() const {
    return hytlass::Tensor4DCoord(
      input_size.n(),
      (input_size.h() + padding.n() + padding.h() - filter_size.h()) / conv_stride.row() + 1,
      (input_size.w() + padding.w() + padding.c() - filter_size.w()) / conv_stride.column() + 1,
      filter_size.n());
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of multiply-adds = NPQK * CRS
    int64_t fmas = output_size().product() * int64_t(filter_size.h() * filter_size.w() * filter_size.c());
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Result {
  double runtime_ms;
  double gflops;
  hytlass::Status status;
  hytlass::Status reference_check;
  hipError_t error;

  Result(): 
    runtime_ms(0), 
    gflops(0),
    status(hytlass::Status::kSuccess),
    reference_check(hytlass::Status::kInvalid),
    error(hipSuccess) { }

  static std::ostream & print_header(std::ostream &out, Options const &options) {

    if (!options.tag.empty()) {
      out << "Name,";
    }

    out << "Layer,N,H,W,C,K,R,S,G,Runtime,GFLOPs";

    return out;
  }

  std::ostream & print(std::ostream &out, int idx, Options const &options) {

    if (!options.tag.empty()) {
      out << options.tag << ",";
    }

    out
      << "conv_" << idx << ","
      << options.input_size.n() << ","
      << options.input_size.h() << ","
      << options.input_size.w() << ","
      << options.input_size.c() << ","
      << options.filter_size.n() << ","
      << options.filter_size.h() << ","
      << options.filter_size.w() << ","
      << options.groups << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
template <typename Conv2dOperation>
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the HYTLASS Utilities.
  //

  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.output_size());
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.output_size());
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(7),
      ElementInputA(-8),
      0);

  // Fill tensor B on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(7),
      ElementInputB(-8),
      0);

  // Fill tensor C on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(7),
      ElementOutput(-8),
      0);

  // Fill tensor D on host with zeros
  hytlass::reference::host::TensorFill(
      tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  hytlass::reference::host::TensorFill(
      tensor_ref_d.host_view());

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  //
  // Define arguments for HYTLASS Convolution
  //

  hytlass::conv::Mode mode = hytlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Construct Conv2dProblemSize with user defined output size
  hytlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices,
      options.groups
  );

  // Construct Conv2dOperation::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename Conv2dOperation::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    tensor_c.device_ref(),
    tensor_d.device_ref(),
    {options.alpha, options.beta},
  };

  //
  // Initialize HYTLASS Convolution
  //

  Conv2dOperation implicit_gemm_op;

  size_t workspace_size = implicit_gemm_op.get_workspace_size(arguments);

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm_op.can_implement(arguments);
  HYTLASS_CHECK(result.status);

  result.status = implicit_gemm_op.initialize(arguments, workspace.get());
  HYTLASS_CHECK(result.status);

  //
  // Launch initialized HYTLASS kernel
  //
  result.status = implicit_gemm_op();

  HYTLASS_CHECK(result.status);

  //
  // Optional reference check
  //

  if (options.reference_check) {
    std::cout << "Verification on device...\n";

    // Compute with reference implementation
    hytlass::reference::device::Conv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementComputeEpilogue,
      ElementAccumulator,
      hytlass::NumericConverter<ElementOutput, ElementComputeEpilogue>
    >(
      problem_size,
      tensor_a.device_ref(),
      tensor_b.device_ref(),
      tensor_c.device_ref(),
      tensor_ref_d.device_ref(),
      options.alpha,
      options.beta
    );

    tensor_ref_d.sync_host();

    // Check if output from HYTLASS kernel and reference kernel are equal or not
    tensor_d.sync_host();

    bool passed = hytlass::reference::host::TensorEquals(
      tensor_d.host_view(),
      tensor_ref_d.host_view());

    if (!passed) {
      result.reference_check = hytlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    } else {
      result.reference_check = hytlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  } else {
    result.reference_check = hytlass::Status::kInvalid;
  }

  //
  // Performance measurement
  //

  if (options.measure_performance) {

    hipEvent_t events[2];
    
    for (auto & event : events) {
      result.error = hipEventCreate(&event);
      if (result.error != hipSuccess) {
        std::cerr << "hipEventCreate() failed: " << hipGetErrorString(result.error) << std::endl;
        return result;
      }
    }

    // Record an event at the start of a series of convolution operations.
    result.error = hipEventRecord(events[0]);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
      return result;
    }

    // Launch a sequence of implicit GEMM operations on the device
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      result.status = implicit_gemm_op();
      HYTLASS_CHECK(result.status);
    }

    // Record an event when the convolutions have been launched.
    result.error = hipEventRecord(events[1]);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
      return result;
    }

    // Wait for work on the device to complete.
    result.error = hipEventSynchronize(events[1]);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventSynchronize() failed: " << hipGetErrorString(result.error) << std::endl;
      return result;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    result.error = hipEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (result.error != hipSuccess) {
      std::cerr << "hipEventElapsed() failed: " << hipGetErrorString(result.error) << std::endl;
      return result;
    }

    // Print average runtime and GFLOPs.
    result.runtime_ms = double(runtime_ms) / double(options.iterations);
    result.gflops = options.gflops(result.runtime_ms / 1000.0);

    // Cleanup
    for (auto event : events) {
      (void)hipEventDestroy(event);
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

Result profile_convolution_selecter(Options const &options) {
  int k_per_group = options.filter_size.n() / options.groups;

  // In group conv, if k_per_group < threadblock_N, one Threadblock will calculate multiple groups
  if (k_per_group < ThreadblockShape::kN) { // MultipleGroup mode
    if (options.optimized) {
      std::cerr << "Invalid problem: optimized group conv kernel doesn't support MultipleGroup (one CTA calculate multiple groups) mode" << std::endl;
      exit(-1);
    } else {
      std::cout << "Select AnalyticMultipleGroupOperation\n";
      return profile_convolution<AnalyticMultipleGroupOperation>(options);
    }
  } else { // SingleGroup mode
    if (options.optimized) {
      std::cout << "Select OptimizedSingleGroupOperation\n";
      return profile_convolution<OptimizedSingleGroupOperation>(options);
    } else {
      std::cout << "Select AnalyticSingleGroupOperation\n";
      return profile_convolution<AnalyticSingleGroupOperation>(options);
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  Options options;
  
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  // Execute one problem size
  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  Result result = profile_convolution_selecter(options);

  Result::print_header(std::cout, options) << std::endl;
  result.print(std::cout, 1, options) << std::endl;

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

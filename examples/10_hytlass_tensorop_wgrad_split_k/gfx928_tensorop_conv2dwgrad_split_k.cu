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

/*
This example shows how to compute conv2d gradient with respect to weight (wgrad). In wgrad, the K dimension of
impligit GEMM, corresponding to the sequential reduction loop, is very large (N * P * Q). Split-k with parallel
reduction is highly effective for such cases. Given split_k_slices parameter, it partitions the K loop into
split_k_slices chunks and computes partial reductions in parallel across different blocks. After that,
a parallel reduction kernel is launched to accumulate partial reductions.
In practice, wgrad requires fp32 accumulation to avoid overflow. When the input is fp16, some care is needed
to correctly instantiate the GEMM template.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/conv/kernel/default_conv2d_wgrad.h"
#include "hytlass/conv/device/implicit_gemm_convolution.h"

#include "hytlass/util/command_line.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/device/convolution.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/reduction/device/reduce_split_k.h"
#include "hytlass/reduction/thread/reduction_operators.h"

#include "helper.h"

// The code section below describes datatype for input, output tensors and computation between
// elements
// In Wgrad, fp32 accumulation is necessary in practice.
using ElementAccumulator = float;                  // Data type of accumulator
using ElementComputeEpilogue = float;              // Data type of epilogue computation (alpha, beta)
using ElementInputA = hytlass::half_t;             // Data type of elements in input tensor
using ElementInputB = hytlass::half_t;             // Data type of elements in input tensor
using ElementOutput = hytlass::half_t;                       // Data type of elements in output tensor
using ElementC = ElementOutput;
using ElementCompute = ElementComputeEpilogue;
using LayoutInputA = hytlass::layout::TensorNHWC;
using LayoutInputB = hytlass::layout::TensorNHWC;
using LayoutOutput = hytlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes hip Gfx architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>; // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = hytlass::gemm::GemmShape<64, 64, 32>;          // Warp tile shape

// This code section describes the size of MMA op
using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Number of pipelines you want to use
constexpr int NumStages = 2;

// This code section describe iterator algorithm selected is Analytic or Optimized
static hytlass::conv::IteratorAlgorithm const IteratorAlgorithm = hytlass::conv::IteratorAlgorithm::kOptimized;

// We need two epilogue functors - one for GEMM and another for the final reduction.
// The epilogue for GEMM is not used, but needed to instantiate the HYTLASS kernel template.
// Note that, when the input is fp16 and accumulation is fp32, the output of GEMM needs to be fp32,
// the final reduction is done in fp32, and the reduction epilogue converts fp32 outputs to fp16.
// Therefore, the output type of the GEMM epilogue is ElementCompute, not ElementOutput.

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOpGEMM = hytlass::epilogue::thread::LinearCombination<
    ElementCompute,                                     // Data type of output matrix.
    128 / hytlass::sizeof_bits<ElementCompute>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination

// The epilogue functor for reduction. This is the one that is actually used.
using EpilogueOpReduction = hytlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Data type of output matrix.
    128 / hytlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
    // memory access. This becomes the vector width of
    // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue>;                           // Data type for alpha/beta in lin

using Conv2dWgradKernel = typename hytlass::conv::kernel::DefaultConv2dWgrad<
    ElementInputA, LayoutInputA,
    ElementInputB, LayoutInputB,
    ElementAccumulator, LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOpGEMM,
    SwizzleThreadBlock,
    NumStages,
    hytlass::arch::OpMultiplyAdd,
    IteratorAlgorithm
    >::Kernel;

using ImplicitGemm = hytlass::conv::device::ImplicitGemmConvolution<Conv2dWgradKernel>;

using EpilogueOutputOp = EpilogueOpReduction;

/// Reduction kernel
using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator,
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
   >;

using ReductionKernel = hytlass::reduction::kernel::ReduceSplitK<
    hytlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
   >;

using ReductionDevice = hytlass::reduction::device::ReduceSplitK<ReductionKernel>;
using ReductionStrideIndex = typename ReductionDevice::StrideIndex;

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  hytlass::Tensor4DCoord input_size;
  hytlass::Tensor4DCoord filter_size;
  hytlass::Tensor4DCoord padding;
  hytlass::MatrixCoord conv_stride;
  hytlass::MatrixCoord dilation;
  bool reference_check;
  bool measure_performance;
  int iterations;
  bool save_workspace;
  ElementComputeEpilogue alpha;
  ElementComputeEpilogue beta;
  int split_k_slices;
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 32, 32, 32),
    filter_size(32, 3, 3, 32),
    padding(1, 1, 1, 1),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(true),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(0),
    split_k_slices(8),
    benchmark(false) 
  {}

  // Verify the problem size is compatible with the HYTLASS Convolution implementation.
  bool valid() {

    //
    // HYTLASS attempts to load 128b vectors of hytlass::half_t (F16) elements. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 8 elements.
    //
    int const kAlignment = 8;

    if ((input_size.c() % kAlignment) || (filter_size.n() % kAlignment)) {

      // misaligned tensors
      return false;
    }

    // Invalid padding
    if ((padding.h() != filter_size.h() / 2) || (padding.w() != filter_size.w() / 2)) {

      return false;
    }

    return true;
  }

  /// Updates input and filter sizes
  void update(
	      hytlass::Tensor4DCoord input_size,
	      hytlass::Tensor4DCoord filter_size,
	      hytlass::MatrixCoord stride) {

    this->input_size = input_size;
    this->filter_size = filter_size;
    conv_stride = stride;

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

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = true;
    }

    cmd.get_cmd_line_argument("n", input_size.n());
    cmd.get_cmd_line_argument("h", input_size.h());
    cmd.get_cmd_line_argument("w", input_size.w());
    cmd.get_cmd_line_argument("c", input_size.c());

    cmd.get_cmd_line_argument("k", filter_size.n());
    cmd.get_cmd_line_argument("r", filter_size.h());
    cmd.get_cmd_line_argument("s", filter_size.w());
    filter_size.c() = input_size.c();

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    cmd.get_cmd_line_argument("split-k-slices", split_k_slices);

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

    out << "10_hytlass_tensorop_wgrad_split_k example\n\n"
      << "  This example shows how to compute conv2d gradient with respect to weight (wgrad).\n"
      << "  In wgrad, the K dimension of impligit GEMM, corresponding to the sequential reduction loop, is very large (N * P * Q).\n"
      << "  Split-k with parallel reduction is highly effective for such cases.\n\n"
      << "Options:\n\n"
      << "  --help               If specified, displays this usage statement.\n\n"
      << "  --n=<int>            Input tensor extent N\n"
      << "  --h=<int>            Input tensor extent H\n"
      << "  --w=<int>            Input tensor extent W\n"
      << "  --c=<int>            Input tensor extent C\n"
      << "  --k=<int>            Filter extent K\n"
      << "  --r=<int>            Filter extent R\n"
      << "  --s=<int>            Filter extent S\n\n"
      << "  --alpha=<float>      Epilogue scalar alpha\n"
      << "  --beta=<float>       Epilogue scalar beta\n\n"
      << "  --split-k-slices=<int>   Split-k factor \n\n"
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/10_hytlass_tensorop_wgrad_split_k/gfx928_tensorop_conv2dwgrad_split_k --n=32 --h=224 --w=224 --c=128 --k=256 --r=3 --s=3 --split-k-slices=8\n\n";

        return out;
  }

  /// Computes the output tensor size (NPQK)
  hytlass::Tensor4DCoord output_size() const {
    return hytlass::Tensor4DCoord(input_size.n(),
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

    out << "Layer,N,H,W,C,K,R,S,Stride_H,Stride_W,Runtime,GFLOPs";

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
      << options.conv_stride.row() << ","
      << options.conv_stride.column() << ","
      << runtime_ms << ","
      << gflops;

    return out;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Runs one benchmark
Result profile_convolution(Options const &options) {

  Result result;

  //
  // Allocate host-device tensors using the HYTLASS Utilities.
  //

  // Inputs are the output gradient and the original activation.
  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.output_size());
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.input_size);
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(options.filter_size);
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.filter_size);
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.filter_size);

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

  // Fill tensor C, D on host with zeros
  hytlass::reference::host::TensorFill(tensor_c.host_view());

  hytlass::reference::host::TensorFill(tensor_d.host_view());

  // Fill tensor D for reference on host with zeros
  hytlass::reference::host::TensorFill(tensor_ref_d.host_view());

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

  // Partition the GEMM K loop into split_k_slices chunks
  int split_k_slices = options.split_k_slices;

  // Construct Conv2dProblemSize with user defined output size
  // Do not forget to pass the last argument.
  hytlass::conv::Conv2dProblemSize problem_size(
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices
  );

  using hytlass::layout::TensorNHWC;

  hytlass::conv::SplitKMode const split_k_mode = hytlass::conv::SplitKMode::kParallel;

  // Since the epilogue is not computed after GEMM, there is no need to pass the C tensor and
  // alpha and beta can be set to 1 and 0 respectively.
  // Moreover, since the output will be written to the workspace, there is no need to pass
  // the D tensor as well.
  // Do not forget to pass the last argument.
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    {nullptr, TensorNHWC()},
    {nullptr, TensorNHWC()},
    {ElementCompute(1), ElementCompute(0)},
    split_k_mode
  };

  //
  // Initialize HYTLASS Convolution
  //

  ImplicitGemm implicit_gemm;

  size_t workspace_size = implicit_gemm.get_workspace_size(arguments);

  // Split-K requires non-zero workspace size. The workspace size grows linearly with split_k_slices.
  std::cout << "split-k-slices: " << split_k_slices << std::endl;
  std::cout << "workspace size: " << workspace_size << std::endl;

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  result.status = implicit_gemm.can_implement(arguments);
  HYTLASS_CHECK(result.status);

  // After the workspace is allocated, we point the GEMM destination pointer to the workspace.
  TensorNHWC layout_D{TensorNHWC::packed(options.filter_size)};
  arguments.ref_D.reset(reinterpret_cast<ElementCompute*>(workspace.get()), layout_D);

  result.status = implicit_gemm.initialize(arguments, workspace.get());
  HYTLASS_CHECK(result.status);

  //
  // Launch initialized HYTLASS kernel
  //
  result.status = implicit_gemm();

  HYTLASS_CHECK(result.status);

  if (split_k_mode == hytlass::conv::SplitKMode::kParallel) {
    // Do reduction
    ReductionDevice reduction_op;
    auto& status = result.status;
    static hytlass::conv::Operator const kConvolutionalOperator = ImplicitGemm::kConvolutionalOperator;
    typename ReductionDevice::Arguments reduction_args(
        hytlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        hytlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        // Reduction input
        {
            reinterpret_cast<ElementAccumulator*> (workspace.get()),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        // Destination
        {
            tensor_d.device_data(),
            ReductionStrideIndex(tensor_d.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        // Source
        {
            tensor_c.device_data(),
            ReductionStrideIndex(tensor_c.stride()[ImplicitGemm::UnderlyingKernel::kTensorCStrideIdx])
        },
        {options.alpha, options.beta}
    );

    status = reduction_op.initialize(reduction_args, nullptr);
    status = reduction_op();
  }

  //
  // Optional reference check
  //

  if (options.reference_check) {
    std::cout << "Verification on device...\n";

    // Compute with reference implementation
    hytlass::reference::device::Conv2dWgrad<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementComputeEpilogue,
        ElementAccumulator,
        hytlass::NumericConverter<ElementOutput, ElementComputeEpilogue>>(
        problem_size,
        tensor_a.device_ref(),
        tensor_b.device_ref(),
        tensor_c.device_ref(),
        tensor_ref_d.device_ref(),
        options.alpha,
        options.beta);

    // Check if output from HYTLASS kernel and reference kernel are equal or not
    tensor_c.sync_host();
    tensor_d.sync_host();
    tensor_ref_d.sync_host();

    const ElementOutput non_zero_floor(1e-6f);
    ElementOutput eps(0.05);

    bool passed = hytlass::reference::host::TensorRelativelyEquals(tensor_d.host_view(), tensor_ref_d.host_view(), eps, non_zero_floor);

    if (!passed) {
      result.reference_check = hytlass::Status::kErrorInternal;
      std::cout << "ERROR - results miscompared.\n";
    }
    else {
      result.reference_check = hytlass::Status::kSuccess;
      std::cout << "Passed.\n";
    }
  }
  else {
    result.reference_check = hytlass::Status::kInvalid;
  }

  if (options.save_workspace) {

    std::stringstream ss;

    ss << "10_hytlass_tensorop_wgrad_split_k"
       << options.input_size.n() << "x" << options.input_size.h() << "x" << options.input_size.w() << "x" << options.input_size.c()
       << "_"
       << options.filter_size.n() << "x" << options.filter_size.h() << "x" << options.filter_size.w() << "x" << options.filter_size.c()
       << ".dat";

    std::ofstream output_workspace(ss.str());

    output_workspace
      << "Input = \n" << tensor_a.host_view() << "\n\n"
      << "Filters = \n" << tensor_b.host_view() << "\n\n";

    if (options.reference_check) {
      output_workspace << "Reference = \n" << tensor_ref_d.host_view() << "\n\n";
    }

    output_workspace << "Computed = \n" << tensor_c.host_view() << std::endl;

    std::cout << "Results written to '" << ss.str() << "'." << std::endl;
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
      result.status = implicit_gemm();
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

int main(int argc, char const **args) {

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {34, 408};

    struct Benchmark {
      int h, w, c, k, r, s, stride_h, stride_w;
    } layers[] = {
	 {56, 56,   64,  256, 1, 1, 1, 1},
	 {56, 56,   64,   64, 1, 1, 1, 1},
	 {56, 56,   64,   64, 3, 3, 1, 1},
	 {56, 56,  256,   64, 1, 1, 1, 1},
	 {56, 56,  256,  512, 1, 1, 2, 2},
	 {56, 56,  256,  128, 1, 1, 1, 1},
	 {56, 56,  128,  128, 3, 3, 2, 2},
	 {28, 28,  128,  512, 1, 1, 1, 1},
	 {28, 28,  512,  128, 1, 1, 1, 1},
	 {28, 28,  128,  128, 3, 3, 1, 1},
	 {28, 28,  512, 1024, 1, 1, 2, 2},
	 {28, 28,  512,  256, 1, 1, 1, 1},
	 {28, 28,  256,  256, 3, 3, 2, 2},
	 {14, 14,  256, 1024, 1, 1, 1, 1},
	 {14, 14, 1024,  256, 1, 1, 1, 1},
	 {14, 14,  256,  256, 3, 3, 1, 1},
	 {14, 14, 1024, 2048, 1, 1, 2, 2},
	 {14, 14, 1024,  512, 1, 1, 1, 1},
	 {14, 14,  512,  512, 3, 3, 2, 2},
	 { 7,  7,  512, 2048, 1, 1, 1, 1},
	 { 7,  7, 2048,  512, 1, 1, 1, 1},
	 { 7,  7,  512,  512, 3, 3, 1, 1},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {
        options.update({N, layer.h, layer.w, layer.c},
                       {layer.k, layer.r, layer.s, layer.c},
                       {layer.stride_h, layer.stride_w});

        Result result = profile_convolution(options);
        result.print(std::cout, idx, options) << std::endl;
      }

      ++idx;
    }
  }
  else {

    // Execute one problem size
    if (!options.valid()) {
      std::cerr << "Invalid problem." << std::endl;
      return -1;
    }

    Result result = profile_convolution(options);

    Result::print_header(std::cout, options) << std::endl;
    result.print(std::cout, 1, options) << std::endl;
  }

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

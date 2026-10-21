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


This example shows how to run convolution kernels using functions and data structures
provided by HYTLASS using tensor cores.

Writing a single high performance convolution kernel is hard but do-able. Whereas writing
high performance kernels at scale which works for multiple problem sizes with good abstractions is
really hard. HYTLASS solves this problem by providing simplified abstractions to compose
multiple sections of implicit gemm kernel. When used properly, the kernels can hit peak performance
of GPU easily.

HYTLASS divides a kernel into hierarchical composable sections. Which means, at each thread, warp
and thread-block level, they compute on their own tile-size with higher level of tile sizes being
composed from lower level ones. Multiple thread-tiles (tile size each thread computes) can be used
to form warp-tiles (tile size each warp computes) and multiple warp tiles can be used to compute
threadblock-tile (tile size computed by a threadblock).

In thie example, we split variable initialization into
1. Setting up data properties : describes how tensors are laid out in the memory and how the kernel
can view them (logical to physical mapping)
2. Setting up computation properties : describes how the above set tensors will be used to compute
output of convolution.

First, we setup the data types of the input tensor A, weights' tensor B and output tensor C along
with alpha, beta as the equation for convolution is C = alpha * Conv(A, B) + beta * C. In HYTLASS,
the kernels first compute Conv(A, B) and leave the rest of the computation to end of the kernel as
alpha * X + beta * C is a simple element-wise operation on X (Conv(A, B)) and C. We call this as 
epilogue of kernel. Hence, we setup data types for alpha and beta to be equal to 
ElementComputeEpilogue = float. We want to use MMA instructions on Turing and they support 4-bit
signed integer. But int4b_t is not fully supported by HYGON software stack, so HYTLASS introduces
hytlass::int4b_t. We use the data type for elements in input tensor A and B as hytlass::int4b_t. We
convey this to HYTLASS kernel by initializing template variables ElementAccumulator (int32_t),
ElementComputeEpilogue (float), ElementInputA (hytlass::int4b_t), ElementInputB (hytlass::int4b_t),
ElementOutput (int32_t). Communicating just the data type is not enough. As the data is laid out 
linearly in memory, we have to convey the layout of tensors. We do that by initializing template
variables LayoutInputA, LayoutInputB and LayoutOutput to TensorNHWC hytlass variable. Next, we setup
rules to comptue alpha * X + beta * C which is called epilogue of the kernel. We initialize template
variable EpilogueOp, which takes the data type of output ElementOutput (int32_t), the number of
elements per vector memory access (32), data type of accumulator (int32_t) and data type of
computation of linear combination (alpha * X + beta * C).

Now that we setup the properties of data, we have to setup properties of computation.

Second, we create template variables of tile sizes for thread-block, warp and mma-op to 128x128x128,
64x64x128, 8x8x32 (MxNxK) respectively. When passed to instantiate HYTLASS Implicit GEMM kernel, it
internally deduces the amount of threads needed per thread-block, amount of shared memory, storing
data in bank-conflict free manner, and ton of other variables required to compose, initialize and
launch a high performance Implicit GEMM kernel. This is the beauty of HYTLASS, it relieves developer
from understanding and coding complicated hardware optimizations which can easily go wrong.

HYTLASS also supports multiple MMA pipelines in a threadblock. What are MMA pipelines? MMA pipelines
constitute the whole process of loading input data from global memory to shared memory, loading data
from shared memory to registers, doing matrix multiplication, store to global memory. The below flow
sequence shows a typical mma pipeline.

tensor in global memory -> registers -> tile in shared memory -> registers -> mma -> registers ->
output to global memory

The problem with single pipeline is, each stage is synchronous which means, each stage has to wait
until the previous finished executing. There are stages in the pipeline which do not have fixed
latency, for example, the loads from global memory and shared memory. Therefore, we can add one more
pipeline with a phase shift in mma kernel to hide latency from global and shared memory loads.
Finally, the pipeline in a kernel looks like

(1) tensor in global memory -> (2) registers -> (3) tile in shared memory -> (4) registers -> (5)
mma -> (6) registers -> (7) output to global memory (1) <null> -> (2) <null> -> (3) tensor in global
memory -> (4) registers -> (5) tile in shared memory -> (6) registers -> (7) mma -> (8) registers ->
(9) output to global memory

This way, you can hide the second global memory load latency by doing computation on already loaded
input data.

There are few more template variables initialized such as, which threadblock tile of output matrix
is done which threadblock launched on an SM, GFX architecture of GPU you want to run on.

These are all put together to create a template variable which describes HYTLASS Implicit GEMM
kernel using hytlass::conv::device::ImplicitGemm template.

The next step is to initialize physical data, instantiate and initialize HYTLASS kernel and run it.
We use HYTLASS utilities to initialize, fill, compare tensors as they are simple and doesn't come
in the way of learning HYTLASS.

Once all the tensors are initialized and filled with data, create arguments tuple to launch HYTLASS
kernel which takes problem size (N = 1, H = 64, W = 64, C = 128), filter size (K = 64,
R = 3, S = 3, C = 128 ), padding, strides, dilation, tensors, alpha, beta and the
important one, split k-dimension factor. Along with that, we query HYTLASS if any scratch-space
memory required by the kernel we instantiated. If yes, we create it and pass it along with other
arguments created to initialize HYTLASS kernel then, the kernel is launched.

In this example, we later on launch a reference convolution kernel (from HYTLASS utilities) to
compare if the output from HYTLASS kernel is same as the reference implicit GEMM kernel.
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hytlass/gemm/device/gemm.h"
#include "hytlass/conv/kernel/default_conv2d_fprop.h"
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
using ElementAccumulator = float;                 // Data type of accumulator
using ElementComputeEpilogue = float;               // Data type of epilogue computation (alpha, beta)
using ElementInputA = hytlass::half_t;             // Data type of elements in input tensor
using ElementInputB = hytlass::half_t;             // Data type of elements in input tensor
using ElementOutput = float;                       // Data type of elements in output tensor

using LayoutInputA = hytlass::layout::TensorNHWC;
using LayoutInputB = hytlass::layout::TensorNHWC;
using LayoutOutput = hytlass::layout::TensorNHWC;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = hytlass::arch::OpClassTensorOp;

// This code section describes GFX architecture number
using SmArch = hytlass::arch::Gfx928;

// This code section describes the tile size a thread block will compute
using ThreadblockShape = hytlass::gemm::GemmShape<64, 64, 128>;  // Threadblock tile shape

// This code section describes tile size a warp will compute
using WarpShape = hytlass::gemm::GemmShape<32, 32, 128>;         // Warp tile shape

// This code section describes the size of MMA opS
using InstructionShape = hytlass::gemm::GemmShape<16, 16, 32>;    // TensorCore instruction shape

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;

// 1 -> singlestage
// 2 -> pipelined
constexpr int NumStages = 1;
// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = hytlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,                                     // Data type of output matrix.
    128 / hytlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
                                                       // memory access. This becomes the vector width of
                                                       // math instructions in the epilogue too.
    ElementAccumulator,                                // Data type of accumulator
    ElementComputeEpilogue,                            // Data type for alpha/beta in linear combination
    hytlass::epilogue::thread::ScaleType::NoBetaScaling>;                           

// EpilogueOp::Params params(ElementComputeEpilogue(0.5), ElementComputeEpilogue(0));

using Conv2dFpropKernel = typename hytlass::conv::kernel::DefaultConv2dFprop<
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
  hytlass::conv::IteratorAlgorithm::kOptimized,
  hytlass::conv::StrideSupport::kStrided,
  8,
  8
>::Kernel;

using ImplicitGemm = hytlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;


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
  bool benchmark;
  std::string tag;

  Options():
    help(false),
    input_size(1, 14, 14, 1024),
    filter_size(256, 1, 1, 1024),
    padding(0, 0, 0, 0),
    conv_stride(1, 1),
    dilation(1, 1),
    reference_check(true),
    measure_performance(true),
    iterations(20),
    save_workspace(false),
    alpha(1),
    beta(1),
    benchmark(false) { }

  // Verify the problem size is compatible with the HYTLASS Convolution implementation.
  bool valid() {

    //
    // HYTLASS attempts to load 128b vectors. Consequently,
    // all pointers, strides, and tensor extents must be divisible by 32 elements.
    //
    // int const kAlignment = 128 / hytlass::sizeof_bits<ElementInputA>::value;

    // if ((input_size.c() % kAlignment) ||
    //   (filter_size.n() % kAlignment)) {

    //   // misaligned tensors
    //   return false;
    // }
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

    if (cmd.check_cmd_line_flag("save-workspace")) {
      save_workspace = true;
    }

    if (cmd.check_cmd_line_flag("benchmark")) {
      benchmark = false;
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
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tag", tag);

    int32_t padding_h = filter_size.h() / 2;
    int32_t padding_w = filter_size.w() / 2;
    padding = {padding_h, padding_h, padding_w, padding_w};

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "13_tensorop_fused_conv2d_fprop example\n\n"
      << "  This example uses Turing's Tensor Core operators on int4 data types to compute\n"
      << "  forward convolution on tensors of layout NHWC.\n\n"
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
      << "  --ref-check          If set (true), reference check on the host is computed\n"
      << "  --perf-check         If set (true), performance is measured.\n"
      << "  --benchmark          If set (true), performance benchmarking on several layers and batch-size.\n"
      << "  --iterations=<int>   Number of profiling iterations to perform.\n"
      << "  --save-workspace     If set, workspace is written to a text file.\n"
      << "  --tag=<string>       String to replicate across the first column in the results table\n";

    out << "\n\nExamples:\n\n"
      << "$ ./gfx928_tensorop_conv2d_bias_relu  --n=32 --h=224 --w=224 --c=128 --k=256 --r=1 --s=1\n\n"
      << "$ ./gfx928_tensorop_conv2d_bias_relu  --n=1 --h=224 --w=224 --c=32 --k=32 --r=3 --s=3 --ref-check\n\n";

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

    out << "Layer,N,H,W,C,K,R,S,Runtime,GFLOPs";

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

  hytlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(options.input_size);
  hytlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(options.filter_size);
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_c_bias({1, 1, 1, options.output_size().c()});
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(options.output_size());
  hytlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(options.output_size());

  //
  // Initialize tensors
  //

  // Fill tensor A on host with uniform-distribution random data

  hytlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      hytlass::MantissaInBits<ElementOutput>::bits);
  
  // Fill tensor B on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      2,
      ElementInputB(4),
      ElementInputB(-4),
      hytlass::MantissaInBits<ElementOutput>::bits);
  // <- Fill matrix C on host with uniform-distribution random data
     hytlass::reference::host::TensorFillRandomUniform(
      tensor_c_bias.host_view(),
      1,
      ElementOutput(1),
      ElementOutput(-1),
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
  tensor_c_bias.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();
  
  //
  // Define arguments for HYTLASS Convolution
  //

  // mode (kCrossCorrelation or kConvolution)
  hytlass::conv::Mode mode = hytlass::conv::Mode::kCrossCorrelation;

  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  ElementComputeEpilogue alpha     = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta      = ElementComputeEpilogue(0);
  // Construct Conv2dProblemSize with user defined output size
  hytlass::conv::Conv2dProblemSize problem_size(      
      options.input_size,
      options.filter_size,
      options.padding,
      options.conv_stride,
      options.dilation,
      options.output_size(),
      mode,
      split_k_slices);

  // Construct ImplicitGemm::Argument structure with conv2d 
  // problem size, data pointers, and epilogue values
  typename ImplicitGemm::Arguments arguments{
    problem_size,
    tensor_a.device_ref(),
    tensor_b.device_ref(),
    {tensor_c_bias.device_data(), LayoutOutput::Stride(0)},
    tensor_d.device_ref(),
    {alpha,beta},
  };

  //
  // Initialize HYTLASS Convolution
  //

  ImplicitGemm implicit_gemm_op;

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
    std::cout << "Verification on host...\n";

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
      tensor_c_bias.device_ref(), 
      tensor_ref_d.device_ref(),
      alpha,
      beta
    );

    // Check if output from HYTLASS kernel and reference kernel are equal or not
    tensor_d.sync_host();
    tensor_ref_d.sync_host();
    // Compute bias + relu in host code
    for (int n = 0; n < problem_size.N; ++n) {
      for (int p = 0; p < problem_size.P; ++p) {
        for (int q = 0; q < problem_size.Q; ++q) {
          for (int k = 0; k < problem_size.K; ++k) {
            
            tensor_ref_d.at({n, p, q, k}) =
                std::max(ElementOutput(0),
                        ElementOutput(tensor_ref_d.at({n, p, q, k}) +
                                     tensor_c_bias.at({0, 0, 0, k})));
          }
        }
      }
    }

    const ElementOutput non_zero_floor(1e-6);
    ElementOutput eps(1e-3);

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

    ss << "09_tensor_conv_workspace_conv2dfprop_"
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

    output_workspace << "Computed = \n" << tensor_d.host_view() << std::endl;

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

int main(int argc, char const **args) {

  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.benchmark) {
    // Benchmark several layers

    int batch_sizes[] = {1, 32, 64, 128, 256, 512};

    struct Benchmark {
      int h, w, c, k, r, s;
    } layers[] = {
      {56,  56,   64,   256, 1, 1},
      {56,  56,   64,    64, 1, 1},
      {56,  56,   64,    64, 3, 3},
      {56,  56,  256,    64, 1, 1},
      {56,  56,  256,   512, 1, 1},
      {56,  56,  256,   128, 1, 1},
      {28,  28,  128,   128, 3, 3},
      {28,  28,  128,   512, 1, 1},
      {28,  28,  512,   128, 1, 1},
      {28,  28,  512,  1024, 1, 1},
      {28,  28,  512,   256, 1, 1},
      {14,  14,  256,   256, 3, 3},
      {14,  14,  256,  1024, 1, 1},
      {14,  14,  1024,  256, 1, 1},
      {14,  14,  1024, 2048, 1, 1},
      {14,  14,  1024,  512, 1, 1},
      {7,    7,   512,  512, 3, 3},
    };

    Result::print_header(std::cout, options) << std::endl;

    int idx = 1;

    for (auto const &layer : layers) {
      for (auto N : batch_sizes) {

        options.update({N, layer.h, layer.w, layer.c}, {layer.k, layer.r, layer.s, layer.c});

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

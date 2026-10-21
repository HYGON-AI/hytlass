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

/***************************************************************************************************
 Example contrasting the Stream-K parallel decomposition for GEMM threadblocks versus the
 "classic data-parallel" and "Split-K" decompositions.

 For more details regarding the Stream-K method, see "Stream-K: Work-centric Parallel Decomposition
 for Dense Matrix-Matrix Multiplication on the GPU" (https://arxiv.org/abs/2301.03598)

 Requires HYGON gfx928 or newer device (Gfx928+).

 - Build and run:

     hytlass$ mkdir build

     hytlass$ cd build

     hytlass/build$ cmake .. -DHYTLASS_HIPCC_ARCHS=928

     hytlass/build$ make gfx928_streamk_gemm

     hytlass/build$ ./examples/03_hytlass_streamk_gemm/gfx928_streamk_gemm

        10000 timing iterations of 2048 x 2048 x 2048 matrix-matrix multiply

        Basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.112633 ms
          GFLOPs: 152530

        StreamK GEMM with default load-balancing
          Disposition: Passed
          Avg runtime: 0.0941929 ms
          GFLOPs: 182390
          Speedup vs Basic-DP: 1.196

        StreamK emulating basic data-parallel GEMM
          Disposition: Passed
          Avg runtime: 0.113119 ms
          GFLOPs: 151875
          Speedup vs Basic-DP: 0.996

        Basic split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.104772 ms
          GFLOPs: 163973

        StreamK emulating Split-K GEMM with tile-splitting factor 2
          Disposition: Passed
          Avg runtime: 0.105379 ms
          GFLOPs: 163029
          Speedup vs Basic-SplitK: 0.994

 **************************************************************************************************/

#include <iostream>
#include <string>

#include "hytlass/hytlass.h"
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
/// GEMM kernel configurations (hytlass_tensorop_h16816gemm_128x128_32x4_nn_align8)
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = hytlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = hytlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / hytlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = hytlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = hytlass::layout::ColumnMajor;                      // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / hytlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = hytlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = hytlass::layout::RowMajor;                      // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / hytlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

// Multiply-accumulate blocking/pipelining details
using ElementAccumulator  = float;                          // Element type for internal accumulation
using ArchTag             = hytlass::arch::Gfx928;                     // Tag indicating the minimum Gfx that supports the intended feature
using OperatorClass       = hytlass::arch::OpClassTensorOp;            // Operator class tag
using ThreadblockShape    = hytlass::gemm::GemmShape<128, 128, 32>;    // Threadblock-level tile size (concept: GemmShape)
using WarpShape           = hytlass::gemm::GemmShape<64, 64, 32>;     // Warp-level tile size (concept: GemmShape)
using InstructionShape    = hytlass::gemm::GemmShape<16, 16, 16>;      // Instruction-level tile size (concept: GemmShape)
constexpr int NumStages   = 1;                                         // Number of global->shared pipeline stages used in the GEMM mainloop

// Epilogue output operator
using EpilogueOp = hytlass::epilogue::thread::LinearCombination<
    ElementC,               // Element type for C and D matrix operands
    AlignmentC,             // Memory access granularity of C and D matrix in units of elements
    ElementAccumulator,     // Element type from internal accumaccumulation
    ElementAccumulator>;    // Data type used to compute linear combination

// Reference device GEMM implementation type
using DeviceGemmReference = hytlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

// StreamK device GEMM implementation type
using DeviceGemmStreamK = hytlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    hytlass::gemm::threadblock::ThreadblockSwizzleStreamK, // <-- Only difference
    NumStages,
    AlignmentA,
    AlignmentB>;


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {
  double avg_runtime_ms;
  double gflops;
  hytlass::Status status;
  hipError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    hytlass::Status status = hytlass::Status::kSuccess,
    hipError_t error = hipSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(true)
  {}

};


/// Command line options parsing
struct Options {
  std::string               command_name;
  bool                      help;
  hytlass::gemm::GemmCoord  problem_size;
  float                     alpha;
  float                     beta;
  int                       split_k_factor;
  int                       avail_sms;
  bool                      reference_check;
  int                       iterations;

  hytlass::HostTensor<ElementA, LayoutA> tensor_a;
  hytlass::HostTensor<ElementB, LayoutB> tensor_b;
  hytlass::HostTensor<ElementC, LayoutC> tensor_c;
  hytlass::HostTensor<ElementC, LayoutC> tensor_d;
  hytlass::HostTensor<ElementC, LayoutC> tensor_ref_d;

  Options(std::string command_name) :
    command_name(command_name),
    help(false),
    problem_size({2048, 2048, 2048}),
    alpha(1.0f),
    beta(0.0f),
    split_k_factor(1),
    avail_sms(-1),              // Number of device SMs to use is unlimited
    reference_check(true),
    iterations(100)
  {}

  bool valid() const {
    return true;
  }

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
    cmd.get_cmd_line_argument("split", split_k_factor);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {
  out << "03_hytlass_streamk_gemm example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --split=<int>               Split-K factor to emulate\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

  out << "\n\nExamples:\n\n"
      << "$ ./examples/03_hytlass_streamk_gemm/gfx928_streamk_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    // Two flops per multiply-add
    return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Populates a DeviceGemmStreamK::Arguments structure from the given commandline options
typename DeviceGemmStreamK::Arguments args_from_options(
    const DeviceGemmStreamK &device_gemm,
    const Options &options,
    hytlass::HostTensor<ElementA, LayoutA> &tensor_a,
    hytlass::HostTensor<ElementB, LayoutB> &tensor_b,
    hytlass::HostTensor<ElementC, LayoutC> &tensor_c,
    hytlass::HostTensor<ElementC, LayoutC> &tensor_d)
{
  return typename DeviceGemmStreamK::Arguments(
    hytlass::gemm::GemmUniversalMode::kGemm,  // universal mode
    options.problem_size,                     // problem_size
    options.split_k_factor,                   // batch count / splitk slices
    {                                         // epilogue parameters
      ElementAccumulator(options.alpha),
      ElementAccumulator(options.beta)
    },
    tensor_a.device_data(),                   // ptr_A
    tensor_b.device_data(),                   // ptr_B
    tensor_c.device_data(),                   // ptr_C
    tensor_d.device_data(),                   // ptr_D
    options.problem_size.mk().product(),      // batch_stride_A
    options.problem_size.nk().product(),      // batch_stride_B
    options.problem_size.mn().product(),      // batch_stride_C
    options.problem_size.mn().product(),      // batch_stride_D
    tensor_a.layout().stride(0),              // stride_a
    tensor_b.layout().stride(0),              // stride_b
    tensor_c.layout().stride(0),              // stride_c
    tensor_d.layout().stride(0),              // stride_d
    options.avail_sms);                       // avail_sms
}


/// Execute a given example GEMM computation
template <typename DeviceGemmT>
Result run(std::string description, Options &options) {
  // Display test description
  std::cout << std::endl << description << std::endl;

  // Zero-initialize test output matrix D
  hytlass::reference::host::TensorFill(options.tensor_d.host_view());
  options.tensor_d.sync_device();

  // Instantiate HYTLASS kernel depending on templates
  DeviceGemmT device_gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of DeviceGemmT
  auto arguments = args_from_options(device_gemm, options, options.tensor_a, options.tensor_b, options.tensor_c, options.tensor_d);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = DeviceGemmT::get_workspace_size(arguments);

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check the problem size is supported or not
  HYTLASS_CHECK(device_gemm.can_implement(arguments));

  // Initialize HYTLASS kernel with arguments and workspace pointer
  HYTLASS_CHECK(device_gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  HYTLASS_CHECK(device_gemm());

  // Copy output data from HYTLASS and reference kernel to host for comparison
  options.tensor_d.sync_host();

  // Check if output from HYTLASS kernel and reference kernel are equal or not
  Result result;
  ElementC eps(1e-3);
  ElementC non_zero_floor(1e-6);
  // Reference check
  result.passed = hytlass::reference::host::TensorRelativelyEquals(options.tensor_d.host_view(), 
    options.tensor_ref_d.host_view(), eps, non_zero_floor);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  // Run profiling loop
  if (options.iterations > 0) {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      HYTLASS_CHECK(device_gemm());
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPs: " << result.gflops << std::endl;
  }

  if (!result.passed) {
    printf("verify failed");
  }

  return result;
}


/// Program entrypoint
int main(int argc, const char **argv) {
  // Parse commandline options
  Options options("gfx928_streamk_gemm");
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  std::cout <<
    options.iterations << " timing iterations of " <<
    options.problem_size.m() << " x " <<
    options.problem_size.n() << " x " <<
    options.problem_size.k() << " matrix-matrix multiply" << std::endl;

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }


  //
  // Initialize GEMM datasets
  //

  // Initialize tensors using HYTLASS helper functions
  options.tensor_a.resize(options.problem_size.mk());       // <- Create matrix A with dimensions M x K
  options.tensor_b.resize(options.problem_size.kn());       // <- Create matrix B with dimensions K x N
  options.tensor_c.resize(options.problem_size.mn());       // <- Create matrix C with dimensions M x N
  options.tensor_d.resize(options.problem_size.mn());       // <- Create matrix D with dimensions M x N used to store output from HYTLASS kernel
  options.tensor_ref_d.resize(options.problem_size.mn());   // <- Create matrix D with dimensions M x N used to store output from reference kernel

  // Fill matrix A on host with uniform-random data [-2, 2]
  hytlass::reference::host::TensorFillRandomUniform(
      options.tensor_a.host_view(),
      1,
      ElementA(2),
      ElementA(-2),
      0);

  // Fill matrix B on host with uniform-random data [-2, 2]
  hytlass::reference::host::TensorFillRandomUniform(
      options.tensor_b.host_view(),
      1,
      ElementB(2),
      ElementB(-2),
      0);

  // Fill matrix C on host with uniform-random data [-2, 2]
  hytlass::reference::host::TensorFillRandomUniform(
      options.tensor_c.host_view(),
      1,
      ElementC(2),
      ElementC(-2),
      0);


  //
  // Compute reference output
  //

  // Copy data from host to GPU
  options.tensor_a.sync_device();
  options.tensor_b.sync_device();
  options.tensor_c.sync_device();

  // Zero-initialize reference output matrix D
  hytlass::reference::host::TensorFill(options.tensor_ref_d.host_view());
  options.tensor_ref_d.sync_device();

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    options.problem_size,
    ElementAccumulator(options.alpha),
    options.tensor_a.device_ref(),
    options.tensor_b.device_ref(),
    ElementAccumulator(options.beta),
    options.tensor_c.device_ref(),
    options.tensor_ref_d.device_ref());

  // Wait for kernels to finish
  HIP_CHECK(hipDeviceSynchronize());

  // Copy output data from reference kernel to host for comparison
  options.tensor_ref_d.sync_host();


  //
  // Evaluate HYTLASS kernels
  //

  // Test default operation
  if (options.split_k_factor == 1) {

    Result streamk_default  = run<DeviceGemmStreamK>("StreamK GEMM with default load-balancing", options);

    // Show that StreamK can emulate basic data-parallel GEMM when we set the number of SMs to load-balance across = 1
    options.avail_sms       = 1;        // Set loadbalancing width to 1 SM (no load balancing)
    Result streamk_dp       = run<DeviceGemmStreamK>("StreamK emulating basic data-parallel GEMM", options);
    options.avail_sms       = -1;       // Reset loadbalancing width to unspecified SMs (i.e., the number of device SMs)

    options.split_k_factor++;     // Increment splitting factor for next evaluation

  }

  Result streamk_splitk = run<DeviceGemmStreamK>(
    std::string("StreamK emulating Split-K GEMM with tile-splitting factor ") + std::to_string(options.split_k_factor),
    options);

  return 0;
}

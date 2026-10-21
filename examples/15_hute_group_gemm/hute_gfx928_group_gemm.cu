/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Grouped GEMM example using HYTLASS 3 APIs for HCU architecture.

    For this example all scheduling work is performed on the device.

    To run this example:

      $ ./hute_gfx928_group_gemm --m=2048 --n=2048 --k=2048 --groups=10

      The above example command makes all 10 groups to be sized at the given m, n, k sizes. 
      Skipping any of the problem dimensions randomizes it across the different groups.
      Same applies for alpha and beta values that are randomized across the different groups.

    To run this example for a set of problems using the benchmark option:

      $ ./hute_gfx928_group_gemm --benchmark=./test_benchmark.txt

      Where the test_benchmark.txt may look as such:
        0 256x512x128
        1 256x512x512
        2 512x256x128
        3 256x256x128
        4 256x512x1024
        5 1024x512x128 and so on
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>

#include "hytlass/hytlass.h"

#include "hute/tensor.hpp"
#include "hytlass/tensor_ref.h"
#include "hytlass/epilogue/collective/default_epilogue.hpp"
#include "hytlass/epilogue/thread/linear_combination.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/gemm/group_array_problem_shape.hpp"
#include "hytlass/gemm/collective/collective_builder.hpp"
#include "hytlass/epilogue/collective/collective_builder.hpp"
#include "hytlass/gemm/device/gemm_universal_adapter.h"
#include "hytlass/gemm/kernel/gemm_universal.hpp"

#include "hytlass/util/command_line.h"
#include "hytlass/util/distribution.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/packed_stride.hpp"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/device/tensor_compare.h"
#include "hytlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace hute;
using ProblemShape = hytlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group
using ElementA = hytlass::half_t;                          // Element type for A matrix operand
using ElementB = hytlass::half_t;                          // Element type for B matrix operand
using ElementC = hytlass::half_t;                                // Element type for C and D matrix operands


/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         LayoutA     = hytlass::layout::ColumnMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / hytlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         LayoutB     = hytlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / hytlass::sizeof_bits<ElementB>::value;    // Alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         LayoutC     = hytlass::layout::ColumnMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / hytlass::sizeof_bits<ElementC>::value;    // Alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = hytlass::arch::Gfx928;                            // Tag indicating the minimum Gfx that supports the intended feature
using OperatorClass       = hytlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_32>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_1,_1,_1>; 
using WarpShape_MNK = Shape<_2, _2, _1>;
using InstructionShape_MNK = Shape<_32, _32, _16>;
using StageCountType = hytlass::gemm::collective::StageCount<2>;  
using KernelSchedule = hytlass::gemm::KernelPtrArraySpecialized; // Kernel to launch
using EpilogueSchedule = hytlass::epilogue::PtrArrayNoSmemWarpSpecialized;                     // Epilogue to launch


using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
    hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    hytlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC *, AlignmentC,
    ElementC, LayoutC *, AlignmentC,
    EpilogueSchedule
  >::CollectiveOp;

using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA *, AlignmentA,
    ElementB, LayoutB *, AlignmentB,
    ElementAccumulator,
    TileShape, WarpShape_MNK, InstructionShape_MNK, ClusterShape,
    StageCountType,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

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

using StrideA = typename Gemm::GemmKernel::UnderlyingStrideA;
using StrideB = typename Gemm::GemmKernel::UnderlyingStrideB;
using StrideC = typename Gemm::GemmKernel::UnderlyingStrideC;
using StrideD = typename Gemm::GemmKernel::UnderlyingStrideD;

// Host-side allocations
std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

std::vector<StrideA> stride_A_host;
std::vector<StrideB> stride_B_host;
std::vector<StrideC> stride_C_host;
std::vector<StrideD> stride_D_host;

std::vector<ElementAccumulator> alpha_host;
std::vector<ElementAccumulator> beta_host;

// Device-side allocations
hytlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

hytlass::DeviceAllocation<typename Gemm::ElementA> block_A;
hytlass::DeviceAllocation<typename Gemm::ElementB> block_B;
hytlass::DeviceAllocation<typename Gemm::ElementC> block_C;
hytlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
hytlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

hytlass::DeviceAllocation<const typename Gemm::ElementA *> ptr_A;
hytlass::DeviceAllocation<const typename Gemm::ElementB *> ptr_B;
hytlass::DeviceAllocation<const typename Gemm::ElementC *> ptr_C;
hytlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_D;
hytlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput *> ptr_ref_D;

hytlass::DeviceAllocation<StrideA> stride_A;
hytlass::DeviceAllocation<StrideB> stride_B;
hytlass::DeviceAllocation<StrideC> stride_C;
hytlass::DeviceAllocation<StrideD> stride_D;

// Note, this is an array of pointers to alpha and beta scaling values per group
hytlass::DeviceAllocation<ElementAccumulator*> alpha_device;
hytlass::DeviceAllocation<ElementAccumulator*> beta_device;
hytlass::DeviceAllocation<ElementAccumulator> block_alpha;
hytlass::DeviceAllocation<ElementAccumulator> block_beta;



/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = FLT_MAX;
  float beta  = FLT_MAX;
  int iterations = 10;
  int m = 1024, n = 2048, k = 512, groups = 10;
  std::string benchmark_path;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  int const tma_alignment_bits = 128;
  int const alignment = tma_alignment_bits / hytlass::sizeof_bits<ElementA>::value; 

  // Parses the command line
  void parse(int argc, char const **args) {
    hytlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m);
    cmd.get_cmd_line_argument("n", n);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("groups", groups);
    cmd.get_cmd_line_argument("alpha", alpha, FLT_MAX);
    cmd.get_cmd_line_argument("beta",  beta,  FLT_MAX);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("benchmark", benchmark_path);

    // Decide how to initialize the problems
    if (!benchmark_path.empty()) {
      if (!benchmark_problems()) {
        problem_sizes_host.clear();
        return;
      }
    }
    else {
      randomize_problems(cmd);
    }
  }

  void randomize_problems(hytlass::CommandLine &cmd) {
    int cmd_line_m = -1, cmd_line_n = -1, cmd_line_k = -1;
    cmd.get_cmd_line_argument("m", cmd_line_m);
    cmd.get_cmd_line_argument("n", cmd_line_n);
    cmd.get_cmd_line_argument("k", cmd_line_k);

    problem_sizes_host.reserve(groups);

    for (int i = groups; i > 0; i--) {
      int m = cmd_line_m;
      int n = cmd_line_n;
      int k = cmd_line_k;
      if (m < 1) {
        m = ((rand() % 512) + 1);
      }
      if (n < 1) {
        n = ((rand() % 512) + 1);
      }
      if (k < 1) {
        k = alignment * ((rand() % 64) + 1);
      }
      problem_sizes_host.push_back({m, n, k});
    }
  }

  /// Load a benchmark
  bool benchmark_problems() {
    std::ifstream file(benchmark_path);
    if (!file.good()) {
      return false;
    }

    while (file.good()) {

      int idx = -1;
      std::string extent_str;

      file >> idx >> extent_str;

      if (idx < 0 || extent_str.empty()) {
        break;
      }

      hytlass::gemm::GemmCoord extent;
      std::vector<std::string> tokens;

      hytlass::CommandLine::tokenize(tokens, extent_str, 'x');

      for (int i = 0; i < int(tokens.size()); ++i) {
        int x = std::atoi(tokens.at(i).c_str());

        // round up
        if (x % alignment) {
          x += (alignment - (x % alignment));
        }

        extent.at(i) = x;
      }

      if (extent.product()) {
        problem_sizes_host.push_back({extent.m(), extent.n(), extent.k()});
      }
    }
    groups = static_cast<int>(problem_sizes_host.size());

    return true;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "hute_group_gemm\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM for all groups\n"
      << "  --n=<int>                   Sets the N extent of the GEMM for all groups\n"
      << "  --k=<int>                   Sets the K extent of the GEMM for all groups\n"
      << "  --groups=<int>              Sets the number of individual GEMM problems for Grouped GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n"
      << "  --benchmark=<str>           Executes a benchmark problem size.\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "hute_gfx928_group_gemm" << " --m=1024 --n=512 --k=1024 --groups=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  hytlass::Status status = hytlass::Status::kSuccess;
  hipError_t error = hipSuccess;
  bool passed = false;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  hytlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023) {

  Element scope_max, scope_min;
  int bits_input = hytlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = static_cast<Element>(1);
    scope_min = static_cast<Element>(1);
  } else if (bits_input <= 8) {
    scope_max = static_cast<Element>(1);
    scope_min = static_cast<Element>(1);
  } else {
    scope_max = static_cast<Element>(4);
    scope_min = static_cast<Element>(2);
  }

  hytlass::reference::device::BlockFillRandomUniform(
    block.get(), block.size(), seed, scope_max, scope_min, 0);

  return true;
}

/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;

    stride_A_host.push_back(hytlass::make_hute_packed_stride(StrideA{}, hute::make_shape(M, K, Int<1>{})));
    stride_B_host.push_back(hytlass::make_hute_packed_stride(StrideB{}, hute::make_shape(N, K, Int<1>{})));
    stride_C_host.push_back(hytlass::make_hute_packed_stride(StrideC{}, hute::make_shape(M, N, Int<1>{})));
    stride_D_host.push_back(hytlass::make_hute_packed_stride(StrideD{}, hute::make_shape(M, N, Int<1>{})));

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {
  
  uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.groups);
  std::vector<ElementB *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementC *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>((rand() % 5) + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  ptr_A.reset(options.groups);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options, bool host_problem_shapes_available = true)
{
  hytlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = hytlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::EpilogueOutputOp::Params params;
  if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
    // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
    params = typename Gemm::EpilogueOutputOp::Params(
      ElementAccumulator(options.alpha), ElementAccumulator(options.beta));
  }
  else {
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
    params = typename Gemm::EpilogueOutputOp::Params(alpha_device.get(), beta_device.get());
  }

  typename Gemm::Arguments arguments;
  if (host_problem_shapes_available) {
    arguments = typename Gemm::Arguments {
      hytlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {params, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info
    };
  }
  else {
    arguments = typename Gemm::Arguments {
      hytlass::gemm::GemmUniversalMode::kGrouped,
      {options.groups, problem_sizes.get(), nullptr},
      {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
      {params, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
      hw_info
    };
  }

  return arguments;
}

bool verify(const Options &options) {
  bool passed = true;
  for (int32_t i = 0; i < options.groups; ++i) {
    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);
    hytlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({M, K}));
    hytlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({K, N}));
    hytlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({M, N}));
    hytlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({M, N}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(
      {M, N, K},
      ElementAccumulator(alpha_host.at(i)),
      ref_A,
      ref_B,
      ElementAccumulator(beta_host.at(i)),
      ref_C,
      ref_D);

    // Wait for kernel to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Check if output from HYTLASS kernel and reference kernel are equal or not
    passed &= hytlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N);
    #if 0
    std::cout << "Group: " << i << " Status: " << passed << std::endl;
    #endif
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options, bool host_problem_shapes_available = true)
{
  allocate(options);
  initialize(options);

  // Instantiate HYTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options, host_problem_shapes_available);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  HYTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize HYTLASS kernel with arguments and workspace pointer
  HYTLASS_CHECK(gemm.initialize(arguments, workspace.get()));

  // Correctness / Warmup iteration
  HYTLASS_CHECK(gemm.run());

  // Check if output from HYTLASS kernel and reference kernel are equal or not
  Result result;
  result.passed = verify(options);

  std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

  if (!result.passed) {
    exit(-1);
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      HYTLASS_CHECK(gemm.initialize(arguments, workspace.get()));
      HYTLASS_CHECK(gemm.run());
    }
    timer.stop();

    // Compute average setup and runtime and GFLOPs.
    float elapsed_ms       = timer.elapsed_millis();
    result.avg_runtime_ms  = double(elapsed_ms) / double(options.iterations);
    result.gflops          = options.gflops(result.avg_runtime_ms / 1000.0, options.problem_sizes_host);

    std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
    for (int32_t i = 0; i < options.groups; ++i) {
      std::cout << "    " << options.problem_sizes_host.at(i);
      std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
    }
    std::cout << "  Groups      : " << options.groups  << std::endl;
    std::cout << "  Avg runtime : " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS      : " << result.gflops << std::endl;
  }

  return 0;
}



///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

 
  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate HYTLASS kernels
  //


  run<Gemm>(options);
  run<Gemm>(options, false /*host_problem_shapes_available*/);


  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

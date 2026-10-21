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
/*! \file
    \brief Tests for device-wide GEMM interface
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hute/tensor.hpp"
#include "hute/atom/mma_atom.hpp"

#include "hytlass/numeric_types.h"
#include "hytlass/matrix_coord.h"

#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/gemm/group_array_problem_shape.hpp"
#include "hytlass/gemm/device/gemm_universal_adapter.h"
#include "hytlass/gemm/kernel/gemm_universal.hpp"
#include "hytlass/gemm/collective/collective_builder.hpp"

#include "hytlass/epilogue/collective/default_epilogue.hpp"
#include "hytlass/epilogue/collective/collective_builder.hpp"
#include "hytlass/epilogue/thread/linear_combination.h"

#include "hytlass/util/command_line.h"
#include "hytlass/util/packed_stride.hpp"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/device/gemm.h"
#include "hytlass/util/reference/device/tensor_compare.h"
#include "hytlass/util/reference/device/tensor_fill.h"

#include "helper.h"

using namespace hute;
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = hytlass::half_t;
using LayoutA = hytlass::layout::ColumnMajor;
constexpr int AlignmentA = 128 / hytlass::sizeof_bits<ElementA>::value; 

using ElementB = hytlass::half_t;
using LayoutB = hytlass::layout::RowMajor;
constexpr int AlignmentB = 128 / hytlass::sizeof_bits<ElementB>::value;

using ElementC = hytlass::half_t;
using LayoutC = hytlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / hytlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ElementCompute = ElementAccumulator;

using TileShape_MNK = Shape<_128, _128, _32>;
using WarpShape_MNK = Shape<_2, _2, _1>;
using InstructionShape_MNK = Shape<_32, _32, _16>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using StageCountType = hytlass::gemm::collective::StageCount<2>;  
// kernel to launch
using KernelScheduleType = hytlass::gemm::KernelPtrArraySpecialized;
using EpilogueTileType = hytlass::epilogue::collective::EpilogueTileAuto;
// epilogue to launch
using EpilogueScheduleType = hytlass::epilogue::PtrArrayNoSmemWarpSpecialized; 
using TileSchedulerType = hytlass::gemm::PersistentScheduler;

using CollectiveMainloop = typename hytlass::gemm::collective::CollectiveBuilder<
    hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_MNK, WarpShape_MNK, InstructionShape_MNK, ClusterShape_MNK,
    StageCountType, KernelScheduleType
  >::CollectiveOp;

using CollectiveEpilogue = typename hytlass::epilogue::collective::CollectiveBuilder<
    hytlass::arch::Gfx928, hytlass::arch::OpClassTensorOp,  
    TileShape_MNK, ClusterShape_MNK, 
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,   
    ElementC, LayoutC, AlignmentC,
    EpilogueScheduleType
  >::CollectiveOp;
  
using GemmKernel = hytlass::gemm::kernel::GemmUniversal<
    hytlass::gemm::ArrayProblemShape<Shape<int,int,int,int>>,
    CollectiveMainloop,
    CollectiveEpilogue,
    TileSchedulerType
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

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

std::vector<int64_t> offset_A;
std::vector<int64_t> offset_B;
std::vector<int64_t> offset_C;
std::vector<int64_t> offset_D;

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


/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help = false;

  float alpha = 1.0f;
  float beta = 0.0f;
  int iterations = 10;
  int m = 1024, n = 512, k = 1024, l = 10;

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
    cmd.get_cmd_line_argument("l", l);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "hute_ptr_array_batched_gemm\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --l=<int>                   Sets the batch count for Ptr-Array GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n";

    out
      << "\n\nExamples:\n\n"
      << "$ " << "hute_gfx928_ptr_array_batched_gemm" << " --m=1024 --n=512 --k=1024 --l=10 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k * l;
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
    scope_max = 2;
    scope_min = 0;
  } else if (bits_input <= 8) {
    scope_max = 2;
    scope_min = -2;
  } else {
    scope_max = 8;
    scope_min = -8;
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

  for (int32_t i = 0; i < options.l; ++i) {

    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = options.m * options.k;
    int64_t elements_B = options.k * options.n;
    int64_t elements_C = options.m * options.n;
    int64_t elements_D = options.m * options.n;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;
  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = hytlass::make_hute_packed_stride(StrideA{}, hute::make_shape(options.m, options.k, options.l));
  stride_B = hytlass::make_hute_packed_stride(StrideB{}, hute::make_shape(options.n, options.k, options.l));
  stride_C = hytlass::make_hute_packed_stride(StrideC{}, hute::make_shape(options.m, options.n, options.l));
  stride_D = hytlass::make_hute_packed_stride(StrideD{}, hute::make_shape(options.m, options.n, options.l));

  //
  // Assign pointers
  //

  std::vector<ElementA *> ptr_A_host(options.l);
  std::vector<ElementB *> ptr_B_host(options.l);
  std::vector<ElementC *> ptr_C_host(options.l);
  std::vector<ElementC *> ptr_D_host(options.l);

  for (int32_t i = 0; i < options.l; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
  }

  ptr_A.reset(options.l);
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.l);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.l);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.l);
  ptr_D.copy_from_host(ptr_D_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  hytlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count = hytlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  typename Gemm::Arguments arguments{
    hytlass::gemm::GemmUniversalMode::kArray,
    {{options.m, options.n, options.k, options.l}},
    {ptr_A.get(), stride_A, ptr_B.get(), stride_B},
    {{options.alpha, options.beta}, ptr_C.get(), stride_C, ptr_D.get(), stride_D},
    hw_info
  };

  return arguments;
}

bool verify(const Options &options) {
  bool passed = true;
  for (int32_t i = 0; i < options.l; ++i) {
    hytlass::TensorRef ref_A(block_A.get() + offset_A.at(i), Gemm::LayoutA::packed({options.m, options.k}));
    hytlass::TensorRef ref_B(block_B.get() + offset_B.at(i), Gemm::LayoutB::packed({options.k, options.n}));
    hytlass::TensorRef ref_C(block_C.get() + offset_C.at(i), Gemm::LayoutC::packed({options.m, options.n}));
    hytlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), Gemm::LayoutD::packed({options.m, options.n}));

    //
    // Compute reference output
    //

    // Create instantiation for device reference gemm kernel
    DeviceGemmReference gemm_reference;

    // Launch device reference gemm kernel
    gemm_reference(
      {options.m, options.n, options.k},
      ElementAccumulator(options.alpha),
      ref_A,
      ref_B,
      ElementAccumulator(options.beta),
      ref_C,
      ref_D);

    // Wait for kernel to finish
    HIP_CHECK(hipDeviceSynchronize());

    // Check if output from HYTLASS kernel and reference kernel are equal or not
    passed &= hytlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), options.m * options.n);
  }
  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  allocate(options);
  initialize(options);

  // Instantiate HYTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

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
    result.gflops          = options.gflops(result.avg_runtime_ms / 1000.0);

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Batches     : " << options.l  << std::endl;
    std::cout << "  Alpha, Beta : " << options.alpha << ',' << options.beta << std::endl;
    std::cout << "  Avg runtime : " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS      : " << result.gflops << std::endl;
  }

  return 0;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

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


  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

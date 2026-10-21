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
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/device/gett.hpp"

#include "helper.h"

using namespace hute;
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
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;

  Options():
    help(false),
    problem_size({8192, 8192, 2048}),
    batch_count(1),
    reference_check(false),
    iterations(10),
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

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("batch_count", batch_count);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "07_hute_batch_gemm example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/07_hute_batch_gemm/gfx928_batch_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 --batch_count=2 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds
    int64_t fmas = problem_size.product() * batch_count;

    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA = hytlass::half_t;
using LayoutA = hytlass::layout::ColumnMajor;
constexpr int AlignmentA = 128 / sizeof_bits_v<ElementA>;;

using ElementB = hytlass::half_t;
using LayoutB = hytlass::layout::RowMajor;
constexpr int AlignmentB = 128 / sizeof_bits_v<ElementB>;;

using ElementC = hytlass::half_t;
using LayoutC = hytlass::layout::ColumnMajor;
constexpr int AlignmentC = 128 / sizeof_bits_v<ElementC>;

using ElementAccumulator = float;
using ElementCompute = ElementAccumulator;

using TileShape_MNK = Shape<_128, _128, _32>;
using WarpShape_MNK = Shape<_2, _2, _1>;
using InstructionShape_MNK = Shape<_32, _32, _16>;
using ClusterShape_MNK = Shape<_1, _1, _1>;

using StageCountType = hytlass::gemm::collective::StageCount<2>;  
using KernelScheduleType = hytlass::gemm::KernelMultistage;
using EpilogueTileType = hytlass::epilogue::collective::EpilogueTileAuto;
using EpilogueScheduleType = hytlass::epilogue::collective::EpilogueScheduleAuto;

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
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = hytlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using ElementD = typename Gemm::GemmKernel::ElementD;
using StrideD = typename Gemm::GemmKernel::StrideD;
using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
using LayoutTagA = hytlass::detail::StrideToLayoutTagA_t<StrideA>;
using LayoutTagB = hytlass::detail::StrideToLayoutTagB_t<StrideB>;
using LayoutTagC = hytlass::detail::StrideToLayoutTagA_t<StrideC>;
using LayoutTagD = hytlass::detail::StrideToLayoutTagA_t<StrideD>;
using ElementScalar = typename EpilogueOutputOp::ElementScalar;

int run(Options &options) {
  int m = options.problem_size.m();
  int n = options.problem_size.n();
  int k = options.problem_size.k();
  int batch_count = options.batch_count;
  float alpha = options.alpha;
  float beta = options.beta;

  StrideA stride_a = hytlass::make_hute_packed_stride(StrideA{}, hute::make_shape(m, k, batch_count));
  StrideB stride_b = hytlass::make_hute_packed_stride(StrideB{}, hute::make_shape(n, k, batch_count));
  StrideC stride_c = hytlass::make_hute_packed_stride(StrideC{}, hute::make_shape(m, n, batch_count));
  StrideD stride_d = hytlass::make_hute_packed_stride(StrideD{}, hute::make_shape(m, n, batch_count));
  auto a_coord = hytlass::make_Coord(m * batch_count, k);
  auto b_coord = hytlass::make_Coord(k, n * batch_count);
  auto c_coord = hytlass::make_Coord(m * batch_count, n);
  
  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;

  hytlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  hytlass::HostTensor<ElementC, LayoutTagD> tensor_D;
  hytlass::HostTensor<ElementC, LayoutTagD> reference_D;

  ProblemShapeType problem_size = ProblemShapeType{m, n, k, batch_count};
  
  tensor_A.resize(a_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
  tensor_B.resize(b_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
  tensor_C.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
  tensor_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
  reference_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
  
  auto A = hute::make_tensor(tensor_A.host_data(),
    hute::make_layout(hute::make_shape(m, k, batch_count), stride_a));
  auto B = hute::make_tensor(tensor_B.host_data(),
    hute::make_layout(hute::make_shape(n, k, batch_count), stride_b));
  auto C = hute::make_tensor(tensor_C.host_data(),
    hute::make_layout(hute::make_shape(m, n, batch_count), stride_c));
  auto D = hute::make_tensor(reference_D.host_data(),
    hute::make_layout(hute::make_shape(m, n, batch_count), stride_d));

  hytlass::reference::host::TensorFillRandomUniform(
    tensor_A.host_view(),
    6118,
    ElementA(5),
    ElementA(-5),
    0);  // <- Fill matrix A on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
    tensor_B.host_view(),
    6117,
    ElementB(5),
    ElementB(-5),
    0);  // <- Fill matrix B on host with uniform-distribution random data
  hytlass::reference::host::TensorFillRandomUniform(
    tensor_C.host_view(),
    6116,
    ElementC(5),
    ElementC(-5),
    0);  // <- Fill matrix C on host with uniform-distribution random data
  hytlass::reference::host::TensorFill(
    tensor_D.host_view());  // <- fill matrix D on host with zeros
  hytlass::reference::host::TensorFill(
    reference_D.host_view());  // <- fill matrix D for reference on host with zeros

  hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();
  tensor_D.sync_device();

  auto arguments = typename Gemm::Arguments {
    hytlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    {
      tensor_A.device_data(), stride_a,
      tensor_B.device_data(), stride_b
    },
    {
      {hytlass::from_real<ElementScalar>(alpha), hytlass::from_real<ElementScalar>(beta)},
      tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
    }
  };
  
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;

  hytlass::Status status = gemm_op.can_implement(arguments);
  HYTLASS_CHECK(status);

  // Run the GEMM
  status = gemm_op.initialize(arguments, workspace.get());
  HYTLASS_CHECK(status);

  // Result structure
  Result result;

  // Launch initialized HYTLASS kernel
  status = gemm_op.run();
  (void)hipDeviceSynchronize();
  HYTLASS_CHECK(status);

  // verify
  hytlass::reference::device::gett<
      ProblemShapeType, ElementA, StrideA, ElementB, StrideB,
      ElementAccumulator, ElementC, StrideC, ElementD, StrideD, ElementScalar>(
      problem_size, tensor_A.device_data(), stride_a, tensor_B.device_data(),
      stride_b, ElementAccumulator(0), tensor_C.device_data(), stride_c,
      reference_D.device_data(), stride_d, alpha, beta);

  tensor_D.sync_host();
  reference_D.sync_host();
  
  result.passed = hytlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

  if (result.passed) {
    std::cout << "Reference check passed." << std::endl;
  }
  else {
    std::cerr << "Error - reference check failed." << std::endl;
  }

  // warmup
  for (int iter = 0; iter < 10; ++iter) {
    status = gemm_op(arguments, workspace.get());
  }
  (void)hipDeviceSynchronize();

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

  // Record an event at the start of a series of GEMMs
  result.error = hipEventRecord(events[0]);
  if (result.error != hipSuccess) {
    std::cerr << "hipEventRecord() failed: " << hipGetErrorString(result.error) << std::endl;
    return -1;
  }

  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized HYTLASS kernel
    status = gemm_op(arguments, workspace.get());
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

  std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
  std::cout << " GFLOPs: " << result.gflops << std::endl;

  return 0;
}

int main(int argc, const char **argv) {

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d  tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}

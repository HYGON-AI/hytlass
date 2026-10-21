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

#include <cmath>
#include <iostream>
#include <vector>
#include <limits>

#include "hytlass/hytlass.h"
#include "hytlass/arch/memory.h"
#include "hytlass/arch/memory_gfx928.h"

#include "hytlass/util/command_line.h"
#include "hytlass/util/host_tensor.h"

#include "hytlass/util/reference/host/gemm_complex.h"
#include "hytlass/util/reference/device/gemm_complex.h"
#include "hytlass/util/reference/host/tensor_reduce.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_norm.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/device/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/error_metrics.h"
#include "hytlass/util/tensor_view_io.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/epilogue/thread/linear_combination.h"
/////////////////////////////////////////////////////////////////////////////////////////////////

#include "gemm_with_softmax.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

#define TRACE(x) { std::cout << "gemm_softmax.cu:" << __LINE__ << "  " << x << std::endl; }

/////////////////////////////////////////////////////////////////////////////////////////////////

enum class Disposition {
  kPassed,
  kIncorrect,
  kNotVerified
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;
  hytlass::gemm::GemmCoord problem_size;
  int batch_count;
  int iterations;
  unsigned seed;
  float alpha;
  float beta;
  bool verification_enabled;
  float tolerance;

  Options():
    help(false),
    problem_size({16, 24, 64}),
    batch_count(16),
    iterations(20),
    seed(2022),
    alpha(1),
    beta(0),
    verification_enabled(true),
    tolerance(1e-5f)
  { }

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

    cmd.get_cmd_line_argument("batch_count", batch_count);

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);

    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("verify", verification_enabled);
    cmd.get_cmd_line_argument("seed", seed);
    cmd.get_cmd_line_argument("tolerance", tolerance);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "16_gemm_softmax example\n\n"
      << "  This example uses the HYTLASS Library to compute GEMM + Softmax for arbitrary problem sizes.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --batch_count=<int>         Batch number\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --seed=<int>                Random number seed (1*)\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform (0 to disable profiling).\n\n"
      << "  --verify=<bool>             If true, performs reference calculation.\n\n"
      << "  --tolerance <float>         Error tolerance\n"
    ;

    out << "\n\nExamples:\n\n"
      << "$ ./gemm_softmax --m=1024 --n=512 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Returns true if the environment and Toolkit support this
  bool supported(bool verbose = true) const {

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

struct Testbed {

  //
  // Type definitions
  //


  using ElementA = hytlass::half_t;
  using ElementB = hytlass::half_t;
  using ElementC = hytlass::half_t;
  using ElementCompute = float;
  using ElementD = ElementC;
  using ElementSoftmax = ElementC;

  using LayoutA = hytlass::layout::RowMajor;
  using LayoutB = hytlass::layout::ColumnMajor;

  using ThreadblockShape = hytlass::gemm::GemmShape<128, 128, 32>;
  using WarpShape        = hytlass::gemm::GemmShape<64, 64, 32>;
  using InstructionShape = hytlass::gemm::GemmShape<16, 16, 16>;

  using OperatorClass = hytlass::arch::OpClassTensorOp;
  using ArchTag = hytlass::arch::Gfx928;

  // ApplyShape impacts the final Softmax performance a lot.
  // Set ApplyShape::kColumn to be the next multiple of 32 number that is after
  // (gemm_N / alignment).
  // Set ApplyShape::kRow to max(1, 128 / ApplyShape::kColumn).
  using ApplyShape = hytlass::MatrixShape<1, 1024>;

  static int const kStages = 1;

  /// Linear scaling operator
  using EpilogueFunctorOp = hytlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / hytlass::sizeof_bits<ElementC>::value,
    ElementCompute,
    ElementCompute
  >;

  using GemmSoftmax = hytlass::GemmSoftmax<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueFunctorOp,
    kStages,
    ApplyShape
  >;

  using ElementNorm = typename GemmSoftmax::ElementNorm;
  using ElementSum = typename GemmSoftmax::ElementSum;
  using LayoutC = typename GemmSoftmax::LayoutC;
  using LayoutN = typename GemmSoftmax::LayoutN;
  using LayoutS = typename GemmSoftmax::LayoutS;
  using MatrixCoord = typename LayoutC::TensorCoord;

  //
  // Data members
  //

  Options const &options;


  hytlass::HostTensor<ElementNorm, LayoutC>     reference_N;

  hytlass::DeviceAllocation<ElementA> block_A;
  hytlass::DeviceAllocation<ElementB> block_B;
  hytlass::DeviceAllocation<ElementC> block_C;
  hytlass::DeviceAllocation<ElementD> block_D;
  hytlass::DeviceAllocation<ElementD> block_Ref;
  hytlass::DeviceAllocation<ElementSoftmax> block_Softmax;
  hytlass::DeviceAllocation<ElementNorm> block_Norm;
  hytlass::DeviceAllocation<ElementSum> block_Sum;

  int block_num = (options.problem_size.n() + GemmSoftmax::ThreadblockShape::kN - 1) / GemmSoftmax::ThreadblockShape::kN;

  hytlass::gemm::GemmCoord problem = options.problem_size;

  int64_t lda = LayoutA::packed({problem.m(), problem.k()}).stride(0);
  int64_t ldb = LayoutB::packed({problem.k(), problem.n()}).stride(0);
  int64_t ldc = LayoutC::packed({problem.m(), problem.n()}).stride(0);

  // fixed rowmajor for norm and sum
  int64_t ldn = problem.m();
  int64_t lds = ldn;

  int64_t total_elements_A_per_batch = problem.m() * problem.k();
  int64_t total_elements_B_per_batch = problem.k() * problem.n();
  int64_t total_elements_C_per_batch = problem.m() * problem.n();
  int64_t total_elements_D_per_batch = problem.m() * problem.n();
  int64_t total_elements_partial_norm_per_batch = block_num * problem.m();

  int64_t total_elements_A = total_elements_A_per_batch * options.batch_count;
  int64_t total_elements_B = total_elements_B_per_batch * options.batch_count;
  int64_t total_elements_C = total_elements_C_per_batch * options.batch_count;
  int64_t total_elements_D = total_elements_D_per_batch * options.batch_count;
  int64_t total_elements_partial_norm = total_elements_partial_norm_per_batch * options.batch_count;

  //
  // Methods
  //

  Testbed(
    Options const &options_
  ):
    options(options_)
  {
    reference_N.reset({options.problem_size.m(), 1}, false);
  }

  /// Run
  Disposition run() {

    Disposition disposition = Disposition::kNotVerified;

    //
    // Initialize the workspace
    //

    initialize();

    //
    // Launch device kernel
    //
    hytlass::Status status = hytlass::Status::kSuccess;

    status = execute_device_kernel();

    if (status != hytlass::Status::kSuccess) {
      std::cerr << "Device execution failed." << std::endl;
      return disposition;
    }

    hipError_t result = hipDeviceSynchronize();
    if (result != hipSuccess) {
      std::cerr << "Device synchronize failed with error "
        << hipGetErrorString(result) << std::endl;
      return disposition;
    }

    //
    // Verify
    //

    if (options.verification_enabled) {

      bool passed = verify();

      if (passed) {
        disposition = Disposition::kPassed;
      }
      else {
        disposition = Disposition::kIncorrect;
      }
    }

    //
    // Profiling
    //
    if (options.iterations) {
      profile();
    }

    return disposition;
  }

  /// Random initialization
  void initialize() {

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_Softmax.reset(total_elements_D);
    block_Ref.reset(total_elements_D_per_batch);
    block_Norm.reset(total_elements_partial_norm);
    block_Sum.reset(total_elements_partial_norm);

    hytlass::reference::device::BlockFillRandomUniform(
            block_A.get(), total_elements_A, options.seed, ElementA(5), ElementA(-5), 0);

    hytlass::reference::device::BlockFillRandomUniform(
            block_B.get(), total_elements_B, options.seed + 1, ElementB(5), ElementB(-5), 0);

    hytlass::reference::device::BlockFillRandomUniform(
            block_C.get(), total_elements_C, options.seed + 2, ElementC(5), ElementC(-5), 0);

    hytlass::reference::device::BlockFillRandomUniform(
            block_D.get(), total_elements_D, options.seed + 3, ElementD(5), ElementD(-5), 0);

    hytlass::reference::device::BlockFillRandomUniform(
            block_Ref.get(), total_elements_D_per_batch, options.seed + 3, ElementD(5), ElementD(-5), 0);

    hytlass::reference::device::BlockFillRandomUniform(
            block_Softmax.get(), total_elements_D, options.seed + 3, ElementSoftmax(5), ElementSoftmax(-5), 0);

    hytlass::reference::host::TensorFill(
      reference_N.host_view(),
      ElementNorm()
    );

  }

  hytlass::Status execute_device_kernel() {

    hytlass::Status status = hytlass::Status::kSuccess;

    //
    // Setup arguments
    //

    GemmSoftmax::Arguments args(
      options.problem_size,
      options.batch_count,
      {block_A.get(), lda},
      {block_B.get(), ldb},
      {block_C.get(), ldc},
      {block_D.get(), ldc},
      {
        ElementCompute(options.alpha),
        ElementCompute(options.beta)
      },
      {block_Norm.get(), ldn},
      {block_Sum.get(), lds},
      {block_Softmax.get(), ldc},
      total_elements_A_per_batch,
      total_elements_B_per_batch,
      total_elements_C_per_batch,
      total_elements_D_per_batch,
      total_elements_partial_norm_per_batch,
      total_elements_partial_norm_per_batch,
      total_elements_D_per_batch
    );

    //
    // Launch
    //

    GemmSoftmax gemm_softmax;

    // Initialize
    status = gemm_softmax.initialize(args);
    if (status != hytlass::Status::kSuccess) {
      return status;
    }

    // Run
    status = gemm_softmax();

    return status;
  }

  template<typename Element>
  bool verify_tensor(std::vector<Element> vector_Input, \
                       std::vector<Element> vector_Input_Ref) {

    auto size = int64_t((vector_Input.size() < vector_Input_Ref.size()) ? vector_Input.size() : vector_Input_Ref.size());
    float abs_tol = options.tolerance;
    float rel_tol = options.tolerance;
    
    for (int64_t i = 0; i < size; ++i) {
      float diff = (float)(vector_Input.at(i) - vector_Input_Ref.at(i));
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref.at(i));
      float relative_diff = abs_ref > abs_tol ? abs_diff / abs_ref : 0;
      if ( (isnan(abs_diff) || isinf(abs_diff)) ||  (abs_diff > rel_tol && relative_diff > rel_tol)) {
        printf("diff = %f, {%f, %f}.\n", abs_diff, (float)(vector_Input.at(i)), (float)(vector_Input_Ref.at(i)));
        return false;
      }

    }

    return true;
  }

  /// Verifies the reference matches
  bool verify() {

    LayoutA layout_A(lda);
    LayoutB layout_B(ldb);
    LayoutC layout_C(ldc);
    LayoutN Layout_N(ldn);
    LayoutS Layout_S(lds);

    MatrixCoord extent_A{problem.m(), problem.k()};
    MatrixCoord extent_B{problem.k(), problem.n()};
    MatrixCoord extent_C{problem.m(), problem.n()};

    for (int batch_idx = 0; batch_idx < options.batch_count; batch_idx++) {

      hytlass::TensorView<ElementA, LayoutA> view_A(block_A.get() + total_elements_A_per_batch * batch_idx, layout_A, extent_A);
      hytlass::TensorView<ElementB, LayoutB> view_B(block_B.get() + total_elements_B_per_batch * batch_idx, layout_B, extent_B);
      hytlass::TensorView<ElementC, LayoutC> view_C(block_C.get() + total_elements_C_per_batch * batch_idx, layout_C, extent_C);
      hytlass::TensorView<ElementC, LayoutC> view_Ref_device(block_Ref.get(), layout_C, extent_C);

      hytlass::reference::device::GemmComplex<
          ElementA, LayoutA,
          ElementB, LayoutB,
          ElementC, LayoutC, 
          ElementCompute, ElementCompute
      >(
        problem,
        options.alpha, 
        view_A,
        hytlass::ComplexTransform::kNone,
        view_B,
        hytlass::ComplexTransform::kNone,
        options.beta, 
        view_C, 
        view_Ref_device, 
        ElementCompute(0)
      );

      // Copy reference results to host memory for verification
      std::vector<ElementD> matrix_D_Ref(layout_C.capacity(extent_C));
      hytlass::device_memory::copy_to_host(matrix_D_Ref.data(), block_Ref.get(), matrix_D_Ref.size());
      hytlass::TensorView<ElementD, LayoutC> view_Ref(matrix_D_Ref.data(), layout_C, extent_C);

      std::vector<ElementSoftmax> matrix_Softmax_Ref(layout_C.capacity(extent_C));
      hytlass::TensorView<ElementSoftmax, LayoutC> view_Softmax_Ref(matrix_Softmax_Ref.data(), layout_C, extent_C);

      // Copy computed results to host memory
      std::vector<ElementD> matrix_D(layout_C.capacity(extent_C));
      hytlass::device_memory::copy_to_host(matrix_D.data(), block_D.get() + total_elements_D_per_batch * batch_idx, matrix_D.size());

      std::vector<ElementD> matrix_Softmax(layout_C.capacity(extent_C));
      hytlass::device_memory::copy_to_host(matrix_Softmax.data(), block_Softmax.get() + total_elements_D_per_batch * batch_idx, matrix_Softmax.size());

      // Compute the norm
      for (int m = 0; m < options.problem_size.m(); ++m) {
        reference_N.at({m, 0}) = view_Ref.ref().at({m, 0});
        for (int n = 1; n < options.problem_size.n(); ++n) {
          reference_N.at({m, 0}) = std::max(reference_N.at({m, 0}), ElementNorm(view_Ref.ref().at({m, n})));
        }
      }

      // Compute softmax
      for (int m = 0; m < options.problem_size.m(); ++m) {

        float sum = float();

        for (int n = 0; n < options.problem_size.n(); ++n) {
          sum += std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) );
        }

        float inv_sum = float(1.0f / sum);

        for (int n = 0; n < options.problem_size.n(); ++n) {

          view_Softmax_Ref.ref().at({m, n}) = ElementSoftmax(
            std::exp( float(view_Ref.ref().at({m, n})) - float(reference_N.at({m, 0})) ) * inv_sum
          );
        }
      }

      // Verification checks - set any of these to 'true' to override the verification checks.
      bool verified_D = true;
      bool verified_Softmax = true;

      // Verify softmax output
      if (!verified_D) {
        verified_D = verify_tensor<ElementC>(matrix_D, matrix_D_Ref);
      }

      if (!verified_Softmax) {
        verified_Softmax = verify_tensor<ElementSoftmax>(matrix_Softmax, matrix_Softmax_Ref);
      }

      if (!verified_D || !verified_Softmax) {

        std::cerr << "Verification check failed for tensor Softmax at batch " << batch_idx << "\n";

        // Summarize which checks failed
        if (!verified_D) {
          std::cerr << "Verification of D tensor failed\n";
        }

        if (!verified_Softmax) {
          std::cerr << "Verification of Softmax tensor failed\n";
        }

        return false;
      }

    }

    return true;
  }

  /// Profiles
  bool profile() {

    //
    // Profile
    //

    hytlass::Status status = hytlass::Status::kSuccess;
    hipError_t result;
    hipEvent_t events[2];
    int const kIterations = options.iterations;

    for (hipEvent_t &evt : events) {
      result = hipEventCreate(&evt);
      if (result != hipSuccess) {
        std::cerr << "hipEventCreate failed with error " << hipGetErrorString(result) << std::endl;
        return false;
      }
    }

    result = hipEventRecord(events[0]);

    if (result != hipSuccess) {
      std::cerr << "hipEventRecord() failed with error " << hipGetErrorString(result) << std::endl;
      return false;
    }

    for (int iter = 0; iter < kIterations; ++iter) {

      status = execute_device_kernel();

      if (status != hytlass::Status::kSuccess) {
        std::cerr << "Device execution failed." << std::endl;
        return false;
      }
    }

    result = hipEventRecord(events[1]);

    if (result != hipSuccess) {
      std::cerr << "hipEventRecord() failed with error " << hipGetErrorString(result) << std::endl;
      return false;
    }

    result = hipDeviceSynchronize();

    if (result != hipSuccess) {
      std::cerr << "hipDeviceSynchronize() failed with error " << hipGetErrorString(result) << std::endl;
      return false;
    }

    float elapsed_ms = 0;
    result = hipEventElapsedTime(&elapsed_ms, events[0], events[1]);

    if (result != hipSuccess) {
      std::cerr << "hipEventElapsedTime() failed with error " << hipGetErrorString(result) << std::endl;
      return false;
    }

    for (hipEvent_t &evt : events) {
      result = hipEventDestroy(evt);
      if (result != hipSuccess) {
        std::cerr << "hipEventDestroy() failed with error " << hipGetErrorString(result) << std::endl;
        return false;
      }
    }

    int64_t flops = int64_t(options.problem_size.m()) * options.problem_size.n() * options.problem_size.k() * 2;
    int64_t bytes = (sizeof(ElementD) * 2 + sizeof(ElementSoftmax)) * options.problem_size.m() * options.problem_size.n();

    double gflops_per_second = double(flops) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1.0e9);
    double gbytes_per_second = double(bytes) * kIterations * options.batch_count / double(elapsed_ms / 1000.0f) / double(1 << 30);

    double elapsed_ms_per_iter = double(elapsed_ms) / kIterations;

    std::cout << "         Problem: "
              << options.problem_size.m() << "-by-" << options.problem_size.n() << "-by-" << options.problem_size.k()
              << ", batch size: " << options.batch_count
              << std::endl;

    std::cout << "         Runtime: " << elapsed_ms_per_iter << " ms\n" << std::endl;

    std::cout << "          GFLOPs: " << gflops_per_second << "  GFLOPs" << std::endl;
    std::cout << "Memory bandwidth: " << gbytes_per_second << "  GiB/s" << std::endl;

    return true;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv) {

  // Options parsing
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (!options.supported()) {
    return 0;
  }

  // Run
  Testbed testbed(options);

  Disposition disposition = testbed.run();

  std::cout << std::endl;

  switch (disposition) {
    case Disposition::kPassed:
      std::cout << "Passed" << std::endl;
      break;
    case Disposition::kIncorrect:
      std::cout << "Incorrect" << std::endl;
      break;
    case Disposition::kNotVerified:
      std::cout << "Not verified" << std::endl;
      break;
  }

  return (disposition == Disposition::kPassed ? 0 : -1);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

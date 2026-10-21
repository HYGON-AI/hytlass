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
    \brief Example of running an scale AbsMax GEMM.

    In addition to using Fp16 Tensor Core instructions, the GEMM uses a distinct epilogue
    that enables additional scaling of operands/outputs, storing a pre-activation-function output
    tensor (called the "auxiliary" output), and computing the absolute maximum value of the
    outputs.

    Pseudocode for this epilogue is as follows:

    Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
    D = activation(Aux)

    if Aux is fp8 type:
        abs_max_output = max( abs(aux) | (for every aux in Aux))
        Aux = scale_aux * Aux
    endif

    if D is fp8 type:
        abs_max_output = max( abs(d) | (for every d in D))
        D = scale_d * D
    endif

    Parameter Aux is optionally stored to global memory
*/

#include <iostream>
#include <fstream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hytlass/numeric_conversion.h"
#include "hytlass/util/command_line.h"
#include "hytlass/util/host_tensor.h"
#include "hytlass/util/reference/host/gemm_complex.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/distribution.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_norm.h"
#include "hytlass/util/reference/host/gemm.h"

#include "hytlass/epilogue/thread/activation.h"
#include "hytlass/epilogue/thread/linear_combination_generic_with_scaling.h"
#include "hytlass/gemm/device/gemm_universal_with_absmax.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/matrix_coord.h"
#include "hytlass/gemm/device/gemm_universal_adapter.h"


using ElementA = hytlass::half_t;
using ElementB = hytlass::half_t;
using ElementOutput = hytlass::float_e4m3_t;
using ElementAuxOutput = ElementOutput;
using ElementAccumulator = float;
using LayoutA = hytlass::layout::RowMajor;
using LayoutB = hytlass::layout::ColumnMajor;
using LayoutC = hytlass::layout::RowMajor;
static int const kStages = 1;
static int const kAlignmentA = 8;
static int const kAlignmentB = 8;

using EpilogueOutputOp = hytlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
    hytlass::epilogue::thread::ReLu,
    ElementOutput,
    ElementAuxOutput,
    8,
    ElementAccumulator,
    ElementAccumulator
    >;

template <typename MathOperator>
using Gemm_ = hytlass::gemm::device::GemmUniversalWithAbsMax<
    ElementA, LayoutA, ElementB, LayoutB, ElementOutput, LayoutC,
    ElementAccumulator, hytlass::arch::OpClassTensorOp, hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 128>, hytlass::gemm::GemmShape<64, 32, 128>, hytlass::gemm::GemmShape<16, 16, 16>,
    EpilogueOutputOp, hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, kStages,
    kAlignmentA, kAlignmentB, MathOperator
  >;

using ElementAbsmax = typename EpilogueOutputOp::ElementAbsmax;


// Command line options parsing
struct Options {

  bool help;
  bool error;
  bool reference_check;
  hytlass::gemm::GemmCoord problem_size;

  int iterations;
  int warmup_iterations;

  bool scale_A;
  bool scale_B;
  bool scale_C;

  float alpha;
  float beta;

  Options():
    help(false),
    error(false),
    reference_check(false),
    iterations(20),
    warmup_iterations(5),
    scale_A(true),
    scale_B(true),
    scale_C(true),
    alpha(1.f),
    beta(0.f)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    hytlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("iterations", iterations, 20);
    cmd.get_cmd_line_argument("warmup_iterations", warmup_iterations, 5);
    cmd.get_cmd_line_argument("reference-check", reference_check, false);
    cmd.get_cmd_line_argument("scale-A", scale_A, true);
    cmd.get_cmd_line_argument("scale-B", scale_B, true);
    cmd.get_cmd_line_argument("scale-C", scale_C, true);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);

    int m, n, k;
    cmd.get_cmd_line_argument("m", m, 1024);
    cmd.get_cmd_line_argument("n", n, 1024);
    cmd.get_cmd_line_argument("k", k, 1024);

    problem_size = hytlass::gemm::GemmCoord{m, n, k};
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "18_gemm_with_abs_max\n\n"
      << "  This example executes a GEMM with AbsMax Tensor Core operations. In addition to performing\n"
      << "  a normal GEMM, the kernel performs the following operations:\n"
      << "      Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias\n"
      << "        D = activation(Aux)\n\n"
      << "      if Aux is fp8:\n"
      << "         abs_max_output = max( abs(aux) | (for every aux in Aux) )\n"
      << "         Aux = scale_aux * Aux\n\n"
      << "      if D is fp8 type:\n"
      << "         abs_max_output = max( abs(d) | (for every d in D) )\n"
      << "         D = scale_d * D\n\n"
      << "Options:\n\n"
      << "  --help                           If specified, displays this usage statement\n\n"
      << "  --m=<int>                        Sets the M dimension of the GEMM\n"
      << "  --n=<int>                        Sets the N dimension of the GEMM\n"
      << "  --k=<int>                        Sets the K dimension of the GEMM\n"
      << "  --scale-A=<bool>                 Whether to apply a scaling factor to operand A (default: true)\n"
      << "  --scale-B=<bool>                 Whether to apply a scaling factor to operand B (default: true)\n"
      << "  --scale-C=<bool>                 Whether to apply a scaling factor to operand C (default: true)\n"
      << "  --iterations=<int>               Number of profiling iterations to perform\n"
      << "  --warmup-iterations=<int>        Number of warmup iterations to perform\n"
      << "  --reference-check=<bool>         If true, performs reference check\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  float gflops(float runtime_s) const {
    // Two flops per multiply-add
    return 2.0f * float(problem_size.product()) / float(1.0e9) / runtime_s;
  }
};

/// Helper class to run the kernel
template <typename Gemm>
struct TestbedRunner {

  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;
  using ElementScalingFactor = typename Gemm::EpilogueOutputOp::ElementScalingFactor;

  static bool const kScaleAux = Gemm::EpilogueOutputOp::kIsScalingAndAmaxAuxOutputNeeded;
  static bool const kScaleOutput = Gemm::EpilogueOutputOp::kIsScalingAndAmaxOutputNeeded;

  /// Initialization
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;

  hytlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  hytlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  hytlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> tensor_Aux;
  hytlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> tensor_D;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_Vector;
  hytlass::HostTensor<ElementAccumulator, typename Gemm::LayoutC> tmp_D;
  hytlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementOutput, typename Gemm::LayoutC> reference_D;
  hytlass::HostTensor<typename Gemm::EpilogueOutputOp::ElementAuxOutput, typename Gemm::LayoutC> reference_Aux;
  hytlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_A;
  hytlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_B;
  hytlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_C;
  hytlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_D;
  hytlass::HostTensor<ElementScalingFactor, typename Gemm::LayoutC> scale_Aux;
  hytlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_Aux;
  hytlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> abs_max_D;
  hytlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_Aux;
  hytlass::HostTensor<ElementAbsmax, typename Gemm::LayoutC> reference_abs_max_D;

  //
  // Methods
  //

  TestbedRunner(
    bool scaleA = true,
    bool scaleB = true,
    bool scaleC = true,
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  /// Helper to initialize scaling factors
  template <typename Element, typename Layout>
  bool initialize_scale_factor(hytlass::TensorView<Element, Layout> view, uint64_t seed, int bits=0) {
    hytlass::reference::host::TensorFillRandomUniform(view, seed, double(1.), double(0.), bits);
    return true;
  }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    hytlass::TensorView<Element, Layout> view,
    hytlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == hytlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = hytlass::sizeof_bits<Element>::value;
      int bits_output = hytlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      } else {
        scope_max = 8;
        scope_min = -8;
      }

      hytlass::reference::host::TensorFillRandomUniform(
        view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == hytlass::Distribution::Identity) {

      hytlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == hytlass::Distribution::Gaussian) {

      hytlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == hytlass::Distribution::Sequential) {

      hytlass::reference::host::BlockFillSequential(
        view.data(), view.capacity());
    }
    else {
      std::cerr << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(const Options& options) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(options.problem_size.mk());
    tensor_B.resize(options.problem_size.kn());
    tensor_C.resize(options.problem_size.mn());
    tensor_D.resize(options.problem_size.mn());
    tensor_Vector.resize({1, options.problem_size.n()});
    reference_D.resize(options.problem_size.mn(), false);
    tmp_D.resize(options.problem_size.mn(), false);

    initialize_tensor(tensor_A.host_view(), init_A, seed + 2019);
    initialize_tensor(tensor_B.host_view(), init_B, seed + 2018);
    initialize_tensor(tensor_C.host_view(), init_C, seed + 2017);
    initialize_tensor(tensor_Vector.host_view(), init_C, seed + 2020);

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    hytlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);
    tensor_Vector.host_view().at(origin) = typename Gemm::ElementC(1);

    hytlass::reference::host::TensorFill(tensor_D.host_view());
    hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_Vector.sync_device();

    int scale_bits = 2;
    if (options.scale_A) {
      scale_A.resize({1, 1});
      initialize_scale_factor(scale_A.host_view(), seed + 2021, scale_bits);
      scale_A.sync_device();
    }

    if (options.scale_B) {
      scale_B.resize({1, 1});
      initialize_scale_factor(scale_B.host_view(), seed + 2022, scale_bits);
      scale_B.sync_device();
    }

    if (options.scale_C) {
      scale_C.resize({1, 1});
      initialize_scale_factor(scale_C.host_view(), seed + 2023, scale_bits);
      scale_C.sync_device();
    }

    if (kScaleOutput) {
      scale_D.resize({1, 1});
      initialize_scale_factor(scale_D.host_view(), seed + 2024, scale_bits);
      scale_D.sync_device();

      abs_max_D.resize({1, 1});
      hytlass::reference::host::TensorFill(abs_max_D.host_view());
      abs_max_D.sync_device();

      reference_abs_max_D.resize({1, 1});
    }

    if (kScaleAux) {
      tensor_Aux.resize(options.problem_size.mn());
      hytlass::reference::host::TensorFill(tensor_Aux.host_view());
      tensor_Aux.sync_device();

      scale_Aux.resize({1, 1});
      initialize_scale_factor(scale_Aux.host_view(), seed + 2025, scale_bits);
      scale_Aux.sync_device();

      abs_max_Aux.resize({1, 1});
      hytlass::reference::host::TensorFill(abs_max_Aux.host_view());
      abs_max_Aux.sync_device();

      reference_Aux.resize(options.problem_size.mn(), false);
      reference_abs_max_Aux.resize({1, 1});
    }
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(const Options& options) {

    tensor_D.sync_host();

    bool passed = hytlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (kScaleAux) {
      tensor_Aux.sync_host();
      abs_max_Aux.sync_host();
      passed &= hytlass::reference::host::TensorEquals(reference_Aux.host_view(), tensor_Aux.host_view());
      passed &= hytlass::reference::host::TensorEquals(abs_max_Aux.host_view(), reference_abs_max_Aux.host_view());
    }

    if (kScaleOutput) {
      abs_max_D.sync_host();
      passed &= hytlass::reference::host::TensorEquals(abs_max_D.host_view(), reference_abs_max_D.host_view());
    }

    if (!passed) {
      std::cerr << "Reference check failed" << std::endl;

      std::string output_file = "testbed_with_amax_errors.txt";
      std::ofstream file(output_file);

      file
        << "problem: " << options.problem_size
        << ", alpha: " << options.alpha << ", beta: " << options.beta << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\nVector =\n" << tensor_Vector.host_view()
        << "\nScaleA = " << scale_A.host_view()
        << "\nScaleB = " << scale_B.host_view()
        << "\nScaleC = " << scale_C.host_view()
        << "\nScaleD = " << scale_D.host_view()
        << "\nScaleAux = " << scale_Aux.host_view()
        << "\n\nReference D =\n" << reference_D.host_view()
        << "\nComputed D =\n" << tensor_D.host_view();
      if (kScaleAux) {
        file
          << "\n\nReference Aux =\n" << reference_Aux.host_view()
          << "\nComputed Aux =\n" << tensor_Aux.host_view()
          << "\n\nReference Absmax Aux = " << reference_abs_max_Aux.host_view()
          << "\nComputed Absmax Aux = " << abs_max_Aux.host_view();
      }
      if (kScaleOutput) {
        file
          << "\n\nReference Absmax D = " << reference_abs_max_D.host_view()
          << "\nComputed Absmax D = " << abs_max_D.host_view();
      }

      std::cerr << "Dumped results to " << output_file << std::endl;

    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(const Options& options) {

    hytlass::Coord<2> origin(0);
    ElementCompute scaled_alpha = options.alpha;
    if (options.scale_A) {
      scaled_alpha *= scale_A.host_view().at(origin);
    }
    if (options.scale_B) {
      scaled_alpha *= scale_B.host_view().at(origin);
    }

    ElementCompute scaled_beta = options.beta;
    if (options.scale_C) {
      scaled_beta *= scale_C.host_view().at(origin);
    }

    //
    // Verify
    //

    hytlass::reference::host::GemmComplex<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC,
        ElementCompute, ElementAccumulator, ElementAccumulator
    >(
      options.problem_size,
      scaled_alpha,
      tensor_A.host_ref(),
      Gemm::kTransformA,
      tensor_B.host_ref(),
      Gemm::kTransformB,
      scaled_beta,
      tensor_C.host_ref(),
      tmp_D.host_ref(),
      ElementAccumulator(0)
    );

    ElementCompute tmp_abs_max_Aux(0.);
    ElementCompute tmp_abs_max_D(0.);

    hytlass::NumericConverter<ElementCompute, typename Gemm::ElementC> cvt_c_to_compute;
    hytlass::NumericConverter<ElementCompute, ElementAccumulator> cvt_accum_to_compute;
    hytlass::NumericConverter<ElementAccumulator, ElementCompute> cvt_compute_to_accum;
    hytlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementOutput, ElementCompute> cvt_compute_to_d;
    hytlass::NumericConverter<typename Gemm::EpilogueOutputOp::ElementAuxOutput, ElementCompute> cvt_compute_to_aux;

    hytlass::absolute_value_op<ElementCompute> abs;
    hytlass::maximum_with_nan_propogation<ElementCompute> max;
    hytlass::epilogue::thread::ReLu<ElementCompute> act;

    ElementScalingFactor d_scale = kScaleOutput ? scale_D.host_view().at(origin) : ElementScalingFactor(1.);

    for (int m = 0; m < options.problem_size.m(); ++m) {
      for (int n = 0; n < options.problem_size.n(); ++n) {
        ElementCompute intermediate = cvt_accum_to_compute(tmp_D.host_view().at({m, n}));
        ElementCompute bias = cvt_c_to_compute(tensor_Vector.host_view().at({0, n}));
        ElementCompute aux = intermediate + bias;
        ElementCompute d = act(aux);
        tmp_abs_max_Aux = max(abs(aux), tmp_abs_max_Aux);
        tmp_abs_max_D = max(abs(d), tmp_abs_max_D);
        reference_D.host_view().at({m, n}) = cvt_compute_to_d(d * d_scale);

        if (kScaleAux) {
          reference_Aux.host_view().at({m, n}) = cvt_compute_to_aux(aux * scale_Aux.host_view().at(origin));
        }
      }
    }

    if (kScaleAux) {
      reference_abs_max_Aux.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_Aux);
    }

    if (kScaleOutput) {
      reference_abs_max_D.host_view().at(origin) = cvt_compute_to_accum(tmp_abs_max_D);
    }

    return compare_reference(options);
  }

  /// Returns true if the hip device is sufficient to execute the kernel.
  bool sufficient() const {

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

    hipDeviceProp_t properties;
    int device_idx;
    hipError_t result = hipGetDevice(&device_idx);

    if (result != hipSuccess) {
      std::cerr << "hipGetDevice() failed with error: " << hipGetErrorString(result) << std::endl;
      return false;
    }

    result = hipGetDeviceProperties(&properties, device_idx);

    if (result != hipSuccess) {
      std::cerr << "hipGetDeviceProperties() failed with error: " << hipGetErrorString(result) << std::endl;
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(Options& options)
  {

    // Waive test if insufficient hip device
    if (!sufficient()) {
      std::cerr << "Insufficient resources to run the kernel." << std::endl;
      return false;
    }

    this->initialize(options);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::EpilogueOutputOp::Params::ActivationParams activation_params{
      ElementCompute(options.alpha),
      ElementCompute(options.beta)
    };
    typename Gemm::EpilogueOutputOp::Params epilogue_params{
      activation_params,
      scale_A.device_data(),
      scale_B.device_data(),
      scale_C.device_data(),
      scale_D.device_data(),
      scale_Aux.device_data(),
      abs_max_Aux.device_data(),
      abs_max_D.device_data()
    };

    typename Gemm::Arguments arguments{
      hytlass::gemm::GemmUniversalMode::kGemm,
      options.problem_size,
      /* batch_count = */ 1,
      epilogue_params,
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data(),
      tensor_D.device_data(),
      tensor_Aux.device_data(),
      tensor_Vector.device_data(),
      options.problem_size.m() * options.problem_size.k(),
      options.problem_size.n() * options.problem_size.k(),
      options.problem_size.m() * options.problem_size.n(),
      options.problem_size.m() * options.problem_size.n(),
      (int)options.problem_size.m(), // Batch stride vector
      tensor_A.layout().stride(0),
      tensor_B.layout().stride(0),
      tensor_C.layout().stride(0),
      tensor_D.layout().stride(0),
      (int64_t)0 // Leading dimension of vector. This must be 0
    };

    Gemm gemm_op;

    hytlass::Status status = gemm_op.can_implement(arguments);
    if (status != hytlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() failed" << std::endl;
      return false;
    }

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    status = gemm_op.initialize(arguments, workspace.get());
    if (status != hytlass::Status::kSuccess) {
      std::cerr << "Gemm::initialize() failed" << std::endl;
      return false;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    if (status != hytlass::Status::kSuccess) {
      std::cerr << "Gemm::run() failed" << std::endl;
      return false;
    }

    hipError_t hip_error = hipDeviceSynchronize();
    if (hip_error != hipSuccess) {
      std::cerr << "hip error: " << hipGetErrorString(hip_error) << std::endl;
      return false;
    }

    //
    // Verify
    //

    bool passed = true;
    if (options.reference_check) {
      passed &= this->verify(options);
    } else {
      std::cout << "Skipped reference check" << std::endl;
    }

    //
    // Warm up
    //

    for (int i = 0; i < options.warmup_iterations; ++i) {
      gemm_op();
    }

    //
    // Profile
    //

    hipEvent_t events[2];
    hipError_t error;
    for (auto & event : events) {
      error = hipEventCreate(&event);
      if (error != hipSuccess) {
        std::cerr << "hipEventCreate() failed: " << hipGetErrorString(error) << std::endl;
        return false;
      }
    }

    // Record an event at the start of a series of GEMM operations
    error = hipEventRecord(events[0]);
    if (error != hipSuccess) {
      std::cerr << "hipEventRecord() failed: " << hipGetErrorString(error) << std::endl;
      return false;
    }

    // Run profiling loop
    for (int iter = 0; iter < options.iterations; ++iter) {
      gemm_op();
    }

    // Record an event when the GEMM operations have been launched.
    error = hipEventRecord(events[1]);
    if (error != hipSuccess) {
      std::cerr << "hipEventRecord() failed: " << hipGetErrorString(error) << std::endl;
      return false;
    }

    // Wait for work on the device to complete.
    error = hipEventSynchronize(events[1]);
    if (error != hipSuccess) {
      std::cerr << "hipEventSynchronize() failed: " << hipGetErrorString(error) << std::endl;
      return false;
    }

    // Measure elapsed runtime
    float runtime_ms = 0;
    error = hipEventElapsedTime(&runtime_ms, events[0], events[1]);
    if (error != hipSuccess) {
      std::cerr << "hipEventElapsed() failed: " << hipGetErrorString(error) << std::endl;
      return false;
    }

    // Compute average runtime and GFLOPs.
    runtime_ms = runtime_ms / float(options.iterations);
    float gflops = options.gflops(runtime_ms / 1000.0f);

    std::cout << "Problem size: " << options.problem_size.m() << 'x' << options.problem_size.n() << 'x' << options.problem_size.k() << std::endl;
    std::cout << "Runtime (ms): " << runtime_ms << std::endl;
    std::cout << "GFLOPs/sec:   " << gflops << std::endl;

    // Cleanup
    for (auto event : events) {
      (void)hipEventDestroy(event);
    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const** argv) {

  hipDeviceProp_t props;

  hipError_t error = hipGetDeviceProperties(&props, 0);
  if (error != hipSuccess) {
    std::cerr << "hipGetDeviceProperties() returned an error: " << hipGetErrorString(error) << std::endl;
    return -1;
  }


  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  std::cout << "Running GEMM with staged accumulation (OpMultiplyAdd)" << std::endl;
  std::cout << "=====================================================" << std::endl;
  TestbedRunner<Gemm_<hytlass::arch::OpMultiplyAdd>> testbed_staged_accum;
  bool passed = testbed_staged_accum.run(options);

  if (passed) {
    std::cout << "Passed" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  return 0;
}

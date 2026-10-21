/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Implicit GEMM testbed
*/
#pragma once

#include <fstream>

#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"

#include "hytlass/conv/device/implicit_gemm_convolution.h"
#include "hytlass/reduction/device/tensor_reduce.h"
#include "hytlass/reduction/device/reduce_split_k.h"
#include "hytlass/reduction/thread/reduction_operators.h"

#include "conv2d_problems.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/device/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_compare.h"

#include "hytlass/util/reference/host/convolution.h"
#include "hytlass/util/reference/device/convolution.h"

#include "hytlass/core_io.h"
#include "hytlass/util/tensor_view_io.h"

#include "../cache_testbed_output.h"

namespace test {
namespace conv {
namespace device {

template <typename Conv2d>
class TestbedConv2dWithReduction {
public:

  using ElementA = typename Conv2d::ElementA;
  using LayoutA = typename Conv2d::LayoutA;
  using ElementB = typename Conv2d::ElementB;
  using LayoutB = typename Conv2d::LayoutB;
  using ElementC = typename Conv2d::ElementC;
  using LayoutC = typename Conv2d::LayoutC;
  using ElementAccumulator = typename Conv2d::ElementAccumulator;
  using ElementCompute = typename Conv2d::ElementCompute;
  using EpilogueOutputOp = typename Conv2d::EpilogueOutputOp;
  using ElementT = typename EpilogueOutputOp::ElementTensor;

  static hytlass::conv::Operator const kConvolutionalOperator = Conv2d::kConvolutionalOperator;

public:

  /// Initialization
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;

  hytlass::HostTensor<ElementA, LayoutA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutC> tensor_C;

  hytlass::HostTensor<ElementAccumulator, LayoutC> tensor_Reduction;
  hytlass::HostTensor<ElementT,           hytlass::layout::RowMajor> tensor_Tensor;
  hytlass::HostTensor<ElementAccumulator, LayoutC> tensor_Final_Reduction;

  hytlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  hytlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

public:

  TestbedConv2dWithReduction(
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) {

  }

    /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  void initialize_tensor(
    hytlass::TensorView<Element, Layout> view, 
    hytlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    if (dist_kind == hytlass::Distribution::Uniform) {

      int scope = 2;

      hytlass::reference::host::TensorFillRandomUniform(
        view, seed, scope, -scope, 0);
    } 
    else if (dist_kind == hytlass::Distribution::Identity) {

      hytlass::reference::host::TensorFillIdentity(view);
    } 
    else if (dist_kind == hytlass::Distribution::Gaussian) {

      hytlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == hytlass::Distribution::Sequential) {

      hytlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    } 
    else {
    }
  }

  void initialize(
    hytlass::conv::Conv2dProblemSize const &problem_size, uint64_t seed = 2019) {
        
    tensor_A.resize(implicit_gemm_tensor_a_extent(kConvolutionalOperator, problem_size));
    tensor_B.resize(implicit_gemm_tensor_b_extent(kConvolutionalOperator, problem_size));
    tensor_C.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));

    tensor_Reduction.resize({
      1,
      1,
      (problem_size.N * problem_size.P * problem_size.Q - 1 + Conv2d::ThreadblockShape::kM) / Conv2d::ThreadblockShape::kM,
      (problem_size.K)
    });

    tensor_Final_Reduction.resize({
      1,
      1,
      1,
      (problem_size.K)
    });

    tensor_Tensor.resize({(problem_size.N * problem_size.P * problem_size.Q), problem_size.K});

    tensor_D_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_D_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));

    initialize_tensor(tensor_A.host_view(), init_A, seed); 
    initialize_tensor(tensor_B.host_view(), init_B, seed * 17); 
    initialize_tensor(tensor_C.host_view(), init_C, seed * 39);
    
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();
    tensor_D_reference.sync_device();
  }

  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Conv2d::UnderlyingKernel::SharedStorage);

    hipDeviceProp_t properties;
    int device_idx;
    hipError_t result = hipGetDevice(&device_idx);

    if (result != hipSuccess) {
      throw std::runtime_error("hipGetDevice() API call failed.");
    }

    result = hipGetDeviceProperties(&properties, device_idx);

    if (result != hipSuccess) {
      throw std::runtime_error("hipGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlock < smem_size) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    hytlass::conv::Conv2dProblemSize const &problem_size,
    hytlass::conv::SplitKMode const &split_k_mode = hytlass::conv::SplitKMode::kSerial,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0)) {

    // Waive test if insufficient HIP device
    if (!sufficient()) {
      if (HYTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient HIP device." << std::endl;
      }
      return true;
    }

#if 0 //display conv2d problem size for debugging
    std::cout << problem_size << std::endl
              << "alpha, beta: (" << alpha << ", " << beta << ")" << std::endl
              << "split_k_mode: " << ((split_k_mode == hytlass::conv::SplitKMode::kSerial) ? "(serial)" : "(parallel)") << std::endl
              << std::endl;
#endif

    initialize(problem_size);

    // configure the operator
    Conv2d conv2d_op;

    typename Conv2d::Arguments conv2d_args(
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D_computed.device_ref(),
      {alpha, beta},
      split_k_mode,
      tensor_Reduction.device_data(),
      tensor_Tensor.device_data(),
      static_cast<int>(tensor_Reduction.stride()[0]),
      static_cast<int>(tensor_Tensor.stride()[0])
    );

    // find workspace requirement for parallel split-k reduction
    size_t workspace_size = Conv2d::get_workspace_size(conv2d_args);

    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    hytlass::Status status = conv2d_op.initialize(conv2d_args, workspace.get());

    if (status != hytlass::Status::kSuccess) {
      hipError_t error = hipGetLastError();
      std::cerr << "This test is not supported: " << hipGetErrorString(error) << "\n";
      return true;
    }

    // conv2d operation with parallel split-k-mode
    if (split_k_mode == hytlass::conv::SplitKMode::kParallel) {

      // conv2d output is written to workspace in global memory
      conv2d_args.ref_D.reset(reinterpret_cast<ElementC*>(workspace.get()));
      // accumulate mma for each cta in k-dimension (1.0 * A * B)
      conv2d_args.output_op = {ElementCompute(1), ElementCompute(0)}; 
      // update conv2d operator arguments
      status = conv2d_op.update(conv2d_args, workspace.get());
    }
    
    EXPECT_TRUE(status == hytlass::Status::kSuccess);
    if (status != hytlass::Status::kSuccess) {
      return false;
    }

    // run conv2d operator
    status = conv2d_op();
    
    EXPECT_TRUE(status == hytlass::Status::kSuccess);
    if (status != hytlass::Status::kSuccess) {
      return false;
    }

    bool passed = false;

    hipError_t result = hipDeviceSynchronize();
    EXPECT_EQ(result, hipSuccess) << " device reference error: " 
                                   << hipGetErrorString(result);

    // Final reduction over the partial reduction tensor
    using Functor = hytlass::plus<ElementAccumulator>;
    using TensorReduction = hytlass::reduction::device::TensorReduction<
      ElementAccumulator,
      ElementAccumulator,
      LayoutC, 
      Functor,
      8,
      ElementAccumulator
    >;

    TensorReduction reduction(tensor_Reduction.extent(), 2);

    hytlass::DeviceAllocation<uint8_t> reduction_device_workspace(reduction.workspace_size());

    status = reduction.reduce(
      tensor_Final_Reduction.device_ref(),
      tensor_Reduction.device_ref(),
      reduction_device_workspace.get(),
      ElementAccumulator());

    EXPECT_EQ(status, hytlass::Status::kSuccess);
    EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

    //
    // Reference check
    //

    tensor_D_computed.sync_host();

#if HYTLASS_CONV_TEST_UNIT_REFERENCE_DEVICE_ENABLED

    hytlass::reference::device::Conv2d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementCompute,
      ElementAccumulator 
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D_reference.device_ref(),
      alpha, 
      beta);

    // sync host (copy device data to host) for dumping error output in case of mismatches
    tensor_D_reference.sync_host();
    
#else 

    hytlass::reference::host::Conv2d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementCompute,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      tensor_C.host_ref(),
      tensor_D_reference.host_ref(),
      alpha, 
      beta);

#endif

      passed = hytlass::reference::host::TensorRelativelyEquals(
        tensor_D_computed.host_view(), 
        tensor_D_reference.host_view(),
        (ElementC)1e-3f, (ElementC)1e-4f
      );

    EXPECT_TRUE(passed);

    //
    // Reference check on reduction results
    //

    tensor_Reduction.sync_host();
    tensor_Final_Reduction.sync_host();

    // compute backwards for reduction results
    hytlass::HostTensor<ElementAccumulator, LayoutC> reference_Reduction;
    reference_Reduction.resize({
      1,
      1,
      1,
      (problem_size.K) 
    });

    for (int k = 0; k < problem_size.K; ++k) {
      ElementAccumulator reduced_value = ElementAccumulator();
      for (int n = 0; n < problem_size.N; ++n) {
        for (int p = 0; p < problem_size.P; ++p) {
          for (int q = 0; q < problem_size.Q; ++q) {
            reduced_value += tensor_D_reference.at({n, p, q, k});
          }
        }
      }
      reference_Reduction.at({0, 0, 0, k}) = reduced_value;
    }

    // passed = hytlass::reference::host::TensorEquals(
    //   tensor_Final_Reduction.host_view(),
    //   reference_Reduction.host_view()
    // );

    passed = hytlass::reference::host::TensorRelativelyEquals(
      tensor_Final_Reduction.host_view(), 
      reference_Reduction.host_view(),
      (ElementAccumulator)1e-3f, (ElementAccumulator)1e-4f
    );

    EXPECT_TRUE(passed);

    if (!passed) {
      std::stringstream fname;

      fname << "error_Conv2d_ImplicitGemm_device_"
        << (split_k_mode == hytlass::conv::SplitKMode::kSerial ? "serial_reduction_" : "parallel_reduction_")
        << (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kFprop ? "fprop_" :
            (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ? "dgrad_" : "wgrad_")) 
        << "nhwc_"
        << problem_size.N << "x"
        << problem_size.H << "x"
        << problem_size.W << "x"
        << problem_size.C 
        << "_krsc_"
        << problem_size.K << "x"
        << problem_size.R << "x"
        << problem_size.S << "x"
        << problem_size.C 
        << "_padding_" 
        << problem_size.pad_h << "x"
        << problem_size.pad_w 
        << "_stride_"  
        << problem_size.stride_h << "x"
        << problem_size.stride_w 
        << "_dilation_"
        << problem_size.dilation_h << "x"
        << problem_size.dilation_w << "_"
        << (problem_size.mode == hytlass::conv::Mode::kCrossCorrelation ? "xcorr_" : "conv_")
        << Conv2d::ThreadblockShape::kM << "x"  
        << Conv2d::ThreadblockShape::kN << "x"  
        << Conv2d::ThreadblockShape::kK << "_"
        << Conv2d::WarpShape::kM << "x"  
        << Conv2d::WarpShape::kN << "x"  
        << Conv2d::WarpShape::kK << ".txt";

      std::cout << fname.str() << std::endl;

      std::ofstream results(fname.str());

      results << problem_size << std::endl;

      results
        << "\nA:\n" << tensor_A.host_view() << "\n"
        << "\nB:\n" << tensor_B.host_view() << "\n"
        << "\nC:\n" << tensor_C.host_view() << "\n"
        << "\nD reference:\n" << tensor_D_reference.host_view() << "\n"
        << "\nD computed:\n" << tensor_D_computed.host_view() << "\n"
        << "\nreduction reference:\n" << reference_Reduction.host_view() << "\n"
        << "\nreduction computed:\n" << tensor_Reduction.host_view() << "\n";
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// TestAllConv: Runs hytlass::conv::device::ImplicitGemmConvolution operator and compares it with reference
// TestAllConv runs conv operator on default conv problem sizes from test::conv::device::TestbedConv2dProblemSizes
// Additionally, each conv2d test can provide conv problem sizes (conv_test_sizes) and blacklist of sizes 
// (conv_blacklist_sizes)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ImplicitGemm>
bool TestAllConv2dWithReduction(
  const Conv2dProblemVector & conv_test_sizes = Conv2dProblemVector(),
  const Conv2dProblemVector & conv_blacklist_sizes = Conv2dProblemVector()) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2dWithReduction<ImplicitGemm> testbed;

  //
  // Get conv problem sizes to run conv operator 
  //
  TestbedConv2dProblemSizes conv_problems(128/hytlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);

  // Vector of conv2d problem sizes to avoid duplicate runs
  Conv2dProblemVector conv_tested_sizes;

  Conv2dProblemVector const *problem_vectors[] = {
    &conv_test_sizes,                               // run user specified sizes
    &conv_problems.conv2d_default_sizes,            // run default
    &conv_problems.conv2d_resnet50_sizes,           // run resnet50 sizes
#if HYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED 
    &conv_problems.conv2d_rigorous_sizes,           // run large and rigorous sizes if enabled
#endif
  };

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for (Conv2dProblemVector const * problem_vector : problem_vectors) {

    //  Run conv testbed on default convolution sizes
    for(auto conv_problem : *problem_vector) {

      // Skip blacklist and avoid duplicate problem sizes
      if (std::find(conv_blacklist_sizes.begin(), conv_blacklist_sizes.end(), conv_problem) != conv_blacklist_sizes.end() ||
          std::find(conv_tested_sizes.begin(), conv_tested_sizes.end(), conv_problem) != conv_tested_sizes.end()) {
        continue;
      }

      //
      // Procedurally disable certain cases
      //
  
      // HYTLASS DGRAD's *unity* stride specialization only support stride {1, 1} 
      if ((ImplicitGemm::kConvolutionalOperator == 
            hytlass::conv::Operator::kDgrad) && 
          (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
            hytlass::conv::StrideSupport::kUnity)) {
        if (!((conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
          continue;
        }
      }

#if 0 // relax restrictions on analytic strided dgrad
      // HYTLASS DGRAD's *strided* specialization only support stride >= {2, 2} 
      if ((ImplicitGemm::kConvolutionalOperator == 
            hytlass::conv::Operator::kDgrad) && 
          (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
            hytlass::conv::StrideSupport::kStrided)) {
         if (((conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
           continue;
         }
      }
#endif
      
      //
      // Test
      //
      // push back tested problem size to avoid re-running duplicates
      conv_tested_sizes.push_back(conv_problem);

      // test mode = xcross
      passed = testbed.run(
        conv_problem,
        hytlass::conv::SplitKMode::kSerial);
    
      if (!passed) {
        return false;
      }
      
      // test mode = convolution
      passed = testbed.run(
        conv_problem.reset_mode(hytlass::conv::Mode::kConvolution),
        hytlass::conv::SplitKMode::kSerial);
    
      if (!passed) {
        return false;
      }
    }
  }

  // HYTLASS DGRAD's *strided* specialization does not support split-k mode 
  if ((ImplicitGemm::kConvolutionalOperator == 
          hytlass::conv::Operator::kDgrad) && 
      (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
        hytlass::conv::StrideSupport::kStrided)) {

    passed = testbed.run(
      hytlass::conv::Conv2dProblemSize(
      {1, 56, 56, 8},   // input size (NHWC)
      {8, 1, 1, 8},     // filter size (KRSC)
      {0, 0, 0, 0},     // padding (pad_h, _, pad_w, _)
      {2, 2},           // stride (stride_h, stride_w)
      {1, 1}),          // dilation (dilation_h, dilation_w)
      hytlass::conv::SplitKMode::kSerial,
      hytlass::from_real<typename ImplicitGemm::ElementCompute>(2.0), 
      hytlass::from_real<typename ImplicitGemm::ElementCompute>(2.0));

    if (!passed) {
      return false;
    }

    return passed;
  }

  // Sweep split-k-slice using serial and prallel reduction with non-unity alpha and non-zero beta for 
  // a single conv2d problem size. Convolution unit tests take a long time to run so only sweep parameters 
  // which are abolutely necessary to catch functional bugs. The below code does provide option to sweep
  // alpha and beta for local testing, but only runs one value for alpha and beta.
  hytlass::conv::Conv2dProblemSize conv2d_split_k_test_size (
      {1, 17, 11, 288},   // input size (NHWC)
      {160, 3, 3, 288},   // filter size (KRSC)
      {1, 1, 1, 1},       // padding (pad_h, _, pad_w, _)
      {1, 1},             // stride (stride_h, stride_w)
      {1, 1}              // dilation (dilation_h, dilation_w)
    );

  // Parallel SplitK is not tested.
  hytlass::conv::SplitKMode split_k_modes [] = {
    hytlass::conv::SplitKMode::kSerial,
  };

  int split_k_slices[] = {
    1, 2, 3, 4, 201
  };

  double problem_alpha[] = {
    2.0
  };

  double problem_beta[] = {
    2.0
  };

  for (auto split_k_mode : split_k_modes) {
    for (auto split_k_slice : split_k_slices) {
      for (auto alpha : problem_alpha) {
        for (auto beta : problem_beta) {

          passed = testbed.run(
            conv2d_split_k_test_size.reset_split_k_slices(split_k_slice),
            split_k_mode,
            hytlass::from_real<typename ImplicitGemm::ElementCompute>(alpha), 
            hytlass::from_real<typename ImplicitGemm::ElementCompute>(beta));

          if (!passed) {
            return false;
          }
        }
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace test

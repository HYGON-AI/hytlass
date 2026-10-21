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
    \brief Implicit GEMM for fused epilogue broadcast testbed

    Parallel split-k is not tested because we can just use regular conv kernel
    when we need to use parallel-splitk.  Broadcast can happen in the reduction
    kernel.
*/
#pragma once

#include <fstream>

#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"

#include "hytlass/conv/device/implicit_gemm_convolution.h"
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

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Conv2d>
struct Conv2dWithBroadcastReferenceOp {

  using OutputOp = typename Conv2d::EpilogueOutputOp;

  using ElementCompute = typename OutputOp::ElementCompute;
  using ElementZ = typename OutputOp::ElementZ;
  using ElementT = typename OutputOp::ElementT;

  typename OutputOp::BinaryOp binary_op;
  typename OutputOp::ElementwiseOp elementwise_op;

  Conv2dWithBroadcastReferenceOp() { }

  void operator()(ElementZ &Z, ElementT &T, ElementCompute conv2d, ElementCompute bias) {
    ElementCompute t_full = binary_op(conv2d, bias);
    T = ElementT(t_full);

    ElementCompute z_full = elementwise_op(t_full);
    Z = ElementZ(z_full);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Fused testbed
//
//  Y = CONV(AB, C)
//
//  T[n, p, q, k] = ReductionOp(Y[n, p, q, k], Broadcast[k])
//
//  Z[n, p, q, k] = Elementwise(T[n, p, q, k])
//

template <
  typename Conv2d,
  typename ReferenceOp,
  bool AddBroadcastFirst = false
>
class TestbedConv2dWithBroadcast {
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
  using ElementZ = typename EpilogueOutputOp::ElementZ;
  using ElementT = typename EpilogueOutputOp::ElementT;
  using ElementVector = typename EpilogueOutputOp::ElementVector;

  static hytlass::conv::Operator const kConvolutionalOperator = Conv2d::kConvolutionalOperator;
  static const bool kAddBroadcastFirst = AddBroadcastFirst;
  static const bool kStoreT = EpilogueOutputOp::kStoreT;

public:

  /// Initialization
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;

  hytlass::HostTensor<ElementA, LayoutA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutC> tensor_C;
  hytlass::HostTensor<ElementAccumulator, LayoutC> tensor_C_reference;
  hytlass::HostTensor<ElementZ, LayoutC> tensor_Z_computed;
  hytlass::HostTensor<ElementZ, LayoutC> tensor_Z_reference;
  hytlass::HostTensor<ElementT, LayoutC> tensor_T_computed;
  hytlass::HostTensor<ElementT, LayoutC> tensor_T_reference;
  hytlass::HostTensor<ElementAccumulator, LayoutC> tensor_Y_reference;
  hytlass::HostTensor<ElementVector, LayoutC> tensor_Broadcast;            // Input Broadcast

public:

  TestbedConv2dWithBroadcast(
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

      int scope;
      int bits = hytlass::sizeof_bits<Element>::value;

      if (bits <= 8) {
        scope = 2;
      }
      else if (bits == 16) {
        if (hytlass::sizeof_bits<ElementAccumulator>::value <= 16) {
          scope = 3;
        }
        else {
          scope = 5;
        }
      }
      else {
        scope = 8;
      }
      
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
    tensor_C_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Z_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Z_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_T_computed.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_T_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Y_reference.resize(implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size));
    tensor_Broadcast.resize({
      1,
      1,
      1,
      implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size).c(),
    });

    initialize_tensor(tensor_A.host_view(), init_A, seed); 
    initialize_tensor(tensor_B.host_view(), init_B, seed * 17); 
    initialize_tensor(tensor_C.host_view(), init_C, seed * 39);
    initialize_tensor(tensor_Broadcast.host_view(), init_C, seed * 39);
 
    for (int n = 0; n < tensor_C_reference.extent().n(); ++n) {
      for (int p = 0; p < tensor_C_reference.extent().h(); ++p) {
        for (int q = 0; q < tensor_C_reference.extent().w(); ++q) {
          for (int k = 0; k < tensor_C_reference.extent().c(); ++k) {
            tensor_C_reference.at({n, p, q, k}) = ElementAccumulator(tensor_C.at({n, p, q, k}));
          }
        }
      }
    }
   
    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_Broadcast.sync_device();
    tensor_C_reference.sync_device();
    tensor_Z_computed.sync_device();
    tensor_Z_reference.sync_device();
    tensor_T_computed.sync_device();
    tensor_T_reference.sync_device();
    tensor_Y_reference.sync_device();
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
    ElementCompute beta = ElementCompute(1)) {

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
      tensor_Z_computed.device_ref(),
      {alpha, beta},
      split_k_mode,
      tensor_Broadcast.device_data(),
      kStoreT ? tensor_T_computed.device_data() : nullptr,
      0,         // This must be zero
      implicit_gemm_tensor_c_extent(kConvolutionalOperator, problem_size).c()
    );

    // initialize the kernel 
    size_t workspace_size = Conv2d::get_workspace_size(conv2d_args);

    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    hytlass::Status status = conv2d_op.initialize(conv2d_args, workspace.get());

    if (status != hytlass::Status::kSuccess) {
      hipError_t error = hipGetLastError();
      std::cerr << "This test is not supported: " << hipGetErrorString(error) << "\n";
      return true;
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

    tensor_T_computed.sync_host();
    tensor_Z_computed.sync_host();

    //
    // Reference check
    //

    // When kAddBroadcastFirst is true, add bias on the host
    ElementCompute beta_ref = kAddBroadcastFirst ? ElementCompute(0) : beta;

#if HYTLASS_CONV_TEST_UNIT_REFERENCE_DEVICE_ENABLED

    hytlass::reference::device::Conv2d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementAccumulator,
      LayoutC,
      ElementAccumulator,
      ElementAccumulator 
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C_reference.device_ref(),
      tensor_Y_reference.device_ref(),
      alpha, 
      beta_ref);

    // sync host (copy device data to host) for dumping error output in case of mismatches
    tensor_Y_reference.sync_host();

    if (kConvolutionalOperator == hytlass::conv::Operator::kDeconv) {
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
        tensor_Y_reference.host_ref(),
        alpha,
        beta_ref);
    }
    
#else 

    hytlass::reference::host::Conv2d<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementAccumulator,
      LayoutC,
      ElementAccumulator,
      ElementAccumulator
    >(
      kConvolutionalOperator,
      problem_size,
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      tensor_C_reference.host_ref(),
      tensor_Y_reference.host_ref(),
      alpha, 
      beta_ref);

#endif
    ReferenceOp reference_op;

    // compute tensor Z and tensor T
    for (int n = 0; n < problem_size.N; ++n) {
      for (int p = 0; p < (kConvolutionalOperator == hytlass::conv::Operator::kFprop ? problem_size.P : problem_size.H); ++p) {
        for (int q = 0; q < (kConvolutionalOperator == hytlass::conv::Operator::kFprop ? problem_size.Q : problem_size.W); ++q) {
          for (int k = 0; k < (kConvolutionalOperator == hytlass::conv::Operator::kFprop ? problem_size.K : problem_size.C); ++k) {
  
            ElementZ z{};
            ElementT t{};
    
            ElementCompute accum = tensor_Y_reference.at({n, p, q, k});
	          ElementCompute bias = ElementCompute(tensor_Broadcast.at({0, 0, 0, k}));


            if (kAddBroadcastFirst) {
              reference_op(z, t, accum + bias,
                           beta * ElementCompute(tensor_C_reference.at({n, p, q, k})));
            } else {
              reference_op(z, t, accum, bias);
            }   
 
            tensor_Z_reference.at({n, p, q, k}) = z;
            tensor_T_reference.at({n, p, q, k}) = t;
          }
        }
      }
    }

    if (kStoreT) {
      passed = hytlass::reference::host::TensorRelativelyEquals(
        tensor_T_computed.host_view(), 
        tensor_T_reference.host_view(),
        (ElementT)1e-3, (ElementT)1e-4
      );

      EXPECT_TRUE(passed);
    }

    passed = hytlass::reference::host::TensorRelativelyEquals(
      tensor_Z_computed.host_view(), 
      tensor_Z_reference.host_view(),
      (ElementZ)1e-3, (ElementZ)(1e-4)
    );

    EXPECT_TRUE(passed);

    if (!passed) {
      std::stringstream fname;

      fname << "error_Conv2d_ImplicitGemm_device_"
        << (split_k_mode == hytlass::conv::SplitKMode::kSerial ? "serial_reduction_" : "parallel_reduction_")
        << (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kFprop ? "fprop_" :
            (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ? "dgrad_" :
              (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kDeconv ? "deconv_" : "wgrad_")))
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
        << "\nBroadcast:\n" << tensor_Broadcast.host_view() << "\n"
        << "\nY reference:\n" << tensor_Y_reference.host_view() << "\n"
        << "\nT reference:\n" << tensor_T_reference.host_view() << "\n"
        << "\nT computed:\n" << tensor_T_computed.host_view() << "\n"
        << "\nZ reference:\n" << tensor_Z_reference.host_view() << "\n"
        << "\nZ computed:\n" << tensor_Z_computed.host_view() << "\n";
    }

    return passed;
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ImplicitGemm,
          typename ReferenceOp = Conv2dWithBroadcastReferenceOp<ImplicitGemm>,
          bool AddBroadcastFirst = false>
bool TestSpecificConv2dWithBroadcast(
  const Conv2dProblemVector & problem_sizes) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2dWithBroadcast<ImplicitGemm, ReferenceOp, AddBroadcastFirst> testbed;

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for(auto conv_problem : problem_sizes) {

    //
    // Test
    //

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

  return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// TestAllConv: Runs hytlass::conv::device::ImplicitGemmConvolution operator and compares it with reference
// TestAllConv runs conv operator on default conv problem sizes from test::conv::device::TestbedConv2dProblemSizes
// Additionally, each conv2d test can provide conv problem sizes (conv_test_sizes) and blacklist of sizes 
// (conv_blacklist_sizes)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ImplicitGemm,
          typename ReferenceOp = Conv2dWithBroadcastReferenceOp<ImplicitGemm>,
          bool AddBroadcastFirst = false,
          bool TestSplitK = true 
>
bool TestAllConv2dWithBroadcast(
  const Conv2dProblemVector &conv_test_sizes = Conv2dProblemVector(),
  const Conv2dProblemVector &conv_blacklist_sizes = Conv2dProblemVector()) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2dWithBroadcast<ImplicitGemm, ReferenceOp, AddBroadcastFirst> testbed;

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
      if ((ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ||
            ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDeconv) && 
          (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
            hytlass::conv::StrideSupport::kUnity)) {
        if (!((conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
          continue;
        }
      }

#if 0 // relax restrictions on analytic strided dgrad
      // HYTLASS DGRAD's *strided* specialization only support stride >= {2, 2} 
      if ((ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ||
            ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDeconv) && 
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
  if ((ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ||
        ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDeconv) && 
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

  if (!TestSplitK)
    return passed;

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

  hytlass::conv::SplitKMode split_k_modes [] = {
    hytlass::conv::SplitKMode::kSerial
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

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
#include "hytlass/reduction/device/reduce_split_k.h"
#include "hytlass/reduction/thread/reduction_operators.h"

#include "conv2d_problems.h"
#include "hytlass/complex.h"

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
class TestbedConv2d {
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

  static hytlass::conv::Operator const kConvolutionalOperator = Conv2d::kConvolutionalOperator;

  /// Reduction kernel
  using ReductionOp = hytlass::reduction::thread::ReduceAdd<
    ElementAccumulator, 
    typename EpilogueOutputOp::ElementAccumulator,
    EpilogueOutputOp::kCount
  >;

  using ReductionKernel = hytlass::reduction::kernel::ReduceSplitK<
    hytlass::MatrixShape<2, 64 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  using ReductionDevice = hytlass::reduction::device::ReduceSplitK<ReductionKernel>;
  using ReductionStrideIndex = typename ReductionDevice::StrideIndex;

public:

  /// Initialization
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;

  hytlass::HostTensor<ElementA, LayoutA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutC> tensor_C;
  hytlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  hytlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

  int tested_problem_count;

public:

  TestbedConv2d(
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_), tested_problem_count(0) {

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

  void get_compare_args(ElementC& eps, ElementC& non_zero_floor) {
    if constexpr (std::is_same_v<ElementA, float>) {
      eps = ElementC(1e-6f);
      non_zero_floor = ElementC(1e-8f);
    } else if constexpr (std::is_same_v<ElementA, hytlass::tfloat32_t>) {
      eps = ElementC(0.05f);
      non_zero_floor = ElementC(1e-3f);
    } else if constexpr (std::is_same_v<ElementA, hytlass::half_t>) {
      eps = ElementC(1e-3f);
      non_zero_floor = ElementC(1e-4f);
    } else if constexpr (std::is_same_v<ElementA, hytlass::bfloat16_t>) {
      eps = ElementC(0.05f);
      non_zero_floor = ElementC(1e-3f);
    }
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

    // increment tested problem count run by the testbed
    tested_problem_count++;

#if 0 // display conv2d problem size for debugging
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
      split_k_mode
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
      std::cerr << "Failed to run." << std::endl;
      return false;
    }


    if (split_k_mode == hytlass::conv::SplitKMode::kParallel) {

      // configure parallel reduction operator 
      ReductionDevice reduction_op;

      typename ReductionDevice::Arguments reduction_args(
        hytlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, problem_size).mn(),
        problem_size.split_k_slices,
        hytlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, problem_size),
        {
          reinterpret_cast<ElementAccumulator*> (workspace.get()),
          ReductionStrideIndex(tensor_C.stride()[Conv2d::UnderlyingKernel::kTensorCStrideIdx])
        },
        {
          tensor_D_computed.device_data(),
          ReductionStrideIndex(tensor_C.stride()[Conv2d::UnderlyingKernel::kTensorCStrideIdx])
        },
        {
          tensor_C.device_data(),
          ReductionStrideIndex(tensor_C.stride()[Conv2d::UnderlyingKernel::kTensorCStrideIdx])
        },
        // apply alpha, beta to obtain the following equation alpha * ReduceAdd(A * B) + beta * C 
        {alpha, beta} 
      );

      status = reduction_op.initialize(reduction_args, nullptr);

      EXPECT_TRUE(status == hytlass::Status::kSuccess);
      if (status != hytlass::Status::kSuccess) {
        return false;
      }

      // run prallel reduction kernel
      status = reduction_op();

      EXPECT_TRUE(status == hytlass::Status::kSuccess);
      if (status != hytlass::Status::kSuccess) {
        return false;
      }
    }
    bool passed = false;

    hipError_t result = hipDeviceSynchronize();
    EXPECT_EQ(result, hipSuccess) << " device reference error: " 
                                   << hipGetErrorString(result);

    tensor_D_computed.sync_host();

    //
    // Reference check - support caching results
    //

    CachedTestKey cached_test_key = CreateCachedConv2dTestKey<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator,
        ElementCompute
      >(
        kConvolutionalOperator,
        problem_size, 
        alpha, 
        beta, 
        tensor_A.host_view(),
        tensor_B.host_view(),
        tensor_C.host_view()
      );

    //
    // Look for the cached key
    //

    bool cached_result_loaded = false;
    CachedTestResult cached_test_result;

    std::string conv2d_result_cache_name = 
      std::string("cached_results_") + HYTLASS_TARGET_NAME + ".txt";

    if (HYTLASS_TEST_ENABLE_CACHED_RESULTS) {

      CachedTestResultListing cached_results(conv2d_result_cache_name);

      auto cached = cached_results.find(cached_test_key);

      cached_result_loaded = cached.first;
      if (cached_result_loaded) {
        cached_test_result = cached.second;
      }
    }
    
    if (!cached_result_loaded) {

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
          tensor_D_reference.host_ref(),
          alpha,
          beta);
      }
    
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

      if (HYTLASS_TEST_ENABLE_CACHED_RESULTS) {

        cached_test_result.D = TensorHash(tensor_D_reference.host_view());

        CachedTestResultListing cached_results(conv2d_result_cache_name);

        cached_results.append(cached_test_key, cached_test_result);
        cached_results.write(conv2d_result_cache_name);
      }
    } // if (!cached_result_loaded)

    uint32_t tensor_D_hash = TensorHash(tensor_D_computed.host_view());

    if (HYTLASS_TEST_ENABLE_CACHED_RESULTS) {
      // Due to floating point errors, a failed hash verification 
      // does not necessarily indicate an incorrect result.
      passed = (tensor_D_hash == cached_test_result.D);

      EXPECT_EQ(tensor_D_hash, cached_test_result.D) 
        << "Hash-based comparison failed for key:" << "\n" << cached_test_key << "\n";
    }
    else {
      ElementC eps, non_zero_floor;

      if constexpr (std::is_same_v<ElementA, float> ||
                    std::is_same_v<ElementA, hytlass::tfloat32_t> ||
                    std::is_same_v<ElementA, hytlass::half_t> ||
                    std::is_same_v<ElementA, hytlass::bfloat16_t>) {
        get_compare_args(eps, non_zero_floor);
        passed = hytlass::reference::host::TensorRelativelyEquals(
            tensor_D_computed.host_view(),
            tensor_D_reference.host_view(),
            eps,
            non_zero_floor);

      } else {
        passed = hytlass::reference::host::TensorEquals(
            tensor_D_computed.host_view(),
            tensor_D_reference.host_view());
      }
    }

    EXPECT_TRUE(passed);

    std::stringstream ss_problem_size_text;
    ss_problem_size_text         << "nhwc_"
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
        << (problem_size.mode == hytlass::conv::Mode::kCrossCorrelation ? "xcorr_" : "conv_");

    if (!passed) {
      std::stringstream fname;

      fname << "error_Conv2d_ImplicitGemm_device_"
        << (split_k_mode == hytlass::conv::SplitKMode::kSerial ? "serial_reduction_" : "parallel_reduction_")
        << (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kFprop ? "fprop_" :
            (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ? "dgrad_" :
              (Conv2d::kConvolutionalOperator == hytlass::conv::Operator::kDeconv ? "deconv_" : "wgrad_")))
        << ss_problem_size_text.str()
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
        << "\nC:\n" << tensor_C.host_view() << "\n";

      results << "\nD reference (hash: " << cached_test_result.D << ")\n";

      if (!cached_result_loaded) {
        results
          << tensor_D_reference.host_view() << "\n";  
      }

      results
        << "\nD computed (hash: " << tensor_D_hash << ")\n" 
        << tensor_D_computed.host_view() << "\n";

    }

    return passed;
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename ImplicitGemm>
bool TestSpecificConv2d(
  const Conv2dProblemVector & problem_sizes) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2d<ImplicitGemm> testbed;

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
template <typename ImplicitGemm>
bool TestAllConv2d(
  const Conv2dProblemVector & conv_test_sizes = Conv2dProblemVector(),
  const Conv2dProblemVector & conv_blacklist_sizes = Conv2dProblemVector()) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2d<ImplicitGemm> testbed;

  //
  // Get conv problem sizes to run conv operator 
  //
  TestbedConv2dProblemSizes conv_problems(128/hytlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);

  // Vector of conv2d problem sizes to avoid duplicate runs
  Conv2dProblemVector conv_tested_sizes;

  // Vectors of Conv2dProblemVector (lenient/easiest to rigorous problem sizes)
  std::vector<Conv2dProblemVector> problem_vectors = {
    conv_test_sizes,                               // run user specified sizes
    conv_problems.conv2d_default_sizes,            // run default
    //conv_problems.conv2d_resnet50_sizes,         // run resnet50 sizes
#if HYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
    conv_problems.conv2d_rigorous_sizes,           // run large and rigorous sizes if enabled
#endif
  };

  // Flatten 2D problem_vectors into a 1D problem_sizes
  std::vector<hytlass::conv::Conv2dProblemSize> problem_sizes;
  for (auto problem_vector : problem_vectors) {
    for(auto conv_problem : problem_vector) {
      problem_sizes.push_back(conv_problem);
    }
  }  

  // If HYTLASS_UNIT_TEST_PROBLEM_COUNT is set reverse the order (rigorous to lenient) 
  // run the most rigorous problem size first
  if (HytlassUnitTestProblemCount()) {
    std::reverse(problem_sizes.begin(), problem_sizes.end());
  }

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for(auto conv_problem : problem_sizes) {

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

    // Fixed channels algorithm requires channel count to match access size
    if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFixedChannels) {
      if (conv_problem.C != ImplicitGemm::UnderlyingKernel::Mma::IteratorA::AccessType::kElements) {
        continue;
      }
    }

    // Few channels algorithm requires channel count to match access size
    if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFewChannels) {
      if (conv_problem.C % ImplicitGemm::UnderlyingKernel::Mma::IteratorA::AccessType::kElements) {
        continue;
      }
    }

    // HYTLASS DGRAD's *strided* stride specialization supports all stride {stride_h, stride_w} 
    // Although strided dgrad works for all stride combinations, we are only going 
    // to run strided dgrad for non-unity strides 
    if ((ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ||
          ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDeconv) &&
        (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
          hytlass::conv::StrideSupport::kStrided)) {
       if (((conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
         continue;
       }
    }
    
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

    // If HYTLASS_UNIT_TEST_PROBLEM_COUNT is set reduce the number of tested problem counts
    if (HytlassUnitTestProblemCount() && 
        testbed.tested_problem_count > HytlassUnitTestProblemCount()) {
      return true;
    }
  }

  // Small-channels convolution can't run here.
  if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFixedChannels) {

    return true;
  }

  // Small-channels convolution can't run here.
  if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFewChannels) {

    return true;
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
    hytlass::conv::SplitKMode::kSerial,
    hytlass::conv::SplitKMode::kParallel,
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

          // If HYTLASS_UNIT_TEST_PROBLEM_COUNT is set reduce the number of tested problem counts
          if (HytlassUnitTestProblemCount() && 
              testbed.tested_problem_count > HytlassUnitTestProblemCount()) {
            return true;
          }
        }
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// TestAllConv: Runs hytlass::conv::device::ImplicitGemmConvolution operator and compares it with reference
// TestAllConv runs conv operator on default conv problem sizes from test::conv::device::TestbedConv2dProblemSizes
// Additionally, each conv2d test can provide conv problem sizes (conv_test_sizes) and blacklist of sizes 
// (conv_blacklist_sizes)
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename ImplicitGemm>
bool TestWithoutReduceConv2d(
  const Conv2dProblemVector & conv_test_sizes = Conv2dProblemVector(),
  const Conv2dProblemVector & conv_blacklist_sizes = Conv2dProblemVector()) {

  bool passed = true;

  //
  // Testbed object
  //

  TestbedConv2d<ImplicitGemm> testbed;

  //
  // Get conv problem sizes to run conv operator 
  //
  TestbedConv2dProblemSizes conv_problems(128/hytlass::sizeof_bits<typename ImplicitGemm::ElementA>::value);

  // Vector of conv2d problem sizes to avoid duplicate runs
  Conv2dProblemVector conv_tested_sizes;

  // Vectors of Conv2dProblemVector (lenient/easiest to rigorous problem sizes)
  std::vector<Conv2dProblemVector> problem_vectors = {
    conv_test_sizes,                               // run user specified sizes
    conv_problems.conv2d_default_sizes,            // run default
    //conv_problems.conv2d_resnet50_sizes,         // run resnet50 sizes
#if HYTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED
    conv_problems.conv2d_rigorous_sizes,           // run large and rigorous sizes if enabled
#endif
  };

  // Flatten 2D problem_vectors into a 1D problem_sizes
  std::vector<hytlass::conv::Conv2dProblemSize> problem_sizes;
  for (auto problem_vector : problem_vectors) {
    for(auto conv_problem : problem_vector) {
      problem_sizes.push_back(conv_problem);
    }
  }  

  // If HYTLASS_UNIT_TEST_PROBLEM_COUNT is set reverse the order (rigorous to lenient) 
  // run the most rigorous problem size first
  if (HytlassUnitTestProblemCount()) {
    std::reverse(problem_sizes.begin(), problem_sizes.end());
  }

  // Sweep conv2d problem sizes (split-k-mode=kSerial, split-k-slice=1, alpha=1.0, beta=0.0)
  for(auto conv_problem : problem_sizes) {

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

    // Fixed channels algorithm requires channel count to match access size
    if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFixedChannels) {
      if (conv_problem.C != ImplicitGemm::UnderlyingKernel::Mma::IteratorA::AccessType::kElements) {
        continue;
      }
    }

    // Few channels algorithm requires channel count to match access size
    if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFewChannels) {
      if (conv_problem.C % ImplicitGemm::UnderlyingKernel::Mma::IteratorA::AccessType::kElements) {
        continue;
      }
    }

    // HYTLASS DGRAD's *strided* stride specialization supports all stride {stride_h, stride_w} 
    // Although strided dgrad works for all stride combinations, we are only going 
    // to run strided dgrad for non-unity strides 
    if ((ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDgrad ||
          ImplicitGemm::kConvolutionalOperator == hytlass::conv::Operator::kDeconv) &&
        (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kStrideSupport == 
          hytlass::conv::StrideSupport::kStrided)) {
       if (((conv_problem.stride_h == 1) && (conv_problem.stride_w == 1))) {
         continue;
       }
    }
    
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

    // If HYTLASS_UNIT_TEST_PROBLEM_COUNT is set reduce the number of tested problem counts
    if (HytlassUnitTestProblemCount() && 
        testbed.tested_problem_count > HytlassUnitTestProblemCount()) {
      return true;
    }
  }

  // Small-channels convolution can't run here.
  if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFixedChannels) {

    return true;
  }

  // Small-channels convolution can't run here.
  if (ImplicitGemm::UnderlyingKernel::Mma::IteratorA::kIteratorAlgorithm ==
        hytlass::conv::IteratorAlgorithm::kFewChannels) {

    return true;
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

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////


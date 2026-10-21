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
    \brief Tests for device-wide GEMM interface
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>

#include "../../common/hytlass_unit_test.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/distribution.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_norm.h"
#include "hytlass/util/reference/host/gemm.h"

#include "testbed_utils.h"
#include "testbed_universal.h"

#include "hytlass/layout/matrix.h"
#include "hytlass/matrix_coord.h"
#include "hytlass/gemm/device/gemm_universal_adapter.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu = false>
struct Testbed {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  typename Gemm::LayoutA::Stride stride_factor_A;
  typename Gemm::LayoutB::Stride stride_factor_B;
  typename Gemm::LayoutC::Stride stride_factor_C;
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;

  hytlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  hytlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;

  //
  // Methods
  //

  Testbed(
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    stride_factor_A(typename Gemm::LayoutA::Stride()),
    stride_factor_B(typename Gemm::LayoutB::Stride()),
    stride_factor_C(typename Gemm::LayoutC::Stride()),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  Testbed(
    typename Gemm::LayoutA::Stride stride_factor_A_,
    typename Gemm::LayoutB::Stride stride_factor_B_,
    typename Gemm::LayoutC::Stride stride_factor_C_,
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_C(stride_factor_C_),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

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
        scope_max = 1;
        scope_min = -1;
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
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(hytlass::gemm::GemmCoord problem_size) {
    //
    // Allocate the GEMM workspace
    //

    tensor_A.resize(problem_size.mk(), hytlass::layout::Affine2Layout_Factory<typename Gemm::LayoutA>::layout_factory(problem_size.mk(), stride_factor_A));
    tensor_B.resize(problem_size.kn(), hytlass::layout::Affine2Layout_Factory<typename Gemm::LayoutB>::layout_factory(problem_size.kn(), stride_factor_B));
    tensor_C.resize(problem_size.mn(), hytlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C));
    tensor_D.resize(problem_size.mn(), hytlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C));
    reference_D.resize(problem_size.mn(), hytlass::layout::Affine2Layout_Factory<typename Gemm::LayoutC>::layout_factory(problem_size.mn(), stride_factor_C), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = typename Gemm::ElementA(1);
    tensor_B.host_view().at({0, 0}) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(hytlass::make_Coord(0, 0)) = typename Gemm::ElementC(1);

    hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
    hytlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    tensor_D.sync_host();

    EXPECT_GT(hytlass::reference::host::TensorNorm(tensor_A.host_view()), 0);
    EXPECT_GT(hytlass::reference::host::TensorNorm(tensor_B.host_view()), 0);
    EXPECT_GT(hytlass::reference::host::TensorNorm(tensor_C.host_view()), 0);

    if (tensor_D.size() > 1)
      EXPECT_GT(hytlass::reference::host::TensorNorm(tensor_D.host_view()), 0);

    if (reference_D.size() > 1)
      EXPECT_GT(hytlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = hytlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed);

    if (!passed) {

      std::stringstream fname;

      fname << "error_Gemm_device_" 
        << problem_size.m() << "x"
        << problem_size.n() << "x"
        << problem_size.k() << "_"
        << Gemm::ThreadblockShape::kM << "x"  
        << Gemm::ThreadblockShape::kN << "x"  
        << Gemm::ThreadblockShape::kK << "_"
        << Gemm::WarpShape::kM << "x"  
        << Gemm::WarpShape::kN << "x"  
        << Gemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());

      file
        << "problem: " << problem_size 
        << ", alpha: " << alpha << ", beta: " << beta << "\n\n";

      file 
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
    hytlass::gemm::GemmCoord problem_size, 
    ElementCompute alpha, 
    ElementCompute beta) {

    //
    // Verify
    //
    
    hytlass::reference::host::Gemm<
        typename Gemm::ElementA, typename Gemm::LayoutA,
        typename Gemm::ElementB, typename Gemm::LayoutB,
        typename Gemm::ElementC, typename Gemm::LayoutC, ElementCompute,
        ElementAccumulator, typename Gemm::Operator>
        reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
      reference_D.host_ref(), 
      ElementAccumulator(0)
    );

    if (Relu) {
      for (int i = 0; i < problem_size.m(); ++i) {
        for (int j = 0; j < problem_size.n(); ++j) {
           reference_D.at(hytlass::MatrixCoord(i, j)) = 
                  ((ElementCompute)reference_D.at(hytlass::MatrixCoord(i, j)) < (ElementCompute)0)
                  ? (typename Gemm::ElementC)0
                  : reference_D.at(hytlass::MatrixCoord(i, j));
        }
      }
    }

    return compare_reference(problem_size, alpha, beta);
  }

	/// Determine if the HIP device is sufficient to run the kernel
  bool sufficient() const {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    size_t smem_size = sizeof(typename Gemm::GemmKernel::SharedStorage);

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
    hytlass::gemm::GemmCoord problem_size,
    int split_k_slices = 1,
    ElementCompute alpha = ElementCompute(1),
    ElementCompute beta = ElementCompute(0))
  {

    // Waive test if insufficient HIP device
    if (!sufficient()) {
      if (HYTLASS_TEST_UNIT_ENABLE_WARNINGS) {
        std::cerr << "Test waived due to insufficient HIP device." << std::endl;
      }
      return true;
    }

    this->initialize(problem_size);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::Arguments arguments{
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      {alpha, beta},
      split_k_slices
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    hytlass::Status status = gemm_op.initialize(arguments, workspace.get());

    if (status != hytlass::Status::kSuccess) {
      hipError_t error = hipGetLastError();
      std::cerr << "This test is not supported: " << hipGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == hytlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, alpha, beta);

    if (!passed) {
      std::cout << "Error with split_k_slices = " << split_k_slices << ", alpha: " << alpha << std::endl;
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu=false>
bool TestAllGemmBasic(
    const typename Gemm::LayoutA::Stride& stride_factor_A = typename Gemm::LayoutA::Stride(),
    const typename Gemm::LayoutB::Stride& stride_factor_B = typename Gemm::LayoutB::Stride(),
    const typename Gemm::LayoutC::Stride& stride_factor_C = typename Gemm::LayoutC::Stride()) {
  bool passed = true;

  int const kMinimumOperandElementSize = 
    std::min(
      int(hytlass::sizeof_bits<typename Gemm::ElementA>::value), 
      int(hytlass::sizeof_bits<typename Gemm::ElementB>::value));

  int const kAlignment = hytlass::platform::is_same<
                              typename Gemm::OperatorClass, 
                              hytlass::arch::OpClassSimt>::value ? 1 : 128 / kMinimumOperandElementSize;

  // int8_t gemm alignment constraints
  int const kAlignmentM = hytlass::platform::is_same<typename Gemm::OperatorClass, hytlass::arch::OpClassSimt>::value &&
                          hytlass::platform::is_same<typename Gemm::ElementA, int8_t>::value &&
                          hytlass::platform::is_same<typename Gemm::LayoutA, hytlass::layout::ColumnMajor>::value ? 4 : kAlignment;

  int const kAlignmentN = hytlass::platform::is_same<typename Gemm::OperatorClass, hytlass::arch::OpClassSimt>::value &&
                          hytlass::platform::is_same<typename Gemm::ElementB, int8_t>::value &&
                          hytlass::platform::is_same<typename Gemm::LayoutB, hytlass::layout::RowMajor>::value ? 4 : kAlignment;

  int const kAlignmentK = hytlass::platform::is_same<typename Gemm::OperatorClass, hytlass::arch::OpClassSimt>::value &&
                          hytlass::platform::is_same<typename Gemm::ElementA, int8_t>::value &&
                          hytlass::platform::is_same<typename Gemm::ElementB, int8_t>::value &&
                          (hytlass::platform::is_same<typename Gemm::LayoutA, hytlass::layout::RowMajor>::value ||
                          hytlass::platform::is_same<typename Gemm::LayoutB, hytlass::layout::ColumnMajor>::value) ? 4 : kAlignment;

  int problem_size_m[] = {kAlignmentM, 512 - 3 * kAlignmentM};

  int problem_size_n[] = {kAlignmentN, 512 - 2 * kAlignmentN};

  int problem_size_k[] = {
      kAlignmentK, Gemm::ThreadblockShape::kK * (Gemm::kStages + 1) - kAlignmentK};

  int split_k_slices[] = {
    1, 2, 3
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    2.0
  };

  Testbed<Gemm, Relu> testbed(stride_factor_A, stride_factor_B, stride_factor_C);

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {

          if (!Gemm::kSplitKSerial && split_k > 1) {
            continue;
          }

          if (split_k > 1 && k / Gemm::ThreadblockShape::kK < split_k) {
            continue;
          }

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {

              hytlass::gemm::GemmCoord problem_size(m, n, k);
              passed = testbed.run(
                problem_size, 
                split_k,
                hytlass::from_real<ElementCompute>(alpha), 
                hytlass::from_real<ElementCompute>(beta)
              );

              if (!passed) {
                return false;
              }
            }
          }
        }
      }
    }
  }

  return passed;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm, bool Relu=false>
bool TestAllGemm(
    const typename Gemm::LayoutA::Stride& stride_factor_A,
    const typename Gemm::LayoutB::Stride& stride_factor_B = typename Gemm::LayoutB::Stride(),
    const typename Gemm::LayoutC::Stride& stride_factor_C = typename Gemm::LayoutC::Stride())
{
  // Test basic GEMM with non-default stride factors
  return TestAllGemmBasic<Gemm, Relu>(stride_factor_A, stride_factor_B, stride_factor_C);
}

template <typename Gemm, bool Relu=false>
bool TestAllGemm()
{
#ifdef NDEBUG
  // Non-debug builds also test basic GEMM with default stride factors
  if (!TestAllGemmBasic<Gemm, Relu>()) {
    return false;
  }
#endif // NDEBUG

  // Test universal GEMM
#if 1
  // Define the universal kernel
  using UniversalKernel = hytlass::gemm::kernel::GemmUniversal<
    typename Gemm::GemmKernel::Mma,                                 // Mma
    typename Gemm::GemmKernel::Epilogue,                            // Epilogue
    hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>    // ThreadblockSwizzle
  >;
#else 
  // Define the streamk universal kernel
  using UniversalKernel = hytlass::gemm::kernel::GemmUniversalStreamk<
    typename Gemm::GemmKernel::Mma,                                 // Mma
    typename Gemm::GemmKernel::Epilogue,                            // Epilogue
    hytlass::gemm::threadblock::ThreadblockSwizzleStreamK           // ThreadblockSwizzle
  >;
#endif

  // Define the universal adaptor
  using UniversalGemm = hytlass::gemm::device::GemmUniversalAdapter<UniversalKernel>;

  // Test universal GEMM
  return TestAllGemmUniversal<UniversalGemm, Relu>();
}

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////


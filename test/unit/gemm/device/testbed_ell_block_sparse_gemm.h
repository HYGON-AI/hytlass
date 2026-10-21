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
    \brief Tests for device-wide ELL block sparse GEMM interface
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
#include "hytlass/util/reference/host/gemm_complex.h"
#include "hytlass/gemm/device/ell_gemm.h"
#include "testbed_utils.h"
#include "hytlass/util/host_uncompress.h"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
struct TestbedEllBlockSparseGemm {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;
  using ElementAccumulator = typename Gemm::ElementAccumulator;
  using ElementCompute = typename Gemm::GemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  hytlass::Distribution::Kind init_ELL;
  uint64_t seed;

  hytlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A;
  hytlass::HostTensor<typename Gemm::ElementB, typename Gemm::LayoutB> tensor_B;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_C;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> tensor_D;
  hytlass::HostTensor<typename Gemm::ElementC, typename Gemm::LayoutC> reference_D;

  hytlass::HostTensor<typename Gemm::ElementA, typename Gemm::LayoutA> tensor_A_uncompressed;
  hytlass::HostTensor<int32_t, typename Gemm::LayoutA> tensor_ell_idx;

  //
  // Methods
  //

  TestbedEllBlockSparseGemm(
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_ELL_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = 2080
  ):
    init_A(init_A_), init_B(init_B_), init_C(init_C_), init_ELL(init_ELL_), seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
    hytlass::TensorView<Element, Layout> view,
    hytlass::Distribution::Kind dist_kind,
    uint64_t seed) {

    constexpr bool is_unsigned_int = std::is_unsigned<Element>::value;

    if (dist_kind == hytlass::Distribution::Uniform) {

      double scope_max, scope_min;
      int bits_input = hytlass::sizeof_bits<Element>::value;
      int bits_output = hytlass::sizeof_bits<typename Gemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      } else if (bits_input <= 8) {
        scope_max = is_unsigned_int ? 2 : 1;
        scope_min = is_unsigned_int ? 0 : -1;
      } else if (bits_output == 16) {
        constexpr auto u8_bf16 =
          (hytlass::platform::is_same<ElementA, uint8_t>::value &&
           hytlass::platform::is_same<ElementB, hytlass::bfloat16_t>::value) ||
          (hytlass::platform::is_same<ElementA, hytlass::bfloat16_t>::value &&
           hytlass::platform::is_same<ElementB, uint8_t>::value);
        scope_max = is_unsigned_int ? 10 : (u8_bf16 ? 3 : 5);
        scope_min = is_unsigned_int ? 0 : (u8_bf16 ? -3 : -5);
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
  void initialize(hytlass::gemm::GemmCoord problem_size, int a_ell_num_columns, int a_ell_blocksize) {
  
    tensor_A.resize(hytlass::make_Coord(problem_size.m(), a_ell_num_columns));
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);
    tensor_A_uncompressed.resize(problem_size.mk());
    tensor_ell_idx.resize(hytlass::make_Coord(problem_size.m() / a_ell_blocksize, a_ell_num_columns / a_ell_blocksize));

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    if (init_ELL == hytlass::Distribution::Uniform) {
      hytlass::reference::host::TensorFillRandomEllIdx(
          tensor_ell_idx.host_view(), seed,
          problem_size.m() / a_ell_blocksize,
          a_ell_num_columns / a_ell_blocksize,
          problem_size.k() / a_ell_blocksize);

    } else {
      for(int i = 0; i < problem_size.m() / a_ell_blocksize; ++i) {
        for(int j = 0; j < a_ell_num_columns / a_ell_blocksize; ++j) {
          tensor_ell_idx.at({i, j}) = j+3;
        }
      }
    }

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    hytlass::Coord<2> origin(0);
    tensor_A.host_view().at(origin) = typename Gemm::ElementA(1);
    tensor_B.host_view().at(origin) = typename Gemm::ElementB(1);
    tensor_C.host_view().at(origin) = typename Gemm::ElementC(1);

    hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
    tensor_ell_idx.sync_device();
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

    EXPECT_GT(hytlass::reference::host::TensorNorm(tensor_D.host_view()), 0);
    EXPECT_GT(hytlass::reference::host::TensorNorm(reference_D.host_view()), 0);

    bool passed = hytlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    EXPECT_TRUE(passed) << " mismatched reference";

    if (!passed) {

      std::ofstream file("testbed_ell_block_sparse_gemm_errors.txt");

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
    int a_ell_num_columns,
    int a_ell_blocksize,
    ElementCompute alpha,
    ElementCompute beta) {

    //
    // Verify
    //

    hytlass::uncompress_ell_block_sparse(
      tensor_A_uncompressed.host_ref(),
      tensor_A.host_ref(),
      tensor_ell_idx.host_ref(),
      problem_size.m(),
      problem_size.k(),
      a_ell_num_columns,
      a_ell_blocksize
    );

    hytlass::reference::host::Gemm<
      typename Gemm::ElementA, typename Gemm::LayoutA,
      typename Gemm::ElementB, typename Gemm::LayoutB,
      typename Gemm::ElementC, typename Gemm::LayoutC, ElementCompute,
      ElementAccumulator, typename Gemm::Operator>
      reference_gemm;

    reference_gemm(
      problem_size,
      alpha, 
      tensor_A_uncompressed.host_ref(), 
      tensor_B.host_ref(), 
      beta, 
      reference_D.host_ref(), 
      ElementAccumulator(0)
    );

    return compare_reference(problem_size, alpha, beta);
  }

  /// Returns true if the HIP device is sufficient to execute the kernel.
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
    int a_ell_num_columns,
    int a_ell_blocksize,
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

    this->initialize(problem_size, a_ell_num_columns, a_ell_blocksize);

    //
    // Initialize the GEMM operator
    //

    typename Gemm::EpilogueOutputOp::Params epilogue_op(alpha, beta);

    typename Gemm::Arguments arguments{
      problem_size,
      tensor_A.device_ref(),
      tensor_B.device_ref(),
      tensor_C.device_ref(),
      tensor_D.device_ref(),
      tensor_ell_idx.device_data(),
      a_ell_num_columns,
      a_ell_blocksize,
      0,
      epilogue_op
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);

    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    hytlass::Status status = gemm_op.initialize(arguments, workspace.get());

    EXPECT_TRUE(status == hytlass::Status::kSuccess) << to_string(status);

    //
    // Run the GEMM
    //

    status = gemm_op();

    EXPECT_TRUE(status == hytlass::Status::kSuccess) << to_string(status);

    //
    // Verify
    //

    bool passed = this->verify(problem_size, a_ell_num_columns, a_ell_blocksize, alpha, beta);

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Gemm>
bool TestAll() {

  bool passed = true;

  int problem_size_m[] = {
    256, 512
  };

  int problem_size_n[] = {
    128, 256
  };

  int problem_size_k[] = {
    512, 1024
  };

  int ell_blocksize[] = {
    32, 64
  };

  double problem_alpha[] = {
    1
  };

  double problem_beta[] = {
    0.0
  };

  using ElementCompute = typename Gemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for(int block : ell_blocksize) {
          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {

              hytlass::gemm::GemmCoord problem_size(m, n, 2048);
              TestbedEllBlockSparseGemm<Gemm> testbed;
              passed = testbed.run(
                problem_size,
                k,
                block,
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


template <typename Gemm>
bool TestAllEllBlockSparseGemm()
{
  // Test Ell block Sparse GEMM
  return TestAll<Gemm>();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

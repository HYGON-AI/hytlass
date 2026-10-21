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
    \brief Unit tests for thread-level GEMM
*/

#pragma once

#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include "hytlass/hytlass.h"
#include "hytlass/aligned_buffer.h"
#include "hytlass/numeric_types.h"
#include "hytlass/subbyte_reference.h"
#include "hytlass/platform/platform.h"
#include "hytlass/arch/arch.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"

#include "hytlass/util/distribution.h"
#include "hytlass/util/reference/host/gemm.h"
#include "hytlass/util/reference/host/gemm_complex.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/host_reorder.h"
#include "hytlass/util/host_uncompress.h"

namespace test {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Test kernel
template <typename Mma, typename ThreadblockShape>
__global__ void kernel(
  typename Mma::ElementC *output_C, 
  typename Mma::ElementA const *input_A,
  typename Mma::ElementB const *input_B,
  typename Mma::ElementC const *input_C,
  int iterations = 1) {

  // Use AlignedBuffer to store trivially copyable objects in unions and __shared__ buffers.
  __shared__ hytlass::AlignedBuffer<
    typename Mma::ElementA, ThreadblockShape::kM * ThreadblockShape::kK> smem_buffer_A;

  __shared__ hytlass::AlignedBuffer<
    typename Mma::ElementB, ThreadblockShape::kN * ThreadblockShape::kK> smem_buffer_B;

  if (threadIdx.x == 0) {
    typename Mma::ElementA *smem_ptr_A = smem_buffer_A.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_A.size(); ++i) {
      hytlass::ReferenceFactory<typename Mma::ElementA>::get(smem_ptr_A, i) =
          hytlass::ReferenceFactory<typename hytlass::platform::remove_const<
              typename Mma::ElementA>::type>::get(input_A, i);
    }

    typename Mma::ElementB *smem_ptr_B = smem_buffer_B.data();
    #pragma unroll 1
    for (size_t i = 0; i < smem_buffer_B.size(); ++i) {
      hytlass::ReferenceFactory<typename Mma::ElementB>::get(smem_ptr_B, i) =
          hytlass::ReferenceFactory<typename hytlass::platform::remove_const<
              typename Mma::ElementB>::type>::get(input_B, i);
    }
  }

  __syncthreads();
  __threadfence();

  //
  // Construct warp-level matrix product
  //

  using FragmentA = typename Mma::FragmentA;
  using FragmentB = typename Mma::FragmentB;
  using FragmentC = typename Mma::FragmentC;

  typename Mma::LayoutA layout_A = Mma::LayoutA::packed({ThreadblockShape::kM, ThreadblockShape::kK});
  typename Mma::LayoutB layout_B = Mma::LayoutB::packed({ThreadblockShape::kK, ThreadblockShape::kN});
  typename Mma::LayoutC layout_C = Mma::LayoutC::packed({Mma::Shape::kM, Mma::Shape::kN});

  typename Mma::IteratorA iter_A({smem_buffer_A.data(), layout_A}, hytlass::arch::LaneId());

  typename Mma::IteratorB iter_B({smem_buffer_B.data(), layout_B}, hytlass::arch::LaneId());

  FragmentA frag_A;
  FragmentB frag_B;

  FragmentC accum;

  Mma mma;

  accum.clear();

  HYTLASS_PRAGMA_NO_UNROLL
  for (int iter = 0; iter < iterations; ++iter) {     // place in loop that is not unrolled 

    HYTLASS_PRAGMA_UNROLL
    for (int k = 0; k < ThreadblockShape::kK;
         k += Mma::Policy::MmaShape::kK) {
      iter_A.load(frag_A);
      iter_B.load(frag_B);

      ++iter_A;
      ++iter_B;

      mma(accum, frag_A, frag_B, accum);
    }
  }
  
  typename Mma::IteratorC iter_C({output_C, layout_C}, hytlass::arch::LaneId());

  iter_C.store(accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Warp-level matrix multiply-accumulate
  typename Mma_,
  /// Size of threadblock-scoped shape used to store SMEM
  typename ThreadblockShape_,
  /// The inner product operation performed by GEMM 
  typename Operator_ = hytlass::arch::OpMultiplyAdd
>
struct Testbed {

  /// Thread-level matrix multiply-accumulate operator
  using Mma = Mma_;
  using ThreadblockShape = ThreadblockShape_;
  using Operator = Operator_;

  using Shape = typename Mma::Shape;
  using ElementA = typename Mma::ElementA;
  using LayoutA = typename Mma::LayoutA;
  using ElementB = typename Mma::ElementB;
  using LayoutB = typename Mma::LayoutB;
  using ElementC = typename Mma::ElementC;
  using LayoutC = typename Mma::LayoutC;

  //
  // Data members
  //

  hytlass::HostTensor<ElementA, LayoutA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutC> tensor_C;
  hytlass::HostTensor<ElementC, LayoutC> tensor_D_computed;
  hytlass::HostTensor<ElementC, LayoutC> tensor_D_reference;

  //
  // Methods
  //

  /// Allocates workspace in device memory
  Testbed() {

    tensor_A.reset(hytlass::make_Coord(ThreadblockShape::kM, ThreadblockShape::kK));
    tensor_B.reset(hytlass::make_Coord(ThreadblockShape::kK, ThreadblockShape::kN));
    tensor_C.reset(hytlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_computed.reset(hytlass::make_Coord(Shape::kM, Shape::kN));
    tensor_D_reference.reset(hytlass::make_Coord(Shape::kM, Shape::kN), false);
  }

  /// Returns true if the HIP device is sufficient to execute the kernel.
  bool sufficient() const {

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

    return true;
  }


  /// Runs the test
  bool run(
      hytlass::Distribution::Kind init_A = hytlass::Distribution::Uniform,
      hytlass::Distribution::Kind init_B = hytlass::Distribution::Uniform) {

    if (!sufficient()) {
      return true;
    }

    //
    // initialize device memory
    //

    if (init_A == hytlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (hytlass::sizeof_bits<ElementA>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (hytlass::sizeof_bits<ElementA>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;

      hytlass::reference::host::BlockFillRandomUniform(tensor_A.host_data(),
        tensor_A.capacity(), seed, scope_max, scope_min, 0);

    } else if (init_A == hytlass::Distribution::Sequential) {
      hytlass::reference::host::BlockFillSequential(tensor_A.host_data(),
                                                    tensor_A.capacity());
    } else if (init_A == hytlass::Distribution::Identity) {
      hytlass::reference::host::TensorFillIdentity(tensor_A.host_view());
    } else {
      return false;
    }

    if (init_B == hytlass::Distribution::Uniform) {
      int scope_max = 8;
      int scope_min = -8;

      if (hytlass::sizeof_bits<ElementB>::value == 4) {
        scope_max = 2;
        scope_min = -2;
      } else if (hytlass::sizeof_bits<ElementB>::value == 1) {
        scope_max = 2;
        scope_min = 0;
      }

      uint64_t seed = 7;

      hytlass::reference::host::BlockFillRandomUniform(tensor_B.host_data(),
        tensor_B.capacity(), seed, scope_max, scope_min, 0);

    } else if (init_B == hytlass::Distribution::Sequential) {
      hytlass::reference::host::BlockFillSequential(tensor_B.host_data(),
                                                    tensor_B.capacity());
    } else if (init_B == hytlass::Distribution::Identity) {
      hytlass::reference::host::TensorFillIdentity(tensor_B.host_view());
    } else {
      return false;
    }

    hytlass::reference::host::TensorFill(
      tensor_C.host_view(),
      ElementC(0)
    );

    hytlass::reference::host::TensorFill(
      tensor_D_computed.host_view(),
      ElementC(0)
    );

    hytlass::reference::host::TensorFill(
      tensor_D_reference.host_view(),
      ElementC(0)
    );

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D_computed.sync_device();

    // launch kernel
    kernel<Mma, ThreadblockShape><<< dim3(1, 1), dim3(WARP_SIZE_GPU, 1, 1) >>>(
      tensor_D_computed.device_data(),
      tensor_A.device_data(),
      tensor_B.device_data(),
      tensor_C.device_data());

    // verify no errors
    hipError_t result = hipDeviceSynchronize();

    EXPECT_EQ(result, hipSuccess) << "HIP ERROR: " << hipGetErrorString(result);
    if (result != hipSuccess) {
      return false;
    }

    tensor_D_computed.sync_host();

    //
    // Reference implementation
    //

    hytlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                   ElementC, LayoutC, ElementC, ElementC,
                                   Operator>
        reference_gemm;

    reference_gemm(
      {Shape::kM, Shape::kN, ThreadblockShape::kK},
      ElementC(1),
      tensor_A.host_ref(),
      tensor_B.host_ref(),
      ElementC(0),
      tensor_D_reference.host_ref()
    );

    //
    // Verify equivalence
    //

    // compare
    bool passed = hytlass::reference::host::TensorEquals(
      tensor_D_computed.host_view(),
      tensor_D_reference.host_view()
    );

    EXPECT_TRUE(passed);

    if (!passed) {
      
      hytlass::TensorView<ElementA, hytlass::layout::ColumnMajor> tensor_A_physical(
        tensor_A.host_data(), 
        tensor_A.stride()[0], 
        tensor_A.extent());

      hytlass::TensorView<ElementB, hytlass::layout::RowMajor> tensor_B_physical(
        tensor_B.host_data(), 
        tensor_B.stride()[0], 
        tensor_B.extent());

      std::ofstream file("warp_testbed_errors.txt");

      file <<"hytlass::sizeof_bits<ElementA>::value = "<<hytlass::sizeof_bits<ElementA>::value<<"\n";
      file
        << "A:\n" << tensor_A.host_view() << "\n\n"
        << "A(physical - stride: " << tensor_A.stride()[0] 
        << ", extent: " << tensor_A.extent() << "):\n" << tensor_A_physical << "\n\n";

      file <<"hytlass::sizeof_bits<ElementB>::value = "<<hytlass::sizeof_bits<ElementB>::value<<"\n";
      file
        << "B:\n" << tensor_B.host_view() << "\n\n"
        << "B(physical - stride: " << tensor_B.stride()[0] 
        << ", extent: " << tensor_B.extent() << "):\n" << tensor_B_physical << "\n\n";

      file
        << "C:\n" << tensor_C.host_view() << "\n\n"
        << "Reference:\n" << tensor_D_reference.host_view() << "\n\n"
        << "Computed:\n" << tensor_D_computed.host_view() << std::endl;
    }
    
    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace test

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

#include "hytlass_unit_test.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/tensor_view_io.h"
#include "hytlass/util/distribution.h"
#include "hytlass/util/packed_stride.hpp"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_norm.h"
#include "hytlass/util/reference/host/gett.hpp"
#include "hytlass/util/GPU_Clock.hpp"
#include "testbed_utils.h"

#include "hytlass/kernel_hardware_info.hpp"
#include "hytlass/layout/matrix.h"
#include "hytlass/matrix_coord.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/epilogue/fusion/operations.hpp"

#include "hute/int_tuple.hpp"

namespace test {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail{

// Helper classes that take default data type when 
// the Gemm::EpilogueOutputOp does not have ElementCompute
// and ElementScalar. 
template <typename Gemm, typename Default, typename = void>
struct ElementComputeType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementComputeType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementCompute>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementCompute;
};

template <typename Gemm, typename Default, typename = void>
struct ElementScalarType {
  using Type = Default;
};

template <typename Gemm, typename Default>
struct ElementScalarType<Gemm, Default, std::void_t<typename Gemm::EpilogueOutputOp::ElementScalar>> {
  using Type = typename Gemm::EpilogueOutputOp::ElementScalar;
};



// The number of splits to test.
//
// This class makes it harder to confuse the order of arguments
// of the various run(...) functions in this file.  The constructor
// is explicit, so one can't just type 42 (or false, which the
// compiler unhelpfully turns into 0); one has to type Splits(42).
// Splits() picks the default number of splits, 1.
//
// The conversion-to-int operator (operator int()) MUST be explicit!
// Conversion to int MUST require static_cast<int>.
// Otherwise, that defeats a key purpose of this class,
// which is to catch common errors of confusing the order
// of function arguments.
class Splits {
public:
  Splits() = default;

  template<class IntegralNotBool,
    __HUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !std::is_same_v<IntegralNotBool, bool>)) >
  explicit Splits(IntegralNotBool splits) : splits_(splits) {}
  explicit operator int() const { return splits_; }
private:
  int splits_ = 1;
};

// The number of iterations to test.
//
// This class, like Splits above makes it harder to confuse
// the order of arguments of the various run(...) functions in this file.
// Iterations() picks the default number of iterations, 20.
class Iterations {
public:
  Iterations() = default;

  template<class IntegralNotBool,
    __HUTE_REQUIRES((std::is_integral_v<IntegralNotBool> &&
      !std::is_same_v<IntegralNotBool, bool>)) >
  explicit Iterations(IntegralNotBool iterations) : iterations_(iterations) {}
  explicit operator int() const { return iterations_; }
private:
  int iterations_ = 20;
};

template <
  typename Gemm,
  template <class T> class ActivationFunctor_ = hytlass::epilogue::thread::Identity
>
struct TestbedImpl {
  // Kernel data types
  using ElementA = typename Gemm::GemmKernel::ElementA;
  using StrideA  = typename Gemm::GemmKernel::StrideA;
  using ElementB = typename Gemm::GemmKernel::ElementB;
  using StrideB  = typename Gemm::GemmKernel::StrideB;
  using ElementC = std::conditional_t<std::is_void_v<typename Gemm::GemmKernel::ElementC>,
                    typename Gemm::GemmKernel::ElementD,typename Gemm::GemmKernel::ElementC>;
  using StrideC  = typename Gemm::GemmKernel::StrideC;
  using ElementD = typename Gemm::GemmKernel::ElementD;
  using StrideD  = typename Gemm::GemmKernel::StrideD;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using EpilogueOutputOp = typename Gemm::EpilogueOutputOp;
  /// For custom EVTs
  using ElementCompute = typename ElementComputeType<Gemm, ElementAccumulator>::Type;
  using ElementScalar = typename ElementScalarType<Gemm, ElementCompute>::Type;
  using ActivationFunctor = ActivationFunctor_<ElementCompute>;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  static constexpr uint32_t mma_promotion_interval = 4;

  // Looks at Hute Stride to check Row / Column Major
  template<typename Stride>
  static constexpr bool is_row_or_col_major(){
    int stride_0 = int(hute::size<0>(Stride{}));
    int stride_1 = int(hute::size<1>(Stride{}));
    int depth = hute::depth(Stride{});
    return ((stride_0 == 1) || (stride_1 == 1)) && (depth == 1);
  }

  // Note: this limitation comes from testbed / not the library
  static_assert(is_row_or_col_major<StrideA>(),
    "ERROR : A Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideB>(),
    "ERROR : B Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideC>(),
    "ERROR : C Layout is neither Row / Column Major)");
  static_assert(is_row_or_col_major<StrideD>(),
    "ERROR : D Layout is neither Row / Column Major)");

  // Deduce Hytlass Layouts (RowMajor & ColumnMajor)
  using LayoutTagA = hytlass::detail::StrideToLayoutTagA_t<StrideA>;
  using LayoutTagB = hytlass::detail::StrideToLayoutTagB_t<StrideB>;
  using LayoutTagC = hytlass::detail::StrideToLayoutTagA_t<StrideC>;
  using LayoutTagD = hytlass::detail::StrideToLayoutTagA_t<StrideD>;

  /// Initialization
  StrideA stride_a;
  StrideB stride_b;
  StrideC stride_c;
  StrideD stride_d;
  typename LayoutTagA::Stride stride_factor_A;
  typename LayoutTagB::Stride stride_factor_B;
  typename LayoutTagC::Stride stride_factor_C;
  typename LayoutTagD::Stride stride_factor_D;
  hytlass::Distribution::Kind init_A;
  hytlass::Distribution::Kind init_B;
  hytlass::Distribution::Kind init_C;
  uint64_t seed;
  static constexpr uint64_t kDefaultSeed = 4096;

  hytlass::HostTensor<ElementA, LayoutTagA> tensor_A;
  hytlass::HostTensor<ElementB, LayoutTagB> tensor_B;
  hytlass::HostTensor<ElementC, LayoutTagC> tensor_C;
  hytlass::HostTensor<ElementD, LayoutTagD> tensor_D;
  hytlass::HostTensor<ElementD, LayoutTagD> reference_D;
  uint32_t sm_count;

  // Used to force multi-wave tests for persistent kernel schedules
  constexpr static int MaxSmCount = 16;
  //
  // Methods
  //

  TestbedImpl(
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(typename LayoutTagA::Stride()),
    stride_factor_B(typename LayoutTagB::Stride()),
    stride_factor_C(typename LayoutTagC::Stride()),
    stride_factor_D(typename LayoutTagD::Stride()),
    init_A(init_A_), init_B(init_B_), init_C(init_C_), seed(seed_) { }

  TestbedImpl(
    typename LayoutTagA::Stride stride_factor_A_,
    typename LayoutTagB::Stride stride_factor_B_,
    typename LayoutTagC::Stride stride_factor_C_,
    typename LayoutTagD::Stride stride_factor_D_,
    hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
    hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
    uint64_t seed_ = kDefaultSeed
  ):
    stride_factor_A(stride_factor_A_),
    stride_factor_B(stride_factor_B_),
    stride_factor_C(stride_factor_C_),
    stride_factor_D(stride_factor_D_),
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
      int bits_output = hytlass::sizeof_bits<ElementD>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      }
      else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      }
      else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      }
      else {
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

    else if (dist_kind == hytlass::Distribution::AllOnes) {
      hytlass::reference::host::TensorFill(view, Element(1));
    }

    else {
      return false;
    }

    return true;
  }

  /// Initializes data structures, this is batch Specialization
  void initialize(ProblemShapeType problem_size) {
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = hute::append<4>(problem_size, 1);
    auto M = hute::size<0>(problem_shape_MNKL);
    auto N = hute::size<1>(problem_shape_MNKL);
    auto K = hute::size<2>(problem_shape_MNKL);
    auto L = hute::size<3>(problem_shape_MNKL);

    stride_a = hytlass::make_hute_packed_stride(StrideA{}, hute::make_shape(M, K, L));
    stride_b = hytlass::make_hute_packed_stride(StrideB{}, hute::make_shape(N, K, L));
    stride_c = hytlass::make_hute_packed_stride(StrideC{}, hute::make_shape(M, N, L));
    stride_d = hytlass::make_hute_packed_stride(StrideD{}, hute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = hytlass::make_Coord(M * L, K);
    auto c_coord = hytlass::make_Coord(M * L, N);
    // Hytlass has Row/Col major refers to MxK times KxN matrix product, 
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = hytlass::make_Coord(K, N * L);
    

    tensor_A.resize(a_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_C.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);

    (initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    (initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));
    (initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Initializes data structures, this is splitk Specialization
  void initialize(ProblemShapeType problem_size, int slice_k) {
    //
    // Allocate the GEMM workspace
    //
    auto problem_shape_MNKL = hute::append<4>(problem_size, 1);
    auto M = hute::size<0>(problem_shape_MNKL);
    auto N = hute::size<1>(problem_shape_MNKL);
    auto K = hute::size<2>(problem_shape_MNKL);
    auto L = 1; // The L dimension stores slice_k, so split-K and batching cannot coexist; batch count must be 1.

    stride_a = hytlass::make_hute_packed_stride(StrideA{}, hute::make_shape(M, K, L));
    stride_b = hytlass::make_hute_packed_stride(StrideB{}, hute::make_shape(N, K, L));
    stride_c = hytlass::make_hute_packed_stride(StrideC{}, hute::make_shape(M, N, L));
    stride_d = hytlass::make_hute_packed_stride(StrideD{}, hute::make_shape(M, N, L));

    // 2.x host tensor does not natively contain a batch stride or coord, so we spoof if by folding it into the outer mode
    auto a_coord = hytlass::make_Coord(M * L, K);
    auto c_coord = hytlass::make_Coord(M * L, N);
    // Hytlass has Row/Col major refers to MxK times KxN matrix product, 
    // so the HostTensorB should be treated as KxN in "coord"'s view
    auto b_coord = hytlass::make_Coord(K, N * L);
    

    tensor_A.resize(a_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagA>::layout_factory(a_coord, stride_factor_A));
    tensor_B.resize(b_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagB>::layout_factory(b_coord, stride_factor_B));
    tensor_C.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagC>::layout_factory(c_coord, stride_factor_C));
    tensor_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D));
    reference_D.resize(c_coord, hytlass::layout::Affine2Layout_Factory<LayoutTagD>::layout_factory(c_coord, stride_factor_D), false);

    (initialize_tensor(tensor_A.host_view(), init_A, seed + 2022));
    (initialize_tensor(tensor_B.host_view(), init_B, seed + 2021));
    (initialize_tensor(tensor_C.host_view(), init_C, seed + 2020));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({0, 0}) = ElementA(1);
    tensor_B.host_view().at({0, 0}) = ElementB(1);
    tensor_C.host_view().at({0, 0}) = ElementC(1);

    hytlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      hute::Shape<int,int,int,int> problem_shape_MNKL,
      ElementScalar alpha,
      ElementScalar beta)
  {
    auto [M, N, K, L] = problem_shape_MNKL;

    tensor_D.sync_host();

    bool passed = hytlass::reference::host::TensorEquals(reference_D.host_view(), tensor_D.host_view());

    if (!passed) 
    {
      std::stringstream fname;
      fname << "error_Gemm_device_"
        << M << "x" << N << "x" << K << "x" << L << "_"
        << hute::get<0>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << hute::get<1>(typename Gemm::GemmKernel::TileShape{}) << "_"
        << hute::get<2>(typename Gemm::GemmKernel::TileShape{}) << ".csv";

      std::ofstream file(fname.str());
      file
        << "problem: " << ' ' << M << "x" << N << "x" << K << ", Batch count = " << L
        << ", alpha: " << float(alpha) << ", beta: " << float(beta) << "\n\n";

      file
        << "A =\n" << tensor_A.host_view()
        << "\nB =\n" << tensor_B.host_view()
        << "\nC =\n" << tensor_C.host_view()
        << "\n\nReference =\n" << reference_D.host_view()
        << "\n\nComputed =\n" << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      ProblemShapeType problem_size,
      ElementScalar alpha,
      ElementScalar beta) 
  {
    auto problem_shape_MNKL = hute::append<4>(problem_size, 1);
    auto M = hute::size<0>(problem_shape_MNKL);
    auto N = hute::size<1>(problem_shape_MNKL);
    auto K = hute::size<2>(problem_shape_MNKL);
    auto L = hute::size<3>(problem_shape_MNKL);

    auto A = hute::make_tensor(tensor_A.host_data(),
        hute::make_layout(hute::make_shape(M, K, L), stride_a));
    auto B = hute::make_tensor(tensor_B.host_data(),
        hute::make_layout(hute::make_shape(N, K, L), stride_b));
    auto C = hute::make_tensor(tensor_C.host_data(),
        hute::make_layout(hute::make_shape(M, N, L), stride_c));
    auto D = hute::make_tensor(reference_D.host_data(),
        hute::make_layout(hute::make_shape(M, N, L), stride_d));
    auto Bias = hute::make_tensor(static_cast<ElementCompute*>(nullptr),
        hute::make_layout(hute::make_shape(M, hute::_1{})));
    auto Aux = hute::make_tensor(static_cast<ElementD*>(nullptr),
        hute::make_layout(hute::make_shape(M, N, L), stride_d));
    auto Valpha = hute::make_tensor(static_cast<ElementCompute*>(nullptr),
        hute::make_layout(hute::make_shape(M, hute::_1{})));
    auto Vbeta = hute::make_tensor(static_cast<ElementCompute*>(nullptr),
        hute::make_layout(hute::make_shape(M, hute::_1{})));

    hytlass::reference::host::GettMainloopParams<ElementAccumulator, decltype(A), decltype(B)> mainloop_params{A, B};

    hytlass::reference::host::GettEpilogueParams<
        ElementScalar,
        ElementScalar,
        ElementAccumulator,
        ElementCompute,
        decltype(C),
        decltype(D),
        decltype(Bias),
        decltype(Aux),
        decltype(Valpha),
        decltype(Vbeta),
        ActivationFunctor
        >
        epilogue_params{
          alpha, beta,
          C, D, Bias, Aux
          , Valpha, Vbeta
        };

    hytlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    return compare_reference(problem_shape_MNKL, alpha, beta);
  }

	/// Determine if the GFX device is sufficient to run the kernel
  bool sufficient() {
    //
    // Determine SMEM requirements and waive if not satisfied
    //

    int smem_size = Gemm::GemmKernel::SharedStorageSize;

    int device_idx;
    hipError_t result = hipGetDevice(&device_idx);

    if (result != hipSuccess) {
      throw std::runtime_error("hipGetDevice() API call failed.");
    }

    hipDeviceProp_t properties;
    result = hipGetDeviceProperties(&properties, device_idx);
    this->sm_count = properties.multiProcessorCount;

    if (result != hipSuccess) {
      throw std::runtime_error("hipGetDeviceProperties() failed");
    }

    if (properties.sharedMemPerBlock < smem_size) {
      // return false;
    }

    return true;
  }

  bool profile(
    ProblemShapeType problem_size,
    int iterations,
    Gemm& gemm_op,
    typename Gemm::Arguments& arguments,
    hytlass::device_memory::allocation<uint8_t>& workspace) {
    int M = hute::size<0>(problem_size);
    int N = hute::size<1>(problem_size);
    int K = hute::size<2>(problem_size);
    int L = 1;
    if constexpr(hute::rank(ProblemShapeType{}) == 4) {
      L = hute::size<3>(problem_size);
    }


    hytlass::Status status;
    // warm-up
    for (int iter = 0; iter < 10; ++iter) {
      status = gemm_op(arguments, workspace.get());
    }
    (void)hipDeviceSynchronize();
    //
    // Run the GEMM
    //
    hipError_t result;
    GPU_Clock timer;
    timer.start();
    double gflops = (2.0*M*N*K) * 1e-9;
    for (int iter = 0; iter < iterations; ++iter) {
      status = gemm_op(arguments, workspace.get());
      if (status != hytlass::Status::kSuccess) {
        return false;
      }
    }

    result = hipDeviceSynchronize();
    double hute_time = timer.seconds() / iterations;
    HUTE_CHECK_LAST();
    printf("HUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / hute_time, hute_time*1000);
    if (result != hipSuccess) {
      return false;
    }

    return true;
  }

  /// Executes one test
  bool run(
    ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    bool profiling = false,
    detail::Iterations iterations = Iterations{},
    detail::Splits splits = Splits{})
  {
    // Fail test if insufficient GFX device
    if (!sufficient()) {
      std::cout << "Test failed due to insufficient GFX device." << std::endl;
      return false;
    }
    
    auto slice_k = hute::get<3>(problem_size);
    this->initialize(problem_size, slice_k);

    //
    // Initialize the GEMM operator
    //

    hytlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    if (not profiling) {
      this->sm_count = min(MaxSmCount, hytlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id));
      hw_info.sm_count = this->sm_count;
    }
    else {
      this->sm_count = hytlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
      hw_info.sm_count = this->sm_count;
    }

    typename Gemm::GemmKernel::TileScheduler::Arguments scheduler_args;
    if constexpr (std::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, hytlass::gemm::StreamKScheduler>) {
      scheduler_args = { static_cast<int>(splits) };
    }

    // DefaultEpilogue
    auto arguments = typename Gemm::Arguments {
      hytlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {
        tensor_A.device_data(), stride_a,
        tensor_B.device_data(), stride_b
      },
      {
        {alpha, beta},
        tensor_C.device_data(), stride_c, tensor_D.device_data(), stride_d
      },
      hw_info,
      scheduler_args
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    hytlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    hytlass::Status status = gemm_op.can_implement(arguments);

    if (status != hytlass::Status::kSuccess) {
      hipError_t error = hipGetLastError();
      std::cerr << "This test is not supported: " << hipGetErrorString(error) << "\n";
      return true;
    }

    //
    // Run the GEMM
    //

    if (profiling) {
      printf("first step: verify results\n");
      hipError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = hipDeviceSynchronize();
      if (result != hipSuccess) {
        printf("Error at Kernel Sync.\n");
        return false;
      }
      bool passed = this->verify(problem_size, alpha, beta);
      if (!passed) {
        printf("%s:%d\n",__FILE__,__LINE__);
        std::cout << "Error : Failed : with alpha: " << float(alpha) << ", beta: " << float(beta)
                  << "\n";
      }
      else
      {
        printf("%s:%d check passed\n",__FILE__,__LINE__);
      }
      return profile(problem_size, static_cast<int>(iterations), gemm_op, arguments, workspace);
    }
    else {
      hipError_t result;
      status = gemm_op.initialize(arguments, workspace.get());
      status = gemm_op.run();
      result = hipDeviceSynchronize();
      if (result != hipSuccess) {
        printf("Error at Kernel Sync.\n");
        return false;
      }

      printf("verify results\n");
      bool passed = this->verify(problem_size, alpha, beta);
      if (!passed) {
        printf("%s:%d\n",__FILE__,__LINE__);
        std::cout << "Error : Failed : with alpha: " << float(alpha) << ", beta: " << float(beta)
                  << "\n";
      }
      else
      {
        printf("%s:%d check passed\n",__FILE__,__LINE__);
      }

      return passed;
    }
  }
};

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  template <class T> class ActivationFunctor
>
struct Testbed3x {

  using TestBedImpl = typename detail::TestbedImpl<Gemm, ActivationFunctor>;
  using Kernel      = typename Gemm::GemmKernel;
  using Epilogue    = typename Gemm::GemmKernel::CollectiveEpilogue;

  using ElementAccumulator   = typename TestBedImpl::ElementAccumulator;
  using ElementCompute       = typename TestBedImpl::ElementCompute;
  using ElementScalar        = typename TestBedImpl::ElementScalar;

  using LayoutTagA = typename TestBedImpl::LayoutTagA;
  using LayoutTagB = typename TestBedImpl::LayoutTagB;
  using LayoutTagC = typename TestBedImpl::LayoutTagC;
  using LayoutTagD = typename TestBedImpl::LayoutTagD;

  // Detail Implementation
  TestBedImpl impl_;

  //
  // Methods
  //
  Testbed3x(
      hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
      hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
      hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
      uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(init_A_, init_B_, init_C_, seed_) {}

  Testbed3x(    
      typename LayoutTagA::Stride stride_factor_A_,
      typename LayoutTagB::Stride stride_factor_B_,
      typename LayoutTagC::Stride stride_factor_C_,
      typename LayoutTagD::Stride stride_factor_D_,
      hytlass::Distribution::Kind init_A_ = hytlass::Distribution::Uniform,
      hytlass::Distribution::Kind init_B_ = hytlass::Distribution::Uniform,
      hytlass::Distribution::Kind init_C_ = hytlass::Distribution::Uniform,
          uint64_t seed_ = TestBedImpl::kDefaultSeed)
      : impl_(stride_factor_A_,
              stride_factor_B_,
              stride_factor_C_,
              stride_factor_D_,
              init_A_,
              init_B_,
              init_C_,
              seed_) {}

  /// Executes one test
  bool run(
    typename TestBedImpl::ProblemShapeType problem_size,
    ElementScalar alpha = ElementScalar(1),
    ElementScalar beta = ElementScalar(0),
    detail::Splits splits = detail::Splits{},
    bool profiling = false,
    detail::Iterations iterations = detail::Iterations{})
  {
    return impl_.run(
        problem_size, alpha, beta, profiling, iterations, splits
        );
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Gemm,
  typename Testbed = Testbed3x<Gemm, hytlass::epilogue::thread::Identity>
>
bool TestAll(double alpha = 1.0, double beta = 0.0, Testbed testbed = {}) {
  using ElementScalar = typename Gemm::EpilogueOutputOp::ElementScalar;
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  int max_alignment = std::max(Gemm::kAlignmentA, Gemm::kAlignmentB);
  std::vector<int> problem_size_m = {256};
  std::vector<int> problem_size_n = {256};
  if constexpr (std::is_same_v<typename Gemm::GemmKernel::DispatchPolicy::Schedule,
                hytlass::gemm::KernelTmaWarpSpecializedPingpong>) {
    problem_size_m.push_back(768);
    problem_size_n.push_back(768);
  }

  constexpr int Stages = Gemm::GemmKernel::DispatchPolicy::Stages;
  constexpr int TileShapeK = hute::size<2>(typename Gemm::GemmKernel::TileShape{});

  std::vector<int> problem_size_k = {32};
  std::vector<int> problem_splits = {1};
  if constexpr (std::is_same_v<typename Gemm::GemmKernel::TileSchedulerTag, hytlass::gemm::StreamKScheduler>) {
    problem_splits.push_back(2);
    problem_splits.push_back(3);

    // As many splits as there are maximum k tiles
    problem_splits.push_back(Stages + 1);
  }

  bool passed = true;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int splits : problem_splits) {
          ProblemShapeType problem_size;
          if constexpr (hute::rank(ProblemShapeType{}) == 4) {
            problem_size = ProblemShapeType{m, n, k, /* l */ 1};
          }
          else {
            problem_size = ProblemShapeType{m, n, k};
          }
          printf("problem size:%d %d %d\n",m,n,k);
          passed = testbed.run(
            problem_size,
            hytlass::from_real<ElementScalar>(alpha),
            hytlass::from_real<ElementScalar>(beta),
            detail::Splits(splits)
          );

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

template <typename Gemm>
bool TestAllBiasElementwise(double alpha = 1.0, double beta = 0.0, bool check_relative_equality=false) {

  return TestAll<Gemm>(alpha, beta, testbed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Gemm>
bool TestGemmPerf3x(int iterations = 20,int m = 4096,int n = 4096,int k = 128) {
  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;
  using ElementAccumulator = typename Gemm::GemmKernel::ElementAccumulator;
  using ElementScalar = ElementAccumulator;
  bool passed = true;

  std::vector<int> problem_size_m = { m };
  std::vector<int> problem_size_n = { n };
  std::vector<int> problem_size_k = { k };

  Testbed3x<Gemm, hytlass::epilogue::thread::Identity> testbed;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        ProblemShapeType problem_size;
        if constexpr (hute::rank(ProblemShapeType{}) == 4) {
          problem_size = ProblemShapeType{m, n, k, /* l */ 1};
        }
        else {
          problem_size = ProblemShapeType{m, n, k};
        }
        printf("perf test:{%d %d %d}\n",m,n,k);
        passed = testbed.run(
          problem_size,
          hytlass::from_real<ElementScalar>(1),
          hytlass::from_real<ElementScalar>(0),
          detail::Splits(1),
          true,
          detail::Iterations(iterations)
        );

        if (!passed) {
          return false;
        }
      }
    }
  }

  return passed;
}


} // namespace device
} // namespace gemm
} // namespace test

/////////////////////////////////////////////////////////////////////////////////////////////////

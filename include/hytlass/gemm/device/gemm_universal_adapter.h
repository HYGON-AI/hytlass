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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

// common
#include "hytlass/detail/layout.hpp"
#include "hytlass/detail/mma.hpp"
#include "hytlass/device_kernel.h"
#include "hytlass/gemm/gemm.h"
#include "hytlass/hip_host_adapter.hpp"
#include "hytlass/hytlass.h"

#include "hytlass/trace.h"

// 2.x
#include "hytlass/gemm/device/gemm_universal_base.h"
#include "hytlass/gemm/kernel/gemm_transpose_operands.h"
#include "hytlass/gemm/threadblock/threadblock_swizzle.h"
#include "hytlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

// 3.x
#include "hytlass/gemm/kernel/gemm_universal.hpp"
#include "hip/hip_runtime_api.h"
////////////////////////////////////////////////////////////////////////////////

namespace hytlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  GemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type hytlass::gemm::kernel::Gemm or hytlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of GemmUniversalAdapter
  on the two kernel API types, and thus, GemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/
template <class GemmKernel_, class Enable = void>
class GemmUniversalAdapter;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// HYTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class GemmKernel_>
class GemmUniversalAdapter<
  GemmKernel_,
  hute::enable_if_t<gemm::detail::IsHytlass3GemmKernel<GemmKernel_>::value>>
{
public:
  using GemmKernel = GemmKernel_;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  static bool const kEnableHipHostAdapter = HYTLASS_ENABLE_HIP_HOST_ADAPTER;

  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = hytlass::arch::OpMultiplyAdd;

  using OperatorClass = hytlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;

  using ArchTag = typename GemmKernel::ArchTag;

  // NOTE: Assume identity swizzle for now
  using ThreadblockSwizzle = hytlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = hytlass::gemm::GemmShape<
      hute::size<0>(TileShape{}),
      hute::size<1>(TileShape{}),
      hute::size<2>(TileShape{})>;

  using ClusterShape = hytlass::gemm::GemmShape<
      hute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      hute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      hute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = hytlass::gemm::GemmShape<
      hute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      hute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      hute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads

  static constexpr int WarpsInMma = hute::max(2, HUTE_STATIC_V(hute::size(typename GemmKernel::TiledMma{})) / WARP_SIZE_GPU);
  static constexpr int WarpsInMmaM = 2;
  static constexpr int WarpsInMmaN = hute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = hytlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = hytlass::gemm::GemmShape<
      HUTE_STATIC_V(hute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      HUTE_STATIC_V(hute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      HUTE_STATIC_V(hute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = CollectiveMainloop::DispatchPolicy::Stages;

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = hytlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA, typename CollectiveMainloop::TiledMma::ValTypeA>();
  static int constexpr kAlignmentB = hytlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB, typename CollectiveMainloop::TiledMma::ValTypeB>();
  static int constexpr kAlignmentC = hytlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = hytlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  // Split-K preserves splits that are 128b aligned
  static int constexpr kSplitKAlignment = hute::max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  /// Argument structure: User API
  using Arguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename GemmKernel::Params;

private:

  /// Kernel API parameters object
  Params params_;

public:

  /// Access the Params structure
  Params const& params() const {
    return params_;
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (GemmKernel::can_implement(args)) {
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    HYTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    workspace_bytes += GemmKernel::get_workspace_size(args);
    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    HYTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    hipError_t result;
    if (smem_size >= (48 << 10)) {
      HYTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = hipFuncSetAttribute(
          (const void*)device_kernel<GemmKernel>,
          hipFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (hipSuccess != result) {
        result = hipGetLastError(); // to clear the error bit
        HYTLASS_TRACE_HOST(
          "  hipFuncSetAttribute() returned error: "
          << hipGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
        smem_size);

    if (hipSuccess != result) {
      result = hipGetLastError(); // to clear the error bit
      HYTLASS_TRACE_HOST(
        "  hipOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << hipGetErrorString(result));
      return -1;
    }

    HYTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    void* workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter* hip_adapter = nullptr) {

    HYTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernel::initialize_workspace(args, workspace, stream, hip_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    // Don't set the function attributes - require the HipHostAdapter to set it.
    if constexpr (kEnableHipHostAdapter) {
      HYTLASS_ASSERT(hip_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = GemmKernel::SharedStorageSize;

      HYTLASS_ASSERT(hip_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        HYTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        hipError_t result = hipFuncSetAttribute(
          (const void*)device_kernel<GemmKernel>,
            hipFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (hipSuccess != result) {
          result = hipGetLastError(); // to clear the error bit
          HYTLASS_TRACE_HOST("  hipFuncSetAttribute() returned error: " << hipGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    HYTLASS_TRACE_HOST("GemmUniversal()::update() - workspace: " << workspace);

    size_t workspace_bytes = get_workspace_size(args);
    if (workspace_bytes > 0 && nullptr == workspace) {
      return Status::kErrorWorkspaceNull;
    }

    params_ = GemmKernel::to_underlying_arguments(args, workspace);
    return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underling_arguments()
  static Status
  run(Params& params,
      hipStream_t stream = nullptr,
      HipHostAdapter *hip_adapter = nullptr) {
    HYTLASS_TRACE_HOST("GemmUniversal::run()");
    
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    
    launch_result = Status::kSuccess;
    if constexpr (kEnableHipHostAdapter) {
      HYTLASS_ASSERT(hip_adapter);
      if (hip_adapter) {
        void* kernel_params[] = {&params};

        launch_result = hip_adapter->launch(
          grid, block, smem_size, stream, kernel_params, 0
        );

      }
      else {
        return Status::kErrorInternal;
      }
    }
    else {
      HYTLASS_ASSERT(hip_adapter == nullptr);
      device_kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
    }
    

    hipError_t result = hipGetLastError();
    if (hipSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      HYTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  run(
    Arguments const& args,
    void* workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter *hip_adapter = nullptr
  ) {
    Status status = initialize(args, workspace, stream, hip_adapter);

    if (Status::kSuccess == status) {
      status = run(params_, stream, hip_adapter);
    }
    return status;
  }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  Status
  operator()(
    Arguments const& args,
    void* workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter *hip_adapter = nullptr) {
    return run(args, workspace, stream, hip_adapter);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  run(hipStream_t stream = nullptr, HipHostAdapter *hip_adapter = nullptr) {
    return run(params_, stream, hip_adapter);
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  Status
  operator()(hipStream_t stream = nullptr, HipHostAdapter *hip_adapter = nullptr) {
    return run(params_, stream, hip_adapter);
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// HYTLASS 2.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <class GemmKernel_>
class GemmUniversalAdapter<
  GemmKernel_,
  hute::enable_if_t<not gemm::detail::IsHytlass3GemmKernel<GemmKernel_>::value>>
{
public:

  using GemmKernel = GemmKernel_;

  static bool const kInternalTranspose =
    !hytlass::epilogue::threadblock::detail::is_2x_evt_v<typename GemmKernel::Epilogue> &&  // 2.x EVT does not require internal transpose
    hute::is_same<typename GemmKernel::LayoutC, hytlass::layout::RowMajor>::value;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;
  using WarpShape = typename GemmKernel::WarpShape;
  using InstructionShape = typename GemmKernel::InstructionShape;

  // warp-level, arch-level (instruction), math operator
  using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;

  // Operator class and arch tag extract bottom-up
  // set it for top-level gemm device-level template
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  // Type, layout, and complex transform deliberately exchanged with B
  using MapArguments = kernel::detail::MapArguments<
    typename GemmKernel::ElementA,
    typename GemmKernel::LayoutA,
    GemmKernel::kTransformA,
    GemmKernel::kAlignmentA,
    typename GemmKernel::ElementB,
    typename GemmKernel::LayoutB,
    GemmKernel::kTransformB,
    GemmKernel::kAlignmentB,
    typename GemmKernel::LayoutC,
    kInternalTranspose
  >;

  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static int const kAlignmentA = MapArguments::kAlignmentA;

  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kAlignmentB = MapArguments::kAlignmentB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename MapArguments::LayoutC;
  static int const kAlignmentC = GemmKernel::kAlignmentC;

  // C and D same type for 2.x kernel
  using ElementD = ElementC;
  using LayoutD = LayoutC;

  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementD, LayoutD>;

  static int const kStages = GemmKernel::Mma::kStages;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using UnderlyingOperator = GemmUniversalBase<GemmKernel>;
  using Arguments = typename UnderlyingOperator::Arguments;

  HAS_MEMBER_DEF(EnStaggerK)
  static bool constexpr EnStaggerK =
      get_constant_member_EnStaggerK<typename GemmKernel::Mma::IteratorA>(false) &&
      get_constant_member_EnStaggerK<typename GemmKernel::Mma::IteratorB>(false);

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  GemmUniversalAdapter() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    if (kInternalTranspose) {
      return args.transposed_problem();
    }
    else {
      return args;
    }
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, HipHostAdapter *hip_adapter = nullptr) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args), hip_adapter);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args, HipHostAdapter *hip_adapter = nullptr) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args), hip_adapter);
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {
    return UnderlyingOperator::get_grid_shape(to_underlying_arguments(args));
  }

  /// Computes the tiled shape
  static GemmCoord get_tiled_shape(Arguments const &args) {
    return UnderlyingOperator::get_tiled_shape(to_underlying_arguments(args));
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    return UnderlyingOperator::maximum_active_blocks(smem_capacity);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter *hip_adapter = nullptr
  ) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream, hip_adapter);
  }

  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args) {

    return underlying_operator_.update(to_underlying_arguments(args));
  }

  /// Runs the kernel using initialized state.
  Status run(
    hipStream_t stream = nullptr,
    HipHostAdapter *hip_adapter = nullptr) {

    return underlying_operator_.run(stream, hip_adapter);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    hipStream_t stream = nullptr, 
    HipHostAdapter *hip_adapter = nullptr) {

    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter *hip_adapter = nullptr) {

    Status status = initialize(args, workspace, stream, hip_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, hip_adapter);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::gemm::device

////////////////////////////////////////////////////////////////////////////////

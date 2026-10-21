/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/kernel_hardware_info.hpp"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/tcc_padding.hpp"

#include "hute/tensor.hpp"

namespace hytlass::gemm::kernel {

template <class IntT>
HYTLASS_DEVICE
hute::Stride<IntT, hute::Int<1>, int64_t>
make_hute_packed_stride(hute::Stride<IntT, hute::Int<1>, int64_t> s, hute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  hute::get<0>(s_copy) = static_cast<IntT>(hute::get<1>(shape_MKL));
  int batch_count =  hute::get<2>(shape_MKL);
  if (batch_count > 1) {
    hute::get<2>(s_copy) = static_cast<IntT>(hute::get<0>(shape_MKL) * hute::get<1>(shape_MKL));
  }
  else {
    hute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}


template <class IntT>
HYTLASS_DEVICE
hute::Stride<hute::Int<1>, IntT, int64_t>
make_hute_packed_stride(hute::Stride<hute::Int<1>, IntT, int64_t> s, hute::Shape<int,int,int> shape_MKL) {
  static_assert(std::is_integral_v<IntT>,
    "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
  auto s_copy = s;
  hute::get<1>(s_copy) = static_cast<IntT>(hute::get<0>(shape_MKL));
  int batch_count =  hute::get<2>(shape_MKL);
  if (batch_count > 1) {
    hute::get<2>(s_copy) = static_cast<IntT>(hute::get<0>(shape_MKL) * hute::get<1>(shape_MKL));
  }
  else {
    hute::get<2>(s_copy) = static_cast<IntT>(0);
  }
  return s_copy;
}


///////////////////////////////////////////////////////////////////////////////
// batch_gemm & gemm
template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  hute::enable_if_t<hute::is_base_of_v<KernelMultistage, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static_assert(hute::is_void_v<TileScheduler_> or hute::is_same_v<TileScheduler_, RegularScheduler>,
    "Gfx928 kernel does not support specializing the tile scheduler.");

  using TileSchedulerTag = TileScheduler_;

  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    hute::Shape<hute::Int<1>, hute::Int<1>, hute::Int<1>>>::Scheduler;

  using TileSchedulerArguments = typename TileScheduler::Arguments;

  using TileSchedulerParams = typename TileScheduler::Params;

  static_assert(std::is_same_v<TileScheduler, hytlass::gemm::kernel::detail::RegularTileScheduler>, "must be RegularTileScheduler \n");

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(hute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int SmemEpilogue =  static_cast<int>(hute::max(sizeof(typename CollectiveEpilogue::SharedStorage),  hute::tile_size<0>(TiledMma{}) * hute::tile_size<1>(TiledMma{}) * sizeof(ElementAccumulator)));

  static constexpr int SharedStorageSize = static_cast<int>(hute::max(
        sizeof(typename CollectiveMainloop::SharedStorage),
        SmemEpilogue));

  static constexpr uint32_t MaxThreadsPerBlock = HUTE_STATIC_V(hute::size(TiledMma{}));
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    TileSchedulerParams scheduler{};
    void * workspace{};
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      TileScheduler::to_underlying_arguments(args.problem_shape, TileShape{}, ClusterShape{}, args.scheduler),
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    if (check_padding_arch()) {
      workspace_size += hytlass::get_padding_workspace_size<Arguments, ElementA, ElementB, StrideA, StrideB>(args);
    }
    return workspace_size;
  }

  static
  hytlass::Status
  initialize_workspace(Arguments const& args,
                       void* workspace = nullptr, 
                       hipStream_t stream = nullptr,
                       HipHostAdapter *hip_adapter = nullptr) {
    if (check_padding_arch()) {
      Status status = hytlass::initialize_padding_workspace<Arguments, ElementA, ElementB, StrideA, StrideB>(args, workspace, stream);
      if (status != Status::kSuccess) {
        return status;
      }
    }
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    int batch_count = 1;
    if constexpr (hute::rank(ProblemShape{}) == 4) {
      batch_count = hute::size<3>(params.problem_shape);
    }

    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = params.scheduler.swizzle_size_;
    }

    auto cluster_shape = Shape<_1, _1, _1>{};

    auto grid_shape = TileScheduler::get_grid_shape(params.problem_shape, TileShape{}, cluster_shape, args);
    return grid_shape;
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  HYTLASS_DEVICE
  auto get_mA_mkl_tensor(Params const& params) {
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);
    if (is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_padding_required<ElementA>(K)) {
      auto da_copy = make_hute_packed_stride(StrideA{}, make_shape(M, K + PADDING_A, L));
      return make_tensor(make_gmem_ptr((const ElementA *)params.workspace), make_shape(M, K, L), da_copy);
    }
    if (is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_padding_required<ElementA>(M)) {
      auto da_copy = make_hute_packed_stride(StrideA{}, make_shape(M + PADDING_A, K, L));
      return make_tensor(make_gmem_ptr((const ElementA *)params.workspace), make_shape(M, K, L), da_copy);
    }

    return make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M, K, L), params.mainloop.dA);
  }

  HYTLASS_DEVICE
  auto get_mB_nkl_tensor(Params const& params) {
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);
    uint64_t offset_padding = 0;
    if (is_same_v<StrideA, typename TagToStrideA<layout::RowMajor>::type> && is_padding_required<ElementA>(K)) {
      offset_padding += (PADDING_A + K) * M * sizeof(ElementA);
    }
    if (is_same_v<StrideA, typename TagToStrideA<layout::ColumnMajor>::type> && is_padding_required<ElementA>(M)) {
      offset_padding += (PADDING_A + M) * K * sizeof(ElementA);
    }

    if (is_same_v<StrideB, typename TagToStrideB<layout::ColumnMajor>::type> && is_padding_required<ElementB>(K)) {
      auto db_copy = make_hute_packed_stride(StrideB{}, make_shape(N, K + PADDING_B, L));
      return make_tensor(make_gmem_ptr((const ElementB *)((reinterpret_cast<uint8_t *>(params.workspace) + offset_padding))), make_shape(N, K, L), db_copy);
    }

    if (is_same_v<StrideB, typename TagToStrideB<layout::RowMajor>::type> && is_padding_required<ElementB>(N)) {
      auto db_copy = make_hute_packed_stride(StrideB{}, make_shape(N + PADDING_B, K, L));
      return make_tensor(
          make_gmem_ptr((const ElementB *)((reinterpret_cast<uint8_t *>(params.workspace) + offset_padding))), make_shape(N, K, L), db_copy);
    }

    return make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N, K, L), params.mainloop.dB);
  }

  HYTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace hute;
    using X = Underscore;

    // Preconditions
    HUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(hute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(hute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(hute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(hute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};

    TileScheduler scheduler{params.scheduler};

    auto [m_coord, n_coord, l_coord, is_valid_tile] = scheduler.get_current_work();
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);

#if (defined(__gfx928__) && defined(__HIPCC__))
    Tensor mA_mkl = get_mA_mkl_tensor(params);
    Tensor mB_nkl = get_mB_nkl_tensor(params);
    
#else
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA);
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB);
#endif
    Tensor mA_mk = mA_mkl(_, _, l_coord);
    Tensor mB_nk = mB_nkl(_, _, l_coord);

    // Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)

    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
    auto k_residue = K - size<1>(gA) * size<2>(gA);                                          // K - BLK_K * k_coord_max
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulators);

    auto k_tile_iter  = hute::make_coord_iterator(shape<2>(gA));
    int k_tile_count = size<2>(gA);

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      accumulators,
      gA,
      gB,
      accumulators,
      k_tile_iter, k_tile_count,
      residue_mnk,
      thread_idx,
      smem_buf
    );
    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.epilogue};
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      blk_coord_mnkl,
      accumulators,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::gemm::kernel

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

#include "hute/tensor.hpp"

namespace hytlass::gemm::kernel {

// Stream-K mainloop
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
  hute::enable_if_t<hute::is_base_of_v<KernelStreamKSpecialized, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
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
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  using ClusterShape = typename DispatchPolicy::ClusterShape;

  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    hute::Shape<hute::Int<1>, hute::Int<1>, hute::Int<1>>>::Scheduler;   
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;

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

  static constexpr uint32_t MaxThreadsPerBlock = hute::size(TiledMma{});
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
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    KernelHardwareInfo hw_info;
    TileSchedulerParams scheduler;
    void* workspace;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    HYTLASS_TRACE_HOST("to_underlying_arguments():");

    auto problem_shape = args.problem_shape;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);

    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      HYTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }

    HYTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);
    
    auto * reduction_workspace = workspace;

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
      problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, reduction_workspace);

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
      hw_info,
      scheduler,
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
    TileScheduler t;
    workspace_size +=
        t.template get_workspace_size<ProblemShape, ElementAccumulator>(
            args.scheduler, args.problem_shape, args.hw_info);
    return workspace_size;
  }

  static
  hytlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, hipStream_t stream = nullptr,
    HipHostAdapter* hip_adapter = nullptr) {
    TileScheduler t;
    return t.template initialize_workspace<ProblemShape, ElementAccumulator>(
        args.scheduler, workspace, stream, args.problem_shape,
        args.hw_info);
  }

  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.problem_shape, TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
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
    decltype(params.problem_shape) problem_shape_MNKL;
    if constexpr (hute::rank(decltype(params.problem_shape){}) == 3) {
      problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    } else {
      problem_shape_MNKL = params.problem_shape;
    }

    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};

    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA); //(m,k,l)  
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); //(n,k,l) 

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, blk_shape, make_coord(_,_,_), Step<_1, X,_1>{});          // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, blk_shape, make_coord(_,_,_), Step< X,_1,_1>{});          // (BLK_N,BLK_K,n,k,l)

    TileScheduler scheduler{params.scheduler};
    auto work_tile_info = scheduler.get_current_work();

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    
    while (work_tile_info.is_valid_tile) {
      clear(accumulators);        

      // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
      auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
      auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
      auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
      auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

      // Slice with our work tile coordinates to construct mainloop tensor views
      Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);  // (BLK_N,BLK_K,k)

      // Get the number of K tiles to compute for this work as well as the
      // starting K tile offset of the work.
      auto work_k_tile_count = TileScheduler::get_work_k_tile_count(
          work_tile_info, problem_shape_MNKL, blk_shape);
      auto work_k_tile_start =
          TileScheduler::get_work_k_tile_start(work_tile_info);
      auto k_tile_iter = hute::make_coord_iterator(
          idx2crd(work_k_tile_start, shape<2>(gA)), shape<2>(gA));

      // Compute tile residues for predication
      auto m_max_coord = M - size<0>(gA_mkl) * get<0>(blk_coord);                             // M - BLK_M * m_coord
      auto n_max_coord = N - size<0>(gB_nkl) * get<1>(blk_coord);                             // N - BLK_N * n_coord
      auto k_residue = (!TileScheduler::compute_epilogue(work_tile_info)) ? 0 : K - size<1>(gA) * size<2>(gA);  // K - BLK_K * k_coord_max

      auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

      // Perform the collective scoped MMA
      CollectiveMainloop collective_mma;
      collective_mma(
        accumulators,
        gA,
        gB,
        accumulators,
        k_tile_iter, work_k_tile_count,
        residue_mnk,
        thread_idx,
        smem_buf
      );

      TileScheduler::template fixup<MaxThreadsPerBlock>(
          params.scheduler, work_tile_info, accumulators);

      // Epilogue and write to gD
      CollectiveEpilogue epilogue{params.epilogue}; 
      if (TileScheduler::compute_epilogue(work_tile_info)) {
        epilogue(
          problem_shape_MNKL,
          blk_shape,
          blk_coord,
          accumulators,
          tiled_mma,
          residue_mnk,
          thread_idx,
          smem_buf
        );
      }
      
      // Get next work tile
      work_tile_info = fetch_next_work(work_tile_info, scheduler);
    } // Scheduler work fetch loop

  }

private:
  // Round up number of bytes to the nearest multiple of L2 cache line alignment
  HYTLASS_HOST_DEVICE
  static size_t
  round_up_to_l2_alignment(size_t bytes) {
    constexpr static size_t L2CacheLineSizeBytes = 128;
    return (bytes + L2CacheLineSizeBytes - 1) / L2CacheLineSizeBytes * L2CacheLineSizeBytes;
  }

  // Kernel helper function to get next work unit
  HYTLASS_DEVICE
  typename TileScheduler::WorkTileInfo
  fetch_next_work(
    typename TileScheduler::WorkTileInfo& work_tile_info,
    TileScheduler& scheduler) const {
    // Check whether we should continue on with the current work unit. If this is the case,
    // the work unit will have been updated in continue_current_work to reflect the new
    // tile to be computed.
    if (scheduler.continue_current_work(work_tile_info)) {
      return work_tile_info;
    }

    // Get next work tile
    scheduler.advance_to_next_work();
    return scheduler.get_current_work();
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace hytlass::gemm::kernel

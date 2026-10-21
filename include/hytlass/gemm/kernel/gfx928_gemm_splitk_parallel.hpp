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
#include "hytlass/util/exceptions.h"
#include "hytlass/kernel_hardware_info.hpp"
#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/dispatch_policy.hpp"
#include "hytlass/epilogue/dispatch_policy.hpp"

#include "hute/tensor.hpp"

namespace hytlass::gemm::kernel {

// split-k parallel
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
  hute::enable_if_t<hute::is_base_of_v<KernelSplitkParallelSpecialized, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
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

  static_assert(hute::is_void_v<TileScheduler_> or hute::is_same_v<TileScheduler_, PersistentScheduler>,
    "Gfx928 kernel does not support specializing the tile scheduler.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    hute::Shape<hute::Int<1>, hute::Int<1>, hute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

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

  static constexpr int SmemEpilogue =  static_cast<int>(
    hute::max(sizeof(typename CollectiveEpilogue::SharedStorage),
    hute::tile_size<0>(TiledMma{}) * hute::tile_size<1>(TiledMma{}) * sizeof(ElementAccumulator)));

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
    void * workspace;  
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
      workspace
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemmSplitKParallel;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;
    workspace_size = sizeof(ElementC) * size_t(get<0>(args.problem_shape)) *
                     size_t(get<1>(args.problem_shape)) *
                     size_t(get<3>(args.problem_shape));
    return workspace_size;
  }

  static
  hytlass::Status
  initialize_workspace(Arguments const& args, 
    void* workspace = nullptr,
    hipStream_t stream = nullptr,
    HipHostAdapter* hip_adapter = nullptr) {
    hytlass::Status status = Status::kSuccess;
    return status;
  }

  static dim3
  get_grid_shape(Params const& params) {
    int slice_k = 1;
    if constexpr (rank(ProblemShape{}) == 4) {
      slice_k = get<3>(params.problem_shape);
      auto blk_shape = TileShape{};
      auto gemm_k_size = ceil_div(get<2>(params.problem_shape), slice_k);
      slice_k = ceil_div(get<2>(params.problem_shape), gemm_k_size);
    }
    return dim3(
      hute::size(hute::ceil_div(hute::shape<0>(params.problem_shape), hute::shape<0>(TileShape{}))),
      hute::size(hute::ceil_div(hute::shape<1>(params.problem_shape), hute::shape<1>(TileShape{}))),
      slice_k
    );
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
    // auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});

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

#ifndef __HIPCC__                                                                // (BLK_M,BLK_N,BLK_K)
    auto [m_coord, n_coord, l_coord] = blockIdx;
#else
    auto m_coord = blockIdx.x;
    auto n_coord = blockIdx.y;
    auto l_coord = blockIdx.z;
#endif
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,Int<1>{}), params.mainloop.dA); //(m,k,1)
    Tensor mB_nkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,Int<1>{}), params.mainloop.dB); //(n,k,1)

    auto slice_k = L;
    int grid_tiled_shape_k = 0;
    int gemm_k_size = 0;
    int k_offset = 0;
    gemm_k_size = ceil_div(K, slice_k);
    k_offset = l_coord * gemm_k_size;
    grid_tiled_shape_k = ceil_div(gemm_k_size, size<2>(blk_shape));
    if (K > k_offset && K <= gemm_k_size * (l_coord + 1)) {
      grid_tiled_shape_k = ceil_div(K - k_offset, size<2>(blk_shape));
    } else if (K < gemm_k_size * (l_coord + 1)) {
      return;
    }
    
    auto split_k_shape = make_shape(M, N, gemm_k_size);
    
    // Slice to get the tiles this thread block is responsible for
    Tensor mA_mk_split = local_tile(mA_mkl, split_k_shape, make_coord(_,_,l_coord), Step<_1, X,_1>{});                    // (m,gemm_k_size)
    Tensor gA_layout = local_tile(mA_mk_split, blk_shape, make_coord(m_coord,n_coord,_), Step<_1, X,_1>{});              // (BLK_M,BLK_K,k_subproblem) 
    Tensor gA = take<0,3>(gA_layout);
    Tensor mB_nk_split = local_tile(mB_nkl, split_k_shape, make_coord(_,_,l_coord), Step<X,_1,_1>{});                     // (n,gemm_k_size)
    Tensor gB_layout = local_tile(mB_nk_split, blk_shape, make_coord(m_coord,n_coord,_), Step<X,_1,_1>{});               // (BLK_N,BLK_K,k_subproblem)
    Tensor gB = take<0,3>(gB_layout);

    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
                                 
    int k_residue = 0;
    k_residue = gemm_k_size - grid_tiled_shape_k * size<1>(gA);
    if (K > k_offset && K <= gemm_k_size * (l_coord + 1)) {
      k_residue = K - k_offset - grid_tiled_shape_k * size<1>(gA);
    }

    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulators);

    K = min(K, (l_coord + 1) * gemm_k_size);

    int k_tile_count = ceil_div((K - k_offset), size<2>(blk_shape));
    auto k_tile_iter  = hute::make_coord_iterator(k_tile_count); 

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

    EpilogueParams epilogue_params = params.epilogue;

    size_t splitk_slice_stride = size_t(M) * size_t(N) * sizeof(ElementC);
    auto * parallel_workspace = reinterpret_cast<uint8_t*>(params.workspace);
    epilogue_params.ptr_D = reinterpret_cast<ElementD*>(
        parallel_workspace + splitk_slice_stride * l_coord);

    epilogue_params.thread.beta = 0;  
    epilogue_params.thread.alpha = 1;

    // Epilogue and write to workspace
    CollectiveEpilogue epilogue{epilogue_params};
    
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

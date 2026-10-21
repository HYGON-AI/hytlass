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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/array.h"
#include "hytlass/aligned_buffer.h"

#include "hytlass/numeric_types.h"
#include "hytlass/matrix_shape.h"

#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/threadblock/mma_base.h"



/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting HIP cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  /// Transformation applied to B operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaSingleStage : public MmaBase<Shape_, Policy_, 1> {
public:

  ///< Base class
  using Base = MmaBase<Shape_, Policy_, 1>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  using ArchTag = arch::Gfx928;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  // staticaly assert kStages for MmaSingleStage is 1 (single stage mma pipeline)
  static_assert((Base::kStages == 1), "MmaSingleStage requires kStages set to value 1");
private:

  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;

protected:

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

public:

  /// Construct from tensor references
  HYTLASS_DEVICE
  MmaSingleStage(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx                                        ///< ID of each thread within a warp
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx){

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

  }

  HYTLASS_DEVICE
  void gemm_wait() {
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                 "s_barrier \n\t"
                 : : : "memory");
    // __syncthreads();
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  HYTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,            ///< number of iterations of the mainloop
    FragmentC &accum,                 ///< destination accumulator tile
    IteratorA iterator_A,             ///< iterator over A operand in global memory
    IteratorB iterator_B,             ///< iterator over B operand in global memory
    FragmentC const &src_accum) {     ///< source accumualtor tile

    ///< transformation applied to A fragment
    TransformA transform_A_;

    ///< transformation applied to B fragment
    TransformB transform_B_;  

    ///< Mark register memory access state.

    int warp_stage_idx_ = 0;

    Operator warp_mma;

    //
    // Prologue
    //

    FragmentA tb_frag_A;
    tb_frag_A.clear();
    iterator_A.load(tb_frag_A);
    ++iterator_A;

    FragmentB tb_frag_B;
    tb_frag_B.clear();
    iterator_B.load(tb_frag_B);
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

    __syncthreads();
    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[Base::kWarpGemmIterations == 1 ? 1 : 2];
    WarpFragmentB warp_frag_B[Base::kWarpGemmIterations == 1 ? 1 : 2];

    // Load A fragment from shared B
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;

    // Load B fragment from shared B
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;
    
    // When kWarpGemmIterations == 1, the LDS data for this iteration is read directly,
    // so double-buffering registers is unnecessary.
    if constexpr (Base::kWarpGemmIterations != 1) { 
      warp_stage_idx_ ^= 1;
    }
 

    //
    // Mainloop
    //

    // Avoid the final memory access in buffer load.
    HYTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 1; --gemm_k_iterations) {

      HYTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // If kWarpGemmIterations == 1, write to the current iteration's LDS.
        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          // Ensure the previous LDS read has completed before writing the next round's LDS data in advance.
          gemm_wait();

          this->smem_iterator_A_.store(transform_A_(tb_frag_A));

          this->smem_iterator_B_.store(transform_B_(tb_frag_B)); 

          // reset for LDS
          this->warp_tile_iterator_A_.add_tile_offset(
            {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
          this->warp_tile_iterator_B_.add_tile_offset(
            {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});           


          __syncthreads();
        }


        if constexpr (Base::kWarpGemmIterations == 1) {
          this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations);
          this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations);
        } else {
          this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        }
        
        this->warp_tile_iterator_A_.load(warp_frag_A[warp_stage_idx_]);
        this->warp_tile_iterator_B_.load(warp_frag_B[warp_stage_idx_]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if constexpr (Base::kWarpGemmIterations != 1) {
          warp_stage_idx_ ^= 1;
        }

        if (warp_mma_k == 0) {
          iterator_A.load(tb_frag_A);
          ++iterator_A;

          iterator_B.load(tb_frag_B);
          ++iterator_B;

        }

        warp_mma(
          accum, 
          warp_frag_A[warp_stage_idx_], 
          warp_frag_B[warp_stage_idx_], 
          accum);
      }
    }


    HYTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      // If kWarpGemmIterations == 1, write to the current iteration's LDS.
      if constexpr (Base::kWarpGemmIterations == 1) {
        gemm_wait();

        this->smem_iterator_A_.store(transform_A_(tb_frag_A));

        this->smem_iterator_B_.store(transform_B_(tb_frag_B)); 

        // reset for LDS
        this->warp_tile_iterator_A_.add_tile_offset(
          {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
        this->warp_tile_iterator_B_.add_tile_offset(
          {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});           

          __syncthreads();
      }

      if constexpr (Base::kWarpGemmIterations == 1) {
        this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k % Base::kWarpGemmIterations);
      } else {
        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      }
        
      this->warp_tile_iterator_A_.load(warp_frag_A[warp_stage_idx_]);
      this->warp_tile_iterator_B_.load(warp_frag_B[warp_stage_idx_]);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;

      if constexpr (Base::kWarpGemmIterations != 1) {
        warp_stage_idx_ ^= 1;
      }

      warp_mma(
        accum, 
        warp_frag_A[warp_stage_idx_], 
        warp_frag_B[warp_stage_idx_], 
        accum);
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace hytlass

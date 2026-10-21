/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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
#include "hip/hip_runtime.h"
#include <iostream>

#include "../../common/hytlass_unit_test.h"
#include "hytlass/hytlass.h"

#include "hytlass/gemm/gemm.h"
#include "hytlass/gemm/kernel/gemm_grouped.h"
#include "hytlass/gemm/kernel/default_gemm_grouped.h"
#include "hytlass/gemm/device/gemm_grouped.h"

#include "hytlass/util/host_tensor.h"
#include "hytlass/util/reference/host/gemm.h"
#include "hytlass/util/reference/host/tensor_compare.h"
#include "hytlass/util/reference/host/tensor_copy.h"
#include "hytlass/util/reference/host/tensor_fill.h"
#include "hytlass/util/tensor_view_io.h"

#include "testbed_grouped.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Visitor class to abstract away the algorithm for iterating over tiles.
//
// This is the prototype. We will delete this when the efficient kernel is
// available.
struct GemmGroupedProblemVisitor {

  struct Params {
    hytlass::gemm::GemmCoord const *problem_sizes;
    int32_t                         problem_count;
    int64_t const                  *tile_count;
  };

  struct SharedStorage {
    //
    // Nothing for now. As an optimization step, we could consider parallel
    // argmin or prefix sums across the block.
    //
  };

  //
  // Data members
  //

  SharedStorage &shared_storage;
  Params const &params;
  hytlass::MatrixCoord threadblock_shape;

  int64_t tile_idx;
  int64_t tile_count_sum;
  int64_t problem_tile_start;
  int32_t problem_idx;

  //
  // Methods
  //
  HYTLASS_DEVICE
  GemmGroupedProblemVisitor(
    SharedStorage &shared_storage_,
    Params const &params_,
    hytlass::MatrixCoord threadblock_shape_,
    int32_t block_idx
  ):
    shared_storage(shared_storage_),
    params(params_),
    threadblock_shape(threadblock_shape_),
    tile_idx(block_idx),
    tile_count_sum(0),
    problem_idx(0)
  {

    hytlass::gemm::GemmCoord problem = params.problem_sizes[problem_idx];

    hytlass::gemm::GemmCoord  grid = grid_shape(problem);

    problem_tile_start = 0;
    tile_count_sum = grid.m() * grid.n();
  }

  /// Get the grid shape
  HYTLASS_HOST_DEVICE
  static hytlass::gemm::GemmCoord grid_shape(
    hytlass::gemm::GemmCoord const &problem,
    hytlass::MatrixCoord const & block_shape) {

    return hytlass::gemm::GemmCoord(
      ((problem.m() - 1 + block_shape.row()) / block_shape.row()),
      ((problem.n() - 1 + block_shape.column()) / block_shape.column()),
      1);
  }

  /// Get the grid shape
  HYTLASS_DEVICE
  hytlass::gemm::GemmCoord grid_shape(hytlass::gemm::GemmCoord const &problem) const {
    return grid_shape(problem, threadblock_shape);
  }

  /// Returns true if there is a tile to compute
  HYTLASS_DEVICE
  bool next_tile() {

    if (tile_idx < tile_count_sum) {
      return true;
    }

    do {
      ++problem_idx;

      if (problem_idx >= params.problem_count) {
        return false;
      }

      hytlass::gemm::GemmCoord problem = params.problem_sizes[problem_idx];
      hytlass::gemm::GemmCoord  grid = grid_shape(problem);

      int64_t tile_count = grid.m() * grid.n();

      problem_tile_start = tile_count_sum;
      tile_count_sum += tile_count;

    } while (tile_count_sum <= tile_idx);

    return true;
  }

  /// Gets the global tile index
  HYTLASS_HOST_DEVICE
  int64_t tile_index() const {
    return tile_idx;
  }

  /// Gets the index of the problem
  HYTLASS_HOST_DEVICE
  int32_t problem_index() const {
    return problem_idx;
  }

  /// Returns the problem size for the current problem
  HYTLASS_HOST_DEVICE
  hytlass::gemm::GemmCoord problem_size() const {
    return params.problem_sizes[problem_idx];
  }

  HYTLASS_HOST_DEVICE
  int64_t threadblock_idx() const {
    return tile_idx - problem_tile_start;
  }

  HYTLASS_DEVICE
  void advance(int32_t grid_size) {
    tile_idx += grid_size;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int ThreadblockShapeM, int ThreadblockShapeN>
__global__ void GroupedBatchedKernel(GemmGroupedProblemVisitor::Params params) {

  __shared__ GemmGroupedProblemVisitor::SharedStorage shared_storage;

  GemmGroupedProblemVisitor problem_visitor(
    shared_storage,
    params,
    {ThreadblockShapeM, ThreadblockShapeN},
    blockIdx.x);

  while (problem_visitor.next_tile()) {

    hytlass::gemm::GemmCoord problem_size = problem_visitor.problem_size();
    int64_t threadblock_idx                       = problem_visitor.threadblock_idx();

    hytlass::gemm::GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

    int threadblock_tile_m_idx = int(threadblock_idx / grid_shape.n());
    int threadblock_tile_n_idx = int(threadblock_idx % grid_shape.n());

    //
    // Do the MMA
    //

    if (threadIdx.x == 0) {
      #if 0
      printf("Block %d - tile: %lld, problem %d, threadblock_idx: %lld, threadblock(m: %d, n: %d)\n",
        blockIdx.x,
        static_cast<long long>(problem_visitor.tile_index()),
        problem_visitor.problem_index(),
        threadblock_idx,
        threadblock_tile_m_idx,
        threadblock_tile_n_idx);
      #endif
    }

    // Next tile
    problem_visitor.advance(gridDim.x);
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_scheduler, 64x64x32_32x32x32) {

  int32_t problem_count = 16;

  int const kThreadblockShapeM = 64;
  int const kThreadblockShapeN = 64;

  std::vector<hytlass::gemm::GemmCoord> problem_sizes(problem_count);
  std::vector<int64_t> tile_counts(problem_count);

  // construct a few problems of random sizes
  srand(1921);
  for (int32_t i = 0; i < problem_count; ++i) {
    problem_sizes.at(i) = hytlass::gemm::GemmCoord(
      8 * (rand() % 48) + 64,
      8 * (rand() % 48) + 64,
      8 * (rand() % 48) + 64);
  }

  // compute prefix sum
  int64_t tile_count = 0;

  for (int32_t i = 0; i < problem_count; ++i) {

    hytlass::gemm::GemmCoord grid_shape = GemmGroupedProblemVisitor::grid_shape(
      problem_sizes.at(i), {kThreadblockShapeM, kThreadblockShapeN});

    int32_t problem_tile_count = (grid_shape.m() * grid_shape.n());

    int64_t tile_start = tile_count;

    tile_count += problem_tile_count;
    tile_counts.at(i) = tile_count;

    if (false) {
      std::cout << "Problem " << i << " size("
        << problem_sizes.at(i).m() << "-by-" << problem_sizes.at(i).n()
        << ") - tiles: " << problem_tile_count << ",  grid(" << grid_shape.m() << ", " << grid_shape.n()
        << "), tiles[" << tile_start << ", " << tile_count << ")" << std::endl;
    }
  }

  // Copy to device memory
  hytlass::DeviceAllocation<hytlass::gemm::GemmCoord> problem_sizes_device(problem_count);
  hytlass::DeviceAllocation<int64_t>                  tile_counts_device(problem_count);

  problem_sizes_device.copy_from_host(problem_sizes.data());
  tile_counts_device.copy_from_host(tile_counts.data());

  GemmGroupedProblemVisitor::Params params;
  params.problem_sizes = problem_sizes_device.get();
  params.problem_count = problem_count;
  params.tile_count = tile_counts_device.get();

  // Launch the kernel
  dim3 grid(108, 1, 1);
  dim3 block(128, 1, 1);

  GroupedBatchedKernel<kThreadblockShapeM, kThreadblockShapeN><<< grid, block >>>(params);

  // wait
  auto st = hipDeviceSynchronize();
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f16n_f16t_f32n_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    ElementOutput, hytlass::layout::ColumnMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(24);
  EXPECT_TRUE(passed);

}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f16n_f16t_f32t_tensor_op_f32, 128x128x32_64x64x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    ElementOutput, hytlass::layout::RowMajor,    // row major
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 128, 32>,
    hytlass::gemm::GemmShape<64, 64, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(24);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f16t_f16n_f32n_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    8,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    ElementOutput, hytlass::layout::ColumnMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 32>,
    hytlass::gemm::GemmShape<64, 32, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f16t_f16n_f32t_tensor_op_f32, 128x64x32_64x32x32) {

  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    hytlass::half_t,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    8,
    hytlass::half_t,
    hytlass::layout::ColumnMajor,
    hytlass::ComplexTransform::kNone,
    8,
    ElementOutput, hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassTensorOp,
    hytlass::arch::Gfx928,
    hytlass::gemm::GemmShape<128, 64, 32>,
    hytlass::gemm::GemmShape<64, 32, 32>,
    hytlass::gemm::GemmShape<16, 16, 16>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / hytlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f32t_f32t_f32n_simt_f32, 128x128x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementOutput, hytlass::layout::ColumnMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 128, 8>,
    hytlass::gemm::GemmShape<64, 32, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f32t_f32t_f32t_simt_f32, 128x128x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementOutput, hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 128, 8>,
    hytlass::gemm::GemmShape<64, 32, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f32t_f32t_f32n_simt_f32, 128x64x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementOutput, hytlass::layout::ColumnMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 64, 8>,
    hytlass::gemm::GemmShape<64, 32, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(GFX928_Device_GemmGrouped_f32t_f32t_f32t_simt_f32, 128x64x8_64x32x1) {

  using ElementInput = float;
  using ElementOutput = float;
  using ElementAccumulator = float;

  using GemmKernel = typename hytlass::gemm::kernel::DefaultGemmGrouped<
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementInput,
    hytlass::layout::RowMajor,
    hytlass::ComplexTransform::kNone,
    1,
    ElementOutput, hytlass::layout::RowMajor,
    ElementAccumulator,
    hytlass::arch::OpClassSimt,
    hytlass::arch::Gfx906,
    hytlass::gemm::GemmShape<128, 64, 8>,
    hytlass::gemm::GemmShape<64, 32, 8>,
    hytlass::gemm::GemmShape<1, 1, 1>,
    hytlass::epilogue::thread::LinearCombination<
        ElementOutput, 1,
        ElementAccumulator, ElementAccumulator>,
    hytlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    2>::GemmKernel;

  using Gemm = hytlass::gemm::device::GemmGrouped<GemmKernel>;

  //
  // Test
  //

  test::gemm::device::TestbedGrouped<Gemm> testbed;

  bool passed = testbed.run(27);
  EXPECT_TRUE(passed);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
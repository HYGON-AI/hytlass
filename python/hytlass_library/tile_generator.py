#################################################################################################
# Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################
from itertools import product

try:
  import builtins
  if hasattr(builtins, "HYTLASS_IGNORE_PACKAGE") and HYTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import hytlass_library")
  from hytlass_library.library import *
except ImportError:
  from library import *

class TileGeneratorGfx928:

  def check_shared_memory_constraint(self, Tile_M, Tile_N, Tile_K, stage, smem_size, byte_size):
    # TODO: Currently assumes a two-stage reg-LDS pipeline.
    total_used_smem = Tile_M * Tile_K * byte_size + Tile_N * Tile_K * byte_size
    if total_used_smem >= smem_size:
      return False
    return True

  def check_vgpr_constraint(self, block_shape, warp_count, math_inst, byte_size):
    # TODO: Include alignment in the analysis.
    block_m, block_n, block_k = block_shape
    warp_cnt_m, warp_cnt_n, warp_cnt_k = warp_count

    thread_cnt = warp_cnt_m * warp_cnt_n * warp_cnt_k * 64
    thread_cnt_mn = warp_cnt_m * warp_cnt_n * 64
    
    # gmem -> reg
    elements_per_thread = (block_m + block_n) * block_k // thread_cnt
    tg_vgpr_cost = elements_per_thread * (byte_size / 4)

    per_warp_m = block_m // warp_cnt_m
    per_warp_n = block_n // warp_cnt_n

    k_elements_per_iterator = math_inst.instruction_shape[2]

    # lds -> reg
    # Prefetching requires twice as many registers.
    elements_per_thread = (per_warp_m + per_warp_n) * k_elements_per_iterator * 2 / 64
    sr_vgpr_cost = elements_per_thread * (byte_size / 4)

    # accum c
    elements_per_thread = (block_m * block_n) // thread_cnt_mn
    c_vgpr_cost = elements_per_thread

    total_vgpr = tg_vgpr_cost + sr_vgpr_cost + c_vgpr_cost

    return total_vgpr < 224

  def check_k_major_copy_constraints(self, Tile_MN, Tile_K, thread_count, aligment, byte_size):
    MaxElementsPerThread = Tile_MN * Tile_K // thread_count
    Aligment_ = min(MaxElementsPerThread, aligment)
    if Aligment_ == 0:
      return False
    threads_major = Tile_K // Aligment_
    threads_minor = thread_count // threads_major
    if (Aligment_ * byte_size * 8) & (Aligment_ * byte_size * 8 - 1) != 0:
      return False
    if Aligment_ * byte_size * 8 > 128:
      return False
    if threads_major <= 0:
      return False
    if thread_count % threads_major != 0:
      return False
    if not (threads_minor == 0 or (Tile_MN % threads_minor == 0)):
      return False
    return True

  def check_mn_major_copy_constraints(self, Tile_MN, Tile_K, thread_count, aligment, byte_size):
    MaxElementsPerThread = Tile_MN * Tile_K // thread_count
    Aligment_ = min(MaxElementsPerThread, aligment)
    if Aligment_ == 0:
      return False
    threads_major = Tile_MN // Aligment_
    threads_minor = thread_count // threads_major
    if (Aligment_ * byte_size * 8) & (Aligment_ * byte_size * 8 - 1) != 0:
      return False
    if Aligment_ * byte_size * 8 > 128:
      return False
    if threads_major <= 0:
      return False
    if thread_count % threads_major != 0:
      return False
    if not (threads_minor == 0 or (Tile_K % threads_minor == 0)):
      return False
    return True

  def check_common_constraints(self, Tile_M, Tile_N, Tile_K, warp_count, math_inst, smem_size, stage, thread_count, layouts,
                               byte_size):
    for layout in layouts:
      if layout[0][0] == LayoutType.RowMajor:
        if not self.check_k_major_copy_constraints(Tile_M, Tile_K, thread_count, layout[0][1], byte_size):
          return False
      if layout[1][0] == LayoutType.ColumnMajor:
        if not self.check_k_major_copy_constraints(Tile_N, Tile_K, thread_count, layout[1][1], byte_size):
          return False
      if layout[0][0] == LayoutType.ColumnMajor:
        if not self.check_mn_major_copy_constraints(Tile_M, Tile_K, thread_count, layout[0][1], byte_size):
          return False
      if layout[1][0] == LayoutType.RowMajor:
        if not self.check_mn_major_copy_constraints(Tile_N, Tile_K, thread_count, layout[1][1], byte_size):
          return False
    if not self.check_vgpr_constraint((Tile_M, Tile_N, Tile_K), warp_count, math_inst, byte_size):
      return False
    if not self.check_shared_memory_constraint(Tile_M, Tile_N, Tile_K, stage, smem_size, byte_size):
      return False
    return True

  def check_8b_constraints(self, Tile_M, Tile_N, Tile_K, math_inst, warp_count, smem_size, stage, thread_count, layouts,
                           byte_size):
    if not self.check_common_constraints(Tile_M, Tile_N, Tile_K, warp_count, math_inst, smem_size, stage, thread_count, layouts,
                                         byte_size):
      return False
    if (Tile_M // 32 % 2 != 0 and warp_count[0] != 1) or (Tile_N // 32 % 2 != 0 and warp_count[1] != 1):
      return False
    if (Tile_M // 32 % 2 == 0 and Tile_M // 32 % 4 != 0 and warp_count[0] in {4, 8}) or \
        (Tile_N // 32 % 2 == 0 and Tile_N // 32 % 4 != 0 and warp_count[1] in {4, 8}):
      return False
    if (Tile_M // 32 % 2 == 0 and Tile_M // 32 // 4 % 2 != 0 and warp_count[0] == 8) or \
        (Tile_N // 32 % 2 == 0 and Tile_N // 32 // 4 % 2 != 0 and warp_count[1] == 8):
      return False
    if Tile_K > 128:
      return False
    if Tile_K > 64 and math_inst.instruction_shape[2] == 64:
      return False
    if Tile_M % math_inst.instruction_shape[0] != 0 or Tile_N % math_inst.instruction_shape[1] != 0 or Tile_K % math_inst.instruction_shape[2] != 0:
      return False
    return True

  def check_16b_constraints(self, Tile_M, Tile_N, Tile_K, math_inst, warp_count, smem_size, stage, thread_count, layouts,
                            byte_size):
    if not self.check_common_constraints(Tile_M, Tile_N, Tile_K, warp_count, math_inst, smem_size, stage, thread_count, layouts,
                                         byte_size):
      return False
    if (Tile_M // 32 % 2 != 0 and warp_count[0] != 1) or (Tile_N // 32 % 2 != 0 and warp_count[1] != 1):
      return False
    if (Tile_M // 32 % 2 == 0 and Tile_M // 32 % 4 != 0 and warp_count[0] in {4, 8}) or \
        (Tile_N // 32 % 2 == 0 and Tile_N // 32 % 4 != 0 and warp_count[1] in {4, 8}):
      return False
    if (Tile_M // 32 % 2 == 0 and Tile_M // 32 // 4 % 2 != 0 and warp_count[0] == 8) or \
        (Tile_N // 32 % 2 == 0 and Tile_N // 32 // 4 % 2 != 0 and warp_count[1] == 8):
      return False
    if Tile_K > 64:
      return False
    if Tile_K > 32 and math_inst.instruction_shape[2] == 32:
      return False
    if Tile_M % math_inst.instruction_shape[0] != 0 or Tile_N % math_inst.instruction_shape[1] != 0 or Tile_K % math_inst.instruction_shape[2] != 0:
      return False
    return True

  def check_32b_constraints(self, Tile_M, Tile_N, Tile_K, math_inst, warp_count, smem_size, stage, thread_count, layouts,
                            byte_size):
    if not self.check_common_constraints(Tile_M, Tile_N, Tile_K, warp_count, math_inst, smem_size, stage, thread_count, layouts,
                                         byte_size):
      return False
    if (Tile_M // 32 % 2 != 0 and warp_count[0] in {4, 8}) or (Tile_N // 32 % 2 != 0 and warp_count[1] in {4, 8}):
      return False
    if (Tile_M // 64 % 2 != 0 and warp_count[0] == 8) or (Tile_N // 64 % 2 != 0 and warp_count[1] == 8):
      return False
    return True

  def generate_tile_descriptions(self, tile_configs, math_insts, data_type, layouts):
    tile_descriptions = []
    stages = tile_configs.stages
    cluster_shapes = tile_configs.cluster_shapes
    min_cc = tile_configs.min_cc
    max_cc = tile_configs.max_cc
    smem_size = tile_configs.smem_size
    warp_count_mapping = tile_configs.warp_count_mapping
    thread_counts = [
      [thread_count, warp_count]
      for thread_count, warp_counts in warp_count_mapping.items()
      for warp_count in warp_counts
    ]

    if data_type == '8b':
      byte_size = 1
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(32, 513, 32)))
      check_constraints = self.check_8b_constraints
    elif data_type == '16b':
      byte_size = 2
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(16, 513, 16)))
      check_constraints = self.check_16b_constraints
    elif data_type == '32b':
      byte_size = 4
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(8, 513, 8)))
      check_constraints = self.check_32b_constraints
    else:
      raise ValueError("Unsupported data type: {}".format(data_type))

    combinations = product(threadblock_shapes, stages, thread_counts, [math_insts], [min_cc], [max_cc], cluster_shapes)

    for threadblock_shape, stage, (thread_count, warp_count), math_inst, min_cc, max_cc, cluster_shape in combinations:
      Tile_M, Tile_N, Tile_K = threadblock_shape

      if not check_constraints(Tile_M, Tile_N, Tile_K, math_inst, warp_count, smem_size, stage, thread_count, layouts, byte_size):
        continue

      tile_description = TileDescription(threadblock_shape, stage, warp_count, math_inst, min_cc, max_cc, cluster_shape)
      tile_descriptions.append(tile_description)

    return tile_descriptions


class TileGeneratorGfx906:
  def check_shared_memory_constraint(self, Tile_M, Tile_N, Tile_K, stage, smem_size, byte_size):
    # TODO: Currently assumes a two-stage reg-LDS pipeline.
    total_used_smem = Tile_M * Tile_K * byte_size + Tile_N * Tile_K * byte_size
    if total_used_smem >= smem_size:
      return False
    return True

  def check_vgpr_constraint(self, Tile_M, Tile_N,thread_count, byte_size):
    if Tile_M * Tile_N // thread_count >= 192:
      return False
    return True

  def check_k_major_copy_constraints(self, Tile_MN, Tile_K, thread_count, aligment, byte_size):
    MaxElementsPerThread = Tile_MN * Tile_K // thread_count
    Aligment_ = min(MaxElementsPerThread, aligment)
    if Aligment_ == 0:
      return False
    threads_major = Tile_K // Aligment_
    threads_minor = thread_count // threads_major
    if (Aligment_ * byte_size * 8) & (Aligment_ * byte_size * 8 - 1) != 0:
      return False
    if Aligment_ * byte_size * 8 > 128:
      return False
    if threads_major <= 0:
      return False
    if thread_count % threads_major != 0:
      return False
    if not (threads_minor == 0 or (Tile_MN % threads_minor == 0)):
      return False
    return True

  def check_mn_major_copy_constraints(self, Tile_MN, Tile_K, thread_count, aligment, byte_size):
    MaxElementsPerThread = Tile_MN * Tile_K // thread_count
    Aligment_ = min(MaxElementsPerThread, aligment)
    if Aligment_ == 0:
      return False
    threads_major = Tile_MN // Aligment_
    threads_minor = thread_count // threads_major
    if (Aligment_ * byte_size * 8) & (Aligment_ * byte_size * 8 - 1) != 0:
      return False
    if Aligment_ * byte_size * 8 > 128:
      return False
    if threads_major <= 0:
      return False
    if thread_count % threads_major != 0:
      return False
    if not (threads_minor == 0 or (Tile_K % threads_minor == 0)):
      return False
    return True

  def check_constraints(self, Tile_M, Tile_N, Tile_K, warp_count, stage, smem_size, thread_count, layouts, byte_size):
    for layout in layouts:
      if layout[0][0] == LayoutType.RowMajor:
        if not self.check_k_major_copy_constraints(Tile_M, Tile_K, thread_count, layout[0][1], byte_size):
          return False
      if layout[1][0] == LayoutType.ColumnMajor:
        if not self.check_k_major_copy_constraints(Tile_N, Tile_K, thread_count, layout[1][1], byte_size):
          return False
      if layout[0][0] == LayoutType.ColumnMajor:
        if not self.check_mn_major_copy_constraints(Tile_M, Tile_K, thread_count, layout[0][1], byte_size):
          return False
      if layout[1][0] == LayoutType.RowMajor:
        if not self.check_mn_major_copy_constraints(Tile_N, Tile_K, thread_count, layout[1][1], byte_size):
          return False
    if not self.check_vgpr_constraint(Tile_M, Tile_N, thread_count, byte_size):
      return False
    if not self.check_shared_memory_constraint(Tile_M, Tile_N, Tile_K, stage, smem_size, byte_size):
      return False
    return True

  def generate_tile_descriptions(self, tile_configs, math_inst, data_type, layouts):
    tile_descriptions = []
    stages = tile_configs.stages
    cluster_shapes = tile_configs.cluster_shapes
    min_cc = tile_configs.min_cc
    max_cc = tile_configs.max_cc
    smem_size = tile_configs.smem_size
    warp_count_mapping = tile_configs.warp_count_mapping
    thread_counts = [
      [thread_count, warp_count]
      for thread_count, warp_counts in warp_count_mapping.items()
      for warp_count in warp_counts
    ]

    if data_type == '8b':
      byte_size = 1
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(32, 513, 32)))
      check_constraints = self.check_constraints
    elif data_type == '16b':
      byte_size = 2
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(16, 513, 16)))
      check_constraints = self.check_constraints
    elif data_type == '32b':
      byte_size = 4
      threadblock_shapes = list(product(range(32, 513, 32), range(32, 513, 32), range(8, 513, 8)))
      check_constraints = self.check_constraints
    else:
      raise ValueError("Unsupported data type: {}".format(data_type))

    combinations = product(threadblock_shapes, stages, thread_counts, [math_inst], [min_cc], [max_cc], cluster_shapes)

    for threadblock_shape, stage, (thread_count, warp_count), math_inst, min_cc, max_cc, cluster_shape in combinations:
      Tile_M, Tile_N, Tile_K = threadblock_shape

      if not check_constraints(Tile_M, Tile_N, Tile_K, warp_count, stage, smem_size, thread_count, layouts, byte_size):
        continue

      tile_description = TileDescription(threadblock_shape, stage, warp_count, math_inst, min_cc, max_cc, cluster_shape)
      tile_descriptions.append(tile_description)

    return tile_descriptions

class TileGeneratorGfx928_2x:
  # The hytlass2 mainloop always uses 16-byte vectorized memory accesses.
  ACCESS_SIZE = 16
  WARP_SIZE_GPU = 64

  def check_power_of_two(self, val) -> bool:
    return val > 0 and ((val & (val - 1)) == 0)

  def check_shared_memory_constraint(self, Tile: tuple, Warp_cnt: tuple, stage, smem_size, byte_size):
    _block_m, _block_n, _block_k = Tile
    _warp_cnt_m, _warp_cnt_n, _warp_cnt_k = Warp_cnt

    mainloop_used_smem = (_block_m + _block_n) * _block_k * stage * byte_size
    
    k_row_per_iterator = 16
    byte_size_lds = 4
    epilogue_used_smem = _warp_cnt_m * _warp_cnt_k *  _warp_cnt_n * (_block_n // _warp_cnt_n) * k_row_per_iterator * byte_size_lds

    return max(mainloop_used_smem, epilogue_used_smem) <= smem_size

  def check_gemm_vgpr_constraint(self, block_shape, warp_count, math_inst, byte_size, align, buffer_access):
    # Rough register estimate for single-stage pipelines only.
    block_m, block_n, block_k = block_shape
    warp_cnt_m, warp_cnt_n, warp_cnt_k = warp_count

    thread_cnt = warp_cnt_m * warp_cnt_n * warp_cnt_k * self.WARP_SIZE_GPU
    thread_cnt_mn = warp_cnt_m * warp_cnt_n * self.WARP_SIZE_GPU

    element_per_thread = (block_m * block_k + block_n * block_k) // thread_cnt
    tg_vgpr_cost = element_per_thread * (byte_size / 4)

    buffer_offsets_vgpr_cost = 0
    access_cnt = element_per_thread // align

    if buffer_access:
      # The guard offset is stored as an int, so each memory access incurs 4 bytes of register overhead.
      buffer_offsets_vgpr_cost = access_cnt
    else:
      # Predicate register constraint for global loads.
      if access_cnt > 64:
        return False

    per_warp_m = block_m // warp_cnt_m
    per_warp_n = block_n // warp_cnt_n

    k_elements_per_iterator = math_inst.instruction_shape[2]
    
    element_per_thread = (per_warp_m * k_elements_per_iterator + per_warp_n * k_elements_per_iterator) // self.WARP_SIZE_GPU
    sr_vgpr_cost = element_per_thread * (byte_size / 4) * 2

    element_per_thread = (block_m * block_n) // thread_cnt_mn
    c_vpgr_cost = element_per_thread

    total_vgpr = tg_vgpr_cost + sr_vgpr_cost + c_vpgr_cost + buffer_offsets_vgpr_cost
    return total_vgpr < 224

  def check_conv_vgpr_constraint_conv(self, block_shape, warp_count, math_inst, byte_size, align,
                                 conv_types, iterator_algorithms):
    # Register estimates cover only single-stage Fprop.
    block_m, block_n, block_k = block_shape
    warp_cnt_m, warp_cnt_n, warp_cnt_k = warp_count

    thread_cnt = warp_cnt_m * warp_cnt_n * warp_cnt_k * self.WARP_SIZE_GPU
    thread_cnt_mn = warp_cnt_m * warp_cnt_n * self.WARP_SIZE_GPU

    element_per_thread = (block_m * block_k + block_n * block_k) // thread_cnt
    tg_vgpr_cost = element_per_thread * (byte_size / 4)

    per_warp_m = block_m // warp_cnt_m
    per_warp_n = block_n // warp_cnt_n

    k_elements_per_iterator = math_inst.instruction_shape[2]
    
    element_per_thread = (per_warp_m * k_elements_per_iterator + per_warp_n * k_elements_per_iterator) // self.WARP_SIZE_GPU
    sr_vgpr_cost = element_per_thread * (byte_size / 4) * 2

    element_per_thread = (block_m * block_n) // thread_cnt_mn

    # Accumulators use 32-bit types.
    c_vpgr_cost = element_per_thread

    data_vgpr = tg_vgpr_cost + sr_vgpr_cost + c_vpgr_cost

    total_vgpr = 0

    # Convolution iterators need auxiliary arrays for global-memory offsets.
    if ConvKind.Fprop in conv_types:
      logical_align = self.ACCESS_SIZE / (byte_size)
      # Fprop does not iterate along the contiguous dimension.
      iteration_strided_a = (block_m * block_k) // logical_align // thread_cnt
      iteration_strided_b = (block_n * block_k) // logical_align // thread_cnt

      if IteratorAlgorithm.Analytic in iterator_algorithms:
        # n,p,q offset
        vgpr_a = 3 * iteration_strided_a
        # k, group_idx_offset_k
        vgpr_b = 2 * iteration_strided_b
        iterator_algo_vgpr_cost = vgpr_a + vgpr_b
        total_vgpr = data_vgpr + iterator_algo_vgpr_cost
      if IteratorAlgorithm.Optimized in iterator_algorithms:
        vgpr_a = (logical_align // align) * iteration_strided_a * 2 + (iteration_strided_a // 4)
        vgpr_b = (logical_align // align)
        iterator_algo_vgpr_cost = vgpr_a + vgpr_b
        total_vgpr = data_vgpr + iterator_algo_vgpr_cost
      if IteratorAlgorithm.FewChannels in iterator_algorithms or \
         IteratorAlgorithm.FewChannels in iterator_algorithms:
        vgpr_a = 3 * iteration_strided_a
        vgpr_b = iteration_strided_b
        iterator_algo_vgpr_cost = vgpr_a + vgpr_b
        total_vgpr = data_vgpr + c_vpgr_cost + iterator_algo_vgpr_cost

    return total_vgpr < 224


  def check_epilogue_constraints(self, block_shape, warp_cnt, alignment, byte_size):
    warp_remaining = warp_cnt[1] * warp_cnt[2]
    shape_row = 16
    shape_column = block_shape[1]
    k_shape_row = shape_row // warp_remaining
    k_shape_width = shape_column // alignment

    k_target_memory_access_width = 256 // (alignment * byte_size)
    k_target_memroy_access_row = self.WARP_SIZE_GPU // k_target_memory_access_width

    k_access_width = self.WARP_SIZE_GPU // k_shape_row if k_target_memroy_access_row > k_shape_row else \
                     min(k_shape_width, min(self.WARP_SIZE_GPU, 256 // (alignment * byte_size)))
    k_access_row = k_shape_row if k_target_memroy_access_row > k_shape_row else \
                     min(shape_row, self.WARP_SIZE_GPU // k_access_width)

    k_iterations_row = k_shape_row // k_access_row
    k_iterations_column = k_shape_width // k_access_width

    if k_iterations_column == 0:
      return False
    if k_iterations_row == 0:
      return False
    if k_access_width * alignment > shape_column:
      return False
    if k_access_width * k_iterations_column * alignment != shape_column:
      return False
    if k_access_row * k_access_width != self.WARP_SIZE_GPU:
      return False

    return True

  def check_k_major_copy_constraints(self, blk_mn, blk_k, warp_cnt_mn, warp_cnt_k, thread_cnt, math_inst, byte_size):
    inst_m, inst_n, inst_k = math_inst.instruction_shape

    if not self.check_power_of_two(blk_k):
      return False
      
    if blk_k % warp_cnt_k != 0:
      return False
    warp_shape_k = blk_k // warp_cnt_k

    if warp_shape_k % inst_k != 0:
      return False
    
    if blk_mn % warp_cnt_mn != 0:
      return False
    warp_shape_mn = blk_mn // warp_cnt_mn
    if warp_shape_mn % inst_m != 0:
      return False

    warp_major_lane = (blk_k * byte_size) // self.ACCESS_SIZE

    if warp_major_lane == 0:
      return False

    warp_minor_lane = self.WARP_SIZE_GPU // warp_major_lane
    # Require a regular tile partition with one iteration along the major dimension.
    if warp_major_lane * warp_minor_lane != self.WARP_SIZE_GPU:
      return False

    warp_cnt = thread_cnt // self.WARP_SIZE_GPU

    blk_minor_thread = warp_cnt * warp_minor_lane
    if (blk_mn < blk_minor_thread) or (blk_mn % blk_minor_thread != 0):
      return False

    # regularIterator requires a power-of-two kPointer count.
    swizzle_unit = 0

    if byte_size == 4:
      # 32-bit data types do not yet support the splice iterator.
      swizzle_unit = 16
    elif byte_size == 2:
      swizzle_unit = 8 if inst_k == 32 else 16
    elif byte_size == 1:
      swizzle_unit = 8 if inst_k == 64 else 16
    else:
      return False

    strided_iteration_cnt = blk_mn // (warp_cnt * warp_minor_lane)
    k_pointer_count = min(strided_iteration_cnt, max(swizzle_unit // warp_minor_lane, 1))
    
    if not self.check_power_of_two(k_pointer_count):
      return False

    if k_pointer_count > swizzle_unit:
      return False
    
    return True

  def check_mn_major_copy_constraints(self, blk_mn, blk_k, warp_cnt_mn, warp_cnt_k, thread_cnt, math_inst, byte_size):

    if not self.check_power_of_two(blk_mn):
      return False

    if blk_mn != 0 and (blk_mn % warp_cnt_mn != 0):
      return False
    warp_shape_mn = blk_mn // warp_cnt_mn

    ds_read_matrix_in_mn = 32

    if warp_shape_mn % ds_read_matrix_in_mn != 0:
      return False

    if blk_k % warp_cnt_k != 0:
      return False
    warp_shape_k = blk_k // warp_cnt_k
    if warp_shape_k % math_inst.instruction_shape[2] != 0:
      return False

    warp_major_lane = (blk_mn * byte_size) // self.ACCESS_SIZE

    if warp_major_lane == 0:
      return False

    warp_minor_lane = self.WARP_SIZE_GPU // warp_major_lane

    # Require a regular tile partition with one iteration along the major dimension.
    if warp_major_lane * warp_minor_lane != self.WARP_SIZE_GPU:
      return False

    warp_cnt_mnk = thread_cnt // self.WARP_SIZE_GPU 

    blk_minor_thread = warp_cnt_mnk * warp_minor_lane
    if (blk_k < blk_minor_thread) or (blk_k % blk_minor_thread != 0):
      return False

    # regularIterator requires a power-of-two kPointer count.
    swizzle_unit = 0
    if byte_size == 4:
      swizzle_unit = 1
    elif byte_size == 2:
      swizzle_unit = 1 if blk_mn == 32 else 2
    elif byte_size == 1:
      swizzle_unit = 1 if blk_mn == 32 else \
                     2 if blk_mn == 64 else 4
    else:
      return False

    strided_iteration_cnt = blk_k // (warp_cnt_mnk * warp_minor_lane)

    k_pointer_count = min(strided_iteration_cnt, swizzle_unit)

    if not self.check_power_of_two(k_pointer_count):
      return False

    return True

  def check_gemm_constraints(self, blk_shape, warp_cnt, math_inst, layouts, stage, smem_size, 
                             byte_size_abc, align_abc, buffer_access):
    blk_shape_m, blk_shape_n, blk_shape_k = blk_shape
    warp_cnt_m, warp_cnt_n, warp_cnt_k = warp_cnt
    align_a, align_b, align_c = align_abc
    inst_shape_m, inst_shape_n, inst_shape_k = math_inst.instruction_shape
    byte_size_a, byte_size_b, byte_size_c = byte_size_abc

    thread_cnt = (warp_cnt_m * warp_cnt_n * warp_cnt_k) * self.WARP_SIZE_GPU

    if (not self.check_power_of_two(warp_cnt_m)) or \
       (not self.check_power_of_two(warp_cnt_n)) or \
       (not self.check_power_of_two(warp_cnt_k)):
      return False

    for layout in layouts:
      # The generator swaps A/B and transposes their layouts.
      if layout[0] == LayoutType.RowMajor:
        if not self.check_k_major_copy_constraints(blk_shape_n, blk_shape_k, warp_cnt_n, warp_cnt_k, 
                                                   thread_cnt, math_inst, byte_size_a):
          return False
      if layout[0] == LayoutType.ColumnMajor:
        if not self.check_mn_major_copy_constraints(blk_shape_n, blk_shape_k, warp_cnt_n, warp_cnt_k, 
                                                    thread_cnt, math_inst, byte_size_a):
          return False
      if layout[1] == LayoutType.RowMajor:
        if not self.check_mn_major_copy_constraints(blk_shape_m, blk_shape_k, warp_cnt_m, warp_cnt_k, 
                                                    thread_cnt, math_inst, byte_size_a):
          return False
      if layout[1] == LayoutType.ColumnMajor:
        if not self.check_k_major_copy_constraints(blk_shape_m, blk_shape_k, warp_cnt_m, warp_cnt_k, 
                                                   thread_cnt, math_inst, byte_size_a):
          return False

      if not self.check_gemm_vgpr_constraint(blk_shape, warp_cnt, math_inst, byte_size_a, align_a, buffer_access):
        return False
      if not self.check_shared_memory_constraint(blk_shape, warp_cnt, stage, smem_size, byte_size_a):
        return False
      if not self.check_epilogue_constraints(blk_shape, warp_cnt, align_c, byte_size_c):
        return False

      if stage > 1:
        k_iterations = blk_shape_k // inst_shape_k // warp_cnt_k
        if k_iterations < 2:
          return False
      return True

  def check_conv_constraints(self, blk_shape, warp_cnt, math_inst, layouts, stage, smem_size, 
                             byte_size_abc, align_abc, conv_kinds, iterator_algorithms):
    blk_shape_m, blk_shape_n, blk_shape_k = blk_shape
    warp_cnt_m, warp_cnt_n, warp_cnt_k = warp_cnt
    align_a, align_b, align_c = align_abc
    inst_shape_m, inst_shape_n, inst_shape_k = math_inst.instruction_shape
    byte_size_a, byte_size_b, byte_size_c = byte_size_abc

    thread_cnt = (warp_cnt_m * warp_cnt_n * warp_cnt_k) * self.WARP_SIZE_GPU

    if (not self.check_power_of_two(warp_cnt_m)) or \
       (not self.check_power_of_two(warp_cnt_n)) or \
       (not self.check_power_of_two(warp_cnt_k)):
      return False

    for layout in layouts:
      if layout[0] == LayoutType.RowMajor and layout[1] == LayoutType.ColumnMajor:
        if ConvKind.Fprop in conv_kinds:
          if not self.check_k_major_copy_constraints(blk_shape_m, blk_shape_k, warp_cnt_m, warp_cnt_k,
                                                     thread_cnt, math_inst, byte_size_a) or \
             not self.check_k_major_copy_constraints(blk_shape_n, blk_shape_k, warp_cnt_n, warp_cnt_k,
                                                     thread_cnt, math_inst, byte_size_b):
            return False
      elif layout[0] == LayoutType.RowMajor and layout[1] == LayoutType.RowMajor:
        if ConvKind.Dgrad in conv_kinds:
          if not self.check_k_major_copy_constraints(blk_shape_m, blk_shape_k, warp_cnt_m, warp_cnt_k,
                                                      thread_cnt, math_inst, byte_size_a) or \
            not self.check_mn_major_copy_constraints(blk_shape_n, blk_shape_k, warp_cnt_n, warp_cnt_k,
                                                      thread_cnt, math_inst, byte_size_b):
            return False
      elif layout[0] == LayoutType.ColumnMajor and layout[1] == LayoutType.RowMajor:
        if ConvKind.Wgrad in conv_kinds:
          if not self.check_mn_major_copy_constraints(blk_shape_m, blk_shape_k, warp_cnt_m, warp_cnt_k,
                                                      thread_cnt, math_inst, byte_size_a) or \
             not self.check_mn_major_copy_constraints(blk_shape_n, blk_shape_k, warp_cnt_n, warp_cnt_k,
                                                      thread_cnt, math_inst, byte_size_b):
            return False

    if not self.check_conv_vgpr_constraint_conv(blk_shape, warp_cnt, math_inst, byte_size_a,
                                             align_a, conv_kinds, iterator_algorithms):
      return False

    if not self.check_shared_memory_constraint(blk_shape, warp_cnt, stage, smem_size, byte_size_a):
      return False

    if not self.check_epilogue_constraints(blk_shape, warp_cnt, align_c, byte_size_c):
      return False

    if stage > 1:
      k_iterations = blk_shape_k // inst_shape_k // warp_cnt_k
      if k_iterations < 2:
        return False
    return True

  def generate_gemm_tile_descriptions(self, tile_configs, math_insts, byte_size_abc, layouts, align_abc, buffer_access = True):
    tile_descriptions = []
    stages = tile_configs.stages
    min_cc = tile_configs.min_cc
    max_cc = tile_configs.max_cc
    smem_size = tile_configs.smem_size
    warp_count_mapping = tile_configs.warp_count_mapping
    thread_counts = [
      [thread_count, warp_count]
      for thread_count, warp_counts in warp_count_mapping.items()
      for warp_count in warp_counts
    ]

    threadblock_shapes = [
      [m, n, k] 
      for m, n, k in product(range(32, 513, 32), range(32, 513, 32), range(16, 257, 16))
    ]

    combinations = product(threadblock_shapes, stages, thread_counts, [math_insts], [min_cc], [max_cc])

    for threadblock_shape, stage, (thread_count, warp_count), math_inst, min_cc, max_cc in combinations:
      if not self.check_gemm_constraints(threadblock_shape, warp_count, math_inst, layouts, stage,
                                         smem_size, byte_size_abc, align_abc, buffer_access):
        continue

      tile_description = TileDescription(threadblock_shape, stage, warp_count, math_inst, min_cc, max_cc)
      tile_descriptions.append(tile_description)

    return tile_descriptions

  def generate_conv_tile_descriptions(self, tile_configs, math_insts, byte_size_abc, layouts, 
                                      align_abc, conv_kinds, iterator_algorithms):
    tile_descriptions = []
    stages = tile_configs.stages
    min_cc = tile_configs.min_cc
    max_cc = tile_configs.max_cc
    smem_size = tile_configs.smem_size
    warp_count_mapping = tile_configs.warp_count_mapping
    thread_counts = [
      [thread_count, warp_count]
      for thread_count, warp_counts in warp_count_mapping.items()
      for warp_count in warp_counts
    ]

    if ConvKind.Dgrad in conv_kinds or ConvKind.Wgrad in conv_kinds:
      raise Exception("generate_conv_tile_descriptions do not support Dgrad and Wgrad")

    threadblock_shapes = [
      [m, n, k] 
      for m, n, k in product(range(32, 513, 32), range(32, 513, 32), range(16, 257, 16))
    ]

    combinations = product(threadblock_shapes, stages, thread_counts, [math_insts], [min_cc], [max_cc])

    for threadblock_shape, stage, (thread_count, warp_count), math_inst, min_cc, max_cc in combinations:

      if not self.check_conv_constraints(threadblock_shape, warp_count, math_inst, layouts, stage, smem_size,
                                         byte_size_abc, align_abc, conv_kinds, iterator_algorithms):
        continue

      tile_description = TileDescription(threadblock_shape, stage, warp_count, math_inst, min_cc, max_cc)
      tile_descriptions.append(tile_description)

    return tile_descriptions

  def generate_conv_few_channels_tile_descriptions(self, tile_configs, math_insts, byte_size_abc, layouts, 
                                      align_abc, conv_kinds):
    tile_descriptions = []
    stages = tile_configs.stages
    min_cc = tile_configs.min_cc
    max_cc = tile_configs.max_cc
    smem_size = tile_configs.smem_size
    warp_count_mapping = tile_configs.warp_count_mapping
    thread_counts = [
      [thread_count, warp_count]
      for thread_count, warp_counts in warp_count_mapping.items()
      for warp_count in warp_counts
    ]

    if ConvKind.Dgrad in conv_kinds or ConvKind.Wgrad in conv_kinds:
      raise Exception("generate_conv_tile_descriptions do not support Dgrad and Wgrad")

    threadblock_shapes = [
      [m, n, k] 
      for m, n, k in product(range(32, 513, 32), range(32, 513, 32), range(16, 65, 16))
    ]

    byte_size_c = byte_size_abc[2]

    combinations = product(threadblock_shapes, stages, thread_counts, [math_insts], [min_cc], [max_cc])

    def deduce_align_c(blk_shape, thread_cnt, align_abc, epilogue_steps = 8):
      align_abc_tmp = list(align_abc)

      elements_per_thread = blk_shape[0] * blk_shape[1] // thread_cnt // epilogue_steps
      elements_per_thread = min(elements_per_thread, align_abc_tmp[2])
      elements_per_thread = elements_per_thread if (elements_per_thread == 0) or self.check_power_of_two(elements_per_thread) else 1

      align_c = min(elements_per_thread, min(8, self.ACCESS_SIZE // byte_size_c))
      align_abc_tmp[2] = align_c
      return align_abc_tmp

    for threadblock_shape, stage, (thread_count, warp_count), math_inst, min_cc, max_cc in combinations:
      align_abc_tmp = deduce_align_c(threadblock_shape, thread_count, align_abc)

      if align_abc_tmp[2] == 0:
        continue

      if not self.check_conv_constraints(threadblock_shape, warp_count, math_inst, layouts, stage, smem_size,
                                         byte_size_abc, align_abc_tmp, conv_kinds, [IteratorAlgorithm.FewChannels]):
        continue

      tile_description = TileDescription(threadblock_shape, stage, warp_count, math_inst, min_cc, max_cc)
      tile_descriptions.append(tile_description)

    return tile_descriptions

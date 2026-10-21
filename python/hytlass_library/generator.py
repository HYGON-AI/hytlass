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

"""
Utilities for enumerating HYTLASS library kernels
"""

import argparse
import enum
import logging
import os.path
import shutil
import argparse
import logging

from enum import Enum, auto
import sys
from itertools import product
import copy
from typing import Any, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)


# Certain usecases of hytlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using HYTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of hytlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-hytlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-hytlass-package-imports", action='store_true', required=False,
                     help="Disable use of hytlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `HYTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.HYTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the hytlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.HYTLASS_IGNORE_PACKAGE = _args.disable_hytlass_package_imports

try:
  if HYTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import hytlass_library")
  from hytlass_library.library import *
  from hytlass_library.manifest import *
  from hytlass_library.operator_builder import *
except ImportError:
  from library import *
  from manifest import *
  from operator_builder import *
###################################################################################################
#
def GenerateGfx906_Simt_8b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx906","8b")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx906()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"8b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    # # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)

#
def GenerateGfx906_Simt_16b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx906","16b")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx906()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"16b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    # # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)

#
def GenerateGfx906_Simt_32b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx906","32b")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx906()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"32b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    # # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)

#

###################################################################################################

#
def GenerateGfx906(manifest,dtk_version):
  GenerateGfx906_Simt_8b_gemm(manifest, dtk_version)
  GenerateGfx906_Simt_16b_gemm(manifest, dtk_version)
  GenerateGfx906_Simt_32b_gemm(manifest, dtk_version)

###################################################################################################
def GenerateGfx928_TensorOp_8b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx928","8b", "nn")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx928()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"8b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    stream_k_schedules = tile_configs.stream_k_schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])  
  
    # # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
      
#
def GenerateGfx928_TensorOp_16b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx928","16b", "nn")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx928()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"16b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    stream_k_schedules = tile_configs.stream_k_schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])  
  
    # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
      
#
def GenerateGfx928_TensorOp_32b_gemm(manifest, dtk_version):
  tile_configs = TileConfig("Gfx928","32b")
  layouts = tile_configs.current_layouts
  math_instructions = tile_configs.math_instructions
  tile_gen = TileGeneratorGfx928()
  for math_inst in math_instructions:
    tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,"32b",layouts)
    data_type = {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }

    # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
    for layout in layouts:
      layout[2][1] = 128 // DataTypeSize[data_type["c_type"]]   

    schedules = tile_configs.schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, schedules)

    stream_k_schedules = tile_configs.stream_k_schedules
    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])  
  
    # # for mixed precision kernels, also generate kernels that write output matrix in the A/B format
    # # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:
      data_type_mixed = {
        "a_type"   : math_inst.element_a,
        "b_type"   : math_inst.element_b,
        "c_type"   : math_inst.element_a,
        "d_type"   : math_inst.element_a,
        "acc_type" : math_inst.element_accumulator,
        "epi_type" : math_inst.element_accumulator
      }

      # Set alignment c based on Destination format. 128 / sizeof(c_type) / 8
      for layout in layouts:
        layout[2][1] = 128 // DataTypeSize[data_type_mixed["c_type"]]

      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, schedules)
      CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_type_mixed, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK])
      
#

###################################################################################################

#
def GenerateGfx928(manifest,dtk_version):
  GenerateGfx928_TensorOp_8b_gemm(manifest, dtk_version)
  GenerateGfx928_TensorOp_16b_gemm(manifest, dtk_version)
  GenerateGfx928_TensorOp_32b_gemm(manifest, dtk_version)
  
###################################################################################################

def GenerateGfx928_2x_TensorOp_32b_gemm(manifest, dtk_version):
  tile_configs = TileConfig_2x("Gfx928", "32b", "nn")
  layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Alignment must not exceed the maximum vectorized memory access length for this data type.
  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [4, 4, 4],
  #   [1, 1, 1],
  # ]

  enable_buffer_access = True

  tile_gen = TileGeneratorGfx928_2x()
  for math_inst in math_instructions:

    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[3]]))
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_gemm_tile_descriptions(tile_configs, math_inst, 
                                                                   byte_size_abc, layouts, 
                                                                   align_abc, enable_buffer_access)

      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False)
      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False,
                         swizzling_functor=SwizzlingFunctor.StreamK)


def GenerateGfx928_2x_TensorOp_32b_conv(manifest, dtk_version):
  # Fprop uses the TN GEMM MmaCore, so use the TN base configuration here.
  tile_configs = TileConfig_2x("Gfx928", "32b", "tn")
  tile_layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Only support Fprop for now
  conv_kinds = [ConvKind.Fprop, ]

  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [4, 4, 4],
  #   [1, 1, 1],
  # ]

  channel_cnts = [
    [1, 1, 1],
  ]

  tile_gen = TileGeneratorGfx928_2x()

  for math_inst in math_instructions:
    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    # Auxiliary-array register costs make valid convolution tiles depend on alignment.
    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        conv_kinds, [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type, [align_abc, ], [ConvKind.Fprop], EpilogueFunctor.LinearCombination) 

    # few channels
    for align in channel_cnts:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_few_channels_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        [ConvKind.Fprop, ]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
                                      data_type, [align_abc, ], conv_kinds)

def GenerateGfx928_2x_TensorOp_16b_gemm(manifest, dtk_version):
  tile_configs = TileConfig_2x("Gfx928", "16b", "nn")
  layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Alignment must not exceed the maximum vectorized memory access length for this data type.
  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [8, 8, 8],
  #   [1, 1, 1],
  # ]

  enable_buffer_access = True

  tile_gen = TileGeneratorGfx928_2x()
  for math_inst in math_instructions:

    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[3]]))
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_gemm_tile_descriptions(tile_configs, math_inst, 
                                                                   byte_size_abc, layouts, 
                                                                   align_abc, enable_buffer_access)

      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False)
      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False,
                         swizzling_functor=SwizzlingFunctor.StreamK)


def GenerateGfx928_2x_TensorOp_16b_conv(manifest, dtk_version):
  # Fprop uses the TN GEMM MmaCore, so use the TN base configuration here.
  tile_configs = TileConfig_2x("Gfx928", "16b", "tn")
  tile_layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Only support Fprop for now
  conv_kinds = [ConvKind.Fprop, ]

  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [8, 8, 8],
  #   [1, 1, 1],
  # ]

  channel_cnts = [
    [1, 1, 1],
  ]

  tile_gen = TileGeneratorGfx928_2x()

  for math_inst in math_instructions:
    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    # Auxiliary-array register costs make valid convolution tiles depend on alignment.
    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        conv_kinds, [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type, [align_abc, ], [ConvKind.Fprop], EpilogueFunctor.LinearCombination) 

    # few channels
    for align in channel_cnts:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_few_channels_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        [ConvKind.Fprop, ]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
                                      data_type, [align_abc, ], conv_kinds)

def GenerateGfx928_2x_TensorOp_8b_gemm(manifest, dtk_version):
  tile_configs = TileConfig_2x("Gfx928", "16b", "nn")
  layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Alignment must not exceed the maximum vectorized memory access length for this data type.
  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [16, 16, 8],
  #   [1, 1, 1],
  # ]

  enable_buffer_access = True

  tile_gen = TileGeneratorGfx928_2x()
  for math_inst in math_instructions:

    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[3]]))
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_gemm_tile_descriptions(tile_configs, math_inst, 
                                                                   byte_size_abc, layouts, 
                                                                   align_abc, enable_buffer_access)

      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False)
      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [align_abc, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False,
                         swizzling_functor=SwizzlingFunctor.StreamK)


def GenerateGfx928_2x_TensorOp_8b_conv(manifest, dtk_version):
  # Fprop uses the TN GEMM MmaCore, so use the TN base configuration here.
  tile_configs = TileConfig_2x("Gfx928", "8b", "tn")
  tile_layouts = tile_configs.layouts
  math_instructions = tile_configs.math_instructions

  # Only support Fprop for now
  conv_kinds = [ConvKind.Fprop, ]

  alignment_constraints = tile_configs.data_type_aligment
  # align_a, align_b, align_c
  # alignment_constraints = [
  #   [16, 16, 8],
  #   [1, 1, 1],
  # ]

  channel_cnts = [
    [1, 1, 1],
  ]

  tile_gen = TileGeneratorGfx928_2x()

  for math_inst in math_instructions:
    data_type = [
      math_inst.element_a,            # dataType of A
      math_inst.element_b,            # dataType of B
      math_inst.element_a,            # dataType of output
      math_inst.element_accumulator   # dataType of accum
    ]

    # Auxiliary-array register costs make valid convolution tiles depend on alignment.
    for align in alignment_constraints:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        conv_kinds, [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions,
        data_type, [align_abc, ], [ConvKind.Fprop], EpilogueFunctor.LinearCombination) 

    # few channels
    for align in channel_cnts:
      align_a, align_b, align_c = align
      align_c = min(align_c, min(8, 128 // DataTypeSize[data_type[2]])) 
      align_abc = [align_a, align_b, align_c]

      byte_size_abc = [
        DataTypeSize[data_type[0]] // 8, 
        DataTypeSize[data_type[1]] // 8, 
        DataTypeSize[data_type[2]] // 8
      ]

      tile_descriptions = tile_gen.generate_conv_few_channels_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layouts, align_abc,
        [ConvKind.Fprop, ]
      )

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)
      CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_descriptions,
                                      data_type, [align_abc, ], conv_kinds)

def GenerateGfx928_2x(manifest,dtk_version):
  GenerateGfx928_2x_TensorOp_32b_gemm(manifest, dtk_version)
  GenerateGfx928_2x_TensorOp_32b_conv(manifest, dtk_version)

  GenerateGfx928_2x_TensorOp_16b_gemm(manifest, dtk_version)
  GenerateGfx928_2x_TensorOp_16b_conv(manifest, dtk_version)

  GenerateGfx928_2x_TensorOp_8b_gemm(manifest, dtk_version)
  GenerateGfx928_2x_TensorOp_8b_conv(manifest, dtk_version)

###################################################################################################
def Generate_kernels_by_problems(maifest, dtk_version, src_problem_path):
  class GemmHeader(Enum):
    operation = 0
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    m = auto()
    n = auto()
    k = auto()
  
  class Conv2dHeader(Enum):
    operation = 0
    conv_kind = auto()
    Activation = auto()
    Filter = auto()
    Output = auto()
    n = auto()
    h = auto()
    w = auto()
    c = auto()
    k = auto()
    r = auto()
    s = auto()
    p = auto()
    q = auto()
    g = auto()
    pad_h = auto()
    pad_w = auto()
    stride_h = auto()
    stride_w = auto()
    dilation_h = auto()
    dilation_w = auto()

  def parse_type_and_layout(target_value):
    pair = target_value.split(":")
    if len(pair) != 2:
      raise ValueError(f"Invalid element and layout {target_value}")
    
    type_str, layout_str = pair
    type_enum = DataType[type_str.lower()]

    layout_enum = None

    if layout_str == "row":
      layout_enum = LayoutType.RowMajor
    elif layout_str == "column":
      layout_enum = LayoutType.ColumnMajor
    elif layout_str == "nhwc":
      layout_enum = LayoutType.TensorNHWC
    else:
      raise ValueError(f"Invalid layout {layout_str}. Support Only row, column or nhwc")
    return type_enum, layout_enum

  def get_alignment(target_dim, current_align):
    assert target_dim > 0 and current_align > 0
    mask = current_align - 1
    if (target_dim & mask) == 0:
        return current_align
    return target_dim & -target_dim

  float_types = {DataType.f32, DataType.tf32, DataType.f16, 
                 DataType.bf16, DataType.e4m3, DataType.e5m2}

  operation_descriptions = []

  problems_with_filter_name = os.path.basename(src_problem_path)
  helper_path = os.path.join(os.path.dirname(__file__), "../../scripts/profiler_helper")
  gemm_with_filter_path = os.path.normpath(rf"{helper_path}/gemm_{problems_with_filter_name}")
  conv2d_with_filter_path = os.path.normpath(rf"{helper_path}/conv2d_{problems_with_filter_name}")

  with open(src_problem_path, "r", encoding='utf-8') as src_file, \
       open(gemm_with_filter_path, "w", encoding='utf-8') as gemm_dst_file, \
       open(conv2d_with_filter_path, "w", encoding="utf-8") as conv2d_dst_file:

    csv_gemm_header = list(GemmHeader.__members__.keys())
    csv_conv2d_header = list(Conv2dHeader.__members__.keys())

    csv_gemm_header.append("kernels")
    csv_gemm_header.append("compute-capability")
    csv_conv2d_header.append("kernels")

    gemm_dst_file.write(",".join(csv_gemm_header)+"\n")
    conv2d_dst_file.write(",".join(csv_conv2d_header)+"\n")

    for line in src_file:
      if line is None or not line.strip():
        continue

      values = [value.strip() for value in line.strip().split(',')]

      if values[0] == "Gemm":
        if not len(values) == len(GemmHeader.__members__):
          raise ValueError(f"Number of params for Gemm does not match, {len(GemmHeader.__members__)} \
                           are required, but {len(values)} are provide")

        element_a, layout_a = parse_type_and_layout(values[GemmHeader.A.value])
        element_b, layout_b = parse_type_and_layout(values[GemmHeader.B.value])
        element_c, layout_c = parse_type_and_layout(values[GemmHeader.C.value])

        element_acc = DataType.f32 if element_a in float_types else DataType.s32

        m = int(values[GemmHeader.m.value])
        n = int(values[GemmHeader.n.value])
        k = int(values[GemmHeader.k.value])

        max_align_ab = 128 // DataTypeSize[element_a]
        max_align_c = 128 // DataTypeSize[element_c]

        align_a = get_alignment(k if layout_a == LayoutType.RowMajor else m, max_align_ab)
        align_b = get_alignment(n if layout_b == LayoutType.RowMajor else k, max_align_ab)
        align_c = get_alignment(n if layout_c == LayoutType.RowMajor else m, max_align_c)

        gemm_desc = GemmDescription(
          element_a, layout_a, align_a, element_b, layout_b, align_b,
          element_c, layout_c, align_c,
          element_c, element_acc, element_acc
        )

        if gemm_desc not in operation_descriptions:
          operation_descriptions.append(gemm_desc)
        
        # align_c does not need to be checked: for a GEMM problem, (align_a, align_b, layout, element_output) determines align_c.
        # For the few-channel iterator, align_c also depends on blk_shape.
        kernel_filter = f"*_align_a{gemm_desc.align_a}_b{gemm_desc.align_b}*"
        values.append(kernel_filter)
        values.append("75")

        gemm_dst_file.write(",".join(values)+"\n")

      elif values[0] == "Conv2d":
        if not len(values) == len(Conv2dHeader.__members__):
          raise ValueError(f"Number of params for Conv2d does not match, {len(Conv2dHeader.__members__)} \
                           are required, but {len(values)} are provide")
        
        conv_kind = ConvKind[values[Conv2dHeader.conv_kind.value].capitalize()]
        element_a, layout_a = parse_type_and_layout(values[Conv2dHeader.Activation.value])
        element_f, layout_f = parse_type_and_layout(values[Conv2dHeader.Filter.value])
        element_o, layout_o = parse_type_and_layout(values[Conv2dHeader.Output.value])
        element_acc = DataType.f32 if element_a in float_types else DataType.s32

        max_align_a = 128 // DataTypeSize[element_a]
        max_align_f = 128 // DataTypeSize[element_f]
        max_align_o = 128 // DataTypeSize[element_o]
        
        align_a = get_alignment(int(values[Conv2dHeader.c.value]), max_align_a)
        align_f = get_alignment(int(values[Conv2dHeader.c.value]), max_align_f)
        align_o = get_alignment(int(values[Conv2dHeader.k.value]), max_align_o)

        enable_few_channel = (
            conv_kind == ConvKind.Fprop and (
              (DataTypeSize[element_a] == 8 and int(values[Conv2dHeader.c.value]) <= 64)
              or int(values[Conv2dHeader.c.value]) <= 32
            )
        )

        conv2d_desc = Conv2dDescription(conv_kind, element_a, layout_a, align_a,
                                        element_f, layout_f, align_f,
                                        element_o, layout_o, align_o,
                                        element_acc, element_acc, enable_few_channel)

        if not conv2d_desc in operation_descriptions:
          operation_descriptions.append(conv2d_desc)

        kernel_filter = f"*_align_a{conv2d_desc.align_a}_b{conv2d_desc.align_b}*"
        values.append(kernel_filter)

        conv2d_dst_file.write(",".join(values)+"\n")
      else:
        raise ValueError(f"parse faild, check data")


  for item in operation_descriptions:
    item.create_operation(manifest)

###################################################################################################

def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level into the numeric identifier used
  in setting the log level

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level


# This function for defining the ArgumentParser is used to make it easy for the HYTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for HYTLASS Kernels")
  parser.add_argument("--operations", default="all", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default=".", required=False, help="HYTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default=".", help="HYTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of HYTLASS Library Generator.")
  parser.add_argument("--architectures", default='906;928;936', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
  parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--dtk-version", default="25.10", help="Semantic version string of DTK")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--interface-dir", default=None, required=False, help="Interface header to kernels")
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  parser.add_argument("--problem-size-path", required=False, help="Gererate kernels for given problems")
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()

  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)
  manifest = Manifest(args)

  if args.problem_size_path is not None and len(args.problem_size_path) != 0:
    _LOGGER.info(f"Generate kernels by given problems, path is {args.problem_size_path}")
    Generate_kernels_by_problems(manifest, args.dtk_version, args.problem_size_path)
  else:
    GenerateGfx906(manifest, args.dtk_version)
    GenerateGfx928(manifest, args.dtk_version)
    GenerateGfx928_2x(manifest, args.dtk_version)

  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################

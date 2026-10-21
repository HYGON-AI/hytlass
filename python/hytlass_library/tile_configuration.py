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
from copy import deepcopy

try:
  import builtins
  if hasattr(builtins, "HYTLASS_IGNORE_PACKAGE") and HYTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import hytlass_library")
  from hytlass_library.library import *
except ImportError:
  from library import *

class TileConfig:
  device_config_params = {
    "Gfx906": {
        "min_cc": 906,
        "max_cc": 1024,
        "smem_size": 65536,
        "stages": [2],
        "cluster_shapes": [[1,1,1]],
        "warp_count_mapping":{
            64: [[1, 1, 1]],
            128: [[2, 1, 1], [1, 2, 1]],
            256: [[2, 2, 1], [1, 4, 1], [4, 1, 1]]
        },
        "schedules": [
            [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]
          ]
    },
    "Gfx928": {
        "min_cc": 928,
        "max_cc": 1024,
        "smem_size": 65536,
        "stages": [2],
        "cluster_shapes": [[1,1,1]],
        "warp_count_mapping":{
            64: [[1, 1, 1]],
            128: [[2, 1, 1], [1, 2, 1]],
            256: [[2, 2, 1], [1, 4, 1], [4, 1, 1]],
            512: [[2, 4, 1], [4, 2, 1], [1, 8, 1], [8, 1, 1]]
        },
        "schedules": [
            [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto],
            [KernelScheduleType.ScheduleAuto, EpilogueScheduleType.NoSmemWarpSpecialized]
          ],
        "stream_k_schedules": [
            [KernelScheduleType.StreamK, EpilogueScheduleType.ScheduleAuto]
          ]
    }
  }

  data_type_math_instructions_params = {
    "Gfx906": {
        "8b": [
          MathInstruction(
            [1, 1, 4],
            DataType.s8, DataType.s8, DataType.s32,
            OpcodeClass.Simt,
            MathOperation.multiply_add),
        ],
        "16b": [
          MathInstruction(
            [1, 1, 2],
            DataType.f16, DataType.f16, DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add),
        ],
        "32b": [
          MathInstruction(
            [1, 1, 1],
            DataType.f32, DataType.f32, DataType.f32,
            OpcodeClass.Simt,
            MathOperation.multiply_add),
        ]
  },
    "Gfx928": {
        "8b": [
          MathInstruction(
            [32, 32, 32],
            DataType.s8, DataType.s8, DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [32, 32, 32],
            DataType.u8, DataType.u8, DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ],
        "16b": [
          MathInstruction(
            [32, 32, 16],
            DataType.f16, DataType.f16, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [32, 32, 16],
            DataType.bf16, DataType.bf16, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ],
        "32b": [
          MathInstruction(
            [16, 16, 8],
            DataType.f32, DataType.f32, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [16, 16, 8],
            DataType.tf32, DataType.tf32, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ]
        }
  }

  data_type_aligment_params = {
    "8b": [16],
    "16b": [8],
    "32b": [4]
  }

  layout_mappings = {
    "nt": [[LayoutType.ColumnMajor, 1], [LayoutType.RowMajor, 1], [LayoutType.ColumnMajor, 1]],
    "nn": [[LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
    "tt": [[LayoutType.RowMajor, 1], [LayoutType.RowMajor, 1], [LayoutType.ColumnMajor, 1]],
    "tn": [[LayoutType.RowMajor, 1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
  }

  instruction_mappings = {
    "nt": {
      "16b": [
        MathInstruction(
          [32, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [32, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "nn": {
      "16b": [
        MathInstruction(
          [32, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add)
      ],
      "8b": [
        MathInstruction(
          [32, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "tt": {
      "16b": [
        MathInstruction(
          [32, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [32, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "tn": {
      "16b": [
        MathInstruction(
          [16, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [16, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 64],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 64],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 64],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 64],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 64],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 32, 64],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 32, 64],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [32, 16, 64],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    }
  }

  def __init__(self, device_name, data_type, layout=""):

    device_config = self.device_config_params.get(device_name)
    if device_config is None:
        raise ValueError(f"Unsupported device: {device_name}")
    self.min_cc = device_config["min_cc"]
    self.max_cc = device_config["max_cc"]
    self.smem_size = device_config["smem_size"]
    self.stages = device_config["stages"]
    self.cluster_shapes = device_config["cluster_shapes"]
    self.warp_count_mapping = device_config["warp_count_mapping"]
    self.schedules = device_config["schedules"]
    self.stream_k_schedules = device_config.get("stream_k_schedules", [])
    
    if layout != "" :
      if layout not in ["nt", "nn", "tn", "tt"]:
        raise ValueError(f"Unsupported layout type: {layout}")
      math_instructions = self.instruction_mappings.get(layout).get(data_type)
    else :
      math_instructions = self.data_type_math_instructions_params.get(device_name,{}).get(data_type)
    if math_instructions is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    self.math_instructions = math_instructions

    data_type_aligment = self.data_type_aligment_params.get(data_type)
    if data_type_aligment is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    if layout != "" :
      self.current_layouts = []
      for alignment in data_type_aligment:
        raw_layout = deepcopy(self.layout_mappings.get(layout))
        raw_layout[0][1] = alignment
        raw_layout[1][1] = alignment
        self.current_layouts.append(raw_layout)
    else :
      self.current_layouts = self.layout_aligment_config(data_type_aligment)
  
  def layout_aligment_config(self, data_type_aligment):
    layouts = [
      [[LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
      [[LayoutType.ColumnMajor, 1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
      [[LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1], [LayoutType.ColumnMajor, 1]],
      [[LayoutType.RowMajor,    1], [LayoutType.RowMajor,    1], [LayoutType.ColumnMajor, 1]],
    ]

    if data_type_aligment :
      current_layouts = []
      for aligment in data_type_aligment:
        for layout in layouts:
           new_layout = [list(layout[0]), list(layout[1]), list(layout[2])]
           new_layout[0][1] = aligment
           new_layout[1][1] = aligment
           current_layouts.append(new_layout)
      return current_layouts
    else:
      return layouts

    


class TileConfig_2x:

  device_config_params = {
    "Gfx928": {
        "min_cc": 928,
        "max_cc": 1024,
        "smem_size": 65536,
        "stages": [1],
        "warp_count_mapping":{
            64:  [[1, 1, 1]],
            128: [[2, 1, 1], [1, 2, 1], [1, 1, 2]],
            256: [[2, 2, 1], [1, 4, 1], [4, 1, 1], [2, 1, 2], [1, 2, 2], [1, 1, 4]],
            512: [[2, 4, 1], [4, 2, 1], [1, 8, 1], [8, 1, 1], [4, 1, 2], [2, 2, 2], [1, 4, 2]]
        },
    }
  }

  data_type_math_instructions_params = {
    "Gfx928": {
        "8b": [
          MathInstruction(
            [16, 16, 32],
            DataType.s8, DataType.s8, DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [16, 16, 32],
            DataType.u8, DataType.u8, DataType.s32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ],
        "16b": [
          MathInstruction(
            [16, 16, 16],
            DataType.f16, DataType.f16, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [16, 16, 16],
            DataType.bf16, DataType.bf16, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ],
        "32b": [
          MathInstruction(
            [16, 16, 8],
            DataType.f32, DataType.f32, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
          MathInstruction(
            [16, 16, 8],
            DataType.tf32, DataType.tf32, DataType.f32,
            OpcodeClass.TensorOp,
            MathOperation.multiply_add),
        ]
      }
  }

  data_type_aligment_params = {
    "8b" : [[16, 16, 8]],
    "16b": [[8, 8, 8]],
    "32b": [[4, 4, 4]]
  }

  layout_mappings = {
    "nt" :  (LayoutType.ColumnMajor, LayoutType.RowMajor,    LayoutType.ColumnMajor),
    "tt" :  (LayoutType.RowMajor,    LayoutType.RowMajor,    LayoutType.ColumnMajor),
    "tn" :  (LayoutType.RowMajor,    LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    "nn" :  (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
  }

  instruction_mappings = {
    "nt": {
      "32b": [
        MathInstruction(
          [16, 16, 8],
          DataType.f32, DataType.f32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 8],
          DataType.tf32, DataType.tf32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "16b": [
        MathInstruction(
          [16, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [16, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "tt": {
      "32b": [
        MathInstruction(
          [16, 16, 8],
          DataType.f32, DataType.f32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 8],
          DataType.tf32, DataType.tf32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "16b": [
        MathInstruction(
          [16, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [16, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "nn": {
      "32b": [
        MathInstruction(
          [16, 16, 8],
          DataType.f32, DataType.f32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 8],
          DataType.tf32, DataType.tf32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],

      "16b": [
        MathInstruction(
          [16, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "8b": [
        MathInstruction(
          [16, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
    "tn": {
      "32b": [
        MathInstruction(
          [16, 16, 8],
          DataType.f32, DataType.f32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 8],
          DataType.tf32, DataType.tf32, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
      "16b": [
        MathInstruction(
          [16, 16, 16],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 16],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.f16, DataType.f16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.bf16, DataType.bf16, DataType.f32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),

      ],
      "8b": [
        MathInstruction(
          [16, 16, 32],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 32],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 64],
          DataType.s8, DataType.s8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
        MathInstruction(
          [16, 16, 64],
          DataType.u8, DataType.u8, DataType.s32,
          OpcodeClass.TensorOp,
          MathOperation.multiply_add),
      ],
    },
  }

  def __init__(self, device_name, data_type, layout=""):

    device_config = self.device_config_params.get(device_name)
    if device_config is None:
        raise ValueError(f"Unsupported device: {device_name}")
    self.min_cc = device_config["min_cc"]
    self.max_cc = device_config["max_cc"]
    self.smem_size = device_config["smem_size"]
    self.stages = device_config["stages"]
    self.warp_count_mapping = device_config["warp_count_mapping"]
    
    if layout != "" :
      if layout not in ["nt", "nn", "tn", "tt"]:
        raise ValueError(f"Unsupported layout type: {layout}")
      math_instructions = self.instruction_mappings.get(layout).get(data_type)
    else :
      math_instructions = self.data_type_math_instructions_params.get(device_name,{}).get(data_type)
    if math_instructions is None:
        raise ValueError(f"Unsupported data type: {data_type}")
    self.math_instructions = math_instructions

    self.data_type_aligment = self.data_type_aligment_params.get(data_type)

    if self.data_type_aligment is None:
        raise ValueError(f"Unsupported data type: {data_type}")

    self.layouts = []
    if layout != "":
      self.layouts.append(self.layout_mappings.get(layout))
    else:
      for _layout in range(["nt", "tn", "nn", "tt"]):
        self.layouts.append(self.layout_mappings.get(_layout))

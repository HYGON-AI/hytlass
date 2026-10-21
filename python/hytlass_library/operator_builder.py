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
from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple

try:
  import builtins
  if hasattr(builtins, "HYTLASS_IGNORE_PACKAGE") and HYTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import hytlass_library")
  from hytlass_library.library import *
  from hytlass_library.manifest import *
except ImportError:
  from library import *
  from manifest import *

_LOGGER = logging.getLogger(__name__)

def logging_prefix(indent_level: int = 0) -> str:
  """String prefix for start of each debug log entry"""
  prefix = '*** '
  indent = '  '
  return f"{prefix}{indent_level * indent}"

def log_debug_line(line: str, indent_level: int = 0) -> None:
  """Log one line of debug output"""
  prefix = logging_prefix(indent_level)
  _LOGGER.debug(prefix + line)

#
def DTKVersionSatisfies(semantic_ver_string, major, minor, patch = 0):

  # by default, use the latest DTK version
  dtk_version = [11, 0, 132]

  # Update dtk_version based on parsed string
  if semantic_ver_string != '':
    for i, x in enumerate([int(x) for x in semantic_ver_string.split('.')]):
      if i < len(dtk_version):
        dtk_version[i] = x
      else:
        dtk_version.append(x)
  return dtk_version >= [major, minor, patch]


###################################################################################################
###################################################################################################

#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 64 // epilogue_steps
  result = min(max_alignment, elements_per_thread)

  if result == 0:
    return 0
  elif (result & (result - 1)) == 0:
    return result
  else:
    return 1

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8;
    # return SwizzlingFunctor.StreamK
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

#
def CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = DefaultSwizzlingFunctor(), BufferAccess = True, EnStaggerK = False):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:
            assert(isinstance(alignment, (tuple, list)) and len(alignment) == 3)
            alignment_a, alignment_b, alignment_c = alignment
            alignment_c = min(8, alignment_c)

            A = TensorDescription(element_a, layout[0], alignment_a, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment_b, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Universal, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor,
              BufferAccess = BufferAccess, EnStaggerK = EnStaggerK)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    complex_transforms=None,
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Regular]):

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedules:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none), ]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, complex_transforms, schedules, tile_schedulers)
  for layout, tile_description, data_type, complex_transform, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1], complex_transform[0])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1], complex_transform[1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    gemm_op_extra_args = {}
    gemm_kind = GemmKind.Universal3x
    element_compute = data_type.get("epi_type", data_type["acc_type"])
 
    operation = GemmOperation(
        gemm_kind, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler, **gemm_op_extra_args)

    manifest.append(operation)
    operations.append(operation)

  return operations

#
def CreateSparseGemmOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  gemm_kinds = [GemmKind.Sparse]

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GemmOperation(GemmKind.Sparse, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations


#
def CreateGemmGroupedOperator(manifest, layouts, tile_descriptions, data_type, \
  alignment_constraints, complex_transforms = None, epilogue_functor = EpilogueFunctor.LinearCombination, \
  swizzling_functor = SwizzlingFunctor.Identity8):

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none),]

  element_a, element_b, element_c, element_epilogue = data_type

  operations = []

  # by default, only generate the largest tile and largest alignment
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  for layout in layouts:
    for tile_description in tile_descriptions:
      for alignment in alignment_constraints:
        for complex_transform in complex_transforms:

            alignment_c = min(8, alignment)

            A = TensorDescription(element_a, layout[0], alignment, complex_transform[0])
            B = TensorDescription(element_b, layout[1], alignment, complex_transform[1])
            C = TensorDescription(element_c, layout[2], alignment_c)

            new_operation = GroupedGemmOperation(GemmKind.Grouped, tile_description.minimum_compute_capability, \
              tile_description, A, B, C, element_epilogue, epilogue_functor, swizzling_functor)

            manifest.append(new_operation)
            operations.append(new_operation)

  return operations

###########################################################################################################
#   ConvolutionOperator support variations
#        ____________________________________________________________________
#         ConvolutionalOperator |      Analytic          |    Optimized
#        ____________________________________________________________________
#        |       Fprop          |     (strided)          |    (strided)
#        |       Dgrad          |     (strided, unity*)  |    (strided, unity)
#        |       Wgrad          |     (strided)          |    (strided)
#        ____________________________________________________________________
#
# Note :  Operator marked (*) are supported but not generated to keep the instantiated kernel count low
###########################################################################################################
# Convolution for 2D operations
def CreateConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:
      assert(isinstance(alignment, (list, tuple)) and len(alignment) == 3)
      alignment_a, alignment_b, alignment_c = alignment
      alignment_c = min(8, alignment_c)

      A = TensorDescription(element_a, layout[0], alignment_a)
      B = TensorDescription(element_b, layout[1], alignment_b)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operations = [
            # None grouped kernel
            Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_),
          ]

          # Instance group conv kernel
          if tile.math_instruction.opcode_class == OpcodeClass.TensorOp and A.layout == LayoutType.TensorNHWC and \
            tile.minimum_compute_capability >= 80:
            # SingleGroup kernel
            new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
              A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.SingleGroup))

            # Analytic iterator supports MultipleGroup mode
            if iterator_algorithm == IteratorAlgorithm.Analytic:
              new_operations.append(Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_, group_mode=GroupMode.MultipleGroup))

          for new_operation in new_operations:
            manifest.append(new_operation)
            operations.append(new_operation)

      #
      # Conv2d Dgrad
      #
      if ConvKind.Dgrad in conv_kinds:

        # Unity stride for Analytic and Optimized Dgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Analytic Dgrad
        # strided dgrad uses a special threadblock swizzle
        # note that SwizzlingFunctor.StridedDgradHorizontal might be
        # better for problem sizes with large activation channel count
        swizzling_functor_strided_dgrad_ = SwizzlingFunctor.StridedDgradIdentity1

        if IteratorAlgorithm.Analytic in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

        # Strided support for Optimized Dgrad
        if IteratorAlgorithm.Optimized in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_strided_dgrad_)

          manifest.append(new_operation)
          operations.append(new_operation)

      #
      # Conv2d Wgrad
      #
      if ConvKind.Wgrad in conv_kinds:

        # Strided support for Analytic and Optimized Wgrad
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFixedChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized) 
  iterator_algorithms = [IteratorAlgorithm.FixedChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []



  for tile in tile_descriptions:
    for channel_count in channel_counts:
      assert(isinstance(channel_count, (list, tuple)) and len(channel_count) == 3)
      alignment_a, alignment_b, alignment_c = channel_count
      alignment_c = EpilogueAlignment(alignment_c, tile)

      if alignment_c == 0:
        continue

      A = TensorDescription(element_a, layout[0], alignment_a)
      B = TensorDescription(element_b, layout[1], alignment_b)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 2D operations specialized for few channels
def CreateConv2dFewChannelsOperator(manifest, layout, tile_descriptions, data_type, channel_counts, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.FewChannels,]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    channel_counts = [channel_counts[0],]

  operations = []

  for tile in tile_descriptions:
    for channel_count in channel_counts:
      assert(isinstance(channel_count, (tuple, list)) and len(channel_count)==3)
      alignment_a, alignment_b, alignment_c = channel_count
      alignment_c = EpilogueAlignment(alignment_c, tile)

      if alignment_c == 0:
        continue

      A = TensorDescription(element_a, layout[0], alignment_a)
      B = TensorDescription(element_b, layout[1], alignment_b)
      C = TensorDescription(element_c, layout[2], alignment_c)
      swizzling_functor_ = swizzling_functor

      #
      # Conv2d Fprop
      #
      if ConvKind.Fprop in conv_kinds:

        # Strided support for Analytic and Optimized Fprop
        for iterator_algorithm in iterator_algorithms:
          new_operation = Conv2dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
            A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor, swizzling_functor_)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

# Convolution for 3D operations
def CreateConv3dOperator(manifest, layout, tile_descriptions, data_type, alignment, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], epilogue_functor = EpilogueFunctor.LinearCombination):

  element_a, element_b, element_c, element_epilogue = data_type

  # one exceptional case
  alignment_c = min(8, alignment)

  # iterator algorithm (analytic and optimized)
  iterator_algorithms = [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size and optimized iterators
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    iterator_algorithms = [IteratorAlgorithm.Optimized]

  operations = []

  # All tile sizes for Conv3dFprop and Conv3dWgrad
  for tile in tile_descriptions:
    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Fprop
    #
    if ConvKind.Fprop in conv_kinds:
      # Strided support for Analytic and Optimized Fprop
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Fprop, iterator_algorithm, tile.minimum_compute_capability, tile,\
                                        A, B, C, element_epilogue, StrideSupport.Strided)
        manifest.append(new_operation)
        operations.append(new_operation)
    #
    # Conv3d Wgrad
    #
    if ConvKind.Wgrad in conv_kinds:

      # Strided support for Analytic and Optimized Wgrad
      for iterator_algorithm in iterator_algorithms:
        new_operation = Conv3dOperation(ConvKind.Wgrad, iterator_algorithm, tile.minimum_compute_capability, tile,\
          A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)
        manifest.append(new_operation)
        operations.append(new_operation)

  # All tile sizes for Conv3dDgrad
  for tile in tile_descriptions:

    A = TensorDescription(element_a, layout, alignment)
    B = TensorDescription(element_b, layout, alignment)
    C = TensorDescription(element_c, layout, alignment_c)

    #
    # Conv3d Dgrad
    #
    if ConvKind.Dgrad in conv_kinds:
      # Unity stride for Optimized Dgrad
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Optimized, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Unity, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

      # Strided support for Analytic Dgrad
      # Conv3dDgrad has a naive strided support which does not cut down redundant MMAs
      new_operation = Conv3dOperation(ConvKind.Dgrad, IteratorAlgorithm.Analytic, tile.minimum_compute_capability, tile,\
        A, B, C, element_epilogue, StrideSupport.Strided, epilogue_functor)

      manifest.append(new_operation)
      operations.append(new_operation)

  return operations

# Convolution for Depthwise 2d conv
def CreateDepthwiseConv2dOperator(manifest, layout, tile_descriptions, data_type, alignment_constraints, \
  conv_kinds = [ConvKind.Fprop, ConvKind.Dgrad, ConvKind.Wgrad], \
  epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity4):

  element_a, element_b, element_c, element_epilogue = data_type

  # iterator algorithm (FixedStrideDilation, Optimized)
  iterator_algorithms = [IteratorAlgorithm.FixedStrideDilation, IteratorAlgorithm.Optimized]

  # by default, only generate the largest tile size, largest alignment, and optimized iterator
  if manifest.kernel_filter == '':
    tile_descriptions = [tile_descriptions[0],]
    alignment_constraints = [alignment_constraints[0],]

  operations = []

  for tile in tile_descriptions:
    for alignment in alignment_constraints:

      alignment_c = min(8, alignment)

      A = TensorDescription(element_a, layout[0], alignment)
      B = TensorDescription(element_b, layout[1], alignment)
      C = TensorDescription(element_c, layout[2], alignment_c)

      swizzling_functor_ = swizzling_functor

      if ConvKind.Fprop in conv_kinds:

        # Strided support for Optimized and FixedStridedDilation Depthwise Conv
        for iterator_algorithm in iterator_algorithms:
          stride_support = StrideSupport.Strided
          if iterator_algorithm == IteratorAlgorithm.FixedStrideDilation:
              if tile.stride == [-1, -1] or tile.dilation == [-1,-1]:
                continue
              stride_support = StrideSupport.Fixed

          if iterator_algorithm == IteratorAlgorithm.Optimized:
              if tile.stride != [-1, -1] or tile.dilation != [-1,-1]:
                continue
          new_operation = Conv2dOperation(ConvKind.Fprop,
                                          iterator_algorithm,
                                          tile.minimum_compute_capability,
                                          tile,
                                          A, B, C,
                                          element_epilogue,
                                          stride_support,
                                          epilogue_functor,
                                          swizzling_functor_,
                                          group_mode=GroupMode.Depthwise)

          manifest.append(new_operation)
          operations.append(new_operation)

  return operations

class ConvOperation3x:
  """All parameters of a HYTLASS 3 convolution operation.

  Unlike HYTLASS 2 convolutions, HYTLASS 3 convolutions do not
  distinguish between 2-D and 3-D convolutions by kernel class name.
  Instead, for HYTLASS 3 convolutions, the tensor layouts encode
  whether the convolution is 2-D or 3-D.  Thus, this class deduces
  the OperationKind (either Conv2d or Conv3d) from the layouts,
  rather than taking it as a constructor parameter.
  """
  def __init__(self,
               conv_kind: ConvKind,
               tile_description: TileDescription,
               A: TensorDescription,
               B: TensorDescription,
               C: TensorDescription,
               element_compute: Optional[DataType] = None,
               D: Optional[TensorDescription] = None,
               kernel_schedule: KernelScheduleType = KernelScheduleType.ScheduleAuto,
               epilogue_schedule: EpilogueScheduleType = EpilogueScheduleType.ScheduleAuto,
               tile_scheduler: TileSchedulerType = TileSchedulerType.Default,
               log_indent_level: int = 1):
    log_debug_line(f'ConvOperation3x::init: conv_kind: {conv_kind}', log_indent_level)
    log_indent_level = log_indent_level + 1

    self.conv_kind = conv_kind
    self.tile_description = tile_description
    self.A = A
    self.B = B
    self.C = C
    self.element_compute = C.element if element_compute is None else element_compute
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule

    self.arch = tile_description.minimum_compute_capability
    self.tile_scheduler = tile_scheduler
    if D == None:
      self.D = C
    else:
      self.D = D

    self.is_3x = True
    self.group_mode = GroupMode.NoneGroup # HYTLASS 3 convolutions currently aren't grouped

    operation_kind = None
    for layout in (A.layout, B.layout, C.layout):
      assert(isinstance(layout, LayoutType))
      new_operation_kind = convolution_tensor_layout_type_to_operation_kind(layout)
      if operation_kind is None:
        operation_kind = new_operation_kind
      else: # HYTLASS 3 convolutions don't permit mixing 2-D and 3-D layouts.
        assert(operation_kind == new_operation_kind)
    assert(operation_kind is not None)
    self.operation_kind = operation_kind

  def __str__(self):
    return f"ConvOperation3x: operation_kind={self.operation_kind}, conv_kind={self.conv_kind}, tile_description={self.tile_description}"

  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  def is_mixed_input(self):
    return self.A.element != self.B.element

  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator
    if self.is_complex():
      return get_complex_from_real(accum)
    return accum

  def short_math_name(self):
    prefix = ''
    if self.tile_description.math_instruction.math_operation == MathOperation.multiply_add_complex_gaussian:
      prefix = 'g'
    return prefix + ShortDataTypeNames[self.accumulator_type()]

  def is_tensor_op(self):
    tensor_ops = [
      OpcodeClass.TensorOp
    ]
    return self.tile_description.math_instruction.opcode_class in tensor_ops

  def instruction_shape_string(self):
    math_operations_map = {
      MathOperation.xor_popc: 'xor',
      MathOperation.and_popc: 'and'
    }
    if self.is_tensor_op():
      is0, is1, is2 = self.tile_description.math_instruction.instruction_shape
      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = math_operations_map[math_op] if math_op in math_operations_map.keys() else ''
      return f"{is0}x{is1}x{is2}{math_op_string}"
    else:
      return ''

  def intermediate_type_string(self):
    '''
    Name of the distinct intermediate type used by the tensor operation,
    or the empty string if none.

    Tensor ops (opcode_clas *TensorOp) may use an intermediate data type
    that differs from the element type of A or the accumulator type.
    '''
    if not self.is_tensor_op():
      return ''
    elif self.tile_description.math_instruction.element_a == self.A.element:
      return ''
    elif self.tile_description.math_instruction.element_a == self.tile_description.math_instruction.element_accumulator:
      return ''
    else:
      return DataTypeNames[self.tile_description.math_instruction.element_a]

  def core_name(self):
    inst_shape = self.instruction_shape_string()
    intermediate_type = self.intermediate_type_string()
    conv_kind_name = ConvKindNames[self.conv_kind]
    return f"{self.short_math_name()}{inst_shape}{intermediate_type}{conv_kind_name}"

  def extended_name(self):
    core_name = self.core_name()
    element_a = DataTypeNames[self.A.element]
    element_b = DataTypeNames[self.B.element]
    element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator]
    element_c = DataTypeNames[self.C.element]
    element_d = DataTypeNames[self.D.element]
    return f"{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}"

  def is_complex(self):
    complex_operators = [
      MathOperation.multiply_add_complex,
      MathOperation.multiply_add_complex_gaussian,
      MathOperation.multiply_add_complex_fast_f32
    ]
    return self.tile_description.math_instruction.math_operation in complex_operators

  def layout_names(self):
    '''Layout strings for A and B, respectively'''
    if self.is_complex():
      return (ShortComplexLayoutNames[(self.A.layout, self.A.complex_transform)],
              ShortComplexLayoutNames[(self.B.layout, self.B.complex_transform)])
    else:
      return (ShortLayoutTypeNames[self.A.layout],
              ShortLayoutTypeNames[self.B.layout])

  def extended_name(self):
    core_name = self.core_name()
    element_a = DataTypeNames[self.A.element]
    element_b = DataTypeNames[self.B.element]
    element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator]
    element_c = DataTypeNames[self.C.element]
    element_d = DataTypeNames[self.D.element]
    layout_a, layout_b = self.layout_names()
    return f"{core_name}_{element_a}{layout_a}_{element_b}{layout_b}_{element_acc}_{element_c}_{element_d}"

  def configuration_name(self):
    prefix = 'hytlass3x'
    arch = self.arch
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    tbm = self.tile_description.tile_shape[0]
    tbn = self.tile_description.tile_shape[1]
    tbk = self.tile_description.tile_shape[2]
    cm = self.tile_description.cluster_shape[0]
    cn = self.tile_description.cluster_shape[1]
    ck = self.tile_description.cluster_shape[2]
    alignment = max(self.A.alignment, self.B.alignment)
    tile_scheduler = TileSchedulerSuffixes[self.tile_scheduler]
    kernel_schedule = KernelScheduleSuffixes[self.kernel_schedule]
    epilogue_schedule = EpilogueScheduleSuffixes[self.epilogue_schedule]

    return f"{prefix}_gfx{arch}_{opcode_class_name}_{self.extended_name()}_{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{self.tile_description.stages}_align{alignment}{tile_scheduler}{kernel_schedule}{epilogue_schedule}"

  def procedural_name(self):
    return self.configuration_name()

def convolution_tensor_layout_type_to_operation_kind(layout: LayoutType) -> OperationKind:
  if layout == LayoutType.TensorNHWC or layout == LayoutType.TensorKCSR:
    return OperationKind.Conv2d
  elif layout == LayoutType.TensorNDHWC or layout == LayoutType.TensorKCSRT:
    return OperationKind.Conv3d
  else:
    raise RuntimeError(f'LayoutType {layout} does not have a corresponding OperationKind')

def CreateConvOperator3x(manifest: Manifest,
                         dims_and_alignments: Sequence[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]],
                         tile_descriptions: Sequence[Sequence[TileDescription]],
                         data_types,
                         schedule_pairs: Sequence[Tuple[KernelScheduleType, KernelScheduleType]] = \
                           [(KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto)],
                         complex_transforms: Optional[Sequence[ComplexTransform]] = None,
                         tile_schedulers: Sequence[TileSchedulerType] = [TileSchedulerType.Persistent],
                         conv_kind: ConvKind = ConvKind.Fprop,
                         log_indent_level: int = 1):
  """
  Create zero or more HYTLASS 3 two-dimensional convolution operators.

  Create a HYTLASS 3 two-dimensional convolution operator
  for all feasible combinations of the input parameters.
  Add the operators to the manifest.

  dims_and_alignments: 3-level list.  Each outer list term is a list [A, B, C].
    Each inner list (A, B, or C) has the form [num_spatial_dimensions, alignment].
    Both are integers; the first is the number of spatial dimensions
    (currently, only 2 or 3 are supported), and the second is the byte alignment.
    We deduce the operation_kind (either OperationKind.Conv2d or OperationKind.Conv3d)
    from num_spatial_dimensions.

  This function doesn't take layouts, unlike the GEMM functions.
  HYTLASS 3 convolutions currently support three input layouts:

  * TensorNWC for 1-D convolutions,
  * TensorNHWC for 2-D convolutions, and
  * TensorNDHWC for 3-D convolutions.

  Output (C and D) layouts are the same as input layouts,
  except for Wgrad convolutions, where the layouts are

  * TensorKCS for 1-D convolutions,
  * TensorKCSR for 2-D convolutions, and
  * TensorKCSRT for 3-D convolutions.

  The output layouts are completely constrained by the input layouts
  and the convolution kind.

  tile_descriptions: 2-level list.
    Outer level has one list per math instruction.
    Inner level has one TileDescription for each cluster shape.

  data_types: Either a single data_type dictionary, or a list of them.
    Keys: 'a_type', 'b_type', 'c_type', 'd_type', 'acc_type', 'epi_type'

  complex_transforms: Optional list of pairs.
    First element of each pair is the complex transform for A, and
    second element of each pair is the complex transform for B.

  schedule_pairs: [(kernel_schedule, epilogue_schedule), ...]

  conv_kind: Convolution kind (Fprop, Dgrad, or Wgrad).
  """
  log_debug_line('CreateConvOperator3x', log_indent_level)
  log_indent_level = log_indent_level + 1
  log_debug_line(f'conv_kind: {conv_kind}', log_indent_level)

  for triple in dims_and_alignments:
    spatial_dimensionality = None # to be determined by loop below
    assert(len(triple) == 3)
    for entry in triple: # [A, B, C]
      assert(len(entry) == 2)
      [dim, alignment] = entry
      assert(type(dim) is int)
      assert(dim == 2 or dim == 3)
      assert(type(alignment) is int)
      assert(alignment > 0)
      if spatial_dimensionality is None:
        spatial_dimensionality = dim
      else:
        # A, B, and C need to have the same spatial dimensionality
        assert(spatial_dimensionality == dim)

  def input_and_output_layouts(spatial_dim: int, kind: ConvKind) -> Tuple[LayoutType, LayoutType]:
    if spatial_dim == 1:
      input_layout = LayoutType.TensorNWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCS
      else:
        output_layout = input_layout
    elif spatial_dim == 2:
      input_layout = LayoutType.TensorNHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSR
      else:
        output_layout = input_layout
    elif spatial_dim == 3:
      input_layout = LayoutType.TensorNDHWC
      if kind == ConvKind.Wgrad:
        output_layout = LayoutType.TensorKCSRT
      else:
        output_layout = input_layout
    else:
      assert(False)
    return (input_layout, output_layout)

  def dims_to_layouts(A_B_C: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> \
      Tuple[Tuple[LayoutType, int], Tuple[LayoutType, int], Tuple[LayoutType, int]]:
    [A, B, C] = A_B_C
    [spatial_dim, alignment] = A
    [input_layout, output_layout] = input_and_output_layouts(spatial_dim, conv_kind)
    return ((input_layout, A[1]),
            (input_layout, B[1]),
            (output_layout, C[1]))

  # layouts: list of triples (A, B, C).
  # Each of A, B, and C has the form [layout, alignment].
  layouts = [dims_to_layouts(A_B_C) for A_B_C in dims_and_alignments]

  if type(data_types) is dict:
    data_types = [data_types]

  for s in schedule_pairs:
    assert(len(s) == 2)

  if complex_transforms is None:
    complex_transforms = [(ComplexTransform.none, ComplexTransform.none)]

  # product produces a one-pass generator, so the loop must call it anew each time.
  def make_combinations():
    return product(
      layouts,
      tile_descriptions,
      data_types,
      complex_transforms,
      schedule_pairs,
      tile_schedulers
    )

  operations = []
  for layout_triple, tile_description, data_type, complex_transform_pair, schedule_pair, tile_scheduler in make_combinations():
    A_layout, A_alignment = layout_triple[0]
    A_xform = complex_transform_pair[0]
    B_layout, B_alignment = layout_triple[1]
    B_xform = complex_transform_pair[1]
    C_layout, C_alignment = layout_triple[2]
    D_layout = C_layout
    D_alignment = C_alignment

    A = TensorDescription(data_type["a_type"], A_layout, A_alignment, A_xform)
    B = TensorDescription(data_type["b_type"], B_layout, B_alignment, B_xform)
    C = TensorDescription(data_type["c_type"], C_layout, C_alignment)
    D = TensorDescription(data_type["d_type"], D_layout, D_alignment)
    element_compute = data_type.get("epi_type", data_type["acc_type"])
    kernel_schedule, epilogue_schedule = schedule_pair

    operation = ConvOperation3x(conv_kind=conv_kind,
                                tile_description=tile_description,
                                A=A,
                                B=B,
                                C=C,
                                element_compute=element_compute,
                                D=D,
                                kernel_schedule=kernel_schedule,
                                epilogue_schedule=epilogue_schedule,
                                tile_scheduler=tile_scheduler,
                                log_indent_level=log_indent_level)
    log_debug_line(f'Created ConvOperation3x: {str(operation)}', log_indent_level)
    manifest.append(operation)
    operations.append(operation)

  return operations

###################################################################################################
###################################################################################################



class BaseOperatorDescription:
  def __init__(self, element_a, layout_a, align_a, element_b, layout_b, align_b,
               element_c, layout_c, align_c,
               element_d, element_acc, element_epi):
    self.element_a = element_a
    self.element_b = element_b
    self.element_c = element_c
    self.element_d = element_d
    self.element_acc = element_acc
    self.element_epi = element_epi

    self.layout_a = layout_a
    self.layout_b = layout_b
    self.layout_c = layout_c

    self.align_a = align_a
    self.align_b = align_b
    self.align_c = align_c

  @classmethod
  @abstractmethod
  def create_operation(cls, manifest):
    pass

  def __eq__(self, other):
    if not isinstance(other, BaseOperatorDescription):
      return False
    return (
      self.element_a == other.element_a and
      self.element_b == other.element_b and
      self.element_c == other.element_c and
      self.element_d == other.element_d and
      self.element_acc == other.element_acc and
      self.element_epi == other.element_epi and
      self.layout_a == other.layout_a and
      self.layout_b == other.layout_b and
      self.layout_c == other.layout_c and
      self.align_a == other.align_a and
      self.align_b == other.align_b and
      self.align_c == other.align_c
    )

  def __hash__(self):
    return hash((
      self.element_a,
      self.element_b,
      self.element_c,
      self.element_d,
      self.element_acc,
      self.element_epi,
      self.layout_a,
      self.layout_b,
      self.layout_c,
      self.align_a,
      self.align_b,
      self.align_c,
    ))

  def __str__(self):
    return (
      f"BaseOperatorDescription("
      f"element_a={self.element_a}, layout_a={self.layout_a}, align_a={self.align_a}, "
      f"element_b={self.element_b}, layout_b={self.layout_b}, align_b={self.align_b}, "
      f"element_c={self.element_c}, layout_c={self.layout_c}, align_c={self.align_c}, "
      f"element_d={self.element_d}, element_acc={self.element_acc}, element_epi={self.element_epi})"
    )

  def _validate_data_types(self, math):
   return (
      self.element_a == math.element_a and
      self.element_b == math.element_b and
      self.element_acc == math.element_accumulator
      )

class GemmDescription(BaseOperatorDescription):
  def __init__(self, element_a, layout_a, align_a, element_b, layout_b, align_b, 
               element_c, layout_c, align_c,
               element_d, element_acc, element_epi):
    align_c =  min(align_c, min(8, 128 // DataTypeSize[element_c]))

    super().__init__(element_a, layout_a, align_a, element_b, layout_b, align_b,
                   element_c, layout_c, align_c,
                   element_d, element_acc, element_epi)
    
  def __short_layout(self):
    def current_short(layout):
      if layout == LayoutType.RowMajor:
        return "t"
      elif layout == LayoutType.ColumnMajor:
        return "n"
      else:
        raise ValueError(f"Only support Row/Column-major")

    return f"{current_short(self.layout_a)}{current_short(self.layout_b)}"

  def __str__(self):
    return (
      f"GemmDescription("
      f"{super().__str__()}"
      f")"
    )

  def create_operation(self, manifest):
    self.__create_hute_gemm(manifest)
    self.__create_hytlass_gemm(manifest)

  def __create_hute_gemm(self, manifest):
    bit_size_ab_str = f"{DataTypeSize[self.element_a]}b"
    tile_configs = TileConfig("Gfx928", bit_size_ab_str, self.__short_layout())

    target_layout = [[
      [self.layout_a, self.align_a],
      [self.layout_b, self.align_b],
      [self.layout_c, self.align_c]
    ]]

    data_type = {
      "a_type" :   self.element_a,
      "b_type" :   self.element_b,
      "c_type" :   self.element_c,
      "d_type" :   self.element_d,
      "acc_type" : self.element_acc,
      "epi_type" : self.element_epi
    }

    math_instructions = tile_configs.math_instructions

    tile_gen = TileGeneratorGfx928()

    for math_inst in math_instructions:
      if not self._validate_data_types(math_inst):
        continue

      tile_descriptions = tile_gen.generate_tile_descriptions(tile_configs, math_inst,
                                                              bit_size_ab_str, target_layout)

      regular_schedules = tile_configs.schedules
      stream_k_schedules = tile_configs.stream_k_schedules

      CreateGemmUniversal3xOperator(manifest, target_layout, tile_descriptions, 
                                    data_type, regular_schedules)
      CreateGemmUniversal3xOperator(manifest, target_layout, tile_descriptions,
                                    data_type, stream_k_schedules, tile_schedulers=[TileSchedulerType.StreamK,])

  def __create_hytlass_gemm(self, manifest):
    bit_size_ab_str = f"{DataTypeSize[self.element_a]}b"
    tile_configs = TileConfig_2x("Gfx928", bit_size_ab_str, self.__short_layout())
    layouts = tile_configs.layouts
    math_instructions = tile_configs.math_instructions
    enable_buffer_access = True

    tile_gen = TileGeneratorGfx928_2x()

    data_type = [
      self.element_a,
      self.element_b,
      self.element_c,
      self.element_acc
    ]

    for math_inst in math_instructions:
      if not self._validate_data_types(math_inst):
        continue

      target_align = (
        self.align_a, self.align_b, self.align_c
      )

      byte_size_abc = (
        DataTypeSize[self.element_a] // 8, 
        DataTypeSize[self.element_b] // 8, 
        DataTypeSize[self.element_c] // 8
      )

      tile_descriptions = tile_gen.generate_gemm_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, layouts, 
        target_align, enable_buffer_access
      )

      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [target_align, ],
                        BufferAccess=enable_buffer_access, EnStaggerK= False)

      CreateGemmOperator(manifest, layouts, tile_descriptions, data_type, [target_align, ],
                         BufferAccess = enable_buffer_access, EnStaggerK = False, 
                         swizzling_functor=SwizzlingFunctor.StreamK)

class Conv2dDescription(BaseOperatorDescription):
  def __init__(self, conv_kind, 
               element_a, layout_a, align_a, 
               element_f, layout_f, align_f, 
               element_o, layout_o, align_o,
               element_acc, element_epi, enable_few_channel):
    self.conv_kind = conv_kind
    self.enable_few_channel = enable_few_channel

    element_a, element_b, element_c = self.__remap2abc((element_a, element_f, element_o))
    layout_a, layout_b, layout_c = self.__remap2abc((layout_a, layout_f, layout_o))
    align_a, align_b, align_c = self.__remap2abc((align_a, align_f, align_o))
  
    align_c = min(align_c, min(8, 128 // DataTypeSize[element_c]))

    element_d = element_c

    super().__init__(element_a, layout_a, align_a,
                     element_b, layout_b, align_b,
                     element_c, layout_c, align_c,
                     element_d, element_acc, element_epi)

  def __remap2abc(self, raw: list):
    # [a, f, o] -> [a, b, c]
    result = None
    if self.conv_kind == ConvKind.Fprop:
      result = list(raw)
    elif self.conv_kind == ConvKind.Dgrad:
      result = [raw[2], raw[1], raw[0]]
    elif self.conv_kind == ConvKind.Wgrad:
      result = [raw[2], raw[0], raw[1]]
    return result

  def __short_layout(self):
    result = None
    if self.layout_a == LayoutType.TensorNHWC and self.layout_b == LayoutType.TensorNHWC:
      if self.conv_kind == ConvKind.Fprop:
        result = "tn"
      elif self.conv_kind == ConvKind.Dgrad:
        result = "tt"
      elif self.conv_kind == ConvKind.Wgrad:
        result = "nt"

    if result == None:
      raise ValueError(f"only support TensorNHWC")
    return result

  def __eq__(self, other):
    if not isinstance(other, Conv2dDescription):
      return False
    return (super().__eq__(other) and 
            self.conv_kind == other.conv_kind and
            self.enable_few_channel == other.enable_few_channel)

  def __hash__(self):
    return hash(super().__hash__(), self.conv_kind, self.enable_few_channel)

  def __str__(self):
    return (
      f"Conv2dDescription("
      f"conv_kind {self.conv_kind}"
      f"{super().__str__()}"
      f")"
    )

  def create_operation(self, manifest):
    self.__create_hytlass_conv(manifest)
    if self.enable_few_channel:
      self.__create_hytlass_conv_few_channels(manifest)

  def __create_hytlass_conv(self, manifest):
    bit_size_ab_str = f"{DataTypeSize[self.element_a]}b"

    tile_configs = TileConfig_2x("Gfx928", bit_size_ab_str, self.__short_layout())
    tile_gen = TileGeneratorGfx928_2x()

    tile_layout = tile_configs.layouts
    math_insts = tile_configs.math_instructions
    align_constaints = [self.align_a, self.align_b, self.align_c]

    data_type = [
      self.element_a,
      self.element_b,
      self.element_c,
      self.element_epi
    ]

    byte_size_abc = [
      DataTypeSize[self.element_a] // 8,
      DataTypeSize[self.element_b] // 8,
      DataTypeSize[self.element_c] // 8
    ]

    conv_layout = [
      self.layout_a,
      self.layout_b,
      self.layout_c
    ]

    for math_inst in math_insts:
      if not self._validate_data_types(math_inst):
        continue

      tile_desc = tile_gen.generate_conv_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layout, align_constaints,
        [self.conv_kind, ],
        [IteratorAlgorithm.Analytic, IteratorAlgorithm.Optimized]
      )

      CreateConv2dOperator(manifest, conv_layout, tile_desc, data_type,
                           [align_constaints, ], [self.conv_kind, ],
                           EpilogueFunctor.LinearCombination)

  def __create_hytlass_conv_few_channels(self, manifest):
    bit_size_ab_str = f"{DataTypeSize[self.element_a]}b"

    tile_configs = TileConfig_2x("Gfx928", bit_size_ab_str, self.__short_layout())
    tile_gen = TileGeneratorGfx928_2x()

    tile_layout = tile_configs.layouts
    math_insts = tile_configs.math_instructions

    align_constraints = [self.align_a, self.align_b, self.align_c]

    data_type = [
      self.element_a,
      self.element_b,
      self.element_c,
      self.element_epi
    ]

    byte_size_abc = [
      DataTypeSize[self.element_a] // 8,
      DataTypeSize[self.element_b] // 8,
      DataTypeSize[self.element_c] // 8
    ]

    conv_layout = [
      self.layout_a,
      self.layout_b,
      self.layout_c
    ]

    for math_inst in math_insts:
      if not self._validate_data_types(math_inst):
        continue

      tile_desc = tile_gen.generate_conv_few_channels_tile_descriptions(
        tile_configs, math_inst, byte_size_abc, tile_layout, align_constraints,
        [self.conv_kind, ]
      )

      CreateConv2dFewChannelsOperator(manifest, conv_layout, tile_desc, data_type, 
                                      [align_constraints, ], [self.conv_kind])

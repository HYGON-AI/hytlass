# Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 
list(SORT HYTLASS_HIPCC_ARCHS_ENABLED)
set(HYTLASS_HIPCC_ARCHS_ENABLED_REVERSED ${HYTLASS_HIPCC_ARCHS_ENABLED})
list(REVERSE HYTLASS_HIPCC_ARCHS_ENABLED_REVERSED)
list(GET HYTLASS_HIPCC_ARCHS_ENABLED_REVERSED 0 HYTLASS_HIPCC_MAX_ARCH)

add_custom_target(
  hytlass_test_unit_conv_device
  DEPENDS
  hytlass_test_unit_conv_device_simt
)

add_custom_target(
  test_unit_conv_device
  DEPENDS
  test_unit_conv_device_simt
)

if (HYTLASS_HIPCC_MAX_ARCH GREATER_EQUAL 928)
  add_dependencies(
    hytlass_test_unit_conv_device
    hytlass_test_unit_conv_device_tensorop_f32_gfx928
    hytlass_test_unit_conv_device_tensorop_f16_gfx928
    hytlass_test_unit_conv_device_tensorop_f32_tf32_gfx928
    hytlass_test_unit_conv_device_tensorop_s32_gfx928
  )

  add_dependencies(
    test_unit_conv_device
    test_unit_conv_device_tensorop_f32_gfx928
    test_unit_conv_device_tensorop_f16_gfx928
    test_unit_conv_device_tensorop_f32_tf32_gfx928
    test_unit_conv_device_tensorop_s32_gfx928
  )
endif()


#
# OpClassSimt (HIP cores)
#
hytlass_test_unit_add_executable(
  hytlass_test_unit_conv_device_simt
  
  # F32  
  conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_gfx906.cu
  conv2d_dgrad_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_gfx906.cu
  conv2d_wgrad_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_gfx906.cu
  conv2d_fprop_with_broadcast_simt_gfx906.cu

  # CF32
  conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_gfx906.cu
  conv2d_dgrad_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_gfx906.cu
  conv2d_wgrad_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_gfx906.cu 

  # F16
  conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_simt_f16_gfx906.cu
  depthwise_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_simt_f16_gfx906.cu
  depthwise_conv2d_fprop_direct_conv_f16nhwc_f16nhwc_f16nhwc_simt_f16_gfx906.cu
  depthwise_conv2d_fprop_direct_conv_fixed_stride_dilation_f16nhwc_f16nhwc_f16nhwc_simt_f16_gfx906.cu

  # deconv2d
  deconv2d_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_gfx906.cu
  deconv2d_with_broadcast_simt_gfx906.cu
)

#
# OpClassTensorOp (Tensor cores)
#

if (HYTLASS_HIPCC_MAX_ARCH GREATER_EQUAL 928)
  # Conv - F16 input, F32 output, F32 accumulation - gfx928
  hytlass_test_unit_add_executable(
    hytlass_test_unit_conv_device_tensorop_f32_gfx928

    conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_dgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_wgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_gfx928.cu

    conv2d_fprop_with_broadcast_gfx928.cu
    conv2d_fprop_with_reduction_gfx928.cu

    # Conv2d (small channel count specializations)
    conv2d_fprop_fixed_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu
    conv2d_fprop_few_channels_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu

    # Conv2d (Strided Dgrad)
    conv2d_strided_dgrad_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_strided_dgrad_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_strided_dgrad_implicit_gemm_swizzling4_gfx928.cu

    # Group Conv2d
    group_conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu
  )

  # Conv - F16 input, F16 output, F32 accumulation 
  hytlass_test_unit_add_executable(
    hytlass_test_unit_conv_device_tensorop_f16_gfx928

    conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu
    conv2d_dgrad_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu 
    conv2d_wgrad_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_gfx928.cu 
  )

  # Conv - TF32 input, F32 output, F32 accumulation
  hytlass_test_unit_add_executable(
    hytlass_test_unit_conv_device_tensorop_f32_tf32_gfx928

    conv2d_fprop_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_dgrad_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_gfx928.cu
    conv2d_wgrad_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_gfx928.cu

  )

  #   # Conv2d - S8 input, S32 output, S32 accumulation
  hytlass_test_unit_add_executable(
    hytlass_test_unit_conv_device_tensorop_s32_gfx928

    conv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_gfx928.cu
  )
  
endif()
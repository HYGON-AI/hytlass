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
  hytlass_test_unit_gemm_device_3x
  DEPENDS
  hytlass_test_unit_gemm_device_simt_3x
  hytlass_test_unit_gemm_device_tensorop_gfx928_3x
  hytlass_test_unit_gemm_device_tensorop_gfx928_stream_k
  hytlass_test_unit_gemm_device_tensorop_alignx_gfx928
)

add_custom_target(
  test_unit_gemm_device_3x
  DEPENDS
  test_unit_gemm_device_simt
  test_unit_gemm_device_tensorop_gfx928_3x
  test_unit_gemm_device_tensorop_gfx928_stream_k
  test_unit_gemm_device_tensorop_alignx_gfx928
)

# OpClassSimt
hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_device_simt_3x

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gfx906_gemm_f32_f32_f32_simt.cu
  gfx906_gemm_s8_s8_s32_simt.cu
)

# OpClassTensorOp (Tensor cores)
if (HYTLASS_HIPCC_MAX_ARCH GREATER_EQUAL 928)
  hytlass_test_unit_add_executable(
    hytlass_test_unit_gemm_device_tensorop_gfx928_3x

    BATCH_SOURCES ON
    BATCH_SIZE 4

    gfx928_gemm_f16_f16_f16_tensor_op_f32.cu
    gfx928_gemm_f16_f16_f32_tensor_op_f32.cu
    gfx928_gemm_bf16_bf16_bf16_tensor_op_f32.cu
    gfx928_gemm_s8_s8_s8_tensor_op_s32.cu
    gfx928_gemm_s8_s8_s32_tensor_op.cu
    gfx928_gemm_tf32_tf32_f32_tensor_op_f32.cu
    gfx928_gemm_f32_f32_f32_tensor_op_f32.cu
    gfx928_gemm_u8_u8_u8_tensor_op_s32.cu
  )

  hytlass_test_unit_add_executable(
    hytlass_test_unit_gemm_device_tensorop_gfx928_stream_k

    gfx928_gemm_stream_k_scheduler.cu
    gfx928_gemm_f16_f16_f16_tensor_op_f32_stream_k.cu
    gfx928_gemm_bf16_bf16_bf16_tensor_op_f32_stream_k.cu
    gfx928_gemm_tf32_tf32_tf32_tensor_op_f32_stream_k.cu
    gfx928_gemm_s8_s8_s8_tensor_op_s32_stream_k.cu
    gfx928_gemm_u8_u8_u8_tensor_op_s32_stream_k.cu
  )

  #Alignment tests
  hytlass_test_unit_add_executable(
    hytlass_test_unit_gemm_device_tensorop_alignx_gfx928

    BATCH_SOURCES ON
    BATCH_SIZE 4

    gfx928_gemm_f16_f16_f16_alignx_tensor_op_f32.cu
    gfx928_gemm_bf16_bf16_bf16_alignx_tensor_op_f32.cu
    gfx928_gemm_s8_s8_s8_alignx_tensor_op_s32.cu
    gfx928_gemm_tf32_tf32_f32_alignx_tensor_op_f32.cu
    gfx928_gemm_u8_u8_u8_alignx_tensor_op_s32.cu
  )
    
endif()


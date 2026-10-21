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

add_library(hytlass_test_util_includes INTERFACE)

add_custom_target(
  hytlass_test_unit_gemm_device
  DEPENDS
  hytlass_test_unit_ell_block_sparse_gemm_device
  hytlass_test_unit_gemm_device_simt
  hytlass_test_unit_gemm_device_tensorop_gfx928
  hytlass_test_unit_gemm_device_singlestage_tensorop_gfx928
  hytlass_test_unit_gemm_device_tensorop_slicedk_gfx928
  hytlass_test_unit_gemm_grouped_device
  hytlass_test_unit_gemm_splitk_device
)

add_custom_target(
  test_unit_gemm_device
  DEPENDS
  test_unit_ell_block_sparse_gemm_device
  test_unit_gemm_device_simt
  test_unit_gemm_device_singlestage_tensorop_gfx928
  test_unit_gemm_device_tensorop_slicedk_gfx928
  test_unit_gemm_device_tensorop_gfx928
  test_unit_gemm_grouped_device
  test_unit_gemm_splitk_device
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_ell_block_sparse_gemm_device

  # TensorOp
  ell_block_sparse_gemm_f16t_f16t_f16t_tensor_op_f32_gfx928.cu
  ell_block_sparse_gemm_f16t_f16n_f16t_tensor_op_f32_gfx928.cu
  ell_block_sparse_gemm_f32t_f32t_f32t_tensor_op_f32_gfx928.cu
  ell_block_sparse_gemm_f32t_f32n_f32t_tensor_op_f32_gfx928.cu

  # Simt
  ell_block_sparse_gemm_f32t_f32t_f32t_simt_f32_gfx906.cu
  ell_block_sparse_gemm_f32t_f32n_f32t_simt_f32_gfx906.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_device_simt
  simt_hgemm_nn_gfx906.cu
  simt_hgemm_nt_gfx906.cu
  simt_hgemm_tn_gfx906.cu
  simt_hgemm_tt_gfx906.cu
  simt_igemm_nn_gfx906.cu
  simt_igemm_nt_gfx906.cu
  simt_igemm_tn_gfx906.cu
  simt_igemm_tt_gfx906.cu
  simt_sgemm_nn_gfx906.cu
  simt_sgemm_nt_gfx906.cu
  simt_sgemm_tn_gfx906.cu
  simt_sgemm_tt_gfx906.cu
  simt_cgemm_nn_gfx906.cu
  simt_cgemm_nt_gfx906.cu
  simt_cgemm_tn_gfx906.cu
  simt_cgemm_tt_gfx906.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_device_tensorop_gfx928

  # TensorOp
  gemm_bf16n_bf16n_f32t_tensor_op_f32_gfx928.cu
  gemm_bf16t_bf16t_bf16t_tensor_op_f32_gfx928.cu
  gemm_f16n_f16n_f16t_tensor_op_f32_gfx928.cu
  gemm_f16n_f16n_f32n_tensor_op_f32_gfx928.cu
  gemm_f16n_f16n_f32t_tensor_op_f32_gfx928.cu
  gemm_f16n_f16t_f16t_tensor_op_f32_gfx928.cu
  gemm_f16n_f16t_f32t_tensor_op_f32_gfx928.cu
  gemm_f16t_f16n_f16t_tensor_op_f32_gfx928.cu
  gemm_f16t_f16n_f32n_tensor_op_f32_gfx928.cu
  gemm_f16t_f16n_f32t_tensor_op_f32_gfx928.cu
  gemm_f16t_f16t_f32n_tensor_op_f32_gfx928.cu
  gemm_f16t_f16t_f32t_tensor_op_f32_gfx928.cu
  gemm_f32n_f32n_f32t_tensor_op_f32_gfx928.cu
  gemm_f32n_f32t_f32t_tensor_op_f32_gfx928.cu
  gemm_f32t_f32n_f32t_tensor_op_f32_gfx928.cu
  gemm_f32t_f32t_f32t_tensor_op_f32_gfx928.cu
  gemm_s8t_s8n_s8n_tensor_op_s32_gfx928.cu
  gemm_s8t_s8n_s8t_tensor_op_s32_gfx928.cu
  gemm_s8t_s8n_s32n_tensor_op_s32_gfx928.cu
  gemm_s8t_s8n_s32t_tensor_op_s32_gfx928.cu
  gemm_tf32n_tf32n_f32t_tensor_op_f32_gfx928.cu
  gemm_tf32n_tf32t_f32t_tensor_op_f32_gfx928.cu
  gemm_tf32t_tf32n_f32t_tensor_op_f32_gfx928.cu
  gemm_tf32t_tf32t_f32t_tensor_op_f32_gfx928.cu
  gemm_universal_f16n_f16t_f32n_tensor_op_f32_gfx928.cu
  gemm_universal_f16n_f16t_f32t_tensor_op_f32_gfx928.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_device_singlestage_tensorop_gfx928

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16t_f16n_f16n_singlestage_tensor_op_f32_gfx928.cu
  gemm_f16t_f16n_f16t_singlestage_tensor_op_f32_gfx928.cu
  gemm_f16t_f16n_f32t_singlestage_tensor_op_f32_gfx928.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_device_tensorop_slicedk_gfx928

  BATCH_SOURCES ON
  BATCH_SIZE 4

  gemm_f16n_f16t_f16t_tensor_op_f32_slicedk_gfx928.cu
  gemm_f16t_f16n_f16t_tensor_op_f32_slicedk_gfx928.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_grouped_device
  gemm_grouped_gfx928.cu
  gemm_grouped_scheduler_gfx928.cu
)

hytlass_test_unit_add_executable(
  hytlass_test_unit_gemm_splitk_device
  gemm_splitk_serial_tensor_op_gfx928.cu
  gemm_splitk_simt_gfx906.cu
  gemm_splitk_tensor_op_gfx928.cu
)

target_include_directories(hytlass_test_unit_ell_block_sparse_gemm_device PRIVATE ${CMAKE_SOURCE_DIR}/tools/util/include)


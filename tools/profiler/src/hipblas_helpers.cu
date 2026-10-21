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
/* \file
   \brief Helper functions for mapping HYTLASS concepts to hipBLAS.
*/

#include <stdexcept>

#if HYTLASS_ENABLE_HIPBLAS
#include "hytlass/profiler/hipblas_helpers.h"

namespace hytlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a hipBLAS status to hytlass::Status
Status get_hytlass_status(hipblasStatus_t hipblas) {

  switch (hipblas) {
    case HIPBLAS_STATUS_SUCCESS: 
      return Status::kSuccess;
    case HIPBLAS_STATUS_INVALID_VALUE:
      return Status::kErrorInvalidProblem;
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return Status::kErrorNotSupported;
    default: break;
  }
  return Status::kErrorInternal;
}

/// Converts a hipBLAS status to hytlass::profiler::Disposition
Disposition get_hytlass_disposition(hipblasStatus_t hipblas_status) {

  if (hipblas_status == HIPBLAS_STATUS_INVALID_VALUE) {
    return Disposition::kInvalidProblem;
  }
  else if (hipblas_status == HIPBLAS_STATUS_NOT_SUPPORTED) {
    return Disposition::kNotSupported;
  }
  return Disposition::kFailed;
}

/// Maps a HYTLASS tensor layout to a hipBLAS transpose operation
bool get_hipblas_transpose_operation(
  hipblasOperation_t &operation,
  library::LayoutTypeID layout, 
  library::ComplexTransform transform) {

  switch (layout) {
    case library::LayoutTypeID::kColumnMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = HIPBLAS_OP_N;
        return true;
      }
      else {
        return false;
      }
      break;
    case library::LayoutTypeID::kRowMajor:
      if (transform == library::ComplexTransform::kNone) {
        operation = HIPBLAS_OP_T;
        return true;
      }
      else if (transform == library::ComplexTransform::kConjugate) {
        operation = HIPBLAS_OP_C;
        return true;
      }
      break;
    default: break;
  }

  return false;
}

/// Maps a HYTLASS numeric type to a hipBLAS data type enumeration
bool get_hipblas_datatype(hipblasDatatype_t &data_type, library::NumericTypeID element_type) {
  switch (element_type) {
  case library::NumericTypeID::kFE4M3:
    // TODO: support fp8
    return false;
    break;
  
  case library::NumericTypeID::kFE5M2:
    // TODO: support fp8
    return false;
    break;

  case library::NumericTypeID::kF16:
    data_type = HIPBLAS_R_16F;
    return true;
    
  case library::NumericTypeID::kBF16:
    data_type = HIPBLAS_R_16B;
    return true;
  
  case library::NumericTypeID::kTF32: 
    break;
  
  case library::NumericTypeID::kF32:
    data_type = HIPBLAS_R_32F;
    return true;
    
  case library::NumericTypeID::kF64: 
    data_type = HIPBLAS_R_64F;
    return true;
  
  case library::NumericTypeID::kS4: 
    break;
  
  case library::NumericTypeID::kS8: 
    data_type = HIPBLAS_R_8I;
    return true;
    
  case library::NumericTypeID::kS16: 
    break;
 
  case library::NumericTypeID::kS32: 
    data_type = HIPBLAS_R_32I;
    return true;
    
  case library::NumericTypeID::kS64: 
    break;
  
  case library::NumericTypeID::kU4: 
    break;
  
  case library::NumericTypeID::kU8: 
    data_type = HIPBLAS_R_8U;
    return true;
    
  case library::NumericTypeID::kU16: 
    break;
    
  case library::NumericTypeID::kU32: 
    data_type = HIPBLAS_R_32U;
    return true;
    
  case library::NumericTypeID::kU64: 
    break;

  case library::NumericTypeID::kB1: 
    break;

  case library::NumericTypeID::kCF32:
    data_type = HIPBLAS_C_32F;
    return true;

  case library::NumericTypeID::kCF64:
    data_type = HIPBLAS_C_64F;
    return true;
  
  case library::NumericTypeID::kInvalid:
  
  default: 
    break;
  }

  return false;
}

/// Maps a hytlass::SideMode to hipBLAS side mode
bool get_hipblas_side_mode(hipblasSideMode_t& side, SideMode side_mode) {

  switch (side_mode) {
    case SideMode::kLeft: 
      side = HIPBLAS_SIDE_LEFT;
      return true;
    case SideMode::kRight: 
      side = HIPBLAS_SIDE_RIGHT;
      return true;
    default: break;
  }

  return false;
}

/// Maps a hytlass::FillMode to hipBLAS fill mode
bool get_hipblas_fill_mode(hipblasFillMode_t& uplo, FillMode fill_mode) {

  switch (fill_mode) {
    case FillMode::kLower: 
      uplo = HIPBLAS_FILL_MODE_LOWER;
      return true;
    case FillMode::kUpper: 
      uplo = HIPBLAS_FILL_MODE_UPPER;
      return true;
    default: break;
  }

  return false;
}

/// Maps a hytlass::DiagType to hipBLAS diag type
bool get_hipblas_diag_type(hipblasDiagType_t& diag, DiagType diag_type) {

  switch (diag_type) {
    case DiagType::kNonUnit: 
      diag = HIPBLAS_DIAG_NON_UNIT;
      return true;
    case DiagType::kUnit: 
      diag = HIPBLAS_DIAG_UNIT;
      return true;
    default: break;
  }

  return false;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gets the hipblas algorithm given threadblock tile dimensions and math opcode class
hipblasGemmAlgo_t get_hipblas_gemm_algo(int cta_m, int cta_n, int cta_k, library::OpcodeClassID opcode_class) {
  // return (opcode_class == library::OpcodeClassID::kSimt ? 
  //   HIPBLAS_GEMM_DEFAULT : HIPBLAS_GEMM_DEFAULT_TENSOR_OP);
  // use HIPBLAS_GEMM_DEFAULT duo to only it's provided
  return HIPBLAS_GEMM_DEFAULT;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Returns a status if hipBLAS can satisfy a particular GEMM description
Status hipblas_satisfies(library::GemmDescription const &desc) {
  auto const &math_instruction = desc.tile_description.math_instruction;

  if (math_instruction.element_accumulator == library::NumericTypeID::kS32 && 
    math_instruction.opcode_class == library::OpcodeClassID::kTensorOp) {

    return Status::kErrorNotSupported;
  }

  // output type S4 and S8 not supported in hipBLAS
  if (desc.C.element == library::NumericTypeID::kS4 || 
    desc.C.element == library::NumericTypeID::kS8) {

    return Status::kErrorNotSupported;
  }

  // input type BF16 and TF32 not supported in hipBLAS
  if (desc.A.element == library::NumericTypeID::kBF16 || 
    desc.A.element == library::NumericTypeID::kTF32) {

    return Status::kErrorNotSupported;
  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

hipblasGemmExDispatcher::hipblasGemmExDispatcher(
  library::GemmDescription const &op_desc,
  library::GemmUniversalConfiguration configuration_,
  library::GemmUniversalArguments arguments_,
  hipblasGemmAlgo_t algorithm
):
  configuration(configuration_), arguments(arguments_), algo(algorithm), status(Status::kSuccess) {

  bool good = true;

  good = (good && get_hipblas_transpose_operation(trans_A, op_desc.A.layout, op_desc.transform_A));
  good = (good && get_hipblas_transpose_operation(trans_B, op_desc.B.layout, op_desc.transform_B));
  good = (good && get_hipblas_datatype(data_type_A, op_desc.A.element));
  good = (good && get_hipblas_datatype(data_type_B, op_desc.B.element));
  good = (good && get_hipblas_datatype(data_type_C, op_desc.C.element));

  good = (good && get_hipblas_datatype(
    compute_data_type,
    op_desc.tile_description.math_instruction.element_accumulator));

  if (!good) {
    status = Status::kErrorNotSupported;
  }
}

/// Executes GEMM using these arguments
hipblasStatus_t hipblasGemmExDispatcher::operator()(hipblasHandle_t handle) {

  if (configuration.mode == library::GemmUniversalMode::kBatched) {
    return hipblasGemmStridedBatchedEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.batch_stride_A,
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.batch_stride_B,
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
      arguments.batch_stride_C,
      configuration.batch_count,
      compute_data_type,
      algo
    );
  }
  else {
    return hipblasGemmEx(
      handle,
      trans_A,
      trans_B,
      configuration.problem_size.m(),
      configuration.problem_size.n(),
      configuration.problem_size.k(),
      arguments.alpha,
      arguments.A,
      data_type_A,
      int(configuration.lda),
      arguments.B,
      data_type_B,
      int(configuration.ldb),
      arguments.beta,
      arguments.D,
      data_type_C,
      int(configuration.ldc),
      compute_data_type,
      algo
    );
  }
}

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace hytlass

#endif // #if HYTLASS_ENABLE_HIPBLAS

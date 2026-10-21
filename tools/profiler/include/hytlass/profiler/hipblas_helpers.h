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

#pragma once

#if HYTLASS_ENABLE_HIPBLAS
#include <hipblas.h>

#include "hytlass/hytlass.h"
#include "hytlass/library/library.h"
#include "hytlass/library/util.h"
#include "hytlass/blas3.h"

#include "options.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Converts a hipBLAS status to hytlass::Status
Status get_hytlass_status(hipblasStatus_t hipblas);

/// Converts a hipBLAS status to hytlass::profiler::Disposition
Disposition get_hytlass_disposition(hipblasStatus_t hipblas_status);

/// Maps a HYTLASS tensor layout to a hipBLAS transpose operation
bool get_hipblas_transpose_operation(
  hipblasOperation_t &operation,
  library::LayoutTypeID layout,
  library::ComplexTransform transform = library::ComplexTransform::kNone);

/// Maps a HYTLASS numeric type to a hipBLAS data type enumeration
bool get_hipblas_datatype(hipblasDatatype_t &data_type, library::NumericTypeID element_type);

/// Gets the hipblas algorithm given threadblock tile dimensions and math opcode class
hipblasGemmAlgo_t get_hipblas_gemm_algo(
  int cta_m, 
  int cta_n, 
  int cta_k, 
  library::OpcodeClassID opcode_class);

/// Returns a status if hipBLAS can satisfy a particular GEMM description
Status hipblas_satisfies(library::GemmDescription const &desc);

/// This is a helper class to create hipblasHandle_t automatically on HipblasCreate object creation and 
/// to destroy hipblasHandle_t on HipblasCreate object destruction. 
/// Additionally, it provides implicit cast from HipblasCreate's object to hipblasHandle_t's object
class HipblasCreate {
private:
	hipblasHandle_t handle;
	hipblasStatus_t status;

public:
	HipblasCreate() {
		status = hipblasCreate(&handle);
	}

	~HipblasCreate() {
		hipblasDestroy(handle);
	}

    /// Implicit cast HipblasCreate object to hipblasHandle_t
    operator hipblasHandle_t() const { return handle; }

    /// returns hipblasStatus_t for handle creation
    hipblasStatus_t get_hipblas_create_status() { return status; }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

/// Selects one or more hipBLAS algorithms.
static void select_hipblas_algorithms(
  std::vector<hipblasGemmAlgo_t> &algorithms,
  Options const &options, 
  library::GemmDescription const &op_desc) {
    // for hipblas, only HIPBLAS_GEMM_DEFAULT is provided
    algorithms.push_back(HIPBLAS_GEMM_DEFAULT); 
}

/// Dispatcher to hipblasGemmEx() 
struct hipblasGemmExDispatcher {

  //
  // Data members
  //
  library::GemmUniversalConfiguration configuration;
  library::GemmUniversalArguments arguments;

  // hipblas-specific data structures to fill hipblas API call arguments
  hipblasOperation_t trans_A;
  hipblasOperation_t trans_B;
  hipblasDatatype_t data_type_A;
  hipblasDatatype_t data_type_B;
  hipblasDatatype_t data_type_C;
  hipblasDatatype_t compute_data_type;

  hipblasGemmAlgo_t algo;
  Status status;
  
  //
  // Methods
  //

  hipblasGemmExDispatcher( 
    library::GemmDescription const &op_desc,
    library::GemmUniversalConfiguration configuration_,
    library::GemmUniversalArguments arguments_,
    hipblasGemmAlgo_t algorithm = HIPBLAS_GEMM_DEFAULT
  );

  /// Executes GEMM using these arguments
  hipblasStatus_t operator()(hipblasHandle_t handle);
};
///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

} // namespace profiler
} // namespace hytlass


#endif // #if HYTLASS_ENABLE_HIPBLAS

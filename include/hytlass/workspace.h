/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Utilities for initializing workspaces
*/

#pragma once

#include "hip/hip_runtime.h"

#include "hytlass/trace.h"

#include "hytlass.h"
#include "hytlass/hip_host_adapter.hpp"
namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int MinWorkspaceAlignment = 16;

static Status
zero_workspace(void* workspace, size_t workspace_size, hipStream_t stream = nullptr) {
  if (workspace_size > 0) {
    if (workspace == nullptr) {
      HYTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    HYTLASS_TRACE_HOST("  clearing barrier workspace");
    hipError_t result = hipMemsetAsync(workspace, 0, workspace_size, stream);
    if (hipSuccess != result) {
      result = hipGetLastError(); // to clear the error bit
      HYTLASS_TRACE_HOST("  hipMemsetAsync() returned error " << hipGetErrorString(result));
      return Status::kErrorInternal;
    }
  }

  return Status::kSuccess;
}

template <typename T>
Status
fill_workspace(void* workspace, T fill_value, size_t fill_count, hipStream_t stream = nullptr, HipHostAdapter *hip_adapter = nullptr) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 2 || sizeof(T) == 1, "Unsupported fill type");
  if (fill_count > 0) {
    if (workspace == nullptr) {
      HYTLASS_TRACE_HOST("  error: device workspace must not be null");
      return Status::kErrorWorkspaceNull;
    }

    HYTLASS_TRACE_HOST("  filling workspace");
    hipDeviceptr_t d_workspace = reinterpret_cast<hipDeviceptr_t>(workspace);

    hipError_t result = hipSuccess;
    if (sizeof(T) == 4) {
      result = hipMemsetD32Async(d_workspace, reinterpret_cast<uint32_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 2) {
      result = hipMemsetD16Async(d_workspace, reinterpret_cast<uint16_t&>(fill_value), fill_count, stream);
    }
    else if (sizeof(T) == 1) {
      result = hipMemsetD8Async(d_workspace, reinterpret_cast<uint8_t&>(fill_value), fill_count, stream);
    }

    if (hipSuccess != result) {
      const char** error_string_ptr = nullptr;
      (void) hipGetErrorString(result);
      if (error_string_ptr != nullptr) {
        HYTLASS_TRACE_HOST("  cuMemsetD" << sizeof(T) * 8 << "Async() returned error " << *error_string_ptr);
      }
      else {
        HYTLASS_TRACE_HOST("  cuMemsetD" << sizeof(T) * 8 << "Async() returned unrecognized error");
      }
      return Status::kErrorInternal;
    }

  }

  return Status::kSuccess;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

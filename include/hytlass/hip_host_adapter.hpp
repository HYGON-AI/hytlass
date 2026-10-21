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

/*! \file
    \brief Interface betweeen a HYTLASS device-wide operator and HIP.
*/

#pragma once

#include <hip/hip_runtime_api.h>
#include "hytlass/hytlass.h"
#include "hytlass/trace.h"

#include "hytlass/platform/platform.h"
#include <cstdio>

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Macro-level guard for HIP Host Adapter
//
#if !defined(HYTLASS_ENABLE_HIP_HOST_ADAPTER)
#define HYTLASS_ENABLE_HIP_HOST_ADAPTER false
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This class defines an object which abstracts interactions between the HYTLASS device-wide GEMM and
/// HIP. The intention is to enable HYTLASS to be used with both the HIP Runtime API and HIP Driver API.
struct HipHostAdapter {

  /// Limit the number of kernels
  static constexpr int32_t kMaximumKernelCount = 4;

  /// Maximum cluster size
  static constexpr int MaxClusterSize = 32;

  //
  // Data members
  //

  /// Handles
  void        *kernel_handles[kMaximumKernelCount];
  int32_t      kernel_count = 0;

  //
  // Methods
  //

  /// Ctor
  HipHostAdapter() = default;

  /// Dtor
  virtual ~HipHostAdapter() {}

  /// Copy Ctor
  inline HipHostAdapter(const HipHostAdapter & rhs):
    kernel_count(rhs.kernel_count)
  {
    HYTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Copy Assignment
  inline HipHostAdapter& operator=(const HipHostAdapter & rhs) {

    HYTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
    kernel_count = rhs.kernel_count;
    return *this;
  }

  /// Move ctor
  inline HipHostAdapter(HipHostAdapter && rhs):
    kernel_count(rhs.kernel_count)
  {
    HYTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }
  }

  /// Move assignment
  inline HipHostAdapter& operator=(HipHostAdapter && rhs) {

    HYTLASS_ASSERT(rhs.kernel_count >= 0 && rhs.kernel_count < kMaximumKernelCount);
    for (int32_t i = 0; i < rhs.kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = rhs.kernel_handles[i];
    }

    kernel_count = rhs.kernel_count;

    return *this;
  }

  /// Ctor
  inline HipHostAdapter(
    void **kernel_handles_, 
    int32_t kernel_count_
  ): 
    kernel_count(kernel_count_)
  {
    HYTLASS_ASSERT(kernel_count >= 0);
    for (int32_t i = 0; i < kernel_count && i < kMaximumKernelCount; ++i) {
      kernel_handles[i] = kernel_handles_[i];
    }
  }

  /// Returns true if the HipHostAdapter is empty (kernel_count == 0)
  inline bool empty() const { return !kernel_count; }

  /// Returns kernel_count
  inline size_t size() const { return static_cast<size_t>(kernel_count); }

  /// Queries the occupancy of a kernel
  virtual Status query_occupancy(
    int32_t *device_sms, 
    int32_t *sm_occupancy,
    int32_t kernel_index,
    int32_t thread_count,
    int32_t smem_size) const = 0;
 
  /// Launches a kernel without using Threadblock Clusters. 
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const block_dims,
    size_t const smem_size,
    hipStream_t hip_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

  /// Launches a kernel using the HIP Extensible Launch API and Threadblock Clusters.
  virtual Status launch(
    dim3 const grid_dims,
    dim3 const cluster_dims,
    dim3 const block_dims,
    size_t const smem_size,
    hipStream_t hip_stream,
    void** kernel_params,
    int32_t kernel_index) const = 0;

protected:

  /**
   * Fills a buffer in Global Memory with a byte sequence copied from host memory.
   * This function can be overriden to dispatch to the appropriate cuMemsetD*Async API
  */
  virtual Status memsetDeviceImpl(
    void* destination, ///< Device memory pointer to be filled
    void const* fill_value, ///< Value to be filled in the buffer
    size_t fill_size, ///< Size of the data type to be used for filling the buffer
    size_t count, ///< Number of elements of size fill_size
    hipStream_t stream) const = 0;

public:

  /// Fills a buffer in Global Memory with a byte sequence copied from host memory
  template<class FillValueType>
  Status memsetDevice(
    void* destination,
    FillValueType fill_value, 
    size_t count,
    hipStream_t stream) const
  {
    return this->memsetDeviceImpl(
      destination,
      &fill_value,
      sizeof(FillValueType),
      count,
      stream);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

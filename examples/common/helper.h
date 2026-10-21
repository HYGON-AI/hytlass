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
#pragma once

#include "hip/hip_runtime.h"
#include <iostream>

/**
 * Panic wrapper for unwinding HYTLASS errors
 */
#define HYTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    hytlass::Status error = status;                                                              \
    if (error != hytlass::Status::kSuccess) {                                                    \
      std::cerr << "Got hytlass error: " << hytlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding hip runtime errors
 */
#define HIP_CHECK(status)                                              \
  {                                                                    \
    hipError_t error = status;                                         \
    if (error != hipSuccess) {                                         \
      std::cerr << "Got bad hip status: " << hipGetErrorString(error)  \
                << " at line: " << __LINE__ << std::endl;              \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  }


/**
 * GPU timer for recording the elapsed time across kernel(s) launched in GPU stream
 */
struct GpuTimer
{
    hipStream_t _stream_id;
    hipEvent_t _start;
    hipEvent_t _stop;

    /// Constructor
    GpuTimer() : _stream_id(0)
    {
        HIP_CHECK(hipEventCreate(&_start));
        HIP_CHECK(hipEventCreate(&_stop));
    }

    /// Destructor
    ~GpuTimer()
    {
        HIP_CHECK(hipEventDestroy(_start));
        HIP_CHECK(hipEventDestroy(_stop));
    }

    /// Start the timer for a given stream (defaults to the default stream)
    void start(hipStream_t stream_id = 0)
    {
        _stream_id = stream_id;
        HIP_CHECK(hipEventRecord(_start, _stream_id));
    }

    /// Stop the timer
    void stop()
    {
        HIP_CHECK(hipEventRecord(_stop, _stream_id));
    }

    /// Return the elapsed time (in milliseconds)
    float elapsed_millis()
    {
        float elapsed = 0.0;
        HIP_CHECK(hipEventSynchronize(_stop));
        HIP_CHECK(hipEventElapsedTime(&elapsed, _start, _stop));
        return elapsed;
    }
};

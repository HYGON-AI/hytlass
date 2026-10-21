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
    \brief Implementation of a CTA-wide semaphore for inter-CTA synchronization.
*/

#pragma once

#include "hytlass/hytlass.h"

#include "hytlass/array.h"

#include "hytlass/numeric_types.h"
#include "hytlass/matrix_shape.h"

#include "hytlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace hytlass {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// CTA-wide semaphore for inter-CTA synchronization.
class Semaphore { 
public:

  int *lock;
  bool wait_thread;
  int state;

public:

  /// Implements a semaphore to wait for a flag to reach a given value
  HYTLASS_HOST_DEVICE
  Semaphore(int *lock_, int thread_id): 
    lock(lock_), 
    wait_thread(thread_id < 0 || thread_id == 0),
    state(-1) {

  }

  /// Permit fetching the synchronization mechanism early
  HYTLASS_DEVICE
  void fetch() {
    if (wait_thread) {
#ifdef __HIPCC__
      __builtin_hcu_buffer_wbinvl1_vol();
      state = *lock;
#endif
    }
  }

  /// Gets the internal state
  HYTLASS_DEVICE
  int get_state() const {
    return state;
  }

  /// Waits until the semaphore is equal to the given value
  HYTLASS_DEVICE
  void wait(int status = 0) {
    while( wait_thread && (state != status) ) {
      fetch();
    }

    __syncthreads();
  }

  /// Updates the lock with the given result
  HYTLASS_DEVICE
  void release(int status = 0) {
    __syncthreads();

    if (wait_thread) {
#ifdef __HIPCC__
      *lock = status;
#endif
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////


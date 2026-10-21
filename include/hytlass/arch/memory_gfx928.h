/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
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

#include "hytlass/array.h"
#include "hytlass/layout/matrix.h"
#include "hute/arch/util.hpp"
#include "hytlass/arch/mma_gfx928.h"
#include "hytlass/arch/memory_buffer.h"
#include "hytlass/arch/cache_operation.h"

namespace hytlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element>
HYTLASS_DEVICE
void ds_read(void const* ptr, __uint128_t& D);

template<>
HYTLASS_DEVICE
void ds_read<float>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _d;
    _d = __builtin_hcu_ds_read_m32x8f32((float *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
HYTLASS_DEVICE
void ds_read<hytlass::tfloat32_t>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    v4f _d;
    _d = __builtin_hcu_ds_read_m32x8f32((float *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
HYTLASS_DEVICE
void ds_read<hytlass::half_t>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    __fp16x8_t _d;
    _d = __builtin_hcu_ds_read_m32x16f16((__fp16 *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
HYTLASS_DEVICE
void ds_read<hytlass::bfloat16_t>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    __fp16x8_t _d;
    _d = __builtin_hcu_ds_read_m32x16f16((__fp16 *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
HYTLASS_DEVICE
void ds_read<int8_t>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    __i8x16_t _d;
    _d = __builtin_hcu_ds_read_m32x32i8((int *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<>
HYTLASS_DEVICE
void ds_read<uint8_t>(void const* ptr, __uint128_t& D) {
  #if (defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__))
    __i8x16_t _d;
    _d = __builtin_hcu_ds_read_m32x32u8((int *)ptr, short(0));
    __uint128_t* fd = reinterpret_cast<__uint128_t*>(&_d);
    D = *fd;
  #else
    HYTLASS_UNUSED(ptr);
    HYTLASS_NOT_IMPLEMENTED();
  #endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Initiates an asynchronous copy from global memory to shared memory.
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async;

template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_zfill;

template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct cp_async_nan;

template <
   /// Type of Element
   typename Element,
   /// If the data is for a Hermitian matrix diagonal
   bool IsHermitianData = false>
struct cp_async_diag;


/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Always> {
  /// Copy
  HYTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    using AccessType  = Array<uint8_t, SizeInBytes>;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    }
  }
};

template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Always> {
  /// Copy with zero fill
  HYTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard) {
    using AccessType  = Array<uint8_t, SizeInBytes>;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    }
    else {
      AccessType zeros;
      zeros.clear();
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async<SizeInBytes, CacheOperation::Global> {
  /// Copy
  HYTLASS_DEVICE
  cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    using AccessType  = Array<uint8_t, SizeInBytes>;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    }
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, CacheOperation::Global> {
  /// Copy with zero fill
  HYTLASS_DEVICE
  cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    using AccessType  = Array<uint8_t, SizeInBytes>;
    if (pred_guard) {
      *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    }
    else {
      AccessType zeros;
      zeros.clear();
      *static_cast<AccessType *>(smem_ptr) = zeros;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// HYTLASS helper to get SMEM pointer
inline __device__ unsigned hytlass_get_smem_pointer(void *ptr) {
  return hute::cast_smem_ptr_to_uint(ptr);
}

/// HYTLASS helper to get SMEM pointer
inline __device__ unsigned hytlass_get_smem_pointer(void const *ptr) {
  return hytlass_get_smem_pointer(const_cast<void *>(ptr));
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct shared_load_op {
  HYTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    D = *reinterpret_cast<AccessType const *>(ptr);  
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
HYTLASS_DEVICE void shared_load(AccessType &D, void const *ptr) {
  shared_load_op<AccessType, int(sizeof(AccessType))>(D, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 16> {
  HYTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    uint4 const *mptr = reinterpret_cast<uint4 const *>(ptr);
    uint4 v = *mptr;
    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_load_op<AccessType, 8> {
  HYTLASS_DEVICE
  shared_load_op(AccessType &D, void const *ptr) {
    uint2 const *mptr = reinterpret_cast<uint2 const *>(ptr);
    uint2 v = *mptr;
    D = reinterpret_cast<AccessType const &>(v);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct shared_store_op {
  HYTLASS_DEVICE
  shared_store_op(AccessType &D, void *ptr) {
    *reinterpret_cast<AccessType *>(ptr) = D;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
HYTLASS_DEVICE void shared_store(AccessType &D, void *ptr) {
  shared_store_op<AccessType, int(sizeof(AccessType))>(D, ptr);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_store_op<AccessType, 16> {
  HYTLASS_DEVICE
  shared_store_op(AccessType &D, void *ptr) {
    uint4 v = reinterpret_cast<uint4 const &>(D);
    uint4 *mptr = reinterpret_cast<uint4 *>(ptr);
    *mptr = v;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct shared_store_op<AccessType, 8> {
  HYTLASS_DEVICE
  shared_store_op(AccessType &D, void *ptr) {
    uint2 v = reinterpret_cast<uint2 const &>(D);
    uint2 *mptr = reinterpret_cast<uint2 *>(ptr);
    *mptr = v;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


} // namespace arch
} // namespace hytlass
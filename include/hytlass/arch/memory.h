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
    \brief Architecture-specific operators on memory
*/

#pragma once

#include "hytlass/hytlass.h"
#include "hytlass/arch/cache_operation.h"
#include "hytlass/platform/platform.h"

namespace hytlass {
namespace arch {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always
    >
struct global_load;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

// The redundant mov PTX instruction is used to enforce the compiler to
// keep the initializing code before ld.global
template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 *data = reinterpret_cast<uint4 *>(&D);
    if(pred_guard)
    {
      uint4 const *tmp_ptr = reinterpret_cast<uint4 const *>(ptr);
      data[0] = *(tmp_ptr + 0);
      data[1] = *(tmp_ptr + 1);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   32,
                   CacheOperation::LastUse
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 *data = reinterpret_cast<uint4 *>(&D);
    if(pred_guard)
    {
      uint4 const *tmp_ptr = reinterpret_cast<uint4 const *>(ptr);
      data[0] = *(tmp_ptr + 0);
      data[1] = *(tmp_ptr + 1);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 &data = reinterpret_cast<uint4 &>(D);
    if(pred_guard)
    {
      uint4 const *tmp_ptr = reinterpret_cast<uint4 const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   16,
                   CacheOperation::LastUse
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint4 &data = reinterpret_cast<uint4 &>(D);
    if(pred_guard)
    {
      uint4 const *tmp_ptr = reinterpret_cast<uint4 const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint2 &data = reinterpret_cast<uint2 &>(D);
    if(pred_guard)
    {
      uint2 const *tmp_ptr = reinterpret_cast<uint2 const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   8,
                   CacheOperation::LastUse
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint2 &data = reinterpret_cast<uint2 &>(D);
    if(pred_guard)
    {
      uint2 const *tmp_ptr = reinterpret_cast<uint2 const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    unsigned &data = reinterpret_cast<unsigned &>(D);
    if(pred_guard)
    {
      unsigned const *tmp_ptr = reinterpret_cast<unsigned const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   4,
                   CacheOperation::LastUse
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    unsigned &data = reinterpret_cast<unsigned &>(D);
    if(pred_guard)
    {
      unsigned const *tmp_ptr = reinterpret_cast<unsigned const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint16_t &data = reinterpret_cast<uint16_t &>(D);
    if(pred_guard)
    {
      uint16_t const *tmp_ptr = reinterpret_cast<uint16_t const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   2,
                   CacheOperation::LastUse
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    uint16_t &data = reinterpret_cast<uint16_t &>(D);
    if(pred_guard)
    {
      uint16_t const *tmp_ptr = reinterpret_cast<uint16_t const *>(ptr);
      data = *(tmp_ptr + 0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_load<AccessType,
                   1,
                   CacheOperation::Always
                  > {
  HYTLASS_DEVICE
  global_load(AccessType &D, void const *ptr, bool pred_guard) {
    if (pred_guard){
      D = *(reinterpret_cast<AccessType const *>(ptr));
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store data
    typename AccessType,
    /// The bytes of storing
    int StoreBytes
    >
struct global_store;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Specializations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct global_store<AccessType, 64> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint4 const *data = reinterpret_cast<uint4 const *>(&D);
    if(pred_guard)
    {
      uint4 *tmp_ptr = reinterpret_cast<uint4 *>(ptr);
      *(tmp_ptr + 0) = data[0];
      *(tmp_ptr + 1) = data[1];
      *(tmp_ptr + 2) = data[2];
      *(tmp_ptr + 3) = data[3];
      __threadfence();
    }
  }
};


template <typename AccessType>
struct global_store<AccessType, 32> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint4 const *data = reinterpret_cast<uint4 const *>(&D);
    if(pred_guard)
    {
      uint4 *tmp_ptr = reinterpret_cast<uint4 *>(ptr);
      *(tmp_ptr + 0) = data[0];
      *(tmp_ptr + 1) = data[1];
      __threadfence();
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 16> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    __uint128_t const &data = reinterpret_cast<__uint128_t const &>(D);
    if(pred_guard)
    {
      __uint128_t *tmp_ptr = reinterpret_cast<__uint128_t *>(ptr);
      *(tmp_ptr) = data;
      __threadfence();
    } 
  }
};

template <typename AccessType>
struct global_store<AccessType, 8> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint2 const &data = reinterpret_cast<uint2 const &>(D);
    if(pred_guard)
    {
      uint2 *tmp_ptr = reinterpret_cast<uint2 *>(ptr);
      *(tmp_ptr + 0) = data;
      __threadfence();
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 4> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    if(pred_guard)
    {
      uint32_t *tmp_ptr = reinterpret_cast<uint32_t *>(ptr);
      *(tmp_ptr + 0) = data;
      __threadfence();
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 2> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    uint16_t const &data = reinterpret_cast<uint16_t const &>(D);
    if(pred_guard)
    {
      uint16_t *tmp_ptr = reinterpret_cast<uint16_t *>(ptr);
      *(tmp_ptr + 0) = data;
      __threadfence();
    }
  }
};

template <typename AccessType>
struct global_store<AccessType, 1> {
  HYTLASS_DEVICE
  global_store(AccessType const &D, void *ptr, bool pred_guard) {
    if(pred_guard) 
    {
      *(reinterpret_cast<AccessType *>(ptr)) = D;
      __threadfence();
    } 
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// ld.shared
template <int Bytes>
HYTLASS_DEVICE
void shared_load(void *dst, uint32_t ptr);

/// ld.shared - 16b
template <>
HYTLASS_DEVICE
void shared_load<2>(void *dst, uint32_t ptr) {
  (*reinterpret_cast<uint16_t *>(dst)) = *reinterpret_cast<uint16_t *>(ptr);
}

/// ld.shared - 32b
template <>
HYTLASS_DEVICE
void shared_load<4>(void *dst, uint32_t ptr) {
  (*reinterpret_cast<uint32_t *>(dst)) = *reinterpret_cast<uint32_t *>(ptr);
}

/// ld.shared - 64b
template <>
HYTLASS_DEVICE
void shared_load<8>(void *dst, uint32_t ptr) {
  uint2 *dst_u64 = reinterpret_cast<uint2 *>(dst);
  *dst_u64 = *reinterpret_cast<uint2 *>(ptr);
}

/// ld.shared - 128b
template <>
HYTLASS_DEVICE
void shared_load<16>(void *dst, uint32_t ptr) {
  uint4 *dst_u128 = reinterpret_cast<uint4 *>(dst);
  *dst_u128 = *reinterpret_cast<uint4 *>(ptr);
}


/////////////////////////////////////////////////////////////////////////////////////////////////

/// st.shared
template <int Bytes>
HYTLASS_DEVICE
void shared_store(uint32_t ptr, void const *src);

/// st.shared - 16b
template <>
HYTLASS_DEVICE
void shared_store<2>(uint32_t ptr, void const *src) {
  (*reinterpret_cast<uint16_t *>(ptr)) = *reinterpret_cast<uint16_t const*>(src);
}

/// st.shared - 32b
template <>
HYTLASS_DEVICE
void shared_store<4>(uint32_t ptr, void const *src) {
  uint1 const *dst_u32 = reinterpret_cast<uint1 const *>(src);
  (*reinterpret_cast<uint1 *>(ptr)) = *dst_u32;
}

/// st.shared - 64b
template <>
HYTLASS_DEVICE
void shared_store<8>(uint32_t ptr, void const *src) {
  uint2 const *dst_u64 = reinterpret_cast<uint2 const *>(src);
  (*reinterpret_cast<uint2 *>(ptr)) = *dst_u64;
}

/// st.shared - 128b
template <>
HYTLASS_DEVICE
void shared_store<16>(uint32_t ptr, void const *src) {
  uint4 const *dst_u128 = reinterpret_cast<uint4 const *>(src);
  (*reinterpret_cast<uint4 *>(ptr)) = *dst_u128;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace hytlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "hytlass/arch/memory_gfx928.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/***************************************************************************************************
 * Copyright (c) 2023 - 2025 Hygon Information Technology Co., Ltd. All rights reserved.
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    \brief Architecture-specific operators on memory added for AMDGPU
*/

#pragma once

#include "hytlass/array.h"
#include "hute/arch/util.hpp"

namespace hytlass {
namespace arch {

typedef uint32_t uint32x4_t __attribute__((ext_vector_type(4)));
typedef int32_t int32x4_t __attribute__((ext_vector_type(4)));
typedef int32_t int32x2_t __attribute__((ext_vector_type(2)));


struct alignas(16) BufferAccessor {
  int32x4_t buffer_res;
  
  /// Intra-block offset, require: threadblockShape::kK * matrix::strided * sizeof(Element) < 2^31
  int32_t vofft = 0;

  HYTLASS_HOST_DEVICE void add_to_vofft(int32_t offt) {
    vofft += offt;
  }

  /// Add the inter-block offset (sofft) to the pointer
  HYTLASS_HOST_DEVICE void add_to_sofft(int64_t offt) {
    // Offset is limited by the 48-bit address space of the buffer resource
    uint64_t* buffer_res_ptr = reinterpret_cast<uint64_t*>(&buffer_res);

    buffer_res_ptr[0] += offt;
  }

  HYTLASS_HOST_DEVICE BufferAccessor(void* ptr) {
    // Offset mode of buffer load
    uint64_t* buffer_res_ptr = reinterpret_cast<uint64_t*>(&buffer_res);
    buffer_res_ptr[0] = reinterpret_cast<uint64_t>(ptr);
    buffer_res_ptr[1] = (((((long)0x20000) << 32) | 0xFFFFFFFE));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType, int Bytes>
struct buffer_load_op {};

template <typename AccessType>
HYTLASS_DEVICE void buffer_load(AccessType &D, BufferAccessor &ptr, int32_t voffset, int64_t soffset) {
#if defined(__HIPCC__)
  uint64_t* buffer_res_ptr = reinterpret_cast<uint64_t*>(&ptr.buffer_res);
  buffer_res_ptr[0] += soffset;
  buffer_load_op<AccessType, int(sizeof(AccessType))>(D, ptr, voffset);
#else
  hytlass::arch::global_load<AccessType, sizeof(AccessType)>(
    D, ptr.buffer_pointer.ptr + voffset, true);
#endif
}

#if defined(__HIPCC__)

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename AccessType>
struct buffer_load_op<AccessType, 16> {
  HYTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, int32_t voffset) {
    int32x4_t &data = reinterpret_cast<int32x4_t &>(D);
    data = __builtin_hcu_buffer_load_dwordx4(ptr.buffer_res, 0, ptr.vofft + voffset, false, false);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AccessType>
struct buffer_load_op<AccessType, 8> {
  HYTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, int32_t voffset) {
    int32x2_t &data = reinterpret_cast<int32x2_t &>(D);
    data = __builtin_hcu_buffer_load_dwordx2(ptr.buffer_res, 0, ptr.vofft + voffset, false, false);
  }
};

// /////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AccessType>
struct buffer_load_op<AccessType, 4> {
  HYTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, int32_t voffset) {
    int &data = reinterpret_cast<int &>(D);
    data = __builtin_hcu_buffer_load_dword(ptr.buffer_res, 0, ptr.vofft + voffset, false, false);
  }
};

// /////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AccessType>
struct buffer_load_op<AccessType, 2> {
  HYTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, int32_t voffset) {
    unsigned short &data = reinterpret_cast<unsigned short &>(D);
    data = __builtin_hcu_buffer_load_ushort(ptr.buffer_res, 0, ptr.vofft + voffset, false, false);
  }
};

// /////////////////////////////////////////////////////////////////////////////////////////////////
template <typename AccessType>
struct buffer_load_op<AccessType, 1> {
  HYTLASS_DEVICE
  buffer_load_op(AccessType &D, BufferAccessor &ptr, int32_t voffset) {
    unsigned char &data = reinterpret_cast<unsigned char &>(D);
    data = __builtin_hcu_buffer_load_ubyte(ptr.buffer_res, 0, ptr.vofft + voffset, false, false);
  }
};
#endif

} // namespace arch
} // namespace hytlass
